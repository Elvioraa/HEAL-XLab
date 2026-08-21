"""Bounded, detached training diagnostics for Dual-Space extensions."""

from collections import defaultdict
import json
import math
import os

import torch
import torch.nn.functional as F

from opencood.models.sub_modules.dual_space_config import (
    resolve_dual_space_diagnostics,
)


GRADIENT_GROUP_PREFIXES = {
    "object_adapter": ("dual_space_object_adapter_",),
    "context_adapter": ("dual_space_context_adapter_",),
    "aligner": ("aligner_",),
    "encoder_backbone": ("encoder_", "backbone_"),
}


class DualSpaceTrainingDiagnostics(object):
    """Observe training tensors without retaining GPU graphs or parameters."""

    def __init__(self, dual_space_config, output_dir, print_fn=print):
        self.config = resolve_dual_space_diagnostics(dual_space_config)
        if not self.config["enabled"]:
            raise ValueError("training diagnostics require diagnostics.enabled=true")
        self.output_dir = os.path.abspath(output_dir)
        self.print_fn = print_fn
        self.quality_target_enabled = bool(
            self.config["quality_target"]["enabled"]
        )
        self.adapter_residual_enabled = bool(
            self.config["adapter_residual"]["enabled"]
        )
        self.gradient_flow_enabled = bool(
            self.config["gradient_flow"]["enabled"]
        )
        self._forward_step = -1
        self._sample_index = 0
        self._quality_record_count = 0
        self._quality_values = defaultdict(list)
        self._adapter_values = defaultdict(list)
        self._jsonl = None
        self._jsonl_failed = False
        self._ddp_gradient_skip_reported = False

    @property
    def forward_step(self):
        return self._forward_step

    def begin_forward(self):
        """Advance the observer-only forward counter."""
        self._forward_step += 1

    def next_sample_index(self):
        """Return a stable process-local sample index for one scene."""
        value = self._sample_index
        self._sample_index += 1
        return value

    def end_forward(self):
        """Print bounded summaries at the configured observer cadence."""
        if self._forward_step < 0:
            return
        if (self._forward_step + 1) % int(self.config["every_n_steps"]):
            return
        if self.quality_target_enabled:
            self._print_quality_summary()
        if self.adapter_residual_enabled:
            self._print_adapter_summary()

    def record_quality_scene(
        self,
        sample_index,
        scene_index,
        proposals,
        targets,
        matched_gt_indices,
        matched_valid,
        raw_iou,
        pair_indices,
        refined_iou,
        predicted_quality,
        agent_modalities,
    ):
        """Record the exact proposal assignment and quality supervision pairs."""
        if not self.quality_target_enabled:
            return
        cpu = {
            "proposals": _cpu_float(proposals),
            "targets": _cpu_float(targets),
            "matched_gt_indices": matched_gt_indices.detach().to(
                device="cpu", dtype=torch.long
            ),
            "matched_valid": matched_valid.detach().to(
                device="cpu", dtype=torch.bool
            ),
            "raw_iou": _cpu_float(raw_iou),
            "pair_indices": pair_indices.detach().to(device="cpu", dtype=torch.long),
            "refined_iou": _cpu_float(refined_iou),
            "predicted_quality": _cpu_float(predicted_quality),
        }
        pair_count = int(cpu["pair_indices"].shape[0])
        if cpu["refined_iou"].shape != (pair_count,):
            raise ValueError("refined IoU must match valid quality pair count")
        if cpu["predicted_quality"].shape != (pair_count,):
            raise ValueError("predicted quality must match valid pair count")
        for pair_position in range(pair_count):
            proposal_index = int(cpu["pair_indices"][pair_position, 0].item())
            agent_index = int(cpu["pair_indices"][pair_position, 1].item())
            record = {
                "sample": int(sample_index),
                "scene_in_batch": int(scene_index),
                "agent": int(agent_index),
                "modality": str(agent_modalities[agent_index]),
                "proposal_id": proposal_index,
                "matched_gt_id": int(
                    cpu["matched_gt_indices"][proposal_index].item()
                ),
                "matched": bool(cpu["matched_valid"][proposal_index].item()),
                "proposal_box": _rounded_box(cpu["proposals"][proposal_index]),
                "matched_gt_box": _rounded_box(cpu["targets"][proposal_index]),
                "raw_iou": float(cpu["raw_iou"][proposal_index].item()),
                "refined_iou": float(cpu["refined_iou"][pair_position].item()),
                "quality_target": float(cpu["refined_iou"][pair_position].item()),
                "quality_pred": float(
                    cpu["predicted_quality"][pair_position].item()
                ),
            }
            self._append_quality_values(record)
            if self._quality_record_count < int(
                self.config["quality_target"]["max_records"]
            ):
                self._write_quality_record(record)
                self._quality_record_count += 1

    def record_adapter(
        self,
        modality,
        adapter_kind,
        inputs,
        raw_residual,
        outputs,
        feature_dim,
        v6_stats=None,
    ):
        """Record detached object/context residual statistics for one route."""
        if not self.adapter_residual_enabled:
            return
        input_norm = torch.linalg.vector_norm(inputs, dim=feature_dim)
        residual_norm = torch.linalg.vector_norm(raw_residual, dim=feature_dim)
        output_norm = torch.linalg.vector_norm(outputs, dim=feature_dim)
        eps = 1.0e-6
        ratio = residual_norm / (input_norm + eps)
        cosine = F.cosine_similarity(inputs, outputs, dim=feature_dim, eps=eps)
        summary = {
            "input_norm": _scalar_mean(input_norm),
            "residual_norm": _scalar_mean(residual_norm),
            "output_norm": _scalar_mean(output_norm),
            "residual_ratio": _tensor_summary(ratio),
            "cos_input_output": _scalar_mean(cosine),
            "residual_mean": float(raw_residual.detach().float().mean().cpu()),
            "residual_std": float(
                raw_residual.detach().float().std(unbiased=False).cpu()
            ),
            "residual_abs_max": float(
                raw_residual.detach().float().abs().max().cpu()
            ) if raw_residual.numel() else 0.0,
        }
        if v6_stats is not None:
            summary.update(
                {
                    "raw_residual_ratio": _tensor_summary(
                        v6_stats["raw_residual_ratio"]
                    ),
                    "safe_residual_ratio": _tensor_summary(
                        v6_stats["safe_residual_ratio"]
                    ),
                    "cap_scale_mean": _scalar_mean(v6_stats["cap_scale"]),
                    "cap_hit_rate": float(
                        (v6_stats["cap_scale"] < 1.0)
                        .detach()
                        .float()
                        .mean()
                        .cpu()
                    ),
                }
            )
        key = "%s/%s" % (str(modality), str(adapter_kind))
        values = self._adapter_values[key]
        if len(values) < int(self.config["max_records"]):
            values.append(summary)

    def should_record_gradient_flow(self, step):
        """Return whether expensive gradient observation is due at ``step``."""
        return bool(
            self.gradient_flow_enabled
            and (int(step) + 1)
            % int(self.config["gradient_flow"]["every_n_steps"])
            == 0
        )

    def record_gradient_contributions(self, step, detection, quality):
        """Print unscaled autograd.grad norms without writing parameter grads."""
        self.print_fn(
            "[DSDiag][Gradient][step=%d] det=%s quality=%s ratio=%s"
            % (
                int(step),
                _format_group_norms(detection),
                _format_group_norms(quality),
                _format_gradient_ratios(detection, quality),
            )
        )

    def record_total_gradients(self, step, values):
        """Print actual parameter ``.grad`` norms after normal backward."""
        self.print_fn(
            "[DSDiag][GradientTotal][step=%d] %s"
            % (int(step), _format_group_norms(values))
        )

    def report_ddp_gradient_skip(self):
        """Report the documented autograd.grad/DDP boundary once."""
        if not self._ddp_gradient_skip_reported:
            self.print_fn(
                "[DSDiag][Gradient] SKIP contribution split under initialized "
                "DDP; normal reduced .grad norms remain observable"
            )
            self._ddp_gradient_skip_reported = True

    def close(self):
        """Flush and close the bounded JSONL stream."""
        if self._jsonl is not None:
            self._jsonl.flush()
            self._jsonl.close()
            self._jsonl = None

    def _append_quality_values(self, record):
        if len(self._quality_values["quality_target"]) >= int(
            self.config["max_records"]
        ):
            return
        for key in ("quality_target", "quality_pred", "raw_iou"):
            self._quality_values[key].append(float(record[key]))

    def _write_quality_record(self, record):
        if not self.config["quality_target"]["dump_jsonl"] or self._jsonl_failed:
            return
        try:
            if self._jsonl is None:
                os.makedirs(self.output_dir, exist_ok=True)
                path = os.path.join(self.output_dir, "quality_target.jsonl")
                self._jsonl = open(path, "a", encoding="utf-8")
            self._jsonl.write(json.dumps(record, sort_keys=True) + "\n")
            self._jsonl.flush()
        except OSError as error:
            self._jsonl_failed = True
            self.print_fn(
                "[DSDiag][Quality] JSONL disabled after %s: %s"
                % (type(error).__name__, error)
            )
            if self._jsonl is not None:
                self._jsonl.close()
                self._jsonl = None

    def _print_quality_summary(self):
        parts = []
        for name in ("quality_target", "quality_pred", "raw_iou"):
            summary = _list_summary(self._quality_values[name])
            parts.append("%s=%s" % (name, _format_summary(summary)))
        target_unique = len(
            {round(value, 4) for value in self._quality_values["quality_target"]}
        )
        pred_unique = len(
            {round(value, 4) for value in self._quality_values["quality_pred"]}
        )
        self.print_fn(
            "[DSDiag][Quality] %s target_unique_count=%d pred_unique_count=%d"
            % (" ".join(parts), target_unique, pred_unique)
        )

    def _print_adapter_summary(self):
        for key in sorted(self._adapter_values):
            rows = self._adapter_values[key]
            if not rows:
                continue
            latest = rows[-1]
            self.print_fn(
                "[DSDiag][Adapter][%s] input_norm=%.6g residual_norm=%.6g "
                "output_norm=%.6g ratio=%s cos_in_out=%.6g "
                "residual_mean=%.6g residual_std=%.6g residual_abs_max=%.6g"
                % (
                    key,
                    latest["input_norm"],
                    latest["residual_norm"],
                    latest["output_norm"],
                    _format_summary(latest["residual_ratio"]),
                    latest["cos_input_output"],
                    latest["residual_mean"],
                    latest["residual_std"],
                    latest["residual_abs_max"],
                )
            )
            if "cap_hit_rate" in latest:
                self.print_fn(
                    "[DSDiag][V6][%s] raw_ratio=%s safe_ratio=%s "
                    "cap_scale_mean=%.6g cap_hit_rate=%.6g"
                    % (
                        key,
                        _format_summary(latest["raw_residual_ratio"]),
                        _format_summary(latest["safe_residual_ratio"]),
                        latest["cap_scale_mean"],
                        latest["cap_hit_rate"],
                    )
                )


def attach_dual_space_training_diagnostics(
    model, model_dir, rank=0, print_fn=print
):
    """Attach a plain observer only when explicitly enabled and on rank 0."""
    base = unwrap_model(model)
    if int(rank) != 0 or not getattr(base, "dual_space_enabled", False):
        return None
    config = resolve_dual_space_diagnostics(base.dual_space_config)
    if not config["enabled"]:
        return None
    recorder = DualSpaceTrainingDiagnostics(
        base.dual_space_config,
        os.path.join(model_dir, "diagnostics"),
        print_fn=print_fn,
    )
    base._dual_space_training_diagnostics = recorder
    return recorder


def get_dual_space_training_diagnostics(model):
    """Return an attached observer without creating one."""
    return getattr(unwrap_model(model), "_dual_space_training_diagnostics", None)


def close_dual_space_training_diagnostics(model):
    """Close an attached observer; disabled configurations remain no-ops."""
    base = unwrap_model(model)
    recorder = getattr(base, "_dual_space_training_diagnostics", None)
    if recorder is None:
        return
    recorder.close()
    delattr(base, "_dual_space_training_diagnostics")


def maybe_record_loss_gradient_contributions(model, output_dict, step):
    """Observe raw-loss gradient contributions before the normal backward."""
    recorder = get_dual_space_training_diagnostics(model)
    if recorder is None or not recorder.should_record_gradient_flow(step):
        return False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        recorder.report_ddp_gradient_skip()
        return False
    payload = output_dict.get("dual_space_object")
    losses = payload.get("_diagnostic_losses") if isinstance(payload, dict) else None
    if not isinstance(losses, dict):
        return False
    detection = loss_gradient_norms(losses["detection"], model)
    quality = loss_gradient_norms(losses["quality"], model)
    recorder.record_gradient_contributions(step, detection, quality)
    return True


def maybe_record_total_gradients(model, step, grad_scale=1.0):
    """Observe actual grads after normal backward at the configured cadence."""
    recorder = get_dual_space_training_diagnostics(model)
    if recorder is None or not recorder.should_record_gradient_flow(step):
        return False
    recorder.record_total_gradients(
        step, parameter_gradient_norms(model, grad_scale=grad_scale)
    )
    return True


def loss_gradient_norms(loss, model):
    """Return grouped ``autograd.grad`` norms without populating ``.grad``."""
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise ValueError("diagnostic loss must be a scalar tensor")
    named = _selected_named_parameters(model)
    if not named or not loss.requires_grad:
        return {name: 0.0 for name in GRADIENT_GROUP_PREFIXES}
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named],
        retain_graph=True,
        allow_unused=True,
    )
    return _group_gradient_norms(named, gradients)


def parameter_gradient_norms(model, grad_scale=1.0):
    """Return grouped norms from existing parameter ``.grad`` tensors."""
    grad_scale = float(grad_scale)
    if not math.isfinite(grad_scale) or grad_scale <= 0.0:
        raise ValueError("grad_scale must be finite and positive")
    named = _selected_named_parameters(model)
    return _group_gradient_norms(
        named,
        [
            parameter.grad / grad_scale if parameter.grad is not None else None
            for _, parameter in named
        ],
    )


def unwrap_model(model):
    """Return the wrapped module for DDP-like containers."""
    return model.module if hasattr(model, "module") else model


def _selected_named_parameters(model):
    return [
        (name, parameter)
        for name, parameter in unwrap_model(model).named_parameters()
        if parameter.requires_grad
        and any(
            name.startswith(prefixes)
            for prefixes in GRADIENT_GROUP_PREFIXES.values()
        )
    ]


def _group_gradient_norms(named, gradients):
    totals = {name: 0.0 for name in GRADIENT_GROUP_PREFIXES}
    for (parameter_name, _), gradient in zip(named, gradients):
        if gradient is None:
            continue
        squared = float(
            gradient.detach().float().pow(2).sum().cpu().item()
        )
        for group, prefixes in GRADIENT_GROUP_PREFIXES.items():
            if parameter_name.startswith(prefixes):
                totals[group] += squared
                break
    return {name: math.sqrt(value) for name, value in totals.items()}


def _cpu_float(value):
    if not torch.is_tensor(value):
        raise TypeError("diagnostic values must be tensors")
    return value.detach().float().cpu()


def _rounded_box(value):
    return [round(float(item), 6) for item in value.tolist()]


def _scalar_mean(value):
    return float(value.detach().float().mean().cpu()) if value.numel() else 0.0


def _tensor_summary(value):
    flat = value.detach().float().reshape(-1)
    if not flat.numel():
        return _list_summary([])
    quantiles = torch.quantile(
        flat, flat.new_tensor([0.05, 0.25, 0.50, 0.75, 0.95])
    ).cpu().tolist()
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "min": float(flat.min().cpu()),
        "max": float(flat.max().cpu()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
    }


def _list_summary(values):
    if not values:
        return {
            "count": 0, "mean": 0.0, "std": 0.0, "min": 0.0,
            "max": 0.0, "p05": 0.0, "p25": 0.0, "p50": 0.0,
            "p75": 0.0, "p95": 0.0,
        }
    return _tensor_summary(torch.tensor(values, dtype=torch.float32))


def _format_summary(summary):
    return (
        "count=%d mean=%.6g std=%.6g min=%.6g max=%.6g "
        "p05=%.6g p25=%.6g p50=%.6g p75=%.6g p95=%.6g"
        % (
            summary["count"], summary["mean"], summary["std"],
            summary["min"], summary["max"], summary["p05"],
            summary["p25"], summary["p50"], summary["p75"],
            summary["p95"],
        )
    )


def _format_group_norms(values):
    return ",".join(
        "%s=%.6g" % (name, float(values.get(name, 0.0)))
        for name in GRADIENT_GROUP_PREFIXES
    )


def _format_gradient_ratios(detection, quality, eps=1.0e-12):
    return ",".join(
        "%s=%.6g" % (
            name,
            float(quality.get(name, 0.0))
            / (float(detection.get(name, 0.0)) + eps),
        )
        for name in GRADIENT_GROUP_PREFIXES
    )

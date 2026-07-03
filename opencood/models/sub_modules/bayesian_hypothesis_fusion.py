"""Bayesian hypothesis fusion for HVP-CBEA."""

import math

import torch
import torch.nn as nn


class BayesianHypothesisFusion(nn.Module):
    """Scatter updated hypotheses into BEV feature space with gated fusion."""

    def __init__(self, in_channels=256, mid_channels=64, pc_range=None,
                 confirm_boost=1.5, refute_penalty=-2.5, refine_boost=0.8,
                 residual_gate=None):
        super().__init__()
        self.pc_range = pc_range or [-102.4, -102.4, -3.0, 102.4, 102.4, 1.0]
        self.confirm_boost = float(confirm_boost)
        self.refute_penalty = float(refute_penalty)
        self.refine_boost = float(refine_boost)
        if isinstance(residual_gate, bool):
            residual_gate = {"enabled": residual_gate}
        else:
            residual_gate = residual_gate or {}
        self.residual_gate_enabled = bool(residual_gate.get("enabled", True))
        self.collaboration_aware_cfg = self.normalize_collaboration_aware_cfg(
            residual_gate.get("collaboration_aware")
        )
        self.residual_alpha_max = max(float(residual_gate.get("alpha_max", 0.3)), 1e-6)
        alpha_init = float(residual_gate.get("alpha_init", 0.05))
        alpha_init = min(max(alpha_init, 1e-6), self.residual_alpha_max * (1.0 - 1e-6))
        self.residual_alpha_learnable = bool(residual_gate.get("learnable", True))
        alpha_ratio = alpha_init / self.residual_alpha_max
        alpha_logit = math.log(alpha_ratio / (1.0 - alpha_ratio))
        alpha_logit = torch.tensor(alpha_logit, dtype=torch.float32)
        if self.residual_alpha_learnable:
            self.residual_alpha_logit = nn.Parameter(alpha_logit)
        else:
            self.register_buffer("residual_alpha_logit", alpha_logit)
        self.hyp_proj = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.last_residual_debug = self._make_residual_debug()
        self.last_aux_tensors = {}

    def forward(self, ego_hyps, verif_logits, refine_delta, novel_hyps,
                ego_bev_feat, collaboration_scale=None, collaboration_debug=None):
        if ego_bev_feat is None or ego_bev_feat.ndim != 4:
            self.last_residual_debug = self._make_residual_debug(fallback_reason="invalid_ego_bev_feat")
            self.last_aux_tensors = {}
            return ego_bev_feat, ego_hyps
        if ego_hyps is None or ego_hyps.ndim != 3 or ego_hyps.shape[1] == 0:
            self.last_residual_debug = self._make_residual_debug(fallback_reason="no_ego_hypotheses")
            self.last_aux_tensors = {}
            return ego_bev_feat, ego_hyps
        try:
            updated_hyps = self._update_hypotheses(ego_hyps, verif_logits, refine_delta)
            hyp_map = self._scatter(updated_hyps, novel_hyps, ego_bev_feat)
            hyp_feat = self.hyp_proj(hyp_map)
            gate = self.gate(torch.cat([ego_bev_feat, hyp_feat], dim=1))
            candidate = ego_bev_feat * (1.0 - gate) + hyp_feat * gate
            delta_feature = candidate - ego_bev_feat
            fused = self.apply_residual_delta(
                ego_bev_feat,
                delta_feature,
                collaboration_scale=collaboration_scale,
                collaboration_debug=collaboration_debug,
            )
            return fused, updated_hyps
        except Exception:
            self.last_residual_debug = self._make_residual_debug(fallback_reason="exception")
            self.last_aux_tensors = {}
            return ego_bev_feat, ego_hyps

    def compute_loss(self, *args, **kwargs):
        ref = self._find_tensor(args, kwargs)
        if ref is None:
            return torch.tensor(0.0)
        return ref.sum() * 0.0

    def _update_hypotheses(self, ego_hyps, verif_logits, refine_delta):
        if verif_logits is None or refine_delta is None:
            return ego_hyps
        probs = torch.softmax(verif_logits, dim=-1)
        base_score = torch.clamp(ego_hyps[..., 7:8], 1e-4, 1.0 - 1e-4)
        log_odds = torch.logit(base_score)
        log_odds = log_odds + probs[..., 0:1] * self.confirm_boost
        log_odds = log_odds + probs[..., 1:2] * self.refute_penalty
        log_odds = log_odds + probs[..., 2:3] * self.refine_boost
        updated_score = torch.sigmoid(log_odds)
        updated_box = ego_hyps[..., :7] + refine_delta[..., :7] * probs[..., 2:3]
        updated_valid = (updated_score > 0.05).to(ego_hyps.dtype)
        return torch.cat([updated_box, updated_score, updated_valid], dim=-1)

    def _scatter(self, hyps, novel_hyps, ref_feat):
        bsz, _, height, width = ref_feat.shape
        all_hyps = hyps
        if novel_hyps is not None and torch.is_tensor(novel_hyps) and novel_hyps.numel() > 0:
            all_hyps = torch.cat([hyps, novel_hyps.to(device=hyps.device, dtype=hyps.dtype)], dim=1)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range[:6]
        x = ((all_hyps[..., 0] - x_min) / max(x_max - x_min, 1e-6) * width).long()
        y = ((all_hyps[..., 1] - y_min) / max(y_max - y_min, 1e-6) * height).long()
        valid = (all_hyps[..., 8] > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        rows = []
        for bidx in range(bsz):
            flat = ref_feat.new_zeros((height * width,))
            if not torch.any(valid[bidx]):
                rows.append(flat)
                continue
            flat_idx = y[bidx][valid[bidx]] * width + x[bidx][valid[bidx]]
            values = all_hyps[bidx, valid[bidx], 7]
            rows.append(flat.scatter_add(0, flat_idx, values))
        return torch.stack(rows, dim=0).view(bsz, 1, height, width)

    def get_residual_debug(self):
        return dict(self.last_residual_debug)

    def get_aux_tensors(self):
        return dict(self.last_aux_tensors)

    def clear_aux_tensors(self):
        self.last_aux_tensors = {}

    def apply_residual_delta(self, ego_bev_feat, delta_feature,
                             collaboration_scale=None, collaboration_debug=None):
        alpha = self._residual_alpha().to(device=ego_bev_feat.device, dtype=ego_bev_feat.dtype)
        collaboration_scale = self._format_collaboration_scale(
            collaboration_scale,
            ego_bev_feat,
        )
        effective_alpha = alpha * collaboration_scale
        if self.residual_gate_enabled:
            hvp_residual = effective_alpha * delta_feature
            fused = ego_bev_feat + hvp_residual
        else:
            hvp_residual = delta_feature
            fused = ego_bev_feat + delta_feature
            effective_alpha = alpha
        self.last_aux_tensors = {
            "delta_feature": delta_feature,
            "hvp_residual": hvp_residual,
            "alpha": alpha,
            "effective_alpha": effective_alpha,
        }
        self.last_residual_debug = self._make_residual_debug(
            alpha,
            delta_feature,
            collaboration_scale=collaboration_scale,
            effective_alpha=effective_alpha,
            collaboration_debug=collaboration_debug,
        )
        return fused

    @classmethod
    def normalize_collaboration_aware_cfg(cls, cfg):
        normalized = cls.default_collaboration_aware_cfg()
        if isinstance(cfg, bool):
            normalized["enabled"] = cfg
        elif isinstance(cfg, dict):
            normalized.update(cfg)
        normalized["enabled"] = bool(normalized.get("enabled", False))
        normalized["no_collab_scale"] = float(normalized.get("no_collab_scale", 0.0))
        normalized["collab_scale"] = float(normalized.get("collab_scale", 1.0))
        normalized["min_cav"] = int(normalized.get("min_cav", 2))
        normalized["use_record_len"] = bool(normalized.get("use_record_len", True))
        normalized["fallback_scale"] = float(normalized.get("fallback_scale", 1.0))
        normalized["debug"] = bool(normalized.get("debug", False))
        return normalized

    @staticmethod
    def default_collaboration_aware_cfg():
        return {
            "enabled": False,
            "no_collab_scale": 0.0,
            "collab_scale": 1.0,
            "min_cav": 2,
            "use_record_len": True,
            "fallback_scale": 1.0,
            "debug": False,
        }

    @classmethod
    def compute_collaboration_scale(cls, record_len, cfg, device=None, dtype=None,
                                    batch_size=None, fallback_on_error=True):
        cfg = cls.normalize_collaboration_aware_cfg(cfg)
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        batch_size = int(batch_size or 1)
        debug = {
            "hvp_cbea_collaboration_aware_enabled": bool(cfg["enabled"]),
            "hvp_cbea_record_len": [],
            "hvp_cbea_has_collaborator": False,
            "hvp_cbea_collaboration_scale": cls._debug_value(
                torch.ones((batch_size,), device=device, dtype=dtype)
            ),
            "hvp_cbea_collaboration_fallback_reason": "",
        }
        if not cfg["enabled"]:
            return torch.ones((batch_size,), device=device, dtype=dtype), debug
        if not cfg["use_record_len"]:
            scale = torch.full((batch_size,), cfg["fallback_scale"], device=device, dtype=dtype)
            debug["hvp_cbea_collaboration_scale"] = cls._debug_value(scale)
            debug["hvp_cbea_collaboration_fallback_reason"] = "record_len_disabled"
            return scale, debug
        try:
            if record_len is None:
                raise ValueError("record_len is missing")
            lengths = torch.as_tensor(record_len, device=device, dtype=dtype).flatten()
            if lengths.numel() == 0:
                raise ValueError("record_len is empty")
            if lengths.numel() != batch_size:
                if lengths.numel() == 1:
                    lengths = lengths.expand(batch_size)
                else:
                    raise ValueError("record_len batch size mismatch")
            has_collab = lengths >= float(cfg["min_cav"])
            scale = torch.where(
                has_collab,
                torch.full_like(lengths, cfg["collab_scale"]),
                torch.full_like(lengths, cfg["no_collab_scale"]),
            )
            debug["hvp_cbea_record_len"] = [int(v) for v in lengths.detach().cpu().tolist()]
            debug["hvp_cbea_has_collaborator"] = cls._debug_bool(has_collab)
            debug["hvp_cbea_collaboration_scale"] = cls._debug_value(scale)
            return scale, debug
        except Exception as exc:
            if not fallback_on_error:
                raise ValueError("Cannot compute collaboration-aware scale: %s" % exc) from exc
            scale = torch.full((batch_size,), cfg["fallback_scale"], device=device, dtype=dtype)
            debug["hvp_cbea_collaboration_scale"] = cls._debug_value(scale)
            debug["hvp_cbea_collaboration_fallback_reason"] = type(exc).__name__
            return scale, debug

    def _residual_alpha(self):
        if not self.residual_gate_enabled:
            return self.residual_alpha_logit.new_tensor(1.0)
        return self.residual_alpha_max * torch.sigmoid(self.residual_alpha_logit)

    def _make_residual_debug(self, alpha=None, delta_feature=None, fallback_reason="",
                             collaboration_scale=None, effective_alpha=None,
                             collaboration_debug=None):
        if alpha is None:
            alpha = self._residual_alpha()
        if collaboration_scale is None:
            collaboration_scale = alpha.new_tensor(1.0)
        if effective_alpha is None:
            effective_alpha = alpha * collaboration_scale
        collaboration_debug = collaboration_debug or {}
        debug = {
            "hvp_cbea_residual_gate_enabled": bool(self.residual_gate_enabled),
            "hvp_cbea_residual_alpha": float(alpha.detach().cpu()),
            "hvp_cbea_residual_alpha_max": float(self.residual_alpha_max),
            "hvp_cbea_residual_alpha_learnable": bool(self.residual_alpha_learnable),
            "hvp_cbea_collaboration_aware_enabled": bool(
                collaboration_debug.get(
                    "hvp_cbea_collaboration_aware_enabled",
                    self.collaboration_aware_cfg.get("enabled", False),
                )
            ),
            "hvp_cbea_record_len": collaboration_debug.get("hvp_cbea_record_len", []),
            "hvp_cbea_has_collaborator": collaboration_debug.get("hvp_cbea_has_collaborator", False),
            "hvp_cbea_collaboration_scale": collaboration_debug.get(
                "hvp_cbea_collaboration_scale",
                self._debug_value(collaboration_scale),
            ),
            "hvp_cbea_effective_alpha": self._debug_value(effective_alpha),
            "hvp_cbea_collaboration_fallback_reason": collaboration_debug.get(
                "hvp_cbea_collaboration_fallback_reason",
                "",
            ),
            "hvp_cbea_delta_norm": 0.0,
            "hvp_cbea_delta_mean": 0.0,
            "hvp_cbea_delta_std": 0.0,
            "hvp_cbea_residual_fallback_reason": fallback_reason,
        }
        if torch.is_tensor(delta_feature) and delta_feature.numel() > 0:
            detached = delta_feature.detach().float()
            debug["hvp_cbea_delta_norm"] = float(detached.norm().cpu())
            debug["hvp_cbea_delta_mean"] = float(detached.mean().cpu())
            debug["hvp_cbea_delta_std"] = float(detached.std(unbiased=False).cpu())
        return debug

    @staticmethod
    def _format_collaboration_scale(collaboration_scale, ref_tensor):
        if collaboration_scale is None:
            collaboration_scale = ref_tensor.new_tensor(1.0)
        elif not torch.is_tensor(collaboration_scale):
            collaboration_scale = ref_tensor.new_tensor(collaboration_scale)
        else:
            collaboration_scale = collaboration_scale.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        if collaboration_scale.ndim == 0:
            return collaboration_scale
        return collaboration_scale.view(-1, 1, 1, 1)

    @staticmethod
    def _debug_value(value):
        if not torch.is_tensor(value):
            return value
        detached = value.detach().float().flatten().cpu()
        if detached.numel() == 1:
            return float(detached[0])
        return [float(v) for v in detached.tolist()]

    @staticmethod
    def _debug_bool(value):
        if not torch.is_tensor(value):
            return bool(value)
        detached = value.detach().flatten().cpu()
        values = [bool(v) for v in detached.tolist()]
        return values[0] if len(values) == 1 else values

    @staticmethod
    def _find_tensor(args, kwargs):
        for item in list(args) + list(kwargs.values()):
            if torch.is_tensor(item):
                return item
        return None

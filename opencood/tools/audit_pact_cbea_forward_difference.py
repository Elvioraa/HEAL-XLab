"""Read-only runtime comparison of PACT-CBEA v1 and Aligned Uniform.

Hooks only detach observations and always return ``None``; neither model's
forward result, dtype, device, order, nor checkpoint is modified.
"""
from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import math
import os
import random

import numpy as np
import torch


NUMERIC_ABS_THRESHOLD = 1e-5
NUMERIC_REL_THRESHOLD = 1e-4
CORE_PREFIXES = (
    "encoder_", "backbone_", "aligner_", "pyramid_backbone.",
    "shrink_conv.", "cls_head.", "reg_head.", "dir_head.", "compressor.",
)
NODE_ORDER = (
    "encoder", "backbone", "aligner", "pyramid_forward_single_input",
    "pyramid_forward_single_output", "pyramid_forward_collab_input",
    "pyramid_forward_collab_output", "single_feature", "local_evidence",
    "pact_rule_input", "pact_alpha", "pact_rule_output",
    "geometry_input", "geometry_output", "uniform_validity", "uniform_alpha",
    "uniform_router_output", "shrink_input", "shrink_output", "cls_head_input",
    "cls_head_output", "reg_head_output", "dir_head_output", "output_",
)
CSV_FIELDS = (
    "scene_index", "node", "status", "pact_shape", "aligned_shape",
    "pact_dtype", "aligned_dtype", "pact_device", "aligned_device",
    "pact_min", "pact_max", "pact_mean", "pact_std", "pact_absolute_mean",
    "pact_nonzero_ratio", "aligned_min", "aligned_max", "aligned_mean",
    "aligned_std", "aligned_absolute_mean", "aligned_nonzero_ratio",
    "l1_mean_absolute_difference", "l2_relative_error", "cosine_similarity",
    "normalized_correlation", "pact_has_nan", "aligned_has_nan", "pact_has_inf",
    "aligned_has_inf",
)


def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


def tensor_stats(tensor):
    if not torch.is_tensor(tensor):
        raise TypeError("tensor_stats expects a Tensor")
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    stats = {
        "shape": list(detached.shape), "dtype": str(detached.dtype),
        "device": str(detached.device), "has_nan": bool(torch.isnan(detached).any()),
        "has_inf": bool(torch.isinf(detached).any()),
        "min": None, "max": None, "mean": None, "std": None,
        "absolute_mean": None, "nonzero_ratio": None,
    }
    if finite_values.numel():
        stats.update({
            "min": _scalar(finite_values.min()), "max": _scalar(finite_values.max()),
            "mean": _scalar(finite_values.mean()),
            "std": _scalar(finite_values.float().std(unbiased=False)),
            "absolute_mean": _scalar(finite_values.abs().mean()),
            "nonzero_ratio": _scalar((finite_values.abs() > 1e-8).float().mean()),
        })
    return stats


def compare_tensors(left, right):
    left_stats, right_stats = tensor_stats(left), tensor_stats(right)
    result = {"pact": left_stats, "aligned": right_stats}
    if tuple(left.shape) != tuple(right.shape):
        result.update({"status": "structural_divergence", "metrics": None})
        return result
    left_value = left.detach().to(dtype=torch.float64)
    right_value = right.detach().to(device=left.device, dtype=torch.float64)
    difference = left_value - right_value
    denominator = torch.norm(left_value).clamp(min=1e-12)
    left_centered = left_value.reshape(-1) - left_value.mean()
    right_centered = right_value.reshape(-1) - right_value.mean()
    cosine_denominator = torch.norm(left_value.reshape(-1)) * torch.norm(right_value.reshape(-1))
    correlation_denominator = torch.norm(left_centered) * torch.norm(right_centered)
    metrics = {
        "l1_mean_absolute_difference": _scalar(difference.abs().mean()),
        "l2_relative_error": _scalar(torch.norm(difference) / denominator),
        "cosine_similarity": _scalar(torch.dot(left_value.reshape(-1), right_value.reshape(-1)) /
                                     cosine_denominator) if float(cosine_denominator) > 1e-12 else None,
        "normalized_correlation": _scalar(torch.dot(left_centered, right_centered) /
                                           correlation_denominator) if float(correlation_denominator) > 1e-12 else None,
    }
    diverged = (metrics["l1_mean_absolute_difference"] or 0.0) > NUMERIC_ABS_THRESHOLD or \
        (metrics["l2_relative_error"] or 0.0) > NUMERIC_REL_THRESHOLD
    result.update({"status": "numeric_divergence" if diverged else "close", "metrics": metrics})
    return result


class ObservationHooks(object):
    """Temporary, non-mutating tensor observations for one model forward."""

    def __init__(self, model, kind):
        self.model = model
        self.kind = kind
        self.values = {}
        self.handles = []

    def _record(self, name, value):
        if torch.is_tensor(value):
            self.values[name] = value.detach()

    def _forward_hook(self, name, selector=None):
        def hook(module, inputs, output):
            value = selector(output) if selector else output
            self._record(name, value)
            return None
        return hook

    def _pre_hook(self, name, selector=None):
        def hook(module, inputs):
            value = selector(inputs) if selector else inputs[0]
            self._record(name, value)
            return None
        return hook

    def install(self):
        for modality in getattr(self.model, "modality_name_list", []):
            for prefix, selector in (
                    ("encoder", None),
                    ("backbone", lambda output: output["spatial_features_2d"]),
                    ("aligner", None)):
                module = getattr(self.model, "%s_%s" % (prefix, modality), None)
                if module is not None:
                    self.handles.append(module.register_forward_hook(
                        self._forward_hook("%s_%s" % (prefix, modality), selector)))
        pyramid = self.model.pyramid_backbone
        self.handles.append(pyramid.register_forward_pre_hook(
            self._pre_hook("pyramid_forward_single_input")))
        self.handles.append(pyramid.register_forward_hook(
            self._forward_hook("pyramid_module_output")))
        original_single = pyramid.forward_single
        original_collab = pyramid.forward_collab

        def single_observer(*args, **kwargs):
            result = original_single(*args, **kwargs)
            self._record("pyramid_forward_single_input", args[0])
            self._record("pyramid_forward_single_output", result[0])
            self._record("single_feature", result[0])
            return result

        def collab_observer(*args, **kwargs):
            result = original_collab(*args, **kwargs)
            self._record("pyramid_forward_collab_input", args[0])
            self._record("pyramid_forward_collab_output", result[0])
            return result

        # These wrappers return exactly the original object. They exist because
        # forward_single/forward_collab are methods, not named child modules.
        pyramid.forward_single = single_observer
        pyramid.forward_collab = collab_observer
        self._pyramid_originals = (original_single, original_collab)
        for name in ("shrink_conv", "cls_head", "reg_head", "dir_head"):
            module = getattr(self.model, name, None)
            if module is not None:
                self.handles.append(module.register_forward_pre_hook(self._pre_hook(name + "_input")))
                self.handles.append(module.register_forward_hook(self._forward_hook(name + "_output")))
        if hasattr(self.model, "pact_cbea_rule"):
            rule = self.model.pact_cbea_rule
            self.handles.append(rule.register_forward_pre_hook(self._pre_hook("pact_rule_input")))

            def rule_hook(module, inputs, output):
                self._record("pact_rule_output", output[0])
                if isinstance(output[1], dict):
                    for key, node in (("pact_alpha", "pact_alpha"), ("pact_reliability", "pact_reliability")):
                        if key in output[1]:
                            self._record(node, output[1][key])
                return None
            self.handles.append(rule.register_forward_hook(rule_hook))
        for modality in getattr(self.model, "modality_name_list", []):
            module = getattr(self.model, "pact_cbea_evidence_head_%s" % modality, None)
            if module is not None:
                self.handles.append(module.register_forward_hook(
                    self._forward_hook("local_evidence_%s" % modality,
                                       lambda output: output["evidence_heatmap_logits"])))
        if hasattr(self.model, "pact_geometry_aligner"):
            geometry = self.model.pact_geometry_aligner
            self.handles.append(geometry.register_forward_pre_hook(self._pre_hook("geometry_input")))
            self.handles.append(geometry.register_forward_hook(
                self._forward_hook("geometry_output", lambda output: output["feature"])))
        if hasattr(self.model, "pact_aligned_uniform_router"):
            router = self.model.pact_aligned_uniform_router

            def router_pre(module, inputs):
                feature, validity, record_len = inputs
                self._record("uniform_validity", validity)
                values, start = [], 0
                for count in record_len.detach().cpu().tolist():
                    scene_validity = validity[start:start + int(count)].detach()
                    values.append(scene_validity / scene_validity.sum(dim=0, keepdim=True))
                    start += int(count)
                self._record("uniform_alpha", torch.cat(values, dim=0))
                return None
            self.handles.append(router.register_forward_pre_hook(router_pre))
            self.handles.append(router.register_forward_hook(
                self._forward_hook("uniform_router_output", lambda output: output[0])))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        if hasattr(self, "_pyramid_originals"):
            self.model.pyramid_backbone.forward_single = self._pyramid_originals[0]
            self.model.pyramid_backbone.forward_collab = self._pyramid_originals[1]


def _core_keys(state):
    return sorted(key for key in state if key.startswith(CORE_PREFIXES))


def validate_checkpoint_equivalence(pact_model, aligned_model):
    pact_state, aligned_state = pact_model.state_dict(), aligned_model.state_dict()
    pact_core, aligned_core = set(_core_keys(pact_state)), set(_core_keys(aligned_state))
    if pact_core != aligned_core:
        raise RuntimeError("core checkpoint key sets differ: pact_only=%s aligned_only=%s" %
                           (sorted(pact_core - aligned_core), sorted(aligned_core - pact_core)))
    mismatched = [key for key in sorted(pact_core)
                  if not torch.equal(pact_state[key].cpu(), aligned_state[key].cpu())]
    if mismatched:
        raise RuntimeError("common core checkpoint tensors differ: %s" % ", ".join(mismatched[:16]))
    detector = ("cls_head.", "reg_head.", "dir_head.")
    detector_mismatch = [key for key in pact_state if key.startswith(detector)
                         and key in aligned_state and not torch.equal(pact_state[key].cpu(), aligned_state[key].cpu())]
    if detector_mismatch:
        raise RuntimeError("detection head tensors differ: %s" % ", ".join(detector_mismatch[:16]))
    return {
        "common_core_key_count": len(pact_core), "missing_pact_core_keys": [],
        "missing_aligned_core_keys": [], "different_core_tensors": [],
        "detection_head_equal": True,
        "pact_extra_keys": sorted(set(pact_state) - set(aligned_state))[:64],
        "aligned_extra_keys": sorted(set(aligned_state) - set(pact_state))[:64],
        "pact_trainable_total": sum(p.numel() for p in pact_model.parameters() if p.requires_grad),
        "aligned_trainable_total": sum(p.numel() for p in aligned_model.parameters() if p.requires_grad),
        "pact_all_batchnorm_eval": all(not item.training for item in pact_model.modules()
                                        if isinstance(item, torch.nn.modules.batchnorm._BatchNorm)),
        "aligned_all_batchnorm_eval": all(not item.training for item in aligned_model.modules()
                                           if isinstance(item, torch.nn.modules.batchnorm._BatchNorm)),
    }


def structured_yaml_diff(left, right, path=""):
    differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            next_path = "%s.%s" % (path, key) if path else str(key)
            if key not in left or key not in right:
                differences.append({"path": next_path, "pact": left.get(key), "aligned": right.get(key)})
            else:
                differences.extend(structured_yaml_diff(left[key], right[key], next_path))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append({"path": path, "pact": left, "aligned": right})
        else:
            for index, (item_left, item_right) in enumerate(zip(left, right)):
                differences.extend(structured_yaml_diff(item_left, item_right, "%s[%d]" % (path, index)))
    elif left != right:
        differences.append({"path": path, "pact": left, "aligned": right})
    return differences


def classify_evidence_source(output):
    debug = output.get("pact_cbea", {}) if isinstance(output, dict) else {}
    if not debug:
        return "未启用原始规则分支"
    if debug.get("pact_used_base_heal_fallback"):
        return "本地证据头缺失，回退到原始多尺度协同融合"
    if debug.get("pact_local_evidence_enabled"):
        fallbacks = debug.get("pact_fallbacks", [])
        if any("ones" in str(item) for item in fallbacks):
            return "训练后的本地证据头，部分权重回退为全一"
        return "训练后的本地证据头"
    return "未识别证据来源"


def static_code_audit(repo_root):
    paths = {
        "pact": os.path.join(repo_root, "opencood", "models", "heter_pyramid_collab_pact_cbea.py"),
        "aligned": os.path.join(repo_root, "opencood", "models", "heter_pyramid_collab_pact_cbea_aligned_uniform.py"),
        "rule": os.path.join(repo_root, "opencood", "models", "sub_modules", "pact_cbea_rule.py"),
        "pyramid": os.path.join(repo_root, "opencood", "models", "fuse_modules", "pyramid_fuse.py"),
    }
    source = dict((name, open(path, "r", encoding="utf-8").read()) for name, path in paths.items())
    checks = {
        "original_calls_forward_collab": ".forward_collab(" in source["pact"],
        "aligned_calls_forward_collab": ".forward_collab(" in source["aligned"],
        "both_call_forward_single": ".forward_single(" in source["pact"] and ".forward_single(" in source["aligned"],
        "original_rule_receives_warped_single_feature": "warped_feature = self._warp_to_ego(single_feature" in source["pact"],
        "aligned_routes_geometry_feature": "aligned[\"feature\"]" in source["aligned"],
        "pyramid_multiscale_weighted_fuse": "weighted_fuse(feature_list[i]" in source["pyramid"],
        "rule_uniform_fallback": "uniform = feature_tensor.new_full" in source["rule"],
    }
    required_checks = dict(checks)
    required_checks.pop("aligned_calls_forward_collab")
    if not all(required_checks.values()) or checks["aligned_calls_forward_collab"]:
        raise RuntimeError("static PACT/Aligned forward contract does not match the expected audit path: %s" % checks)
    return {
        "source_files": paths,
        "source_checks": checks,
        "original_pact": "先执行单车解码用于本地证据，再额外执行多尺度协同融合；有本地证据时规则在已对齐的单车解码特征上加权，否则使用多尺度协同结果。",
        "aligned_uniform": "只执行单车解码；几何对齐后按每像素有效区域均匀聚合，不调用多尺度协同融合接口。",
        "fusion_scale": "原规则分支与均匀基线均使用解码后的单车特征；原始回退路径额外使用每个金字塔尺度的协同加权和再解码。",
        "residual_or_bypass": "原 PACT 始终计算多尺度基础融合；证据可用时最终检测使用规则聚合特征，证据不可用时直接使用该基础融合结果。没有显式相加残差。",
        "uniform_mean": "原规则的全一权重仅在规则分支中导致按车辆数的均匀权重；它不等价于原始多尺度协同融合，后者每层使用占据头分数、空间变换和软最大化权重。",
        "detection_input": "原 PACT：规则聚合特征或多尺度协同融合特征；Aligned Uniform：有效区域归一化后的单车解码特征。两者随后均经 shrink（若启用）和共享检测头。",
    }


def _node_label(node):
    labels = {
        "pyramid_forward_collab_output": "多尺度协同融合输出",
        "pyramid_forward_collab_input": "多尺度协同融合输入",
        "pyramid_forward_single_output": "单车解码输出",
        "pact_rule_input": "规则聚合输入",
        "pact_rule_output": "规则聚合输出",
        "pact_alpha": "原始规则权重",
        "uniform_alpha": "有效区域均匀权重",
        "uniform_router_output": "均匀聚合输出",
        "cls_head_input": "分类检测头输入",
    }
    return labels.get(node, "未命名观测节点")


def _write_outputs(output_dir, report, rows, summary_lines):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "forward_difference_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "forward_difference_audit.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, "forward_difference_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")


def _comparison_rows(scene_index, pact_values, aligned_values):
    rows, comparisons = [], {}
    for node in sorted(set(pact_values) | set(aligned_values)):
        if node not in pact_values or node not in aligned_values:
            item = {"status": "structural_divergence", "pact": tensor_stats(pact_values[node]) if node in pact_values else None,
                    "aligned": tensor_stats(aligned_values[node]) if node in aligned_values else None, "metrics": None}
        else:
            item = compare_tensors(pact_values[node], aligned_values[node])
        comparisons[node] = item
        row = {"scene_index": scene_index, "node": node, "status": item["status"]}
        for prefix, stats in (("pact", item.get("pact")), ("aligned", item.get("aligned"))):
            if stats:
                for key, value in stats.items():
                    row["%s_%s" % (prefix, key)] = json.dumps(value) if isinstance(value, list) else value
        if item["metrics"]:
            row.update(item["metrics"])
        rows.append(row)
    return comparisons, rows


def _first_nodes(comparisons):
    ordered = sorted(comparisons, key=lambda name: next((index for index, prefix in enumerate(NODE_ORDER)
                                                         if name.startswith(prefix)), len(NODE_ORDER)))
    structural = next((name for name in ordered if comparisons[name]["status"] == "structural_divergence"), None)
    numeric = next((name for name in ordered if comparisons[name]["status"] == "numeric_divergence"), None)
    numeric_items = [(name, item["metrics"]["l2_relative_error"] or 0.0) for name, item in comparisons.items()
                     if item["metrics"]]
    largest = max(numeric_items, key=lambda item: item[1])[0] if numeric_items else None
    return structural, numeric, largest


def audit(args):
    from torch.utils.data import DataLoader
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    set_deterministic_seed(args.seed)
    pact_hypes = yaml_utils.load_yaml(os.path.join(args.pact_model_dir, "config.yaml"))
    aligned_hypes = yaml_utils.load_yaml(os.path.join(args.aligned_model_dir, "config.yaml"))
    pact_hypes["validate_dir"] = pact_hypes["test_dir"]
    aligned_hypes["validate_dir"] = aligned_hypes["test_dir"]
    yaml_differences = structured_yaml_diff(pact_hypes, aligned_hypes)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pact_model, aligned_model = train_utils.create_model(pact_hypes), train_utils.create_model(aligned_hypes)
    state = torch.load(args.checkpoint, map_location="cpu")
    pact_model.load_state_dict(state, strict=False)
    aligned_model.load_state_dict(state, strict=False)
    pact_model.to(device).eval()
    aligned_model.to(device).eval()
    checkpoint = validate_checkpoint_equivalence(pact_model, aligned_model)
    if checkpoint["pact_trainable_total"] or checkpoint["aligned_trainable_total"]:
        raise RuntimeError("runtime comparison requires both models fully frozen")
    dataset = build_dataset(pact_hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=dataset.collate_batch_test)
    report = {"thresholds": {"absolute_mean": NUMERIC_ABS_THRESHOLD, "relative_l2": NUMERIC_REL_THRESHOLD},
              "static_code_audit": static_code_audit(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
              "checkpoint": checkpoint, "yaml_differences": yaml_differences, "scenes": []}
    rows = []
    for scene_index, batch in enumerate(loader):
        if scene_index >= args.num_scenes:
            break
        batch = train_utils.to_device(batch, device)
        pact_hook, aligned_hook = ObservationHooks(pact_model, "pact"), ObservationHooks(aligned_model, "aligned")
        pact_hook.install()
        aligned_hook.install()
        try:
            with torch.no_grad():
                pact_output = pact_model(batch["ego"])
                aligned_output = aligned_model(batch["ego"])
        finally:
            pact_hook.remove()
            aligned_hook.remove()
        for key, value in pact_output.items():
            if torch.is_tensor(value):
                pact_hook._record("output_" + key, value)
        for key, value in aligned_output.items():
            if torch.is_tensor(value):
                aligned_hook._record("output_" + key, value)
        comparisons, scene_rows = _comparison_rows(scene_index, pact_hook.values, aligned_hook.values)
        first_structural, first_numeric, largest = _first_nodes(comparisons)
        report["scenes"].append({
            "scene_index": scene_index, "cav_id_list": batch["ego"].get("cav_id_list", []),
            "agent_modality_list": list(batch["ego"]["agent_modality_list"]),
            "record_len": batch["ego"]["record_len"].detach().cpu().tolist(),
            "comparisons": comparisons, "first_structural_divergence": first_structural,
            "first_numeric_divergence": first_numeric, "largest_relative_error_node": largest,
            "evidence_source_used_by_original_pact": classify_evidence_source(pact_output),
        })
        rows.extend(scene_rows)
        pact_hook.values.clear()
        aligned_hook.values.clear()
    first_scene = report["scenes"][0] if report["scenes"] else {}
    report["summary"] = {
        "first_structural_divergence": first_scene.get("first_structural_divergence"),
        "first_numeric_divergence": first_scene.get("first_numeric_divergence"),
        "largest_relative_error_node": first_scene.get("largest_relative_error_node"),
        "detection_head_input_difference": first_scene.get("comparisons", {}).get("cls_head_input"),
        "evidence_source_used_by_original_pact": first_scene.get("evidence_source_used_by_original_pact"),
        "original_fusion_operation_summary": report["static_code_audit"]["original_pact"],
        "aligned_uniform_fusion_operation_summary": report["static_code_audit"]["aligned_uniform"],
    }
    summary_lines = [
        "检查点核心参数一致：%s。" % ("是" if checkpoint["detection_head_equal"] else "否"),
        "首个结构差异：%s。" % _node_label(report["summary"]["first_structural_divergence"]),
        "首个数值差异：%s。" % _node_label(report["summary"]["first_numeric_divergence"]),
        "原始模型证据来源：%s。" % report["summary"]["evidence_source_used_by_original_pact"],
        "统一权重不等于原始多尺度协同融合的简单平均；两条路径的加权位置和尺度不同。",
        "最小下一步消融：在保留原始单车特征与检测头的前提下，单独替换原始规则聚合权重为有效区域均匀权重。",
    ]
    _write_outputs(args.output_dir, report, rows, summary_lines)
    print("FORWARD_DIFFERENCE_AUDIT_COMPLETE scenes=%d" % len(report["scenes"]))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="Read-only PACT forward difference audit")
    parser.add_argument("--pact_model_dir", required=True)
    parser.add_argument("--aligned_model_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_scenes", type=int, default=4)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    audit(build_parser().parse_args())

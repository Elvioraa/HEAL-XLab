"""Inference-only calibration diagnostics for Dual-Space quality consensus.

The collector consumes detached metadata produced by the real DS inference
path.  It never feeds data back to the model, and moves values to CPU only
after the forward has finished so long inference runs retain no GPU tensors.
"""

import csv
import json
import math
import os
from statistics import median

import torch

from opencood.models.sub_modules.dual_space_box_coder import (
    aligned_rotated_bev_iou_hwl,
    decode_box_residual,
    pairwise_rotated_bev_iou_hwl,
)
from opencood.models.sub_modules.dual_space_config import resolve_dual_space_diagnostics
from opencood.utils.dual_space_refinement_diagnostics import OBSERVATION_KEY


def _empty_stats():
    return {"count": 0, "pred_quality_mean": 0.0, "pred_quality_std": 0.0,
            "true_iou_mean": 0.0, "true_iou_std": 0.0, "mae": 0.0,
            "rmse": 0.0, "pearson": 0.0, "spearman": 0.0}


class DualSpaceQualityDiagnostics(object):
    """Accumulate DS-V3.1 quality calibration without changing inference."""

    def __init__(self, dual_space_config=None):
        self.config = resolve_dual_space_diagnostics(dual_space_config)
        quality = (dual_space_config or {}).get("quality", {})
        self.enabled = bool(
            self.config["enabled"]
            and self.config["quality_target"]["enabled"]
            and quality.get("enabled", False)
        )
        self.yaw_mode = (dual_space_config or {}).get("refiner", {}).get("yaw_mode")
        self._pairs = []
        self._proposals = []

    @classmethod
    def from_model(cls, model):
        return cls(getattr(model, "dual_space_config", None) if getattr(
            model, "dual_space_enabled", False) else None)

    def update_inference_result(self, inference_result, scene_index):
        """Collect one actual inference scene, or no-op when disabled."""
        if not self.enabled:
            return None
        observation = inference_result.get(OBSERVATION_KEY)
        if not isinstance(observation, dict):
            raise RuntimeError("quality diagnostics enabled but observation is missing")
        metadata = observation.get("metadata", {})
        gt_boxes = observation.get("quality_gt_boxes")
        return self.update_metadata(metadata, gt_boxes, scene_index)

    def update_metadata(self, metadata, gt_boxes, scene_index):
        """Match selected proposals to GT and record per-agent true IoU."""
        if not self.enabled:
            return None
        required = ("selected_proposals", "valid_mask", "per_agent_residuals",
                    "per_agent_quality", "consensus_weights", "agent_modalities")
        missing = [key for key in required if key not in metadata]
        if missing:
            # Empty/identity-only refinement frames have no selected proposal
            # payload. They are valid inference outcomes, not diagnostics errors.
            return None
        proposals = metadata["selected_proposals"]
        valid = metadata["valid_mask"]
        residuals = metadata["per_agent_residuals"]
        predicted = metadata["per_agent_quality"]
        weights = metadata["consensus_weights"]
        modalities = tuple(metadata["agent_modalities"])
        _validate_metadata(proposals, valid, residuals, predicted, weights, modalities)
        fallback = metadata.get("quality_fallback")
        if fallback is None:
            fallback = torch.zeros(proposals.shape[0], dtype=torch.bool, device=proposals.device)
        if self.yaw_mode is None:
            raise RuntimeError("quality diagnostics requires refiner.yaw_mode")
        gt_boxes = _validate_gt(gt_boxes, proposals)
        matched_iou, matched_indices = _proposal_matches(proposals, gt_boxes)
        true_quality = torch.zeros_like(predicted)
        if gt_boxes.shape[0] and valid.any():
            expanded_proposals = proposals[:, None, :].expand_as(residuals[..., :7])
            individual = decode_box_residual(expanded_proposals, residuals, self.yaw_mode)
            targets = gt_boxes.index_select(0, matched_indices).unsqueeze(1).expand_as(individual)
            flat_valid = valid.reshape(-1)
            true_quality[valid] = aligned_rotated_bev_iou_hwl(
                individual.reshape(-1, 7)[flat_valid], targets.reshape(-1, 7)[flat_valid]
            )
        self._append_scene(
            scene_index, valid, predicted, true_quality, weights, matched_iou,
            modalities, fallback, has_gt=bool(gt_boxes.shape[0]),
        )

    def _append_scene(self, scene_index, valid, predicted, true_quality, weights,
                      matched_iou, modalities, fallback, has_gt):
        for proposal_index in range(valid.shape[0]):
            agent_indices = valid[proposal_index].nonzero(as_tuple=False).flatten()
            count = int(agent_indices.numel())
            entropy, max_weight, uniform_l1 = _weight_stats(weights[proposal_index], agent_indices)
            record = {"scene_index": scene_index, "proposal_index": proposal_index,
                      "num_valid_agents": count, "proposal_match_iou": float(matched_iou[proposal_index]),
                      "pred_best_agent": None, "true_best_agent": None, "top1_correct": None,
                      "pred_spread": 0.0, "true_spread": 0.0,
                      "normalized_weight_entropy": entropy, "max_weight": max_weight,
                      "uniform_l1_distance": uniform_l1, "fallback": bool(fallback[proposal_index]),
                      "has_gt": bool(has_gt), "proposal_spearman": None}
            if count:
                pred = predicted[proposal_index, agent_indices]
                truth = true_quality[proposal_index, agent_indices]
                record["pred_spread"] = float((pred.max() - pred.min()).item())
                if has_gt:
                    record["true_spread"] = float((truth.max() - truth.min()).item())
                if count >= 2 and has_gt:
                    pred_best = int(agent_indices[pred.argmax()].item())
                    true_max = truth.max()
                    true_best = int(agent_indices[truth.argmax()].item())
                    record.update({"pred_best_agent": pred_best, "true_best_agent": true_best,
                                   "top1_correct": bool(truth[pred.argmax()].eq(true_max)),
                                   "proposal_spearman": _spearman(
                                       pred.detach().cpu().tolist(), truth.detach().cpu().tolist()
                                   )})
                if has_gt:
                    for agent_index in agent_indices.tolist():
                        self._pairs.append({"scene_index": scene_index, "proposal_index": proposal_index,
                            "agent_index": agent_index, "modality": modalities[agent_index],
                            "proposal_match_iou": float(matched_iou[proposal_index]),
                            "pred_quality": float(predicted[proposal_index, agent_index]),
                            "true_quality": float(true_quality[proposal_index, agent_index]),
                            "consensus_weight": float(weights[proposal_index, agent_index]), "valid": True})
            self._proposals.append(record)

    def summary(self):
        """Return JSON-serializable global, threshold, modality and rank statistics."""
        subsets = {"global": self._pair_stats(self._pairs)}
        for name, threshold, strict in (("match_iou_gt_0", 0.0, True),
                                        ("match_iou_ge_0.1", 0.1, False),
                                        ("match_iou_ge_0.3", 0.3, False)):
            rows = [row for row in self._pairs if (row["proposal_match_iou"] > threshold if strict else row["proposal_match_iou"] >= threshold)]
            subsets[name] = self._pair_stats(rows)
        modalities = sorted({row["modality"] for row in self._pairs})
        subsets["per_modality"] = {name: self._modality_stats([r for r in self._pairs if r["modality"] == name]) for name in modalities}
        weighted = [row for row in self._proposals if row["num_valid_agents"] >= 2]
        for name, threshold, strict in (("ranking_match_iou_gt_0", 0.0, True),
                                        ("ranking_match_iou_ge_0.1", 0.1, False),
                                        ("ranking_match_iou_ge_0.3", 0.3, False)):
            ranked = _matched_proposals(self._proposals, threshold, strict)
            subsets[name] = _ranking_stats(ranked)
        for name, threshold, strict in (("match_iou_gt_0", 0.0, True),
                                        ("match_iou_ge_0.1", 0.1, False),
                                        ("match_iou_ge_0.3", 0.3, False)):
            spread_rows = _matched_proposals(self._proposals, threshold, strict)
            subsets["quality_spread_statistics_" + name] = _spread_stats(spread_rows)
        subsets["weight_statistics"] = {"eligible_proposal_count": len(weighted), "mean_normalized_entropy": _mean([r["normalized_weight_entropy"] for r in weighted]), "median_normalized_entropy": _median([r["normalized_weight_entropy"] for r in weighted]), "mean_max_weight": _mean([r["max_weight"] for r in weighted]), "mean_uniform_l1_distance": _mean([r["uniform_l1_distance"] for r in weighted]), "quality_fallback_ratio": _mean([float(r["fallback"]) for r in self._proposals])}
        return subsets

    def _pair_stats(self, rows):
        if not rows:
            return _empty_stats()
        pred = [row["pred_quality"] for row in rows]
        truth = [row["true_quality"] for row in rows]
        result = _empty_stats()
        result.update({"count": len(rows), "pred_quality_mean": _mean(pred), "pred_quality_std": _std(pred), "true_iou_mean": _mean(truth), "true_iou_std": _std(truth), "mae": _mean([abs(a-b) for a,b in zip(pred, truth)]), "rmse": math.sqrt(_mean([(a-b)**2 for a,b in zip(pred, truth)])), "pearson": _pearson(pred, truth), "spearman": _spearman(pred, truth)})
        return result

    def _modality_stats(self, rows):
        stats = self._pair_stats(rows)
        stats["bias"] = _mean([r["pred_quality"] - r["true_quality"] for r in rows])
        return stats

    def save(self, output_dir, suffix=None):
        """Write quality summary plus pair and proposal CSVs when enabled."""
        if not self.enabled:
            return None
        os.makedirs(output_dir, exist_ok=True)
        suffix = "" if not suffix else "_" + suffix
        summary_path = os.path.join(output_dir, "quality_diag_summary%s.json" % suffix)
        with open(summary_path, "w", encoding="utf-8") as stream:
            json.dump(self.summary(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        _write_csv(os.path.join(output_dir, "quality_diag_pairs%s.csv" % suffix), self._pairs)
        _write_csv(os.path.join(output_dir, "quality_diag_proposals%s.csv" % suffix), self._proposals)
        print("[DualSpace Quality Diagnostics] saved %s" % summary_path)
        return summary_path


def _validate_metadata(proposals, valid, residuals, predicted, weights, modalities):
    if not torch.is_tensor(proposals) or proposals.ndim != 2 or proposals.shape[1] != 7:
        raise ValueError("selected_proposals must have shape [P,7]")
    if not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.ndim != 2:
        raise ValueError("valid_mask must be bool [P,A]")
    if not torch.is_tensor(residuals) or tuple(residuals.shape) != tuple(valid.shape) + (8,):
        raise ValueError("per_agent_residuals must have shape [P,A,8]")
    if not torch.is_tensor(predicted) or tuple(predicted.shape) != tuple(valid.shape):
        raise ValueError("per_agent_quality must have shape [P,A]")
    if not torch.is_tensor(weights) or tuple(weights.shape) != tuple(valid.shape):
        raise ValueError("consensus_weights must have shape [P,A]")
    if proposals.shape[0] != valid.shape[0] or len(modalities) != valid.shape[1]:
        raise ValueError("quality metadata proposal or agent dimensions disagree")


def _validate_gt(gt_boxes, proposals):
    if gt_boxes is None:
        return proposals.new_empty((0, 7))
    if not torch.is_tensor(gt_boxes) or gt_boxes.ndim != 2 or gt_boxes.shape[1] != 7:
        raise ValueError("quality GT boxes must have shape [G,7]")
    return gt_boxes.to(device=proposals.device, dtype=proposals.dtype)


def _proposal_matches(proposals, gt_boxes):
    if not gt_boxes.shape[0]:
        return proposals.new_zeros((proposals.shape[0],)), torch.zeros(proposals.shape[0], dtype=torch.long, device=proposals.device)
    matrix = pairwise_rotated_bev_iou_hwl(proposals, gt_boxes)
    return matrix.max(dim=1)


def _weight_stats(weights, indices):
    if int(indices.numel()) < 2:
        return 0.0, float(weights[indices].max()) if indices.numel() else 0.0, 0.0
    values = weights[indices].clamp_min(0)
    values = values / values.sum().clamp_min(torch.finfo(values.dtype).eps)
    entropy = -(values * values.clamp_min(torch.finfo(values.dtype).eps).log()).sum() / math.log(float(indices.numel()))
    return float(entropy), float(values.max()), float((values - 1.0 / indices.numel()).abs().sum())


def _mean(values): return float(sum(values) / len(values)) if values else 0.0
def _median(values): return float(median(values)) if values else 0.0
def _std(values): return float(torch.tensor(values).std(unbiased=False)) if values else 0.0
def _pearson(x, y):
    if len(x) < 2: return 0.0
    a, b = torch.tensor(x), torch.tensor(y)
    denom = a.std(unbiased=False) * b.std(unbiased=False)
    return float(((a-a.mean())*(b-b.mean())).mean()/denom) if float(denom) > 0 else 0.0
def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i]); ranks = [0.0] * len(values); start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for pos in order[start:end]: ranks[pos] = rank
        start = end
    return ranks
def _spearman(x, y): return _pearson(_rank(x), _rank(y))
def _matched_proposals(rows, threshold, strict):
    return [row for row in rows if row["has_gt"] and row["num_valid_agents"] >= 2 and (
        row["proposal_match_iou"] > threshold if strict else row["proposal_match_iou"] >= threshold)]
def _ranking_stats(rows):
    return {"eligible_proposal_count": len(rows),
            "top1_agent_accuracy": _mean([float(row["top1_correct"]) for row in rows]),
            "mean_proposal_spearman": _mean([row["proposal_spearman"] for row in rows])}
def _spread_stats(rows):
    return {"eligible_proposal_count": len(rows),
            "mean_predicted_spread": _mean([row["pred_spread"] for row in rows]),
            "median_predicted_spread": _median([row["pred_spread"] for row in rows]),
            "mean_true_spread": _mean([row["true_spread"] for row in rows]),
            "median_true_spread": _median([row["true_spread"] for row in rows])}
def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

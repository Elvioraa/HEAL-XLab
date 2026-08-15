"""Inference-only geometry diagnostics for Dual-Space refinement.

IoU intentionally reuses OpenCOOD's official AP implementation primitives:
the first four BEV corners are converted to Shapely polygons and intersected
with :func:`opencood.utils.common_utils.compute_iou`.  This is rotated BEV IoU,
not 3D IoU.  The observer never changes prediction tensors or scores.
"""

import csv
import json
import os
from statistics import median

import numpy as np
import torch

from opencood.models.sub_modules.dual_space_config import (
    resolve_dual_space_diagnostics,
)


OBSERVATION_KEY = "dual_space_refinement_observation"


def official_pairwise_rotated_bev_iou(boxes_a, boxes_b):
    """Return the official Shapely rotated-BEV IoU matrix as float32 NumPy."""
    from opencood.utils import common_utils

    _validate_corner_boxes(boxes_a, "boxes_a")
    _validate_corner_boxes(boxes_b, "boxes_b")
    result = np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    if result.size == 0:
        return result
    polygons_a = list(
        common_utils.convert_format(common_utils.torch_tensor_to_numpy(boxes_a.detach()))
    )
    polygons_b = list(
        common_utils.convert_format(common_utils.torch_tensor_to_numpy(boxes_b.detach()))
    )
    for row, polygon in enumerate(polygons_a):
        result[row] = common_utils.compute_iou(polygon, polygons_b)
    return result


def deterministic_before_matching(iou_matrix, match_iou_min):
    """Greedily match unique proposal/GT pairs by BEFORE IoU and stable indices."""
    if not isinstance(iou_matrix, np.ndarray) or iou_matrix.ndim != 2:
        raise ValueError("iou_matrix must be a two-dimensional NumPy array")
    match_iou_min = float(match_iou_min)
    candidates = []
    for proposal_index in range(iou_matrix.shape[0]):
        for gt_index in range(iou_matrix.shape[1]):
            value = float(iou_matrix[proposal_index, gt_index])
            if value >= match_iou_min:
                candidates.append((-value, proposal_index, gt_index))
    candidates.sort()
    used_proposals = set()
    used_gt = set()
    matches = []
    for negative_iou, proposal_index, gt_index in candidates:
        if proposal_index in used_proposals or gt_index in used_gt:
            continue
        used_proposals.add(proposal_index)
        used_gt.add(gt_index)
        matches.append((proposal_index, gt_index, -negative_iou))
    return matches


def make_refinement_observation(
    pred_box_before_ds,
    pred_score_before_ds,
    pred_box_after_ds,
    pred_score_after_ds,
    metadata,
    quality_gt_boxes=None,
):
    """Clone and detach one inference scene without altering prediction outputs."""
    if not isinstance(metadata, dict):
        raise TypeError("Dual-Space refinement metadata must be a mapping")
    observation = {
        "pred_box_before_ds": _clone_optional(pred_box_before_ds),
        "pred_score_before_ds": _clone_optional(pred_score_before_ds),
        "pred_box_after_ds": _clone_optional(pred_box_after_ds),
        "pred_score_after_ds": _clone_optional(pred_score_after_ds),
        "metadata": dict(metadata),
    }
    if quality_gt_boxes is not None:
        observation["quality_gt_boxes"] = _clone_optional(quality_gt_boxes)
    return observation


class DualSpaceRefinementDiagnostics(object):
    """Accumulate matched before/after IoU statistics and save JSON/optional CSV."""

    def __init__(self, dual_space_config=None, iou_function=None):
        self.config = resolve_dual_space_diagnostics(dual_space_config)
        self.enabled = bool(self.config["enabled"])
        self._iou_function = iou_function or official_pairwise_rotated_bev_iou
        if not callable(self._iou_function):
            raise TypeError("iou_function must be callable")
        self.scene_count = 0
        self.proposal_count_before = 0
        self.proposal_count_after = 0
        self.rescued_proposal_count = 0
        self._iou_before = []
        self._iou_after = []
        self._deltas = []
        self._per_object = []

    @classmethod
    def from_model(cls, model):
        config = getattr(model, "dual_space_config", None)
        if not getattr(model, "dual_space_enabled", False):
            config = None
        return cls(config)

    def update_inference_result(self, inference_result, scene_id):
        """Consume one result returned by ``inference_utils`` when enabled."""
        if not self.enabled:
            return None
        if not isinstance(inference_result, dict):
            raise TypeError("inference_result must be a mapping")
        if OBSERVATION_KEY not in inference_result:
            raise RuntimeError(
                "diagnostics enabled but Dual-Space refinement observation is missing"
            )
        return self.update_observation(
            inference_result[OBSERVATION_KEY],
            inference_result.get("gt_box_tensor"),
            scene_id,
        )

    def update_observation(self, observation, gt_boxes, scene_id):
        """Match using BEFORE boxes, then compare the same proposal/GT pairs."""
        if not self.enabled:
            return None
        if not isinstance(observation, dict):
            raise TypeError("observation must be a mapping")
        before = observation.get("pred_box_before_ds")
        after = observation.get("pred_box_after_ds")
        scores = observation.get("pred_score_before_ds")
        metadata = observation.get("metadata", {})
        before_count = _optional_box_count(before, "pred_box_before_ds")
        after_count = _optional_box_count(after, "pred_box_after_ds")
        gt_count = _optional_box_count(gt_boxes, "gt_box_tensor")
        original_after_indices = _resolve_original_indices(
            before_count, after_count, metadata
        )
        rescued_count = _nonnegative_metadata_int(
            metadata, "rescued_proposal_count", after_count - before_count
        )
        if rescued_count != after_count - before_count:
            raise ValueError(
                "rescued_proposal_count must equal after_count - before_count"
            )
        if scores is not None:
            if not torch.is_tensor(scores) or scores.ndim != 1:
                raise ValueError("pred_score_before_ds must have shape [N]")
            if int(scores.shape[0]) != before_count:
                raise ValueError("before score count must match before boxes")

        self.scene_count += 1
        self.proposal_count_before += before_count
        self.proposal_count_after += after_count
        self.rescued_proposal_count += rescued_count
        scene_record = {
            "scene_id": str(scene_id),
            "proposal_count_before": before_count,
            "proposal_count_after": after_count,
            "rescued_proposal_count": rescued_count,
            "matched_count": 0,
        }
        if before_count == 0 or gt_count == 0:
            return scene_record

        if after is None:
            raise ValueError("after boxes are required when before boxes are present")
        index_tensor = torch.as_tensor(
            original_after_indices, dtype=torch.long, device=after.device
        )
        original_after = after.index_select(0, index_tensor)
        before_iou = self._iou_function(before, gt_boxes)
        after_iou = self._iou_function(original_after, gt_boxes)
        if not isinstance(before_iou, np.ndarray) or before_iou.shape != (
            before_count, gt_count
        ):
            raise ValueError("iou_function returned an invalid BEFORE matrix")
        if not isinstance(after_iou, np.ndarray) or after_iou.shape != (
            before_count, gt_count
        ):
            raise ValueError("iou_function returned an invalid AFTER matrix")
        matches = deterministic_before_matching(
            before_iou, self.config["match_iou_min"]
        )
        scene_record["matched_count"] = len(matches)
        for proposal_index, gt_index, iou_before in matches:
            iou_after = float(after_iou[proposal_index, gt_index])
            delta = iou_after - float(iou_before)
            self._iou_before.append(float(iou_before))
            self._iou_after.append(iou_after)
            self._deltas.append(delta)
            if self.config["save_per_object"]:
                score = None if scores is None else float(scores[proposal_index].item())
                self._per_object.append(
                    {
                        "scene_id": str(scene_id),
                        "proposal_index": proposal_index,
                        "gt_index": gt_index,
                        "score": score,
                        "iou_before": float(iou_before),
                        "iou_after": iou_after,
                        "delta_iou": delta,
                    }
                )
        return scene_record

    def summary(self):
        """Return a JSON-serializable aggregate with threshold crossings."""
        matched_count = len(self._deltas)
        epsilon = float(self.config["improvement_epsilon"])
        improved = [value for value in self._deltas if value > epsilon]
        worsened = [value for value in self._deltas if value < -epsilon]
        unchanged_count = matched_count - len(improved) - len(worsened)
        denominator = float(matched_count) if matched_count else 1.0
        result = {
            "iou_definition": "official_rotated_bev_polygon_iou",
            "matching": "before_iou_descending_one_to_one",
            "match_iou_min": float(self.config["match_iou_min"]),
            "improvement_epsilon": epsilon,
            "scene_count": self.scene_count,
            "proposal_count_before": self.proposal_count_before,
            "proposal_count_after": self.proposal_count_after,
            "rescued_proposal_count": self.rescued_proposal_count,
            "matched_count": matched_count,
            "mean_iou_before": _mean(self._iou_before),
            "mean_iou_after": _mean(self._iou_after),
            "mean_delta_iou": _mean(self._deltas),
            "median_delta_iou": float(median(self._deltas)) if self._deltas else 0.0,
            "mean_positive_delta": _mean(improved),
            "mean_negative_delta": _mean(worsened),
            "improved_count": len(improved),
            "improved_fraction": len(improved) / denominator,
            "worsened_count": len(worsened),
            "worsened_fraction": len(worsened) / denominator,
            "unchanged_count": unchanged_count,
            "unchanged_fraction": unchanged_count / denominator,
        }
        for threshold in self.config["thresholds"]:
            label = _threshold_label(threshold)
            result["cross_up_%s" % label] = sum(
                before < threshold <= after
                for before, after in zip(self._iou_before, self._iou_after)
            )
            result["cross_down_%s" % label] = sum(
                after < threshold <= before
                for before, after in zip(self._iou_before, self._iou_after)
            )
        return result

    def save(self, output_dir, suffix=None):
        """Save enabled diagnostics; disabled observers create no files."""
        if not self.enabled:
            return None
        if not isinstance(output_dir, str) or not output_dir:
            raise ValueError("output_dir must be a non-empty path string")
        suffix_text = _safe_suffix(suffix)
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(
            output_dir, "dual_space_refinement_stats%s.json" % suffix_text
        )
        with open(json_path, "w", encoding="utf-8") as stream:
            json.dump(self.summary(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        if self.config["save_per_object"]:
            csv_path = os.path.join(
                output_dir, "dual_space_refinement_objects%s.csv" % suffix_text
            )
            fields = (
                "scene_id", "proposal_index", "gt_index", "score",
                "iou_before", "iou_after", "delta_iou",
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self._per_object)
        print("[DualSpace Diagnostics] saved %s" % json_path)
        return json_path


def _validate_corner_boxes(boxes, name):
    if not torch.is_tensor(boxes) or boxes.ndim != 3:
        raise ValueError("%s must have shape [N,8,3] or [N,4,2]" % name)
    if tuple(boxes.shape[1:]) not in ((4, 2), (8, 3)):
        raise ValueError("%s must have shape [N,8,3] or [N,4,2]" % name)
    if not torch.is_floating_point(boxes):
        raise TypeError("%s must use a floating-point dtype" % name)


def _optional_box_count(boxes, name):
    if boxes is None:
        return 0
    _validate_corner_boxes(boxes, name)
    return int(boxes.shape[0])


def _clone_optional(value):
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise TypeError("diagnostic tensors must be torch.Tensor or None")
    return value.detach().clone()


def _resolve_original_indices(before_count, after_count, metadata):
    if not isinstance(metadata, dict):
        raise TypeError("refinement metadata must be a mapping")
    indices = metadata.get("original_after_indices")
    if indices is None:
        if before_count != after_count:
            raise RuntimeError(
                "proposal count changed without explicit original_after_indices metadata"
            )
        indices = tuple(range(before_count))
    if not isinstance(indices, (tuple, list)) or len(indices) != before_count:
        raise ValueError("original_after_indices must map every BEFORE proposal")
    normalized = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("original_after_indices must contain integers")
        if index < 0 or index >= after_count:
            raise ValueError("original_after_indices contains an out-of-range index")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError("original_after_indices must be unique")
    return tuple(normalized)


def _nonnegative_metadata_int(metadata, key, default):
    value = metadata.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % key)
    return value


def _mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _threshold_label(value):
    return ("%.6f" % float(value)).rstrip("0").rstrip(".")


def _safe_suffix(suffix):
    if suffix is None or suffix == "":
        return ""
    if not isinstance(suffix, str):
        raise TypeError("diagnostics suffix must be a string")
    if not all(character.isalnum() or character in ("-", "_") for character in suffix):
        raise ValueError("diagnostics suffix may contain only letters, digits, '-' and '_'")
    return "_" + suffix

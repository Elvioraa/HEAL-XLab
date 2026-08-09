"""Detached training proposal sampling for the full Dual-Space framework."""

import math

import torch
import torch.nn as nn

from opencood.models.sub_modules.dual_space_box_coder import (
    boxes_lwh_to_hwl,
    pairwise_rotated_bev_iou_hwl,
    wrap_angle,
)


class DualSpaceDetectorProposalDecoder(object):
    """Decode HEAL dense heads with the repository's validated decoder path.

    This wrapper has no parameters and therefore adds no state-dict keys.  It
    maps the normal repository ``postprocess`` schema to
    :class:`CollaborativeProposalDecoder`, then converts its ``xyzlwhr`` output
    back to the Dual-Space/repository ``xyzhwlr`` convention.
    """

    def __init__(self, postprocess_config, dir_args, lidar_range):
        if not isinstance(postprocess_config, dict):
            raise TypeError("postprocess must be a mapping")
        target_args = postprocess_config.get("target_args", {})
        if not isinstance(target_args, dict):
            raise TypeError("postprocess.target_args must be a mapping")
        max_num = _positive_int(
            postprocess_config.get("max_num", 100), "postprocess.max_num"
        )
        decoder_config = {
            "proposal_score_threshold": target_args.get(
                "score_threshold", 0.2
            ),
            "proposal_pre_nms_topk": max(512, max_num * 4),
            "proposal_post_nms_topk": max_num,
            "proposal_nms_threshold": postprocess_config.get(
                "nms_thresh", 0.15
            ),
        }
        from opencood.models.sub_modules.pact_cbea_object_stage3_utils import (
            CollaborativeProposalDecoder,
        )

        self.decoder = CollaborativeProposalDecoder(
            decoder_config, dir_args, lidar_range
        )

    def decode(self, cls_preds, reg_preds, dir_preds, anchor_box):
        """Return detached hwl proposal and score lists for a dense batch."""
        proposals_lwh, scores = self.decoder.decode(
            cls_preds, reg_preds, dir_preds, anchor_box
        )
        return tuple(boxes_lwh_to_hwl(boxes) for boxes in proposals_lwh), tuple(scores)


class DualSpaceTrainingProposalSampler(nn.Module):
    """Create detached hwl proposals and matching GT targets using torch RNG.

    ``source='gt_jitter'`` preserves the DS-V1 path exactly.  ``source='mixed'``
    additionally accepts detached detector proposals, retains only proposals
    matched to a GT with the configured positive rotated-IoU threshold, and
    never injects negative proposals into the geometry-refinement loss.
    """

    def __init__(self, config, max_proposals):
        super().__init__()
        if not isinstance(config, dict):
            raise TypeError("training_proposals must be a mapping")
        self.source = config.get("source")
        if self.source not in ("gt_jitter", "mixed"):
            raise ValueError("training proposal source must be gt_jitter or mixed")
        self.include_gt = _require_bool(config, "include_gt")
        self.jitters_per_gt = _nonnegative_int(config.get("jitters_per_gt"), "jitters_per_gt")
        self.center_xy_std_rel = _nonnegative_real(
            config.get("center_xy_std_rel"), "center_xy_std_rel"
        )
        self.center_z_std_rel = _nonnegative_real(
            config.get("center_z_std_rel"), "center_z_std_rel"
        )
        self.log_size_std = _nonnegative_real(config.get("log_size_std"), "log_size_std")
        self.yaw_std_rad = math.radians(
            _nonnegative_real(config.get("yaw_std_deg"), "yaw_std_deg")
        )
        self.max_proposals = _positive_int(max_proposals, "max_proposals")
        predicted = config.get("predicted", {})
        if not isinstance(predicted, dict):
            raise TypeError("training_proposals.predicted must be a mapping")
        self.predicted_enabled = predicted.get("enabled", False)
        if type(self.predicted_enabled) is not bool:
            raise TypeError("training_proposals.predicted.enabled must be bool")
        if self.source == "mixed":
            if not self.predicted_enabled:
                raise ValueError("mixed proposals require predicted.enabled=true")
            self.predicted_max_per_scene = _positive_int(
                predicted.get("max_per_scene"), "predicted.max_per_scene"
            )
            self.predicted_min_score = predicted.get("min_score")
            if self.predicted_min_score is not None:
                self.predicted_min_score = _unit_interval(
                    self.predicted_min_score, "predicted.min_score"
                )
            self.predicted_positive_iou_min = _unit_interval(
                predicted.get("positive_iou_min"),
                "predicted.positive_iou_min",
            )
        else:
            self.predicted_max_per_scene = 0
            self.predicted_min_score = None
            self.predicted_positive_iou_min = 0.0
        if (
            self.source == "gt_jitter"
            and not self.include_gt
            and self.jitters_per_gt == 0
        ):
            raise ValueError("proposal sampler would generate no proposals")
        self.last_stats = {}

    def forward(
        self,
        gt_boxes,
        gt_mask,
        with_jitter=True,
        predicted_boxes=None,
        predicted_scores=None,
    ):
        """Return ``(proposals, targets)`` in ``[x,y,z,h,w,l,yaw]`` order."""
        if type(with_jitter) is not bool:
            raise TypeError("with_jitter must be bool")
        if not torch.is_tensor(gt_boxes) or gt_boxes.ndim != 2 or gt_boxes.shape[1] != 7:
            raise ValueError("gt_boxes must have shape [M,7] in hwl order")
        if not torch.is_floating_point(gt_boxes):
            raise TypeError("gt_boxes must use a floating-point dtype")
        if not torch.is_tensor(gt_mask) or gt_mask.ndim != 1:
            raise ValueError("gt_mask must have shape [M]")
        if gt_mask.shape[0] != gt_boxes.shape[0] or gt_mask.device != gt_boxes.device:
            raise ValueError("gt_mask must match gt_boxes on length and device")
        valid_gt = gt_boxes[gt_mask.to(dtype=torch.bool)]
        if valid_gt.numel() == 0:
            empty = gt_boxes.new_empty((0, 7))
            self.last_stats = {
                "gt_jitter_count": 0,
                "predicted_input_count": 0,
                "predicted_positive_count": 0,
                "proposal_count": 0,
            }
            return empty.detach(), empty.detach()
        if not bool((valid_gt[:, 3:6] > 0).all()):
            raise ValueError("valid GT height, width, and length must be positive")
        if not bool(torch.isfinite(valid_gt).all()):
            raise ValueError("valid GT boxes must contain only finite values")

        proposal_parts = []
        target_parts = []
        if self.include_gt:
            proposal_parts.append(valid_gt.clone())
            target_parts.append(valid_gt)
        jitter_count = self.jitters_per_gt if with_jitter else 0
        for _ in range(jitter_count):
            jittered = valid_gt.clone()
            noise = torch.randn_like(valid_gt)
            jittered[:, 0] += noise[:, 0] * valid_gt[:, 5] * self.center_xy_std_rel
            jittered[:, 1] += noise[:, 1] * valid_gt[:, 4] * self.center_xy_std_rel
            jittered[:, 2] += noise[:, 2] * valid_gt[:, 3] * self.center_z_std_rel
            jittered[:, 3:6] *= torch.exp(noise[:, 3:6] * self.log_size_std)
            jittered[:, 6] = wrap_angle(
                valid_gt[:, 6] + noise[:, 6] * self.yaw_std_rad
            )
            proposal_parts.append(jittered)
            target_parts.append(valid_gt)

        base_count = sum(int(part.shape[0]) for part in proposal_parts)
        predicted_input_count = 0
        predicted_positive_count = 0
        if self.source == "mixed":
            predicted_boxes, predicted_scores = _validate_predicted_inputs(
                gt_boxes, predicted_boxes, predicted_scores
            )
            predicted_input_count = int(predicted_boxes.shape[0])
            boxes = predicted_boxes.detach()
            scores = predicted_scores.detach()
            if self.predicted_min_score is not None:
                score_mask = scores >= self.predicted_min_score
                boxes = boxes[score_mask]
                scores = scores[score_mask]
            if boxes.shape[0] > self.predicted_max_per_scene:
                order = torch.argsort(scores, descending=True)[
                    :self.predicted_max_per_scene
                ]
                boxes = boxes.index_select(0, order)
            if boxes.shape[0]:
                iou = pairwise_rotated_bev_iou_hwl(boxes, valid_gt.detach())
                matched_iou, matched_indices = iou.max(dim=1)
                positive = matched_iou >= self.predicted_positive_iou_min
                boxes = boxes[positive]
                matched_indices = matched_indices[positive]
                predicted_positive_count = int(boxes.shape[0])
                if boxes.shape[0]:
                    proposal_parts.append(boxes)
                    target_parts.append(valid_gt.index_select(0, matched_indices))

        if proposal_parts:
            proposals = torch.cat(proposal_parts, dim=0)[:self.max_proposals]
            targets = torch.cat(target_parts, dim=0)[:self.max_proposals]
        else:
            proposals = gt_boxes.new_empty((0, 7))
            targets = gt_boxes.new_empty((0, 7))
        self.last_stats = {
            "gt_jitter_count": base_count,
            "predicted_input_count": predicted_input_count,
            "predicted_positive_count": predicted_positive_count,
            "proposal_count": int(proposals.shape[0]),
        }
        # Proposal geometry is a sampling decision and must not receive object loss.
        return proposals.detach(), targets.detach()


def _validate_predicted_inputs(reference, boxes, scores):
    if not torch.is_tensor(boxes) or boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("predicted_boxes must have shape [P,7] in hwl order")
    if not torch.is_floating_point(boxes):
        raise TypeError("predicted_boxes must use a floating-point dtype")
    if not torch.is_tensor(scores) or scores.ndim != 1:
        raise ValueError("predicted_scores must have shape [P]")
    if not torch.is_floating_point(scores):
        raise TypeError("predicted_scores must use a floating-point dtype")
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError("predicted_boxes and predicted_scores counts must match")
    if boxes.device != reference.device or scores.device != reference.device:
        raise ValueError("predicted proposals, scores, and GT must share a device")
    if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all()):
        raise ValueError("predicted proposals and scores must be finite")
    if boxes.numel() and not bool((boxes[:, 3:6] > 0).all()):
        raise ValueError("predicted box height, width, and length must be positive")
    return boxes.to(dtype=reference.dtype), scores.to(dtype=reference.dtype)


def _require_bool(config, key):
    value = config.get(key)
    if type(value) is not bool:
        raise TypeError("training_proposals.%s must be bool" % key)
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return value


def _nonnegative_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("%s must be a non-negative real number" % name)
    return float(value)


def _unit_interval(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a real number" % name)
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("%s must be in [0,1]" % name)
    return value

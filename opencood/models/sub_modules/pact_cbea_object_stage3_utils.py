"""Proposal, matching, freezing, and checkpoint utilities for object Stage 3."""

from collections import OrderedDict
import math
import os

import torch

from opencood.models.sub_modules.pact_cbea_object_refiner import (
    repository_hwl_to_sampler_lwh,
    wrap_to_pi,
)


OBJECT_STAGE3_PREFIX = "object_stage3_refiner."
OBJECT_STAGE3_VERSION = 1


class CollaborativeProposalDecoder(object):
    """Decode frozen collaborative detector maps into ego-frame proposals.

    Dense regression output is decoded in repository ``xyzhwlr`` order, then
    explicitly converted to Stage 3 ``xyzlwhr`` order. Thresholding, direction
    correction, range filtering, and rotated NMS run without gradients.
    """

    def __init__(self, cfg, dir_args, lidar_range):
        self.score_threshold = float(cfg.get("proposal_score_threshold", 0.2))
        self.pre_nms_topk = int(cfg.get("proposal_pre_nms_topk", 512))
        self.post_nms_topk = int(cfg.get("proposal_post_nms_topk", 100))
        self.nms_threshold = float(cfg.get("proposal_nms_threshold", 0.15))
        self.dir_offset = float(dir_args.get("dir_offset", 0.7853))
        self.num_bins = int(dir_args.get("num_bins", 2))
        self.lidar_range = tuple(float(value) for value in lidar_range)
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("proposal_score_threshold must lie in [0,1]")
        if self.pre_nms_topk <= 0 or self.post_nms_topk <= 0:
            raise ValueError("proposal top-K limits must be positive")
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ValueError("proposal_nms_threshold must lie in [0,1]")
        if self.num_bins <= 0:
            raise ValueError("dir_args.num_bins must be positive")
        if len(self.lidar_range) != 6:
            raise ValueError("lidar_range must contain six values")

    @torch.no_grad()
    def decode(self, cls_preds, reg_preds, dir_preds, anchor_box):
        """Return detached proposal and score lists, one tensor per scene."""
        self._validate_dense_inputs(cls_preds, reg_preds, dir_preds, anchor_box)
        from opencood.data_utils.post_processor.voxel_postprocessor import (
            VoxelPostprocessor,
        )
        from opencood.utils import box_utils

        batch_size = cls_preds.shape[0]
        probability = torch.sigmoid(
            cls_preds.detach().float().permute(0, 2, 3, 1)
        ).reshape(batch_size, -1)
        anchors = anchor_box.detach().to(
            device=reg_preds.device, dtype=torch.float32
        )
        decoded_hwl = VoxelPostprocessor.delta_to_boxes3d(
            reg_preds.detach().float(), anchors
        )
        direction = dir_preds.detach().float().permute(
            0, 2, 3, 1
        ).contiguous().reshape(batch_size, -1, self.num_bins)

        proposals = []
        scores_out = []
        period = 2.0 * math.pi / float(self.num_bins)
        for batch_idx in range(batch_size):
            keep = torch.nonzero(
                probability[batch_idx] > self.score_threshold,
                as_tuple=False,
            ).flatten()
            if keep.numel() == 0:
                proposals.append(decoded_hwl.new_empty((0, 7)))
                scores_out.append(probability.new_empty((0,)))
                continue

            scores = probability[batch_idx, keep]
            if keep.numel() > self.pre_nms_topk:
                scores, order = torch.topk(
                    scores, k=self.pre_nms_topk, largest=True, sorted=True
                )
                keep = keep[order]
            boxes_hwl = decoded_hwl[batch_idx, keep].clone()
            dir_labels = direction[batch_idx, keep].argmax(dim=-1)
            dir_rot = _limit_period(
                boxes_hwl[:, 6] - self.dir_offset, 0.0, period
            )
            boxes_hwl[:, 6] = (
                dir_rot
                + self.dir_offset
                + period * dir_labels.to(boxes_hwl.dtype)
            )
            boxes_hwl[:, 6] = wrap_to_pi(boxes_hwl[:, 6])

            corners = box_utils.boxes_to_corners_3d(boxes_hwl, order="hwl")
            range_mask = box_utils.get_mask_for_boxes_within_range_torch(
                corners, self.lidar_range
            )
            boxes_hwl = boxes_hwl[range_mask]
            scores = scores[range_mask]
            corners = corners[range_mask]
            if boxes_hwl.shape[0] == 0:
                proposals.append(decoded_hwl.new_empty((0, 7)))
                scores_out.append(probability.new_empty((0,)))
                continue

            boxes_lwh_all = repository_hwl_to_sampler_lwh(boxes_hwl)
            nms_keep = rotated_nms_indices(
                boxes_lwh_all,
                scores,
                threshold=self.nms_threshold,
                max_boxes=self.post_nms_topk,
            )
            boxes_lwh = boxes_lwh_all[nms_keep]
            proposals.append(boxes_lwh.detach())
            scores_out.append(scores[nms_keep].detach())
        return proposals, scores_out

    def _validate_dense_inputs(self, cls_preds, reg_preds, dir_preds, anchor_box):
        for name, value in (
                ("cls_preds", cls_preds),
                ("reg_preds", reg_preds),
                ("dir_preds", dir_preds)):
            if not isinstance(value, torch.Tensor) or value.ndim != 4:
                raise ValueError("%s must be a 4D tensor" % name)
            if not torch.is_floating_point(value):
                raise TypeError("%s must be floating point" % name)
        if cls_preds.shape[0] != reg_preds.shape[0] or cls_preds.shape[0] != dir_preds.shape[0]:
            raise ValueError("dense prediction batch sizes must match")
        if cls_preds.device != reg_preds.device or cls_preds.device != dir_preds.device:
            raise ValueError("dense predictions must share a device")
        if not isinstance(anchor_box, torch.Tensor) or anchor_box.shape[-1] != 7:
            raise ValueError("anchor_box must be a tensor ending in dimension 7")


@torch.no_grad()
def pairwise_rotated_bev_iou(boxes_a, boxes_b):
    """Compute rotated BEV IoU for ``xyzlwhr`` boxes without gradients.

    The repository CUDA/CPU extension is preferred when importable. A pure
    PyTorch convex-polygon fallback keeps CPU matching available when that
    optional compiled extension is absent; it never participates in ROI
    sampling or a gradient path.
    """
    _validate_box_matrix(boxes_a, "boxes_a")
    _validate_box_matrix(boxes_b, "boxes_b")
    if boxes_a.device != boxes_b.device:
        raise ValueError("boxes_a and boxes_b must share a device")
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))

    try:
        from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import (
            boxes_bev_iou_cpu,
        )
        result = boxes_bev_iou_cpu(
            boxes_a.detach().float().cpu(), boxes_b.detach().float().cpu()
        )
        return torch.as_tensor(
            result, device=boxes_a.device, dtype=boxes_a.dtype
        )
    except (ImportError, OSError, RuntimeError, AttributeError):
        return _pairwise_rotated_iou_torch(boxes_a, boxes_b)


@torch.no_grad()
def rotated_nms_indices(boxes, scores, threshold=0.15, max_boxes=100):
    """Return greedy rotated-NMS indices for ``xyzlwhr`` boxes.

    IoU computation follows :func:`pairwise_rotated_bev_iou`, so the
    repository extension is used when available and the deterministic
    PyTorch fallback is used otherwise. This avoids depending on the optional
    Shapely path, which is not required by object Stage 3.
    """
    _validate_box_matrix(boxes, "boxes")
    if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
        raise ValueError("scores must have shape [P]")
    if scores.shape[0] != boxes.shape[0] or scores.device != boxes.device:
        raise ValueError("scores must align with boxes")
    threshold = float(threshold)
    max_boxes = int(max_boxes)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("NMS threshold must lie in [0,1]")
    if max_boxes <= 0:
        raise ValueError("max_boxes must be positive")
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    iou = pairwise_rotated_bev_iou(boxes, boxes)
    order = torch.argsort(scores, descending=True)
    kept = []
    while order.numel() > 0 and len(kept) < max_boxes:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        order = remaining[iou[current, remaining] <= threshold]
    return torch.stack(kept).to(dtype=torch.long)


@torch.no_grad()
def match_proposals_to_gt(
        proposals,
        gt_boxes,
        positive_iou_threshold=0.55,
        ignore_iou_threshold=0.35):
    """Match each proposal to at most one GT box using rotated BEV IoU."""
    _validate_box_matrix(proposals, "proposals")
    _validate_box_matrix(gt_boxes, "gt_boxes")
    positive_iou_threshold = float(positive_iou_threshold)
    ignore_iou_threshold = float(ignore_iou_threshold)
    if not 0.0 <= ignore_iou_threshold <= positive_iou_threshold <= 1.0:
        raise ValueError(
            "IoU thresholds must satisfy 0 <= ignore <= positive <= 1"
        )
    count = proposals.shape[0]
    if count == 0:
        return {
            "positive_mask": torch.empty(
                (0,), dtype=torch.bool, device=proposals.device
            ),
            "ignore_mask": torch.empty(
                (0,), dtype=torch.bool, device=proposals.device
            ),
            "matched_gt_indices": torch.empty(
                (0,), dtype=torch.long, device=proposals.device
            ),
            "matched_ious": proposals.new_empty((0,)),
            "matched_gt_boxes": proposals.new_empty((0, 7)),
        }
    if gt_boxes.shape[0] == 0:
        return {
            "positive_mask": torch.zeros(
                (count,), dtype=torch.bool, device=proposals.device
            ),
            "ignore_mask": torch.zeros(
                (count,), dtype=torch.bool, device=proposals.device
            ),
            "matched_gt_indices": torch.full(
                (count,), -1, dtype=torch.long, device=proposals.device
            ),
            "matched_ious": proposals.new_zeros((count,)),
            "matched_gt_boxes": proposals.new_zeros((count, 7)),
        }

    iou = pairwise_rotated_bev_iou(proposals, gt_boxes)
    matched_ious, matched_indices = iou.max(dim=1)
    positive = matched_ious >= positive_iou_threshold
    ignore = torch.logical_and(
        matched_ious >= ignore_iou_threshold, ~positive
    )
    matched_boxes = gt_boxes[matched_indices].clone()
    return {
        "positive_mask": positive,
        "ignore_mask": ignore,
        "matched_gt_indices": matched_indices,
        "matched_ious": matched_ious,
        "matched_gt_boxes": matched_boxes,
    }


@torch.no_grad()
def rotated_nms_sampler_boxes(boxes, scores, threshold=0.15, max_boxes=100):
    """Apply rotated NMS to ``xyzlwhr`` boxes and preserve scores."""
    if boxes.shape[0] == 0:
        return boxes.detach(), scores.detach(), torch.empty(
            (0,), dtype=torch.long, device=boxes.device
        )
    keep = rotated_nms_indices(boxes, scores, threshold, max_boxes)
    return boxes[keep].detach(), scores[keep].detach(), keep


def freeze_except_object_stage3(model, prefix=OBJECT_STAGE3_PREFIX):
    """Freeze every parameter except the dedicated object Stage 3 module."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefix))
    return assert_only_object_stage3_trainable(model, prefix=prefix)


def assert_only_object_stage3_trainable(model, prefix=OBJECT_STAGE3_PREFIX):
    """Return trainable names and fail if any base parameter is trainable."""
    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    invalid = [name for name in trainable if not name.startswith(prefix)]
    if invalid:
        raise RuntimeError(
            "non-Stage-3 parameters are trainable: %s" % ", ".join(invalid)
        )
    if not trainable:
        raise RuntimeError("no object Stage 3 parameters are trainable")
    return trainable


def stage3_named_parameters(model):
    """Yield only validated object Stage 3 named parameters."""
    assert_only_object_stage3_trainable(model)
    for name, parameter in model.named_parameters():
        if name.startswith(OBJECT_STAGE3_PREFIX):
            yield name, parameter


def stage3_parameter_summary(model):
    """Return trainable/frozen parameter names and counts."""
    trainable_names = assert_only_object_stage3_trainable(model)
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_count = sum(
        parameter.numel() for parameter in model.parameters()
        if not parameter.requires_grad
    )
    return {
        "trainable_names": trainable_names,
        "trainable_count": trainable_count,
        "frozen_count": frozen_count,
    }


def load_base_checkpoint_compatible(model, checkpoint_path, require_complete=True):
    """Load a frozen base checkpoint while excluding all Stage 3 parameters."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = _extract_state_dict(checkpoint)
    model_state = model.state_dict()
    expected_base = {
        key: value for key, value in model_state.items()
        if not key.startswith(OBJECT_STAGE3_PREFIX)
    }
    loadable = OrderedDict()
    shape_mismatch = []
    unexpected = []
    for key, value in source.items():
        if key.startswith(OBJECT_STAGE3_PREFIX):
            continue
        if key not in expected_base:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(expected_base[key].shape):
            shape_mismatch.append(
                "%s checkpoint=%r model=%r"
                % (key, tuple(value.shape), tuple(expected_base[key].shape))
            )
        else:
            loadable[key] = value
    missing = sorted(set(expected_base) - set(loadable))
    if shape_mismatch:
        raise RuntimeError(
            "base checkpoint shape mismatch:\n  %s"
            % "\n  ".join(shape_mismatch)
        )
    if require_complete and missing:
        raise RuntimeError(
            "base checkpoint is incomplete; missing %d keys, e.g. %s"
            % (len(missing), missing[0])
        )
    model.load_state_dict(loadable, strict=False)
    return {
        "loaded": len(loadable),
        "missing": missing,
        "unexpected": sorted(unexpected),
        "path": checkpoint_path,
    }


def build_stage3_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        global_step,
        config_snapshot,
        base_checkpoint):
    """Build a Stage-3-only checkpoint payload."""
    refiner = _unwrap_model(model).object_stage3_refiner
    return {
        "object_stage3_version": OBJECT_STAGE3_VERSION,
        "stage3_state_dict": refiner.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config_snapshot": config_snapshot,
        "base_checkpoint": str(base_checkpoint),
    }


def strict_load_stage3_checkpoint(
        model,
        checkpoint_or_path,
        optimizer=None,
        scheduler=None,
        expected_version=OBJECT_STAGE3_VERSION):
    """Strictly validate and load all Stage 3 keys, shapes, and version."""
    checkpoint = checkpoint_or_path
    if isinstance(checkpoint_or_path, (str, os.PathLike)):
        if not os.path.isfile(checkpoint_or_path):
            raise FileNotFoundError(str(checkpoint_or_path))
        checkpoint = torch.load(checkpoint_or_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Stage 3 checkpoint must be a mapping")
    version = checkpoint.get("object_stage3_version")
    if version != expected_version:
        raise RuntimeError(
            "Stage 3 checkpoint version mismatch: expected %r, got %r"
            % (expected_version, version)
        )
    state = checkpoint.get("stage3_state_dict")
    if not isinstance(state, dict):
        raise KeyError("Stage 3 checkpoint lacks stage3_state_dict")
    refiner = _unwrap_model(model).object_stage3_refiner
    expected = refiner.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = [
        key for key in expected.keys() & state.keys()
        if tuple(expected[key].shape) != tuple(state[key].shape)
    ]
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "invalid Stage 3 checkpoint: missing=%s unexpected=%s shape_mismatch=%s"
            % (missing, unexpected, shape_mismatch)
        )
    refiner.load_state_dict(state, strict=True)
    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is None:
            raise KeyError("resume checkpoint lacks optimizer_state_dict")
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is None:
            raise KeyError("resume checkpoint lacks scheduler_state_dict")
        scheduler.load_state_dict(scheduler_state)
    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "base_checkpoint": checkpoint.get("base_checkpoint"),
        "version": version,
    }


def _extract_state_dict(checkpoint):
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    if not isinstance(state, dict):
        raise TypeError("checkpoint is not state-dict-like")
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    )


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _validate_box_matrix(boxes, name):
    if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("%s must have shape [N,7]" % name)
    if not torch.is_floating_point(boxes):
        raise TypeError("%s must be floating point" % name)
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("%s contain NaN or Inf" % name)


def _limit_period(value, offset, period):
    return value - torch.floor(value / period + offset) * period


def _pairwise_rotated_iou_torch(boxes_a, boxes_b):
    corners_a = _boxes_to_ccw_corners(boxes_a)
    corners_b = _boxes_to_ccw_corners(boxes_b)
    result = boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    areas_a = boxes_a[:, 3] * boxes_a[:, 4]
    areas_b = boxes_b[:, 3] * boxes_b[:, 4]
    for index_a in range(boxes_a.shape[0]):
        subject = [point for point in corners_a[index_a]]
        for index_b in range(boxes_b.shape[0]):
            clipped = subject
            clip_polygon = corners_b[index_b]
            for edge_idx in range(4):
                clipped = _clip_polygon(
                    clipped,
                    clip_polygon[edge_idx],
                    clip_polygon[(edge_idx + 1) % 4],
                )
                if not clipped:
                    break
            intersection = _polygon_area(clipped, boxes_a)
            union = areas_a[index_a] + areas_b[index_b] - intersection
            result[index_a, index_b] = intersection / union.clamp_min(1e-7)
    return result


def _boxes_to_ccw_corners(boxes):
    half_l = boxes[:, 3] * 0.5
    half_w = boxes[:, 4] * 0.5
    local_x = torch.stack((-half_l, half_l, half_l, -half_l), dim=1)
    local_y = torch.stack((-half_w, -half_w, half_w, half_w), dim=1)
    cosine = torch.cos(boxes[:, 6]).unsqueeze(1)
    sine = torch.sin(boxes[:, 6]).unsqueeze(1)
    x = boxes[:, 0].unsqueeze(1) + cosine * local_x - sine * local_y
    y = boxes[:, 1].unsqueeze(1) + sine * local_x + cosine * local_y
    return torch.stack((x, y), dim=-1)


def _clip_polygon(subject, edge_start, edge_end):
    if not subject:
        return []
    output = []
    previous = subject[-1]
    previous_inside = _inside_edge(previous, edge_start, edge_end)
    for current in subject:
        current_inside = _inside_edge(current, edge_start, edge_end)
        if current_inside:
            if not previous_inside:
                output.append(
                    _segment_line_intersection(
                        previous, current, edge_start, edge_end
                    )
                )
            output.append(current)
        elif previous_inside:
            output.append(
                _segment_line_intersection(
                    previous, current, edge_start, edge_end
                )
            )
        previous = current
        previous_inside = current_inside
    return output


def _inside_edge(point, edge_start, edge_end):
    edge = edge_end - edge_start
    relative = point - edge_start
    cross = edge[0] * relative[1] - edge[1] * relative[0]
    return bool(cross >= -1e-7)


def _segment_line_intersection(start, end, edge_start, edge_end):
    segment = end - start
    edge = edge_end - edge_start
    denominator = edge[0] * segment[1] - edge[1] * segment[0]
    if bool(denominator.abs() < 1e-8):
        return end
    delta = edge_start - start
    numerator = edge[0] * delta[1] - edge[1] * delta[0]
    return start + (numerator / denominator) * segment


def _polygon_area(points, reference):
    if len(points) < 3:
        return reference.new_tensor(0.0)
    polygon = torch.stack(points)
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * torch.abs(
        torch.sum(x * torch.roll(y, shifts=-1) - y * torch.roll(x, shifts=-1))
    )

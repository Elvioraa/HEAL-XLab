"""Apply trained Open-DCSI residuals to official post-processed boxes."""

import torch

from opencood.models.sub_modules.open_dcsi.geometry_refiner import GeometryRefiner
from opencood.utils.box_utils import boxes_to_corners_3d, corner_to_center_torch


def apply_open_dcsi_box_refinement(
    pred_box_tensor,
    pred_score,
    open_output,
    open_config,
):
    """Refine official NMS survivors; unmatched or zero-delta boxes stay exact."""

    if pred_box_tensor is None or pred_score is None:
        return pred_box_tensor, pred_score
    refinement = open_output.get("geometry_refinement")
    tokens = open_output.get("fused_tokens")
    if refinement is None or tokens is None:
        return pred_box_tensor, pred_score
    deltas = refinement.get("box_deltas_hwl")
    if deltas is None or deltas.numel() == 0 or torch.count_nonzero(deltas).item() == 0:
        return pred_box_tensor, pred_score
    scene_mask = tokens["scenario_index"] == 0
    if not scene_mask.any():
        return pred_box_tensor, pred_score
    token_centers = tokens["centers_ego"][scene_mask]
    token_deltas = deltas[scene_mask]
    finite_tokens = torch.isfinite(token_centers).all(dim=-1) & torch.isfinite(
        token_deltas
    ).all(dim=-1)
    if not finite_tokens.any():
        return pred_box_tensor, pred_score
    token_centers = token_centers[finite_tokens]
    token_deltas = token_deltas[finite_tokens]

    official_boxes = corner_to_center_torch(pred_box_tensor, order="hwl").to(
        device=pred_box_tensor.device, dtype=pred_box_tensor.dtype
    )
    distances = torch.cdist(official_boxes[:, :2], token_centers[:, :2])
    nearest_distance, nearest_index = distances.min(dim=1)
    radius = float(
        open_config["innovation_aggregation"]["geometric_clustering"][
            "center_radius"
        ]
    )
    matched = nearest_distance <= radius
    if not matched.any():
        return pred_box_tensor, pred_score
    selected_delta = torch.zeros_like(official_boxes)
    selected_delta[matched] = token_deltas.index_select(0, nearest_index[matched])
    refined_boxes = GeometryRefiner.apply_deltas(official_boxes, selected_delta)
    if torch.equal(refined_boxes, official_boxes):
        return pred_box_tensor, pred_score
    refined_corners = boxes_to_corners_3d(refined_boxes, order="hwl")
    finite = torch.isfinite(refined_corners).reshape(refined_corners.shape[0], -1).all(
        dim=1
    )
    refined_corners = torch.where(
        finite[:, None, None], refined_corners, pred_box_tensor
    )

    refined_score = pred_score
    if open_config["geometry_refiner"]["predict_confidence_delta"]:
        confidence_delta = refinement["confidence_delta"][scene_mask][finite_tokens]
        score_delta = torch.zeros_like(pred_score)
        score_delta[matched] = confidence_delta.index_select(0, nearest_index[matched])
        refined_score = torch.sigmoid(
            torch.logit(pred_score.clamp(1e-6, 1 - 1e-6)) + score_delta
        )
    open_output["inference_refinement"] = {
        "matched_official_boxes": int(matched.sum().item()),
        "official_box_count": int(pred_box_tensor.shape[0]),
        "nms_stage": "official_pre_refinement",
    }
    return refined_corners, refined_score

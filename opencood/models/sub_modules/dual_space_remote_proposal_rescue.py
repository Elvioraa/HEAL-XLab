"""Inference-only remote proposal rescue for Dual-Space HEAL DS-V4.

The policy is deliberately parameter-free.  It filters per-agent detector
proposals in ego coordinates, removes candidates already covered by the fused
detector, greedily deduplicates the remaining remote candidates by exact
rotated BEV IoU, and appends at most a configured number to the fused pool.
No second NMS is applied to the original fused detections.
"""

import torch

from opencood.models.sub_modules.dual_space_box_coder import (
    pairwise_rotated_bev_iou_hwl,
)


@torch.no_grad()
def rescue_remote_proposals(
    fused_boxes,
    fused_scores,
    remote_boxes,
    remote_scores,
    config,
):
    """Append deterministic remote-only candidates to a fused proposal pool.

    Parameters
    ----------
    fused_boxes : torch.Tensor
        ``[F,7]`` boxes in ego-frame repository ``xyzhwlr`` order.
    fused_scores : torch.Tensor
        ``[F]`` fused detector scores.  These values are returned unchanged.
    remote_boxes, remote_scores : sequence of torch.Tensor
        One detached hwl box tensor and score tensor per agent, already
        transformed/aligned to ego coordinates by the shared HEAL pipeline.
    config : dict
        Validated ``dual_space.remote_proposal_rescue`` mapping.

    Returns
    -------
    tuple
        ``(candidate_boxes, candidate_scores, stats)``.  Rescued candidates
        follow all original fused candidates and retain their remote score.
    """
    _validate_inputs(
        fused_boxes, fused_scores, remote_boxes, remote_scores, config
    )
    include_ego = config["include_ego"]
    min_score = float(config["min_score"])
    dedup_iou = float(config["dedup_iou"])
    max_per_agent = int(config["max_per_agent"])
    max_total_added = int(config["max_total_added"])

    ranked = []
    before_filter = 0
    after_score = 0
    examined = 0
    overlap_rejected = 0
    first_agent = 0 if include_ego else 1
    for agent_index in range(first_agent, len(remote_boxes)):
        boxes = remote_boxes[agent_index]
        scores = remote_scores[agent_index]
        before_filter += int(scores.shape[0])
        score_indices = torch.nonzero(scores >= min_score, as_tuple=False).flatten()
        after_score += int(score_indices.numel())
        if score_indices.numel() == 0:
            continue
        order = torch.argsort(scores.index_select(0, score_indices), descending=True)
        score_indices = score_indices.index_select(0, order)[:max_per_agent]
        boxes = boxes.index_select(0, score_indices)
        scores = scores.index_select(0, score_indices)
        examined += int(boxes.shape[0])

        if fused_boxes.shape[0]:
            overlap = pairwise_rotated_bev_iou_hwl(boxes, fused_boxes).amax(dim=1)
            keep = overlap < dedup_iou
            overlap_rejected += int((~keep).sum().item())
            boxes = boxes[keep]
            scores = scores[keep]
            score_indices = score_indices[keep]

        for local_index in range(int(boxes.shape[0])):
            # Agent and original proposal index make ties deterministic.
            ranked.append(
                (
                    -float(scores[local_index].item()),
                    agent_index,
                    int(score_indices[local_index].item()),
                    boxes[local_index],
                    scores[local_index],
                )
            )

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    kept_boxes = []
    kept_scores = []
    remote_duplicate_rejected = 0
    for _, _, _, box, score in ranked:
        if len(kept_boxes) >= max_total_added:
            break
        if kept_boxes:
            previous = torch.stack(kept_boxes, dim=0)
            duplicate = pairwise_rotated_bev_iou_hwl(
                box.unsqueeze(0), previous
            ).amax() >= dedup_iou
            if bool(duplicate.item()):
                remote_duplicate_rejected += 1
                continue
        kept_boxes.append(box)
        kept_scores.append(score)

    if kept_boxes:
        rescued_boxes = torch.stack(kept_boxes, dim=0)
        rescued_scores = torch.stack(kept_scores, dim=0)
        candidate_boxes = torch.cat((fused_boxes, rescued_boxes), dim=0)
        candidate_scores = torch.cat((fused_scores, rescued_scores), dim=0)
    else:
        rescued_boxes = fused_boxes.new_empty((0, 7))
        rescued_scores = fused_scores.new_empty((0,))
        candidate_boxes = fused_boxes
        candidate_scores = fused_scores

    stats = {
        "fused_proposal_count": int(fused_boxes.shape[0]),
        "remote_candidates_before_filter": before_filter,
        "remote_candidates_after_score": after_score,
        "remote_candidates_deduped": int(rescued_boxes.shape[0]),
        "rescued_added": int(rescued_boxes.shape[0]),
        "remote_examined_count": examined,
        "remote_overlap_rejected_count": overlap_rejected,
        "remote_duplicate_rejected_count": remote_duplicate_rejected,
        "rescued_proposal_count": int(rescued_boxes.shape[0]),
        "candidate_proposal_count": int(candidate_boxes.shape[0]),
    }
    return candidate_boxes, candidate_scores, stats


def _validate_inputs(fused_boxes, fused_scores, remote_boxes, remote_scores, config):
    if not torch.is_tensor(fused_boxes) or fused_boxes.ndim != 2 or fused_boxes.shape[1] != 7:
        raise ValueError("fused_boxes must have shape [F,7] in hwl order")
    if not torch.is_tensor(fused_scores) or fused_scores.ndim != 1:
        raise ValueError("fused_scores must have shape [F]")
    if fused_boxes.shape[0] != fused_scores.shape[0]:
        raise ValueError("fused box and score counts must match")
    if not isinstance(remote_boxes, (tuple, list)) or not isinstance(
        remote_scores, (tuple, list)
    ):
        raise TypeError("remote_boxes and remote_scores must be sequences")
    if len(remote_boxes) != len(remote_scores):
        raise ValueError("remote box and score sequences must have equal length")
    for agent_index, (boxes, scores) in enumerate(zip(remote_boxes, remote_scores)):
        if not torch.is_tensor(boxes) or boxes.ndim != 2 or boxes.shape[1] != 7:
            raise ValueError(
                "remote_boxes[%d] must have shape [P,7]" % agent_index
            )
        if not torch.is_tensor(scores) or scores.ndim != 1:
            raise ValueError(
                "remote_scores[%d] must have shape [P]" % agent_index
            )
        if boxes.shape[0] != scores.shape[0]:
            raise ValueError("remote box and score counts must match")
        if boxes.device != fused_boxes.device or scores.device != fused_boxes.device:
            raise ValueError("all rescue tensors must share a device")
    if not isinstance(config, dict):
        raise TypeError("remote_proposal_rescue config must be a mapping")

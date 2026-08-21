"""Parameter-free safety extensions for the Dual-Space V3 core.

The functions in this module do not own parameters or buffers.  Callers gate
them with the resolved V5/V6 configuration so legacy V3 execution never enters
these paths.
"""

import torch
import torch.nn.functional as F


def build_quality_target_mask(targets, matched_valid, config):
    """Return the V5 validity mask for one flattened agent-target vector.

    Parameters
    ----------
    targets : Tensor[N]
        Quality targets already produced by the real proposal/GT assignment.
    matched_valid : BoolTensor[N] or None
        Matching validity propagated by the proposal sampler.  No matching is
        performed in this function.
    config : mapping
        Resolved ``v5_quality_safe.valid_target_mask`` settings.
    """
    if not torch.is_tensor(targets) or targets.ndim != 1:
        raise ValueError("quality targets must have shape [N]")
    if not isinstance(config, dict):
        raise TypeError("valid_target_mask config must be a mapping")
    mask = torch.ones_like(targets, dtype=torch.bool)
    if config["require_matched"]:
        if not torch.is_tensor(matched_valid):
            raise ValueError("matched_valid is required by the V5 target mask")
        if matched_valid.dtype != torch.bool or matched_valid.shape != targets.shape:
            raise ValueError("matched_valid must be bool with the target shape")
        mask = mask & matched_valid
    if config["require_finite"]:
        mask = mask & torch.isfinite(targets)
    mask = mask & (targets >= float(config["min_target"]))
    mask = mask & (targets <= float(config["max_target"]))
    return mask


def cap_quality_loss_ratio(detection_loss, base_quality_loss, config):
    """Cap weighted quality loss against detached detection loss magnitude.

    The returned scale has no gradient path to either input.  Consequently the
    cap changes only the quality branch's effective gradient strength.
    """
    _require_scalar_loss(detection_loss, "detection_loss")
    _require_scalar_loss(base_quality_loss, "base_quality_loss")
    if not isinstance(config, dict):
        raise TypeError("loss_balance config must be a mapping")
    if not config["enabled"]:
        return base_quality_loss, base_quality_loss.detach().new_ones(())
    if config["mode"] != "ratio_cap":
        raise ValueError("unsupported V5 loss balance mode %r" % config["mode"])
    cap = float(config["max_quality_to_detection_ratio"]) * detection_loss.detach()
    denominator = base_quality_loss.detach() + float(config["eps"])
    scale = torch.clamp(cap / denominator, min=0.0, max=1.0)
    return scale * base_quality_loss, scale


def deterministic_quality_ranking_loss(
    predictions,
    targets,
    proposal_agent_indices,
    valid_mask,
    config,
):
    """Compute deterministic within-scene, within-agent logistic ranking.

    Candidate pairs are enumerated by ascending agent and proposal index, then
    truncated without random sampling.  The returned tuple is
    ``(loss, pair_count, accuracy)``.
    """
    if not torch.is_tensor(predictions) or predictions.ndim != 1:
        raise ValueError("quality predictions must have shape [N]")
    if not torch.is_tensor(targets) or targets.shape != predictions.shape:
        raise ValueError("quality targets must match predictions")
    if (
        not torch.is_tensor(proposal_agent_indices)
        or proposal_agent_indices.ndim != 2
        or proposal_agent_indices.shape != (predictions.shape[0], 2)
    ):
        raise ValueError("proposal_agent_indices must have shape [N,2]")
    if (
        not torch.is_tensor(valid_mask)
        or valid_mask.dtype != torch.bool
        or valid_mask.shape != predictions.shape
    ):
        raise ValueError("ranking valid_mask must be bool [N]")
    if not isinstance(config, dict):
        raise TypeError("ranking config must be a mapping")

    zero = predictions.sum() * 0.0
    if not config["enabled"] or float(config["weight"]) <= 0.0:
        return zero, 0, predictions.detach().new_zeros(())
    if config["loss"] != "logistic":
        raise ValueError("unsupported V5 ranking loss %r" % config["loss"])

    minimum_gap = float(config["min_target_gap"])
    maximum_pairs = int(config["max_pairs_per_sample"])
    selected_pairs = []
    agents = torch.unique(proposal_agent_indices[:, 1], sorted=True)
    for agent in agents.detach().cpu().tolist():
        indices = (
            valid_mask
            & (proposal_agent_indices[:, 1] == int(agent))
        ).nonzero(as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        order = torch.argsort(proposal_agent_indices[indices, 0])
        indices = indices.index_select(0, order)
        for left_position in range(int(indices.numel()) - 1):
            left = int(indices[left_position].item())
            for right_position in range(left_position + 1, int(indices.numel())):
                right = int(indices[right_position].item())
                gap = targets[left] - targets[right]
                if bool(torch.abs(gap).detach() >= minimum_gap):
                    selected_pairs.append((left, right))
                    if len(selected_pairs) >= maximum_pairs:
                        break
            if len(selected_pairs) >= maximum_pairs:
                break
        if len(selected_pairs) >= maximum_pairs:
            break

    if not selected_pairs:
        return zero, 0, predictions.detach().new_zeros(())
    left = torch.tensor(
        [pair[0] for pair in selected_pairs],
        device=predictions.device,
        dtype=torch.long,
    )
    right = torch.tensor(
        [pair[1] for pair in selected_pairs],
        device=predictions.device,
        dtype=torch.long,
    )
    target_delta = targets.index_select(0, left) - targets.index_select(0, right)
    sign = torch.sign(target_delta).detach()
    prediction_delta = (
        predictions.index_select(0, left) - predictions.index_select(0, right)
    )
    loss = F.softplus(-sign * prediction_delta).mean()
    accuracy = ((sign * prediction_delta.detach()) > 0).to(
        dtype=predictions.dtype
    ).mean()
    return loss, len(selected_pairs), accuracy


def apply_residual_norm_cap(inputs, residual, config, feature_dim):
    """Return a norm-capped residual and detached diagnostic tensors.

    Norms are computed along the explicitly supplied feature/channel
    dimension.  For the current object and context ROI adapters the real input
    shape is ``[M,C,Rh,Rw]`` and callers pass ``feature_dim=1``.
    """
    if not torch.is_tensor(inputs) or not torch.is_tensor(residual):
        raise TypeError("inputs and residual must be tensors")
    if inputs.shape != residual.shape:
        raise ValueError("inputs and residual must have identical shapes")
    if inputs.ndim < 2:
        raise ValueError("adapter inputs must have at least two dimensions")
    feature_dim = int(feature_dim)
    if feature_dim < 0:
        feature_dim += inputs.ndim
    if feature_dim <= 0 or feature_dim >= inputs.ndim:
        raise ValueError("feature_dim must identify a non-batch dimension")
    if not isinstance(config, dict):
        raise TypeError("V6 residual config must be a mapping")
    if config["mode"] != "norm_cap":
        raise ValueError("unsupported V6 residual mode %r" % config["mode"])

    eps = float(config["eps"])
    input_norm = torch.linalg.vector_norm(inputs, dim=feature_dim, keepdim=True)
    residual_norm = torch.linalg.vector_norm(
        residual, dim=feature_dim, keepdim=True
    )
    raw_ratio = residual_norm / (input_norm + eps)
    cap_scale = torch.clamp(
        float(config["max_residual_ratio"]) / (raw_ratio + eps),
        max=1.0,
    )
    safe_residual = float(config["residual_scale"]) * cap_scale * residual
    safe_norm = torch.linalg.vector_norm(
        safe_residual, dim=feature_dim, keepdim=True
    )
    safe_ratio = safe_norm / (input_norm + eps)
    return safe_residual, {
        "input_norm": input_norm.detach(),
        "raw_residual_norm": residual_norm.detach(),
        "raw_residual_ratio": raw_ratio.detach(),
        "safe_residual_ratio": safe_ratio.detach(),
        "cap_scale": cap_scale.detach(),
    }


def _require_scalar_loss(value, name):
    if not torch.is_tensor(value) or value.ndim != 0:
        raise ValueError("%s must be a scalar tensor" % name)

"""Object residual and optional quality losses for Dual-Space HEAL."""

import torch
import torch.nn.functional as F

from opencood.models.sub_modules.dual_space_extensions import (
    build_quality_target_mask,
    cap_quality_loss_ratio,
    deterministic_quality_ranking_loss,
)


def compute_dual_space_object_loss(payload):
    """Compute dimension-averaged SmoothL1 DS-V1 training loss and stats.

    Stage1 supervises every valid agent residual and the uniform consensus.
    Stage2 has one active modality per sample and therefore uses only the
    individual adaptation term; duplicating it as a one-agent consensus would
    merely double the configured learning signal.
    """
    if not isinstance(payload, dict) or not payload.get("enabled", False):
        raise ValueError("payload must be an enabled dual-space object result")
    scenes = payload.get("scenes")
    if not isinstance(scenes, tuple):
        raise TypeError("dual-space object scenes must be a tuple")
    config = payload["loss_config"]
    mode = payload["mode"]
    v5_config = payload.get("v5_quality_safe")
    v5_enabled = bool(v5_config is not None and mode == "stage2_adapt")

    individual_terms = []
    consensus_terms = []
    quality_terms = []
    quality_zero_terms = []
    ranking_terms = []
    ranking_pair_count = 0
    ranking_correct_sum = 0.0
    quality_valid_count = 0
    quality_predictions = []
    quality_targets = []
    reference = None
    for scene in scenes:
        predictions = scene["individual_residuals"]
        targets = scene["individual_targets"].to(dtype=predictions.dtype)
        reference = predictions if reference is None else reference
        if predictions.numel():
            individual_terms.append(_residual_smooth_l1_per_object(predictions, targets))
        if mode == "stage1_anchor" and bool(scene["any_valid"].any()):
            mask = scene["any_valid"]
            consensus_terms.append(
                _residual_smooth_l1_per_object(
                    scene["fused_residuals"][mask],
                    scene["target_residuals"][mask].to(
                        dtype=scene["fused_residuals"].dtype
                    ),
                )
            )
        if "individual_quality" in scene:
            predictions_q = scene["individual_quality"]
            targets_q = scene["quality_targets"].to(dtype=predictions_q.dtype)
            if predictions_q.shape != targets_q.shape:
                raise ValueError("quality prediction and target shapes must match")
            if predictions_q.numel():
                if not v5_enabled:
                    quality_terms.append(
                        F.smooth_l1_loss(
                            predictions_q, targets_q, reduction="none"
                        )
                    )
                    quality_predictions.append(predictions_q.detach())
                    quality_targets.append(targets_q.detach())
                    continue

                quality_zero_terms.append(predictions_q.sum() * 0.0)
                valid_quality = torch.ones_like(targets_q, dtype=torch.bool)
                if v5_config["valid_target_mask"]["enabled"]:
                    valid_quality = build_quality_target_mask(
                        targets_q,
                        scene.get("quality_matched_valid"),
                        v5_config["valid_target_mask"],
                    )
                quality_error = F.smooth_l1_loss(
                    predictions_q, targets_q, reduction="none"
                )
                quality_terms.append(quality_error[valid_quality])
                if bool(valid_quality.any()):
                    quality_predictions.append(predictions_q[valid_quality].detach())
                    quality_targets.append(targets_q[valid_quality].detach())
                quality_valid_count += int(valid_quality.sum().item())
                if v5_config["ranking"]["enabled"]:
                    ranking_loss, pair_count, ranking_accuracy = (
                        deterministic_quality_ranking_loss(
                            predictions_q,
                            targets_q,
                            scene["quality_pair_indices"],
                            valid_quality,
                            v5_config["ranking"],
                        )
                    )
                    if pair_count:
                        ranking_terms.append(ranking_loss * float(pair_count))
                    ranking_pair_count += pair_count
                    ranking_correct_sum += float(ranking_accuracy.item()) * pair_count

    if reference is None:
        raise ValueError("dual-space object payload contains no scenes")
    zero = reference.sum() * 0.0
    individual_loss = torch.cat(individual_terms).mean() if individual_terms else zero
    consensus_loss = torch.cat(consensus_terms).mean() if consensus_terms else zero
    if not v5_enabled:
        quality_loss = torch.cat(quality_terms).mean() if quality_terms else zero
    else:
        nonempty_quality_terms = [term for term in quality_terms if term.numel()]
        quality_loss = (
            torch.cat(nonempty_quality_terms).mean()
            if nonempty_quality_terms
            else (
                torch.stack(quality_zero_terms).sum()
                if quality_zero_terms else zero
            )
        )
    detection_objective = (
        float(config["individual_loss_weight"]) * individual_loss
        + float(config["consensus_loss_weight"]) * consensus_loss
    )
    unweighted = detection_objective
    quality_enabled = bool(payload.get("quality_enabled", False))
    quality_objective = zero
    balanced_quality_objective = zero
    quality_scale = zero.detach().new_ones(())
    ranking_loss = (
        torch.stack(ranking_terms).sum() / float(ranking_pair_count)
        if ranking_pair_count else zero
    )
    if quality_enabled:
        base_quality = float(config["quality_loss_weight"]) * quality_loss
        if v5_enabled:
            balanced_quality_objective, quality_scale = cap_quality_loss_ratio(
                detection_objective,
                base_quality,
                v5_config["loss_balance"],
            )
            quality_objective = balanced_quality_objective + (
                float(v5_config["ranking"]["weight"]) * ranking_loss
            )
        else:
            quality_objective = base_quality
        unweighted = unweighted + quality_objective
    object_loss = float(config["object_loss_weight"]) * unweighted
    if payload.get("gradient_diagnostics_enabled", False):
        payload["_diagnostic_losses"] = {
            "detection": float(config["object_loss_weight"])
            * detection_objective,
            "quality": float(config["object_loss_weight"])
            * quality_objective,
        }
    stats = {
        "dual_space_enabled": True,
        "dual_space_object_loss": float(object_loss.detach().item()),
        "dual_space_individual_loss": float(individual_loss.detach().item()),
        "dual_space_consensus_loss": float(consensus_loss.detach().item()),
        "dual_space_valid_object_ratio": payload["stats"]["valid_object_ratio"],
        "dual_space_mean_roi_coverage": payload["stats"]["mean_roi_coverage"],
        "dual_space_object_roi_count": payload["stats"]["object_roi_count"],
        "dual_space_valid_agent_object_pairs": payload["stats"][
            "valid_agent_object_pairs"
        ],
    }
    if quality_enabled:
        stats.update(
            {
                "dual_space_quality_loss": float(quality_loss.detach().item()),
                "dual_space_mean_pred_quality": float(
                    torch.cat(quality_predictions).mean().item()
                ) if quality_predictions else 0.0,
                "dual_space_mean_quality_target": float(
                    torch.cat(quality_targets).mean().item()
                ) if quality_targets else 0.0,
            }
        )
    if v5_enabled:
        weighted_quality = balanced_quality_objective.detach()
        denominator = detection_objective.detach().abs() + float(
            v5_config["loss_balance"]["eps"]
        )
        stats.update(
            {
                "dual_space_v5_valid_quality_count": quality_valid_count,
                "dual_space_v5_raw_quality_loss": float(
                    quality_loss.detach().item()
                ),
                "dual_space_v5_weighted_quality_loss": float(
                    weighted_quality.item()
                ),
                "dual_space_v5_quality_scale": float(quality_scale.item()),
                "dual_space_v5_quality_to_detection_ratio": float(
                    (weighted_quality.abs() / denominator).item()
                ),
                "dual_space_v5_ranking_pair_count": ranking_pair_count,
                "dual_space_v5_ranking_loss": float(ranking_loss.detach().item()),
                "dual_space_v5_ranking_accuracy": (
                    ranking_correct_sum / float(ranking_pair_count)
                    if ranking_pair_count else 0.0
                ),
            }
        )
    return object_loss, stats


def _residual_smooth_l1_per_object(predictions, targets):
    # Dimension mean first; the caller then averages all valid objects globally.
    return F.smooth_l1_loss(predictions, targets, reduction="none").mean(dim=-1)

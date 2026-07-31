"""Loss for frozen-base PACT-CBEA object-level Stage 3 training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PactCbeaObjectStage3Loss(nn.Module):
    """Heteroscedastic per-agent and fused residual regression loss."""

    def __init__(self, args):
        super(PactCbeaObjectStage3Loss, self).__init__()
        args = args or {}
        self.fused_loss_weight = float(args.get("fused_loss_weight", 1.0))
        self.agent_loss_weight = float(args.get("agent_loss_weight", 1.0))
        self.variance_reg_weight = float(args.get("variance_reg_weight", 0.01))
        self.heteroscedastic_logvar_weight = float(
            args.get("heteroscedastic_logvar_weight", 0.5)
        )
        self.smooth_l1_beta = float(args.get("smooth_l1_beta", 1.0))
        self.loss_dict = {}
        for name in (
                "fused_loss_weight", "agent_loss_weight",
                "variance_reg_weight", "heteroscedastic_logvar_weight"):
            if getattr(self, name) < 0:
                raise ValueError("%s must be non-negative" % name)
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")

    def forward(self, output_dict, target_dict=None):
        """Compute Stage 3 loss from the model's variable-length scene list."""
        del target_dict
        stage3 = output_dict.get("object_stage3")
        if not isinstance(stage3, dict) or not stage3.get("enabled", False):
            raise KeyError("output_dict lacks enabled object_stage3 outputs")
        zero_anchor = stage3.get("zero_loss_anchor")
        if not isinstance(zero_anchor, torch.Tensor):
            raise KeyError("object_stage3 lacks zero_loss_anchor")

        fused_numerator = zero_anchor
        agent_numerator = zero_anchor
        variance_numerator = zero_anchor
        fused_elements = 0
        agent_elements = 0
        variance_elements = 0
        positive_count = 0
        positive_agent_slots = 0
        valid_positive_agents = 0
        fallback_count = 0
        proposal_count = 0
        log_variance_values = []
        normalized_weight_values = []
        center_errors = []
        size_errors = []
        yaw_errors = []

        for scene in stage3.get("scenes", []):
            positive = scene["positive_mask"].bool()
            valid = scene["valid_mask"].bool()
            proposal_count += int(positive.shape[0])
            fallback_count += int(scene["fallback_mask"].sum().item())
            if positive.numel() == 0 or not bool(positive.any()):
                continue

            target = scene["target_residuals"]
            agent_residuals = scene["agent_residuals"]
            log_variances = scene["agent_log_variances"]
            fused_residual = scene["fused_residual"]
            weights = scene["normalized_agent_weights"]
            positive_count += int(positive.sum().item())
            positive_agent_slots += int(positive.sum().item()) * valid.shape[1]

            positive_valid = torch.logical_and(valid, positive.unsqueeze(1))
            valid_positive_agents += int(positive_valid.sum().item())
            agent_mask = positive_valid.unsqueeze(-1).to(agent_residuals.dtype)
            target_agents = target.unsqueeze(1).expand_as(agent_residuals)
            base_error = F.smooth_l1_loss(
                agent_residuals,
                target_agents,
                reduction="none",
                beta=self.smooth_l1_beta,
            )
            heteroscedastic = (
                torch.exp(-log_variances) * base_error
                + self.heteroscedastic_logvar_weight * log_variances
            )
            agent_numerator = agent_numerator + (heteroscedastic * agent_mask).sum()
            agent_elements += int(positive_valid.sum().item()) * 7

            variance_numerator = variance_numerator + (
                log_variances.square() * agent_mask
            ).sum()
            variance_elements += int(positive_valid.sum().item()) * 7

            fused_error = F.smooth_l1_loss(
                fused_residual[positive],
                target[positive],
                reduction="none",
                beta=self.smooth_l1_beta,
            )
            fused_numerator = fused_numerator + fused_error.sum()
            fused_elements += int(positive.sum().item()) * 7

            residual_abs = (
                fused_residual[positive] - target[positive]
            ).abs()
            center_errors.append(residual_abs[:, 0:3])
            size_errors.append(residual_abs[:, 3:6])
            yaw_errors.append(residual_abs[:, 6:7])
            if bool(positive_valid.any()):
                log_variance_values.append(
                    log_variances[positive_valid].reshape(-1)
                )
                normalized_weight_values.append(
                    weights[positive_valid].reshape(-1)
                )

        fused_loss = fused_numerator / max(fused_elements, 1)
        agent_loss = agent_numerator / max(agent_elements, 1)
        variance_regularization = variance_numerator / max(variance_elements, 1)
        total_loss = (
            self.fused_loss_weight * fused_loss
            + self.agent_loss_weight * agent_loss
            + self.variance_reg_weight * variance_regularization
        )
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("object Stage 3 total loss is NaN or Inf")

        log_stats = _tensor_stats(log_variance_values, zero_anchor)
        self.loss_dict = {
            "total_loss": total_loss.detach(),
            "fused_loss": fused_loss.detach(),
            "agent_loss": agent_loss.detach(),
            "variance_regularization": variance_regularization.detach(),
            "positive_count": positive_count,
            "valid_agent_ratio": (
                float(valid_positive_agents) / float(max(positive_agent_slots, 1))
            ),
            "fallback_ratio": float(fallback_count) / float(max(proposal_count, 1)),
            "mean_log_variance": log_stats[0],
            "min_log_variance": log_stats[1],
            "max_log_variance": log_stats[2],
            "mean_normalized_weight": _mean_or_zero(
                normalized_weight_values, zero_anchor
            ),
            "center_residual_error": _mean_or_zero(center_errors, zero_anchor),
            "size_residual_error": _mean_or_zero(size_errors, zero_anchor),
            "yaw_residual_error": _mean_or_zero(yaw_errors, zero_anchor),
        }
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        """Print and optionally write the current Stage 3 diagnostics."""
        prefix = "ObjectStage3%s" % suffix
        values = self.loss_dict
        print(
            "[%s] epoch %d batch %d/%d total %.6f fused %.6f agent %.6f "
            "var %.6f positives %d valid %.4f fallback %.4f"
            % (
                prefix,
                epoch,
                batch_id,
                batch_len,
                _as_float(values.get("total_loss", 0.0)),
                _as_float(values.get("fused_loss", 0.0)),
                _as_float(values.get("agent_loss", 0.0)),
                _as_float(values.get("variance_regularization", 0.0)),
                int(values.get("positive_count", 0)),
                float(values.get("valid_agent_ratio", 0.0)),
                float(values.get("fallback_ratio", 0.0)),
            )
        )
        if writer is not None:
            step = epoch * max(batch_len, 1) + batch_id
            for key, value in values.items():
                writer.add_scalar("%s/%s" % (prefix, key), _as_float(value), step)


def _mean_or_zero(values, reference):
    if not values:
        return reference.detach().new_tensor(0.0)
    return torch.cat([value.reshape(-1) for value in values]).mean().detach()


def _tensor_stats(values, reference):
    if not values:
        zero = reference.detach().new_tensor(0.0)
        return zero, zero.clone(), zero.clone()
    packed = torch.cat(values)
    return packed.mean().detach(), packed.min().detach(), packed.max().detach()


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)

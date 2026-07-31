"""Shared object-level refinement components for PACT-CBEA Stage 3.

The repository detector uses ``[x,y,z,h,w,l,yaw]`` (``hwl``), while the
object ROI sampler and every component in this module use
``[x,y,z,l,w,h,yaw]`` (``lwh``). Conversion is always explicit at the model
boundary.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


SAMPLER_BOX_ORDER = "xyzlwhr"
REPOSITORY_BOX_ORDER = "xyzhwlr"


def repository_hwl_to_sampler_lwh(boxes):
    """Convert ``[...,7]`` boxes from ``xyzhwlr`` to ``xyzlwhr``."""
    _validate_boxes(boxes, "boxes")
    return boxes[..., [0, 1, 2, 5, 4, 3, 6]]


def sampler_lwh_to_repository_hwl(boxes):
    """Convert ``[...,7]`` boxes from ``xyzlwhr`` to ``xyzhwlr``."""
    _validate_boxes(boxes, "boxes")
    return boxes[..., [0, 1, 2, 5, 4, 3, 6]]


def wrap_to_pi(angle):
    """Wrap angles to ``[-pi, pi]`` with differentiable torch operations."""
    if not isinstance(angle, torch.Tensor):
        raise TypeError("angle must be a torch.Tensor")
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class ObjectResidualCoder(object):
    """Encode and decode proposal-relative ``xyzlwhr`` box residuals.

    Center offsets x/y are normalized by the proposal BEV diagonal, z by
    proposal height, sizes are log ratios, and yaw is wrapped to ``[-pi,pi]``.
    Invalid or extremely small proposal/target sizes are rejected by default.
    """

    def __init__(self, min_size=1e-3, max_log_scale=5.0, strict_sizes=True):
        self.min_size = float(min_size)
        self.max_log_scale = float(max_log_scale)
        self.strict_sizes = bool(strict_sizes)
        if not math.isfinite(self.min_size) or self.min_size <= 0:
            raise ValueError("min_size must be finite and positive")
        if not math.isfinite(self.max_log_scale) or self.max_log_scale <= 0:
            raise ValueError("max_log_scale must be finite and positive")

    def encode(self, proposals, target_boxes):
        """Encode target ``xyzlwhr`` boxes relative to proposals."""
        self._validate_pair(proposals, target_boxes)
        if proposals.numel() == 0:
            return proposals.new_empty(proposals.shape)
        proposal_sizes = self._safe_sizes(proposals[..., 3:6], "proposal")
        target_sizes = self._safe_sizes(target_boxes[..., 3:6], "target")
        diagonal = torch.sqrt(
            proposal_sizes[..., 0] ** 2 + proposal_sizes[..., 1] ** 2
        ).clamp_min(self.min_size)

        residuals = torch.empty_like(proposals)
        residuals[..., 0] = (
            target_boxes[..., 0] - proposals[..., 0]
        ) / diagonal
        residuals[..., 1] = (
            target_boxes[..., 1] - proposals[..., 1]
        ) / diagonal
        residuals[..., 2] = (
            target_boxes[..., 2] - proposals[..., 2]
        ) / proposal_sizes[..., 2]
        residuals[..., 3:6] = torch.log(target_sizes / proposal_sizes)
        residuals[..., 6] = wrap_to_pi(
            target_boxes[..., 6] - proposals[..., 6]
        )
        self._ensure_finite(residuals, "encoded residuals")
        return residuals

    def decode(self, proposals, residuals):
        """Decode residuals into refined ``xyzlwhr`` boxes."""
        self._validate_pair(proposals, residuals, second_name="residuals")
        if proposals.numel() == 0:
            return proposals.new_empty(proposals.shape)
        proposal_sizes = self._safe_sizes(proposals[..., 3:6], "proposal")
        diagonal = torch.sqrt(
            proposal_sizes[..., 0] ** 2 + proposal_sizes[..., 1] ** 2
        ).clamp_min(self.min_size)

        decoded = torch.empty_like(proposals)
        decoded[..., 0] = proposals[..., 0] + residuals[..., 0] * diagonal
        decoded[..., 1] = proposals[..., 1] + residuals[..., 1] * diagonal
        decoded[..., 2] = (
            proposals[..., 2] + residuals[..., 2] * proposal_sizes[..., 2]
        )
        log_scale = residuals[..., 3:6].clamp(
            min=-self.max_log_scale, max=self.max_log_scale
        )
        decoded[..., 3:6] = proposal_sizes * torch.exp(log_scale)
        decoded[..., 6] = wrap_to_pi(proposals[..., 6] + residuals[..., 6])
        self._ensure_finite(decoded, "decoded boxes")
        return decoded

    def _safe_sizes(self, sizes, label):
        if self.strict_sizes and bool((sizes < self.min_size).any()):
            raise ValueError(
                "%s box length, width, and height must be >= %g"
                % (label, self.min_size)
            )
        return sizes.clamp_min(self.min_size)

    @staticmethod
    def _ensure_finite(value, label):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("%s contain NaN or Inf" % label)

    @staticmethod
    def _validate_pair(first, second, second_name="target_boxes"):
        _validate_boxes(first, "proposals")
        _validate_boxes(second, second_name)
        if first.shape != second.shape:
            raise ValueError(
                "proposals and %s must have identical shape, got %r and %r"
                % (second_name, tuple(first.shape), tuple(second.shape))
            )
        if first.device != second.device:
            raise ValueError("box tensors must be on the same device")
        if first.dtype != second.dtype:
            raise ValueError("box tensors must have the same dtype")


class SharedObjectRefiner(nn.Module):
    """Predict per-agent box residual and log variance from ROI features.

    Inputs are ``roi_features[P,A,C,Rh,Rw]`` and ``valid_mask[P,A]``. The same
    convolutional encoder and linear heads are applied independently to every
    proposal-agent pair; no modality or agent-position encoding is used.
    """

    def __init__(
            self,
            in_channels,
            hidden_dim=128,
            min_log_variance=-4.0,
            max_log_variance=4.0,
            initial_log_variance=0.0):
        super(SharedObjectRefiner, self).__init__()
        in_channels = int(in_channels)
        hidden_dim = int(hidden_dim)
        if in_channels <= 0 or hidden_dim <= 0:
            raise ValueError("in_channels and hidden_dim must be positive")
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)
        if not self.min_log_variance < self.max_log_variance:
            raise ValueError("min_log_variance must be below max_log_variance")
        if not (
                self.min_log_variance <= initial_log_variance
                <= self.max_log_variance):
            raise ValueError("initial_log_variance must lie inside clamp range")

        groups = _largest_group_divisor(hidden_dim, 32)
        self.roi_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.residual_head = nn.Linear(hidden_dim, 7)
        self.log_variance_head = nn.Linear(hidden_dim, 7)

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.constant_(self.log_variance_head.bias, initial_log_variance)

    def forward(self, roi_features, valid_mask):
        """Return residuals and clamped log variances with shape ``[P,A,7]``."""
        self._validate_inputs(roi_features, valid_mask)
        proposal_count, agent_count, channels, roi_h, roi_w = roi_features.shape
        if proposal_count == 0:
            empty = roi_features.new_empty((0, agent_count, 7))
            return empty, empty.clone()

        packed = roi_features.reshape(
            proposal_count * agent_count, channels, roi_h, roi_w
        )
        if not bool(torch.isfinite(packed).all()):
            raise FloatingPointError("roi_features contain NaN or Inf")
        embedding = self.roi_encoder(packed).flatten(1)
        residuals = self.residual_head(embedding).reshape(
            proposal_count, agent_count, 7
        )
        log_variances = self.log_variance_head(embedding).reshape(
            proposal_count, agent_count, 7
        ).clamp(min=self.min_log_variance, max=self.max_log_variance)
        if not bool(torch.isfinite(residuals).all()):
            raise FloatingPointError("agent residuals contain NaN or Inf")
        if not bool(torch.isfinite(log_variances).all()):
            raise FloatingPointError("agent log variances contain NaN or Inf")
        return residuals, log_variances

    @staticmethod
    def _validate_inputs(roi_features, valid_mask):
        if not isinstance(roi_features, torch.Tensor) or roi_features.ndim != 5:
            raise ValueError("roi_features must have shape [P,A,C,Rh,Rw]")
        if not torch.is_floating_point(roi_features):
            raise TypeError("roi_features must use a floating-point dtype")
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape [P,A]")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be a bool tensor")
        if tuple(valid_mask.shape) != tuple(roi_features.shape[:2]):
            raise ValueError("valid_mask [P,A] must match roi_features")
        if valid_mask.device != roi_features.device:
            raise ValueError("valid_mask and roi_features must share a device")


def precision_fuse(
        agent_residuals,
        agent_log_variances,
        valid_mask,
        eps=1e-6,
        coverage=None,
        use_coverage_weight=False):
    """Fuse per-agent residuals with per-dimension inverse variances.

    Returns ``fused_residual[P,7]``, normalized weights ``[P,A,7]``, and a
    boolean ``fallback_mask[P]``. If no agent is valid for a proposal, its
    fused residual and all normalized weights are zero.
    """
    _validate_precision_inputs(
        agent_residuals, agent_log_variances, valid_mask, coverage
    )
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if agent_residuals.shape[0] == 0:
        proposal_count, agent_count, _ = agent_residuals.shape
        return (
            agent_residuals.new_empty((proposal_count, 7)),
            agent_residuals.new_empty((proposal_count, agent_count, 7)),
            torch.empty(
                (proposal_count,), dtype=torch.bool,
                device=agent_residuals.device,
            ),
        )

    precision = torch.exp(-agent_log_variances)
    weight_mask = valid_mask.unsqueeze(-1).to(agent_residuals.dtype)
    if use_coverage_weight:
        if coverage is None:
            raise ValueError("coverage is required when use_coverage_weight=true")
        weight_mask = weight_mask * coverage.unsqueeze(-1).clamp(0.0, 1.0)
    weighted_precision = precision * weight_mask
    precision_sum = weighted_precision.sum(dim=1)
    fallback_mask = ~valid_mask.any(dim=1)
    normalized_weights = weighted_precision / precision_sum.clamp_min(eps).unsqueeze(1)
    normalized_weights = torch.where(
        fallback_mask.view(-1, 1, 1),
        torch.zeros_like(normalized_weights),
        normalized_weights,
    )
    fused_residual = (normalized_weights * agent_residuals).sum(dim=1)
    fused_residual = torch.where(
        fallback_mask.unsqueeze(-1),
        torch.zeros_like(fused_residual),
        fused_residual,
    )
    if not bool(torch.isfinite(fused_residual).all()):
        raise FloatingPointError("fused residual contains NaN or Inf")
    return fused_residual, normalized_weights, fallback_mask


def _validate_precision_inputs(residuals, log_variances, valid_mask, coverage):
    if not isinstance(residuals, torch.Tensor) or residuals.ndim != 3:
        raise ValueError("agent_residuals must have shape [P,A,7]")
    if residuals.shape[-1] != 7 or not torch.is_floating_point(residuals):
        raise ValueError("agent_residuals must be floating [P,A,7]")
    if not isinstance(log_variances, torch.Tensor) or log_variances.shape != residuals.shape:
        raise ValueError("agent_log_variances must match agent_residuals")
    if log_variances.dtype != residuals.dtype or log_variances.device != residuals.device:
        raise ValueError("residuals and log variances must share dtype/device")
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be a bool tensor")
    if tuple(valid_mask.shape) != tuple(residuals.shape[:2]):
        raise ValueError("valid_mask must have shape [P,A]")
    if valid_mask.device != residuals.device:
        raise ValueError("valid_mask must share the residual device")
    if coverage is not None:
        if not isinstance(coverage, torch.Tensor) or coverage.shape != valid_mask.shape:
            raise ValueError("coverage must have shape [P,A]")
        if coverage.device != residuals.device:
            raise ValueError("coverage must share the residual device")
    if not bool(torch.isfinite(residuals).all()):
        raise FloatingPointError("agent residuals contain NaN or Inf")
    if not bool(torch.isfinite(log_variances).all()):
        raise FloatingPointError("agent log variances contain NaN or Inf")


def _validate_boxes(boxes, name):
    if not isinstance(boxes, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor" % name)
    if boxes.ndim < 2 or boxes.shape[-1] != 7:
        raise ValueError("%s must have shape [...,7]" % name)
    if not torch.is_floating_point(boxes):
        raise TypeError("%s must use a floating-point dtype" % name)
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("%s contain NaN or Inf" % name)


def _largest_group_divisor(channels, maximum):
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    return 1

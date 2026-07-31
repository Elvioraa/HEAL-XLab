"""Differentiable rotated ROI sampling from per-agent BEV features.

Coordinate contract
-------------------
Proposals are expressed in the ego frame as ``[x, y, z, l, w, h, yaw]``.
The BEV x axis maps to feature-map columns (W), and the BEV y axis maps to
feature-map rows (H). Positive yaw rotates counter-clockwise from +x toward
+y. ROI columns run from the proposal's local -x to +x direction (length);
ROI rows run from local -y to +y (width). Samples are taken at bin centers.

``ego_to_agent[a]`` maps homogeneous points from the ego frame into agent
``a``'s local frame. Each ROI sampling point is constructed in the ego frame
and transformed by that matrix before conversion to the agent feature grid.

This module uses ``grid_sample`` with ``align_corners=False`` and zero
padding. For an agent-frame metric point ``(x, y)``, feature indices are

``column = (x - x_min) / (resolution_x * feature_stride_x) - 0.5``
``row    = (y - y_min) / (resolution_y * feature_stride_y) - 0.5``.

Coverage is the fraction of ROI sample centers whose normalized grid
coordinates both lie in ``[-1, 1]``.
"""

from __future__ import division

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BEVGeometry:
    """Metric geometry of an agent BEV feature map.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        Metric BEV bounds in every agent's local coordinate frame.
    resolution_x, resolution_y : float
        Base-grid metres per cell along x and y before feature downsampling.
    feature_stride_x, feature_stride_y : float
        Feature downsampling factors relative to the base grid. Their product
        with the corresponding resolution is metres per feature-map cell.

    The resulting width and height must match the sampled feature tensor.
    This explicit check prevents silent range/stride mismatches.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution_x: float
    resolution_y: float
    feature_stride_x: float = 1.0
    feature_stride_y: float = 1.0

    def __post_init__(self):
        values = {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y,
            "feature_stride_x": self.feature_stride_x,
            "feature_stride_y": self.feature_stride_y,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("BEVGeometry.%s must be a real number" % name)
            if not math.isfinite(float(value)):
                raise ValueError("BEVGeometry.%s must be finite" % name)
        if self.x_max <= self.x_min:
            raise ValueError("BEVGeometry.x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("BEVGeometry.y_max must be greater than y_min")
        if self.resolution_x <= 0 or self.resolution_y <= 0:
            raise ValueError("BEVGeometry resolutions must be positive")
        if self.feature_stride_x <= 0 or self.feature_stride_y <= 0:
            raise ValueError("BEVGeometry feature strides must be positive")

    @property
    def cell_size_x(self):
        """Metres represented by one feature-map column."""
        return float(self.resolution_x * self.feature_stride_x)

    @property
    def cell_size_y(self):
        """Metres represented by one feature-map row."""
        return float(self.resolution_y * self.feature_stride_y)

    def validate_feature_shape(self, height, width):
        """Raise when ``(height, width)`` disagrees with this geometry."""
        expected_width = (self.x_max - self.x_min) / self.cell_size_x
        expected_height = (self.y_max - self.y_min) / self.cell_size_y
        width_matches = math.isclose(
            expected_width, float(width), rel_tol=1e-6, abs_tol=1e-6
        )
        height_matches = math.isclose(
            expected_height, float(height), rel_tol=1e-6, abs_tol=1e-6
        )
        if not (width_matches and height_matches):
            raise ValueError(
                "agent_features has spatial shape HxW=%dx%d, but BEVGeometry "
                "range/resolution/stride expects %.6fx%.6f"
                % (height, width, expected_height, expected_width)
            )


class RotatedBEVROISampler(nn.Module):
    """Sample ego-frame rotated proposals from every agent BEV feature map.

    ``forward`` accepts ``agent_features[A,C,H,W]``, ``proposals[P,7]`` in
    ``[x,y,z,l,w,h,yaw]`` order, and ``ego_to_agent[A,4,4]``. It returns:

    - ROI features ``[P,A,C,Rh,Rw]``;
    - boolean validity ``[P,A]``;
    - coverage fractions ``[P,A]`` in ``[0,1]``.

    Gradients flow from ROI features to ``agent_features`` through bilinear
    ``grid_sample``. No CPU geometry path or non-PyTorch dependency is used.
    """

    def __init__(self, bev_geometry, roi_size=(7, 7), min_coverage=0.5):
        super(RotatedBEVROISampler, self).__init__()
        if not isinstance(bev_geometry, BEVGeometry):
            raise TypeError("bev_geometry must be a BEVGeometry instance")
        self.bev_geometry = bev_geometry
        self.roi_size = _validate_roi_size(roi_size)
        self.min_coverage = _validate_min_coverage(min_coverage)

    def forward(self, agent_features, proposals, ego_to_agent):
        """Apply rotated BEV ROI sampling using the configured geometry."""
        return rotated_bev_roi_sample(
            agent_features=agent_features,
            proposals=proposals,
            ego_to_agent=ego_to_agent,
            bev_geometry=self.bev_geometry,
            roi_size=self.roi_size,
            min_coverage=self.min_coverage,
        )

    def extra_repr(self):
        return "roi_size=%r, min_coverage=%g, align_corners=False" % (
            self.roi_size,
            self.min_coverage,
        )


def rotated_bev_roi_sample(
        agent_features,
        proposals,
        ego_to_agent,
        bev_geometry,
        roi_size=(7, 7),
        min_coverage=0.5):
    """Sample rotated proposal regions from all agent BEV features.

    Parameters
    ----------
    agent_features : torch.Tensor
        Floating tensor ``[A,C,H,W]`` on the target output device.
    proposals : torch.Tensor
        Floating tensor ``[P,7]`` in ego coordinates and
        ``[x,y,z,l,w,h,yaw]`` order. Length, width, and height must be
        positive. Proposal tensors must share the feature device.
    ego_to_agent : torch.Tensor
        Floating tensor ``[A,4,4]``. Matrix ``a`` transforms ego-frame
        homogeneous points into agent ``a``'s local frame. It must share the
        feature device.
    bev_geometry : BEVGeometry
        Explicit metric range, resolution, and feature stride.
    roi_size : sequence of two int
        ``(Rh,Rw)`` output resolution. Samples are ROI-bin centers.
    min_coverage : float
        Threshold in ``(0,1]``. ``valid_mask[p,a]`` is true exactly when
        ``coverage[p,a] >= min_coverage``.

    Returns
    -------
    roi_features : torch.Tensor
        Tensor ``[P,A,C,Rh,Rw]`` with the feature dtype and device.
    valid_mask : torch.Tensor
        Bool tensor ``[P,A]`` on the feature device.
    coverage : torch.Tensor
        Tensor ``[P,A]`` with the feature dtype and device.
    """
    _validate_inputs(agent_features, proposals, ego_to_agent, bev_geometry)
    roi_height, roi_width = _validate_roi_size(roi_size)
    min_coverage = _validate_min_coverage(min_coverage)

    agent_count, channels, height, width = agent_features.shape
    proposal_count = proposals.shape[0]
    bev_geometry.validate_feature_shape(height, width)

    if proposal_count == 0:
        roi_features = agent_features.new_empty(
            (0, agent_count, channels, roi_height, roi_width)
        )
        valid_mask = torch.empty(
            (0, agent_count), dtype=torch.bool, device=agent_features.device
        )
        coverage = agent_features.new_empty((0, agent_count))
        return roi_features, valid_mask, coverage

    dtype = agent_features.dtype
    device = agent_features.device
    proposals_work = proposals.to(dtype=dtype)
    transforms_work = ego_to_agent.to(dtype=dtype)

    # Bin centers in proposal-local coordinates. Columns follow local +x
    # (length), rows follow local +y (width).
    column_fraction = (
        (torch.arange(roi_width, device=device, dtype=dtype) + 0.5)
        / float(roi_width)
        - 0.5
    )
    row_fraction = (
        (torch.arange(roi_height, device=device, dtype=dtype) + 0.5)
        / float(roi_height)
        - 0.5
    )
    local_x = (
        column_fraction.view(1, 1, roi_width)
        * proposals_work[:, 3].view(proposal_count, 1, 1)
    ).expand(-1, roi_height, -1)
    local_y = (
        row_fraction.view(1, roi_height, 1)
        * proposals_work[:, 4].view(proposal_count, 1, 1)
    ).expand(-1, -1, roi_width)

    yaw = proposals_work[:, 6].view(proposal_count, 1, 1)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    ego_x = (
        proposals_work[:, 0].view(proposal_count, 1, 1)
        + cos_yaw * local_x
        - sin_yaw * local_y
    )
    ego_y = (
        proposals_work[:, 1].view(proposal_count, 1, 1)
        + sin_yaw * local_x
        + cos_yaw * local_y
    )
    ego_z = proposals_work[:, 2].view(proposal_count, 1, 1).expand(
        -1, roi_height, roi_width
    )
    ones = torch.ones_like(ego_x)
    ego_points = torch.stack((ego_x, ego_y, ego_z, ones), dim=-1)

    # [P,Rh,Rw,4] -> [P,A,Rh,Rw,4]. The matrix direction is intentionally
    # ego-to-agent, matching the public contract.
    agent_points = torch.einsum(
        "aij,prsj->parsi", transforms_work, ego_points
    )
    agent_x = agent_points[..., 0]
    agent_y = agent_points[..., 1]

    # align_corners=False maps feature-cell centers to
    # -1 + 1/W ... 1 - 1/W (and analogously for H).
    grid_x = (
        2.0
        * (agent_x - float(bev_geometry.x_min))
        / (float(width) * bev_geometry.cell_size_x)
        - 1.0
    )
    grid_y = (
        2.0
        * (agent_y - float(bev_geometry.y_min))
        / (float(height) * bev_geometry.cell_size_y)
        - 1.0
    )
    sample_grid = torch.stack((grid_x, grid_y), dim=-1)

    inside = torch.logical_and(sample_grid >= -1.0, sample_grid <= 1.0)
    inside = torch.logical_and(inside[..., 0], inside[..., 1])
    coverage = inside.to(dtype=dtype).mean(dim=(-2, -1))
    valid_mask = coverage >= min_coverage

    packed_features = agent_features.unsqueeze(0).expand(
        proposal_count, -1, -1, -1, -1
    ).reshape(proposal_count * agent_count, channels, height, width)
    packed_grid = sample_grid.reshape(
        proposal_count * agent_count, roi_height, roi_width, 2
    )
    sampled = F.grid_sample(
        packed_features,
        packed_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    roi_features = sampled.reshape(
        proposal_count, agent_count, channels, roi_height, roi_width
    )
    return roi_features, valid_mask, coverage


def _validate_inputs(agent_features, proposals, ego_to_agent, bev_geometry):
    if not isinstance(agent_features, torch.Tensor):
        raise TypeError("agent_features must be a torch.Tensor")
    if agent_features.ndim != 4:
        raise ValueError(
            "agent_features must have shape [A,C,H,W], got %r"
            % (tuple(agent_features.shape),)
        )
    if not torch.is_floating_point(agent_features):
        raise TypeError("agent_features must use a floating-point dtype")
    if any(size <= 0 for size in agent_features.shape):
        raise ValueError("agent_features dimensions A,C,H,W must all be positive")

    if not isinstance(proposals, torch.Tensor):
        raise TypeError("proposals must be a torch.Tensor")
    if proposals.ndim != 2 or proposals.shape[1] != 7:
        raise ValueError(
            "proposals must have shape [P,7] in [x,y,z,l,w,h,yaw] order, "
            "got %r" % (tuple(proposals.shape),)
        )
    if not torch.is_floating_point(proposals):
        raise TypeError("proposals must use a floating-point dtype")
    if proposals.device != agent_features.device:
        raise ValueError("proposals and agent_features must be on the same device")

    if not isinstance(ego_to_agent, torch.Tensor):
        raise TypeError("ego_to_agent must be a torch.Tensor")
    expected_transform_shape = (agent_features.shape[0], 4, 4)
    if tuple(ego_to_agent.shape) != expected_transform_shape:
        raise ValueError(
            "ego_to_agent must have shape [A,4,4] matching agent_features; "
            "expected %r, got %r"
            % (expected_transform_shape, tuple(ego_to_agent.shape))
        )
    if not torch.is_floating_point(ego_to_agent):
        raise TypeError("ego_to_agent must use a floating-point dtype")
    if ego_to_agent.device != agent_features.device:
        raise ValueError(
            "ego_to_agent and agent_features must be on the same device"
        )
    if not isinstance(bev_geometry, BEVGeometry):
        raise TypeError("bev_geometry must be a BEVGeometry instance")

    if not bool(torch.isfinite(agent_features).all()):
        raise ValueError("agent_features must contain only finite values")
    if not bool(torch.isfinite(proposals).all()):
        raise ValueError("proposals must contain only finite values")
    if not bool(torch.isfinite(ego_to_agent).all()):
        raise ValueError("ego_to_agent must contain only finite values")
    if proposals.shape[0] > 0 and not bool((proposals[:, 3:6] > 0).all()):
        raise ValueError("proposal length, width, and height must be positive")


def _validate_roi_size(roi_size):
    if not isinstance(roi_size, Sequence) or isinstance(roi_size, (str, bytes)):
        raise TypeError("roi_size must be a two-element sequence (Rh,Rw)")
    if len(roi_size) != 2:
        raise ValueError("roi_size must contain exactly two values (Rh,Rw)")
    values = tuple(roi_size)
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("roi_size values Rh and Rw must be integers")
        if value <= 0:
            raise ValueError("roi_size values Rh and Rw must be positive")
    return values


def _validate_min_coverage(min_coverage):
    if isinstance(min_coverage, bool) or not isinstance(
            min_coverage, (int, float)):
        raise TypeError("min_coverage must be a real number in (0,1]")
    value = float(min_coverage)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("min_coverage must be finite and in (0,1]")
    return value

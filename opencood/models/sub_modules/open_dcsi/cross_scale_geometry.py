"""Sparse cross-scale local geometry sampling around fused innovation tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class CrossScaleGeometrySampler(nn.Module):
    """Sample a few bounded points per token without dense deformable attention."""

    def __init__(self, scale_channels, token_dim, geometry_dim, lidar_range, config):
        super().__init__()
        self.config = config
        self.lidar_range = tuple(float(value) for value in lidar_range)
        self.scale_projections = nn.ModuleList(
            [nn.Linear(channels, geometry_dim) for channels in scale_channels]
        )
        offset_input = token_dim + geometry_dim
        point_count = int(config["sampling_points"])
        if config["shared_across_scales"]:
            self.shared_offset_head = nn.Linear(offset_input, point_count * 2)
        else:
            self.offset_heads = nn.ModuleList(
                [nn.Linear(offset_input, point_count * 2) for _ in scale_channels]
            )
        self.scale_attention = nn.Linear(offset_input, len(scale_channels))

    def _ego_innovation(self, feature, record_len, affine_matrix, align_corners):
        height, width = feature.shape[-2:]
        scene_features = []
        scene_validity = []
        for scene_index, split_feature in enumerate(regroup(feature, record_len)):
            count = int(record_len[scene_index].item())
            transforms = affine_matrix[scene_index, 0, :count]
            warped = warp_affine_simple(
                split_feature,
                transforms,
                (height, width),
                align_corners=align_corners,
            )
            validity = warp_affine_simple(
                feature.new_ones((count, 1, height, width)),
                transforms,
                (height, width),
                align_corners=align_corners,
            ).clamp(0.0, 1.0)
            denominator = validity.sum(dim=0).clamp_min(1e-12)
            fused = (warped * validity).sum(dim=0) / denominator
            scene_features.append(fused)
            scene_validity.append((validity.sum(dim=0) > 0).to(feature.dtype))
        return torch.stack(scene_features), torch.stack(scene_validity)

    def _sample_scale(
        self,
        scene_feature,
        scene_validity,
        centers,
        offsets,
        scene_indices,
        align_corners,
    ):
        token_count, point_count = offsets.shape[:2]
        if token_count == 0:
            return scene_feature.new_empty((0, scene_feature.shape[1])), scene_feature.new_empty((0,))
        x_min, y_min, _, x_max, y_max, _ = self.lidar_range
        metric_x = centers[:, 0, None] + offsets[:, :, 0]
        metric_y = centers[:, 1, None] + offsets[:, :, 1]
        grid_x = 2.0 * (metric_x - x_min) / (x_max - x_min) - 1.0
        grid_y = 2.0 * (metric_y - y_min) / (y_max - y_min) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)
        selected_feature = scene_feature.index_select(0, scene_indices)
        selected_validity = scene_validity.index_select(0, scene_indices)
        sampled = F.grid_sample(
            selected_feature,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        ).squeeze(-1)
        sampled_validity = F.grid_sample(
            selected_validity,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        ).squeeze(1).squeeze(-1)
        weights = sampled_validity.clamp(0.0, 1.0)
        denominator = weights.sum(dim=-1, keepdim=True)
        pooled = (sampled * weights.unsqueeze(1)).sum(dim=-1)
        pooled = torch.where(
            denominator > 0,
            pooled / denominator.clamp_min(1e-12),
            torch.zeros_like(pooled),
        )
        return pooled, (denominator.squeeze(-1) > 0).to(pooled.dtype)

    def forward(
        self,
        tokens,
        innovation_features,
        record_len,
        affine_matrix,
        align_corners=False,
    ):
        token_count = int(tokens["scenario_index"].numel())
        geometry_dim = self.scale_projections[0].out_features
        if token_count == 0:
            return {
                "context": innovation_features[0].new_empty((0, geometry_dim)),
                "offsets": [
                    innovation_features[0].new_empty(
                        (0, int(self.config["sampling_points"]), 2)
                    )
                    for _ in innovation_features
                ],
                "scale_weights": innovation_features[0].new_empty(
                    (0, len(innovation_features))
                ),
                "validity": innovation_features[0].new_empty((0,)),
            }
        token_feature = torch.cat(
            (tokens["innovation_embedding"], tokens["geometry_embedding"]), dim=-1
        )
        scene_indices = tokens["scenario_index"].long()
        localization_control = torch.exp(
            -tokens["localization_uncertainty"].clamp_min(0.0)
        ).reshape(-1, 1, 1)
        max_x = float(self.config["offset_limit"]["x"])
        max_y = float(self.config["offset_limit"]["y"])
        contexts = []
        validities = []
        offsets_per_scale = []
        for scale_index, feature in enumerate(innovation_features):
            scene_feature, scene_validity = self._ego_innovation(
                feature, record_len, affine_matrix, align_corners
            )
            offset_head = (
                self.shared_offset_head
                if hasattr(self, "shared_offset_head")
                else self.offset_heads[scale_index]
            )
            raw_offsets = offset_head(token_feature).reshape(
                token_count, int(self.config["sampling_points"]), 2
            )
            limits = raw_offsets.new_tensor((max_x, max_y)).reshape(1, 1, 2)
            offsets = torch.tanh(raw_offsets) * limits
            if self.config["localization_quality_controls_offset"]:
                offsets = offsets * localization_control
            pooled, validity = self._sample_scale(
                scene_feature,
                scene_validity,
                tokens["centers_ego"],
                offsets,
                scene_indices,
                align_corners,
            )
            contexts.append(self.scale_projections[scale_index](pooled))
            validities.append(validity)
            offsets_per_scale.append(offsets)
        scale_logits = self.scale_attention(token_feature)
        validity_stack = torch.stack(validities, dim=-1)
        scale_logits = scale_logits.masked_fill(validity_stack == 0, -float("inf"))
        scale_weights = torch.softmax(scale_logits, dim=-1)
        scale_weights = torch.where(
            torch.isfinite(scale_weights), scale_weights, torch.zeros_like(scale_weights)
        )
        context_stack = torch.stack(contexts, dim=1)
        context = (context_stack * scale_weights.unsqueeze(-1)).sum(dim=1)
        if self.config["localization_quality_controls_offset"]:
            context = context * localization_control.squeeze(-1)
        context = torch.where(torch.isfinite(context), context, torch.zeros_like(context))
        return {
            "context": context,
            "offsets": offsets_per_scale,
            "scale_weights": scale_weights,
            "validity": (validity_stack.sum(dim=-1) > 0).to(context.dtype),
        }

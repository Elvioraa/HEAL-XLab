"""Standardized object-level innovation token extraction."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_SCHEMA = (
    "scenario_index",
    "agent_global_index",
    "agent_local_index",
    "proposal_id",
    "source_scale",
    "centers_local",
    "size_wlh",
    "yaw_sin_cos_local",
    "boxes_local_hwl",
    "objectness",
    "objectness_logits",
    "semantic_embedding",
    "innovation_embedding",
    "geometry_embedding",
    "evidence_confidence",
    "general_uncertainty",
    "localization_uncertainty",
    "validity",
    "box_quality",
)


class ScaleInnovationTokenizer(nn.Module):
    """Encode one pooled innovation scale into a standardized token."""

    def __init__(self, channels, config, quality_enabled):
        super().__init__()
        self.embedding = nn.Linear(channels, config["token_dim"])
        self.semantic = nn.Linear(channels, config["semantic_dim"])
        self.geometry = nn.Linear(channels + 2, config["geometry_dim"])
        self.box = nn.Linear(channels, 7)
        self.quality = nn.Linear(channels, 4) if quality_enabled else None

    def forward(self, pooled, normalized_xy):
        quality = self.quality(pooled) if self.quality is not None else None
        return {
            "innovation_embedding": self.embedding(pooled),
            "semantic_embedding": self.semantic(pooled),
            "geometry_embedding": self.geometry(
                torch.cat((pooled, normalized_xy), dim=-1)
            ),
            "box_raw": self.box(pooled),
            "quality_raw": quality,
        }


def _roi_pool(feature, xy_index, pool_size, align_corners):
    if xy_index.numel() == 0:
        return feature.new_empty((0, feature.shape[1]))
    _, channels, height, width = feature.shape
    token_count = xy_index.shape[0]
    offsets = torch.arange(pool_size, device=feature.device, dtype=feature.dtype)
    offsets = offsets - (pool_size - 1) / 2.0
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    sample_x = xy_index[:, 0, None, None] + offset_x
    sample_y = xy_index[:, 1, None, None] + offset_y
    if align_corners:
        norm_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
        norm_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
    else:
        norm_x = 2.0 * (sample_x + 0.5) / width - 1.0
        norm_y = 2.0 * (sample_y + 0.5) / height - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    grid = grid.reshape(1, token_count * pool_size, pool_size, 2)
    sampled = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    sampled = sampled.reshape(1, channels, token_count, pool_size, pool_size)
    return sampled.mean(dim=(-1, -2)).squeeze(0).transpose(0, 1)


def _scene_agent_indices(record_len):
    scene_indices = []
    local_indices = []
    for scene_index, count in enumerate(record_len.tolist()):
        for local_index in range(int(count)):
            scene_indices.append(scene_index)
            local_indices.append(local_index)
    return scene_indices, local_indices


class InnovationTokenizer(nn.Module):
    """Extract bounded variable-length tokens from all pyramid scales."""

    def __init__(self, modality_names, scale_channels, lidar_range, config, quality_config):
        super().__init__()
        if not config["standardized_schema"]:
            raise ValueError("Open-DCSI requires standardized innovation token schema")
        self.config = config
        self.quality_config = quality_config
        self.lidar_range = tuple(float(value) for value in lidar_range)
        self.tokenizers = nn.ModuleDict(
            {
                modality: nn.ModuleList(
                    [
                        ScaleInnovationTokenizer(
                            channels, config, quality_config["enabled"]
                        )
                        for channels in scale_channels
                    ]
                )
                for modality in modality_names
            }
        )

    def _empty(self, device, dtype):
        token_dim = int(self.config["token_dim"])
        semantic_dim = int(self.config["semantic_dim"])
        geometry_dim = int(self.config["geometry_dim"])
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        empty_float = torch.empty(0, device=device, dtype=dtype)
        return {
            "scenario_index": empty_long,
            "agent_global_index": empty_long.clone(),
            "agent_local_index": empty_long.clone(),
            "proposal_id": empty_long.clone(),
            "source_scale": empty_long.clone(),
            "centers_local": empty_float.reshape(0, 3),
            "size_wlh": empty_float.reshape(0, 3),
            "yaw_sin_cos_local": empty_float.reshape(0, 2),
            "boxes_local_hwl": empty_float.reshape(0, 7),
            "objectness": empty_float,
            "objectness_logits": empty_float.clone(),
            "semantic_embedding": empty_float.reshape(0, semantic_dim),
            "innovation_embedding": empty_float.reshape(0, token_dim),
            "geometry_embedding": empty_float.reshape(0, geometry_dim),
            "evidence_confidence": empty_float.clone(),
            "general_uncertainty": empty_float.clone(),
            "localization_uncertainty": empty_float.clone(),
            "validity": empty_float.clone(),
            "box_quality": empty_float.clone(),
            "schema": TOKEN_SCHEMA,
            "coordinate_frame": "agent_local_metric",
            "box_order": "x_y_z_h_w_l_yaw",
            "lidar_range": self.lidar_range,
        }

    def _candidate_indices(self, logits):
        score = torch.sigmoid(logits)
        local_max = score == F.max_pool2d(
            score[None, None], kernel_size=3, stride=1, padding=1
        )[0, 0]
        valid = local_max & torch.isfinite(score) & (
            score >= float(self.config["foreground_threshold"])
        )
        candidate_score = torch.where(valid, score, torch.full_like(score, -float("inf")))
        topk = min(int(self.config["proposal_topk"]), candidate_score.numel())
        values, indices = torch.topk(candidate_score.reshape(-1), k=topk)
        keep = torch.isfinite(values)
        return indices[keep], values[keep]

    def forward(
        self,
        innovation_features,
        occ_logits,
        agent_modalities,
        record_len,
        align_corners=False,
    ):
        if not innovation_features:
            raise ValueError("Open-DCSI tokenizer requires innovation feature scales")
        agent_count = int(innovation_features[0].shape[0])
        if len(agent_modalities) != agent_count:
            raise ValueError("Open-DCSI tokenizer agent modality count mismatch")
        if int(record_len.sum().item()) != agent_count:
            raise ValueError("Open-DCSI tokenizer record_len mismatch")
        scene_indices, local_indices = _scene_agent_indices(record_len.cpu())
        per_agent = []
        x_min, y_min, z_min, x_max, y_max, z_max = self.lidar_range

        for agent_index, modality in enumerate(agent_modalities):
            if modality not in self.tokenizers:
                raise KeyError("Open-DCSI tokenizer missing modality {}".format(modality))
            candidates = []
            for scale_index, (feature_scale, logit_scale) in enumerate(
                zip(innovation_features, occ_logits)
            ):
                logits = logit_scale[agent_index, 0]
                flat_indices, objectness = self._candidate_indices(logits)
                if flat_indices.numel() == 0:
                    continue
                height, width = logits.shape
                y_index = torch.div(flat_indices, width, rounding_mode="floor")
                x_index = flat_indices % width
                xy = torch.stack((x_index, y_index), dim=-1).to(feature_scale.dtype)
                pooled = _roi_pool(
                    feature_scale[agent_index : agent_index + 1],
                    xy,
                    int(self.config["roi_pool_size"]),
                    align_corners,
                )
                normalized_xy = torch.stack(
                    (
                        2.0 * (xy[:, 0] + 0.5) / width - 1.0,
                        2.0 * (xy[:, 1] + 0.5) / height - 1.0,
                    ),
                    dim=-1,
                )
                encoded = self.tokenizers[modality][scale_index](
                    pooled, normalized_xy
                )
                raw_box = encoded.pop("box_raw")
                cell_x = (x_max - x_min) / width
                cell_y = (y_max - y_min) / height
                center_x = x_min + (xy[:, 0] + 0.5) * cell_x
                center_y = y_min + (xy[:, 1] + 0.5) * cell_y
                center_x = center_x + torch.tanh(raw_box[:, 0]) * cell_x
                center_y = center_y + torch.tanh(raw_box[:, 1]) * cell_y
                center_z = (z_min + z_max) / 2.0 + torch.tanh(raw_box[:, 2]) * (
                    z_max - z_min
                ) / 2.0
                size_wlh = F.softplus(raw_box[:, 3:6]) + 1e-3
                yaw = math.pi * torch.tanh(raw_box[:, 6])
                centers = torch.stack((center_x, center_y, center_z), dim=-1)
                boxes_hwl = torch.stack(
                    (
                        center_x,
                        center_y,
                        center_z,
                        size_wlh[:, 2],
                        size_wlh[:, 0],
                        size_wlh[:, 1],
                        yaw,
                    ),
                    dim=-1,
                )
                quality_raw = encoded.pop("quality_raw")
                if quality_raw is None:
                    general_uncertainty = torch.zeros_like(objectness)
                    localization_uncertainty = torch.zeros_like(objectness)
                    validity = torch.ones_like(objectness)
                    box_quality = torch.ones_like(objectness)
                else:
                    general_uncertainty = F.softplus(quality_raw[:, 0]) + 1e-4
                    localization_uncertainty = F.softplus(quality_raw[:, 1]) + 1e-4
                    validity = torch.sigmoid(quality_raw[:, 2])
                    box_quality = torch.sigmoid(quality_raw[:, 3])
                candidates.append(
                    {
                        **encoded,
                        "proposal_id": flat_indices,
                        "source_scale": torch.full_like(flat_indices, scale_index),
                        "centers_local": centers,
                        "size_wlh": size_wlh,
                        "yaw_sin_cos_local": torch.stack(
                            (torch.sin(yaw), torch.cos(yaw)), dim=-1
                        ),
                        "boxes_local_hwl": boxes_hwl,
                        "objectness": objectness,
                        "objectness_logits": logits.reshape(-1)[flat_indices],
                        "evidence_confidence": objectness,
                        "general_uncertainty": general_uncertainty,
                        "localization_uncertainty": localization_uncertainty,
                        "validity": validity,
                        "box_quality": box_quality,
                    }
                )
            if not candidates:
                continue
            keys = candidates[0].keys()
            merged = {key: torch.cat([item[key] for item in candidates], dim=0) for key in keys}
            keep_count = min(
                int(self.config["max_tokens_per_agent"]), merged["objectness"].numel()
            )
            keep = torch.topk(merged["objectness"], k=keep_count).indices
            merged = {key: value.index_select(0, keep) for key, value in merged.items()}
            count = keep.numel()
            merged.update(
                {
                    "scenario_index": torch.full(
                        (count,),
                        scene_indices[agent_index],
                        device=keep.device,
                        dtype=torch.long,
                    ),
                    "agent_global_index": torch.full(
                        (count,), agent_index, device=keep.device, dtype=torch.long
                    ),
                    "agent_local_index": torch.full(
                        (count,),
                        local_indices[agent_index],
                        device=keep.device,
                        dtype=torch.long,
                    ),
                }
            )
            per_agent.append(merged)

        if not per_agent:
            return self._empty(
                innovation_features[0].device, innovation_features[0].dtype
            )
        result = {
            key: torch.cat([item[key] for item in per_agent], dim=0)
            for key in per_agent[0]
        }
        result.update(
            {
                "schema": TOKEN_SCHEMA,
                "coordinate_frame": "agent_local_metric",
                "box_order": "x_y_z_h_w_l_yaw",
                "lidar_range": self.lidar_range,
            }
        )
        return result


def transform_tokens_to_ego(token_dict, pairwise_t_matrix, record_len):
    """Transform local token geometry with pairwise[b, ego=0, local_agent]."""

    result = dict(token_dict)
    centers_local = token_dict["centers_local"]
    if centers_local.shape[0] == 0:
        result["centers_ego"] = centers_local.clone()
        result["yaw_sin_cos_ego"] = token_dict["yaw_sin_cos_local"].clone()
        result["boxes_ego_hwl"] = token_dict["boxes_local_hwl"].clone()
        result["coordinate_frame"] = "ego_metric"
        return result
    centers_ego = torch.empty_like(centers_local)
    yaw_ego = torch.empty_like(token_dict["boxes_local_hwl"][:, 6])
    for token_index in range(centers_local.shape[0]):
        scene = int(token_dict["scenario_index"][token_index].item())
        local_agent = int(token_dict["agent_local_index"][token_index].item())
        transform = pairwise_t_matrix[scene, 0, local_agent].to(
            device=centers_local.device, dtype=centers_local.dtype
        )
        homogeneous = torch.cat(
            (centers_local[token_index], centers_local.new_ones(1)), dim=0
        )
        centers_ego[token_index] = (transform @ homogeneous)[:3]
        rotation_yaw = torch.atan2(transform[1, 0], transform[0, 0])
        local_yaw = token_dict["boxes_local_hwl"][token_index, 6]
        yaw = local_yaw + rotation_yaw
        yaw_ego[token_index] = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    boxes_ego = token_dict["boxes_local_hwl"].clone()
    boxes_ego[:, :3] = centers_ego
    boxes_ego[:, 6] = yaw_ego
    result["centers_ego"] = centers_ego
    result["yaw_sin_cos_ego"] = torch.stack(
        (torch.sin(yaw_ego), torch.cos(yaw_ego)), dim=-1
    )
    result["boxes_ego_hwl"] = boxes_ego
    result["coordinate_frame"] = "ego_metric"
    return result

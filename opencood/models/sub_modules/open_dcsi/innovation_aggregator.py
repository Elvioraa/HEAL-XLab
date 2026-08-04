"""Permutation-invariant quality-gated innovation token aggregation."""

import math

import torch
import torch.nn as nn

from opencood.models.sub_modules.open_dcsi.quality_router import (
    absolute_rejection_gate,
)


_AGGREGATED_FLOAT_FIELDS = (
    "centers_ego",
    "size_wlh",
    "semantic_embedding",
    "innovation_embedding",
    "geometry_embedding",
    "objectness",
    "objectness_logits",
    "evidence_confidence",
    "general_uncertainty",
    "localization_uncertainty",
    "validity",
    "box_quality",
)


def _stable_member_key(tokens, index):
    center = tokens["centers_ego"][index]
    embedding_sum = tokens["innovation_embedding"][index].sum()
    return (
        float(center[0].detach().cpu()),
        float(center[1].detach().cpu()),
        float(tokens["boxes_ego_hwl"][index, 6].detach().cpu()),
        float(tokens["objectness"][index].detach().cpu()),
        float(embedding_sum.detach().cpu()),
    )


def _connected_components(tokens, indices, radius, yaw_threshold):
    parent = list(range(len(indices)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            left_index = indices[left]
            right_index = indices[right]
            center_distance = torch.linalg.vector_norm(
                tokens["centers_ego"][left_index, :2]
                - tokens["centers_ego"][right_index, :2]
            )
            yaw_left = tokens["boxes_ego_hwl"][left_index, 6]
            yaw_right = tokens["boxes_ego_hwl"][right_index, 6]
            yaw_delta = torch.atan2(
                torch.sin(yaw_left - yaw_right), torch.cos(yaw_left - yaw_right)
            ).abs()
            if float(center_distance.detach().cpu()) <= radius and float(
                yaw_delta.detach().cpu()
            ) <= yaw_threshold:
                union(left, right)
    groups = {}
    for local_index, token_index in enumerate(indices):
        groups.setdefault(find(local_index), []).append(token_index)
    return list(groups.values())


class InnovationAggregator(nn.Module):
    """Cluster by geometry and aggregate each set with normalized reliability."""

    def __init__(self, config):
        super().__init__()
        if config["pair_specific_parameters"]:
            raise ValueError("Open-DCSI forbids pair-specific aggregation parameters")
        if config["fixed_modality_embeddings"]:
            raise ValueError("Open-DCSI forbids fixed modality embeddings")
        self.config = config

    def _empty(self, tokens):
        result = dict(tokens)
        for key, value in list(result.items()):
            if torch.is_tensor(value) and value.ndim > 0:
                result[key] = value[:0]
        result["source_count"] = tokens["scenario_index"].new_empty(0)
        result["reliability"] = tokens["objectness"].new_empty(0)
        return result

    def forward(self, tokens, quality_router):
        token_count = int(tokens["scenario_index"].numel())
        if token_count == 0:
            return self._empty(tokens)
        finite = torch.ones(token_count, device=tokens["objectness"].device, dtype=torch.bool)
        for key in (
            "centers_ego",
            "boxes_ego_hwl",
            "innovation_embedding",
            "evidence_confidence",
            "general_uncertainty",
            "localization_uncertainty",
            "validity",
            "box_quality",
        ):
            value = tokens[key]
            finite &= torch.isfinite(value).reshape(token_count, -1).all(dim=1)
        reliability = quality_router(tokens)
        reliability = reliability * absolute_rejection_gate(
            reliability,
            self.config["absolute_reject"],
            self.training,
        )
        finite &= reliability > 0
        valid_indices = torch.nonzero(finite, as_tuple=False).flatten().tolist()
        if not valid_indices:
            return self._empty(tokens)

        cluster_config = self.config["geometric_clustering"]
        clusters = []
        for scene in sorted(set(int(tokens["scenario_index"][i]) for i in valid_indices)):
            scene_indices = [
                index
                for index in valid_indices
                if int(tokens["scenario_index"][index]) == scene
            ]
            if cluster_config["enabled"]:
                clusters.extend(
                    _connected_components(
                        tokens,
                        scene_indices,
                        float(cluster_config["center_radius"]),
                        float(cluster_config["yaw_threshold"]),
                    )
                )
            else:
                clusters.extend([[index] for index in scene_indices])

        fused_items = []
        for members in clusters:
            members = sorted(members, key=lambda index: _stable_member_key(tokens, index))
            member_index = torch.as_tensor(
                members, device=reliability.device, dtype=torch.long
            )
            if len(members) == 1:
                weights = reliability.new_ones(1)
            else:
                member_reliability = reliability.index_select(0, member_index)
                total = member_reliability.sum()
                if not torch.isfinite(total) or float(total.detach().cpu()) <= 0:
                    continue
                weights = member_reliability / total
            item = {}
            for field in _AGGREGATED_FLOAT_FIELDS:
                values = tokens[field].index_select(0, member_index)
                expanded_weights = weights.reshape(
                    (weights.shape[0],) + (1,) * (values.ndim - 1)
                )
                item[field] = (values * expanded_weights).sum(dim=0)
            yaw = tokens["boxes_ego_hwl"].index_select(0, member_index)[:, 6]
            yaw_sin = (torch.sin(yaw) * weights).sum()
            yaw_cos = (torch.cos(yaw) * weights).sum()
            fused_yaw = torch.atan2(yaw_sin, yaw_cos)
            center = item["centers_ego"]
            size = item["size_wlh"]
            item["boxes_ego_hwl"] = torch.stack(
                (center[0], center[1], center[2], size[2], size[0], size[1], fused_yaw)
            )
            item["yaw_sin_cos_ego"] = torch.stack(
                (torch.sin(fused_yaw), torch.cos(fused_yaw))
            )
            first = members[0]
            item["scenario_index"] = tokens["scenario_index"][first]
            item["agent_global_index"] = tokens["agent_global_index"].new_tensor(-1)
            item["agent_local_index"] = tokens["agent_local_index"].new_tensor(-1)
            item["proposal_id"] = tokens["proposal_id"][first]
            item["source_scale"] = tokens["source_scale"][first]
            item["source_count"] = tokens["scenario_index"].new_tensor(len(members))
            item["reliability"] = reliability.index_select(0, member_index).sum()
            fused_items.append(item)
        if not fused_items:
            return self._empty(tokens)
        fused_items.sort(
            key=lambda item: (
                int(item["scenario_index"].detach().cpu()),
                float(item["centers_ego"][0].detach().cpu()),
                float(item["centers_ego"][1].detach().cpu()),
            )
        )
        result = {
            key: torch.stack([item[key] for item in fused_items], dim=0)
            for key in fused_items[0]
        }
        result["schema"] = tokens["schema"]
        result["coordinate_frame"] = "ego_metric"
        result["box_order"] = tokens["box_order"]
        result["lidar_range"] = tokens["lidar_range"]
        return result

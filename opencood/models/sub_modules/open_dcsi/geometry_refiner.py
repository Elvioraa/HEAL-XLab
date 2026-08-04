"""Bounded zero-initialized box residual refinement for Open-DCSI."""

import torch
import torch.nn as nn


class GeometryRefiner(nn.Module):
    """Predict token-conditioned residuals later applied to official boxes."""

    def __init__(self, token_dim, geometry_dim, config):
        super().__init__()
        self.config = config
        input_dim = token_dim + geometry_dim * 2 + 1
        hidden_dim = max(32, input_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Linear(
            hidden_dim, 8 if config["predict_confidence_delta"] else 7
        )
        if config["zero_init_output"]:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def _bounded_delta(self, raw):
        delta = torch.zeros_like(raw[:, :7])
        center_limit = float(self.config["max_center_delta"])
        if self.config["predict_center"]:
            delta[:, :3] = torch.tanh(raw[:, :3]) * center_limit
        if self.config["predict_size"] or self.config["predict_height"]:
            size_delta = 0.5 * torch.tanh(raw[:, 3:6])
            if self.config["predict_size"]:
                delta[:, 4:6] = size_delta[:, 1:3]
            if self.config["predict_height"]:
                delta[:, 3] = size_delta[:, 0]
        if self.config["predict_yaw"]:
            delta[:, 6] = torch.tanh(raw[:, 6]) * float(
                self.config["max_yaw_delta"]
            )
        return delta

    @staticmethod
    def apply_deltas(base_boxes_hwl, delta_hwl):
        refined = base_boxes_hwl.clone()
        refined[:, :3] = base_boxes_hwl[:, :3] + delta_hwl[:, :3]
        refined[:, 3:6] = base_boxes_hwl[:, 3:6] * torch.exp(delta_hwl[:, 3:6])
        yaw = base_boxes_hwl[:, 6] + delta_hwl[:, 6]
        refined[:, 6] = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        finite = torch.isfinite(refined).all(dim=-1, keepdim=True)
        exact_identity = (delta_hwl == 0).all(dim=-1, keepdim=True)
        return torch.where(finite & ~exact_identity, refined, base_boxes_hwl)

    def forward(self, tokens, geometry_context=None):
        token_count = int(tokens["scenario_index"].numel())
        if token_count == 0:
            empty = tokens["boxes_ego_hwl"].new_empty((0, 7))
            return {
                "box_deltas_hwl": empty,
                "refined_token_boxes_preview_hwl": empty.clone(),
                "confidence_delta": tokens["objectness"].new_empty((0,)),
            }
        if geometry_context is None:
            geometry_context = torch.zeros_like(tokens["geometry_embedding"])
        refiner_input = torch.cat(
            (
                tokens["innovation_embedding"],
                tokens["geometry_embedding"],
                geometry_context,
                tokens.get("reliability", tokens["objectness"]).reshape(-1, 1),
            ),
            dim=-1,
        )
        raw = self.output(self.network(refiner_input))
        delta = self._bounded_delta(raw)
        preview = self.apply_deltas(tokens["boxes_ego_hwl"], delta)
        confidence_delta = (
            raw[:, 7] if self.config["predict_confidence_delta"] else raw.new_zeros(token_count)
        )
        return {
            "box_deltas_hwl": delta,
            "refined_token_boxes_preview_hwl": preview,
            "confidence_delta": confidence_delta,
        }

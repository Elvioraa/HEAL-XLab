"""Parameter-free uniform aggregation at the original PACT rule boundary."""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class PACTCBEAOriginalRuleUniform(nn.Module):
    """Apply deterministic per-scene 1/N weights to original PACT inputs."""

    def __init__(self):
        super(PACTCBEAOriginalRuleUniform, self).__init__()

    @staticmethod
    def parameter_count():
        return 0

    @staticmethod
    def _record_list(record_len, agent_count):
        if torch.is_tensor(record_len):
            values = record_len.detach().cpu().view(-1).tolist()
        else:
            values = list(record_len)
        values = [int(value) for value in values]
        if not values or any(value <= 0 for value in values):
            raise RuntimeError("record_len must contain positive scene sizes")
        if sum(values) != int(agent_count):
            raise RuntimeError("record_len sum does not match feature agent count")
        return values

    @staticmethod
    def uniform_alpha(feature, agent_count):
        """Return the dense, deterministic alpha used for one scene."""
        if not torch.is_tensor(feature) or feature.dim() != 4:
            raise RuntimeError("feature must have shape [N, C, H, W]")
        if feature.shape[0] != int(agent_count):
            raise RuntimeError("agent_count does not match feature shape")
        return feature.new_full(
            (int(agent_count), 1, feature.shape[2], feature.shape[3]),
            1.0 / float(agent_count),
        )

    def forward(self, feature, record_len):
        if not torch.is_tensor(feature) or feature.dim() != 4:
            raise RuntimeError("PACT rule uniform feature must have shape [N, C, H, W]")
        if not torch.isfinite(feature).all().item():
            raise RuntimeError("PACT rule uniform feature contains NaN or Inf")

        records = self._record_list(record_len, feature.shape[0])
        fused_scenes = []
        alpha_values = []
        alpha_sum_errors = []
        offset = 0
        for agent_count in records:
            scene_feature = feature[offset:offset + agent_count]
            alpha = self.uniform_alpha(scene_feature, agent_count)
            alpha_sum = alpha.sum(dim=0)
            if not torch.allclose(alpha_sum, torch.ones_like(alpha_sum),
                                  atol=1e-6, rtol=1e-6):
                raise RuntimeError("PACT rule uniform alpha does not sum to one")
            alpha_sum_errors.append(
                torch.max(torch.abs(alpha_sum - torch.ones_like(alpha_sum)))
            )
            fused_scenes.append(torch.sum(alpha * scene_feature, dim=0, keepdim=True))
            alpha_values.append(alpha.reshape(-1))
            offset += agent_count

        flattened_alpha = torch.cat(alpha_values, dim=0)
        debug = {
            "per_scene_agent_count": records,
            "uniform_weight_min": float(flattened_alpha.min().item()),
            "uniform_weight_max": float(flattened_alpha.max().item()),
            "uniform_weight_mean": float(flattened_alpha.mean().item()),
            "weight_sum_error": float(torch.stack(alpha_sum_errors).max().item()),
            "parameter_count": 0,
        }
        return torch.cat(fused_scenes, dim=0), debug

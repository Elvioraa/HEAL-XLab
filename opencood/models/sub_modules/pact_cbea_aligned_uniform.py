"""Parameter-free uniform routing over geometrically valid BEV support."""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class PACTCBEAAlignedUniformRouter(nn.Module):
    """Average aligned features only across valid agents at each pixel."""

    def __init__(self, epsilon=1e-6):
        super(PACTCBEAAlignedUniformRouter, self).__init__()
        self.epsilon = float(epsilon)
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @staticmethod
    def parameter_count():
        return 0

    @staticmethod
    def _record_list(record_len):
        if torch.is_tensor(record_len):
            values = record_len.detach().cpu().view(-1).tolist()
        else:
            values = list(record_len)
        return [int(value) for value in values]

    def _validate_inputs(self, aligned_feature, aligned_validity, record_len):
        if not torch.is_tensor(aligned_feature) or aligned_feature.dim() != 4:
            raise RuntimeError("aligned_feature must have shape [N, C, H, W]")
        if not torch.is_tensor(aligned_validity) or aligned_validity.dim() != 4:
            raise RuntimeError("aligned_validity must have shape [N, 1, H, W]")
        if aligned_validity.shape[1] != 1:
            raise RuntimeError("aligned_validity must be single-channel")
        if aligned_feature.shape[0] != aligned_validity.shape[0]:
            raise RuntimeError("feature and validity agent counts do not match")
        if tuple(aligned_feature.shape[-2:]) != tuple(aligned_validity.shape[-2:]):
            raise RuntimeError("feature and validity spatial shapes do not match")
        if not torch.isfinite(aligned_feature).all().item():
            raise RuntimeError("aligned_feature contains NaN or Inf")
        if not torch.isfinite(aligned_validity).all().item():
            raise RuntimeError("aligned_validity contains NaN or Inf")

        records = self._record_list(record_len)
        if not records or any(value <= 0 for value in records):
            raise RuntimeError("record_len must contain positive scene sizes")
        if sum(records) != aligned_feature.shape[0]:
            raise RuntimeError("record_len sum does not match agent count")
        return records

    def _normalized_alpha(self, validity):
        with torch.no_grad():
            clamped = validity.clamp(0.0, 1.0)
            denominator = clamped.sum(dim=0, keepdim=True)
            if (denominator <= self.epsilon).any().item():
                raise RuntimeError(
                    "Aligned uniform routing found a pixel with no valid agent"
                )
            alpha = clamped / denominator
            alpha_sum = alpha.sum(dim=0, keepdim=True)
            if not torch.allclose(
                    alpha_sum,
                    torch.ones_like(alpha_sum),
                    atol=1e-6,
                    rtol=1e-6):
                raise RuntimeError("Aligned uniform alpha does not sum to one")
            return alpha.detach()

    def forward(self, aligned_feature, aligned_validity, record_len):
        with torch.no_grad():
            records = self._validate_inputs(
                aligned_feature, aligned_validity, record_len
            )
            validity = aligned_validity.clamp(0.0, 1.0)
            fused_scenes = []
            alpha_values = []
            valid_ratios = []
            offset = 0
            for count in records:
                scene_feature = aligned_feature[offset:offset + count]
                scene_validity = validity[offset:offset + count]
                alpha = self._normalized_alpha(scene_validity)
                fused_scenes.append(
                    torch.sum(alpha * scene_feature, dim=0, keepdim=True)
                )
                alpha_values.append(alpha.reshape(-1))
                valid_ratios.extend(
                    scene_validity.flatten(1).mean(dim=1).cpu().tolist()
                )
                offset += count

            flattened_alpha = torch.cat(alpha_values)
            debug = {
                "routing_mode": "aligned_uniform",
                "uniform_over_valid_support": True,
                "alpha_sum_verified": True,
                "alpha_min": float(flattened_alpha.min().item()),
                "alpha_max": float(flattened_alpha.max().item()),
                "alpha_mean": float(flattened_alpha.mean().item()),
                "per_scene_agent_count": records,
                "per_agent_valid_ratio": [
                    float(value) for value in valid_ratios
                ],
                "parameter_count": 0,
            }
            return torch.cat(fused_scenes, dim=0).detach(), debug

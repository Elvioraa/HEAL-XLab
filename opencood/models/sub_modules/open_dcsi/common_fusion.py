"""Low-rank, agent-count-independent common consensus fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class LowRankCommonFusion(nn.Module):
    """Fuse compact common features as a quality-weighted residual from ego."""

    def __init__(self, config, align_corners=False):
        super().__init__()
        self.config = config
        self.align_corners = bool(align_corners)

    def _absolute_acceptance(self, reliability):
        reject = self.config["absolute_reject"]
        if not reject["enabled"]:
            return torch.ones_like(reliability)
        threshold = float(reject["threshold"])
        if not self.training and reject["hard_threshold_inference_only"]:
            return (reliability >= threshold).to(reliability.dtype)
        temperature = max(float(reject["temperature"]), 1e-6)
        return torch.sigmoid((reliability - threshold) / temperature)

    def forward(
        self,
        common_feature,
        evidence_score,
        record_len,
        affine_matrix,
        residual_gate,
        quality_prior=None,
    ):
        if common_feature.ndim != 4 or evidence_score.ndim != 4:
            raise ValueError("Open-DCSI common fusion expects NCHW tensors")
        if common_feature.shape[0] != evidence_score.shape[0]:
            raise ValueError("Open-DCSI common feature and score agent counts differ")
        height, width = common_feature.shape[-2:]
        split_feature = regroup(common_feature, record_len)
        split_score = regroup(evidence_score, record_len)
        split_prior = regroup(quality_prior, record_len) if quality_prior is not None else None
        fused_scenes = []
        weight_scenes = []

        for batch_index, scene_feature in enumerate(split_feature):
            agent_count = int(record_len[batch_index].item())
            transforms = affine_matrix[batch_index, 0, :agent_count]
            warped_feature = warp_affine_simple(
                scene_feature,
                transforms,
                (height, width),
                align_corners=self.align_corners,
            )
            warped_score = warp_affine_simple(
                split_score[batch_index],
                transforms,
                (height, width),
                align_corners=self.align_corners,
            )
            validity = warp_affine_simple(
                torch.ones_like(split_score[batch_index]),
                transforms,
                (height, width),
                align_corners=self.align_corners,
            ).clamp(0.0, 1.0)
            if self.config["use_evidence"]:
                reliability = warped_score.clamp_min(0.0) * validity
            else:
                reliability = validity
            if split_prior is not None:
                prior = split_prior[batch_index]
                if prior.shape[-2:] != (height, width):
                    prior = F.interpolate(
                        prior,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                prior = warp_affine_simple(
                    prior,
                    transforms,
                    (height, width),
                    align_corners=self.align_corners,
                )
                reliability = reliability * prior.clamp_min(0.0)
            reliability = reliability * self._absolute_acceptance(reliability)
            reliability = torch.where(
                torch.isfinite(reliability), reliability, torch.zeros_like(reliability)
            )

            ego_feature = warped_feature[0]
            if agent_count == 1:
                fused_scenes.append(ego_feature)
                weight_scenes.append(torch.ones_like(reliability))
                continue

            collaborator_reliability = reliability[1:]
            denominator = collaborator_reliability.sum(dim=0, keepdim=True)
            weights = torch.where(
                denominator > 0,
                collaborator_reliability / denominator.clamp_min(1e-12),
                torch.zeros_like(collaborator_reliability),
            )
            collaborator_delta = warped_feature[1:] - ego_feature.unsqueeze(0)
            residual = (collaborator_delta * weights).sum(dim=0)
            any_valid = (denominator.squeeze(0) > 0).to(residual.dtype)
            gate = torch.tanh(residual_gate).to(residual.dtype)
            fused = ego_feature + gate * any_valid * residual
            fused = torch.where(torch.isfinite(fused), fused, ego_feature)
            fused_scenes.append(fused)
            weight_scenes.append(
                torch.cat((torch.ones_like(reliability[:1]), weights), dim=0)
            )

        return torch.stack(fused_scenes, dim=0), weight_scenes

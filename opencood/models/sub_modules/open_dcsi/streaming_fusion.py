"""Inference-only packet-streaming common fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.open_dcsi.packet_codec import decode_tensor_packet
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


class StreamingCommonFusion(nn.Module):
    """Update low-rank consensus accumulators one decoded packet at a time."""

    def __init__(self, common_fusion_config, streaming_config):
        super().__init__()
        self.common_config = common_fusion_config
        self.streaming_config = streaming_config

    def _acceptance(self, reliability):
        reject = self.common_config["absolute_reject"]
        if not reject["enabled"]:
            return torch.ones_like(reliability)
        threshold = float(reject["threshold"])
        if reject["hard_threshold_inference_only"]:
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
        communication_state,
        scale_index,
        quality_prior=None,
        align_corners=False,
    ):
        if self.training:
            raise RuntimeError("Open-DCSI streaming fusion is inference-only")
        height, width = common_feature.shape[-2:]
        split_common = regroup(common_feature, record_len)
        split_score = regroup(evidence_score, record_len)
        split_prior = regroup(quality_prior, record_len) if quality_prior is not None else None
        packets = [
            packet
            for packet in communication_state["common_packets"]
            if packet["scale_index"] == scale_index
        ]
        packets.sort(
            key=lambda packet: (
                packet["scene_index"], packet["local_agent_index"]
            )
        )
        by_scene = {}
        for packet in packets:
            by_scene.setdefault(packet["scene_index"], []).append(packet)

        fused_scenes = []
        peak_bytes = 0
        processed = 0
        gate = torch.tanh(residual_gate)
        for scene_index, scene_common in enumerate(split_common):
            ego = scene_common[0]
            numerator = torch.zeros_like(ego)
            denominator = ego.new_zeros((1, height, width))
            peak_bytes = max(
                peak_bytes,
                numerator.numel() * numerator.element_size()
                + denominator.numel() * denominator.element_size(),
            )
            for packet_info in by_scene.get(scene_index, []):
                local_agent = packet_info["local_agent_index"]
                decoded = decode_tensor_packet(
                    packet_info["packet"], ego.device, ego.dtype
                )
                transform = affine_matrix[
                    scene_index, 0, local_agent : local_agent + 1
                ]
                warped_feature = warp_affine_simple(
                    decoded,
                    transform,
                    (height, width),
                    align_corners=align_corners,
                )[0]
                local_score = split_score[scene_index][
                    local_agent : local_agent + 1
                ]
                warped_score = warp_affine_simple(
                    local_score,
                    transform,
                    (height, width),
                    align_corners=align_corners,
                )[0]
                validity = warp_affine_simple(
                    torch.ones_like(local_score),
                    transform,
                    (height, width),
                    align_corners=align_corners,
                )[0].clamp(0.0, 1.0)
                reliability = (
                    warped_score.clamp_min(0.0) * validity
                    if self.common_config["use_evidence"]
                    else validity
                )
                if split_prior is not None:
                    local_prior = split_prior[scene_index][
                        local_agent : local_agent + 1
                    ]
                    if local_prior.shape[-2:] != (height, width):
                        local_prior = F.interpolate(
                            local_prior,
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        )
                    warped_prior = warp_affine_simple(
                        local_prior,
                        transform,
                        (height, width),
                        align_corners=align_corners,
                    )[0]
                    reliability = reliability * warped_prior.clamp_min(0.0)
                reliability = reliability * self._acceptance(reliability)
                reliability = torch.where(
                    torch.isfinite(reliability),
                    reliability,
                    torch.zeros_like(reliability),
                )
                numerator = numerator + (warped_feature - ego) * reliability
                denominator = denominator + reliability
                processed += 1
                live_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (
                        numerator,
                        denominator,
                        decoded,
                        warped_feature,
                        warped_score,
                        reliability,
                    )
                )
                peak_bytes = max(peak_bytes, live_bytes)
                del decoded, warped_feature, warped_score, reliability
            residual = torch.where(
                denominator > 0,
                numerator / denominator.clamp_min(1e-12),
                torch.zeros_like(numerator),
            )
            fused = ego + gate * residual
            fused_scenes.append(torch.where(torch.isfinite(fused), fused, ego))
        return torch.stack(fused_scenes), {
            "processed_common_packets": processed,
            "fusion_local_peak_bytes": peak_bytes,
            "recovered_full_dense_stack": False,
        }

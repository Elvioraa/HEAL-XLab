"""Parameter-free top-K local-evidence packets for PACT no-joint inference."""

import torch
import torch.nn as nn


PACKET_SOURCE = "topk_local_evidence"
MODALITY_IDS = {"m1": 1, "m2": 2, "m3": 3, "m4": 4}


class PACTNoJointPacketizer(nn.Module):
    """Sample sparse evidence packets without sending dense evidence maps."""

    def __init__(self, topk=50, confidence_threshold=0.15, quantize="fp16"):
        super().__init__()
        self.topk = int(topk)
        self.confidence_threshold = float(confidence_threshold)
        self.quantize = str(quantize).lower()
        if self.quantize not in ("fp16", "fp32", "none"):
            raise ValueError("no-joint packet quantize must be fp16/fp32/none")

    def forward(self, evidence_heatmap, evidence_uncertainty, modality_name,
                agent_index, agent_to_ego=None):
        if evidence_heatmap.shape != evidence_uncertainty.shape:
            raise ValueError("evidence heatmap and uncertainty must share shape")
        if evidence_heatmap.ndim != 4 or evidence_heatmap.shape[:2] != (1, 1):
            raise ValueError("no-joint packetizer expects [1,1,H,W] local evidence")
        if modality_name not in MODALITY_IDS:
            raise ValueError("unknown PACT modality: %s" % modality_name)
        _, _, height, width = evidence_heatmap.shape
        count = min(self.topk, height * width)
        confidence, flat_index = torch.topk(evidence_heatmap.reshape(-1), count)
        uncertainty = evidence_uncertainty.reshape(-1).gather(0, flat_index)
        x = flat_index.remainder(width).to(dtype=evidence_heatmap.dtype)
        y = torch.div(flat_index, width, rounding_mode="floor").to(dtype=evidence_heatmap.dtype)
        coordinates = torch.stack((
            2.0 * (x + 0.5) / float(width) - 1.0,
            2.0 * (y + 0.5) / float(height) - 1.0,
        ), dim=-1)
        coordinates = self._map_to_ego(coordinates, agent_to_ego)
        valid_mask = confidence >= self.confidence_threshold
        return {
            "coordinates": self._quantize(coordinates),
            "confidence": self._quantize(confidence.unsqueeze(-1)),
            "uncertainty": self._quantize(uncertainty.unsqueeze(-1)),
            "modality_id": torch.full(
                (count, 1), MODALITY_IDS[modality_name], device=confidence.device, dtype=torch.int64
            ),
            "agent_id": torch.full(
                (count, 1), int(agent_index), device=confidence.device, dtype=torch.int64
            ),
            "valid_mask": valid_mask,
            "packet_source": PACKET_SOURCE,
        }

    @staticmethod
    def _map_to_ego(coordinates, affine):
        if affine is None:
            return coordinates
        affine = torch.as_tensor(affine, device=coordinates.device, dtype=coordinates.dtype)
        if affine.shape != (2, 3):
            raise ValueError("packet coordinate transform must be [2,3]")
        homogeneous = torch.cat((coordinates, torch.ones_like(coordinates[:, :1])), dim=-1)
        return homogeneous.matmul(affine.transpose(0, 1))

    def _quantize(self, tensor):
        if self.quantize == "fp16":
            return tensor.to(torch.float16).to(dtype=tensor.dtype)
        return tensor


class PACTNoJointPacketAggregator(nn.Module):
    """Rasterize sparse packets and apply the fixed no-joint modulation rule."""

    def __init__(self, modality_prior=None, fixed_gain=0.1, collision_reduce="max"):
        super().__init__()
        self.modality_prior = dict(modality_prior or {})
        self.fixed_gain = float(fixed_gain)
        self.collision_reduce = str(collision_reduce)
        if self.collision_reduce not in ("max", "sum"):
            raise ValueError("collision_reduce must be max or sum")

    def forward(self, ego_feature, scene_packets):
        if ego_feature.ndim != 4:
            raise ValueError("ego feature must be [B,C,H,W]")
        batch_size, _, height, width = ego_feature.shape
        if len(scene_packets) != batch_size:
            raise ValueError("packet scene count must match ego batch")
        evidence_map = ego_feature.new_zeros((batch_size, 1, height, width))
        valid_total = 0
        for batch_index, packet in enumerate(scene_packets):
            valid_total += self._rasterize_scene(evidence_map[batch_index, 0], packet)
        if valid_total == 0:
            return ego_feature, evidence_map, {"empty_packet": True, "valid_packet_count": 0}
        enhanced = ego_feature * (1.0 + self.fixed_gain * evidence_map)
        return enhanced, evidence_map, {
            "empty_packet": False,
            "valid_packet_count": valid_total,
        }

    def _rasterize_scene(self, target_map, packet):
        valid = packet["valid_mask"].to(device=target_map.device, dtype=torch.bool)
        if valid.numel() == 0 or not bool(valid.any()):
            return 0
        coordinates = packet["coordinates"].to(device=target_map.device, dtype=target_map.dtype)[valid]
        confidence = packet["confidence"].to(device=target_map.device, dtype=target_map.dtype)[valid, 0]
        uncertainty = packet["uncertainty"].to(device=target_map.device, dtype=target_map.dtype)[valid, 0]
        modality_id = packet["modality_id"].to(device=target_map.device)[valid, 0]
        reliability = confidence * torch.exp(-torch.clamp(uncertainty, min=0.0, max=20.0))
        priors = torch.tensor(
            [self.modality_prior.get("m%d" % int(item), 1.0) for item in modality_id.tolist()],
            device=target_map.device,
            dtype=target_map.dtype,
        )
        reliability = reliability * priors
        height, width = target_map.shape
        x = torch.round((coordinates[:, 0] + 1.0) * 0.5 * (width - 1)).long()
        y = torch.round((coordinates[:, 1] + 1.0) * 0.5 * (height - 1)).long()
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not bool(inside.any()):
            return 0
        index = y[inside] * width + x[inside]
        values = reliability[inside]
        flat = target_map.reshape(-1)
        if self.collision_reduce == "max" and hasattr(flat, "scatter_reduce_"):
            flat.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
        elif self.collision_reduce == "max":
            for item_index, value in zip(index.tolist(), values):
                flat[item_index] = torch.maximum(flat[item_index], value)
        else:
            flat.scatter_add_(0, index, values)
        return int(inside.sum().detach().cpu())


class PACTNoJointCommunicationMeter(nn.Module):
    """Parameter-free packet communication accounting."""

    def __init__(self, quantize="fp16", deadline_ms=100, bandwidth_budget_kb=8):
        super().__init__()
        self.quantize = str(quantize).lower()
        self.deadline_ms = float(deadline_ms)
        self.bandwidth_budget_kb = float(bandwidth_budget_kb)

    def forward(self, scene_packets):
        packet_count = sum(int(packet["valid_mask"].sum().detach().cpu()) for packet in scene_packets)
        # Two fp coordinates, fp confidence, fp uncertainty, uint8 modality and agent id.
        float_bytes = 2 if self.quantize == "fp16" else 4
        bytes_per_packet = 4 * float_bytes + 2
        byte_count = packet_count * bytes_per_packet
        seconds = max(self.deadline_ms, 1e-6) / 1000.0
        return {
            "packet_count": packet_count,
            "packet_bytes_per_frame": byte_count,
            "packet_kb_per_frame": byte_count / 1024.0,
            "estimated_mbps": byte_count * 8.0 / 1_000_000.0 / seconds,
            "bytes_per_packet": bytes_per_packet,
            "bandwidth_budget_kb": self.bandwidth_budget_kb,
            "bandwidth_saturated": bool(byte_count >= self.bandwidth_budget_kb * 1024.0 and packet_count > 0),
        }

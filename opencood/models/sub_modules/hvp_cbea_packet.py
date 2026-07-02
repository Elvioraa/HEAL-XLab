"""Packet communication modules for HVP-CBEA.

These modules simulate a deployment path where collaborators send compact
hypothesis/evidence packets instead of dense BEV features.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypothesisEvidencePacketizer(nn.Module):
    """Convert a BEV feature map into top-K hypothesis/evidence packets."""

    def __init__(
        self,
        in_channels=256,
        topk=50,
        packet_dim=16,
        descriptor_dim=8,
        send_uncertainty=True,
        send_agent_quality=True,
        send_timestamp=True,
    ):
        super().__init__()
        self.topk = int(topk)
        self.packet_dim = int(packet_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.send_uncertainty = bool(send_uncertainty)
        self.send_agent_quality = bool(send_agent_quality)
        self.send_timestamp = bool(send_timestamp)
        self.score_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.descriptor_head = nn.Conv2d(in_channels, self.descriptor_dim, kernel_size=1)

    def forward(self, bev_feature, agent_quality=None, timestamp=None):
        if bev_feature is None or bev_feature.ndim != 4:
            raise ValueError("bev_feature must be a [B, C, H, W] tensor")

        bsz, _, height, width = bev_feature.shape
        device = bev_feature.device
        dtype = bev_feature.dtype
        topk = max(self.topk, 0)
        if topk == 0:
            return self.empty_packet(bsz, device=device, dtype=dtype, timestamp=timestamp)

        flat_size = height * width
        actual_k = min(topk, flat_size)
        score_map = self.score_head(bev_feature).flatten(2)
        uncertainty_map = F.softplus(self.uncertainty_head(bev_feature)).flatten(2)
        descriptor_map = self.descriptor_head(bev_feature).flatten(2).transpose(1, 2)

        topk_scores, topk_indices = torch.topk(score_map.squeeze(1), k=actual_k, dim=1)
        gather_index = topk_indices.unsqueeze(-1)
        scores = torch.sigmoid(topk_scores).unsqueeze(-1)
        uncertainty = torch.gather(uncertainty_map.squeeze(1), 1, topk_indices).unsqueeze(-1)
        descriptor = torch.gather(
            descriptor_map,
            1,
            gather_index.expand(-1, -1, self.descriptor_dim),
        )
        boxes = self._pseudo_boxes(topk_indices, height, width, dtype)
        valid_mask = torch.ones((bsz, actual_k), device=device, dtype=torch.bool)

        if actual_k < topk:
            pad_k = topk - actual_k
            boxes = self._pad_k(boxes, pad_k)
            scores = self._pad_k(scores, pad_k)
            uncertainty = self._pad_k(uncertainty, pad_k)
            descriptor = self._pad_k(descriptor, pad_k)
            valid_mask = self._pad_k(valid_mask, pad_k, value=False)

        if not self.send_uncertainty:
            uncertainty = torch.zeros_like(uncertainty)
        agent_quality = self._expand_agent_quality(agent_quality, bsz, topk, device, dtype)
        if not self.send_agent_quality:
            agent_quality = torch.ones_like(agent_quality)

        # TODO: replace normalized pseudo boxes with decoded detector boxes when
        # a deployment-ready box decoder is available at the packet boundary.
        packet = {
            "boxes": boxes,
            "centers": boxes[..., :2],
            "scores": scores,
            "uncertainty": uncertainty,
            "descriptor": descriptor,
            "agent_quality": agent_quality,
            "valid_mask": valid_mask,
            "estimated_bytes": torch.tensor(0, device=device, dtype=torch.long),
            "metadata": self._metadata(timestamp),
        }
        return packet

    def empty_packet(self, batch_size=1, device=None, dtype=None, timestamp=None):
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        topk = max(self.topk, 0)
        return {
            "boxes": torch.zeros((batch_size, topk, 7), device=device, dtype=dtype),
            "centers": torch.zeros((batch_size, topk, 2), device=device, dtype=dtype),
            "scores": torch.zeros((batch_size, topk, 1), device=device, dtype=dtype),
            "uncertainty": torch.zeros((batch_size, topk, 1), device=device, dtype=dtype),
            "descriptor": torch.zeros((batch_size, topk, self.descriptor_dim), device=device, dtype=dtype),
            "agent_quality": torch.ones((batch_size, topk, 1), device=device, dtype=dtype),
            "valid_mask": torch.zeros((batch_size, topk), device=device, dtype=torch.bool),
            "estimated_bytes": torch.tensor(0, device=device, dtype=torch.long),
            "metadata": self._metadata(timestamp),
        }

    def _pseudo_boxes(self, topk_indices, height, width, dtype):
        y_idx = torch.div(topk_indices, width, rounding_mode="floor")
        x_idx = topk_indices.remainder(width)
        x = (x_idx.to(dtype=dtype) + 0.5) / max(float(width), 1.0)
        y = (y_idx.to(dtype=dtype) + 0.5) / max(float(height), 1.0)
        z = torch.zeros_like(x)
        length = torch.full_like(x, 1.0 / max(float(width), 1.0))
        box_width = torch.full_like(y, 1.0 / max(float(height), 1.0))
        box_height = torch.full_like(x, 0.1)
        yaw = torch.zeros_like(x)
        return torch.stack([x, y, z, length, box_width, box_height, yaw], dim=-1)

    @staticmethod
    def _pad_k(tensor, pad_k, value=0.0):
        if pad_k <= 0:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[1] = pad_k
        pad = tensor.new_full(pad_shape, value)
        return torch.cat([tensor, pad], dim=1)

    @staticmethod
    def _expand_agent_quality(agent_quality, batch_size, topk, device, dtype):
        if agent_quality is None:
            return torch.ones((batch_size, topk, 1), device=device, dtype=dtype)
        if not torch.is_tensor(agent_quality):
            agent_quality = torch.tensor(agent_quality, device=device, dtype=dtype)
        agent_quality = agent_quality.to(device=device, dtype=dtype)
        if agent_quality.ndim == 0:
            agent_quality = agent_quality.view(1, 1, 1).expand(batch_size, topk, 1)
        elif agent_quality.ndim == 1:
            agent_quality = agent_quality.view(-1, 1, 1).expand(-1, topk, 1)
        elif agent_quality.ndim == 2:
            agent_quality = agent_quality.unsqueeze(-1)
        return agent_quality.expand(batch_size, topk, 1)

    def _metadata(self, timestamp):
        metadata = {
            "topk": self.topk,
            "packet_dim": self.packet_dim,
            "descriptor_dim": self.descriptor_dim,
            "send_uncertainty": self.send_uncertainty,
            "send_agent_quality": self.send_agent_quality,
            "send_timestamp": self.send_timestamp,
        }
        if self.send_timestamp:
            metadata["timestamp"] = timestamp
        return metadata


class PacketCommunicationMeter(nn.Module):
    """Estimate packet communication cost."""

    FLOAT_FIELDS = ("boxes", "scores", "uncertainty", "descriptor", "agent_quality")

    def __init__(
        self,
        quantize="fp16",
        topk=50,
        descriptor_dim=8,
        packet_dim=16,
        bandwidth_budget_kb=8,
        deadline_ms=100,
    ):
        super().__init__()
        self.quantize = str(quantize or "fp16").lower()
        self.topk = int(topk)
        self.descriptor_dim = int(descriptor_dim)
        self.packet_dim = int(packet_dim)
        self.bandwidth_budget_kb = float(bandwidth_budget_kb)
        self.deadline_ms = float(deadline_ms)

    def forward(self, packet):
        valid_mask = packet.get("valid_mask")
        if valid_mask is None:
            valid_count = 0
            batch_size = 1
        else:
            valid_count = int(valid_mask.detach().sum().cpu())
            batch_size = max(int(valid_mask.shape[0]), 1)
        bytes_per_packet = self._bytes_per_packet(packet)
        total_bytes = int(valid_count * bytes_per_packet)
        bytes_per_frame = float(total_bytes) / float(batch_size)
        kb_per_frame = bytes_per_frame / 1024.0
        budget_saturated = self.bandwidth_budget_kb > 0 and kb_per_frame > self.bandwidth_budget_kb
        mbps_estimate = 0.0
        if self.deadline_ms > 0:
            mbps_estimate = (bytes_per_frame * 8.0) / (self.deadline_ms / 1000.0) / 1.0e6
        return {
            "packet_num": valid_count,
            "num_packets": valid_count,
            "bytes_per_frame": bytes_per_frame,
            "kb_per_frame": kb_per_frame,
            "mbps_estimate": mbps_estimate,
            "quantize": self.quantize,
            "quantize_mode": self.quantize,
            "topk": self.topk,
            "descriptor_dim": self.descriptor_dim,
            "packet_dim": self.packet_dim,
            "bandwidth_budget_kb": self.bandwidth_budget_kb,
            "budget_saturated": bool(budget_saturated),
        }

    def _bytes_per_packet(self, packet):
        values_per_packet = 0
        for key in self.FLOAT_FIELDS:
            tensor = packet.get(key)
            if torch.is_tensor(tensor) and tensor.ndim >= 3:
                values_per_packet += int(tensor.shape[-1])
        values_per_packet += 1
        return values_per_packet * self._bytes_per_value()

    def _bytes_per_value(self):
        if self.quantize in ("fp16", "float16", "half", "int8"):
            return 2 if self.quantize != "int8" else 1
        return 4


class PacketCompressor(nn.Module):
    """Simulate packet quantization and bandwidth-budget filtering."""

    FLOAT_FIELDS = PacketCommunicationMeter.FLOAT_FIELDS

    def __init__(
        self,
        quantize="fp16",
        bandwidth_budget_kb=8,
        topk=50,
        descriptor_dim=8,
        packet_dim=16,
        deadline_ms=100,
        detach_packet=False,
    ):
        super().__init__()
        self.quantize = str(quantize or "fp16").lower()
        self.bandwidth_budget_kb = float(bandwidth_budget_kb)
        self.detach_packet = bool(detach_packet)
        self.meter = PacketCommunicationMeter(
            quantize=self.quantize,
            topk=topk,
            descriptor_dim=descriptor_dim,
            packet_dim=packet_dim,
            bandwidth_budget_kb=bandwidth_budget_kb,
            deadline_ms=deadline_ms,
        )

    def forward(self, packet):
        compressed = self._clone_packet(packet)
        if self.detach_packet:
            for key in self.FLOAT_FIELDS:
                if torch.is_tensor(compressed.get(key)):
                    compressed[key] = compressed[key].detach()
        compressed = self._apply_budget(compressed)
        for key in self.FLOAT_FIELDS:
            tensor = compressed.get(key)
            if torch.is_tensor(tensor):
                compressed[key] = self._quantize_tensor(tensor)
        stats = self.meter(compressed)
        estimated = torch.tensor(
            int(stats["bytes_per_frame"]),
            device=compressed["valid_mask"].device,
            dtype=torch.long,
        )
        compressed["estimated_bytes"] = estimated
        compressed["metadata"] = copy.deepcopy(compressed.get("metadata", {}))
        compressed["metadata"]["communication"] = stats
        return compressed, stats

    def _apply_budget(self, packet):
        valid_mask = packet.get("valid_mask")
        if valid_mask is None or self.bandwidth_budget_kb <= 0:
            return packet
        bytes_per_packet = max(self.meter._bytes_per_packet(packet), 1)
        batch_budget_bytes = int(self.bandwidth_budget_kb * 1024.0)
        max_packets = max(batch_budget_bytes // bytes_per_packet, 1)
        if valid_mask.shape[1] <= max_packets:
            return packet
        keep = torch.arange(valid_mask.shape[1], device=valid_mask.device).view(1, -1)
        packet["valid_mask"] = valid_mask & (keep < max_packets)
        return packet

    def _quantize_tensor(self, tensor):
        if self.quantize in ("none", "fp32", "float32"):
            return tensor.float() if self.quantize in ("fp32", "float32") else tensor
        if self.quantize in ("fp16", "float16", "half"):
            return tensor.half()
        if self.quantize == "int8":
            scale = tensor.detach().abs().amax(dim=tuple(range(1, tensor.ndim)), keepdim=True).clamp_min(1e-6)
            quantized = torch.clamp(torch.round(tensor / scale * 127.0), -128, 127)
            return quantized / 127.0 * scale
        return tensor

    @staticmethod
    def _clone_packet(packet):
        cloned = {}
        for key, value in packet.items():
            if torch.is_tensor(value):
                cloned[key] = value.clone()
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned


class PacketAggregator(nn.Module):
    """Aggregate compact packets into a BEV residual delta feature."""

    def __init__(self, context_channels=256, packet_dim=16, descriptor_dim=8):
        super().__init__()
        self.context_channels = int(context_channels)
        self.packet_dim = int(packet_dim)
        self.descriptor_dim = int(descriptor_dim)
        packet_input_dim = 7 + 1 + 1 + self.descriptor_dim + 1
        self.packet_embed = nn.Sequential(
            nn.Linear(packet_input_dim, self.packet_dim),
            nn.ReLU(inplace=False),
            nn.Linear(self.packet_dim, self.packet_dim),
            nn.ReLU(inplace=False),
        )
        self.token_to_channel = nn.Linear(self.packet_dim, self.context_channels, bias=False)
        self.context_proj = nn.Conv2d(self.context_channels, self.context_channels, kernel_size=1, bias=False)

    def forward(self, ego_context_feature, packet, ego_hypotheses=None):
        if ego_context_feature is None or ego_context_feature.ndim != 4:
            raise ValueError("ego_context_feature must be a [B, C, H, W] tensor")
        bsz, channels, _, _ = ego_context_feature.shape
        device = ego_context_feature.device
        dtype = ego_context_feature.dtype
        packet_features, weight, valid_mask = self._packet_features(packet, bsz, device, dtype)
        if packet_features.shape[1] == 0:
            delta = torch.zeros_like(ego_context_feature)
            debug = self._debug(weight, valid_mask, empty=True)
            return delta, debug, {}

        embedded = self.packet_embed(packet_features)
        weighted = embedded * weight
        denom = weight.sum(dim=1).clamp_min(1e-6)
        token = weighted.sum(dim=1) / denom
        channel_gate = torch.tanh(self.token_to_channel(token)).view(bsz, channels, 1, 1)
        delta = self.context_proj(ego_context_feature) * channel_gate
        has_valid = valid_mask.any(dim=1).to(dtype=dtype).view(bsz, 1, 1, 1)
        delta = delta * has_valid
        debug = self._debug(weight, valid_mask, empty=False)
        debug["ego_hypothesis_count"] = int(ego_hypotheses.shape[1]) if torch.is_tensor(ego_hypotheses) else 0
        return delta, debug, {}

    def _packet_features(self, packet, batch_size, device, dtype):
        boxes = self._field(packet, "boxes", batch_size, 7, device, dtype)
        scores = self._field(packet, "scores", batch_size, 1, device, dtype)
        uncertainty = self._field(packet, "uncertainty", batch_size, 1, device, dtype)
        descriptor = self._field(packet, "descriptor", batch_size, self.descriptor_dim, device, dtype)
        agent_quality = self._field(packet, "agent_quality", batch_size, 1, device, dtype, fill=1.0)
        valid_mask = packet.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(boxes.shape[:2], device=device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        features = torch.cat([boxes, scores, uncertainty, descriptor, agent_quality], dim=-1)
        weight = torch.sigmoid(scores) * torch.sigmoid(agent_quality) * torch.exp(-uncertainty.clamp(0.0, 20.0))
        weight = weight * valid_mask.unsqueeze(-1).to(dtype=dtype)
        return features, weight, valid_mask

    @staticmethod
    def _field(packet, key, batch_size, width, device, dtype, fill=0.0):
        tensor = packet.get(key)
        if not torch.is_tensor(tensor):
            return torch.full((batch_size, 0, width), fill, device=device, dtype=dtype)
        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.shape[0] != batch_size:
            if tensor.shape[0] == 1:
                tensor = tensor.expand(batch_size, -1, -1)
            else:
                raise ValueError("packet batch size does not match ego context batch size")
        return tensor

    @staticmethod
    def _debug(weight, valid_mask, empty=False):
        valid_count = int(valid_mask.detach().sum().cpu()) if torch.is_tensor(valid_mask) else 0
        mean_weight = 0.0
        if torch.is_tensor(weight) and valid_count > 0:
            mean_weight = float(weight.detach().sum().cpu() / max(valid_count, 1))
        return {
            "packet_empty": bool(empty or valid_count == 0),
            "packet_valid_count": valid_count,
            "packet_weight_mean": mean_weight,
        }

"""Packet-only building blocks for the independent PACT-CBEA experiment.

The initial packet format is deliberately feature-derived: it is a compact
top-K pseudo-hypothesis interface, not detector-decoded 3D boxes.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PACKET_SOURCE = "feature_derived_pseudo_hypothesis"


class PACTPacketizer(nn.Module):
    """Convert one or more collaborator BEV maps into standardized packets."""

    def __init__(self, in_channels, packet_dim=16, descriptor_dim=8, topk=50,
                 send_uncertainty=True, send_agent_quality=True,
                 send_timestamp=True):
        super().__init__()
        self.packet_dim = int(packet_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.topk = int(topk)
        self.send_uncertainty = bool(send_uncertainty)
        self.send_agent_quality = bool(send_agent_quality)
        self.send_timestamp = bool(send_timestamp)
        self.score_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.descriptor_head = nn.Conv2d(in_channels, self.descriptor_dim, kernel_size=1)
        self.packet_head = nn.Conv2d(in_channels, self.packet_dim, kernel_size=1)

    def forward(self, bev_feature, agent_quality=None, timestamp=None):
        if not torch.is_tensor(bev_feature) or bev_feature.ndim != 4:
            raise ValueError("PACTPacketizer expects collaborator BEV [N,C,H,W]")
        agents, _, height, width = bev_feature.shape
        if agents == 0:
            return self.empty_packet(bev_feature.device, bev_feature.dtype, 0)

        score_logits = self.score_head(bev_feature)
        uncertainty = F.softplus(self.uncertainty_head(bev_feature)) + 1e-6
        descriptor = self.descriptor_head(bev_feature)
        packet_feature = self.packet_head(bev_feature)
        spatial = height * width
        topk = min(self.topk, spatial)
        score_values, indices = torch.topk(score_logits.flatten(1), k=topk, dim=1)

        def gather(feature_map):
            channels = feature_map.shape[1]
            flat = feature_map.reshape(agents, channels, spatial)
            return torch.gather(
                flat,
                2,
                indices.unsqueeze(1).expand(-1, channels, -1),
            ).transpose(1, 2)

        x = (indices % width).to(dtype=bev_feature.dtype)
        y = torch.div(indices, width, rounding_mode="floor").to(dtype=bev_feature.dtype)
        x = (x + 0.5) / float(max(width, 1))
        y = (y + 0.5) / float(max(height, 1))
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        boxes = torch.stack((x, y, zeros, ones, ones, ones, zeros), dim=-1)
        packet = {
            "boxes": boxes,
            "scores": score_values.unsqueeze(-1),
            "uncertainty": gather(uncertainty),
            "descriptor": gather(descriptor),
            "packet_feature": gather(packet_feature),
            "agent_quality": self._agent_scalar(
                agent_quality,
                agents,
                topk,
                bev_feature,
                default=1.0,
            ),
            "timestamp": self._agent_scalar(
                timestamp,
                agents,
                topk,
                bev_feature,
                default=0.0,
            ),
            "valid_mask": torch.ones(agents, topk, dtype=torch.bool, device=bev_feature.device),
            "packet_source": PACKET_SOURCE,
        }
        return self._pad_packet(packet, self.topk)

    def empty_packet(self, device, dtype, agents=1):
        shape = (int(agents), self.topk)
        return {
            "boxes": torch.zeros(*shape, 7, device=device, dtype=dtype),
            "scores": torch.zeros(*shape, 1, device=device, dtype=dtype),
            "uncertainty": torch.zeros(*shape, 1, device=device, dtype=dtype),
            "descriptor": torch.zeros(*shape, self.descriptor_dim, device=device, dtype=dtype),
            "packet_feature": torch.zeros(*shape, self.packet_dim, device=device, dtype=dtype),
            "agent_quality": torch.ones(*shape, 1, device=device, dtype=dtype),
            "timestamp": torch.zeros(*shape, 1, device=device, dtype=dtype),
            "valid_mask": torch.zeros(*shape, dtype=torch.bool, device=device),
            "packet_source": PACKET_SOURCE,
        }

    @staticmethod
    def _agent_scalar(value, agents, topk, feature, default):
        if value is None:
            return feature.new_full((agents, topk, 1), default)
        value = torch.as_tensor(value, device=feature.device, dtype=feature.dtype)
        if value.numel() == 1:
            return value.reshape(1, 1, 1).expand(agents, topk, 1)
        if value.numel() == agents:
            return value.reshape(agents, 1, 1).expand(-1, topk, -1)
        raise ValueError("packet agent scalar must be scalar or match collaborator count")

    @staticmethod
    def _pad_packet(packet, topk):
        current = int(packet["valid_mask"].shape[1])
        if current == int(topk):
            return packet
        if current > int(topk):
            return {
                key: value[:, :topk] if torch.is_tensor(value) and value.ndim >= 2 else value
                for key, value in packet.items()
            }
        padded = {}
        for key, value in packet.items():
            if not torch.is_tensor(value) or value.ndim < 2:
                padded[key] = value
                continue
            pad_shape = list(value.shape)
            pad_shape[1] = int(topk) - current
            fill = False if value.dtype == torch.bool else 0.0
            padding = torch.full(pad_shape, fill, device=value.device, dtype=value.dtype)
            padded[key] = torch.cat((value, padding), dim=1)
        return padded


class PACTPacketCompressor(nn.Module):
    """Apply a simple differentiable transport quantization and budget cap."""

    FLOAT_FIELDS = ("boxes", "scores", "uncertainty", "descriptor", "packet_feature",
                    "agent_quality", "timestamp")

    def __init__(self, quantize="fp16", bandwidth_budget_kb=8, detach_packet=False):
        super().__init__()
        self.quantize = str(quantize).lower()
        self.bandwidth_budget_kb = float(bandwidth_budget_kb)
        self.detach_packet = bool(detach_packet)
        if self.quantize not in ("none", "fp32", "fp16", "int8"):
            raise ValueError("PACT packet quantize must be none/fp32/fp16/int8")

    def forward(self, packet):
        self._validate_packet(packet)
        packet = dict(packet)
        valid_mask = packet["valid_mask"].clone()
        bytes_per_packet = self._bytes_per_packet(packet)
        budget_bytes = max(int(self.bandwidth_budget_kb * 1024), 0)
        allowed = valid_mask.numel() if bytes_per_packet == 0 else budget_bytes // bytes_per_packet
        valid_indices = torch.nonzero(valid_mask.flatten(), as_tuple=False).flatten()
        if valid_indices.numel() > allowed:
            flat_mask = valid_mask.flatten()
            flat_mask[valid_indices[allowed:]] = False
            valid_mask = flat_mask.view_as(valid_mask)
        packet["valid_mask"] = valid_mask

        for field in self.FLOAT_FIELDS:
            value = packet[field]
            quantized = self._quantize(value)
            if self.detach_packet:
                quantized = quantized.detach()
            packet[field] = quantized
        packet["comm_stats"] = self._stats(valid_mask, bytes_per_packet, budget_bytes)
        return packet

    def _quantize(self, tensor):
        if self.quantize in ("none", "fp32"):
            return tensor
        if self.quantize == "fp16":
            return tensor.to(dtype=torch.float16).to(dtype=tensor.dtype)
        scale = tensor.detach().abs().amax(dim=tuple(range(1, tensor.ndim)), keepdim=True)
        scale = torch.clamp(scale / 127.0, min=1e-6)
        dequantized = torch.round(tensor / scale).clamp(-127, 127) * scale
        return tensor + (dequantized - tensor).detach()

    def _bytes_per_packet(self, packet):
        scalar_count = sum(int(packet[field].shape[-1]) for field in self.FLOAT_FIELDS)
        bytes_per_scalar = {"none": 4, "fp32": 4, "fp16": 2, "int8": 1}[self.quantize]
        return scalar_count * bytes_per_scalar

    def _stats(self, valid_mask, bytes_per_packet, budget_bytes):
        packet_num = int(valid_mask.sum().detach().cpu())
        byte_count = int(packet_num * bytes_per_packet)
        return {
            "packet_num": packet_num,
            "num_packets": packet_num,
            "bytes_per_frame": byte_count,
            "kb_per_frame": byte_count / 1024.0,
            "quantize_mode": self.quantize,
            "bandwidth_budget_kb": self.bandwidth_budget_kb,
            "budget_saturated": bool(byte_count >= budget_bytes and packet_num > 0),
            "bytes_per_packet": bytes_per_packet,
        }

    @classmethod
    def _validate_packet(cls, packet):
        if not isinstance(packet, dict) or "valid_mask" not in packet:
            raise ValueError("PACT packet is missing valid_mask")
        for field in cls.FLOAT_FIELDS:
            if field not in packet or not torch.is_tensor(packet[field]):
                raise ValueError("PACT packet is missing %s" % field)


class PACTPacketAggregator(nn.Module):
    """Aggregate packets only; collaborator dense BEV is never an input."""

    def __init__(self, context_channels, packet_dim=16, descriptor_dim=8, hidden_dim=64):
        super().__init__()
        self.context_channels = int(context_channels)
        input_dim = int(packet_dim) + int(descriptor_dim) + 4
        self.packet_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, self.context_channels),
        )
        self.context_gate = nn.Conv2d(self.context_channels, self.context_channels, kernel_size=1)

    def forward(self, ego_context_feature, packet):
        if ego_context_feature.ndim != 4:
            raise ValueError("PACTPacketAggregator expects ego context [B,C,H,W]")
        if not isinstance(packet, dict):
            raise ValueError("PACTPacketAggregator accepts only packet dictionaries")
        batch_size, channels, height, width = ego_context_feature.shape
        if channels != self.context_channels:
            raise ValueError("ego context channel count does not match packet aggregator")
        packet = _ensure_batched_packet(packet, batch_size)
        valid = packet["valid_mask"].to(device=ego_context_feature.device)
        if valid.dtype != torch.bool:
            valid = valid > 0
        if valid.shape[0] != batch_size:
            raise ValueError("packet batch size does not match ego context")
        if valid.numel() == 0 or not bool(valid.any()):
            delta = torch.zeros_like(ego_context_feature)
            return delta, {"empty_packet": True, "valid_packet_count": 0, "mean_weight": 0.0}

        descriptor = packet["descriptor"].to(dtype=ego_context_feature.dtype)
        packet_feature = packet["packet_feature"].to(dtype=ego_context_feature.dtype)
        scores = packet["scores"].to(dtype=ego_context_feature.dtype)
        uncertainty = packet["uncertainty"].to(dtype=ego_context_feature.dtype)
        quality = packet["agent_quality"].to(dtype=ego_context_feature.dtype)
        timestamp = packet["timestamp"].to(dtype=ego_context_feature.dtype)
        embedding_input = torch.cat((packet_feature, descriptor, scores, uncertainty, quality, timestamp), dim=-1)
        embedding = self.packet_embedding(embedding_input)
        weight = torch.sigmoid(scores) * torch.sigmoid(quality) * torch.exp(-torch.clamp(uncertainty, 0.0, 20.0))
        weight = weight * valid.unsqueeze(-1).to(dtype=weight.dtype)
        denom = weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        token = (embedding * weight).sum(dim=1, keepdim=True) / denom
        spatial_gate = torch.sigmoid(self.context_gate(ego_context_feature))
        delta = token.transpose(1, 2).reshape(batch_size, channels, 1, 1) * spatial_gate
        return delta, {
            "empty_packet": False,
            "valid_packet_count": int(valid.sum().detach().cpu()),
            "mean_weight": float(weight.detach().mean().cpu()),
        }


class PACTPacketResidualFusion(nn.Module):
    """Bounded learned residual gate for packet-derived updates."""

    def __init__(self, alpha_init=0.05, alpha_max=0.3):
        super().__init__()
        self.alpha_max = float(alpha_max)
        initial = min(max(float(alpha_init), 1e-6), self.alpha_max - 1e-6)
        ratio = initial / max(self.alpha_max - initial, 1e-6)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(ratio), dtype=torch.float32))

    def forward(self, ego_feature, delta_feature):
        if ego_feature.shape != delta_feature.shape:
            raise ValueError("packet residual must match ego feature shape")
        alpha = self.alpha_max * torch.sigmoid(self.alpha_logit)
        return ego_feature + alpha * delta_feature, alpha


class PACTPacketCommunicationMeter(nn.Module):
    """Report packet communication totals without introducing parameters."""

    def __init__(self, deadline_ms=100):
        super().__init__()
        self.deadline_ms = float(deadline_ms)

    def forward(self, scene_stats):
        scene_stats = list(scene_stats or [])
        byte_count = sum(int(item.get("bytes_per_frame", 0)) for item in scene_stats)
        packet_count = sum(int(item.get("packet_num", 0)) for item in scene_stats)
        budget = max((float(item.get("bandwidth_budget_kb", 0.0)) for item in scene_stats), default=0.0)
        quantize = scene_stats[0].get("quantize_mode", "none") if scene_stats else "none"
        seconds = max(self.deadline_ms, 1e-6) / 1000.0
        return {
            "packet_count": packet_count,
            "packet_bytes_per_frame": byte_count,
            "packet_kb_per_frame": byte_count / 1024.0,
            "estimated_mbps": byte_count * 8.0 / 1_000_000.0 / seconds,
            "bandwidth_budget_kb": budget,
            "bandwidth_saturated": any(bool(item.get("budget_saturated", False)) for item in scene_stats),
            "quantize": quantize,
        }


def flatten_agent_packets(packet):
    """Flatten [agents, K, ...] packet fields into one scene packet."""
    flattened = {}
    for key, value in packet.items():
        if key == "comm_stats" or not torch.is_tensor(value):
            flattened[key] = value
        elif value.ndim == 2:
            flattened[key] = value.reshape(value.shape[0] * value.shape[1])
        elif value.ndim > 2:
            flattened[key] = value.reshape(-1, *value.shape[2:])
        else:
            flattened[key] = value
    return flattened


def collate_scene_packets(scene_packets):
    """Pad variable packet counts into a [B,K,...] packet dictionary."""
    if not scene_packets:
        raise ValueError("scene packet list cannot be empty")
    tensor_keys = [key for key, value in scene_packets[0].items() if torch.is_tensor(value)]
    max_packets = max(int(packet["valid_mask"].shape[0]) for packet in scene_packets)
    collated = {"packet_source": PACKET_SOURCE}
    for key in tensor_keys:
        values = []
        for packet in scene_packets:
            value = packet[key]
            padding = max_packets - int(value.shape[0])
            if padding:
                shape = (padding, *value.shape[1:])
                fill = False if value.dtype == torch.bool else 0.0
                value = torch.cat((value, torch.full(shape, fill, device=value.device, dtype=value.dtype)), dim=0)
            values.append(value)
        collated[key] = torch.stack(values, dim=0)
    return collated


def _ensure_batched_packet(packet, batch_size):
    result = {}
    for key, value in packet.items():
        if not torch.is_tensor(value):
            result[key] = value
            continue
        if key == "valid_mask" and value.ndim == 1:
            value = value.unsqueeze(0)
        elif key != "valid_mask" and value.ndim == 2:
            value = value.unsqueeze(0)
        result[key] = value
    if result["valid_mask"].shape[0] == 1 and batch_size > 1:
        for key, value in result.items():
            if torch.is_tensor(value) and value.shape[0] == 1:
                result[key] = value.expand(batch_size, *value.shape[1:])
    return result

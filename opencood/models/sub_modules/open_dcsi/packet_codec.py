"""Serializable common/token codecs, budgets, and packet selection."""

from dataclasses import dataclass
import math
import struct

import torch
import torch.nn as nn


_PRECISION_CODE = {"fp16": 1, "int8": 2, "fp32": 3}
_TOKEN_FLOAT_FIELDS = (
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
_TOKEN_INTEGER_FIELDS = (
    "scenario_index",
    "agent_global_index",
    "agent_local_index",
    "proposal_id",
    "source_scale",
)


@dataclass
class TensorPacket:
    payload: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple
    precision: str
    channel_axis: int = -1

    _HEADER_FORMAT = "<4sBBBbIIQ"

    @property
    def payload_bytes(self):
        return self.payload.numel() * self.payload.element_size()

    @property
    def metadata_bytes(self):
        return (
            struct.calcsize(self._HEADER_FORMAT)
            + 4 * len(self.original_shape)
            + 4 * self.scales.numel()
        )

    @property
    def nbytes(self):
        return self.payload_bytes + self.metadata_bytes

    def to_bytes(self):
        header = struct.pack(
            self._HEADER_FORMAT,
            b"ODCT",
            1,
            _PRECISION_CODE[self.precision],
            len(self.original_shape),
            self.channel_axis,
            self.scales.numel(),
            self.payload.numel(),
            self.payload_bytes,
        )
        dimensions = struct.pack(
            "<{}I".format(len(self.original_shape)), *self.original_shape
        )
        scales = (
            self.scales.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
        )
        payload = self.payload.detach().cpu().contiguous().numpy().tobytes()
        serialized = header + dimensions + scales + payload
        if len(serialized) != self.nbytes:
            raise RuntimeError("Open-DCSI tensor packet byte accounting mismatch")
        return serialized


def encode_tensor_packet(tensor, precision, per_channel_scale=False):
    precision = precision.lower()
    if precision == "fp16":
        return TensorPacket(
            payload=tensor.detach().to(torch.float16),
            scales=tensor.new_empty(0, dtype=torch.float32),
            original_shape=tuple(tensor.shape),
            precision=precision,
        )
    if precision == "fp32":
        return TensorPacket(
            payload=tensor.detach().to(torch.float32),
            scales=tensor.new_empty(0, dtype=torch.float32),
            original_shape=tuple(tensor.shape),
            precision=precision,
        )
    if precision != "int8":
        raise ValueError("Open-DCSI tensor precision must be fp16, int8, or fp32")
    if per_channel_scale and tensor.ndim < 2:
        per_channel_scale = False
    if per_channel_scale:
        reduce_dims = tuple(index for index in range(tensor.ndim) if index != 1)
        maximum = tensor.detach().abs().amax(dim=reduce_dims)
        scales = (maximum / 127.0).clamp_min(1e-8).to(torch.float32)
        view_shape = [1] * tensor.ndim
        view_shape[1] = tensor.shape[1]
        quantized = torch.round(tensor.detach() / scales.view(view_shape)).clamp(
            -127, 127
        )
        channel_axis = 1
    else:
        scale = (tensor.detach().abs().max() / 127.0).clamp_min(1e-8)
        scales = scale.reshape(1).to(torch.float32)
        quantized = torch.round(tensor.detach() / scale).clamp(-127, 127)
        channel_axis = -1
    return TensorPacket(
        payload=quantized.to(torch.int8),
        scales=scales,
        original_shape=tuple(tensor.shape),
        precision=precision,
        channel_axis=channel_axis,
    )


def decode_tensor_packet(packet, device, dtype):
    if packet.precision in ("fp16", "fp32"):
        return packet.payload.to(device=device, dtype=dtype).reshape(packet.original_shape)
    payload = packet.payload.to(device=device, dtype=dtype).reshape(packet.original_shape)
    scales = packet.scales.to(device=device, dtype=dtype)
    if packet.channel_axis == 1:
        shape = [1] * len(packet.original_shape)
        shape[1] = packet.original_shape[1]
        return payload * scales.reshape(shape)
    return payload * scales[0]


class TensorCodec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.precision = config["precision"].lower()
        self.per_channel_scale = bool(config.get("per_channel_scale", False))

    def forward(self, tensor):
        packet = encode_tensor_packet(
            tensor, self.precision, self.per_channel_scale
        )
        decoded = decode_tensor_packet(packet, tensor.device, tensor.dtype)
        if self.training:
            decoded = tensor + (decoded - tensor).detach()
        return decoded, packet


@dataclass
class TokenPacket:
    float_packets: dict
    integer_fields: dict
    token_count: int

    _HEADER_FORMAT = "<4sB3xII"
    _DIRECTORY_FORMAT = "<BBHI"

    @property
    def payload_bytes(self):
        float_bytes = sum(packet.payload_bytes for packet in self.float_packets.values())
        integer_bytes = sum(value.numel() * 4 for value in self.integer_fields.values())
        return float_bytes + integer_bytes

    @property
    def metadata_bytes(self):
        directory_count = len(self.float_packets) + len(self.integer_fields)
        tensor_metadata = sum(packet.metadata_bytes for packet in self.float_packets.values())
        return (
            struct.calcsize(self._HEADER_FORMAT)
            + directory_count * struct.calcsize(self._DIRECTORY_FORMAT)
            + tensor_metadata
        )

    @property
    def nbytes(self):
        return self.payload_bytes + self.metadata_bytes

    def to_bytes(self):
        header = struct.pack(
            self._HEADER_FORMAT,
            b"ODTK",
            1,
            self.token_count,
            len(self.float_packets) + len(self.integer_fields),
        )
        chunks = [header]
        field_id = 0
        for _, packet in self.float_packets.items():
            encoded = packet.to_bytes()
            chunks.append(
                struct.pack(self._DIRECTORY_FORMAT, field_id, 1, 0, len(encoded))
            )
            chunks.append(encoded)
            field_id += 1
        for _, value in self.integer_fields.items():
            encoded = value.detach().to(torch.int32).cpu().contiguous().numpy().tobytes()
            chunks.append(
                struct.pack(self._DIRECTORY_FORMAT, field_id, 2, 0, len(encoded))
            )
            chunks.append(encoded)
            field_id += 1
        serialized = b"".join(chunks)
        if len(serialized) != self.nbytes:
            raise RuntimeError("Open-DCSI token packet byte accounting mismatch")
        return serialized


class TokenCodec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.precision = config["precision"].lower()

    def forward(self, tokens):
        float_packets = {
            field: encode_tensor_packet(tokens[field], self.precision)
            for field in _TOKEN_FLOAT_FIELDS
        }
        integer_fields = {field: tokens[field] for field in _TOKEN_INTEGER_FIELDS}
        packet = TokenPacket(
            float_packets=float_packets,
            integer_fields=integer_fields,
            token_count=int(tokens["scenario_index"].numel()),
        )
        decoded = dict(tokens)
        for field, tensor_packet in float_packets.items():
            reconstructed = decode_tensor_packet(
                tensor_packet, tokens[field].device, tokens[field].dtype
            )
            if self.training:
                reconstructed = tokens[field] + (
                    reconstructed - tokens[field]
                ).detach()
            decoded[field] = reconstructed
        return decoded, packet


def _select_token_rows(tokens, indices):
    token_count = int(tokens["scenario_index"].numel())
    result = {}
    for key, value in tokens.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == token_count:
            result[key] = value.index_select(0, indices)
        else:
            result[key] = value
    return result


class CommunicationManager(nn.Module):
    """Apply codecs and scene-local budgets without communicating ego packets."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["common_codec"]["enabled"]:
            self.common_codec = TensorCodec(config["common_codec"])
        if config["token_codec"]["enabled"]:
            self.token_codec = TokenCodec(config["token_codec"])

    def _scene_caps(self, dense_bytes):
        budget = self.config["budget"]
        if not budget["enabled"] or budget["mode"] == "fixed_tokens":
            return [math.inf for _ in dense_bytes]
        if budget["mode"] == "bytes":
            return [int(budget["bytes_per_frame"]) for _ in dense_bytes]
        return [
            int(value * float(budget["ratio_of_dense"])) for value in dense_bytes
        ]

    def process_common(
        self, common_scales, score_scales, record_len, materialize_dense=True
    ):
        scene_count = int(record_len.numel())
        dense_bytes = [0 for _ in range(scene_count)]
        starts = torch.cumsum(
            torch.cat((record_len.new_zeros(1), record_len[:-1])), dim=0
        ).tolist()
        candidates = []
        for scale_index, (common, score) in enumerate(zip(common_scales, score_scales)):
            for scene_index, count_value in enumerate(record_len.tolist()):
                count = int(count_value)
                start = int(starts[scene_index])
                for local_agent in range(1, count):
                    global_agent = start + local_agent
                    tensor = common[global_agent : global_agent + 1]
                    dense_bytes[scene_index] += tensor.numel() * 4
                    if hasattr(self, "common_codec"):
                        if materialize_dense:
                            decoded, packet = self.common_codec(tensor)
                        else:
                            packet = encode_tensor_packet(
                                tensor,
                                self.common_codec.precision,
                                self.common_codec.per_channel_scale,
                            )
                            decoded = None
                    else:
                        packet = encode_tensor_packet(tensor, "fp32")
                        decoded = tensor if materialize_dense else None
                    utility = float(
                        score[global_agent].detach().mean().cpu()
                    )
                    candidates.append(
                        (
                            scene_index,
                            -utility,
                            local_agent,
                            scale_index,
                            global_agent,
                            decoded,
                            packet,
                        )
                    )
        caps = self._scene_caps(dense_bytes)
        used = [0 for _ in range(scene_count)]
        decoded_scales = (
            [scale.clone() for scale in common_scales]
            if materialize_dense
            else list(common_scales)
        )
        quality_masks = [score.new_ones(score.shape) for score in score_scales]
        stats = {
            "dense_baseline_bytes": sum(dense_bytes),
            "common_payload_bytes": 0,
            "token_payload_bytes": 0,
            "metadata_bytes": 0,
            "packet_count": 0,
            "selected_common_packets": 0,
            "selected_tokens": 0,
            "bytes_per_collaborator": {},
            "tokens_per_collaborator": {},
        }
        common_packets = []
        for candidate in sorted(candidates, key=lambda item: item[:4]):
            scene, _, local_agent, scale, global_agent, decoded, packet = candidate
            fits = used[scene] + packet.nbytes <= caps[scene]
            if fits:
                if materialize_dense:
                    decoded_scales[scale][global_agent : global_agent + 1] = decoded
                used[scene] += packet.nbytes
                stats["common_payload_bytes"] += packet.payload_bytes
                stats["metadata_bytes"] += packet.metadata_bytes
                stats["packet_count"] += 1
                stats["selected_common_packets"] += 1
                key = "{}:{}".format(scene, local_agent)
                stats["bytes_per_collaborator"][key] = (
                    stats["bytes_per_collaborator"].get(key, 0) + packet.nbytes
                )
                common_packets.append(
                    {
                        "scene_index": scene,
                        "local_agent_index": local_agent,
                        "global_agent_index": global_agent,
                        "scale_index": scale,
                        "packet": packet,
                    }
                )
            else:
                if materialize_dense:
                    decoded_scales[scale][global_agent].zero_()
                quality_masks[scale][global_agent].zero_()
        return decoded_scales, quality_masks, {
            "caps": caps,
            "used": used,
            "stats": stats,
            "common_packets": common_packets,
        }

    def _token_utility(self, tokens, reliability):
        if not self.config["selection"]["enabled"]:
            return torch.ones_like(reliability)
        mode = self.config["selection"]["score"]
        if mode in ("bgea_reliability", "official_bgea"):
            return reliability
        if mode == "confidence":
            return tokens["objectness"]
        if mode in ("localization_quality", "localization"):
            return torch.exp(-tokens["localization_uncertainty"].clamp_min(0.0))
        innovation = torch.linalg.vector_norm(
            tokens["innovation_embedding"], dim=-1
        )
        if mode in ("marginal_quality", "quality_adjusted_innovation"):
            return reliability * (tokens["box_quality"] - 0.5) * innovation
        if mode == "random":
            if not self.training and self.config["selection"]["deterministic_inference"]:
                value = tokens["proposal_id"].to(torch.float32) * 12.9898
                return torch.frac(torch.sin(value) * 43758.5453)
            return torch.rand_like(reliability)
        raise ValueError("Unsupported Open-DCSI token selection score: {}".format(mode))

    def process_tokens(self, tokens, quality_router, state):
        count = int(tokens["scenario_index"].numel())
        if count == 0:
            self._finish_stats(state)
            return tokens, state["stats"]
        reliability = quality_router(tokens)
        utility = self._token_utility(tokens, reliability)
        ego_mask = tokens["agent_local_index"] == 0
        selected = torch.nonzero(ego_mask, as_tuple=False).flatten().tolist()
        budget = self.config["budget"]
        candidates = []
        for index in torch.nonzero(~ego_mask, as_tuple=False).flatten().tolist():
            value = float(utility[index].detach().cpu())
            if (
                self.config["selection"]["enabled"]
                and self.config["selection"]["allow_negative_reject"]
                and value < 0
            ):
                continue
            candidates.append(
                (
                    int(tokens["scenario_index"][index].item()),
                    -value,
                    int(tokens["agent_local_index"][index].item()),
                    int(tokens["proposal_id"][index].item()),
                    index,
                )
            )
        selected_per_scene = [0 for _ in state["used"]]
        for scene, _, local_agent, _, index in sorted(candidates):
            row = _select_token_rows(
                tokens,
                torch.as_tensor([index], device=utility.device, dtype=torch.long),
            )
            codec = (
                self.token_codec
                if hasattr(self, "token_codec")
                else TokenCodec({"precision": "fp32"})
            )
            _, packet = codec(row)
            if budget["enabled"] and budget["mode"] == "fixed_tokens":
                fits = selected_per_scene[scene] < int(budget["fixed_tokens"])
            else:
                fits = state["used"][scene] + packet.nbytes <= state["caps"][scene]
            if not fits:
                continue
            selected.append(index)
            selected_per_scene[scene] += 1
            state["used"][scene] += packet.nbytes
            state["stats"]["token_payload_bytes"] += packet.payload_bytes
            state["stats"]["metadata_bytes"] += packet.metadata_bytes
            state["stats"]["packet_count"] += 1
            state["stats"]["selected_tokens"] += 1
            key = "{}:{}".format(scene, local_agent)
            state["stats"]["bytes_per_collaborator"][key] = (
                state["stats"]["bytes_per_collaborator"].get(key, 0) + packet.nbytes
            )
            state["stats"]["tokens_per_collaborator"][key] = (
                state["stats"]["tokens_per_collaborator"].get(key, 0) + 1
            )
        selected_tensor = torch.as_tensor(
            sorted(selected), device=utility.device, dtype=torch.long
        )
        selected_tokens = _select_token_rows(tokens, selected_tensor)
        collaborator_mask = selected_tokens["agent_local_index"] != 0
        if collaborator_mask.any():
            codec = (
                self.token_codec
                if hasattr(self, "token_codec")
                else TokenCodec({"precision": "fp32"})
            )
            for collaborator_index in torch.nonzero(
                collaborator_mask, as_tuple=False
            ).flatten().tolist():
                row_index = torch.as_tensor(
                    [collaborator_index], device=utility.device, dtype=torch.long
                )
                row = _select_token_rows(selected_tokens, row_index)
                decoded_row, _ = codec(row)
                for field in _TOKEN_FLOAT_FIELDS:
                    selected_tokens[field] = selected_tokens[field].clone()
                    selected_tokens[field][collaborator_index] = decoded_row[field][0]
        self._finish_stats(state)
        return selected_tokens, state["stats"]

    @staticmethod
    def _finish_stats(state):
        stats = state["stats"]
        stats["total_bytes"] = (
            stats["common_payload_bytes"]
            + stats["token_payload_bytes"]
            + stats["metadata_bytes"]
        )
        dense = stats["dense_baseline_bytes"]
        stats["compression_ratio"] = (
            float(stats["total_bytes"]) / dense if dense > 0 else 0.0
        )

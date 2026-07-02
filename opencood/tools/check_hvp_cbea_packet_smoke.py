"""Smoke test for HEAL-XLab-v3 HVP-CBEA packet modules."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.models.sub_modules.hvp_cbea_packet import (
    HypothesisEvidencePacketizer,
    PacketAggregator,
    PacketCompressor,
)


def main():
    torch.manual_seed(17)
    bev = torch.randn(1, 64, 32, 32, requires_grad=True)

    packetizer = HypothesisEvidencePacketizer(
        in_channels=64,
        topk=20,
        packet_dim=16,
        descriptor_dim=8,
    )
    packet = packetizer(bev)
    assert packet["boxes"].shape == (1, 20, 7)
    assert packet["centers"].shape == (1, 20, 2)
    assert packet["scores"].shape == (1, 20, 1)
    assert packet["uncertainty"].shape == (1, 20, 1)
    assert packet["descriptor"].shape == (1, 20, 8)
    assert packet["valid_mask"].shape == (1, 20)

    compressor = PacketCompressor(
        quantize="fp16",
        bandwidth_budget_kb=8,
        topk=20,
        descriptor_dim=8,
        packet_dim=16,
    )
    compressed_packet, comm_stats = compressor(packet)
    assert comm_stats["bytes_per_frame"] > 0
    assert comm_stats["kb_per_frame"] > 0
    assert comm_stats["num_packets"] == 20
    assert comm_stats["quantize_mode"] == "fp16"

    aggregator = PacketAggregator(
        context_channels=64,
        packet_dim=16,
        descriptor_dim=8,
    )
    delta_feature, aggregation_debug, _ = aggregator(bev, compressed_packet)
    assert delta_feature.shape == bev.shape
    assert torch.isfinite(delta_feature).all()
    assert aggregation_debug["packet_valid_count"] == 20

    loss = delta_feature.mean()
    loss.backward()
    trainable_params = [param for param in aggregator.parameters() if param.requires_grad]
    assert any(param.grad is not None for param in trainable_params)
    assert any(
        param.grad is not None
        and torch.isfinite(param.grad).all()
        and param.grad.detach().abs().sum() > 0
        for param in trainable_params
    )

    empty_packet = packetizer.empty_packet(batch_size=1, device=bev.device, dtype=bev.dtype)
    empty_packet["valid_mask"].zero_()
    empty_delta, empty_debug, _ = aggregator(bev.detach(), empty_packet)
    assert empty_delta.shape == bev.shape
    assert torch.isfinite(empty_delta).all()
    assert empty_delta.detach().abs().max() < 1e-6
    assert empty_debug["packet_empty"]

    print("HVP-CBEA packet smoke OK")
    print("HVP-CBEA packet backward OK")
    print("HVP-CBEA packet empty OK")


if __name__ == "__main__":
    main()

# HEAL-XLab-v3.0 HVP-CBEA Packet Mode

## Goal

Packet Mode models cross-vendor collaborative perception under realistic communication constraints:

- no raw sensor data sharing.
- no dense intermediate BEV feature sharing in deployment.
- no vendor-private model sharing.
- collaborators send standardized hypothesis/evidence packets.
- ego receives packets and performs packet-level evidence aggregation and residual refinement.

The implementation is an incremental path on top of HVP-CBEA v2.5. It is disabled by default.

## Configuration

```yaml
model:
  args:
    hvp_cbea:
      enabled: true
      train_only_hvp: true
      fallback_on_error: true
      residual_gate:
        enabled: true
        alpha_init: 0.05
        alpha_max: 0.3
        learnable: true
      packet:
        enabled: false
        mode: packet_one_round
        topk: 50
        packet_dim: 16
        descriptor_dim: 8
        quantize: fp16
        send_uncertainty: true
        send_agent_quality: true
        send_timestamp: true
        bandwidth_budget_kb: 8
        deadline_ms: 100
        detach_packet: false
        debug: false
```

If `packet` is omitted, `packet.enabled=false`. If `hvp_cbea.enabled=false`, the wrapper still follows the official HEAL path.

## Data Boundary

During OPV2V/HEAL training, collaborator features are locally available inside the model. Packet Mode uses them only before the communication boundary:

```text
collaborator feature -> packetizer -> compressed packet -> ego packet aggregator
```

The packet aggregator does not receive dense collaborator BEV features. It receives compact packet tensors with boxes or centers, scores, uncertainty, descriptor, agent quality, validity mask, and communication metadata.

## Modules

- `HypothesisEvidencePacketizer`: creates top-K packet entries from BEV feature heatmaps. The current v3.0 implementation uses normalized pseudo boxes from top-K feature locations; future work can attach real detector box decoding here.
- `PacketCompressor`: simulates `none`, `fp32`, `fp16`, and placeholder `int8` quantization plus bandwidth-budget masking.
- `PacketAggregator`: applies uncertainty-aware weighting with `sigmoid(score) * sigmoid(agent_quality) * exp(-uncertainty)` and maps packet evidence to a BEV residual delta.
- `PacketCommunicationMeter`: reports packet count, bytes/frame, KB/frame, estimated Mbps, quantization mode, top-K, descriptor dimension, packet dimension, bandwidth budget, and saturation.

## Safety

- `packet.enabled=false` keeps the v2.5 HVP-CBEA feature path unchanged.
- Packet modules are instantiated only when `packet.enabled=true`, avoiding new checkpoint keys in the default v2.5 path.
- `train_only_hvp=true` includes the packet modules under HVP prefixes.
- Non-HVP HEAL BatchNorm buffers remain frozen in train-only mode.
- HVP-CBEA packet outputs are not detached unless `packet.detach_packet=true`.
- Official `heter_pyramid_collab.py`, logs, and v1 HBEC are not modified.

## Validation

CPU smoke tests cover packet import, packetizer shapes, compression stats, packet aggregation, backward gradients through packet aggregator parameters, and all-invalid empty packet behavior.

Follow-up server experiments should start with `packet.enabled=true` untrained sanity final_infer, then a short `train_only_hvp` fine-tune with communication debug enabled.

# PACT-CBEA Packet-Only v1

## Goal

`PACT_CBEA_PACKET_v1` is an independent packet-only collaborative perception
experiment. It does not reuse the HVP-CBEA feature-mode forward path and does
not perform HEAL dense collaborator fusion.

The experiment compares packet-only collaboration with ego-only HEAL, standard
HEAL feature sharing, and HVP-CBEA Feature Mode using both AP and measured
packet communication volume.

## Communication Boundary

Each agent first builds its local BEV feature. For every scene, only the ego
BEV becomes dense detection context. A collaborator BEV is consumed by the
local `PACTPacketizer`, then discarded from the ego-side forward path. The ego
receives only compressed packet tensors and uses `PACTPacketAggregator` plus
`PACTPacketResidualFusion` to create a residual on its local BEV.

The packet v1 source is explicitly:

`feature_derived_pseudo_hypothesis`

It is a top-K pseudo-hypothesis interface derived from BEV features. It is not
a detector-decoded real 3D-box packet.

When `packet_only_strict=true`, a packet failure raises an error and dense
collaborator fallback is prohibited. The only optional recovery policy is
`packet_failure_policy: ego_only`; `dense_feature` is rejected by configuration.
The forward debug payload records that dense collaboration was not used.

## Training and Checkpoints

With `train_only_packet=true`, only these new modules are trainable:

- `pact_packetizer`
- `pact_packet_compressor`
- `pact_packet_aggregator`
- `pact_packet_residual_fusion`
- `pact_packet_comm_meter` (parameter-free meter)

All HEAL encoders, modality backbones, pyramid backbone, shrink layer, and
detection heads are frozen and held in eval mode to prevent BatchNorm buffer
drift. There is no centralized Stage3 joint training and no trainable global
PACT rule.

The packet model accepts a composed m1 Stage1 plus m2/m3/m4 Stage2 expert
checkpoint. Missing keys are accepted only for the new `pact_packet*` modules.
Any missing base key or unexpected HVP Feature Mode key is reported and fails
loading.

## Evaluation

Report AP at the normal IoU thresholds together with `packet_kb_per_frame`,
`estimated_mbps`, packet count, and bandwidth saturation. The key comparisons
must keep the dataset split, base local experts, and detector heads fixed:

- Ego-only local BEV
- Official HEAL dense feature collaboration
- HVP-CBEA Feature Mode
- PACT-CBEA packet-only v1

# PACT-CBEA Packet No-Joint v1

## Positioning

`PACT_CBEA_PACKET_NOJOINT_v1` is an inference-only experiment assembled from
the independently trained PACT-CBEA v1 m1 Stage1 and m2/m3/m4 Stage2 local
experts. It has no Stage3 training, no centralized optimization, and no
cross-vehicle gradient path.

Each vehicle computes its own local evidence head. The collaborator sends only
a top-K packet built from local evidence heatmap and uncertainty. The ego
never receives collaborator dense BEV, dense evidence heatmap, uncertainty
map, descriptor, or evidence feature.

## Packet Interface

The v1 packet source is `topk_local_evidence`. It is not a detector-decoded
box packet. A packet contains normalized BEV `x,y`, confidence, uncertainty,
modality id, agent id, and a valid bit. Descriptor is explicitly disabled:
the existing local experts do not train descriptor loss, so it is not used for
communication or aggregation.

With `fp16`, the transport accounting is 10 bytes per valid packet: four
fp16 values (`x,y,confidence,uncertainty`) plus uint8 modality and agent ids.
The valid mask is packet framing metadata, not an evidence map transmission.

## Fixed Ego Rule

The ego maps coordinates into its local grid and computes:

`reliability = confidence * exp(-uncertainty) * modality_prior`

Packet collisions use configured parameter-free `max` or `sum` reduction. The
packet evidence map then applies the fixed inference rule:

`enhanced_ego_feature = ego_feature * (1 + fixed_gain * packet_evidence_map)`

No learned gate, convolution, MLP, attention, descriptor, or dense PACT rule
is present. No valid packet returns the exact ego feature.

## Communication Boundary

The no-joint wrapper calls `forward_single` independently for local experts;
it never calls collaborative pyramid fusion. Collaborator single features are
used only to evaluate their frozen local evidence head, then discarded after
packetization. The detection heads receive only the ego dense feature after
the fixed packet map modulation.

Runtime debug asserts `packet_only_verified`, `no_joint_training_verified`,
`stage3_training_required=false`, `dense_collab_fusion_used=false`,
`collaborator_dense_after_packet_used=false`, and
`full_evidence_map_transmitted=false`.

## Checkpoints and Evaluation

Load `PACT_CBEA_v1/rule_cbea/net_epoch1.pth`, composed from the four local
experts. The no-joint packet components have zero parameters, so loading must
not require a new Stage3 checkpoint. Unexpected missing or checkpoint keys
fail loading.

Report AP alongside packet count, KB/frame, Mbps estimate, bandwidth
saturation, and comparisons with ego-only, official HEAL, dense PACT-CBEA
rule mode, and the trainable `PACT_CBEA_PACKET_v1` experiment.

# PACT-CBEA Plan

PACT-CBEA stands for Plug-and-play Agent Collaboration with Trust-calibrated Collaborative Bayesian Evidence Aggregation.

PACT-CBEA is the HEAL-compatible, no-joint-training, plug-and-play counterpart to the centralized HVP_CBEA_v3 Stage3 line.

## Position

HVP_CBEA_v3 remains useful as a centralized Stage3 aggregator upper-bound and feature-level mechanism validation experiment. It merges m1/m2/m3/m4 checkpoints and can train a cross-branch HVP-CBEA aggregator.

PACT-CBEA has a different deployment-oriented goal:

- No centralized Stage3 joint training.
- No trainable global aggregator by default.
- No retraining old expert branches when a new vehicle or modality joins.
- Local expert checkpoint composition plus fixed trust-calibrated evidence routing.

## PACT-CBEA v1 Feature Mode

The second increment makes `PACT_CBEA_v1` a formally trainable Feature Mode.
It still transmits BEV feature tensors in the collaborative path; packet-only
communication is a later, separate experiment.

```text
PACT local expert training
  -> m1/m2/m3/m4 local BEV feature
  -> local evidence heatmap logits
  -> local uncertainty / log-variance proxy

PACT collaborative inference
  -> per-agent BEV feature + evidence + uncertainty
  -> warp all three tensors to ego coordinates
  -> parameter-free trust-calibrated rule
  -> enhanced collaborative feature
  -> existing HEAL detection heads
```

Local evidence heads are trainable inside their own expert branches:

- `PACT_CBEA_v1/stage1/m1_base`
- `PACT_CBEA_v1/stage2/m2_alignto_m1`
- `PACT_CBEA_v1/stage2/m3_alignto_m1`
- `PACT_CBEA_v1/stage2/m4_alignto_m1`

There is no centralized Stage3 training and no trainable global PACT
aggregator. The global rule remains parameter-free.

The rule computes:

```text
evidence_confidence = sigmoid(evidence_heatmap)
uncertainty_weight = exp(-evidence_uncertainty)
spatial_consistency = exp(-abs(agent_evidence_confidence - ego_evidence_confidence))
reliability =
  evidence_confidence
  * uncertainty_weight
  * spatial_consistency
  * modality_prior
alpha = reliability / sum(reliability)
enhanced_feature = sum_i alpha_i * feature_i
```

The current main experiment keeps modality priors neutral:

```yaml
modality_prior:
  m1: 1.0
  m2: 1.0
  m3: 1.0
  m4: 1.0
```

If local evidence is unavailable in the collaborative wrapper, PACT-CBEA does
not average unaligned raw agent features. It falls back to the HEAL base
collaborative feature or the ego-only path, and records the fallback in
`output_dict["pact_cbea"]`.

## Plug-and-play Onboarding

When a new vendor or modality such as `m5` is added, the intended path is:

1. Train only the new local expert branch.
2. Emit the unified evidence representation expected by PACT-CBEA.
3. Add a modality prior for `m5` if desired.
4. Compose the local expert checkpoint with existing branches.

No old branch retraining and no global aggregator retraining are required.

## Relation To Packet Mode

Packet mode is a future communication interface enhancement. It can later carry compact evidence across agents or vendors.

The current PACT-CBEA increment does not implement packet mode. Its focus is
the HEAL no-joint-training and plug-and-play paradigm with feature-level
evidence routing. Feature, evidence, and uncertainty maps must all be warped
to the ego coordinate system before rule aggregation.

## Relation To HVP_CBEA_v3

- `HVP_CBEA_v3`: centralized Stage3 aggregator upper-bound and mechanism validation.
- `PACT_CBEA_v1`: HEAL-compatible deployment-oriented method validation.

PACT-CBEA reuses the staged local expert checkpoints but removes the centralized trainable Stage3 aggregator.

## Files

- `opencood/models/sub_modules/pact_cbea_rule.py`
- `opencood/models/sub_modules/pact_cbea_evidence_head.py`
- `opencood/models/heter_pyramid_single_pact_cbea.py`
- `opencood/models/heter_pyramid_collab_pact_cbea.py`
- `opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/pact/stage1/m1_local_evidence.yaml`
- `opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/pact/stage2/m2_local_evidence_adapt.yaml`
- `opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/pact/stage2/m3_local_evidence_adapt.yaml`
- `opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/pact/stage2/m4_local_evidence_adapt.yaml`
- `opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/pact/cbea_rule.yaml`
- `opencood/tools/prepare_pact_cbea.py`
- `opencood/tools/check_pact_cbea_smoke.py`

## Checkpoint Composition

Prepare the initialization checkpoint with:

```bash
python opencood/tools/prepare_pact_cbea.py
```

The output is:

```text
opencood/logs/PACT_CBEA_v1/rule_cbea/net_epoch1.pth
```

The PACT-CBEA rule module is parameter-free, so it has no trainable checkpoint keys by design.
The helper now defaults to `opencood/logs/PACT_CBEA_v1/...` local expert
checkpoints, validates local evidence head keys, avoids silent overwrite unless
`--force` is passed, and writes a manifest with source paths, file sizes,
SHA256 hashes, the output checkpoint path, and the current Git commit.

## Increment Boundary

This v1 Feature Mode increment includes:

- Trainable local evidence heads for m1/m2/m3/m4.
- PACT-specific Stage1/Stage2 local expert YAML.
- Rule-based evidence routing with uncertainty and spatial consistency.
- HEAL-compatible model entry.
- Local expert checkpoint composition helper.
- CPU smoke tests.

This increment does not include:

- Packet-only communication.
- Federated learning.
- Full final-inference automation.
- Centralized Stage3 joint training.
- Any change to existing HVP_CBEA_v3 training logic.

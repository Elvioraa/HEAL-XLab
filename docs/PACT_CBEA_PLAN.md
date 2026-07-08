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

## PACT-CBEA v1

The first increment implements `PACT_CBEA_v1/rule_cbea`:

```text
local expert branches
  -> per-agent BEV/evidence
  -> parameter-free trust-calibrated rule
  -> enhanced collaborative feature
  -> existing HEAL detection heads
```

The rule computes:

```text
evidence_confidence = sigmoid(evidence_heatmap)
uncertainty_weight = exp(-evidence_uncertainty)
reliability = evidence_confidence * uncertainty_weight * modality_prior
alpha = reliability / sum(reliability)
enhanced_feature = sum_i alpha_i * feature_i
```

If evidence is unavailable, PACT-CBEA falls back safely:

- Missing evidence heatmap -> confidence is 1.
- Missing uncertainty -> uncertainty weight is 1.
- Missing modality names -> modality prior is 1.
- Only features available -> uniform average.

Every fallback is recorded in `output_dict["pact_cbea"]`.

## Plug-and-play Onboarding

When a new vendor or modality such as `m5` is added, the intended path is:

1. Train only the new local expert branch.
2. Emit the unified evidence representation expected by PACT-CBEA.
3. Add a modality prior for `m5` if desired.
4. Compose the local expert checkpoint with existing branches.

No old branch retraining and no global aggregator retraining are required.

## Relation To Packet Mode

Packet mode is a future communication interface enhancement. It can later carry compact evidence across agents or vendors.

The current PACT-CBEA increment does not implement packet mode. Its focus is restoring the HEAL no-joint-training and plug-and-play paradigm with a rule-based feature/evidence routing scaffold.

## Relation To HVP_CBEA_v3

- `HVP_CBEA_v3`: centralized Stage3 aggregator upper-bound and mechanism validation.
- `PACT_CBEA_v1`: HEAL-compatible deployment-oriented method validation.

PACT-CBEA reuses the staged local expert checkpoints but removes the centralized trainable Stage3 aggregator.

## Files

- `opencood/models/sub_modules/pact_cbea_rule.py`
- `opencood/models/heter_pyramid_collab_pact_cbea.py`
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

## Increment Boundary

This first increment includes:

- Rule-based evidence routing scaffold.
- HEAL-compatible model entry.
- Local expert checkpoint composition helper.
- CPU smoke tests.

This first increment does not include:

- Packet mode.
- Federated learning.
- Full final-inference automation.
- Stage1/Stage2 rewrites.
- Any change to existing HVP_CBEA_v3 training logic.

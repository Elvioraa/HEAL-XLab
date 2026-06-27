# HEAL-XLab-v2-HVP-CBEA-Trainable

Method: Hypothesis Verification Protocol + Cooperative Bayesian Evidence Accumulation.

HVP-CBEA is a model-forward-level experimental direction. It is independent of v1 HBEC, which remains a post-process hook.

## Design

- `HypothesisEncoder` generates sparse object hypotheses from BEV features.
- `HypothesisVerifier` uses collaborator BEV features to produce CONFIRM / REFUTE / REFINE logits, box deltas, and optional novel hypotheses.
- `BayesianHypothesisFusion` performs a Bayesian-style log-odds update and scatters hypothesis evidence back into BEV feature space before the official detection heads.
- Official `cls_head`, `reg_head`, and `dir_head` remain the final prediction heads.
- Collaborator BEV features are projected with a lightweight 1x1 adapter when their channel count differs from the post-fusion head feature.

## Integration

- Plan B is used: new model file `opencood/models/heter_pyramid_collab_hvp_cbea.py`.
- Official `opencood/models/heter_pyramid_collab.py` is not modified.
- HVP-CBEA is enabled only by selecting `model.core_method: heter_pyramid_collab_hvp_cbea` and setting `model.args.hvp_cbea.enabled: true`.
- With `enabled=false`, the new model bypasses all HVP-CBEA logic.

## Current Scope

- Forward modules are implemented.
- Loss integration is safe and optional through `output_dict['hvp_cbea_loss']`.
- GT-dependent auxiliary losses are defensive and return zero when GT format is not available.
- No inference-time GT is used.
- No raw camera feature bypass is introduced.

## Fallbacks

- No hypotheses: return the original BEV feature.
- No collaborator features: verifier returns neutral logits/deltas; fusion can still fall back.
- Shape mismatch or exception with `fallback_on_error=true`: return the official fused feature.

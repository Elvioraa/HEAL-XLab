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
- v2.1 adds `train_only_hvp`, allowing full collaborative fine-tuning while freezing official HEAL parameters.
- GT-dependent auxiliary losses are defensive and return zero when GT format is not available.
- No inference-time GT is used.
- No raw camera feature bypass is introduced.

## Fallbacks

- No hypotheses: return the original BEV feature.
- No collaborator features: verifier returns neutral logits/deltas; fusion can still fall back.
- Shape mismatch or exception with `fallback_on_error=true`: return the official fused feature.

## v2.1 Training Freeze

`model.args.hvp_cbea.train_only_hvp: true` freezes all inherited HEAL parameters and keeps only HVP-CBEA modules trainable. This protects the HEAL_m1_based baseline during fine-tuning.

This method is collaborative and depends on the `heter_pyramid_collab_hvp_cbea` forward variables (`heter_feature_2d`, `record_len`, collaborator BEV features). It is not compatible with the official `heter_pyramid_single` stage2/m2 training path.

## v2.2 Single-supervision Compatibility

When `model.args.supervise_single: true`, the wrapper preserves train-required single prediction outputs:

- `cls_preds_single`
- `reg_preds_single`
- `dir_preds_single`

These are produced from the official per-agent BEV route: `pyramid_backbone.forward_single(heter_feature_2d)`, optional `shrink_conv`, then the shared detection heads. HVP-CBEA still only injects into the collaborative `fused_feature`.

## v2.3 Autograd-safe HVP-CBEA

The v2.2 fine-tune smoke reached `final_loss.backward()` and exposed an inplace autograd version error in HVP-CBEA hypothesis tensors. v2.3 makes the HVP-CBEA submodules autograd-safe:

- `nn.ReLU(inplace=True)` is replaced with `nn.ReLU(inplace=False)` in HVP-CBEA submodules.
- Novel hypotheses and updated hypotheses are rebuilt out-of-place with `torch.cat()` instead of slice assignment.
- Hypothesis-to-BEV scatter uses out-of-place `scatter_add()` rows and `torch.stack()` instead of indexed assignment into a preallocated map.
- HVP-CBEA outputs are not detached, so detection loss remains able to train HVP-CBEA parameters.

The smoke test now includes `loss.backward()` and verifies that HVP-CBEA parameters receive gradients. `enabled=false` behavior and the `train_only_hvp` freeze rule are unchanged.

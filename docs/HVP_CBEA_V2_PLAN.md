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
- v2.4 keeps inherited HEAL BatchNorm buffers frozen in `train_only_hvp` mode while leaving HVP-CBEA modules trainable.
- v2.5 wraps the HVP-CBEA feature update in a bounded residual gate for identity-start safety.
- v2.6 optionally suppresses the residual gate in ego-only scenes with no collaborator evidence.
- v2.7-a adds default-off auxiliary loss plumbing for residual, alpha, and placeholder refinement consistency terms.
- v2.7-b adds default-off GT/anchor-guided supervision for hypothesis heatmaps and residual focus.
- v3.0 adds an optional packet communication path for low-bandwidth cross-vendor experiments.
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

## v2.4 Frozen HEAL BN Buffer Fix

The v2.3 `train_only_hvp` mode froze inherited HEAL parameters, but the inherited HEAL backbone stayed in train mode. BatchNorm buffers such as `running_mean`, `running_var`, and `num_batches_tracked` could therefore change during HVP-only fine-tuning.

v2.4 tightens the freeze rule:

- In `enabled=true` and `train_only_hvp=true`, all non-HVP BatchNorm and SyncBatchNorm modules are kept in eval mode.
- HVP-CBEA modules still return to train mode after `model.train()`, so their parameters and internal BatchNorm statistics can train normally.
- `enabled=false` remains equivalent to the official HEAL forward path.
- The debug summary reports `train_only_hvp`, `frozen_bn_eval_count`, and `hvp_bn_train_count`.

## v2.5 Residual Gate / Identity-start

The v2.4 standard final_infer result improved AP@0.3 and AP@0.5 clearly and kept AP@0.7 positive across all CAV settings, but high-IoU localization still lagged the +3pp target. The next safety patch makes the HVP-CBEA BEV update less aggressive:

```text
fused_feature_out = fused_feature + alpha * delta_feature
```

`BayesianHypothesisFusion` still builds the hypothesis evidence feature and internal feature gate, but the direct replacement is converted into `delta_feature`. A bounded residual alpha then scales the update before it reaches the official detection heads.

Default residual gate:

- `enabled=true`
- `alpha_init=0.05`
- `alpha_max=0.3`
- `learnable=true`

The alpha parameter lives inside `bayesian_hypothesis_fusion`, so `train_only_hvp=true` keeps it trainable under the existing HVP-CBEA prefix rule. The output is not detached, and detection loss can still backpropagate through `bayesian_hypothesis_fusion`, `hypothesis_verifier`, `hypothesis_encoder`, and the collaborator projection.

Debug fields include `hvp_cbea_residual_alpha`, `hvp_cbea_delta_norm`, `hvp_cbea_delta_mean`, and `hvp_cbea_delta_std`. The smoke test checks that residual alpha exists and receives gradient. Follow-up server work should run an enabled=true untrained sanity pass and a short `train_only_hvp` fine-tune.

## v2.6 Collaboration-aware Residual Gate

v2.5 long training showed that HVP-CBEA is useful in multi-CAV scenes, especially for AP@0.7, but can degrade `use_cav1`. v2.6 adds an optional collaboration-aware multiplier to suppress residual updates when `record_len` indicates no collaborator:

```text
effective_alpha = alpha * collaboration_scale
fused_feature_out = fused_feature + effective_alpha * delta_feature
```

Default config keeps this disabled:

```yaml
residual_gate:
  collaboration_aware:
    enabled: false
    no_collab_scale: 0.0
    collab_scale: 1.0
    min_cav: 2
    use_record_len: true
    fallback_scale: 1.0
    debug: false
```

Expected ablation:

- `use_cav1`: set `collaboration_scale=0.0` and suppress HVP residual perturbation.
- `use_cav2/use_cav3/use_cav4`: keep `collaboration_scale=1.0` and preserve v2.5 residual behavior.

This is intended for inference-only ablation with a v2.5 checkpoint: turn on `collaboration_aware.enabled=true` in config, run final_infer, and compare whether `use_cav1` recovers while multi-CAV gains remain.

## v2.7-a Auxiliary Loss Plumbing

Current training logs can print `HVP-CBEA Loss: 0.0000` because the original HVP-CBEA loss hook is a gradient-safe zero placeholder. v2.7-a keeps that placeholder compatible and adds a config-gated auxiliary loss path:

```yaml
aux_loss:
  enabled: false
  residual_reg:
    enabled: false
    weight: 0.001
    type: l1
  alpha_reg:
    enabled: false
    weight: 0.001
    target: 0.05
  refinement_consistency:
    enabled: false
    weight: 0.05
    mode: feature_delta_l1
```

When enabled, `BayesianHypothesisFusion` exposes `delta_feature`, `hvp_residual`, `alpha`, and `effective_alpha` through `output_dict['hvp_cbea_aux']`. The standard PointPillar loss adds enabled auxiliary terms to the detection loss and reports `hvp_residual_reg_loss`, `hvp_alpha_reg_loss`, `hvp_refinement_consistency_loss`, and `hvp_aux_total_loss`.

The v2.7-a `refinement_consistency` term is deliberately simple: it regularizes `delta_feature` with L1 as a placeholder. v2.7-b should replace or extend it with GT-guided refinement supervision once the target format and matching contract are finalized.

Default behavior is unchanged: if `aux_loss.enabled=false` or the field is omitted, v2.6 residual and collaboration-aware semantics are preserved and no new checkpoint parameters are required.

## v2.7-b GT-guided Auxiliary Loss

v2.7-b turns the v2.7-a auxiliary plumbing into a target-aware supervision path while keeping it fully config-gated:

```yaml
aux_loss:
  enabled: false
  gt_guided:
    enabled: false
    hypothesis_heatmap:
      enabled: false
      weight: 0.05
      source: anchor_pos
      loss: bce
      pos_weight: 2.0
    residual_focus:
      enabled: false
      weight: 0.01
      source: anchor_pos
      bg_weight: 1.0
      fg_weight: 0.25
    residual_fg_boost:
      enabled: false
      weight: 0.005
      source: anchor_pos
      target: 0.0
```

The loss uses `target_dict["pos_equal_one"]` as an anchor-positive map. It supports common layouts such as `[B,H,W,A]`, `[B,A,H,W]`, and `[B,1,H,W]`, then resizes the foreground map to the HVP heatmap or residual spatial size.

The new terms are:

- `hvp_gt_hypothesis_heatmap_loss`: BCE supervision for `hypothesis_hmap`.
- `hvp_gt_residual_focus_loss`: stronger background residual regularization and lighter foreground residual regularization.
- `hvp_gt_residual_fg_boost_loss`: a conservative foreground residual L1 placeholder; v2.7-b does not force residual magnitude to grow.

The goal is to make HVP residuals concentrate on object regions, reduce background perturbation, and help AP@0.3/AP@0.5 while avoiding the ego-only degradation seen in earlier long runs. With `gt_guided.enabled=false`, v2.7-a behavior is unchanged; no new trainable parameters or checkpoint keys are introduced.

## v3.0 Packet Mode

Packet mode is a deployment-oriented experimental path for the no-raw-data, no-dense-feature-sharing setting. It is controlled by `model.args.hvp_cbea.packet.enabled` and defaults to `false`.

When `packet.enabled=false`, v2.5 behavior is unchanged: collaborator BEV features can still enter the existing HVP-CBEA verifier/fusion path. When `packet.enabled=true`, collaborator BEV features are used only before the communication boundary to generate compact packets. The ego-side packet aggregator receives standardized hypothesis/evidence packets and produces a residual delta feature; it does not consume dense collaborator BEV features.

The first packet implementation contains:

- `HypothesisEvidencePacketizer`: feature heatmap top-K pseudo hypotheses and descriptors.
- `PacketCompressor`: fp32/fp16/int8 communication simulation and budget masking.
- `PacketAggregator`: uncertainty-aware packet weighting and trainable packet-to-BEV delta projection.
- `PacketCommunicationMeter`: bytes, KB/frame, estimated Mbps, top-K, quantization, and budget saturation stats.

The packetizer currently uses normalized pseudo boxes from top-K feature locations as an interface placeholder. Future work can replace this with real detector box decoding without changing the packet aggregator boundary.

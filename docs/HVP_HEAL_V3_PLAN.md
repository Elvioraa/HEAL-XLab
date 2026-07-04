# HVP-HEAL v3 Feature Main Plan

HVP-HEAL v3 is the feature-main staged-training direction. It is not the v2.x plug-in route where HEAL is trained first, HVP-CBEA is inserted later, and the inherited HEAL trunk is frozen. In v3, HVP modules are intended to participate in the staged HEAL training process itself.

## Stage1 Hypothesis-aware Base Training

Stage1 adds a lightweight hypothesis head to the m1 base feature path:

```text
m1 input
  -> encoder_m1 / backbone_m1
  -> m1 ego BEV feature
      -> hypothesis head
      -> hypothesis heatmap loss
```

The original HEAL detection path remains intact. The hypothesis head receives the m1 ego pre-fusion BEV feature and predicts `hypothesis_heatmap_logits` plus `hypothesis_heatmap`. The loss is BCE-with-logits against an anchor-positive map derived from `target_dict["pos_equal_one"]`.

## Why This Is Not v2.x Plug-in

v2.x HVP-CBEA is an optional module inserted after a trained HEAL checkpoint and can be trained with `train_only_hvp=true`. v3 starts a new staged-training mainline:

- Stage1: hypothesis-aware base training.
- Stage2: not implemented in this skeleton.
- Stage3: not implemented in this skeleton.
- Packet and hybrid modes are not part of this Stage1 skeleton.

## Safety

- `hvp_v3.enabled=false` is the default.
- The new wrapper calls the official HEAL collab forward unchanged when disabled.
- Stage1 loss is added only when `output_dict["hvp_v3"]` exists and `aux_loss.enabled=true`.
- No v2.x HVP-CBEA files or v1 HBEC files are removed.
- No packet, hybrid, Stage2, or Stage3 logic is introduced.

## Stage1 Follow-up

The next server-side check should run a short Stage1 smoke training with:

```yaml
model:
  core_method: heter_pyramid_collab_hvp_heal_v3
  args:
    hvp_v3:
      enabled: true
      stage: stage1_hypothesis
```

The expected log fields are `HVP-v3 Loss` and `Stage1 Hypothesis Loss`.

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

## Stage1 Smoke Mode

The first Stage1 server run is a low-memory smoke mode:

```yaml
hvp_v3:
  stage1:
    train_mode: hypothesis_head_only
    freeze_base_model: true
    detach_bev_for_hypothesis: true
```

This mode is only meant to verify the real dataloader, train loop, loss wiring, gradients, checkpoint save/load shape, and logging. It freezes the inherited HEAL base model and trains only `hvp_v3_hypothesis_head`. The BEV feature passed into the hypothesis head is detached, so Stage1 hypothesis loss does not retain the full HEAL backbone graph.

## Stage1 Full Joint Mode

Full joint Stage1 training is reserved for follow-up:

```yaml
hvp_v3:
  stage1:
    train_mode: joint
    freeze_base_model: false
    detach_bev_for_hypothesis: false
```

Joint mode can train the base model and hypothesis head together, but it is not the default because it has much higher memory pressure and may require AMP or gradient checkpointing.

## Stage2 Evidence-aware Modality Adaptation

Stage2 adapts non-m1 modalities with evidence-aware outputs. This is the first v3 step where m2/m3/m4 learn to expose standardized evidence in addition to the normal HEAL single-modality detection outputs:

```text
m2/m3/m4 input
  -> modality encoder / backbone / aligner
  -> pyramid backbone / shrink conv
      -> original detection heads
      -> HVP-v3 evidence head
          -> evidence heatmap
          -> evidence uncertainty
          -> evidence descriptor
```

The official `heter_pyramid_single.py` remains unchanged. Stage2 uses `heter_pyramid_single_hvp_heal_v3` only when the yaml explicitly selects it and enables:

```yaml
hvp_v3:
  enabled: true
  stage: stage2_evidence
  stage2:
    train_mode: evidence_adaptation
    use_hvp_v3_evidence: true
  evidence_head:
    enable: true
  evidence_loss:
    enable: true
```

The Stage2 objective augments the existing single-modality HEAL detection training with:

```text
L = L_single_det
    + lambda_hmap * L_evidence_heatmap
    + lambda_unc * L_uncertainty
    + lambda_desc * L_descriptor
```

`L_evidence_heatmap` is BCE-with-logits against the anchor-positive map from `target_dict["pos_equal_one"]`. `L_uncertainty` is a foreground uncertainty penalty. `L_descriptor` is a lightweight descriptor smoothness regularizer until a later stage provides cross-modality descriptor targets. Existing depth and pyramid losses remain part of the normal HEAL Stage2 loss class.

## Why This Is Not v2.x Plug-in

v2.x HVP-CBEA is an optional module inserted after a trained HEAL checkpoint and can be trained with `train_only_hvp=true`. v3 starts a new staged-training mainline:

- Stage1: hypothesis-aware base training.
- Stage2: evidence-aware modality adaptation for m2/m3/m4.
- Stage3: not implemented in this skeleton.
- Packet and hybrid modes are not part of this Stage1 skeleton.

## Safety

- `hvp_v3.enabled=false` is the default.
- The new wrapper calls the official HEAL collab forward unchanged when disabled.
- Stage1 loss is added only when `output_dict["hvp_v3"]` exists and `aux_loss.enabled=true`.
- No v2.x HVP-CBEA files or v1 HBEC files are removed.
- Packet, hybrid, and Stage3 logic are still not introduced.
- `hvp_v3.enabled=false` and old `heter_pyramid_single` configs keep the original HEAL path.
- Stage2 checkpoint loading skips incompatible Stage1-only keys such as `hvp_v3_hypothesis_head`, `encoder_m1`, and `backbone_m1`.

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

## Stage2 Follow-up

The next server-side check should run:

```bash
python opencood/tools/check_hvp_heal_v3_stage2_smoke.py
```

Then launch the explicit v3 Stage2 yaml for the target modality, for example:

```text
opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage2/m2_evidence_adapt.yaml
opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage2/m3_evidence_adapt.yaml
opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage2/m4_evidence_adapt.yaml
```

A valid Stage2 training log should show `HVP-v3 Loss` and `Stage2 Evidence Loss`; a log with only Conf/Loc/Dir/Depth/Pyramid remains ordinary HEAL Stage2, not HVP-v3 Stage2.

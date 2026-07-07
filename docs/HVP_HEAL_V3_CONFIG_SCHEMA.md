# HVP-HEAL v3 Config Schema

Default v3 config is inert:

```yaml
model:
  args:
    hvp_v3:
      enabled: false
      stage: none
```

With `hvp_v3.enabled=false`, the v3 wrapper follows the official HEAL collaborative pyramid forward and emits no `output_dict["hvp_v3"]`.

## Stage1 Hypothesis-aware Base Training

Reference config:

```yaml
model:
  core_method: heter_pyramid_collab_hvp_heal_v3
  args:
    hvp_v3:
      enabled: true
      stage: stage1_hypothesis
      stage1:
        train_mode: hypothesis_head_only
        freeze_base_model: true
        detach_bev_for_hypothesis: true
      feature_main:
        enabled: true
      hypothesis_head:
        enabled: true
        in_channels: 64
        hidden_dim: 64
        out_channels: 1
        use_sigmoid: true
      aux_loss:
        enabled: true
        mode: stage1_hypothesis
        hypothesis_heatmap:
          enabled: true
          weight: 0.01
          pos_weight: 1.0
        residual_reg:
          enabled: false
        alpha_reg:
          enabled: false
        residual_focus:
          enabled: false
```

`stage1.train_mode: hypothesis_head_only` is the low-memory smoke mode. It freezes the inherited HEAL base model, keeps only `hvp_v3_hypothesis_head` trainable, and detaches the BEV feature before the hypothesis head so the backbone graph is not retained for backward.

Future full joint Stage1 training should be explicit:

```yaml
hvp_v3:
  stage1:
    train_mode: joint
    freeze_base_model: false
    detach_bev_for_hypothesis: false
```

The joint mode is not the default because it requires more memory and may need AMP or gradient checkpointing on server GPUs.

The hypothesis head outputs:

- `hypothesis_heatmap_logits`
- `hypothesis_heatmap`
- optional `hypothesis_feature`

The Stage1 loss uses `target_dict["pos_equal_one"]` and supports `[B,H,W,A]`, `[B,A,H,W]`, `[B,1,H,W]`, and compatible flat anchor layouts. If `pos_equal_one` is unavailable, the loss falls back to zero with a debug reason.

## Stage2 Evidence-aware Modality Adaptation

Reference config:

```yaml
model:
  core_method: heter_pyramid_single_hvp_heal_v3
  args:
    hvp_v3:
      enabled: true
      stage: stage2_evidence
      stage2:
        train_mode: evidence_adaptation
        use_hvp_v3_evidence: true
      evidence_head:
        enable: true
        in_channels: 256
        hidden_dim: 64
        descriptor_dim: 16
        use_sigmoid: true
        normalize_descriptor: true
      evidence_loss:
        enable: true
        mode: stage2_evidence
        evidence_heatmap:
          enable: true
          weight: 0.01
          pos_weight: 1.0
        uncertainty:
          enable: true
          weight: 0.001
        descriptor:
          enable: true
          weight: 0.001
```

`evidence_head.enable` and `evidence_loss.enable` are accepted aliases for `enabled`. Defaults remain disabled, so missing `hvp_v3` or missing Stage2 fields do not change official HEAL Stage2.

The Stage2 evidence head outputs:

- `evidence_heatmap_logits`: `[B,1,H,W]`
- `evidence_heatmap`: `[B,1,H,W]`
- `evidence_uncertainty_logits`: `[B,1,H,W]`
- `evidence_uncertainty`: `[B,1,H,W]`
- `evidence_descriptor`: `[B,descriptor_dim,H,W]`
- optional `evidence_feature`: `[B,hidden_dim,H,W]`

The output is stored under:

```python
output_dict["hvp_v3"] = {
    "stage": "stage2_evidence",
    "modality": "m2/m3/m4",
    ...
}
```

The Stage2 evidence loss is added by the existing `point_pillar_*_loss` path when `output_dict["hvp_v3"]["stage"] == "stage2_evidence"`. The total loss includes the normal single-modality detection loss plus the weighted evidence heatmap, uncertainty, and descriptor terms.

Checkpoint compatibility:

- Existing Stage2 checkpoints without evidence head keys can load with missing evidence keys.
- Stage1 checkpoints with extra `hvp_v3_hypothesis_head`, `encoder_m1`, or `backbone_m1` keys are filtered by the Stage2 wrapper.
- The official `heter_pyramid_single.py` and Stage1 wrapper are unchanged by Stage2 configs.

## Template Path

```text
opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage1/m1_hyp_base.yaml
opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage2/m2_evidence_adapt.yaml
opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage2/m3_evidence_adapt.yaml
opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage2/m4_evidence_adapt.yaml
```

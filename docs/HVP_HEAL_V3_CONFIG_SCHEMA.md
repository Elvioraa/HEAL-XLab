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

## Template Path

```text
opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage1/m1_hyp_base.yaml
```

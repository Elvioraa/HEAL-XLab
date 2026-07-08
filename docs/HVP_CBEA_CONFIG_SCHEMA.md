# HVP-CBEA Config Schema

Do not place this template under `logs`. Use it only as a reference for a future evaluated yaml.

The real configuration path is `model.args.hvp_cbea`, because `train_utils.create_model()` passes `hypes['model']['args']` directly to the model constructor.

```yaml
model:
  core_method: heter_pyramid_collab_hvp_cbea
  args:
    hvp_cbea:
      enabled: false
      in_channels: 256
      collaborator_in_channels: 64
      max_hypotheses: 50
      hyp_conf_threshold: 0.15
      mid_channels: 64
      novel_threshold: 0.5
      max_novel: 20
      confirm_boost: 1.5
      refute_penalty: -2.5
      refine_boost: 0.8
      loss_weight_encoder: 0.5
      loss_weight_verifier: 0.3
      loss_weight_fusion: 0.2
      fallback_on_error: true
      train_only_hvp: false
      aux_loss:
        enabled: false
        debug: false
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
        gt_guided:
          enabled: false
          debug: false
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
      residual_gate:
        enabled: true
        alpha_init: 0.05
        alpha_max: 0.3
        learnable: true
        collaboration_aware:
          enabled: false
          no_collab_scale: 0.0
          collab_scale: 1.0
          min_cav: 2
          use_record_len: true
          fallback_scale: 1.0
          debug: false
      packet:
        enabled: false
        mode: packet_one_round
        topk: 50
        packet_dim: 16
        descriptor_dim: 8
        quantize: fp16
        send_uncertainty: true
        send_agent_quality: true
        send_timestamp: true
        bandwidth_budget_kb: 8
        deadline_ms: 100
        detach_packet: false
        debug: false
      debug: false
```

Minimum enablement for server validation:

```yaml
model:
  core_method: heter_pyramid_collab_hvp_cbea
  args:
    hvp_cbea:
      enabled: true
      train_only_hvp: true
      fallback_on_error: true
```

With `enabled: false`, the new model follows the official HEAL forward path and skips all HVP-CBEA modules.

## v2.1 Train-only HVP Fine-tuning

```yaml
model:
  core_method: heter_pyramid_collab_hvp_cbea
  args:
    hvp_cbea:
      enabled: true
      train_only_hvp: true
      fallback_on_error: true
      residual_gate:
        enabled: true
        alpha_init: 0.05
        alpha_max: 0.3
        learnable: true
        collaboration_aware:
          enabled: false
          no_collab_scale: 0.0
          collab_scale: 1.0
          min_cav: 2
          use_record_len: true
          fallback_scale: 1.0
          debug: false
      packet:
        enabled: false
      debug: true
```

When `train_only_hvp: true`, official HEAL backbone parameters are frozen and only these HVP-CBEA modules remain trainable:

- `hvp_collaborator_proj`
- `hypothesis_encoder`
- `hypothesis_verifier`
- `bayesian_hypothesis_fusion`
- `hvp_packetizer`
- `hvp_packet_compressor`
- `hvp_packet_aggregator`
- `hvp_packet_comm_meter`

This is intended for full collaborative fine-tuning from a HEAL_m1_based final_infer checkpoint. It is not intended for the official `heter_pyramid_single` stage2/m2 yaml path.

## v2.4 Frozen HEAL BN Buffer Rule

When `enabled: true` and `train_only_hvp: true`, inherited HEAL BatchNorm and SyncBatchNorm modules are kept in eval mode during training. This freezes their `running_mean`, `running_var`, and `num_batches_tracked` buffers in addition to freezing inherited HEAL parameters with `requires_grad=False`.

HVP-CBEA module BatchNorm layers remain in train mode and can update normally. With `enabled: false`, the wrapper still follows the official HEAL behavior and skips HVP-CBEA logic.

## v2.5 Residual Gate / Identity-start

If `residual_gate` is omitted, these defaults are used:

```yaml
residual_gate:
  enabled: true
  alpha_init: 0.05
  alpha_max: 0.3
  learnable: true
```

With the gate enabled, HVP-CBEA applies its BEV feature update as:

```text
fused_feature_out = fused_feature + alpha * delta_feature
```

`alpha` is stored inside `bayesian_hypothesis_fusion` as a bounded residual parameter. The default initialization is close to identity, reducing the chance that untrained HVP-CBEA modules strongly disturb the inherited HEAL feature. `train_only_hvp: true` still trains only HVP-CBEA prefixes, and the v2.4 non-HVP BatchNorm buffer freeze remains active.

## v2.6 Collaboration-aware Residual Gate

If `collaboration_aware` is omitted, it defaults to disabled:

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

When enabled, the residual update becomes:

```text
effective_alpha = alpha * collaboration_scale
fused_feature_out = fused_feature + effective_alpha * delta_feature
```

`record_len <= 1` uses `no_collab_scale` and suppresses ego-only residual by default. `record_len >= min_cav` uses `collab_scale` and preserves the v2.5 residual strength by default. With `collaboration_aware.enabled: false`, `collaboration_scale=1.0`, so the v2.5 path is unchanged.

## v2.7-a Auxiliary Loss Plumbing

Auxiliary loss is disabled by default. If the `aux_loss` field is omitted, the effective default is:

```yaml
aux_loss:
  enabled: false
  debug: false
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
  gt_guided:
    enabled: false
    debug: false
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

When `aux_loss.enabled: false`, HVP-CBEA keeps the v2.6 behavior. When enabled, the standard detection loss can add optional HVP-CBEA auxiliary terms:

- `residual_reg`: regularizes `effective_alpha * delta_feature`.
- `alpha_reg`: regularizes the bounded residual `alpha` toward a target.
- `refinement_consistency`: currently a feature-delta L1 placeholder on `delta_feature`.

The current v2.7-a consistency term is not GT-guided; v2.7-b can replace it with detection-target-aware refinement supervision. No new trainable parameters or checkpoint keys are introduced.

## v2.7-b GT-guided Auxiliary Loss

`gt_guided` is also disabled by default. If the field is omitted, it is equivalent to:

```yaml
aux_loss:
  gt_guided:
    enabled: false
```

When `aux_loss.enabled: true` and `gt_guided.enabled: true`, HVP-CBEA can use anchor-positive supervision from `target_dict["pos_equal_one"]`:

- `hypothesis_heatmap`: guides `hypothesis_hmap` with anchor-positive BCE.
- `residual_focus`: regularizes `hvp_residual` more strongly in background than foreground.
- `residual_fg_boost`: currently a conservative foreground residual L1-to-target placeholder.

The positive map parser supports `[B,H,W,A]`, `[B,A,H,W]`, `[B,1,H,W]`, and compatible flat anchor layouts. The map is resized to the HVP tensor resolution with nearest-neighbor interpolation. With `gt_guided.enabled: false`, v2.7-a behavior is unchanged.

## v3.0 Packet Mode

Packet mode is disabled by default. If the `packet` field is omitted, the effective default is:

```yaml
packet:
  enabled: false
  mode: packet_one_round
  topk: 50
  packet_dim: 16
  descriptor_dim: 8
  quantize: fp16
  send_uncertainty: true
  send_agent_quality: true
  send_timestamp: true
  bandwidth_budget_kb: 8
  deadline_ms: 100
  detach_packet: false
  debug: false
```

When `packet.enabled: false`, HVP-CBEA uses the v2.5 feature path unchanged. When `packet.enabled: true`, collaborator features are only used inside the local packetizer to produce compact hypothesis/evidence packets; the ego-side packet aggregator receives packets, not dense collaborator BEV features.

## v3 Stage3 Feature-level CBEA Aggregator

Stage3 uses the feature-level HVP-CBEA path, not packet mode:

```yaml
model:
  core_method: heter_pyramid_collab_hvp_cbea
  args:
    hvp_cbea:
      enabled: true
      train_only_hvp: true
      packet:
        enabled: false
      aux_loss:
        enabled: true
        residual_reg:
          enabled: true
          weight: 0.001
        alpha_reg:
          enabled: true
          weight: 0.001
          target: 0.05
        gt_guided:
          enabled: true
          hypothesis_heatmap:
            enabled: true
            weight: 0.05
          residual_focus:
            enabled: true
            weight: 0.01
```

The Stage3 template is:

```text
opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage3/cbea_aggregator.yaml
```

Prepare the Stage3 initial checkpoint with:

```bash
python opencood/tools/prepare_hvp_cbea_stage3.py
```

The merged checkpoint intentionally lacks HVP-CBEA module keys; those modules are initialized by the Stage3 model and remain trainable.

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
      debug: false
```

Minimum enablement for server validation:

```yaml
model:
  core_method: heter_pyramid_collab_hvp_cbea
  args:
    hvp_cbea:
      enabled: true
      fallback_on_error: true
```

With `enabled: false`, the new model follows the official HEAL forward path and skips all HVP-CBEA modules.

# HEAL-XLab Config Schema

Do not add this template under `logs`. Copy the needed block into an evaluated yaml only after a disabled fallback check.

```yaml
xlab:
  enabled: false
  method: hbec
  debug: true
  debug_dir: xlab_debug
  hbec:
    enabled: false
    apply_stage: final_infer_postprocess
    evidence_source: none
    target_modalities: ["m2", "m4", "camera"]
    base_uncertainty: 1.0
    min_score_for_uncertainty: 0.05
    match:
      iou_threshold: 0.1
      center_dist_threshold: 2.0
      iou_weight: 0.7
      dist_weight: 0.3
      dist_scale: 2.0
    refine:
      enabled: true
      refine_strength: 0.5
      evidence_weight: 0.5
    novel:
      enabled: true
      novel_score_threshold: 0.6
      novel_dist_threshold: 2.0
      max_novel: 20
    suppress:
      enabled: false
      suppress_score_threshold: 0.3
      suppress_factor: 1.0
    safety:
      fallback_on_error: true
      max_boxes_after_fusion: 300
      require_no_gt_for_fusion: true
```

Both `xlab.enabled` and `xlab.hbec.enabled` default to `false`. HBEC changes inference output only when both switches are explicitly true and reliable collaborator evidence is available.

## HEAL-XLab-v1.1 Evidence Sources

Default remains `none`, so enabling HBEC without explicitly selecting an evidence source still falls back without changing official outputs.

```yaml
xlab:
  enabled: false
  method: hbec
  debug: true
  hbec:
    enabled: false
    evidence_source: none
    # Optional:
    # evidence_source: late_fusion_reinfer
    # evidence_source: no_fusion_reinfer
    # evidence_source: explicit
```

- `none`: do not extract evidence; safe fallback.
- `late_fusion_reinfer`: call official `opencood.tools.inference_utils.inference_late_fusion()` and use returned object predictions as evidence.
- `no_fusion_reinfer`: call official `opencood.tools.inference_utils.inference_no_fusion()` only when the active dataset supports `post_process_no_fusion`.
- `explicit`: use `infer_context["collaborator_evidence"]` or `infer_context["evidence_packet"]`.

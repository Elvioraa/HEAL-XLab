# HEAL-XLab Change Record

## HEAL-XLab-v1-HBEC

Base hash: `96812ed`

Commit status: implemented in working tree before evaluation.

### Modified Files

- `opencood/tools/inference_heter_in_order.py`
  - Added `apply_xlab_postprocess_hook` import.
  - Inserted the XLab post-process hook inside `main()` after official `infer_result` unpacking and before AP statistics.

### Added Files

- `opencood/xlab/__init__.py`: package marker and disabled-by-default XLab namespace.
- `opencood/xlab/config.py`: safe yaml defaults and method enable checks.
- `opencood/xlab/hooks.py`: official post-process hook entry point with fallback-on-error.
- `opencood/xlab/metrics.py`: per-frame jsonl debug recorder.
- `opencood/xlab/utils.py`: tensor safety, dtype/device, yaw, empty tensor, and payload helpers.
- `opencood/xlab/hbec/__init__.py`: HBEC package marker.
- `opencood/xlab/hbec/packet.py`: `HypothesisPacket`, `EvidencePacket`, and score-derived uncertainty.
- `opencood/xlab/hbec/matcher.py`: BEV IoU / center-distance matcher and greedy matching.
- `opencood/xlab/hbec/refiner.py`: Bayesian refinement, novel insertion, and optional suppression.
- `opencood/xlab/hbec/engine.py`: HBEC final-infer post-process engine.
- `docs/XLAB_OVERVIEW.md`: XLab framework overview.
- `docs/EXPERIMENT_INDEX.md`: experiment registry and baseline.
- `docs/CHANGE_RECORD_XLAB.md`: this change record.
- `docs/HEAL_XLAB_CODE_PATH_AUDIT.md`: real-code path audit.
- `docs/HEAL_XLAB_CONFIG_SCHEMA.md`: yaml schema template, not placed under logs.
- `docs/HEAL_XLAB_V1_HBEC.md`: HBEC method note.

### Hook Location

Official file: `opencood/tools/inference_heter_in_order.py`

Function: `main()`

Inserted after:

- `pred_box_tensor = infer_result['pred_box_tensor']`
- `gt_box_tensor = infer_result['gt_box_tensor']`
- `pred_score = infer_result['pred_score']`

Inserted before:

- `eval_utils.caluclate_tp_fp(..., 0.3)`
- `eval_utils.caluclate_tp_fp(..., 0.5)`
- `eval_utils.caluclate_tp_fp(..., 0.7)`

### Disabled Fallback Mechanism

- If yaml has no `xlab` field, `safe_get_xlab_cfg()` returns `enabled=false`.
- If `xlab.enabled=false`, the hook returns official outputs unchanged and writes no debug record.
- If `xlab.enabled=true` but `xlab.hbec.enabled=false`, the hook returns official outputs unchanged.
- If HBEC is enabled but required inputs or collaborator evidence are unavailable, the engine returns official outputs unchanged and records a fallback reason.
- If HBEC fails and `fallback_on_error=true`, the hook catches the exception and returns official outputs unchanged.

### Ground Truth and Camera Feature Rules

- `gt_box_tensor` is passed through unchanged for evaluation only.
- `gt_box_tensor` is never read by HBEC for fusion.
- Raw camera features are not accessed by XLab.
- Model training and model forward definitions are untouched.

### How To Open Future Experiments

Start from the schema in `docs/HEAL_XLAB_CONFIG_SCHEMA.md` and explicitly set:

```yaml
xlab:
  enabled: true
  hbec:
    enabled: true
```

Then open one experimental mechanism at a time under `match`, `refine`, `novel`, `suppress`, and `evidence_source`.

### Known Risks

- If the official flow cannot provide collaborator-only prediction, HBEC falls back with `fallback_reason = no_collaborator_evidence`.
- The first version may need a follow-up evidence extraction path before AP can change.
- AP improvement must be validated by final_infer experiments.


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

## HEAL-XLab-v1.1-HBEC Evidence Extraction

Base hash: `52eb4ce`

### Modified Files

- `opencood/xlab/config.py`
  - Changed default `hbec.evidence_source` to `none`.
- `opencood/xlab/hbec/engine.py`
  - Integrated `HBECEvidenceExtractor`.
  - Added evidence debug metrics.
  - Falls back safely when extraction fails.
- `opencood/xlab/metrics.py`
  - Added evidence-source and extraction-status fields.
- `opencood/tools/inference_heter_in_order.py`
  - Added `dataset` to `infer_context` so official re-inference can call the active dataset post-processor.

### Added Files

- `opencood/xlab/hbec/evidence.py`
  - Implements object-level evidence extraction from explicit context or official re-inference functions.

### Official Evidence Functions

- `opencood.tools.inference_utils.inference_late_fusion`
- `opencood.tools.inference_utils.inference_no_fusion`

Both return `pred_box_tensor`, `pred_score`, and `gt_box_tensor`. HBEC uses only `pred_box_tensor` and `pred_score` for evidence.

### Safety and Fallback

- Default `evidence_source=none`; enabled HBEC still falls back unless evidence is explicitly selected.
- `late_fusion_reinfer` and `no_fusion_reinfer` are object-level evidence sources after official post-process, not raw camera feature bypass.
- Extraction runs under `torch.no_grad()`.
- The extractor preserves the model training/eval state.
- `gt_box_tensor` is ignored for fusion.
- `batch_data` and `hypes` are not intentionally modified by XLab.
- If extraction fails or returns empty evidence, HBEC returns official outputs unchanged and records `fallback_reason`.
- `enabled=false` still returns official outputs from the hook before any extraction is attempted.

### Follow-up Experiment Order

- disabled equivalence check.
- enabled + `evidence_source=late_fusion_reinfer`.
- enabled refine only.
- enabled refine + novel.
- enabled refine + novel + suppress.

## HEAL-XLab-v1.2 no-fusion evidence extraction fix

Base hash: `27f0e4f`

### Server Finding

- Experiment: `HEAL-XLab-v1.1 step02_hbec_no_refine_only`.
- Requested config used `xlab.enabled=true`, `hbec.enabled=true`, `evidence_source=no_fusion_reinfer`, `refine=true`, `novel=false`, `suppress=false`.
- Result for use_cav2 was `0.8370 / 0.8181 / 0.7247`.
- Debug showed all 2170 frames fell back:
  - `total_evidence_boxes = 0`
  - `total_matched_boxes = 0`
  - `total_refined_boxes = 0`
  - `fallback_reason = dataset_missing_post_process_no_fusion`

### Root Cause

- Official `opencood.tools.inference_utils.inference_no_fusion()` exists, but it calls `dataset.post_process_no_fusion(...)`.
- The active HEAL final-infer dataset is `Intermediate_heter_Infer_Fusion_Dataset`, which inherits `post_process()` from `intermediate_heter_fusion_dataset.py` but does not implement `post_process_no_fusion()`.
- Therefore v1.1 correctly refused to fabricate evidence, but `no_fusion_reinfer` had no usable evidence path.

### Fix

- `opencood/xlab/hbec/evidence.py`
  - Keeps the official `inference_no_fusion()` path when the dataset supports `post_process_no_fusion`.
  - When `evidence_source=no_fusion_reinfer` and the active dataset lacks `post_process_no_fusion`, falls back to a new object-level `single_agent_reinfer` extractor.
  - `single_agent_reinfer` builds an ego-only view of the current batch without mutating `batch_data`, runs the model under `torch.no_grad()`, and uses official `dataset.post_process()` to produce `pred_box_tensor` and `pred_score`.
  - Adds explicit debug fields for requested source, used source, extraction function, dataset class, raw output keys, and dataset capability flags.
- `opencood/xlab/hbec/engine.py`
  - Records the enhanced evidence debug fields in `hbec_debug.jsonl`.
- `opencood/xlab/metrics.py`
  - Adds defaults for the new evidence debug fields.

### Safety

- `gt_box_tensor` is still ignored for HBEC fusion.
- No raw camera feature bypass is introduced; `single_agent_reinfer` uses model object-level predictions and official post-process.
- Training code and training flow are unchanged.
- `enabled=false` returns official outputs before extraction is attempted.
- `evidence_source=none` still falls back and returns official outputs unchanged.
- If both official no-fusion and `single_agent_reinfer` fail, HBEC falls back and records a concrete `fallback_reason`.

## HEAL-XLab-v1.1-logfmt

Change:
- AP metric print precision changed from 2 decimals to 4 decimals.

Reason:
- Two-decimal AP output hides important differences, especially AP@0.7 changes around 0.01.
- Future HEAL-XLab final_infer logs should preserve values like 0.8045 instead of 0.80.

Scope:
- Only AP print formatting was changed.
- AP calculation logic was not changed.
- Inference/model/HBEC logic was not changed.

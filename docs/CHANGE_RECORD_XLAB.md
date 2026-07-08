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

## HEAL-XLab-v2-HVP-CBEA-Trainable

Base hash: `fb533c2`

Method: Hypothesis Verification Protocol + Cooperative Bayesian Evidence Accumulation.

### Difference From v1 HBEC

- v1 HBEC is a final-infer/post-process hook under `opencood/xlab/hbec`.
- v2 HVP-CBEA is a trainable model-forward-level direction inserted before the official detection heads.
- v1 code is preserved and not modified by this change.

### Added Files

- `opencood/models/sub_modules/hypothesis_encoder.py`
  - Generates sparse hypotheses from BEV feature maps.
- `opencood/models/sub_modules/hypothesis_verifier.py`
  - Produces CONFIRM / REFUTE / REFINE logits, refine deltas, and novel hypotheses from collaborator BEV features.
- `opencood/models/sub_modules/bayesian_hypothesis_fusion.py`
  - Performs Bayesian-style log-odds update and scatters hypothesis evidence back into BEV features.
- `opencood/models/heter_pyramid_collab_hvp_cbea.py`
  - New model core method using plan B. Official `heter_pyramid_collab.py` is not modified.
  - Adds a lightweight collaborator BEV projection when per-agent BEV channels differ from the official head feature channels.
- `opencood/tools/check_hvp_cbea_smoke.py`
  - Lightweight tensor-only smoke test.
- `docs/HVP_CBEA_CODE_PATH_AUDIT.md`
- `docs/HVP_CBEA_CONFIG_SCHEMA.md`
- `docs/HVP_CBEA_V2_PLAN.md`

### Modified Files

- `opencood/models/__init__.py`
  - Adds a small `build_model()` compatibility wrapper for import checks.
- `opencood/loss/point_pillar_loss.py`
  - Adds `output_dict['hvp_cbea_loss']` only when present.
- `opencood/loss/point_pillar_depth_loss.py`
  - Logs `hvp_cbea_loss` only when the base loss recorded it.
- `docs/CHANGE_RECORD_XLAB.md`
  - Records this change.

### Enable Path

Real yaml path:

```yaml
model:
  core_method: heter_pyramid_collab_hvp_cbea
  args:
    hvp_cbea:
      enabled: true
      fallback_on_error: true
```

Default is `enabled=false`.

### Fallback Mechanism

- If `model.core_method` remains `heter_pyramid_collab`, official HEAL is unchanged.
- If `model.core_method=heter_pyramid_collab_hvp_cbea` but `model.args.hvp_cbea.enabled=false`, the v2 branch bypasses HVP-CBEA logic and follows the official forward structure.
- If HVP-CBEA is enabled but no hypotheses, no compatible collaborator BEV feature, shape mismatch, or an exception occurs with `fallback_on_error=true`, the original `fused_feature` is returned to the official heads.

### Loss Integration

- Training loss integration is implemented as optional.
- The model may attach `output_dict['hvp_cbea_loss']`.
- `PointPillarLoss` adds it to total loss only when present.
- If the field is absent, official loss is unchanged.
- Current auxiliary module `compute_loss()` methods are defensive and return zero when GT format is unavailable.
- Further stage2 training may be needed to make the auxiliary losses non-zero and task-specific.

### Safety

- Ground truth is not used in inference or feature fusion.
- Raw camera features are not bypassed; modules operate on official BEV tensors after official encoders/backbones/aligners.
- No logs config is added.
- Official `heter_pyramid_collab.py` is not modified.

## HEAL-XLab-v2.1 train_only_hvp

Base hash: `18923bb`

### Motivation

- Server inspection confirmed `HeterPyramidCollabHvpCbea` is collaborative and depends on `heter_feature_2d`, `record_len`, and collaborator BEV features.
- It is not suitable for the official `heter_pyramid_single` stage2/m2 yaml path.
- v2.1 adds a safe fine-tuning mode for full collaborative configs: freeze official HEAL and train only HVP-CBEA modules.

### Modified Files

- `opencood/models/heter_pyramid_collab_hvp_cbea.py`
  - Adds `model.args.hvp_cbea.train_only_hvp`.
  - Adds `_freeze_non_hvp_parameters()`.
  - Adds `_summarize_trainable_parameters()`.
  - Adds train-only summary fields to `hvp_cbea_debug`.
- `docs/HVP_CBEA_CONFIG_SCHEMA.md`
- `docs/HVP_CBEA_V2_PLAN.md`
- `docs/CHANGE_RECORD_XLAB.md`

### YAML

```yaml
model:
  core_method: heter_pyramid_collab_hvp_cbea
  args:
    hvp_cbea:
      enabled: true
      train_only_hvp: true
      fallback_on_error: true
      debug: true
```

### Freeze Rule

When both `enabled=true` and `train_only_hvp=true`, only parameters whose names start with the following prefixes keep `requires_grad=True`:

- `hvp_collaborator_proj`
- `hypothesis_encoder`
- `hypothesis_verifier`
- `bayesian_hypothesis_fusion`

All inherited official HEAL parameters are frozen.

### Safety

- If `enabled=false`, HVP-CBEA logic is bypassed and no HVP fields are added to `output_dict`.
- If `train_only_hvp=false`, current default training behavior is preserved.
- Official `heter_pyramid_collab.py` is not modified.
- v1 HBEC is not modified.
- No logs config is added.

## HEAL-XLab-v2.2 supervise_single compatibility fix

Base hash: `e3ac402`

### Server Finding

- Fine-tune smoke with full collaborative `train_only_hvp` reached the first batch and reported:
  - `Loss: 21.1791`
  - `HVP-CBEA Loss: 0.0000`
- Training then entered `train.py` single-supervision branch:
  - `criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")`
- The loss expected `output_dict['cls_preds_single']` and failed with:
  - `KeyError: 'cls_preds_single'`

### Root Cause

- `HeterPyramidCollabHvpCbea.forward()` did not emit the train-required single prediction fields when `model.args.supervise_single=true`.
- The wrapper preserved collaborative outputs and `occ_single_list`, but missed:
  - `cls_preds_single`
  - `reg_preds_single`
  - `dir_preds_single`

### Fix

- `opencood/models/heter_pyramid_collab_hvp_cbea.py`
  - Reads `self.supervise_single = bool(args.get("supervise_single", False))`.
  - When true, computes single predictions through the official per-agent route:
    - `pyramid_backbone.forward_single(heter_feature_2d)`
    - optional `shrink_conv`
    - shared `cls_head`, `reg_head`, `dir_head`
  - Adds `cls_preds_single`, `reg_preds_single`, and `dir_preds_single` to `output_dict`.
  - HVP-CBEA feature injection remains limited to final collaborative `fused_feature`.

### Safety

- Official `heter_pyramid_collab.py` is not modified.
- `supervise_single` is not disabled or bypassed.
- v1 HBEC is not modified.
- `train_only_hvp` freeze prefixes and behavior are unchanged.
- No logs config is added.

## HEAL-XLab-v2.3 autograd inplace fix

Base hash: `45630d4`

### Server Finding

- v2.2 fine-tune smoke fixed `cls_preds_single`, reached loss computation, and failed at `final_loss.backward()`.
- Error:
  - `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`
- The failing tensor shape `[1, 50]` was consistent with hypothesis-level tensors used inside HVP-CBEA.

### Root Cause

- HVP-CBEA submodules contained autograd-risky operations:
  - `nn.ReLU(inplace=True)` in `HypothesisEncoder`, `HypothesisVerifier`, and `BayesianHypothesisFusion`.
  - Slice writes into `novel_hyps` in `HypothesisVerifier`.
  - Slice writes into `updated` hypotheses in `BayesianHypothesisFusion`.
  - Indexed assignment into the BEV scatter map in `BayesianHypothesisFusion._scatter()`.

### Fix

- `opencood/models/sub_modules/hypothesis_encoder.py`
  - Changes HVP activation to `nn.ReLU(inplace=False)`.
- `opencood/models/sub_modules/hypothesis_verifier.py`
  - Changes HVP activation to `nn.ReLU(inplace=False)`.
  - Rebuilds novel hypotheses with `torch.cat()` instead of slice assignment.
- `opencood/models/sub_modules/bayesian_hypothesis_fusion.py`
  - Changes HVP activation to `nn.ReLU(inplace=False)`.
  - Rebuilds updated hypotheses with `torch.cat()` instead of slice assignment.
  - Replaces indexed BEV map assignment with out-of-place per-batch `scatter_add()` followed by `torch.stack()`.
- `opencood/tools/check_hvp_cbea_smoke.py`
  - Adds a backward smoke test and checks that HVP-CBEA module parameters receive gradients.

### Safety

- HVP-CBEA outputs are not detached; detection loss can still backpropagate into `hypothesis_encoder`, `hypothesis_verifier`, `bayesian_hypothesis_fusion`, and collaborator projection modules.
- Official `heter_pyramid_collab.py` is not modified.
- `enabled=false` behavior is unchanged.
- `train_only_hvp` freeze prefixes and behavior are unchanged.
- v1 HBEC is not modified.
- No logs config is added.

## HEAL-XLab-v2.4 frozen HEAL BN buffer fix

Base hash: `e02731c`

### Server Finding

- A checkpoint diff on the HVP-CBEA v2.3 fine-tune smoke showed:
  - `hvp changed keys: 0`
  - `non-hvp changed keys: 510`
  - `keys missing in epoch1: 36`
  - `keys missing in epoch2: 0`
- The changed non-HVP keys were inherited HEAL BatchNorm buffers, including `running_mean`, `running_var`, and `num_batches_tracked` under `backbone_m1`, `backbone_m2`, and `backbone_m3`.
- The missing epoch1 HVP keys are expected because the starting checkpoint did not contain HVP-CBEA parameters.

### Root Cause

- v2.3 `train_only_hvp` froze parameters with `requires_grad=False`.
- Frozen inherited HEAL modules still stayed in train mode, so BatchNorm running statistics continued to update.
- This polluted the HEAL_m1_based backbone during HVP-only fine-tuning.

### Fix

- `opencood/models/heter_pyramid_collab_hvp_cbea.py`
  - Adds `_set_frozen_heal_modules_eval()`.
  - Overrides `train(self, mode=True)` so each training-mode switch keeps non-HVP BatchNorm and SyncBatchNorm modules in eval mode.
  - Keeps HVP-CBEA modules in train mode after `super().train(True)`.
  - Adds debug summary fields: `frozen_bn_eval_count`, `hvp_bn_train_count`, and `train_only_hvp`.

### Safety

- Official `heter_pyramid_collab.py` is not modified.
- `enabled=false` behavior is unchanged.
- `train_only_hvp=true` now freezes both inherited HEAL parameters and inherited HEAL BN buffers.
- HVP-CBEA internal BatchNorm modules remain trainable and keep updating during training.
- v1 HBEC is not modified.
- No logs config is added.

## HEAL-XLab-v2.5 residual gate / identity-start

Base hash: `180f553`

### Server Finding

- v2.4 standard final_infer best result came from `v24_lr2em4_ep6_epoch4_standard`.
- Epoch4 improved AP@0.3 and AP@0.5 clearly across all CAV settings and kept AP@0.7 positive:
  - `use_cav1: 0.8122 / 0.7909 / 0.6882`
  - `use_cav2: 0.8621 / 0.8355 / 0.7298`
  - `use_cav3: 0.9092 / 0.8911 / 0.8078`
  - `use_cav4: 0.9100 / 0.8920 / 0.8082`
- The high-IoU AP@0.7 gain is still below the +3pp target, suggesting HVP-CBEA helps discovery and medium-quality boxes more than high-precision localization.

### Goal

- Make the HVP-CBEA update to `fused_feature` a controlled residual instead of a strong direct replacement.
- Keep HVP-CBEA trainable from detection loss without detaching outputs.
- Preserve `enabled=false`, `train_only_hvp`, supervise-single compatibility, and the v2.4 non-HVP BN freeze.

### Fix

- `opencood/models/sub_modules/bayesian_hypothesis_fusion.py`
  - Converts the previous direct fusion candidate into `delta_feature`.
  - Applies `fused_feature_out = fused_feature + alpha * delta_feature`.
  - Adds bounded residual alpha as `alpha_max * sigmoid(residual_alpha_logit)`.
  - Defaults to `alpha_init=0.05`, `alpha_max=0.3`, and `learnable=true`.
- `opencood/models/heter_pyramid_collab_hvp_cbea.py`
  - Adds and normalizes `model.args.hvp_cbea.residual_gate`.
  - Passes the residual gate config into `BayesianHypothesisFusion`.
  - Adds debug fields for residual alpha and delta feature statistics while preserving train-only and BN summaries.
- `opencood/tools/check_hvp_cbea_smoke.py`
  - Checks residual alpha existence, bounds, and gradient after backward.
  - Prints `HVP-CBEA residual gate OK`.

### Safety

- Official `heter_pyramid_collab.py` is not modified.
- `enabled=false` behavior is unchanged.
- `train_only_hvp=true` still trains only HVP-CBEA prefixes; the new alpha lives under `bayesian_hypothesis_fusion`.
- Non-HVP HEAL parameters and BatchNorm buffers remain frozen in train-only mode.
- HVP-CBEA internal BatchNorm modules remain trainable and update normally.
- v1 HBEC is not modified.
- No logs config is added.

### Follow-up

- Run an `enabled=true` untrained sanity final_infer to verify identity-start behavior.
- Run a short `train_only_hvp` fine-tune to test whether the safer residual update improves AP@0.7.

## HEAL-XLab-v2.6 collaboration-aware residual gate

Base hash: `f246f2d`

### Motivation

- v2.5 residual-gate long training showed useful multi-CAV gains, especially in `use_cav2`, `use_cav3`, and `use_cav4` AP@0.7.
- `use_cav1` regressed, suggesting residual HVP-CBEA updates can perturb ego-only features when no collaborator evidence exists.
- v2.6 targets an inference-only ablation path: use a v2.5 checkpoint, enable collaboration-aware scaling in config, and suppress residuals only for ego-only scenes.

### Fix

- `opencood/models/sub_modules/bayesian_hypothesis_fusion.py`
  - Adds default-disabled `residual_gate.collaboration_aware`.
  - Computes `effective_alpha = alpha * collaboration_scale`.
  - Supports scalar and batch-wise collaboration scales without adding checkpoint parameters.
  - Adds residual debug fields for record length, collaborator presence, collaboration scale, effective alpha, and fallback reason.
- `opencood/models/heter_pyramid_collab_hvp_cbea.py`
  - Reads `record_len` from the existing forward path.
  - Computes collaboration scale before feature or packet residual application.
  - Passes scale into the existing v2.5 residual gate.
- `opencood/tools/check_hvp_cbea_smoke.py`
  - Adds smoke coverage for `record_len=[1]` suppression, `record_len=[2]` active residual, and disabled-mode v2.5 equivalence.

### Config

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

### Safety

- Default `collaboration_aware.enabled=false` preserves v2.5 behavior.
- No new trainable parameter or checkpoint key is added.
- `enabled=false` still follows the official HEAL path.
- Official `heter_pyramid_collab.py` is not modified.
- v1 HBEC is not modified.
- No logs config is added.

## HEAL-XLab-v2.7-a auxiliary loss plumbing

Base hash: `7bc0127`

### Motivation

- Current training logs can show `HVP-CBEA Loss: 0.0000`.
- The existing `_compute_hvp_loss()` path only returns `tensor.sum() * 0.0` terms. This keeps HVP-CBEA tensors attached to the graph safely, but it does not provide an auxiliary optimization signal.
- v2.7-a adds the plumbing for configurable auxiliary HVP-CBEA losses while keeping the default behavior off.

### Added Files

- `opencood/loss/hvp_cbea_aux_loss.py`
  - Adds default and normalized `aux_loss` config helpers.
  - Adds `compute_hvp_auxiliary_loss()`.
  - Supports residual regularization, alpha regularization, and a placeholder feature-delta consistency term.

### Modified Files

- `opencood/models/sub_modules/bayesian_hypothesis_fusion.py`
  - Records `delta_feature`, `hvp_residual`, `alpha`, and `effective_alpha` after residual fusion.
- `opencood/models/heter_pyramid_collab_hvp_cbea.py`
  - Adds default-off `model.args.hvp_cbea.aux_loss`.
  - Emits `output_dict['hvp_cbea_aux']` only when `aux_loss.enabled=true`.
- `opencood/loss/point_pillar_loss.py`
  - Adds enabled auxiliary HVP-CBEA terms into the standard detection loss.
  - Reports `hvp_residual_reg_loss`, `hvp_alpha_reg_loss`, `hvp_refinement_consistency_loss`, and `hvp_aux_total_loss`.
- `opencood/tools/check_hvp_cbea_smoke.py`
  - Adds an auxiliary-loss CPU smoke with backward gradient checks.

### Config

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
```

### Safety

- `aux_loss.enabled=false` preserves the v2.6 path.
- `enabled=false` still follows the official HEAL path.
- No new trainable parameters or checkpoint keys are added.
- v2.7-a `refinement_consistency` is a placeholder L1 term on `delta_feature`; v2.7-b should replace it with GT-guided refinement supervision.
- Official `heter_pyramid_collab.py` is not modified.
- v1 HBEC is not modified.
- No logs config is added.

## HEAL-XLab-v2.7-b GT-guided auxiliary loss

Base hash: `406cfef`

### Motivation

- v2.7-a made `HVP-CBEA Loss` non-zero through safe residual, alpha, and feature-delta regularization.
- Those terms are useful plumbing and safety regularizers, but they do not explicitly guide the hypothesis heatmap or where residual updates should occur.
- v2.7-b adds optional GT/anchor-guided HVP auxiliary supervision to focus residual changes on target regions and reduce background perturbation.

### Implementation

- `opencood/loss/hvp_cbea_aux_loss.py`
  - Adds default-off `aux_loss.gt_guided`.
  - Parses `target_dict["pos_equal_one"]` as the anchor-positive map.
  - Supports `[B,H,W,A]`, `[B,A,H,W]`, `[B,1,H,W]`, and compatible flat anchor layouts.
  - Adds `hvp_gt_hypothesis_heatmap_loss`, `hvp_gt_residual_focus_loss`, and `hvp_gt_residual_fg_boost_loss`.
- `opencood/models/heter_pyramid_collab_hvp_cbea.py`
  - Adds `hypothesis_hmap` / `hmap` and `hypothesis_reg` / `reg` into `output_dict['hvp_cbea_aux']` when auxiliary loss is enabled.
- `opencood/loss/point_pillar_loss.py`
  - Passes the existing training `target_dict` into `compute_hvp_auxiliary_loss()`.
- `opencood/tools/check_hvp_cbea_smoke.py`
  - Adds GT-guided auxiliary loss smoke coverage and backward checks.

### Config

```yaml
aux_loss:
  enabled: false
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

### Safety

- `aux_loss.enabled=false` preserves the old path.
- `gt_guided.enabled=false` preserves v2.7-a behavior.
- `enabled=false` still follows the official HEAL path.
- No new trainable parameters or checkpoint keys are added.
- The foreground boost term remains a conservative L1-to-target placeholder; stronger hypothesis-level box refinement is left for future work.
- Official `heter_pyramid_collab.py` is not modified.
- v1 HBEC is not modified.
- No logs config is added.

## HEAL-XLab-v3.0 packet mode

Base hash: `25509b7`

### Goal

- Add an optional HVP-CBEA packet communication path for realistic cross-vendor collaboration.
- Deployment assumption: no raw sensor sharing, no dense intermediate BEV feature sharing, and no vendor-private model sharing.
- Collaborators send standardized hypothesis/evidence packets; ego aggregates packet evidence and applies residual refinement.

### Added Files

- `opencood/models/sub_modules/hvp_cbea_packet.py`
  - Adds `HypothesisEvidencePacketizer`.
  - Adds `PacketCompressor`.
  - Adds `PacketAggregator`.
  - Adds `PacketCommunicationMeter`.
- `opencood/tools/check_hvp_cbea_packet_smoke.py`
  - Covers packetizer shape, fp16 communication stats, aggregation shape/finite checks, backward gradients, and empty-packet behavior.
- `docs/HVP_CBEA_PACKET_V3_PLAN.md`
  - Documents packet-mode goals, config, communication boundary, modules, safety, and validation.

### Modified Files

- `opencood/models/heter_pyramid_collab_hvp_cbea.py`
  - Adds `model.args.hvp_cbea.packet` config with default `enabled=false`.
  - Instantiates packet modules only when `packet.enabled=true`.
  - Adds packet path before the v2.5 feature HVP path.
  - Falls back to the v2.5 feature HVP path when packet inputs are unavailable and `fallback_on_error=true`.
  - Extends train-only HVP prefixes to include packet modules.
- `opencood/models/sub_modules/bayesian_hypothesis_fusion.py`
  - Exposes `apply_residual_delta()` so packet mode reuses the v2.5 residual gate.
- `docs/HVP_CBEA_CONFIG_SCHEMA.md`
- `docs/HVP_CBEA_V2_PLAN.md`
- `docs/CHANGE_RECORD_XLAB.md`

### Config

Default packet config:

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

### Safety

- `enabled=false` still follows the official HEAL path.
- `packet.enabled=false` keeps the v2.5 HVP-CBEA feature path unchanged and does not instantiate packet modules.
- Official `heter_pyramid_collab.py` is not modified.
- v1 HBEC is not modified.
- No logs config is added.
- `train_only_hvp=true` still freezes inherited HEAL parameters and non-HVP BN buffers; packet modules are HVP-trainable when enabled.
- Packet aggregator receives packet tensors, not dense collaborator BEV features.

### Follow-up

- Run `packet.enabled=true` untrained sanity final_infer.
- Run short `train_only_hvp` packet fine-tune with packet debug enabled.
- Replace pseudo top-K feature boxes with real detector box decoding at the packetizer boundary.

## HEAL-XLab-v3.0 HVP-HEAL feature-main Stage1 skeleton

Base hash: `d0faa22`

### Goal

- Start the HVP-HEAL Feature Main staged-training direction.
- Add Stage1 Hypothesis-aware Base Training for m1 base features.
- Keep this separate from the v2.x HVP-CBEA plug-in workflow and from the packet-mode experiment.

### Added Files

- `opencood/models/hvp_heal_v3/hypothesis_head.py`
  - Adds a lightweight Conv-BN-ReLU hypothesis head.
  - Outputs `hypothesis_heatmap_logits`, `hypothesis_heatmap`, and optional `hypothesis_feature`.
- `opencood/models/heter_pyramid_collab_hvp_heal_v3.py`
  - Adds a default-inert HEAL collab wrapper for v3 staged training.
  - Calls official HEAL collab forward when `hvp_v3.enabled=false`.
  - In Stage1, attaches the hypothesis head to the m1 ego pre-fusion BEV feature.
- `opencood/tools/check_hvp_heal_v3_stage1_smoke.py`
  - Covers disabled forward, enabled forward, Stage1 loss, backward, and state-dict key separation.
- `opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage1/m1_hyp_base.yaml`
  - Adds the Stage1 config template.
- `docs/HVP_HEAL_V3_PLAN.md`
- `docs/HVP_HEAL_V3_CONFIG_SCHEMA.md`

### Modified Files

- `opencood/loss/hvp_cbea_aux_loss.py`
  - Adds `compute_hvp_v3_stage1_loss()`.
  - Reuses anchor-positive target parsing from v2.7-b.
- `opencood/loss/point_pillar_loss.py`
  - Adds HVP-v3 Stage1 loss only when `output_dict["hvp_v3"]` exists.
- `opencood/loss/point_pillar_depth_loss.py`
- `opencood/loss/point_pillar_pyramid_loss.py`
  - Logs `HVP-v3 Loss` and `Stage1 Hypothesis Loss` only when v3 loss is active.

### Config

```yaml
hvp_v3:
  enabled: false
  stage: none
```

Stage1 template enables:

```yaml
hvp_v3:
  enabled: true
  stage: stage1_hypothesis
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

### Safety

- Default `hvp_v3.enabled=false` emits no `hvp_v3` output and does not change the official HEAL path.
- Stage1 only trains a hypothesis heatmap auxiliary path; residual focus, packet, hybrid, Stage2, and Stage3 are not implemented.
- v2.x HVP-CBEA modules remain intact.
- Official `heter_pyramid_collab.py` is not modified.
- v1 HBEC is not modified.
- No logs directory is modified.

## HEAL-XLab-v3.0 Stage1 smoke training mode fix

Base hash: `74fa8d2`

### Motivation

- The first server Stage1 1-epoch smoke run hit CUDA OOM during the HEAL base forward near `shrink_conv`.
- The trainable-module summary did not list the newly added hypothesis head because the official HEAL constructor printed trainability before the v3 head was registered.
- Smoke training only needs to verify dataloader/train/loss/checkpoint wiring and hypothesis-head gradients, not full joint HEAL training.

### Fix

- `opencood/models/heter_pyramid_collab_hvp_heal_v3.py`
  - Adds `hvp_v3.stage1.train_mode`.
  - Adds `freeze_base_model` and `detach_bev_for_hypothesis`.
  - Registers `self.hvp_v3_hypothesis_head` as the only trainable module in `hypothesis_head_only` mode.
  - Runs the inherited HEAL base path under `torch.no_grad()` when the base is frozen.
  - Runs the hypothesis head outside the no-grad base path so Stage1 loss can backpropagate into the head.
  - Prints a second trainable-module summary after applying the v3 train mode, so `hvp_v3_hypothesis_head` appears in logs.
- `opencood/tools/check_hvp_heal_v3_stage1_smoke.py`
  - Checks base parameters are frozen in head-only mode.
  - Checks hypothesis-head parameters are trainable.
  - Checks backward produces non-zero hypothesis-head gradients and no base gradients.
- `opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage1/m1_hyp_base.yaml`
  - Enables the low-memory Stage1 smoke mode.

### Config

```yaml
hvp_v3:
  enabled: true
  stage: stage1_hypothesis
  stage1:
    train_mode: hypothesis_head_only
    freeze_base_model: true
    detach_bev_for_hypothesis: true
```

Future full joint Stage1 remains explicit:

```yaml
hvp_v3:
  stage1:
    train_mode: joint
    freeze_base_model: false
    detach_bev_for_hypothesis: false
```

### Safety

- The default v3 config is still disabled.
- `hvp_v3.enabled=false` still emits no `hvp_v3` output and follows the old path.
- Stage2, Stage3, packet, and hybrid modes are not implemented.
- v2.x HVP-CBEA modules remain intact.
- Official `heter_pyramid_collab.py` is not modified.
- v1 HBEC is not modified.
- No logs directory is modified.

## HEAL-XLab-v3.0 Stage2 evidence-aware modality adaptation

Base hash: `eac2db4`

### Motivation

- A server run of `m2_alignto_m1` accidentally used ordinary HEAL Stage2.
- Its logs showed only Conf/Loc/Dir/Depth/Pyramid losses and no evidence loss, so it is not an official HVP-v3 Stage2 experiment.
- Stage2 now needs m2/m3/m4 to train normal detection outputs plus evidence heatmap, uncertainty, and descriptor outputs.

### Implementation

- `opencood/models/hvp_heal_v3/evidence_head.py`
  - Adds `HvpHealV3EvidenceHead`.
  - Outputs `evidence_heatmap_logits`, `evidence_heatmap`, `evidence_uncertainty`, and `evidence_descriptor`.
- `opencood/models/heter_pyramid_single_hvp_heal_v3.py`
  - Adds a default-inert Stage2 single-modality wrapper.
  - Calls the official HEAL single forward unchanged unless `hvp_v3.enabled=true`, `stage=stage2_evidence`, and `use_hvp_v3_evidence=true`.
  - Attaches `hvp_v3_evidence_head` to the shrink-stage BEV feature.
  - Filters incompatible checkpoint keys so Stage1 keys such as `hvp_v3_hypothesis_head`, `encoder_m1`, and `backbone_m1` do not crash Stage2 loading.
- `opencood/loss/hvp_cbea_aux_loss.py`
  - Adds Stage2 evidence loss helpers.
  - Keeps Stage1 hypothesis loss unchanged.
- `opencood/loss/point_pillar_loss.py`
- `opencood/loss/point_pillar_depth_loss.py`
- `opencood/loss/point_pillar_pyramid_loss.py`
  - Dispatch HVP-v3 loss by `output_dict["hvp_v3"]["stage"]`.
  - Log `HVP-v3 Loss` and `Stage2 Evidence Loss`.
- `opencood/tools/check_hvp_heal_v3_stage2_smoke.py`
  - Checks m2/m3/m4 forward, evidence tensor shapes, total-loss integration, backward gradients, and checkpoint key filtering.
- `opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage2/`
  - Adds explicit m2/m3/m4 Stage2 evidence adaptation templates.

### Config

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

### Safety

- Default `hvp_v3.enabled=false` still keeps official HEAL behavior.
- Existing ordinary `heter_pyramid_single` configs are not changed.
- Stage1 logic remains intact.
- Stage3, packet, and hybrid modes are not implemented here.
- Official `heter_pyramid_collab.py` is not modified.
- v2.x HVP-CBEA modules and v1 HBEC are not modified.
- No logs directory is modified.

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

## HVP-CBEA-v3 Stage3 collaborative hypothesis-evidence aggregation

Base hash: `972582f`

### Motivation

- Server Stage1 and Stage2 checkpoints are ready.
- Stage3 yaml was missing, so collaborative HVP-CBEA aggregation training could not be launched from the staged checkpoints.
- Stage3 should train the feature-level CBEA aggregator, not packet mode.

### Implementation

- `opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/stage3/cbea_aggregator.yaml`
  - Adds the Stage3 training template.
  - Uses `core_method: heter_pyramid_collab_hvp_cbea`.
  - Enables `hvp_cbea.enabled=true`.
  - Keeps `hvp_cbea.packet.enabled=false`.
  - Enables `train_only_hvp=true` so inherited HEAL parameters and non-HVP BN buffers stay frozen.
  - Enables HVP-CBEA auxiliary losses for hypothesis heatmap, residual focus, alpha regularization, and residual regularization.
- `opencood/tools/prepare_hvp_cbea_stage3.py`
  - Prepares `opencood/logs/HVP_CBEA_v3/stage3/cbea_aggregator/net_epoch1.pth` from Stage1 m1 and Stage2 m2/m3/m4 best checkpoints.
  - Missing HVP-CBEA module keys are expected and random-initialized by Stage3 model construction.
- `opencood/tools/check_hvp_cbea_stage3_smoke.py`
  - Checks yaml loading, dummy Stage3 model creation, packet disabled, forward, loss, backward, and HVP-only gradients.
- `docs/HVP_HEAL_V3_PLAN.md`
- `docs/HVP_HEAL_V3_CONFIG_SCHEMA.md`
- `docs/HVP_CBEA_CONFIG_SCHEMA.md`
  - Document Stage3 config and checkpoint preparation.

### Safety

- No `logs/` files are modified by this patch.
- Official HEAL `heter_pyramid_collab.py` is not modified.
- Stage1 and Stage2 smoke paths remain unchanged.
- Packet mode remains disabled in the Stage3 yaml.

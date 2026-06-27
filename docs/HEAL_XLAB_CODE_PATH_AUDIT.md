# HEAL-XLab Code Path Audit

Base hash: `96812ed`

## Final Inference Entry

- Current HEAL final-infer command in `README.md` points to `opencood/tools/inference_heter_in_order.py`.
- Main entry: `opencood/tools/inference_heter_in_order.py:62`, `main()`.
- The script loops over collaborator counts at `opencood/tools/inference_heter_in_order.py:162` and writes `hypes['use_cav'] = use_cav` at line 163.

## use_cav Control Logic

- CLI argument `--use_cav` is defined in `opencood/tools/inference_heter_in_order.py`.
- Default values are evaluated into `use_cav_and_lidar_config_pair` at `opencood/tools/inference_heter_in_order.py:160`.
- For intermediate HEAL final inference, `hypes['fusion']['core_method'] += 'infer'` switches to `IntermediateHeterinferFusionDataset`.
- Dataset-side participation limit is implemented in `opencood/data_utils/datasets/heter_infer/intermediate_heter_infer_fusion_dataset.py`:
  - `self.use_cav = params['use_cav']` at line 55.
  - agents at index `_i >= self.use_cav` are skipped for model input at line 209.
  - full GT object collection is kept before this skip, so GT can still cover the full scene.

## post_processor.post_process Call Position

- Intermediate final-infer path calls `opencood/tools/inference_utils.py:156`, `inference_intermediate_fusion()`.
- That delegates to `inference_early_fusion()` at `opencood/tools/inference_utils.py:123`.
- `dataset.post_process(batch_data, output_dict)` is called at `opencood/tools/inference_utils.py:145`.
- The intermediate-heter dataset method calls `self.post_processor.post_process(data_dict, output_dict)` in `opencood/data_utils/datasets/intermediate_heter_fusion_dataset.py`.

## Prediction and Ground Truth Tensor Production

- `pred_box_tensor` and `pred_score` are produced by `self.post_processor.post_process(data_dict, output_dict)` in dataset `post_process`.
- `gt_box_tensor` is produced by `self.post_processor.generate_gt_bbx(data_dict)`.
- Voxel post-processing converts model outputs to boxes, applies official NMS, range filtering, and returns `(pred_box3d_tensor, scores)` in `opencood/data_utils/post_processor/voxel_postprocessor.py`.
- In final-infer script, tensors are unpacked from `infer_result` at `opencood/tools/inference_heter_in_order.py:244-246`.

## AP@0.3 / AP@0.5 / AP@0.7 Statistics

- Per-frame TP/FP accumulation uses `opencood/utils/eval_utils.py:40`, `caluclate_tp_fp()`.
- Final AP calculation uses `opencood/utils/eval_utils.py:140`, `eval_final_results()`.
- AP thresholds are calculated at `opencood/utils/eval_utils.py:143-145` for 0.30, 0.50, and 0.70.
- In final-infer script, TP/FP calls occur at `opencood/tools/inference_heter_in_order.py:270`, `275`, and `280`.

## inference_utils Fusion Functions

- Late fusion: `opencood/tools/inference_utils.py:18`, `inference_late_fusion()`.
- No fusion: `opencood/tools/inference_utils.py:51`, `inference_no_fusion()`.
- No fusion with uncertainty: `opencood/tools/inference_utils.py:88`, `inference_no_fusion_w_uncertainty()`.
- Early fusion: `opencood/tools/inference_utils.py:123`, `inference_early_fusion()`.
- Intermediate fusion: `opencood/tools/inference_utils.py:156`, `inference_intermediate_fusion()`.
- CAV pose visualization helper: `opencood/tools/inference_utils.py:266`, `get_cav_box()`.

## Collaborator-only Prediction Availability

- The official intermediate final-infer path passes only `batch_data['ego']` through the fused model and returns a single fused prediction.
- No audited official helper returns collaborator-only object predictions for the current intermediate HEAL final-infer path.
- `get_cav_box()` returns agent pose boxes for visualization, not collaborator object predictions.
- Late fusion can run each CAV through the model, but the current target path is intermediate HEAL final-infer and does not expose per-collaborator predictions after official post-process.

## Safe Fallback Decision

- HBEC v1 does not synthesize collaborator evidence from GT, raw camera features, or guessed model internals.
- If no reliable collaborator evidence is provided in `infer_context`, `HBECPostProcessor` returns official `pred_box_tensor`, `pred_score`, and `gt_box_tensor` unchanged with `fallback_reason = no_collaborator_evidence`.
- The hook is inserted after official post-process and before AP accumulation in `opencood/tools/inference_heter_in_order.py`.


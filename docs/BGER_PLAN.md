# BGER Plan

BGER stands for Box-Guided Evidence Reactivation.

BGER is a HEAL-compatible asymmetric collaboration line: **decision-level
communication, feature-level fusion**. Collaborators transmit only detection
boxes; the ego vehicle uses them as spatial priors to re-examine its own BEV
features.

## Position

- HEAL / PACT-CBEA transmit dense BEV feature (+ evidence) tensors between
  agents. BGER transmits only box messages (~KB per frame), targeting
  weak-communication deployments.
- Versus classic late fusion, BGER does not merge boxes at the decision
  level. The ego projects collaborator boxes into its BEV frame, renders
  them into prior maps, and a small trainable module (`bger_refine`)
  produces a residual update of the ego feature before the shared HEAL
  detection heads. Sub-threshold ego evidence in occluded regions can be
  reactivated by the prior instead of being lost to the ego's own detection
  threshold.
- Versus PACT-CBEA, heterogeneity is handled at the box level: a
  collaborator only needs a standard detector output (box + confidence).
  New vendors need zero adaptation of the ego side and no shared feature
  space at all.

## Method

```text
collaborator j (any modality)
  -> own single-view pipeline -> boxes + confidences   (message, ~KB)

ego (m1)
  -> own single-view pipeline -> ego BEV feature F_ego
  -> project received boxes into ego frame (pairwise_t_matrix[j, 0])
  -> render prior maps (gaussian bump / box mask / optional yaw)
  -> F' = F_ego + gate * refine(concat(F_ego, prior))    (zero-init conv)
  -> shared HEAL cls/reg/dir heads on F'
```

Three occlusion regimes handled by one mechanism:

- partial occlusion: ego has sub-threshold evidence; the prior locally
  reactivates it (main accuracy gain).
- full occlusion: ego has no evidence; refinement learns to synthesize
  detections supported by the prior confidence.
- collaborator false positives: ego has clear free-space evidence; the
  refinement learns to suppress unsupported priors (robustness gain over
  plain late fusion).

## Box sources (`bger.box_source`)

- `oracle`: collaborator messages are the GT boxes visible from the
  collaborator's own viewpoint (`object_bbx_center_single` /
  `object_bbx_mask_single`, already produced by the intermediateheter
  dataset). Measures the upper bound of decision-level communication.
- `single_decode`: collaborator messages are decoded from its own single
  branch (`pyramid_backbone.forward_single` + shared heads +
  `delta_to_boxes3d` + rotated NMS), detached — boxes are messages, not
  gradient paths. This is the realistic deployment mode.

## Increment Boundary

All new files; gating key is `model.args.bger.enabled` (default absent /
false => the model behaves exactly like the official
`heter_pyramid_collab`, verified bit-exact by the smoke test):

- `opencood/models/sub_modules/bger_box_prior.py`
- `opencood/models/sub_modules/bger_refine.py`
- `opencood/models/heter_pyramid_collab_bger.py`
- `opencood/hypes_yaml/HEAL_XLab_v4_BGER/stage_a/m1_bger_oracle.yaml`
- `opencood/hypes_yaml/HEAL_XLab_v4_BGER/stage_b/m1_ego_heter_bger_single.yaml`
- `opencood/hypes_yaml/HEAL_XLab_v4_BGER/final_infer/m1_ego_m2m3m4_bger.yaml`
- `opencood/hypes_yaml/HEAL_XLab_v4_BGER/final_infer/m1_ego_m2m3m4_boxmerge.yaml`
- `opencood/tools/prepare_bger.py`
- `opencood/tools/inference_bger.py`
- `opencood/tools/check_bger_smoke.py`

No official file (models, datasets, losses, train.py, inference.py,
official yaml) is modified. The intermediateheter dataset already collates
per-agent single-view GT (`heterogeneous=True`), so oracle mode needs no
dataset change.

## Training Protocol

Stage A — oracle upper bound (ego=m1, all agents mapped to m1):

```bash
# compose init checkpoint from the official HEAL Stage1 m1 checkpoint
python opencood/tools/prepare_bger.py --stage stage_a \
    --m1-ckpt <path/to/HEAL/stage1/m1/net_epoch_bestval_at*.pth>

# frozen base, trains only bger_refine
python opencood/tools/train.py \
    -y opencood/hypes_yaml/HEAL_XLab_v4_BGER/stage_a/m1_bger_oracle.yaml \
    --model_dir opencood/logs/BGER_v1/stage_a
```

Decision gate: if stage_a (oracle) shows no meaningful AP gain over the
ego-only baseline, the direction is not worth pursuing further.

Stage B — realistic heterogeneous boxes (ego=m1, collaborators m2/m3/m4):

```bash
python opencood/tools/prepare_bger.py --stage stage_b \
    --m1-ckpt <...m1 stage1...> --m2-ckpt <...m2 stage2...> \
    --m3-ckpt <...m3 stage2...> --m4-ckpt <...m4 stage2...> \
    --stage-a-refine opencood/logs/BGER_v1/stage_a/net_epoch_bestval.pth

python opencood/tools/train.py \
    -y opencood/hypes_yaml/HEAL_XLab_v4_BGER/stage_b/m1_ego_heter_bger_single.yaml \
    --model_dir opencood/logs/BGER_v1/stage_b
```

## Evaluation

```bash
# BGER heterogeneous evaluation (+ communication accounting)
python opencood/tools/inference_bger.py --model_dir opencood/logs/BGER_v1/stage_b

# late-fusion control group: same box messages, no feature reactivation
#   (copy final_infer/m1_ego_m2m3m4_boxmerge.yaml as config.yaml into a log
#    dir with the same composed checkpoint)
python opencood/tools/inference_bger.py --model_dir <boxmerge_dir> --box_merge

# ego-only lower bound: bger.mode: box_merge_only WITHOUT --box_merge
```

Comparison axes for the accuracy-bandwidth pareto story:

- ego-only (no fusion, lower bound)
- late fusion / box_merge_only (same bandwidth as BGER)
- BGER oracle (upper bound of decision-level communication)
- BGER single_decode (proposed)
- HEAL intermediate fusion (accuracy upper bound, feature bandwidth)

`output_dict['bger']` reports per-frame `comm_bytes_boxes`,
`comm_bytes_feature_equiv`, and `comm_ratio`; inference_bger.py averages
them over the test set.

## Smoke Test

```bash
python opencood/tools/check_bger_smoke.py
```

CPU-only, no dataset/checkpoint. Verifies yaml sanity, the disabled ==
official bit-exact equivalence, oracle / single_decode forwards, zero-init
identity of the refinement, prior rendering and collaborator->ego box
projection geometry, and gradient isolation under freeze_base.

## Not Included In This Increment

- Any-modality ego (m2/m3/m4 as ego).
- Motion compensation for delayed box messages.
- Explicit box-adoption post-processing beyond the box_merge baseline.
- Occlusion-stratified evaluation tooling (planned analysis, separate
  increment).

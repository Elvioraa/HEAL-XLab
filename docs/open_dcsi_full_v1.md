# Open-DCSI-HEAL Full v1

Open-DCSI is an opt-in extension of the existing HEAL pyramid models. Existing
model classes, YAML files, losses, checkpoints, and training commands remain
unchanged. Every new state-dict key is under `open_dcsi.*`.

## Verified integration points

The official collaborative path in
`opencood/models/heter_pyramid_collab.py:HeterPyramidCollab.forward` is:

```text
modality input -> encoder_mX -> backbone_mX -> aligner_mX
-> heter_feature_2d -> PyramidFusion.forward_single/forward_collab
-> shrink_conv -> cls/reg/dir heads
```

`OpenDCSICommonRuntime` replaces only the selected wrapper instance's
`pyramid_backbone.forward_single` and `forward_collab` methods. It does not
modify the official `PyramidFusion` class. The disabled wrapper returns through
the parent method before creating or executing any Open-DCSI module.

The enabled collaborative path performs, at every actual pyramid scale:

```text
F_i -> modality-local projector -> C_i -> shared decoder -> reconstructed F_i
F_i - reconstructed F_i -> innovation
C_i -> evidence/validity weighted residual consensus -> decoded common feature
innovation -> proposal ROI token -> quality-gated set aggregation
-> bounded local cross-scale sampling -> zero-initialized box residual
```

The final wrapper inherits the optional PACT-CBEA model and therefore remains
compatible with its evidence-prior call signature. Descriptor conflict is off
by default. Open-DCSI's router has no modality ID, modality embedding, or
pair-specific parameter.

## Open heterogeneous training

Stage1 uses homogeneous m1 collaborative batches. Stage2 uses one new modality
per run. With `stage2_independent.enabled: true`, shared fusion, decoders,
aggregation, geometry refinement, detection heads, and their BatchNorm state
are frozen. The Stage2 optimizer iterator exposes only the current modality's
official branch, common projector, tokenizer, and local quality head. Mixed
modality Stage2 batches fail before forward execution.

Templates are under:

```text
opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/open_dcsi/
```

The templates retain the corresponding official batch size, epoch count,
optimizer, scheduler, data roots, and modality configuration.

## Checkpoint preparation

Extract each independently trained Stage2 local state:

```bash
python opencood/tools/prepare_open_dcsi.py extract \
  --checkpoint <m2-stage2-checkpoint> --modality m2 \
  --output <m2-local-checkpoint>
```

Repeat for m3 and m4. Compose the Stage1 shared state and three local states:

```bash
python opencood/tools/prepare_open_dcsi.py compose \
  --config opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/open_dcsi/inference/open_dcsi_cav1_4_full.yaml \
  --stage1 <stage1-checkpoint> \
  --m2 <m2-local-or-full-stage2-checkpoint> \
  --m3 <m3-local-or-full-stage2-checkpoint> \
  --m4 <m4-local-or-full-stage2-checkpoint> \
  --output <final-model-dir>/net_epoch1.pth
```

Both commands reject missing, overlapping, unexpected, and shape-mismatched
state. They do not overwrite an existing output unless `--force` is explicit.
A source SHA256 manifest is written beside every output.

## Inference

Place the selected inference YAML at `<final-model-dir>/config.yaml`, then use
the repository's ordered heterogeneous inference entry point. The CAV count is
controlled by the command, not by a fixed model tensor layout:

```bash
python opencood/tools/inference_heter_in_order.py \
  --model_dir <final-model-dir> \
  --fusion_method intermediate \
  --use_cav '[1,2,3,4]'
```

`inference_utils.inference_early_fusion` applies the Open-DCSI box residual
only when the model switch is enabled and a valid refinement is present.
Unmatched boxes, empty tokens, rejected tokens, and non-finite residuals keep
the official post-NMS result.

## Local verification

```bash
python opencood/tools/check_open_dcsi_smoke.py
python opencood/tools/audit_open_dcsi_baseline_parity.py
python opencood/tools/audit_open_dcsi_open_heterogeneous.py
python opencood/tools/profile_open_dcsi_resources.py --warmup 2 --iterations 5
```

The resource profiler uses the same batch, CAV count, spatial shape, and model
weights for dense and streaming runs. It reports parameter/checkpoint size,
real serialized payload and metadata bytes, per-collaborator bytes and tokens,
CUDA peaks when available, MAC/FLOP estimates, and synchronized latency.

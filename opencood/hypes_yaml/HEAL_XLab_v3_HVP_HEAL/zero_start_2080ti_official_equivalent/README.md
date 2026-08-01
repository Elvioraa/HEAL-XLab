# HEAL zero-start experiment pack for one RTX 2080 Ti

Run all commands from the repository root on the Linux training server. The
commands below use only checkpoints produced under
`opencood/logs/HEAL_ZERO_START_2080TI`.

## One-time run directory setup

```bash
export PACK_DIR=opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/zero_start_2080ti_official_equivalent
export RUN_ROOT=opencood/logs/HEAL_ZERO_START_2080TI
export M1_MODEL_DIR="$RUN_ROOT/stage1/m1_base"
export M2_MODEL_DIR="$RUN_ROOT/stage2/m2_alignto_m1"
export M3_MODEL_DIR="$RUN_ROOT/stage2/m3_alignto_m1"
export M4_MODEL_DIR="$RUN_ROOT/stage2/m4_alignto_m1"
export FINAL_MODEL_DIR="$RUN_ROOT/final_infer"
export STAGE3_MODEL_DIR="$RUN_ROOT/object_stage3"

test ! -e "$RUN_ROOT" || { echo "Refusing to reuse existing run root: $RUN_ROOT"; exit 1; }
mkdir -p "$M1_MODEL_DIR" "$M2_MODEL_DIR" "$M3_MODEL_DIR" "$M4_MODEL_DIR"
mkdir -p "$FINAL_MODEL_DIR" "$STAGE3_MODEL_DIR"
cp "$PACK_DIR/stage1/m1_pyramid.yaml" "$M1_MODEL_DIR/config.yaml"
cp "$PACK_DIR/stage2/m2_single_pyramid.yaml" "$M2_MODEL_DIR/config.yaml"
cp "$PACK_DIR/stage2/m3_single_pyramid.yaml" "$M3_MODEL_DIR/config.yaml"
cp "$PACK_DIR/stage2/m4_single_pyramid.yaml" "$M4_MODEL_DIR/config.yaml"
cp "$PACK_DIR/final_infer/m1m2m3m4.yaml" "$FINAL_MODEL_DIR/config.yaml"
```

## 1. m1 Stage 1, primary FP32 run

`M1_MODEL_DIR` contains no checkpoint at this point, so `train.py` starts from
random HEAL weights.

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  -y None --model_dir "$M1_MODEL_DIR"
```

## 2. m1 Stage 1, AMP fallback after an OOM

Use the same command with the real `--amp` runtime override. If the failed run
already wrote a checkpoint, this command follows the repository's normal
resume behavior; an epoch-zero OOM normally occurs before the first save.

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  -y None --model_dir "$M1_MODEL_DIR" --amp
```

The same `--amp` flag may be appended to an m2, m3, or m4 command if its FP32
micro-batch still exceeds 11 GB. AMP is a fallback, not the primary run.

## 3. Select the self-produced m1 best-validation checkpoint

The training script keeps one `net_epoch_bestval_at*.pth` file. Refuse to
continue unless exactly one candidate exists, then install it as the official
Stage 2 seed basename in each new-modality directory.

```bash
export M1_BEST_COUNT=$(find "$M1_MODEL_DIR" -maxdepth 1 -type f \
  -name 'net_epoch_bestval_at*.pth' | wc -l)
test "$M1_BEST_COUNT" -eq 1
export M1_BEST=$(find "$M1_MODEL_DIR" -maxdepth 1 -type f \
  -name 'net_epoch_bestval_at*.pth' -print)
cp "$M1_BEST" "$M2_MODEL_DIR/net_epoch1.pth"
cp "$M1_BEST" "$M3_MODEL_DIR/net_epoch1.pth"
cp "$M1_BEST" "$M4_MODEL_DIR/net_epoch1.pth"
```

`load_saved_model()` interprets `net_epoch1.pth` as `init_epoch=1`. With the
official `epoches: 25`, each Stage 2 loop is `range(1, 25)`: 24 new-modality
training epochs, with the scheduler pre-advanced once. This exactly follows
the repository README workflow and is intentionally not changed to 25 fresh
epochs.

## 4. m2 Stage 2

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  -y None --model_dir "$M2_MODEL_DIR"
```

## 5. m3 Stage 2

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  -y None --model_dir "$M3_MODEL_DIR"
```

## 6. m4 Stage 2

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  -y None --model_dir "$M4_MODEL_DIR"
```

Run m2, m3, and m4 sequentially on the single 2080 Ti. Their YAML files use
effective global batches 8, 16, and 8 through gradient accumulation.

## 7. Merge m2, m3, m4, and m1

Do not change the argument order. `heal_tools.py` selects the one bestval
checkpoint in each model directory and writes `net_epoch1.pth` to the final
directory.

```bash
python opencood/tools/heal_tools.py merge_final \
  "$M2_MODEL_DIR" \
  "$M3_MODEL_DIR" \
  "$M4_MODEL_DIR" \
  "$M1_MODEL_DIR" \
  "$FINAL_MODEL_DIR"
export FINAL_CHECKPOINT="$FINAL_MODEL_DIR/net_epoch1.pth"
```

## 8. Final four-modality HEAL inference and AP

This is the repository's official ordered heterogeneous evaluation. The final
config uses inference batch size 1 and leaves AP calculation unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_heter_in_order.py \
  --model_dir "$FINAL_MODEL_DIR" --fusion_method intermediate
```

## 9. Object Stage 3 real-data forward preflight

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/check_pact_cbea_object_stage3_realdata.py \
  -y "$PACK_DIR/object_stage3/heter_pyramid_collab_pact_cbea_object_stage3.yaml" \
  --base-checkpoint "$FINAL_CHECKPOINT" \
  --dataset-split validate --device cuda:0 --batch-count 1 \
  --allow-scratch-stage3
```

## 10. Object Stage 3 one-batch backward preflight

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/check_pact_cbea_object_stage3_realdata.py \
  -y "$PACK_DIR/object_stage3/heter_pyramid_collab_pact_cbea_object_stage3.yaml" \
  --base-checkpoint "$FINAL_CHECKPOINT" \
  --dataset-split train --device cuda:0 --batch-count 1 \
  --allow-scratch-stage3 --backward
```

## 11. Object Stage 3 formal training

The strict HEAL base runs frozen in eval/no-grad mode. Only
`object_stage3_refiner` is optimized, starting from random initialization.

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_pact_cbea_object_stage3.py \
  -y "$PACK_DIR/object_stage3/heter_pyramid_collab_pact_cbea_object_stage3.yaml" \
  --base-checkpoint "$FINAL_CHECKPOINT" \
  --output-dir "$STAGE3_MODEL_DIR" --device cuda:0
```

## 12. Object Stage 3 inference and AP

```bash
export STAGE3_CHECKPOINT="$STAGE3_MODEL_DIR/stage3_best.pth"
mkdir -p "$STAGE3_MODEL_DIR/inference_test"
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_pact_cbea_object_stage3.py \
  -y "$PACK_DIR/object_stage3/heter_pyramid_collab_pact_cbea_object_stage3.yaml" \
  --base-checkpoint "$FINAL_CHECKPOINT" \
  --stage3-checkpoint "$STAGE3_CHECKPOINT" \
  --output-dir "$STAGE3_MODEL_DIR/inference_test" \
  --device cuda:0 --dataset-split test
```

## Reproducibility notes

- The primary runs are FP32. `--amp` is an OOM fallback and is not numerically
  identical to the official FP32 run.
- Gradient accumulation preserves the confirmed two-GPU global sample count:
  m1/m2/m4 use `1 x 8 = 8`, and m3 uses `1 x 16 = 16`.
- BatchNorm still observes micro-batch 1. Its running statistics cannot match
  the official per-GPU batches 4/8, even when gradient batches are equivalent.
- m2 deliberately retains ImageNet-pretrained EfficientNet-B0, as required by
  the official implementation. "Zero start" means no existing HEAL checkpoint.
- No public or historical HEAL checkpoint is used. m1, m2, m3, m4, the merged
  base, and the object Stage 3 checkpoint are all produced by this run.
- `--accumulation-steps N` is a real CLI override for diagnosis. Formal runs
  should use the self-contained values stored in the experiment YAML files.

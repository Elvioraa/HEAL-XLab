# PACT-CBEA Object Stage 3 v1 experiment

Run every command from the repository root on the Linux training server. The
entire run stays under the single experiment root
`opencood/logs/PACT_CBEA_OBJECT_STAGE3_v1`. The commands do not use public,
historical, external, or other experiment checkpoints.

## Fixed directories and safety check

The server directories may already exist, but they must be empty or contain
only files produced by the same stage. This setup does not create log files or
weights and never creates a timestamped sibling directory.

```bash
set -euo pipefail
export PACK_DIR=opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/PACT_CBEA_OBJECT_STAGE3_v1
export RUN_ROOT=opencood/logs/PACT_CBEA_OBJECT_STAGE3_v1
export M1_MODEL_DIR="$RUN_ROOT/stage1/m1_base"
export M2_MODEL_DIR="$RUN_ROOT/stage2/m2_alignto_m1"
export M3_MODEL_DIR="$RUN_ROOT/stage2/m3_alignto_m1"
export M4_MODEL_DIR="$RUN_ROOT/stage2/m4_alignto_m1"
export MERGED_MODEL_DIR="$RUN_ROOT/stage2/merged_m1m2m3m4"
export STAGE3_MODEL_DIR="$RUN_ROOT/stage3/object_refiner"
export FINAL_INFER_DIR="$RUN_ROOT/final_infer"

for directory in \
  "$M1_MODEL_DIR" "$M2_MODEL_DIR" "$M3_MODEL_DIR" "$M4_MODEL_DIR" \
  "$MERGED_MODEL_DIR" "$STAGE3_MODEL_DIR" "$FINAL_INFER_DIR"; do
  test -d "$directory" || {
    echo "Required precreated directory is missing: $directory" >&2
    exit 1
  }
done

assert_stage_dir_safe() {
  local directory=$1
  local profile=$2
  local path base
  shopt -s nullglob dotglob
  for path in "$directory"/*; do
    base=${path##*/}
    case "$profile:$base" in
      train:config.yaml|train:train.log|train:events.out.tfevents.*|train:net_epoch*.pth|train:eval*.yaml) ;;
      merge:config.yaml|merge:merge.log|merge:net_epoch1.pth) ;;
      stage3:config.yaml|stage3:train.log|stage3:stage3_best.pth|stage3:stage3_epoch*.pth) ;;
      infer:config.yaml|infer:infer.log|infer:eval_object_stage3_*.yaml) ;;
      *)
        echo "Refusing unknown content in $directory: $base" >&2
        shopt -u nullglob dotglob
        return 1
        ;;
    esac
  done
  shopt -u nullglob dotglob
}

install_config_once() {
  local source=$1
  local destination=$2
  if test -e "$destination"; then
    cmp -s "$source" "$destination" || {
      echo "Refusing to overwrite a different config: $destination" >&2
      return 1
    }
  else
    cp "$source" "$destination"
  fi
}

assert_stage_dir_safe "$M1_MODEL_DIR" train
assert_stage_dir_safe "$M2_MODEL_DIR" train
assert_stage_dir_safe "$M3_MODEL_DIR" train
assert_stage_dir_safe "$M4_MODEL_DIR" train
assert_stage_dir_safe "$MERGED_MODEL_DIR" merge
assert_stage_dir_safe "$STAGE3_MODEL_DIR" stage3
assert_stage_dir_safe "$FINAL_INFER_DIR" infer

install_config_once "$PACK_DIR/stage1/m1_pyramid.yaml" "$M1_MODEL_DIR/config.yaml"
install_config_once "$PACK_DIR/stage2/m2_single_pyramid.yaml" "$M2_MODEL_DIR/config.yaml"
install_config_once "$PACK_DIR/stage2/m3_single_pyramid.yaml" "$M3_MODEL_DIR/config.yaml"
install_config_once "$PACK_DIR/stage2/m4_single_pyramid.yaml" "$M4_MODEL_DIR/config.yaml"
install_config_once "$PACK_DIR/merged_base/m1m2m3m4.yaml" "$MERGED_MODEL_DIR/config.yaml"
install_config_once \
  "$PACK_DIR/final_infer/heter_pyramid_collab_pact_cbea_object_stage3.yaml" \
  "$FINAL_INFER_DIR/config.yaml"
```

The hierarchical YAML `name` values map beneath `opencood/logs/`. The
repository's `setup_train()` also appends a timestamp for new runs, so the
formal Stage 1/2 commands deliberately pass `--model_dir` to select the exact
precreated directories. An empty model directory starts at epoch zero; known
`net_epoch*.pth` files follow the normal resume path.

## 1. Train m1 Stage 1 from random initialization

```bash
set -eu
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  -y None --model_dir "$M1_MODEL_DIR" \
  2>&1 | tee -a "$M1_MODEL_DIR/train.log"
```

If FP32 does not fit the available GPU, append `--amp` to the same command.

## 2. Select the m1 best checkpoint and install Stage 2 seeds

```bash
set -euo pipefail
M1_BEST_COUNT=$(find "$M1_MODEL_DIR" -maxdepth 1 -type f \
  -name 'net_epoch_bestval_at*.pth' | wc -l)
test "$M1_BEST_COUNT" -eq 1
M1_BEST=$(find "$M1_MODEL_DIR" -maxdepth 1 -type f \
  -name 'net_epoch_bestval_at*.pth' -print)
for destination in "$M2_MODEL_DIR" "$M3_MODEL_DIR" "$M4_MODEL_DIR"; do
  test ! -e "$destination/net_epoch1.pth" || {
    echo "Refusing to overwrite Stage 2 seed: $destination/net_epoch1.pth" >&2
    exit 1
  }
  cp "$M1_BEST" "$destination/net_epoch1.pth"
done
```

`load_saved_model()` reads each installed `net_epoch1.pth` as
`init_epoch=1`. With `epoches: 25`, Stage 2 executes `range(1, 25)`: 24
new-modality epochs with the scheduler pre-advanced once.

## 3. Train m2 Stage 2

```bash
set -eu
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  -y None --model_dir "$M2_MODEL_DIR" \
  2>&1 | tee -a "$M2_MODEL_DIR/train.log"
```

## 4. Train m3 Stage 2

```bash
set -eu
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  -y None --model_dir "$M3_MODEL_DIR" \
  2>&1 | tee -a "$M3_MODEL_DIR/train.log"
```

## 5. Train m4 Stage 2

```bash
set -eu
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  -y None --model_dir "$M4_MODEL_DIR" \
  2>&1 | tee -a "$M4_MODEL_DIR/train.log"
```

The fixed single-GPU plans are m1/m2/m4 `1 x 8 = 8`, m3
`1 x 16 = 16`, and Object Stage 3 `1 x 1 = 1`. They are not bound to a
specific GPU model.

## 6. Merge the four-modality HEAL base

The argument order is strictly m2, m3, m4, m1, output. The merged base is not
the final inference directory.

```bash
set -eu
set -o pipefail
python opencood/tools/heal_tools.py merge_final \
  "$M2_MODEL_DIR" \
  "$M3_MODEL_DIR" \
  "$M4_MODEL_DIR" \
  "$M1_MODEL_DIR" \
  "$MERGED_MODEL_DIR" \
  2>&1 | tee -a "$MERGED_MODEL_DIR/merge.log"
export BASE_CHECKPOINT="$MERGED_MODEL_DIR/net_epoch1.pth"
```

## 7. Train the frozen-base Object Stage 3 refiner

Only `object_stage3_refiner` is optimized. The HEAL base comes from the
merged checkpoint above and remains frozen. For a resumed Stage 3 run, add
`--resume-stage3 "$STAGE3_MODEL_DIR/<selected-stage3-checkpoint>.pth"`.

```bash
set -eu
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train_pact_cbea_object_stage3.py \
  -y "$PACK_DIR/stage3/heter_pyramid_collab_pact_cbea_object_stage3.yaml" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$STAGE3_MODEL_DIR" --device cuda:0 \
  2>&1 | tee -a "$STAGE3_MODEL_DIR/train.log"
```

## 8. Final Object Stage 3 inference and AP

This is the only final inference step. It reads the merged base and Stage 3
checkpoint independently, while all evaluation files are written only to
`final_infer/`.

```bash
set -eu
set -o pipefail
export STAGE3_CHECKPOINT="$STAGE3_MODEL_DIR/stage3_best.pth"
CUDA_VISIBLE_DEVICES=0 python opencood/tools/inference_pact_cbea_object_stage3.py \
  -y "$PACK_DIR/final_infer/heter_pyramid_collab_pact_cbea_object_stage3.yaml" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --stage3-checkpoint "$STAGE3_CHECKPOINT" \
  --output-dir "$FINAL_INFER_DIR" \
  --device cuda:0 --dataset-split test \
  2>&1 | tee -a "$FINAL_INFER_DIR/infer.log"
```

## Reproducibility notes

- Logs are created only when the corresponding command starts. `tee -a`
  appends after interruption, mirrors stdout/stderr to the terminal, and
  `pipefail` preserves Python failures.
- FP32 is the primary path. AMP is an optional memory fallback and is not
  numerically identical to FP32.
- Gradient accumulation preserves the confirmed official global sample counts,
  but BatchNorm still observes micro-batch 1.
- m2 retains the official ImageNet-pretrained EfficientNet-B0 dependency.
- Every HEAL and PACT-CBEA checkpoint used by this workflow is produced under
  `opencood/logs/PACT_CBEA_OBJECT_STAGE3_v1`.

"""Prepare HVP-CBEA v3 Stage3 initial checkpoint from Stage1/Stage2 best ckpts."""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.tools.heal_tools import get_model_path_from_dir, merge_and_save_final


DEFAULT_STAGE1_DIR = "opencood/logs/HVP_CBEA_v3/stage1/m1_base"
DEFAULT_STAGE2_M2_DIR = "opencood/logs/HVP_CBEA_v3/stage2/m2_alignto_m1"
DEFAULT_STAGE2_M3_DIR = "opencood/logs/HVP_CBEA_v3/stage2/m3_alignto_m1"
DEFAULT_STAGE2_M4_DIR = "opencood/logs/HVP_CBEA_v3/stage2/m4_alignto_m1"
DEFAULT_OUTPUT_DIR = "opencood/logs/HVP_CBEA_v3/stage3/cbea_aggregator"


def main():
    args = parse_args()
    stage_dirs = [
        _resolve(args.stage2_m2_dir),
        _resolve(args.stage2_m3_dir),
        _resolve(args.stage2_m4_dir),
        _resolve(args.stage1_dir),
    ]
    output_dir = _resolve(args.output_dir)

    print("Preparing HVP-CBEA v3 Stage3 checkpoint from:")
    for stage_dir in stage_dirs:
        print(" - %s" % stage_dir)
    print("Output dir: %s" % output_dir)

    if args.dry_run:
        for stage_dir in stage_dirs:
            if os.path.isdir(stage_dir):
                get_model_path_from_dir(stage_dir)
            else:
                print("missing directory: %s" % stage_dir)
        print("dry-run only; no checkpoint was written")
        return

    os.makedirs(output_dir, exist_ok=True)
    merge_and_save_final(stage_dirs, output_dir)
    output_path = os.path.join(output_dir, "net_epoch1.pth")
    print("Stage3 initial checkpoint written to: %s" % output_path)
    print("HVP-CBEA module keys are intentionally absent and will be randomly initialized.")
    print("Training loads this checkpoint with strict=False via train_utils.load_saved_model().")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Stage1/Stage2 HEAL checkpoints for HVP-CBEA v3 Stage3.",
    )
    parser.add_argument("--stage1-dir", default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--stage2-m2-dir", default=DEFAULT_STAGE2_M2_DIR)
    parser.add_argument("--stage2-m3-dir", default=DEFAULT_STAGE2_M3_DIR)
    parser.add_argument("--stage2-m4-dir", default=DEFAULT_STAGE2_M4_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


if __name__ == "__main__":
    main()

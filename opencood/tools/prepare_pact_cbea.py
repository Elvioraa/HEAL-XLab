"""Prepare PACT-CBEA initialization checkpoint from local expert branches."""

import argparse
from collections import OrderedDict
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.tools.heal_tools import merge_dict


DEFAULT_M1 = "opencood/logs/HVP_CBEA_v3/stage1/m1_base/net_epoch_bestval_at27.pth"
DEFAULT_M2 = "opencood/logs/HVP_CBEA_v3/stage2/m2_alignto_m1/net_epoch_bestval_at13.pth"
DEFAULT_M3 = "opencood/logs/HVP_CBEA_v3/stage2/m3_alignto_m1/net_epoch_bestval_at25.pth"
DEFAULT_M4 = "opencood/logs/HVP_CBEA_v3/stage2/m4_alignto_m1/net_epoch_bestval_at19.pth"
DEFAULT_OUTPUT_DIR = "opencood/logs/PACT_CBEA_v1/rule_cbea"


def main():
    args = parse_args()
    checkpoint_paths = [
        _resolve(args.m2_ckpt),
        _resolve(args.m3_ckpt),
        _resolve(args.m4_ckpt),
        _resolve(args.m1_ckpt),
    ]
    output_dir = _resolve(args.output_dir)

    print("Preparing PACT-CBEA initialization checkpoint from local expert branches...")
    for path in checkpoint_paths:
        print(" - %s" % path)
    print("Output dir: %s" % output_dir)

    if args.dry_run:
        for path in checkpoint_paths:
            print("%s: %s" % ("FOUND" if os.path.exists(path) else "MISSING", path))
        print("dry-run only; no checkpoint was written")
        print("PACT-CBEA rule module is parameter-free and requires no centralized Stage3 joint training.")
        return

    for path in checkpoint_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    merged = OrderedDict()
    for path in checkpoint_paths:
        state_dict = torch.load(path, map_location="cpu")
        merged = merge_dict(merged, state_dict)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "net_epoch1.pth")
    torch.save(merged, output_path)
    print("PACT-CBEA initialization checkpoint written to: %s" % output_path)
    print("PACT-CBEA rule module is parameter-free and requires no centralized Stage3 joint training.")
    print("PACTCBEARule has no trainable checkpoint keys by design.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compose local expert checkpoints for PACT-CBEA.",
    )
    parser.add_argument("--m1-ckpt", default=DEFAULT_M1)
    parser.add_argument("--m2-ckpt", default=DEFAULT_M2)
    parser.add_argument("--m3-ckpt", default=DEFAULT_M3)
    parser.add_argument("--m4-ckpt", default=DEFAULT_M4)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


if __name__ == "__main__":
    main()

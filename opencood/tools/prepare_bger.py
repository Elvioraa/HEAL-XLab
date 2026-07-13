"""Prepare BGER initialization checkpoints from HEAL expert checkpoints.

Two composition modes:

- ``--stage stage_a``: take the official HEAL Stage1 m1 checkpoint and write
  it as the init checkpoint for BGER stage_a training. bger_refine keys are
  absent on purpose; train.py loads with strict=False semantics via the
  missing-key tolerant loader, and bger_refine starts from its zero-conv
  identity initialization.

- ``--stage stage_b``: merge the m1 Stage1 checkpoint, the m2/m3/m4 HEAL
  Stage2 checkpoints, and the bger_refine weights trained in stage_a into a
  single init checkpoint for heterogeneous stage_b finetuning (or direct
  final inference).

A manifest with source paths, sizes, SHA256 hashes, and the current git
commit is written next to the output checkpoint.
"""

import argparse
from collections import OrderedDict
import hashlib
import json
import os
import subprocess
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.tools.heal_tools import merge_dict


DEFAULT_M1 = "opencood/logs/HEAL_m1_based/stage1/m1_base/net_epoch_bestval_at.pth"
DEFAULT_M2 = "opencood/logs/HEAL_m1_based/stage2/m2_alignto_m1/net_epoch_bestval_at.pth"
DEFAULT_M3 = "opencood/logs/HEAL_m1_based/stage2/m3_alignto_m1/net_epoch_bestval_at.pth"
DEFAULT_M4 = "opencood/logs/HEAL_m1_based/stage2/m4_alignto_m1/net_epoch_bestval_at.pth"
DEFAULT_STAGE_A_REFINE = "opencood/logs/BGER_v1/stage_a/net_epoch_bestval.pth"
DEFAULT_OUTPUT_DIR_A = "opencood/logs/BGER_v1/stage_a"
DEFAULT_OUTPUT_DIR_B = "opencood/logs/BGER_v1/stage_b"

BGER_REFINE_PREFIX = "bger_refine."


def main():
    args = parse_args()
    if args.stage == "stage_a":
        source_paths = {"m1": _resolve(args.m1_ckpt)}
        output_dir = _resolve(args.output_dir or DEFAULT_OUTPUT_DIR_A)
    else:
        source_paths = {
            "m1": _resolve(args.m1_ckpt),
            "m2": _resolve(args.m2_ckpt),
            "m3": _resolve(args.m3_ckpt),
            "m4": _resolve(args.m4_ckpt),
            "bger_refine": _resolve(args.stage_a_refine),
        }
        output_dir = _resolve(args.output_dir or DEFAULT_OUTPUT_DIR_B)
    output_path = os.path.join(output_dir, args.output_name)

    print("Preparing BGER %s initialization checkpoint..." % args.stage)
    for name, path in source_paths.items():
        print(" - %s: %s" % (name, path))
    print("Output checkpoint: %s" % output_path)

    if args.dry_run:
        for name, path in source_paths.items():
            status = "FOUND" if os.path.exists(path) else "MISSING"
            print("%s: %s" % (status, path))
        print("dry-run only; no checkpoint was written")
        return

    for name, path in source_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    if os.path.exists(output_path) and not args.force:
        raise FileExistsError(
            "%s already exists; pass --force to overwrite after verifying its source"
            % output_path
        )

    manifest = {
        "stage": args.stage,
        "source_checkpoints": {},
        "output_checkpoint": output_path,
        "git_commit": _git_commit(),
    }

    merged = OrderedDict()
    for name, path in source_paths.items():
        state_dict = _load_state_dict(path)
        if name == "bger_refine":
            refine_state = OrderedDict(
                (key, value) for key, value in state_dict.items()
                if key.startswith(BGER_REFINE_PREFIX)
            )
            if not refine_state:
                raise KeyError(
                    "%s does not contain any %s keys; was stage_a trained with "
                    "bger.enabled=true?" % (path, BGER_REFINE_PREFIX)
                )
            merged = merge_dict(merged, refine_state)
        elif name == "m1":
            _require_prefixes(path, state_dict, ("encoder_m1", "pyramid_backbone", "cls_head"))
            merged = merge_dict(merged, state_dict)
        else:
            _require_prefixes(path, state_dict, ("encoder_%s" % name,))
            branch_state = OrderedDict(
                (key, value) for key, value in state_dict.items()
                if key.startswith((
                    "encoder_%s" % name,
                    "backbone_%s" % name,
                    "aligner_%s" % name,
                ))
            )
            merged = merge_dict(merged, branch_state)
        manifest["source_checkpoints"][name] = {
            "path": path,
            "bytes": os.path.getsize(path),
            "sha256": _sha256(path),
        }

    os.makedirs(output_dir, exist_ok=True)
    torch.save(merged, output_path)
    manifest["output_checkpoint_bytes"] = os.path.getsize(output_path)
    manifest["output_checkpoint_sha256"] = _sha256(output_path)
    manifest_path = os.path.join(output_dir, "bger_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print("BGER initialization checkpoint written to: %s" % output_path)
    print("BGER source manifest written to: %s" % manifest_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compose BGER initialization checkpoints from HEAL experts.",
    )
    parser.add_argument("--stage", choices=("stage_a", "stage_b"), required=True)
    parser.add_argument("--m1-ckpt", default=DEFAULT_M1)
    parser.add_argument("--m2-ckpt", default=DEFAULT_M2)
    parser.add_argument("--m3-ckpt", default=DEFAULT_M3)
    parser.add_argument("--m4-ckpt", default=DEFAULT_M4)
    parser.add_argument("--stage-a-refine", default=DEFAULT_STAGE_A_REFINE,
                        help="stage_a checkpoint carrying trained bger_refine keys")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default="net_epoch1.pth")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _require_prefixes(path, state_dict, prefixes):
    missing = [
        prefix for prefix in prefixes
        if not any(key.startswith(prefix) for key in state_dict.keys())
    ]
    if missing:
        raise KeyError("%s missing expected module prefixes: %s" % (path, missing))


def _load_state_dict(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint is not a state_dict-like mapping: %s" % path)
    return OrderedDict((key[7:] if key.startswith("module.") else key, value)
                       for key, value in state.items())


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _resolve(path):
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(REPO_ROOT, path))


if __name__ == "__main__":
    main()

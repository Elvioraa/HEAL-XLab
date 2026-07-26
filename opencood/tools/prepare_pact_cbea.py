"""Prepare PACT-CBEA initialization checkpoint from local PACT expert branches."""

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

from opencood.models.sub_modules.pact_cbea_rule import PACTCBEARule
from opencood.tools.heal_tools import merge_dict


DEFAULT_M1 = "opencood/logs/PACT_CBEA_v1/stage1/m1_base/net_epoch_bestval.pth"
DEFAULT_M2 = "opencood/logs/PACT_CBEA_v1/stage2/m2_alignto_m1/net_epoch_bestval.pth"
DEFAULT_M3 = "opencood/logs/PACT_CBEA_v1/stage2/m3_alignto_m1/net_epoch_bestval.pth"
DEFAULT_M4 = "opencood/logs/PACT_CBEA_v1/stage2/m4_alignto_m1/net_epoch_bestval.pth"
DEFAULT_OUTPUT_DIR = "opencood/logs/PACT_CBEA_v1/rule_cbea"


def main():
    args = parse_args()
    checkpoint_paths = {
        "m1": _resolve(args.m1_ckpt),
        "m2": _resolve(args.m2_ckpt),
        "m3": _resolve(args.m3_ckpt),
        "m4": _resolve(args.m4_ckpt),
    }
    output_dir = _resolve(args.output_dir)
    output_path = os.path.join(output_dir, args.output_name)

    print("Preparing PACT-CBEA initialization checkpoint from local PACT expert branches...")
    for modality, path in checkpoint_paths.items():
        print(" - %s: %s" % (modality, path))
    print("Output checkpoint: %s" % output_path)

    rule = PACTCBEARule()
    if sum(param.numel() for param in rule.parameters()) != 0 or len(rule.state_dict()) != 0:
        raise RuntimeError("PACTCBEARule must remain parameter-free and checkpoint-free")

    if args.dry_run:
        for modality, path in checkpoint_paths.items():
            status = "FOUND" if os.path.exists(path) else "MISSING"
            print("%s: %s" % (status, path))
            if os.path.exists(path):
                _print_checkpoint_checks(modality, path)
        print("dry-run only; no checkpoint was written")
        print("PACT-CBEA rule module is parameter-free and requires no centralized Stage3 joint training.")
        return

    for modality, path in checkpoint_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        _validate_checkpoint(modality, path)

    if os.path.exists(output_path) and not args.force:
        raise FileExistsError(
            "%s already exists; pass --force to overwrite after verifying its source" % output_path
        )

    merged = OrderedDict()
    manifest = {
        "source_checkpoints": {},
        "output_checkpoint": output_path,
        "git_commit": _git_commit(),
        "pact_rule_parameter_count": 0,
        "pact_rule_checkpoint_keys": [],
    }
    # Official merge_dict(single, stage1) filters its FIRST argument, dropping
    # any key containing 'head_m'. That is meant for modality-specific
    # detection heads (cls_head_m4, ...) but also matches our
    # 'pact_cbea_evidence_head_mX.*' keys. Since the accumulator is passed as
    # that first argument on every iteration, each round silently discards the
    # evidence head accumulated in the previous round, leaving only the last
    # merged modality (m1). Back the evidence heads up here and restore them
    # once after the loop, so no round can filter them away.
    evidence_backup = OrderedDict()
    for modality in ("m2", "m3", "m4", "m1"):
        path = checkpoint_paths[modality]
        state_dict = _load_state_dict(path)
        merged = merge_dict(merged, state_dict)
        prefix = "pact_cbea_evidence_head_%s" % modality
        for key, value in state_dict.items():
            if key.startswith(prefix):
                evidence_backup[key] = value
        manifest["source_checkpoints"][modality] = {
            "path": path,
            "bytes": os.path.getsize(path),
            "sha256": _sha256(path),
            "required_prefix": "pact_cbea_evidence_head_%s" % modality,
        }

    merged = _restore_evidence_heads(merged, evidence_backup)
    _validate_merged_evidence_heads(merged, checkpoint_paths)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(merged, output_path)
    manifest["output_checkpoint_bytes"] = os.path.getsize(output_path)
    manifest["output_checkpoint_sha256"] = _sha256(output_path)
    manifest_path = os.path.join(output_dir, "pact_cbea_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print("PACT-CBEA initialization checkpoint written to: %s" % output_path)
    print("PACT-CBEA source manifest written to: %s" % manifest_path)
    print("PACT-CBEA rule module is parameter-free and requires no centralized Stage3 joint training.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compose PACT-CBEA local expert checkpoints.",
    )
    parser.add_argument("--m1-ckpt", default=DEFAULT_M1)
    parser.add_argument("--m2-ckpt", default=DEFAULT_M2)
    parser.add_argument("--m3-ckpt", default=DEFAULT_M3)
    parser.add_argument("--m4-ckpt", default=DEFAULT_M4)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="net_epoch1.pth")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _restore_evidence_heads(merged, evidence_backup):
    """Put every modality's evidence-head keys back after merge_dict filtering.

    Values come verbatim from each modality's own checkpoint, so this only
    restores what the official 'head_m' filter removed and never overwrites
    anything the merge legitimately produced.
    """
    for key, value in evidence_backup.items():
        merged[key] = value
    if evidence_backup:
        print("Restored %d evidence-head keys" % len(evidence_backup))
    return merged


def _validate_merged_evidence_heads(merged, checkpoint_paths):
    """Fail loudly if any modality's evidence head is missing after merging.

    Without this guard a checkpoint whose evidence heads were silently dropped
    still loads fine, but the CBEA rule then routes on randomly initialized
    evidence, producing meaningless alpha.
    """
    missing = []
    for modality in checkpoint_paths:
        prefix = "pact_cbea_evidence_head_%s" % modality
        source_keys = {
            key for key in _load_state_dict(checkpoint_paths[modality])
            if key.startswith(prefix)
        }
        merged_keys = {key for key in merged if key.startswith(prefix)}
        if not merged_keys:
            missing.append("%s: 0 keys in merged checkpoint" % modality)
        elif source_keys - merged_keys:
            missing.append(
                "%s: %d key(s) lost, e.g. %s"
                % (
                    modality,
                    len(source_keys - merged_keys),
                    sorted(source_keys - merged_keys)[0],
                )
            )
    if missing:
        raise RuntimeError(
            "merged checkpoint is missing evidence-head parameters:\n  %s"
            % "\n  ".join(missing)
        )
    for modality in sorted(checkpoint_paths):
        prefix = "pact_cbea_evidence_head_%s" % modality
        count = sum(1 for key in merged if key.startswith(prefix))
        loc = sum(
            1 for key in merged
            if key.startswith(prefix) and "localization" in key
        )
        print(
            "  %s evidence head: %d keys (%d localization)" % (modality, count, loc)
        )


def _print_checkpoint_checks(modality, path):
    try:
        state_dict = _load_state_dict(path)
    except Exception as exc:
        print("   CHECK FAILED: %s:%s" % (type(exc).__name__, exc))
        return
    prefix = "pact_cbea_evidence_head_%s" % modality
    has_head = any(key.startswith(prefix) for key in state_dict.keys())
    print("   evidence head prefix %s: %s" % (prefix, "FOUND" if has_head else "MISSING"))


def _validate_checkpoint(modality, path):
    state_dict = _load_state_dict(path)
    prefix = "pact_cbea_evidence_head_%s" % modality
    if not any(key.startswith(prefix) for key in state_dict.keys()):
        raise KeyError("%s does not contain local evidence head prefix %s" % (path, prefix))
    # m1 uses a parameter-free Identity aligner, so its checkpoint
    # legitimately contains no aligner_m1.* keys. Other modalities keep
    # strict aligner validation because their aligners are trainable.
    expected_model_prefixes = [
        "encoder_%s" % modality,
        "backbone_%s" % modality,
        prefix,
    ]
    if modality != "m1":
        expected_model_prefixes.append("aligner_%s" % modality)

    missing = [
        prefix_name
        for prefix_name in expected_model_prefixes
        if not any(key.startswith(prefix_name) for key in state_dict.keys())
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

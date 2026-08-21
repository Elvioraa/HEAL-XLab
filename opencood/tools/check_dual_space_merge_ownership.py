"""Read-only checkpoint ownership checker for Dual-Space final merges."""

import argparse
import sys

import torch

from opencood.tools.audit_dual_space_merge import (
    load_checkpoint_state,
    rebuild_expected_merge,
)


SHARED_PREFIXES = (
    "dual_space_shared_object_encoder.",
    "dual_space_shared_geometry_encoder.",
    "dual_space_shared_object_refiner.",
    "dual_space_shared_context_encoder.",
    "dual_space_shared_multiscale_fusion.",
    "dual_space_shared_scale_gate.",
    "dual_space_shared_quality_head.",
)
MODALITIES = ("m2", "m3", "m4")


def check_merge_ownership(m1, stage2, merged, preview_limit=10):
    """Compare merged Dual-Space tensors with production ownership output."""
    if set(stage2) != set(MODALITIES):
        raise ValueError("stage2 must contain m2, m3, and m4")
    expected = rebuild_expected_merge(m1, stage2)
    groups = {"m1_shared": SHARED_PREFIXES}
    groups.update(
        {
            modality: (
                "dual_space_object_adapter_%s." % modality,
                "dual_space_context_adapter_%s." % modality,
            )
            for modality in MODALITIES
        }
    )
    report = {}
    failed = False
    for name, prefixes in groups.items():
        keys = sorted(key for key in expected if key.startswith(prefixes))
        identical = []
        different = []
        missing = []
        for key in keys:
            if key not in merged:
                missing.append(key)
            elif _tensor_identical(expected[key], merged[key]):
                identical.append(key)
            else:
                different.append(key)
        failed = failed or bool(different or missing)
        report[name] = {
            "expected": len(keys),
            "identical": len(identical),
            "different": len(different),
            "missing": len(missing),
            "different_keys": different[:preview_limit],
            "missing_keys": missing[:preview_limit],
        }

    expected_dual = {key for key in expected if key.startswith("dual_space_")}
    merged_dual = {key for key in merged if key.startswith("dual_space_")}
    unexpected = sorted(merged_dual - expected_dual)
    failed = failed or bool(unexpected)
    report["unexpected"] = unexpected[:preview_limit]
    report["status"] = "FAIL" if failed else "PASS"
    return report


def print_report(report):
    """Print the stable human-readable ownership summary."""
    for name in ("m1_shared",) + MODALITIES:
        values = report[name]
        print("[%s]" % name)
        for key in ("expected", "identical", "different", "missing"):
            print("%s=%d" % (key, values[key]))
        for key in values["different_keys"]:
            print("different_key=%s" % key)
        for key in values["missing_keys"]:
            print("missing_key=%s" % key)
    for key in report["unexpected"]:
        print("unexpected_key=%s" % key)
    print("MERGE_OWNERSHIP_RESULT: %s" % report["status"])


def build_parser():
    parser = argparse.ArgumentParser(
        description="Check Dual-Space merge ownership without modifying files"
    )
    parser.add_argument("--m1", required=True)
    parser.add_argument("--m2", required=True)
    parser.add_argument("--m3", required=True)
    parser.add_argument("--m4", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--preview-limit", type=int, default=10)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.preview_limit < 1:
        raise ValueError("preview-limit must be positive")
    report = check_merge_ownership(
        load_checkpoint_state(args.m1),
        {
            "m2": load_checkpoint_state(args.m2),
            "m3": load_checkpoint_state(args.m3),
            "m4": load_checkpoint_state(args.m4),
        },
        load_checkpoint_state(args.merged),
        preview_limit=args.preview_limit,
    )
    print_report(report)
    return 0 if report["status"] == "PASS" else 1


def _tensor_identical(first, second):
    return bool(
        torch.is_tensor(first)
        and torch.is_tensor(second)
        and first.shape == second.shape
        and first.dtype == second.dtype
        and torch.equal(first, second)
    )


if __name__ == "__main__":
    sys.exit(main())

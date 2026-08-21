"""Read-only ownership audit for merged Dual-Space HEAL checkpoints."""

import argparse
from collections import OrderedDict
from contextlib import redirect_stdout
import io
import json
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.dual_space_config import (
    resolve_v6_residual_safe_config,
    validate_dual_space_config,
)
from opencood.models.sub_modules.dual_space_object import (
    SHARED_DUAL_SPACE_PREFIXES,
    install_dual_space_modules,
)
from opencood.tools.heal_tools import (
    apply_dual_space_merge_ownership,
    merge_dict,
)


MODALITIES = ("m2", "m3", "m4")


class DualSpaceMergeAuditError(RuntimeError):
    """Raised when a hard merge/config ownership check fails."""


def rebuild_expected_merge(stage1_state, stage2_states):
    """Rebuild the expected state by calling the production merge ownership."""
    ordered = [stage2_states[name] for name in MODALITIES] + [stage1_state]
    merged = OrderedDict()
    with redirect_stdout(io.StringIO()):
        for state in ordered:
            merged = merge_dict(merged, state)
        merged = apply_dual_space_merge_ownership(merged, ordered)
    return merged


def audit_dual_space_merge(
    stage1_state,
    stage2_states,
    merged_state,
    stage1_config,
    stage2_configs,
    merged_config,
    expected_dual_space_keys=None,
):
    """Audit configs, frozen/shared state, explicit owners, and full merge output."""
    _validate_state_dict(stage1_state, "Stage1")
    _validate_state_dict(merged_state, "merged")
    if set(stage2_states) != set(MODALITIES):
        raise DualSpaceMergeAuditError("Stage2 states must contain m2, m3, and m4")
    if set(stage2_configs) != set(MODALITIES):
        raise DualSpaceMergeAuditError("Stage2 configs must contain m2, m3, and m4")
    for modality in MODALITIES:
        _validate_state_dict(stage2_states[modality], "Stage2 %s" % modality)

    stage1_dual = _dual_config(stage1_config, "Stage1")
    merged_dual = _dual_config(merged_config, "merged inference")
    stage2_dual = {
        modality: _dual_config(stage2_configs[modality], "Stage2 %s" % modality)
        for modality in MODALITIES
    }
    _validate_config_contracts(
        stage1_config, stage1_dual, stage2_configs, stage2_dual,
        merged_config, merged_dual,
    )
    if expected_dual_space_keys is None:
        expected_dual_space_keys = expected_dual_space_state_keys(merged_config)
    expected_dual_space_keys = set(expected_dual_space_keys)
    if not expected_dual_space_keys or not all(
        key.startswith("dual_space_") for key in expected_dual_space_keys
    ):
        raise DualSpaceMergeAuditError(
            "expected Dual-Space schema must contain only dual_space_ keys"
        )

    required_shared_prefixes = _required_shared_prefixes(stage1_dual)
    stage1_shared_keys = {
        key for key in stage1_state if key.startswith(SHARED_DUAL_SPACE_PREFIXES)
    }
    expected_shared_keys = {
        key for key in expected_dual_space_keys
        if key.startswith(SHARED_DUAL_SPACE_PREFIXES)
    }
    if stage1_shared_keys != expected_shared_keys:
        raise DualSpaceMergeAuditError(
            "Stage1 shared key mismatch; missing=%s unexpected=%s"
            % (
                sorted(expected_shared_keys - stage1_shared_keys),
                sorted(stage1_shared_keys - expected_shared_keys),
            )
        )
    for prefix in required_shared_prefixes:
        if not any(key.startswith(prefix) for key in stage1_shared_keys):
            raise DualSpaceMergeAuditError(
                "Stage1 is missing required shared prefix %s" % prefix
            )
    for modality in MODALITIES:
        source = stage2_states[modality]
        source_shared = {
            key for key in source if key.startswith(SHARED_DUAL_SPACE_PREFIXES)
        }
        if source_shared != stage1_shared_keys:
            raise DualSpaceMergeAuditError(
                "Stage2 %s shared key set differs from Stage1" % modality
            )
        for key in sorted(stage1_shared_keys):
            _assert_tensor_exact(stage1_state[key], source[key], key, "Stage2 %s frozen shared" % modality)

    expected = rebuild_expected_merge(stage1_state, stage2_states)
    expected_keys = set(expected)
    merged_keys = set(merged_state)
    missing_all = sorted(expected_keys - merged_keys)
    unexpected_all = sorted(merged_keys - expected_keys)
    if missing_all or unexpected_all:
        raise DualSpaceMergeAuditError(
            "merged checkpoint key mismatch; missing=%s unexpected=%s"
            % (missing_all, unexpected_all)
        )

    merged_ds = {key for key in merged_state if key.startswith("dual_space_")}
    missing_ds = sorted(expected_dual_space_keys - merged_ds)
    unexpected_ds = sorted(merged_ds - expected_dual_space_keys)
    if missing_ds or unexpected_ds:
        raise DualSpaceMergeAuditError(
            "Dual-Space key mismatch; missing=%s unexpected=%s"
            % (missing_ds, unexpected_ds)
        )

    for key in sorted(expected):
        _assert_tensor_exact(expected[key], merged_state[key], key, "production merge")
    for key in sorted(stage1_shared_keys):
        _assert_tensor_exact(stage1_state[key], merged_state[key], key, "Stage1 shared ownership")

    adapter_counts = {}
    for modality in MODALITIES:
        source = stage2_states[modality]
        prefixes = ["dual_space_object_adapter_%s." % modality]
        if stage1_dual["multi_scale"]["enabled"]:
            prefixes.append("dual_space_context_adapter_%s." % modality)
        owned_keys = [key for key in source if key.startswith(tuple(prefixes))]
        expected_owned_keys = {
            key for key in expected_dual_space_keys
            if key.startswith(tuple(prefixes))
        }
        if set(owned_keys) != expected_owned_keys:
            raise DualSpaceMergeAuditError(
                "Stage2 %s adapter key mismatch; missing=%s unexpected=%s"
                % (
                    modality,
                    sorted(expected_owned_keys - set(owned_keys)),
                    sorted(set(owned_keys) - expected_owned_keys),
                )
            )
        for prefix in prefixes:
            if not any(key.startswith(prefix) for key in owned_keys):
                raise DualSpaceMergeAuditError(
                    "Stage2 %s is missing required adapter prefix %s"
                    % (modality, prefix)
                )
        for key in sorted(owned_keys):
            _assert_tensor_exact(source[key], merged_state[key], key, "Stage2 %s adapter ownership" % modality)
        adapter_counts[modality] = {
            "owned_tensor_count": len(owned_keys),
            "training_change": _adapter_change_diagnostic(
                stage1_state, source, tuple(prefixes)
            ),
        }

    checkpoint_profile = stage1_dual.get(
        "experiment_profile", stage1_dual["version"]
    )
    inference_profile = merged_dual.get(
        "experiment_profile", merged_dual["version"]
    )
    return {
        "status": "PASS",
        "checkpoint_profile": checkpoint_profile,
        "inference_profile": inference_profile,
        "checks": [
            "profile consistency",
            "Stage1 shared completeness",
            "Stage2 frozen shared identity",
            "m2 ownership",
            "m3 ownership",
            "m4 ownership",
            "Dual-Space key completeness",
            "HEAL production merge contract",
            "merged inference checkpoint contract",
        ],
        "expected_key_count": len(expected_keys),
        "dual_space_key_count": len(expected_dual_space_keys),
        "missing_dual_space_keys": 0,
        "unexpected_dual_space_keys": 0,
        "adapters": adapter_counts,
    }


def expected_dual_space_state_keys(merged_config):
    """Build the configured DS-only state schema without a detector forward."""
    try:
        args = merged_config["model"]["args"]
        modalities = [
            modality for modality in ("m1", "m2", "m3", "m4")
            if modality in args
        ]
        host = nn.Module()
        host.modality_name_list = modalities
        with redirect_stdout(io.StringIO()):
            install_dual_space_modules(host, args)
        return {
            key for key in host.state_dict() if key.startswith("dual_space_")
        }
    except Exception as error:
        raise DualSpaceMergeAuditError(
            "could not derive configured Dual-Space state schema: %s" % error
        ) from error


def load_checkpoint_state(path):
    """Load a checkpoint on CPU without mutating or rewriting it."""
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict) and "state_dict" in value and isinstance(
        value["state_dict"], dict
    ):
        value = value["state_dict"]
    if not isinstance(value, dict):
        raise TypeError("checkpoint must contain a state-dict mapping: %s" % path)
    state = OrderedDict(value)
    _validate_state_dict(state, path)
    return state


def _validate_config_contracts(
    stage1_config,
    stage1_dual,
    stage2_configs,
    stage2_dual,
    merged_config,
    merged_dual,
):
    if stage1_dual["mode"] != "stage1_anchor":
        raise DualSpaceMergeAuditError("Stage1 config must use mode=stage1_anchor")
    if stage1_dual["allow_untrained_initialization"] is not True:
        raise DualSpaceMergeAuditError("Stage1 config must permit its documented initialization")
    source_profile = stage1_dual.get("experiment_profile", stage1_dual["version"])
    source_signature = _architecture_signature(stage1_dual)
    for modality in MODALITIES:
        dual = stage2_dual[modality]
        profile = dual.get("experiment_profile", dual["version"])
        if profile != source_profile or dual["version"] != stage1_dual["version"]:
            raise DualSpaceMergeAuditError("Stage2 %s profile mismatch" % modality)
        if dual["mode"] != "stage2_adapt" or dual.get("active_modality") != modality:
            raise DualSpaceMergeAuditError("Stage2 %s mode/active_modality mismatch" % modality)
        if dual["allow_untrained_initialization"] is not False:
            raise DualSpaceMergeAuditError("Stage2 %s must require a checkpoint" % modality)
        if _architecture_signature(dual) != source_signature:
            raise DualSpaceMergeAuditError("Stage2 %s architecture mismatch" % modality)
    if merged_dual["mode"] != "inference":
        raise DualSpaceMergeAuditError("merged config must use mode=inference")
    if merged_dual["allow_untrained_initialization"] is not False:
        raise DualSpaceMergeAuditError("merged inference must require a checkpoint")
    inference_profile = merged_dual.get(
        "experiment_profile", merged_dual["version"]
    )
    same_profile = (
        inference_profile == source_profile
        and merged_dual["version"] == stage1_dual["version"]
    )
    v3_to_v4 = (
        source_profile == "ds_v3"
        and inference_profile == "ds_v4"
        and merged_dual["version"] == "ds_v4"
        and merged_dual["remote_proposal_rescue"]["enabled"] is True
    )
    if not same_profile and not v3_to_v4:
        raise DualSpaceMergeAuditError("merged inference profile mismatch")
    if _architecture_signature(merged_dual) != source_signature:
        raise DualSpaceMergeAuditError("merged inference architecture mismatch")
    merged_v6 = resolve_v6_residual_safe_config(merged_dual)
    for modality in MODALITIES:
        if resolve_v6_residual_safe_config(stage2_dual[modality]) != merged_v6:
            raise DualSpaceMergeAuditError(
                "merged inference lost Stage2 %s V6 residual-safe settings"
                % modality
            )
    merged_modalities = {
        key for key in merged_config["model"]["args"] if key in ("m1", "m2", "m3", "m4")
    }
    if merged_modalities != {"m1", "m2", "m3", "m4"}:
        raise DualSpaceMergeAuditError("merged inference must define m1/m2/m3/m4")


def _architecture_signature(dual):
    ignored = {
        "version", "experiment_profile", "mode", "active_modality",
        "allow_untrained_initialization", "remote_proposal_rescue",
        "diagnostics", "v5_quality_safe", "v6_residual_safe", "report_stats",
    }
    return {key: value for key, value in dual.items() if key not in ignored}


def _required_shared_prefixes(dual):
    prefixes = [
        "dual_space_shared_object_encoder.",
        "dual_space_shared_geometry_encoder.",
        "dual_space_shared_object_refiner.",
    ]
    if dual["multi_scale"]["enabled"]:
        prefixes.append("dual_space_shared_context_encoder.")
        if dual["multi_scale"]["fusion"] == "adaptive_gate":
            prefixes.append("dual_space_shared_scale_gate.")
        else:
            prefixes.append("dual_space_shared_multiscale_fusion.")
    if dual["quality"]["enabled"]:
        prefixes.append("dual_space_shared_quality_head.")
    return tuple(prefixes)


def _dual_config(config, label):
    try:
        dual = config["model"]["args"]["dual_space"]
        validate_dual_space_config(dual)
    except Exception as error:
        raise DualSpaceMergeAuditError(
            "%s config is invalid: %s" % (label, error)
        ) from error
    return dual


def _validate_state_dict(state, label):
    if not isinstance(state, dict):
        raise TypeError("%s state must be a mapping" % label)
    non_tensors = [key for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensors:
        raise TypeError("%s contains non-tensor state values: %s" % (label, non_tensors))


def _assert_tensor_exact(expected, actual, key, owner):
    if tuple(expected.shape) != tuple(actual.shape):
        raise DualSpaceMergeAuditError("%s shape mismatch for %s" % (owner, key))
    if expected.dtype != actual.dtype:
        raise DualSpaceMergeAuditError("%s dtype mismatch for %s" % (owner, key))
    if not torch.equal(expected, actual):
        raise DualSpaceMergeAuditError("%s value mismatch for %s" % (owner, key))


def _adapter_change_diagnostic(stage1, stage2, prefixes):
    comparable = sorted(
        key for key in stage2 if key.startswith(prefixes) and key in stage1
    )
    changed = unchanged = 0
    max_abs_diff = 0.0
    for key in comparable:
        first = stage1[key]
        second = stage2[key]
        if first.shape != second.shape:
            changed += 1
            continue
        if torch.equal(first, second):
            unchanged += 1
            continue
        changed += 1
        if first.numel() and torch.is_floating_point(first) and torch.is_floating_point(second):
            difference = (first.to(torch.float64) - second.to(torch.float64)).abs().max()
            max_abs_diff = max(max_abs_diff, float(difference.item()))
    return {
        "comparable_tensor_count": len(comparable),
        "changed_tensor_count": changed,
        "unchanged_tensor_count": unchanged,
        "max_abs_diff": max_abs_diff,
        "warning_only": True,
    }


def _write_json(path, report):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Audit a Dual-Space final merge")
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage1-config", required=True)
    for modality in MODALITIES:
        parser.add_argument("--stage2-%s-checkpoint" % modality, required=True)
        parser.add_argument("--stage2-%s-config" % modality, required=True)
    parser.add_argument("--merged-checkpoint", required=True)
    parser.add_argument("--merged-config", required=True)
    parser.add_argument("--json-out")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = audit_dual_space_merge(
            load_checkpoint_state(args.stage1_checkpoint),
            {
                modality: load_checkpoint_state(
                    getattr(args, "stage2_%s_checkpoint" % modality)
                )
                for modality in MODALITIES
            },
            load_checkpoint_state(args.merged_checkpoint),
            load_yaml(args.stage1_config, None),
            {
                modality: load_yaml(
                    getattr(args, "stage2_%s_config" % modality), None
                )
                for modality in MODALITIES
            },
            load_yaml(args.merged_config, None),
        )
    except Exception as error:
        report = {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print("[FAIL] %s: %s" % (type(error).__name__, error))
        if args.json_out:
            _write_json(args.json_out, report)
        return 1
    for check in report["checks"]:
        print("[PASS] %s" % check)
    print("ALL DUAL-SPACE MERGE CHECKS PASSED")
    if args.json_out:
        _write_json(args.json_out, report)
        print("JSON summary: %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Synthetic hard-failure coverage for the read-only Dual-Space merge audit."""

import copy
from collections import OrderedDict
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.audit_dual_space_merge import (
    DualSpaceMergeAuditError,
    audit_dual_space_merge,
    expected_dual_space_state_keys,
    rebuild_expected_merge,
)
from opencood.tools.dual_space_smoke_common import make_dual_config


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def configs():
    def wrapped(mode, active=None, modalities=()):
        dual = make_dual_config(
            mode=mode, active_modality=active, multi=True, quality=True
        )
        args = {name: {} for name in modalities}
        args["dual_space"] = dual
        return {"model": {"args": args}}

    return (
        wrapped("stage1_anchor", modalities=("m1",)),
        {
            modality: wrapped(
                "stage2_adapt", active=modality, modalities=(modality,)
            )
            for modality in ("m2", "m3", "m4")
        },
        wrapped("inference", modalities=("m1", "m2", "m3", "m4")),
    )


def states():
    shared = OrderedDict(
        (
            ("dual_space_shared_object_encoder.weight", torch.tensor([1.0])),
            ("dual_space_shared_geometry_encoder.weight", torch.tensor([2.0])),
            ("dual_space_shared_object_refiner.weight", torch.tensor([3.0])),
            ("dual_space_shared_context_encoder.weight", torch.tensor([4.0])),
            ("dual_space_shared_multiscale_fusion.weight", torch.tensor([5.0])),
            ("dual_space_shared_quality_head.weight", torch.tensor([6.0])),
        )
    )
    stage1 = OrderedDict(shared)
    stage1["encoder_m1.weight"] = torch.tensor([11.0])
    stage1["pyramid_backbone.shared.weight"] = torch.tensor([12.0])
    stage2 = {}
    for index, modality in enumerate(("m2", "m3", "m4"), start=2):
        state = OrderedDict(shared)
        state["dual_space_object_adapter_%s.weight" % modality] = torch.tensor(
            [float(index * 10 + 1)]
        )
        state["dual_space_context_adapter_%s.weight" % modality] = torch.tensor(
            [float(index * 10 + 2)]
        )
        state["encoder_%s.weight" % modality] = torch.tensor([float(index * 10 + 3)])
        state["aligner_%s.weight" % modality] = torch.tensor([float(index * 10 + 4)])
        stage2[modality] = state
    return stage1, stage2, rebuild_expected_merge(stage1, stage2)


def run_audit(stage1=None, stage2=None, merged=None, config_set=None):
    original_stage1, original_stage2, original_merged = states()
    expected_dual_space_keys = {
        key for key in original_merged if key.startswith("dual_space_")
    }
    stage1 = original_stage1 if stage1 is None else stage1
    stage2 = original_stage2 if stage2 is None else stage2
    merged = original_merged if merged is None else merged
    stage1_config, stage2_configs, merged_config = configs()
    if config_set is not None:
        stage1_config, stage2_configs, merged_config = config_set
    return audit_dual_space_merge(
        stage1, stage2, merged,
        stage1_config, stage2_configs, merged_config,
        expected_dual_space_keys=expected_dual_space_keys,
    )


def assert_rejected(
    stage1=None, stage2=None, merged=None, config_set=None, expected_text=""
):
    try:
        run_audit(
            stage1=stage1,
            stage2=stage2,
            merged=merged,
            config_set=config_set,
        )
    except DualSpaceMergeAuditError as error:
        assert expected_text in str(error), str(error)
    else:
        raise AssertionError("merge audit accepted an injected ownership defect")


@test("correct synthetic ownership passes")
def test_correct():
    report = run_audit()
    assert report["status"] == "PASS"
    assert report["missing_dual_space_keys"] == 0
    assert report["unexpected_dual_space_keys"] == 0


@test("m3 adapter from the wrong source fails")
def test_wrong_m3():
    _, _, merged = states()
    merged = copy.deepcopy(merged)
    merged["dual_space_object_adapter_m3.weight"] = merged[
        "dual_space_object_adapter_m2.weight"
    ].clone()
    assert_rejected(merged=merged, expected_text="value mismatch")


@test("shared refiner overwritten after Stage1 fails")
def test_wrong_shared_refiner():
    _, _, merged = states()
    merged = copy.deepcopy(merged)
    merged["dual_space_shared_object_refiner.weight"] = torch.tensor([99.0])
    assert_rejected(merged=merged, expected_text="value mismatch")


@test("missing Dual-Space key fails")
def test_missing_ds_key():
    _, _, merged = states()
    merged = copy.deepcopy(merged)
    del merged["dual_space_context_adapter_m4.weight"]
    assert_rejected(merged=merged, expected_text="key mismatch")


@test("a DS key missing from every checkpoint still fails the model schema")
def test_globally_missing_ds_key():
    stage1, stage2, merged = states()
    key = "dual_space_shared_object_refiner.weight"
    del stage1[key]
    del merged[key]
    for state in stage2.values():
        del state[key]
    assert_rejected(
        stage1=stage1,
        stage2=stage2,
        merged=merged,
        expected_text="Stage1 shared key mismatch",
    )


@test("profile mismatch fails")
def test_profile_mismatch():
    stage1_config, stage2_configs, merged_config = configs()
    stage2_configs = copy.deepcopy(stage2_configs)
    stage2_configs["m3"]["model"]["args"]["dual_space"][
        "experiment_profile"
    ] = "wrong_profile"
    assert_rejected(
        config_set=(stage1_config, stage2_configs, merged_config),
        expected_text="profile mismatch",
    )


@test("yaw residual contract mismatch fails architecture audit")
def test_yaw_mode_mismatch():
    stage1_config, stage2_configs, merged_config = configs()
    stage2_configs = copy.deepcopy(stage2_configs)
    stage2_configs["m3"]["model"]["args"]["dual_space"]["refiner"][
        "yaw_mode"
    ] = "sin_cos_centered"
    assert_rejected(
        config_set=(stage1_config, stage2_configs, merged_config),
        expected_text="architecture mismatch",
    )


@test("formal DS-V3 config derives the complete state schema")
def test_formal_config_schema():
    path = os.path.join(
        REPO_ROOT,
        "opencood", "hypes_yaml", "HEAL_XLab_v4_DUAL_SPACE",
        "DS_V3", "merged_infer.yaml",
    )
    keys = expected_dual_space_state_keys(load_yaml(path, None))
    for prefix in (
        "dual_space_shared_object_encoder.",
        "dual_space_shared_geometry_encoder.",
        "dual_space_shared_object_refiner.",
        "dual_space_shared_context_encoder.",
        "dual_space_shared_multiscale_fusion.",
        "dual_space_shared_quality_head.",
        "dual_space_object_adapter_m2.",
        "dual_space_context_adapter_m4.",
    ):
        assert any(key.startswith(prefix) for key in keys), prefix


def main():
    passed = 0
    for name, function in TESTS:
        try:
            function()
        except Exception as error:
            print("[FAIL] %s: %s: %s" % (name, type(error).__name__, error))
        else:
            passed += 1
            print("[PASS] %s" % name)
    print("RESULT: %d/%d PASS" % (passed, len(TESTS)))
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())

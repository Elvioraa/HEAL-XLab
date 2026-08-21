"""CPU smoke tests for the production merge_final extension preflight."""

import copy
from collections import OrderedDict
import os
import sys
import tempfile

import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.tools.heal_tools import (
    DualSpaceMergeConfigError,
    merge_and_save_final,
    validate_dual_space_merge_config_contract,
)


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def v5(enabled=True):
    return {
        "enabled": bool(enabled),
        "valid_target_mask": {"enabled": True},
        "loss_balance": {
            "enabled": True,
            "max_quality_to_detection_ratio": 0.25,
        },
        "ranking": {"enabled": False},
    }


def v6(enabled=True):
    return {
        "enabled": bool(enabled),
        "object": {
            "enabled": True,
            "max_residual_ratio": 0.25,
            "residual_scale": 1.0,
            "eps": 1.0e-6,
        },
        "context": {
            "enabled": True,
            "max_residual_ratio": 0.25,
            "residual_scale": 1.0,
            "eps": 1.0e-6,
        },
    }


def config(v5_config=None, v6_config=None, legacy_dual=False):
    args = {}
    if legacy_dual or v5_config is not None or v6_config is not None:
        dual = {"enabled": True, "mode": "stage2_adapt"}
        if v5_config is not None:
            dual["v5_quality_safe"] = copy.deepcopy(v5_config)
        if v6_config is not None:
            dual["v6_residual_safe"] = copy.deepcopy(v6_config)
        args["dual_space"] = dual
    return {"model": {"args": args}}


def write_yaml(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def layout(stage_configs, final_config="missing", checkpoints=False):
    context = tempfile.TemporaryDirectory()
    root = context.name
    stage_dirs = []
    for index, modality in enumerate(("m2", "m3", "m4", "m1")):
        directory = os.path.join(root, modality)
        os.makedirs(directory)
        stage_dirs.append(directory)
        if modality in stage_configs:
            write_yaml(os.path.join(directory, "config.yaml"), stage_configs[modality])
        if checkpoints:
            torch.save(
                OrderedDict((("%s.weight" % modality, torch.tensor([index + 1.0])),)),
                os.path.join(directory, "net_epoch1.pth"),
            )
    final_dir = os.path.join(root, "final")
    os.makedirs(final_dir)
    if final_config != "missing":
        write_yaml(os.path.join(final_dir, "config.yaml"), final_config)
    return context, stage_dirs, final_dir


def validate(stage_configs, final_config="missing"):
    context, stage_dirs, final_dir = layout(stage_configs, final_config)
    try:
        return validate_dual_space_merge_config_contract(stage_dirs, final_dir)
    finally:
        context.cleanup()


def rejected(stage_configs, final_config="missing", expected=""):
    try:
        validate(stage_configs, final_config)
    except DualSpaceMergeConfigError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError("merge config contract accepted an invalid fixture")


@test("A legacy HEAL merge remains unchanged without extensions")
def test_legacy_heal_merge():
    configs = {name: config() for name in ("m2", "m3", "m4")}
    context, stage_dirs, final_dir = layout(configs, checkpoints=True)
    try:
        merge_and_save_final(stage_dirs, final_dir)
        output = torch.load(os.path.join(final_dir, "net_epoch1.pth"))
        assert set(output) == {"m1.weight", "m2.weight", "m3.weight", "m4.weight"}
    finally:
        context.cleanup()


@test("legacy DS-V3 with missing extension blocks is a no-op")
def test_legacy_ds_v3():
    result = validate({name: config(legacy_dual=True) for name in ("m2", "m3", "m4")})
    assert result == {"v5_quality_safe": False, "v6_residual_safe": False}


@test("B explicitly disabled extensions are a no-op")
def test_disabled():
    disabled = config(v5(False), v6(False))
    result = validate({name: disabled for name in ("m2", "m3", "m4")})
    assert result == {"v5_quality_safe": False, "v6_residual_safe": False}


@test("C identical Stage2 V5 with final V5 disabled passes")
def test_v5_pass():
    stages = {name: config(v5()) for name in ("m2", "m3", "m4")}
    result = validate(stages, config(v5(False)))
    assert result == {"v5_quality_safe": True, "v6_residual_safe": False}


@test("D inconsistent Stage2 V5 enabled states fail")
def test_v5_enabled_mismatch():
    stages = {"m2": config(v5()), "m3": config(v5(False)), "m4": config(v5())}
    rejected(stages, expected="V5 quality-safe configuration mismatch")


@test("E inconsistent Stage2 V5 functional settings fail")
def test_v5_functional_mismatch():
    changed = v5()
    changed["loss_balance"]["max_quality_to_detection_ratio"] = 0.5
    stages = {"m2": config(v5()), "m3": config(changed), "m4": config(v5())}
    rejected(stages, expected="V5 quality-safe configuration mismatch")


@test("F identical Stage2 and final V6 passes")
def test_v6_pass():
    stages = {name: config(v6_config=v6()) for name in ("m2", "m3", "m4")}
    result = validate(stages, config(v6_config=v6()))
    assert result == {"v5_quality_safe": False, "v6_residual_safe": True}


@test("G Stage2 V6 with final V6 disabled fails")
def test_v6_final_disabled():
    stages = {name: config(v6_config=v6()) for name in ("m2", "m3", "m4")}
    rejected(stages, config(v6_config=v6(False)), "disabled in final")


@test("H Stage2 V6 with final V6 block missing fails")
def test_v6_final_block_missing():
    stages = {name: config(v6_config=v6()) for name in ("m2", "m3", "m4")}
    rejected(stages, config(legacy_dual=True), "disabled in final")


@test("I Stage2 V6 with final config.yaml missing fails")
def test_v6_final_config_missing():
    stages = {name: config(v6_config=v6()) for name in ("m2", "m3", "m4")}
    rejected(stages, expected="final inference config.yaml is missing")


@test("J Stage2 and final V6 functional mismatch fails")
def test_v6_final_mismatch():
    stages = {name: config(v6_config=v6()) for name in ("m2", "m3", "m4")}
    changed = v6()
    changed["context"]["residual_scale"] = 0.5
    rejected(stages, config(v6_config=changed), "stage2/m2 and final_infer")


@test("K Stage2 V6 functional mismatch fails")
def test_v6_stage2_mismatch():
    changed = v6()
    changed["object"]["max_residual_ratio"] = 0.5
    stages = {
        "m2": config(v6_config=v6()),
        "m3": config(v6_config=changed),
        "m4": config(v6_config=v6()),
    }
    rejected(stages, config(v6_config=v6()), "V6 residual-safe configuration mismatch")


@test("L combined V5 plus V6 lifecycle passes")
def test_v5_v6_pass():
    stages = {name: config(v5(), v6()) for name in ("m2", "m3", "m4")}
    result = validate(stages, config(v5(False), v6()))
    assert result == {"v5_quality_safe": True, "v6_residual_safe": True}


@test("M invalid config fails before checkpoint lookup or write")
def test_fail_before_write():
    stages = {name: config(v6_config=v6()) for name in ("m2", "m3", "m4")}
    context, stage_dirs, final_dir = layout(stages, config(v6_config=v6(False)))
    output = os.path.join(final_dir, "net_epoch1.pth")
    try:
        try:
            merge_and_save_final(stage_dirs, final_dir)
        except DualSpaceMergeConfigError as error:
            assert "disabled in final" in str(error)
        else:
            raise AssertionError("invalid production merge unexpectedly succeeded")
        assert not os.path.exists(output)
    finally:
        context.cleanup()


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

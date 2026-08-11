"""Temporary-directory smoke tests for ``prepare_dual_space_stage2``."""

import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.prepare_dual_space_stage2 import (
    find_unique_stage1_best,
    prepare_dual_space_stage2,
    sha256_file,
)


PROFILE_DIR = os.path.join(
    REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v4_DUAL_SPACE", "DS_V1"
)
PROFILE_V1_1_DIR = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "HEAL_XLab_v4_DUAL_SPACE",
    "DS_V1_1",
)
TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def write_checkpoint(directory, name, payload=b"synthetic checkpoint"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "wb") as stream:
        stream.write(payload)
    return path


def write_stage1_config(directory, profile_dir=PROFILE_DIR):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "config.yaml")
    shutil.copy2(os.path.join(profile_dir, "stage1_m1.yaml"), path)
    return path


@test("Stage2 prepare creates verified independent m2/m3/m4 seeds")
def test_success():
    with tempfile.TemporaryDirectory() as directory:
        stage1 = os.path.join(directory, "stage1")
        best = write_checkpoint(stage1, "net_epoch_bestval_at12.pth")
        write_stage1_config(stage1)
        stage2 = os.path.join(directory, "stage2")
        summary = prepare_dual_space_stage2(PROFILE_DIR, stage1, stage2)
        assert summary["profile"] == "ds_v1"
        assert summary["yaw_mode"] == "sin_cos"
        assert summary["sha256"] == sha256_file(best)
        assert sha256_file(os.path.join(stage2, "net_epoch1.pth")) == summary["sha256"]
        for modality in ("m2", "m3", "m4"):
            item = summary["modalities"][modality]
            assert item["checkpoint_method"] in ("symlink", "copy")
            assert sha256_file(item["checkpoint"]) == summary["sha256"]
            config = load_yaml(item["config"], None)
            dual = config["model"]["args"]["dual_space"]
            assert dual["mode"] == "stage2_adapt"
            assert dual["active_modality"] == modality
            assert dual["allow_untrained_initialization"] is False


@test("missing Stage1 bestval checkpoint fails loudly")
def test_missing_best():
    with tempfile.TemporaryDirectory() as directory:
        try:
            find_unique_stage1_best(directory)
        except RuntimeError as error:
            assert "found 0" in str(error)
        else:
            raise AssertionError("missing bestval checkpoint was accepted")


@test("multiple Stage1 bestval checkpoints fail loudly")
def test_multiple_best():
    with tempfile.TemporaryDirectory() as directory:
        write_checkpoint(directory, "net_epoch_bestval_at1.pth", b"one")
        write_checkpoint(directory, "net_epoch_bestval_at2.pth", b"two")
        try:
            find_unique_stage1_best(directory)
        except RuntimeError as error:
            assert "found 2" in str(error)
        else:
            raise AssertionError("multiple bestval checkpoints were accepted")


@test("existing training output is never overwritten")
def test_existing_output_guard():
    with tempfile.TemporaryDirectory() as directory:
        stage1 = os.path.join(directory, "stage1")
        write_checkpoint(stage1, "net_epoch_bestval_at8.pth")
        write_stage1_config(stage1)
        stage2 = os.path.join(directory, "stage2")
        write_checkpoint(stage2, "train.log", b"existing experiment")
        try:
            prepare_dual_space_stage2(PROFILE_DIR, stage1, stage2)
        except FileExistsError as error:
            assert "refusing to overwrite" in str(error)
        else:
            raise AssertionError("existing Stage2 output was overwritten")
        with open(os.path.join(stage2, "train.log"), "rb") as stream:
            assert stream.read() == b"existing experiment"


@test("symlink permission failure uses verified copy fallback")
def test_copy_fallback():
    original_symlink = os.symlink

    def denied_symlink(*args, **kwargs):
        raise OSError("synthetic symlink permission denial")

    with tempfile.TemporaryDirectory() as directory:
        stage1 = os.path.join(directory, "stage1")
        best = write_checkpoint(stage1, "net_epoch_bestval_at4.pth")
        write_stage1_config(stage1)
        stage2 = os.path.join(directory, "stage2")
        os.symlink = denied_symlink
        try:
            summary = prepare_dual_space_stage2(PROFILE_DIR, stage1, stage2)
        finally:
            os.symlink = original_symlink
        for item in summary["modalities"].values():
            assert item["checkpoint_method"] == "copy"
            assert not os.path.islink(item["checkpoint"])
            assert sha256_file(item["checkpoint"]) == sha256_file(best)


@test("DS-V1.1 Stage2 rejects a legacy DS-V1 Stage1 seed")
def test_centered_rejects_legacy_seed():
    with tempfile.TemporaryDirectory() as directory:
        stage1 = os.path.join(directory, "stage1")
        write_checkpoint(stage1, "net_epoch_bestval_at6.pth")
        write_stage1_config(stage1, PROFILE_DIR)
        stage2 = os.path.join(directory, "stage2")
        try:
            prepare_dual_space_stage2(PROFILE_V1_1_DIR, stage1, stage2)
        except ValueError as error:
            assert "profile/version mismatch" in str(error)
        else:
            raise AssertionError("DS-V1.1 accepted a legacy DS-V1 Stage1 seed")
        assert not os.path.exists(stage2)


@test("DS-V1.1 Stage2 accepts a matching centered Stage1 seed")
def test_centered_accepts_centered_seed():
    with tempfile.TemporaryDirectory() as directory:
        stage1 = os.path.join(directory, "stage1")
        best = write_checkpoint(stage1, "net_epoch_bestval_at9.pth")
        write_stage1_config(stage1, PROFILE_V1_1_DIR)
        stage2 = os.path.join(directory, "stage2")
        summary = prepare_dual_space_stage2(
            PROFILE_V1_1_DIR, stage1, stage2
        )
        assert summary["profile"] == "ds_v1_1"
        assert summary["version"] == "ds_v1_1"
        assert summary["yaw_mode"] == "sin_cos_centered"
        assert summary["sha256"] == sha256_file(best)


@test("DS-V1.1 Stage2 rejects a matching profile with legacy yaw semantics")
def test_centered_rejects_legacy_yaw_mode():
    with tempfile.TemporaryDirectory() as directory:
        stage1 = os.path.join(directory, "stage1")
        write_checkpoint(stage1, "net_epoch_bestval_at10.pth")
        config_path = write_stage1_config(stage1, PROFILE_V1_1_DIR)
        with open(config_path, "r", encoding="utf-8") as stream:
            content = stream.read()
        content = content.replace(
            "yaw_mode: sin_cos_centered", "yaw_mode: sin_cos"
        )
        with open(config_path, "w", encoding="utf-8") as stream:
            stream.write(content)
        stage2 = os.path.join(directory, "stage2")
        try:
            prepare_dual_space_stage2(PROFILE_V1_1_DIR, stage1, stage2)
        except ValueError as error:
            assert "refiner.yaw_mode mismatch" in str(error)
        else:
            raise AssertionError("DS-V1.1 accepted legacy yaw semantics")
        assert not os.path.exists(stage2)


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

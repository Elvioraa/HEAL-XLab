"""Temporary-directory smoke tests for ``prepare_dual_space_stage2``."""

import os
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


@test("Stage2 prepare creates verified independent m2/m3/m4 seeds")
def test_success():
    with tempfile.TemporaryDirectory() as directory:
        stage1 = os.path.join(directory, "stage1")
        best = write_checkpoint(stage1, "net_epoch_bestval_at12.pth")
        stage2 = os.path.join(directory, "stage2")
        summary = prepare_dual_space_stage2(PROFILE_DIR, stage1, stage2)
        assert summary["profile"] == "ds_v1"
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

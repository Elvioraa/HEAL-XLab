"""CPU smoke tests for the read-only Dual-Space ownership checker."""

from collections import OrderedDict
import os
import sys
import tempfile

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.tools.audit_dual_space_merge import rebuild_expected_merge
from opencood.tools.check_dual_space_merge_ownership import (
    check_merge_ownership,
    main as checker_main,
)
from opencood.tools.dual_space_smoke_common import run_registered_tests


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def fixture():
    shared = {
        "dual_space_shared_object_encoder.weight": torch.tensor([1.0]),
        "dual_space_shared_geometry_encoder.weight": torch.tensor([2.0]),
        "dual_space_shared_object_refiner.weight": torch.tensor([3.0]),
    }
    m1 = OrderedDict(shared)
    m1["encoder_m1.weight"] = torch.tensor([11.0])
    stage2 = {}
    for index, modality in enumerate(("m2", "m3", "m4"), start=2):
        state = OrderedDict(shared)
        state["dual_space_object_adapter_%s.weight" % modality] = torch.tensor(
            [float(index)]
        )
        state["encoder_%s.weight" % modality] = torch.tensor([10.0 + index])
        stage2[modality] = state
    merged = rebuild_expected_merge(m1, stage2)
    return m1, stage2, merged


@test("correct production ownership passes")
def test_correct_merge():
    m1, stage2, merged = fixture()
    report = check_merge_ownership(m1, stage2, merged)
    assert report["status"] == "PASS"
    assert report["m1_shared"]["different"] == 0
    for modality in ("m2", "m3", "m4"):
        assert report[modality]["identical"] == 1


@test("different and missing owner tensors fail")
def test_bad_merge():
    m1, stage2, merged = fixture()
    broken = OrderedDict(merged)
    broken["dual_space_object_adapter_m2.weight"] = torch.tensor([99.0])
    del broken["dual_space_object_adapter_m3.weight"]
    report = check_merge_ownership(m1, stage2, broken)
    assert report["status"] == "FAIL"
    assert report["m2"]["different"] == 1
    assert report["m3"]["missing"] == 1


@test("CLI reads checkpoints and returns ownership exit status")
def test_cli():
    m1, stage2, merged = fixture()
    with tempfile.TemporaryDirectory() as directory:
        paths = {}
        for name, state in (("m1", m1),) + tuple(stage2.items()) + (("merged", merged),):
            path = os.path.join(directory, "%s.pth" % name)
            torch.save(state, path)
            paths[name] = path
        code = checker_main(
            [
                "--m1", paths["m1"],
                "--m2", paths["m2"],
                "--m3", paths["m3"],
                "--m4", paths["m4"],
                "--merged", paths["merged"],
            ]
        )
        assert code == 0


def main():
    return run_registered_tests(TESTS)


if __name__ == "__main__":
    sys.exit(main())

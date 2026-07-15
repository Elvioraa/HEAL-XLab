"""CPU smoke for the read-only PACT forward-difference auditor."""

from __future__ import absolute_import, division, print_function

import ast
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opencood.tools.audit_pact_cbea_forward_difference import (
    CSV_FIELDS,
    NUMERIC_ABS_THRESHOLD,
    NUMERIC_REL_THRESHOLD,
    ObservationHooks,
    _comparison_rows,
    _first_nodes,
    _write_outputs,
    classify_evidence_source,
    compare_tensors,
    static_code_audit,
    structured_yaml_diff,
    validate_checkpoint_equivalence,
)


AUDIT_PATH = os.path.join(ROOT, "opencood", "tools", "audit_pact_cbea_forward_difference.py")


class _Pyramid(nn.Module):
    def forward_single(self, tensor):
        return tensor * 2.0, [tensor[:, :1]]

    def forward_collab(self, tensor, *unused):
        return tensor * 3.0, [tensor[:, :1]]


class _Model(nn.Module):
    def __init__(self):
        super(_Model, self).__init__()
        self.modality_name_list = []
        self.pyramid_backbone = _Pyramid()
        self.cls_head = nn.Conv2d(1, 1, 1, bias=False)
        self.reg_head = nn.Conv2d(1, 1, 1, bias=False)
        self.dir_head = nn.Conv2d(1, 1, 1, bias=False)


class _CheckpointModel(nn.Module):
    def __init__(self):
        super(_CheckpointModel, self).__init__()
        self.encoder_m1 = nn.Conv2d(1, 1, 1, bias=False)
        self.backbone_m1 = nn.Conv2d(1, 1, 1, bias=False)
        self.aligner_m1 = nn.Conv2d(1, 1, 1, bias=False)
        self.pyramid_backbone = nn.Conv2d(1, 1, 1, bias=False)
        self.cls_head = nn.Conv2d(1, 1, 1, bias=False)
        self.reg_head = nn.Conv2d(1, 1, 1, bias=False)
        self.dir_head = nn.Conv2d(1, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)


def _assert_hooks_are_read_only():
    model = _Model().eval()
    feature = torch.ones(2, 1, 3, 3)
    original_single = model.pyramid_backbone.forward_single
    original_collab = model.pyramid_backbone.forward_collab
    expected_single = original_single(feature)
    expected_collab = original_collab(feature, None)
    hooks = ObservationHooks(model, "smoke")
    hooks.install()
    try:
        actual_single = model.pyramid_backbone.forward_single(feature)
        actual_collab = model.pyramid_backbone.forward_collab(feature, None)
    finally:
        hooks.remove()
    assert type(actual_single) is tuple and type(actual_collab) is tuple
    assert torch.equal(actual_single[0], expected_single[0])
    assert torch.equal(actual_collab[0], expected_collab[0])
    assert model.pyramid_backbone.forward_single.__self__ is model.pyramid_backbone
    assert model.pyramid_backbone.forward_collab.__self__ is model.pyramid_backbone
    assert torch.equal(feature, torch.ones_like(feature))
    assert "pyramid_forward_single_output" in hooks.values
    assert "pyramid_forward_collab_output" in hooks.values


def _assert_tensor_comparison():
    value = torch.randn(2, 3, 4, 4)
    identical = compare_tensors(value, value.clone())
    assert identical["status"] == "close"
    assert identical["metrics"]["l1_mean_absolute_difference"] == 0.0
    changed = value.clone()
    changed += NUMERIC_ABS_THRESHOLD * 10.0
    divergence = compare_tensors(value, changed)
    assert divergence["status"] == "numeric_divergence"
    shape = compare_tensors(value, torch.randn(1, 3, 4, 4))
    assert shape["status"] == "structural_divergence"
    comparisons, rows = _comparison_rows(0, {"encoder_m1": value, "cls_head_input": value},
                                          {"encoder_m1": changed, "cls_head_input": value})
    first_structural, first_numeric, largest = _first_nodes(comparisons)
    assert first_structural is None
    assert first_numeric == "encoder_m1"
    assert largest == "encoder_m1"
    assert len(rows) == 2


def _assert_checkpoint_and_bn():
    left, right = _CheckpointModel().eval(), _CheckpointModel().eval()
    right.load_state_dict(left.state_dict())
    report = validate_checkpoint_equivalence(left, right)
    assert report["detection_head_equal"] is True
    assert report["pact_all_batchnorm_eval"] is True
    assert report["aligned_all_batchnorm_eval"] is True
    with torch.no_grad():
        right.cls_head.weight.add_(1.0)
    try:
        validate_checkpoint_equivalence(left, right)
    except RuntimeError as error:
        assert "core checkpoint tensors differ" in str(error) or "detection head tensors differ" in str(error)
    else:
        raise AssertionError("different detection head was accepted")


def _assert_yaml_and_evidence():
    differences = structured_yaml_diff(
        {"dataset": {"root": "a"}, "model": {"core": "x"}},
        {"dataset": {"root": "b"}, "model": {"core": "y"}},
    )
    assert {item["path"] for item in differences} == {"dataset.root", "model.core"}
    assert classify_evidence_source({"pact_cbea": {
        "pact_local_evidence_enabled": True, "pact_fallbacks": []
    }}) == "训练后的本地证据头"
    assert "回退" in classify_evidence_source({"pact_cbea": {
        "pact_used_base_heal_fallback": True
    }})
    assert "全一" in classify_evidence_source({"pact_cbea": {
        "pact_local_evidence_enabled": True,
        "pact_fallbacks": ["missing_evidence_heatmap_confidence_ones"],
    }})


def _assert_outputs_have_no_dense_tensors():
    tensor = torch.ones(1, 1, 2, 2)
    comparisons, rows = _comparison_rows(0, {"output_cls": tensor}, {"output_cls": tensor.clone()})
    report = {"scenes": [{"comparisons": comparisons}], "summary": {"node": "output_cls"}}
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        _write_outputs(directory, report, rows, ["中文摘要测试。"])
        parsed = json.load(open(os.path.join(directory, "forward_difference_audit.json"), "r", encoding="utf-8"))
        assert parsed["scenes"][0]["comparisons"]["output_cls"]["pact"]["shape"] == [1, 1, 2, 2]
        assert os.path.isfile(os.path.join(directory, "forward_difference_audit.csv"))
        assert os.path.isfile(os.path.join(directory, "forward_difference_summary.txt"))
    assert "scene_index" in CSV_FIELDS


def _assert_static_audit():
    report = static_code_audit(ROOT)
    assert report["source_checks"]["original_calls_forward_collab"] is True
    assert report["source_checks"]["aligned_calls_forward_collab"] is False
    assert report["source_checks"]["pyramid_multiscale_weighted_fuse"] is True


def _assert_python38():
    for path in (AUDIT_PATH, os.path.abspath(__file__)):
        source = open(path, "r", encoding="utf-8").read()
        ast.parse(source, filename=path, feature_version=8)


def main():
    _assert_hooks_are_read_only()
    _assert_tensor_comparison()
    _assert_checkpoint_and_bn()
    _assert_yaml_and_evidence()
    _assert_outputs_have_no_dense_tensors()
    _assert_static_audit()
    _assert_python38()
    print("hooks preserve original returns and tensors: OK")
    print("identical/numeric/structural first-divergence checks: OK")
    print("checkpoint, detection-head and BatchNorm checks: OK")
    print("structured YAML, evidence source and output files: OK")
    print("static original/aligned/pyramid code-path audit: OK")
    print("Python 3.8 AST: OK")
    print("PACT_CBEA_FORWARD_DIFFERENCE_SMOKE_PASS")


if __name__ == "__main__":
    main()

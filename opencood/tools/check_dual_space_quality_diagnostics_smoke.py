"""CPU smoke tests for DS-V3.1 inference-only quality diagnostics."""

import json
import os
import sys
import tempfile
import inspect

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.models.sub_modules.dual_space_box_coder import boxes_hwl_to_corners_3d
from opencood.models.sub_modules.dual_space_object import refine_dual_space_detections
from opencood.tools.dual_space_smoke_common import TinyDualSpaceHost, make_boxes, make_scene
from opencood.utils.dual_space_quality_diagnostics import (
    DualSpaceQualityDiagnostics,
    _median,
)


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def quality_host():
    host = TinyDualSpaceHost(mode="inference", quality=True)
    host.dual_space_config["diagnostics"] = {
        "enabled": True,
        "quality_target": {"enabled": True},
    }
    host.dual_space_flags["diagnostics"] = True
    return host


def metadata(predicted):
    proposals = make_boxes(1)
    residuals = torch.zeros(1, 3, 8)
    residuals[0, 1, 0] = 0.25
    residuals[0, 2, 0] = 0.75
    valid = torch.ones(1, 3, dtype=torch.bool)
    quality = torch.tensor([predicted], dtype=torch.float32)
    weights = quality / quality.sum(dim=1, keepdim=True)
    return {
        "selected_proposals": proposals.detach(),
        "top_indices": torch.tensor([0]),
        "valid_mask": valid,
        "per_agent_residuals": residuals,
        "per_agent_quality": quality,
        "consensus_weights": weights,
        "quality_fallback": torch.tensor([False]),
        "agent_modalities": ("m1", "m2", "m3"),
    }, proposals.clone()


@test("diagnostics disabled leaves real refinement metadata and outputs unchanged")
def test_disabled_is_noop():
    torch.manual_seed(5)
    host = TinyDualSpaceHost(mode="inference", quality=True)
    boxes = boxes_hwl_to_corners_3d(make_boxes(2))
    scores = torch.tensor([0.9, 0.7])
    scene = make_scene(host, agent_count=2)
    context = {"scenes": (scene,)}
    with torch.no_grad():
        output, output_scores = refine_dual_space_detections(host, boxes, scores, context)
    assert torch.equal(output_scores, scores)
    assert output.shape == boxes.shape
    assert "dual_space_refinement_metadata" not in context


@test("quality-target child disabled creates no quality diagnostic output")
def test_quality_target_child_gate():
    host = TinyDualSpaceHost(mode="inference", quality=True)
    host.dual_space_config["diagnostics"] = {"enabled": True}
    collector = DualSpaceQualityDiagnostics.from_model(host)
    assert collector.enabled is False
    with tempfile.TemporaryDirectory() as directory:
        assert collector.save(directory) is None
        assert os.listdir(directory) == []


@test("enabled real quality diagnostics expose detached metadata without changing outputs")
def test_enabled_metadata_is_detached_and_invariant():
    torch.manual_seed(7)
    host = quality_host()
    boxes = boxes_hwl_to_corners_3d(make_boxes(2))
    scores = torch.tensor([0.9, 0.7])
    scene = make_scene(host, agent_count=2)
    host.dual_space_flags["diagnostics"] = False
    with torch.no_grad():
        off_boxes, off_scores = refine_dual_space_detections(host, boxes, scores, {"scenes": (scene,)})
    host.dual_space_flags["diagnostics"] = True
    context = {"scenes": (scene,)}
    with torch.no_grad():
        on_boxes, on_scores = refine_dual_space_detections(host, boxes, scores, context)
    assert torch.equal(off_boxes, on_boxes)
    assert torch.equal(off_scores, on_scores)
    data = context["dual_space_refinement_metadata"]
    for key in ("selected_proposals", "top_indices", "valid_mask", "per_agent_residuals",
                "per_agent_quality", "consensus_weights", "quality_fallback"):
        assert key in data
        assert not data[key].requires_grad
    assert data["per_agent_quality"].shape == data["valid_mask"].shape
    assert data["per_agent_residuals"].shape[-1] == 8


@test("calibration reports correct top-1 and near-uniform consensus entropy")
def test_calibration_and_entropy():
    host = quality_host()
    collector = DualSpaceQualityDiagnostics.from_model(host)
    data, gt = metadata([0.8, 0.6, 0.3])
    collector.update_metadata(data, gt, 0)
    summary = collector.summary()
    assert summary["global"]["count"] == 3
    assert summary["ranking_match_iou_ge_0.3"]["eligible_proposal_count"] == 1
    assert summary["ranking_match_iou_ge_0.3"]["top1_agent_accuracy"] == 1.0
    assert 0.8 < summary["weight_statistics"]["mean_normalized_entropy"] <= 1.0
    assert summary["match_iou_ge_0.3"]["count"] == 3


@test("reversed predicted ranking is recorded as incorrect")
def test_reversed_ranking():
    host = quality_host()
    collector = DualSpaceQualityDiagnostics.from_model(host)
    data, gt = metadata([0.3, 0.6, 0.8])
    collector.update_metadata(data, gt, 1)
    summary = collector.summary()
    assert summary["ranking_match_iou_ge_0.3"]["top1_agent_accuracy"] == 0.0


@test("one-hot-like consensus has lower normalized entropy than uniform")
def test_weight_entropy_discriminates():
    host = quality_host()
    uniform = DualSpaceQualityDiagnostics.from_model(host)
    uniform_data, gt = metadata([1.0, 1.0, 1.0])
    uniform.update_metadata(uniform_data, gt, 0)
    peaked = DualSpaceQualityDiagnostics.from_model(host)
    peaked_data, _ = metadata([0.999, 0.0005, 0.0005])
    peaked.update_metadata(peaked_data, gt, 0)
    assert peaked.summary()["weight_statistics"]["mean_normalized_entropy"] < 0.1
    assert uniform.summary()["weight_statistics"]["mean_normalized_entropy"] > 0.999


@test("collector saves machine-readable summary and both CSV tables")
def test_save_outputs():
    host = quality_host()
    collector = DualSpaceQualityDiagnostics.from_model(host)
    data, gt = metadata([0.8, 0.6, 0.3])
    collector.update_metadata(data, gt, 2)
    with tempfile.TemporaryDirectory() as directory:
        summary_path = collector.save(directory, "use_cav4")
        with open(summary_path, "r", encoding="utf-8") as stream:
            saved = json.load(stream)
        assert saved["per_modality"]["m1"]["count"] == 1
        assert os.path.isfile(os.path.join(directory, "quality_diag_pairs_use_cav4.csv"))
        assert os.path.isfile(os.path.join(directory, "quality_diag_proposals_use_cav4.csv"))


@test("identity-only observation is skipped without a quality calibration error")
def test_identity_only_observation_is_safe():
    host = quality_host()
    collector = DualSpaceQualityDiagnostics.from_model(host)
    assert collector.update_metadata(
        {
            "original_after_indices": (0,),
            "rescued_proposal_count": 0,
            "original_order_preserved": True,
        },
        make_boxes(1),
        0,
    ) is None
    assert collector.summary()["global"]["count"] == 0


@test("no-GT scene skips calibration while retaining consensus weight statistics")
def test_no_gt_does_not_pollute_calibration():
    host = quality_host()
    collector = DualSpaceQualityDiagnostics.from_model(host)
    data, _ = metadata([0.8, 0.6, 0.3])
    collector.update_metadata(data, torch.empty(0, 7), 0)
    summary = collector.summary()
    assert summary["global"]["count"] == 0
    assert summary["per_modality"] == {}
    assert summary["ranking_match_iou_ge_0.3"]["eligible_proposal_count"] == 0
    assert summary["weight_statistics"]["eligible_proposal_count"] == 1


@test("ranking and spread summaries are match-IoU stratified")
def test_ranking_and_spread_strata():
    host = quality_host()
    collector = DualSpaceQualityDiagnostics.from_model(host)
    data, gt = metadata([0.8, 0.6, 0.3])
    collector.update_metadata(data, gt, 0)
    summary = collector.summary()
    ranking = summary["ranking_match_iou_ge_0.3"]
    spread = summary["quality_spread_statistics_match_iou_ge_0.3"]
    assert ranking["eligible_proposal_count"] == 1
    assert ranking["mean_proposal_spearman"] > 0.9
    assert spread["eligible_proposal_count"] == 1
    assert spread["mean_predicted_spread"] == 0.5
    assert spread["mean_true_spread"] > 0.0


@test("even-count median follows the standard midpoint definition")
def test_even_median():
    assert _median([1.0, 3.0, 9.0, 11.0]) == 6.0


@test("proposal Spearman is recorded once and does not rescan all pairs")
def test_proposal_spearman_is_linear():
    import opencood.utils.dual_space_quality_diagnostics as diagnostics

    source = inspect.getsource(diagnostics.DualSpaceQualityDiagnostics)
    assert "def _proposal_spearman" not in source
    host = quality_host()
    collector = DualSpaceQualityDiagnostics.from_model(host)
    data, gt = metadata([0.8, 0.6, 0.3])
    collector.update_metadata(data, gt, 0)
    assert collector._proposals[0]["proposal_spearman"] is not None


@test("in-order inference gates collector construction behind diagnostics and quality")
def test_inference_collector_gate():
    path = os.path.join(os.path.dirname(__file__), "inference_heter_in_order.py")
    with open(path, "r", encoding="utf-8") as stream:
        source = stream.read()
    needle = "and model.dual_space_flags.get(\"diagnostics\", False)"
    assert needle in source
    assert "and model.dual_space_flags.get(\"quality\", False)" in source
    assert 'model.dual_space_diagnostics_config["quality_target"]' in source


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

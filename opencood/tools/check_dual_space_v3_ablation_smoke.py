"""CPU smoke tests for parameter-free Dual-Space V3 inference ablations."""

import copy
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.dual_space_box_coder import (
    boxes_hwl_to_corners_3d,
)
from opencood.models.sub_modules.dual_space_config import (
    dual_space_ablation_log_lines,
    resolve_dual_space_ablation,
    validate_dual_space_config,
)
import opencood.models.sub_modules.dual_space_object as dual_space_object
from opencood.models.sub_modules.dual_space_object import (
    predict_scene_residuals,
    refine_dual_space_detections,
    uniform_geometry_consensus,
)
from opencood.tools.dual_space_smoke_common import (
    TinyDualSpaceHost,
    make_boxes,
    make_scene,
    run_registered_tests,
)


TESTS = []
PACK_ROOT = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "HEAL_XLab_v4_DUAL_SPACE",
    "V3_ABLATION",
)


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def ablation(quality_fusion=True, refiner=True):
    return {
        "quality_fusion": {"enabled": bool(quality_fusion)},
        "refiner": {"enabled": bool(refiner)},
    }


def host(seed, quality_fusion=True, refiner=True, explicit=True):
    torch.manual_seed(seed)
    return TinyDualSpaceHost(
        modalities=("m1", "m2"),
        mode="inference",
        multi=True,
        quality=True,
        ablation=(
            ablation(quality_fusion, refiner) if explicit else None
        ),
    )


def assert_state_identical(left, right):
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert list(left_state) == list(right_state)
    for key, value in left_state.items():
        assert value.shape == right_state[key].shape
        assert torch.equal(value, right_state[key]), key


@test("missing ablation config preserves exact V3 state and forward")
def test_legacy_default_identity():
    assert resolve_dual_space_ablation(None) == ablation(True, True)
    legacy = host(11, explicit=False)
    original = host(11, True, True)
    assert_state_identical(legacy, original)
    torch.manual_seed(12)
    scene = make_scene(legacy, agent_count=2)
    proposals = make_boxes(3)
    legacy_result = predict_scene_residuals(legacy, scene, proposals)
    original_result = predict_scene_residuals(original, scene, proposals)
    for key in (
        "individual_residuals", "per_agent_residuals", "fused_residuals",
        "refined_boxes", "valid_mask", "any_valid", "coverage",
        "individual_quality", "per_agent_quality", "consensus_weights",
    ):
        assert torch.equal(legacy_result[key], original_result[key]), key


@test("quality_fusion false never calls quality-weighted consensus")
def test_quality_fusion_bypass():
    model = host(21, False, True)
    torch.manual_seed(22)
    scene = make_scene(model, agent_count=2)
    proposals = make_boxes(2)
    original_function = dual_space_object.quality_weighted_geometry_consensus

    def forbidden(*args, **kwargs):
        raise AssertionError("quality-weighted consensus was called")

    dual_space_object.quality_weighted_geometry_consensus = forbidden
    try:
        result = predict_scene_residuals(model, scene, proposals)
    finally:
        dual_space_object.quality_weighted_geometry_consensus = original_function
    expected, expected_valid = uniform_geometry_consensus(
        result["per_agent_residuals"], result["valid_mask"]
    )
    assert torch.equal(result["fused_residuals"], expected)
    assert torch.equal(result["any_valid"], expected_valid)
    assert result["quality_weighting_bypassed"] is True


@test("refiner false never calls refiner and preserves detection tensors exactly")
def test_refiner_bypass():
    model = host(31, True, False)
    calls = []
    handle = model.dual_space_shared_object_refiner.register_forward_pre_hook(
        lambda module, inputs: calls.append(True)
    )
    torch.manual_seed(32)
    scene = make_scene(model, agent_count=2)
    proposals = make_boxes(2, x=1.0, y=-2.0)
    result = predict_scene_residuals(model, scene, proposals)
    assert not calls
    assert torch.count_nonzero(result["individual_residuals"]) == 0
    assert torch.equal(result["refined_boxes"], proposals)

    pred_boxes = boxes_hwl_to_corners_3d(proposals)
    pred_scores = torch.tensor([0.9, 0.8])
    output_boxes, output_scores = refine_dual_space_detections(
        model, pred_boxes, pred_scores, {"scenes": (scene,)}
    )
    handle.remove()
    assert not calls
    assert torch.equal(output_boxes, pred_boxes)
    assert torch.equal(output_scores, pred_scores)


@test("quality and refiner disabled preserve output shapes and original boxes")
def test_combined_bypass():
    original = host(41, True, True)
    combined = host(41, False, False)
    assert_state_identical(original, combined)
    torch.manual_seed(42)
    scene = make_scene(combined, agent_count=2)
    proposals = make_boxes(4)
    result = predict_scene_residuals(combined, scene, proposals)
    assert result["refined_boxes"].shape == proposals.shape
    assert result["fused_residuals"].shape == (4, 8)
    assert result["consensus_weights"].shape == (4, 2)
    assert torch.equal(result["refined_boxes"], proposals)
    assert result["quality_weighting_bypassed"] is True
    assert result["refiner_bypassed"] is True


@test("four formal YAMLs expose only the requested ablation matrix")
def test_yaml_matrix():
    expected = {
        "merged_original.yaml": (True, True),
        "merged_q0.yaml": (False, True),
        "merged_r0.yaml": (True, False),
        "merged_q0_r0.yaml": (False, False),
    }
    source = load_yaml(
        os.path.join(os.path.dirname(PACK_ROOT), "DS_V3", "merged_infer.yaml"),
        None,
    )
    source.pop("name", None)
    source["model"]["args"]["dual_space"].pop("ablation", None)
    reference = source
    for filename, switches in expected.items():
        config = load_yaml(os.path.join(PACK_ROOT, filename), None)
        dual = config["model"]["args"]["dual_space"]
        validate_dual_space_config(dual)
        resolved = resolve_dual_space_ablation(dual)
        assert resolved["quality_fusion"]["enabled"] is switches[0]
        assert resolved["refiner"]["enabled"] is switches[1]
        normalized = copy.deepcopy(config)
        normalized.pop("name", None)
        normalized["model"]["args"]["dual_space"].pop("ablation")
        assert normalized == reference


@test("startup log reports the resolved component switches")
def test_startup_log():
    model = host(51, False, True)
    assert dual_space_ablation_log_lines(model) == (
        "[DualSpace Ablation]",
        "quality_fusion = False",
        "refiner = True",
    )


def main():
    return run_registered_tests(TESTS)


if __name__ == "__main__":
    sys.exit(main())

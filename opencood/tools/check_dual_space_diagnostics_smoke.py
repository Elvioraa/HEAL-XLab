"""CPU smoke tests for inference-only Dual-Space refinement diagnostics."""

import json
import importlib.util
import os
import sys
import tempfile

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.models.sub_modules.dual_space_box_coder import (
    boxes_hwl_to_corners_3d,
    corners_3d_to_boxes_hwl,
    pairwise_rotated_bev_iou_hwl,
)
from opencood.models.sub_modules.dual_space_config import (
    dual_space_feature_flags,
    resolve_dual_space_diagnostics,
    validate_dual_space_config,
)
from opencood.models.sub_modules.dual_space_object import (
    refine_dual_space_detections,
)
from opencood.tools.dual_space_smoke_common import (
    TinyDualSpaceHost,
    make_boxes,
    make_dual_config,
    make_scene,
)
from opencood.utils.dual_space_refinement_diagnostics import (
    OBSERVATION_KEY,
    DualSpaceRefinementDiagnostics,
    deterministic_before_matching,
    make_refinement_observation,
    official_pairwise_rotated_bev_iou,
)


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def diagnostics_config(enabled=True, save_per_object=False):
    return {
        "diagnostics": {
            "enabled": enabled,
            "match_iou_min": 0.3,
            "thresholds": [0.3, 0.5, 0.7],
            "improvement_epsilon": 1.0e-4,
            "save_per_object": save_per_object,
        }
    }


def test_pairwise_iou(boxes_a, boxes_b):
    boxes_a = corners_3d_to_boxes_hwl(boxes_a)
    boxes_b = corners_3d_to_boxes_hwl(boxes_b)
    return pairwise_rotated_bev_iou_hwl(boxes_a, boxes_b).cpu().numpy()


def make_recorder(enabled=True, save_per_object=False):
    return DualSpaceRefinementDiagnostics(
        diagnostics_config(enabled, save_per_object),
        iou_function=test_pairwise_iou,
    )


def corners(center_x_values):
    boxes = torch.zeros(len(center_x_values), 7, dtype=torch.float32)
    boxes[:, 0] = torch.tensor(center_x_values, dtype=torch.float32)
    boxes[:, 3] = 1.5
    boxes[:, 4] = 2.0
    boxes[:, 5] = 4.0
    return boxes_hwl_to_corners_3d(boxes)


def observation(before, after, scores=None, metadata=None):
    if scores is None and before is not None:
        scores = torch.linspace(0.9, 0.8, before.shape[0])
    if metadata is None:
        count = 0 if before is None else int(before.shape[0])
        metadata = {
            "original_after_indices": tuple(range(count)),
            "rescued_proposal_count": 0,
            "original_order_preserved": True,
        }
    return make_refinement_observation(before, scores, after, scores, metadata)


def test_official_iou_definition():
    matrix = official_pairwise_rotated_bev_iou(corners([1.0]), corners([0.0]))
    assert matrix.shape == (1, 1)
    assert abs(float(matrix[0, 0]) - 0.6) < 1.0e-6


@test("deterministic BEFORE matching uses stable proposal and GT indices")
def test_deterministic_matching():
    matrix = torch.tensor([[0.8, 0.8], [0.8, 0.7]]).numpy()
    matches = deterministic_before_matching(matrix, 0.3)
    assert matches == [(0, 0, float(matrix[0, 0])), (1, 1, float(matrix[1, 1]))]


@test("IoU improvement and cross-up at 0.7 are recorded")
def test_improvement_cross_up():
    recorder = make_recorder()
    result = {
        OBSERVATION_KEY: observation(corners([1.0]), corners([0.0])),
        "gt_box_tensor": corners([0.0]),
    }
    recorder.update_inference_result(result, "improved")
    summary = recorder.summary()
    assert summary["matched_count"] == 1
    assert abs(summary["mean_iou_before"] - 0.6) < 1.0e-6
    assert abs(summary["mean_iou_after"] - 1.0) < 1.0e-6
    assert summary["improved_count"] == 1
    assert summary["cross_up_0.7"] == 1
    assert summary["cross_down_0.7"] == 0


@test("IoU worsening and cross-down at 0.7 are recorded")
def test_worsening_cross_down():
    recorder = make_recorder()
    result = {
        OBSERVATION_KEY: observation(corners([0.0]), corners([1.0])),
        "gt_box_tensor": corners([0.0]),
    }
    recorder.update_inference_result(result, "worsened")
    summary = recorder.summary()
    assert summary["worsened_count"] == 1
    assert summary["cross_down_0.7"] == 1
    assert summary["cross_up_0.7"] == 0


@test("empty predictions and empty GT are safe")
def test_empty_inputs():
    recorder = make_recorder()
    empty = corners([])
    recorder.update_inference_result(
        {OBSERVATION_KEY: observation(None, None), "gt_box_tensor": empty},
        "none",
    )
    recorder.update_inference_result(
        {OBSERVATION_KEY: observation(empty, empty), "gt_box_tensor": empty},
        "empty",
    )
    summary = recorder.summary()
    assert summary["scene_count"] == 2
    assert summary["matched_count"] == 0
    assert summary["mean_delta_iou"] == 0.0


@test("RPR count changes require and respect explicit source indices")
def test_rpr_identity_contract():
    before = corners([1.0])
    after = torch.cat((corners([0.0]), corners([10.0])), dim=0)
    metadata = {
        "original_after_indices": (0,),
        "rescued_proposal_count": 1,
        "original_order_preserved": True,
    }
    recorder = make_recorder()
    recorder.update_inference_result(
        {
            OBSERVATION_KEY: observation(before, after, metadata=metadata),
            "gt_box_tensor": corners([0.0]),
        },
        "rpr",
    )
    summary = recorder.summary()
    assert summary["proposal_count_before"] == 1
    assert summary["proposal_count_after"] == 2
    assert summary["rescued_proposal_count"] == 1
    assert summary["matched_count"] == 1

    unsafe = observation(before, after, metadata={"rescued_proposal_count": 1})
    try:
        make_recorder().update_observation(
            unsafe, corners([0.0]), "unsafe"
        )
    except RuntimeError as error:
        assert "original_after_indices" in str(error)
    else:
        raise AssertionError("count-changing RPR observation lacked identity metadata")


@test("diagnostics disabled is a complete no-op")
def test_disabled_noop():
    recorder = make_recorder(enabled=False)
    assert recorder.update_inference_result({}, "disabled") is None
    with tempfile.TemporaryDirectory() as directory:
        assert recorder.save(directory) is None
        assert os.listdir(directory) == []


@test("legacy config defaults diagnostics off and training mode rejects it")
def test_diagnostics_config_contract():
    legacy = make_dual_config(mode="inference")
    validate_dual_space_config(legacy)
    assert dual_space_feature_flags(legacy)["diagnostics"] is False
    assert resolve_dual_space_diagnostics(legacy)["enabled"] is False

    training = make_dual_config(mode="stage1_anchor")
    training["diagnostics"] = diagnostics_config(True)["diagnostics"]
    try:
        validate_dual_space_config(training)
    except ValueError as error:
        assert "inference-only" in str(error)
    else:
        raise AssertionError("training config enabled inference-only diagnostics")

    for invalid in (float("nan"), float("inf")):
        invalid_config = make_dual_config(mode="inference")
        invalid_config["diagnostics"] = diagnostics_config(True)["diagnostics"]
        invalid_config["diagnostics"]["improvement_epsilon"] = invalid
        try:
            validate_dual_space_config(invalid_config)
        except ValueError as error:
            assert "finite" in str(error)
        else:
            raise AssertionError("non-finite diagnostics epsilon was accepted")


@test("real refinement output and state are invariant to diagnostics observer")
def test_real_refinement_invariance():
    torch.manual_seed(17)
    host = TinyDualSpaceHost(mode="inference")
    host.eval()
    with torch.no_grad():
        output_layer = host.dual_space_shared_object_refiner.network[-1]
        output_layer.bias[0] = 0.05
        output_layer.bias[7] = 1.0

    state_keys_before = tuple(host.state_dict().keys())
    state_before = {
        key: value.detach().clone() for key, value in host.state_dict().items()
    }
    boxes = boxes_hwl_to_corners_3d(make_boxes(2, x=0.0, y=0.0))
    scores = torch.tensor([0.9, 0.7], dtype=boxes.dtype)
    scene = make_scene(host, agent_count=2)

    host.dual_space_flags["diagnostics"] = False
    disabled_context = {"scenes": (scene,)}
    with torch.no_grad():
        disabled_boxes, disabled_scores = refine_dual_space_detections(
            host, boxes, scores, disabled_context
        )
    assert "dual_space_refinement_metadata" not in disabled_context

    host.dual_space_flags["diagnostics"] = True
    enabled_context = {"scenes": (scene,)}
    with torch.no_grad():
        enabled_boxes, enabled_scores = refine_dual_space_detections(
            host, boxes, scores, enabled_context
        )

    assert not torch.equal(boxes, disabled_boxes)
    assert torch.equal(disabled_boxes, enabled_boxes)
    assert torch.equal(disabled_scores, enabled_scores)
    assert enabled_context["dual_space_refinement_metadata"] == {
        "original_after_indices": (0, 1),
        "rescued_proposal_count": 0,
        "original_order_preserved": True,
    }
    assert tuple(host.state_dict().keys()) == state_keys_before
    for key, value in host.state_dict().items():
        assert torch.equal(value, state_before[key]), key


@test("diagnostics save deterministic JSON and optional per-object CSV")
def test_save_outputs():
    recorder = make_recorder(save_per_object=True)
    recorder.update_inference_result(
        {
            OBSERVATION_KEY: observation(corners([1.0]), corners([0.0])),
            "gt_box_tensor": corners([0.0]),
        },
        7,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = recorder.save(directory, suffix="use_cav2")
        assert os.path.basename(path) == "dual_space_refinement_stats_use_cav2.json"
        with open(path, "r", encoding="utf-8") as stream:
            saved = json.load(stream)
        assert saved["cross_up_0.7"] == 1
        csv_path = os.path.join(
            directory, "dual_space_refinement_objects_use_cav2.csv"
        )
        assert os.path.isfile(csv_path)


def run_inference_invariance_fixture():
    from opencood.tools import inference_utils
    import opencood.models.sub_modules.dual_space_object as dual_space_object

    original_refiner = dual_space_object.refine_dual_space_detections
    base_boxes = corners([0.0, 5.0])
    base_scores = torch.tensor([0.9, 0.7])
    gt_boxes = corners([0.25, 5.25])

    class FakeDataset(object):
        def post_process(self, batch_data, output_dict):
            return base_boxes.clone(), base_scores.clone(), gt_boxes.clone()

    class FakeModel(object):
        dual_space_enabled = True

        def __init__(self, diagnostics):
            self.dual_space_flags = {
                "diagnostics": diagnostics,
                "report_stats": False,
            }

        def __call__(self, data):
            return {"dual_space_context": {"scenes": ({},)}}

    def fake_refiner(model, boxes, scores, context):
        refined = boxes.clone()
        refined[:, :, 0] += 0.125
        if model.dual_space_flags["diagnostics"]:
            context["dual_space_refinement_metadata"] = {
                "original_after_indices": (0, 1),
                "rescued_proposal_count": 0,
                "original_order_preserved": True,
            }
        return refined, scores

    dual_space_object.refine_dual_space_detections = fake_refiner
    try:
        disabled = inference_utils.inference_early_fusion(
            {"ego": {}}, FakeModel(False), FakeDataset()
        )
        enabled = inference_utils.inference_early_fusion(
            {"ego": {}}, FakeModel(True), FakeDataset()
        )
    finally:
        dual_space_object.refine_dual_space_detections = original_refiner

    return disabled, enabled


def test_inference_invariance():
    disabled, enabled = run_inference_invariance_fixture()

    assert OBSERVATION_KEY not in disabled
    assert OBSERVATION_KEY in enabled
    assert torch.equal(disabled["pred_box_tensor"], enabled["pred_box_tensor"])
    assert torch.equal(disabled["pred_score"], enabled["pred_score"])
    assert torch.equal(
        torch.argsort(disabled["pred_score"], descending=True),
        torch.argsort(enabled["pred_score"], descending=True),
    )


def test_official_ap_invariance():
    from opencood.utils import eval_utils

    disabled, enabled = run_inference_invariance_fixture()

    def official_stats(result):
        stats = {
            value: {"tp": [], "fp": [], "gt": 0, "score": []}
            for value in (0.3, 0.5, 0.7)
        }
        for threshold in stats:
            eval_utils.caluclate_tp_fp(
                result["pred_box_tensor"], result["pred_score"],
                result["gt_box_tensor"], stats, threshold,
            )
        return stats

    assert official_stats(disabled) == official_stats(enabled)


def main():
    if importlib.util.find_spec("shapely") is not None:
        TESTS.append(
            ("official evaluation IoU is rotated BEV polygon IoU", test_official_iou_definition)
        )
        TESTS.append(
            (
                "observer integration preserves final boxes scores and ordering",
                test_inference_invariance,
            )
        )
        TESTS.append(
            (
                "enabled observer leaves final boxes scores ordering and AP inputs unchanged",
                test_official_ap_invariance,
            )
        )
    else:
        print("[SKIP] official Shapely rotated-BEV IoU: Shapely unavailable")
        print("[SKIP] inference-utils observer integration: Shapely unavailable")
        print("[SKIP] official AP invariance integration: Shapely unavailable")
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

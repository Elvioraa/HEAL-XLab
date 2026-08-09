"""CPU smoke tests for DS-V4 inference-only remote proposal rescue."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.models.sub_modules.dual_space_box_coder import (
    boxes_hwl_to_corners_3d,
    corners_3d_to_boxes_hwl,
)
from opencood.models.sub_modules.dual_space_object import (
    _scene_anchor_for_decoder,
    refine_dual_space_detections,
)
from opencood.models.sub_modules.dual_space_remote_proposal_rescue import (
    rescue_remote_proposals,
)
from opencood.tools.dual_space_smoke_common import (
    TinyDualSpaceHost,
    make_boxes,
    make_scene,
    run_registered_tests,
)


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def config(**updates):
    result = {
        "include_ego": False,
        "min_score": 0.5,
        "dedup_iou": 0.5,
        "max_per_agent": 4,
        "max_total_added": 4,
    }
    result.update(updates)
    return result


def empty():
    return make_boxes(0), torch.empty(0)


@test("fused candidates survive when no remote candidate exists")
def test_no_remote():
    fused = make_boxes(1)
    scores = torch.tensor([0.8])
    boxes, output_scores, stats = rescue_remote_proposals(
        fused, scores, (empty()[0], empty()[0]), (empty()[1], empty()[1]), config()
    )
    assert torch.equal(boxes, fused)
    assert torch.equal(output_scores, scores)
    assert stats["rescued_proposal_count"] == 0


@test("fused miss plus remote hit appends candidate")
def test_remote_hit():
    boxes, scores, stats = rescue_remote_proposals(
        make_boxes(0),
        torch.empty(0),
        (make_boxes(0), make_boxes(1, x=5.0)),
        (torch.empty(0), torch.tensor([0.9])),
        config(),
    )
    assert boxes.shape == (1, 7)
    assert torch.allclose(scores, torch.tensor([0.9]))
    assert stats["rescued_proposal_count"] == 1
    assert stats["remote_candidates_before_filter"] == 1
    assert stats["remote_candidates_after_score"] == 1
    assert stats["remote_candidates_deduped"] == 1
    assert stats["rescued_added"] == 1


@test("remote proposal overlapping fused result is rejected")
def test_fused_overlap():
    fused = make_boxes(1, x=2.0)
    boxes, _, stats = rescue_remote_proposals(
        fused,
        torch.tensor([0.7]),
        (make_boxes(0), fused.clone()),
        (torch.empty(0), torch.tensor([0.99])),
        config(),
    )
    assert boxes.shape[0] == 1
    assert stats["remote_overlap_rejected_count"] == 1


@test("remote score threshold is enforced")
def test_low_score():
    boxes, _, _ = rescue_remote_proposals(
        make_boxes(0),
        torch.empty(0),
        (make_boxes(0), make_boxes(1)),
        (torch.empty(0), torch.tensor([0.49])),
        config(),
    )
    assert boxes.shape[0] == 0


@test("remote duplicates keep deterministic highest score")
def test_remote_dedup():
    candidate = make_boxes(1, x=4.0)
    boxes, scores, stats = rescue_remote_proposals(
        make_boxes(0),
        torch.empty(0),
        (make_boxes(0), candidate, candidate.clone()),
        (torch.empty(0), torch.tensor([0.7]), torch.tensor([0.9])),
        config(),
    )
    assert boxes.shape[0] == 1
    assert torch.allclose(scores, torch.tensor([0.9]))
    assert stats["remote_duplicate_rejected_count"] == 1


@test("max_per_agent limits each source independently")
def test_max_per_agent():
    remote = make_boxes(5)
    remote[:, 0] = torch.arange(5, dtype=torch.float32) * 6.0
    boxes, _, _ = rescue_remote_proposals(
        make_boxes(0),
        torch.empty(0),
        (make_boxes(0), remote),
        (torch.empty(0), torch.linspace(0.6, 1.0, 5)),
        config(max_per_agent=2, max_total_added=8),
    )
    assert boxes.shape[0] == 2


@test("max_total_added limits the global rescue pool")
def test_max_total():
    remote = make_boxes(6)
    remote[:, 0] = torch.arange(6, dtype=torch.float32) * 6.0
    boxes, _, _ = rescue_remote_proposals(
        make_boxes(0),
        torch.empty(0),
        (make_boxes(0), remote),
        (torch.empty(0), torch.linspace(0.6, 1.0, 6)),
        config(max_per_agent=6, max_total_added=3),
    )
    assert boxes.shape[0] == 3


@test("include_ego false excludes ego local detections")
def test_exclude_ego():
    boxes, _, _ = rescue_remote_proposals(
        make_boxes(0),
        torch.empty(0),
        (make_boxes(1, x=1.0), make_boxes(0)),
        (torch.tensor([0.99]), torch.empty(0)),
        config(include_ego=False),
    )
    assert boxes.shape[0] == 0


@test("include_ego true permits ego candidates as an ablation")
def test_include_ego():
    boxes, _, _ = rescue_remote_proposals(
        make_boxes(0),
        torch.empty(0),
        (make_boxes(1, x=1.0),),
        (torch.tensor([0.99]),),
        config(include_ego=True),
    )
    assert boxes.shape[0] == 1


@test("rescued proposal retains remote max score")
def test_rescued_score():
    candidate = make_boxes(1, x=8.0)
    _, scores, _ = rescue_remote_proposals(
        make_boxes(0),
        torch.empty(0),
        (make_boxes(0), candidate, candidate.clone()),
        (torch.empty(0), torch.tensor([0.88]), torch.tensor([0.77])),
        config(),
    )
    assert torch.equal(scores, torch.tensor([0.88]))


@test("fused score values and ordering remain unchanged")
def test_fused_scores():
    fused = make_boxes(2)
    fused[:, 0] = torch.tensor([-8.0, 8.0])
    scores = torch.tensor([0.31, 0.93])
    _, output, _ = rescue_remote_proposals(
        fused,
        scores,
        (make_boxes(0), make_boxes(1, x=14.0)),
        (torch.empty(0), torch.tensor([0.8])),
        config(),
    )
    assert torch.equal(output[:2], scores)


@test("candidate coordinates remain ego-frame hwl convention")
def test_coordinate_contract():
    remote = make_boxes(1, x=7.0, y=-3.0)
    boxes, _, _ = rescue_remote_proposals(
        make_boxes(0),
        torch.empty(0),
        (make_boxes(0), remote),
        (torch.empty(0), torch.tensor([0.9])),
        config(),
    )
    assert torch.equal(boxes, remote)
    assert boxes[0, 3:6].tolist() == [1.5, 2.0, 4.0]


def make_rpr_context(host, remote_box, ego_valid=True):
    scene = make_scene(host, 2)
    if not ego_valid:
        scene["agent_support"][0].zero_()
    scene["remote_proposals"] = (make_boxes(0), remote_box)
    scene["remote_scores"] = (torch.empty(0), torch.tensor([0.9]))
    return {"scenes": (scene,), "box_order": "hwl", "aligned_to": "ego"}


@test("rescued candidate enters Object-Space refinement")
def test_rpr_refinement():
    host = TinyDualSpaceHost(rescue=True)
    final = host.dual_space_shared_object_refiner.network[-1]
    final.bias.data.zero_()
    final.bias.data[0] = 0.1
    final.bias.data[7] = 1.0
    remote = make_boxes(1, x=2.0)
    corners, scores = refine_dual_space_detections(
        host, None, None, make_rpr_context(host, remote)
    )
    refined = corners_3d_to_boxes_hwl(corners)
    assert refined[0, 0] > remote[0, 0]
    assert torch.equal(scores, torch.tensor([0.9]))


@test("remote-valid rescue refines when ego ROI is invalid")
def test_remote_only_roi():
    host = TinyDualSpaceHost(rescue=True)
    final = host.dual_space_shared_object_refiner.network[-1]
    final.bias.data.zero_()
    final.bias.data[1] = 0.1
    final.bias.data[7] = 1.0
    remote = make_boxes(1)
    corners, _ = refine_dual_space_detections(
        host, None, None, make_rpr_context(host, remote, ego_valid=False)
    )
    refined = corners_3d_to_boxes_hwl(corners)
    assert refined[0, 1] > remote[0, 1]


@test("top-K refinement retains every unselected candidate")
def test_topk_retention():
    host = TinyDualSpaceHost(rescue=True)
    host.dual_space_config["roi"]["max_infer_proposals"] = 2
    host.dual_space_config["remote_proposal_rescue"].update(
        {"max_per_agent": 8, "max_total_added": 8, "dedup_iou": 0.1}
    )
    final = host.dual_space_shared_object_refiner.network[-1]
    final.bias.data.zero_()
    final.bias.data[0] = 0.1
    final.bias.data[7] = 1.0
    remote = make_boxes(6)
    remote[:, 0] = torch.linspace(-10.0, 10.0, 6)
    scene = make_scene(host, 2)
    scene["remote_proposals"] = (make_boxes(0), remote)
    scene["remote_scores"] = (torch.empty(0), torch.linspace(0.6, 0.95, 6))
    corners, scores = refine_dual_space_detections(
        host, None, None, {"scenes": (scene,)}
    )
    output = corners_3d_to_boxes_hwl(corners)
    assert output.shape[0] == scores.shape[0] == 6
    expected_order = torch.argsort(scene["remote_scores"][1], descending=True)
    ordered_remote = remote.index_select(0, expected_order)
    changed = torch.abs(output[:, 0] - ordered_remote[:, 0]) > 1e-4
    assert int(changed.sum()) == 2


@test("RPR-disabled candidate path is an exact geometric no-op")
def test_rpr_disabled_noop():
    v3 = TinyDualSpaceHost(
        mode="inference", multi=True, quality=True, rescue=False
    )
    v4 = TinyDualSpaceHost(multi=True, quality=True, rescue=True)
    v4.load_state_dict(v3.state_dict(), strict=True)
    boxes = make_boxes(2)
    boxes[:, 0] = torch.tensor([-5.0, 5.0])
    corners = boxes_hwl_to_corners_3d(boxes)
    scores = torch.tensor([0.8, 0.7])
    scene_v3 = make_scene(v3, 2)
    scene_v4 = make_scene(v4, 2)
    scene_v3["agent_support"].zero_()
    scene_v4["agent_support"].zero_()
    scene_v4["remote_proposals"] = (make_boxes(0), make_boxes(0))
    scene_v4["remote_scores"] = (torch.empty(0), torch.empty(0))
    output_v3 = refine_dual_space_detections(
        v3, corners, scores, {"scenes": (scene_v3,)}
    )
    output_v4 = refine_dual_space_detections(
        v4, corners, scores, {"scenes": (scene_v4,)}
    )
    assert torch.equal(output_v3[0], corners)
    assert torch.equal(output_v4[0], corners)
    assert torch.equal(output_v3[1], output_v4[1])


@test("real unbatched and optional batched anchor layouts preserve cardinality")
def test_real_anchor_layout():
    height, width, anchor_count = 2, 3, 2
    anchors = torch.zeros(height, width, anchor_count, 7)
    anchors[..., 3] = 1.5
    anchors[..., 4] = 2.0
    anchors[..., 5] = 4.0
    selected = _scene_anchor_for_decoder(anchors, 0, 2)
    assert selected.shape == (height, width, anchor_count, 7)
    assert selected.reshape(-1, 7).shape[0] == height * width * anchor_count
    batched = anchors.unsqueeze(0).repeat(2, 1, 1, 1, 1)
    selected_batched = _scene_anchor_for_decoder(batched, 1, 2)
    assert torch.equal(selected_batched, anchors)
    assert selected_batched.reshape(-1, 7).shape[0] == (
        height * width * anchor_count
    )


@test("report_stats is gated and exposes the complete scene contract")
def test_report_stats_contract():
    quiet = TinyDualSpaceHost(rescue=True)
    quiet_context = make_rpr_context(quiet, make_boxes(1, x=4.0))
    refine_dual_space_detections(quiet, None, None, quiet_context)
    assert "dual_space_stats" not in quiet_context

    host = TinyDualSpaceHost(
        multi=True, quality=True, rescue=True, report_stats=True
    )
    context = make_rpr_context(host, make_boxes(1, x=4.0))
    refine_dual_space_detections(host, None, None, context)
    stats = context["dual_space_stats"]
    required = {
        "original_fused_proposals",
        "rescued_remote_proposals",
        "final_candidate_count",
        "refined_proposal_count",
        "valid_agent_object_pairs",
        "mean_agents_per_object",
        "mean_roi_coverage",
        "mean_quality",
        "median_quality",
        "low_quality_fraction",
        "remote_candidates_before_filter",
        "remote_candidates_after_score",
        "remote_candidates_deduped",
        "rescued_added",
    }
    assert required.issubset(stats)
    assert stats["original_fused_proposals"] == 0
    assert stats["rescued_remote_proposals"] == 1
    assert stats["final_candidate_count"] == 1


@test("V4 RPR adds no state keys beyond matching V3")
def test_rpr_state_inert():
    v3 = TinyDualSpaceHost(
        mode="inference", multi=True, quality=True, rescue=False
    )
    v4 = TinyDualSpaceHost(multi=True, quality=True, rescue=True)
    assert tuple(v3.state_dict().keys()) == tuple(v4.state_dict().keys())


if __name__ == "__main__":
    sys.exit(run_registered_tests(TESTS))

"""CPU smoke tests for DS-V3 agent-object quality consensus."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.loss.dual_space_object_loss import compute_dual_space_object_loss
from opencood.models.sub_modules.dual_space_box_coder import (
    aligned_rotated_bev_iou_hwl,
)
from opencood.models.sub_modules.dual_space_object import (
    build_collab_dual_space_context,
    normalized_agent_object_distance,
    predict_scene_residuals,
    quality_weighted_geometry_consensus,
    run_dual_space_training,
    uniform_geometry_consensus,
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


def training_payload(host, scene=None):
    if scene is None:
        scene = make_scene(host, 2)
    boxes = make_boxes(1).unsqueeze(0)
    data = {
        "object_bbx_center": boxes,
        "object_bbx_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    return run_dual_space_training(host, {"scenes": (scene,)}, data)


@test("quality disabled creates no parameters or state keys")
def test_quality_absent():
    host = TinyDualSpaceHost(multi=True, quality=False)
    assert not hasattr(host, "dual_space_shared_quality_head")
    assert not any("quality" in key for key in host.state_dict())


@test("quality enabled creates one shared scalar head")
def test_quality_present():
    host = TinyDualSpaceHost(multi=True, quality=True)
    assert hasattr(host, "dual_space_shared_quality_head")
    keys = host.state_dict().keys()
    assert any(key.startswith("dual_space_shared_quality_head.") for key in keys)


@test("quality prediction shape and range are agent-object specific")
def test_quality_shape_range():
    host = TinyDualSpaceHost(multi=True, quality=True)
    result = predict_scene_residuals(host, make_scene(host, 3), make_boxes(4))
    assert result["per_agent_quality"].shape == (4, 3)
    assert result["individual_quality"].shape == (12,)
    assert bool(((result["individual_quality"] >= 0) & (
        result["individual_quality"] <= 1
    )).all())


@test("perfect aligned boxes have quality target one")
def test_perfect_iou():
    boxes = make_boxes(3)
    iou = aligned_rotated_bev_iou_hwl(boxes, boxes.clone())
    assert torch.allclose(iou, torch.ones_like(iou), atol=1e-5)


@test("geometrically wrong boxes have lower quality target")
def test_wrong_iou():
    target = make_boxes(1)
    wrong = make_boxes(1, x=8.0)
    assert float(aligned_rotated_bev_iou_hwl(wrong, target)[0]) < 0.1


@test("quality target is detached in training payload")
def test_target_detached():
    host = TinyDualSpaceHost(multi=False, quality=True)
    payload = training_payload(host)
    target = payload["scenes"][0]["quality_targets"]
    assert target.requires_grad is False


@test("quality loss is valid-pair only and finite")
def test_quality_loss_valid_only():
    host = TinyDualSpaceHost(multi=False, quality=True)
    scene = make_scene(host, 2)
    scene["agent_support"][1].zero_()
    payload = training_payload(host, scene)
    result = payload["scenes"][0]
    assert result["individual_quality"].numel() == int(result["valid_mask"].sum())
    loss, stats = compute_dual_space_object_loss(payload)
    assert torch.isfinite(loss)
    assert stats["dual_space_quality_loss"] >= 0.0


@test("high-quality residual dominates weighted consensus")
def test_high_quality_dominates():
    residuals = torch.zeros(1, 2, 8)
    residuals[0, 0, 0] = -1.0
    residuals[0, 1, 0] = 1.0
    fused, _, weights, _ = quality_weighted_geometry_consensus(
        residuals,
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0.01, 0.99]]),
    )
    assert float(fused[0, 0]) > 0.9
    assert float(weights[0, 1]) > 0.98


@test("remote can dominate when ego quality is weak")
def test_remote_dominates():
    residuals = torch.zeros(1, 3, 8)
    residuals[0, :, 1] = torch.tensor([-2.0, 0.7, 0.8])
    quality = torch.tensor([[0.01, 0.90, 0.80]])
    fused, _, _, _ = quality_weighted_geometry_consensus(
        residuals, torch.ones(1, 3, dtype=torch.bool), quality
    )
    assert 0.70 < float(fused[0, 1]) < 0.81


@test("quality consensus is invariant to agent order")
def test_quality_order():
    residuals = torch.randn(4, 5, 8)
    valid = torch.rand(4, 5) > 0.2
    quality = torch.rand(4, 5)
    order = torch.tensor([3, 0, 4, 1, 2])
    first = quality_weighted_geometry_consensus(
        residuals, valid, quality
    )[0]
    second = quality_weighted_geometry_consensus(
        residuals[:, order], valid[:, order], quality[:, order]
    )[0]
    assert torch.allclose(first, second, atol=1e-6)


@test("near-zero fourth-agent quality leaves three-agent result unchanged")
def test_negligible_fourth_agent():
    residuals = torch.randn(1, 4, 8)
    valid = torch.ones(1, 4, dtype=torch.bool)
    quality = torch.tensor([[0.7, 0.8, 0.9, 0.0]])
    first = quality_weighted_geometry_consensus(
        residuals[:, :3], valid[:, :3], quality[:, :3]
    )[0]
    second = quality_weighted_geometry_consensus(
        residuals, valid, quality
    )[0]
    assert torch.equal(first, second)


@test("weighted sin-cos yaw consensus stays circular across pi boundary")
def test_yaw_circular_consensus():
    residuals = torch.zeros(1, 2, 8)
    angles = torch.deg2rad(torch.tensor([179.0, -179.0]))
    residuals[0, :, 6] = torch.sin(angles)
    residuals[0, :, 7] = torch.cos(angles)
    fused = quality_weighted_geometry_consensus(
        residuals,
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2),
    )[0]
    fused_angle = torch.atan2(fused[0, 6], fused[0, 7]).abs()
    assert torch.allclose(fused_angle, torch.tensor(torch.pi), atol=1e-4)


@test("near-zero quality sum falls back exactly to uniform")
def test_quality_fallback():
    residuals = torch.randn(3, 4, 8)
    valid = torch.tensor(
        [[True, True, False, False], [False, True, True, True], [True] * 4]
    )
    expected = uniform_geometry_consensus(residuals, valid)[0]
    fused, _, _, fallback = quality_weighted_geometry_consensus(
        residuals, valid, torch.zeros(3, 4), min_quality_sum=1e-6
    )
    assert torch.equal(fallback, torch.ones(3, dtype=torch.bool))
    assert torch.equal(fused, expected)


@test("all-invalid quality consensus falls back to original box")
def test_quality_no_valid():
    host = TinyDualSpaceHost(multi=False, quality=True)
    scene = make_scene(host, 2)
    scene["agent_support"].zero_()
    proposals = make_boxes(2)
    result = predict_scene_residuals(host, scene, proposals)
    assert not bool(result["any_valid"].any())
    assert torch.equal(result["refined_boxes"], proposals)


@test("detached consensus weights do not update quality head")
def test_consensus_weight_detach():
    residuals = torch.randn(2, 3, 8, requires_grad=True)
    quality = torch.rand(2, 3, requires_grad=True)
    fused = quality_weighted_geometry_consensus(
        residuals,
        torch.ones(2, 3, dtype=torch.bool),
        quality,
        detach_quality=True,
    )[0]
    fused.sum().backward()
    assert residuals.grad is not None
    assert quality.grad is None


@test("Stage1 quality loss updates the shared Quality Head")
def test_stage1_quality_gradient():
    host = TinyDualSpaceHost(multi=False, quality=True)
    payload = training_payload(host)
    loss, _ = compute_dual_space_object_loss(payload)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in host.dual_space_shared_quality_head.parameters()
    )


@test("Stage2 freezes Quality Head but reaches active adapter")
def test_stage2_quality_gradient():
    host = TinyDualSpaceHost(
        modalities=("m2",),
        mode="stage2_adapt",
        active_modality="m2",
        multi=False,
        quality=True,
    )
    payload = training_payload(host, make_scene(host, 1))
    loss, _ = compute_dual_space_object_loss(payload)
    loss.backward()
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in host.dual_space_shared_quality_head.parameters()
    )
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in host.dual_space_object_adapter_m2.parameters()
    )


@test("quality without multi-scale is a legal executable ablation")
def test_quality_without_multiscale():
    host = TinyDualSpaceHost(multi=False, quality=True)
    result = predict_scene_residuals(host, make_scene(host, 2), make_boxes(2))
    assert "context_coverage" not in result
    assert result["per_agent_quality"].shape == (2, 2)


@test("quality without distance does not require or compute agent positions")
def test_quality_without_distance():
    host = TinyDualSpaceHost(
        multi=False, quality=True, quality_use_distance=False
    )
    features = torch.zeros(2, 4, 8, 8)
    record_len = torch.tensor([2])
    affine = torch.zeros(1, 2, 2, 2, 3)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    context = build_collab_dual_space_context(
        host,
        features,
        record_len,
        affine,
        ["m1", "m2"],
        pairwise_t_matrix=None,
    )
    assert "agent_positions" not in context["scenes"][0]
    result = predict_scene_residuals(
        host, context["scenes"][0], make_boxes(1)
    )
    assert result["per_agent_quality"].shape == (1, 2)


@test("agent-object distance uses normalized physical coordinates")
def test_distance_normalization():
    host = TinyDualSpaceHost(multi=False, quality=True)
    scene = make_scene(host, 2)
    scene["agent_positions"] = torch.tensor([[0.0, 0.0], [16.0, 0.0]])
    proposals = make_boxes(1)
    distance = normalized_agent_object_distance(
        scene, proposals, host.dual_space_bev_geometry
    )
    expected = 16.0 / (32.0 ** 2 + 32.0 ** 2) ** 0.5
    assert torch.allclose(distance, torch.tensor([[0.0, expected]]), atol=1e-6)


@test("raw ego-to-agent translation is inverted for agent position")
def test_pairwise_direction():
    host = TinyDualSpaceHost(multi=False, quality=True)
    features = torch.zeros(2, 4, 8, 8)
    record_len = torch.tensor([2])
    affine = torch.zeros(1, 2, 2, 2, 3)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    pairwise = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    pairwise[0, 0, 1, 0, 3] = -6.0
    context = build_collab_dual_space_context(
        host,
        features,
        record_len,
        affine,
        ["m1", "m2"],
        pairwise_t_matrix=pairwise,
    )
    positions = context["scenes"][0]["agent_positions"]
    assert torch.allclose(positions[1], torch.tensor([6.0, 0.0]))


if __name__ == "__main__":
    sys.exit(run_registered_tests(TESTS))

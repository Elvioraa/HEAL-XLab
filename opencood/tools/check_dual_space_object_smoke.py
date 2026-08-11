"""CPU synthetic smoke coverage for Dual-Space HEAL DS-V1."""

import inspect
import math
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.loss.dual_space_object_loss import compute_dual_space_object_loss
from opencood.models.sub_modules.dual_space_box_coder import (
    boxes_hwl_to_corners_3d,
    decode_box_residual,
    encode_box_residual,
    wrap_angle,
)
from opencood.models.sub_modules.dual_space_object import (
    ResidualObjectAdapter,
    build_collab_dual_space_context,
    build_single_dual_space_context,
    configure_dual_space_trainability,
    install_dual_space_modules,
    predict_scene_residuals,
    proposal_geometry_raw,
    refine_dual_space_detections,
    route_modality_adapters,
    run_dual_space_training,
    uniform_geometry_consensus,
    validate_dual_space_checkpoint_keys,
)
from opencood.models.sub_modules.dual_space_object_roi import (
    ChunkedRotatedBEVROISampler,
    DualSpaceBEVGeometry,
)
from opencood.models.sub_modules.dual_space_proposal_sampler import (
    DualSpaceTrainingProposalSampler,
)
from opencood.tools.heal_tools import apply_dual_space_merge_ownership


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def dual_config(
    mode="stage1_anchor", active_modality=None, yaw_mode="sin_cos"
):
    version = "ds_v1_1" if yaw_mode == "sin_cos_centered" else "ds_v1"
    config = {
        "enabled": True,
        "version": version,
        "experiment_profile": version,
        "mode": mode,
        "allow_untrained_initialization": mode == "stage1_anchor",
        "roi": {
            "output_size": 5,
            "max_train_proposals": 128,
            "max_infer_proposals": 64,
            "chunk_size": 8,
            "min_coverage": 0.5,
        },
        "adapter": {"type": "residual_1x1", "zero_init": True},
        "object_encoder": {
            "embedding_dim": 128,
            "hidden_channels": 64,
            "pooled_size": 2,
        },
        "geometry_encoder": {"enabled": True, "hidden_dim": 32},
        "refiner": {
            "hidden_dim": 128,
            "yaw_mode": yaw_mode,
            "zero_init_output": True,
        },
        "consensus": {
            "mode": "uniform_geometry_mean",
            "fallback_to_original": True,
        },
        "training_proposals": {
            "source": "gt_jitter",
            "include_gt": True,
            "jitters_per_gt": 1,
            "center_xy_std_rel": 0.10,
            "center_z_std_rel": 0.05,
            "log_size_std": 0.05,
            "yaw_std_deg": 5.0,
            "max_proposals": 128,
        },
        "loss": {
            "object_loss_weight": 0.2,
            "individual_loss_weight": 1.0,
            "consensus_loss_weight": 1.0,
            "iou_loss_weight": 0.0,
        },
        "multi_scale": {"enabled": False},
        "quality": {"enabled": False},
        "remote_proposal_rescue": {"enabled": False},
    }
    if active_modality is not None:
        config["active_modality"] = active_modality
    return config


class TinyHost(nn.Module):
    def __init__(
        self,
        modalities,
        mode="stage1_anchor",
        active_modality=None,
        channels=4,
        yaw_mode="sin_cos",
    ):
        super().__init__()
        self.modality_name_list = list(modalities)
        self.sensor_type_dict = OrderedDict((name, "lidar") for name in modalities)
        self.base_branches = nn.ModuleDict(
            (name, nn.Conv2d(channels, channels, 1, bias=False))
            for name in modalities
        )
        args = {
            "lidar_range": [-16.0, -16.0, -3.0, 16.0, 16.0, 1.0],
            "dual_space": dual_config(mode, active_modality, yaw_mode),
        }
        for name in modalities:
            args[name] = {"backbone_args": {"num_filters": [channels]}}
        install_dual_space_modules(self, args)
        configure_dual_space_trainability(self)
        # Synthetic hosts emulate a successfully validated Stage1/merged load.
        self._dual_space_checkpoint_ready = True


def make_geometry():
    return DualSpaceBEVGeometry(-16.0, -16.0, -3.0, 16.0, 16.0, 1.0)


def make_boxes(count=1, device="cpu"):
    boxes = torch.zeros(count, 7, device=device)
    boxes[:, 3] = 1.5
    boxes[:, 4] = 2.0
    boxes[:, 5] = 4.0
    return boxes


def make_scene(features, modalities, support=None):
    if support is None:
        support = features.new_ones((features.shape[0], 1, features.shape[2], features.shape[3]))
    return {
        "agent_features": features,
        "agent_support": support,
        "agent_modalities": tuple(modalities),
    }


def make_object_payload(result, targets, config, mode):
    target_residuals = encode_box_residual(
        result["proposals"],
        targets,
        yaw_mode=config["refiner"]["yaw_mode"],
    )
    indices = result["valid_mask"].nonzero(as_tuple=False)
    result = dict(result)
    result["target_residuals"] = target_residuals
    result["individual_targets"] = target_residuals.index_select(0, indices[:, 0])
    return {
        "enabled": True,
        "mode": mode,
        "scenes": (result,),
        "loss_config": config["loss"],
        "stats": {
            "object_roi_count": int(targets.shape[0]),
            "valid_agent_object_pairs": int(indices.shape[0]),
            "valid_object_ratio": 1.0,
            "mean_roi_coverage": 1.0,
        },
    }


@test("disabled baseline is parameter- and output-inert")
def test_disabled_baseline():
    class Baseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.modality_name_list = ["m1"]
            self.sensor_type_dict = OrderedDict(m1="lidar")
            self.layer = nn.Conv2d(2, 2, 1)

        def forward(self, value):
            return self.layer(value)

    torch.manual_seed(1)
    model = Baseline()
    value = torch.randn(1, 2, 4, 4)
    before = model(value).detach().clone()
    keys_before = tuple(model.state_dict().keys())
    install_dual_space_modules(
        model,
        {"lidar_range": [-1, -1, -1, 1, 1, 1], "m1": {"backbone_args": {"num_filters": [2]}}},
    )
    assert model.dual_space_enabled is False
    assert tuple(model.state_dict().keys()) == keys_before
    assert torch.equal(model(value), before)


@test("8D residual encode/decode round trip")
def test_box_roundtrip():
    proposals = make_boxes(8)
    proposals[:, :3] = torch.randn(8, 3)
    proposals[:, 3:6] = torch.rand(8, 3) + 1.0
    proposals[:, 6] = torch.linspace(-2.8, 2.8, 8)
    targets = proposals.clone()
    targets[:, :3] += torch.randn(8, 3) * 0.3
    targets[:, 3:6] *= torch.exp(torch.randn(8, 3) * 0.1)
    targets[:, 6] = wrap_angle(targets[:, 6] + torch.randn(8) * 0.2)
    residuals = encode_box_residual(proposals, targets, yaw_mode="sin_cos")
    decoded = decode_box_residual(proposals, residuals, yaw_mode="sin_cos")
    assert torch.max(torch.abs(decoded[:, :6] - targets[:, :6])).item() < 2e-5
    assert torch.max(torch.abs(wrap_angle(decoded[:, 6] - targets[:, 6]))).item() < 2e-5


@test("yaw wrap handles +179 versus -179 degrees")
def test_yaw_boundary():
    proposals = make_boxes(2)
    targets = proposals.clone()
    proposals[:, 6] = torch.deg2rad(torch.tensor([179.0, -179.0]))
    targets[:, 6] = torch.deg2rad(torch.tensor([-179.0, 179.0]))
    residual = encode_box_residual(proposals, targets, yaw_mode="sin_cos")
    recovered = decode_box_residual(proposals, residual, yaw_mode="sin_cos")
    assert torch.max(torch.abs(wrap_angle(recovered[:, 6] - targets[:, 6]))).item() < 1e-5
    assert torch.max(torch.abs(torch.atan2(residual[:, 6], residual[:, 7]))).item() < 0.04


@test("legacy sin-cos coder preserves its exact yaw representation")
def test_legacy_yaw_regression():
    proposals = make_boxes(4)
    proposals[:, 6] = torch.tensor([-2.7, -0.2, 0.4, 2.8])
    targets = proposals.clone()
    targets[:, 6] = wrap_angle(
        proposals[:, 6] + torch.tensor([0.0, -0.3, 0.2, 0.5])
    )
    dyaw = wrap_angle(targets[:, 6] - proposals[:, 6])
    residuals = encode_box_residual(
        proposals, targets, yaw_mode="sin_cos"
    )
    assert torch.equal(residuals[:, 6], torch.sin(dyaw))
    assert torch.equal(residuals[:, 7], torch.cos(dyaw))
    decoded = decode_box_residual(
        proposals, residuals, yaw_mode="sin_cos"
    )
    assert torch.equal(decoded[:, :6], targets[:, :6])
    assert torch.equal(
        decoded[:, 6],
        wrap_angle(proposals[:, 6] + torch.atan2(residuals[:, 6], residuals[:, 7])),
    )


@test("centered identity is batched and bit-exact in float32 and float64")
def test_centered_identity_exact():
    for dtype in (torch.float32, torch.float64):
        proposals = make_boxes(6).to(dtype=dtype).reshape(2, 3, 7)
        proposals[..., 0] = torch.arange(6, dtype=dtype).reshape(2, 3)
        proposals[..., 1] = -proposals[..., 0]
        proposals[..., 6] = torch.tensor(
            [-3.0, -0.1, 0.0, 0.1, 1.7, 3.0], dtype=dtype
        ).reshape(2, 3)
        residuals = encode_box_residual(
            proposals, proposals.clone(), yaw_mode="sin_cos_centered"
        )
        assert torch.equal(residuals, torch.zeros_like(residuals))
        decoded = decode_box_residual(
            proposals, torch.zeros_like(residuals),
            yaw_mode="sin_cos_centered",
        )
        assert torch.equal(decoded, proposals)


@test("centered coder round-trips signed yaw including both pi boundaries")
def test_centered_roundtrip():
    offsets = (0.0, 0.01, -0.01, 0.8, math.pi - 1e-4, -math.pi + 1e-4)
    for dtype in (torch.float32, torch.float64):
        proposals = make_boxes(len(offsets)).to(dtype=dtype)
        proposals[:, 0] = torch.linspace(-2.0, 2.0, len(offsets), dtype=dtype)
        proposals[:, 6] = torch.tensor(
            [0.2, -0.4, 0.6, -1.1, 0.0, 0.0], dtype=dtype
        )
        targets = proposals.clone()
        targets[:, 0] += 0.25
        targets[:, 1] -= 0.15
        targets[:, 2] += 0.05
        targets[:, 3:6] *= torch.tensor([1.02, 0.97, 1.04], dtype=dtype)
        targets[:, 6] = wrap_angle(
            proposals[:, 6] + torch.tensor(offsets, dtype=dtype)
        )
        residuals = encode_box_residual(
            proposals, targets, yaw_mode="sin_cos_centered"
        )
        decoded = decode_box_residual(
            proposals, residuals, yaw_mode="sin_cos_centered"
        )
        tolerance = 2e-5 if dtype == torch.float32 else 1e-10
        assert torch.allclose(decoded[:, :6], targets[:, :6], atol=tolerance, rtol=0)
        assert torch.max(torch.abs(wrap_angle(decoded[:, 6] - targets[:, 6]))) < tolerance


@test("box coder rejects every unknown yaw mode")
def test_unknown_yaw_mode():
    proposals = make_boxes(1)
    calls = (
        lambda: encode_box_residual(proposals, proposals, yaw_mode="unknown"),
        lambda: decode_box_residual(
            proposals, torch.zeros(1, 8), yaw_mode="unknown"
        ),
    )
    for call in calls:
        try:
            call()
        except ValueError as error:
            assert "yaw_mode" in str(error)
        else:
            raise AssertionError("unknown yaw mode was silently accepted")


@test("Dual-Space config rejects an unknown yaw mode")
def test_config_unknown_yaw_mode():
    try:
        TinyHost(["m1"], yaw_mode="unknown")
    except ValueError as error:
        assert "refiner.yaw_mode" in str(error)
    else:
        raise AssertionError("unknown config yaw mode was silently accepted")


@test("centered uniform consensus is the legacy circular yaw mean")
def test_centered_uniform_circular_consensus():
    proposal = make_boxes(1)
    angles = torch.deg2rad(torch.tensor([179.0, -179.0, 170.0, -170.0]))
    repeated = proposal.expand(angles.shape[0], -1).clone()
    targets = repeated.clone()
    targets[:, 6] = angles
    legacy = encode_box_residual(repeated, targets, yaw_mode="sin_cos")
    centered = encode_box_residual(
        repeated, targets, yaw_mode="sin_cos_centered"
    )
    valid = torch.ones(1, angles.shape[0], dtype=torch.bool)
    legacy_fused = uniform_geometry_consensus(
        legacy.unsqueeze(0), valid
    )[0]
    centered_fused = uniform_geometry_consensus(
        centered.unsqueeze(0), valid
    )[0]
    legacy_box = decode_box_residual(
        proposal, legacy_fused, yaw_mode="sin_cos"
    )
    centered_box = decode_box_residual(
        proposal, centered_fused, yaw_mode="sin_cos_centered"
    )
    assert torch.allclose(
        wrap_angle(centered_box[:, 6] - legacy_box[:, 6]),
        torch.zeros(1),
        atol=1e-6,
    )
    single_fused = uniform_geometry_consensus(
        centered[:1].unsqueeze(0), torch.ones(1, 1, dtype=torch.bool)
    )[0]
    assert torch.equal(single_fused[0], centered[0])


@test("legacy and centered models keep identical state keys and shapes")
def test_centered_state_schema_compatibility():
    legacy = TinyHost(["m1"], yaw_mode="sin_cos")
    centered = TinyHost(["m1"], yaw_mode="sin_cos_centered")
    legacy_state = legacy.state_dict()
    centered_state = centered.state_dict()
    assert tuple(legacy_state) == tuple(centered_state)
    for key in legacy_state:
        assert legacy_state[key].shape == centered_state[key].shape
    assert legacy.dual_space_config["refiner"]["yaw_mode"] == "sin_cos"
    assert (
        centered.dual_space_config["refiner"]["yaw_mode"]
        == "sin_cos_centered"
    )


def check_roi_shape(agent_count, proposal_count):
    sampler = ChunkedRotatedBEVROISampler(make_geometry(), 5, 8, 0.5)
    features = torch.randn(agent_count, 3, 32, 32)
    proposals = make_boxes(proposal_count)
    rois, valid, coverage = sampler(features, proposals)
    assert rois.shape == (proposal_count, agent_count, 3, 5, 5)
    assert valid.shape == coverage.shape == (proposal_count, agent_count)


@test("ROI shape A=1 N=1")
def test_roi_shape_small():
    check_roi_shape(1, 1)


@test("ROI shape A=3 N=16")
def test_roi_shape_medium():
    check_roi_shape(3, 16)


@test("ROI shape A=5 N=64")
def test_roi_shape_large():
    check_roi_shape(5, 64)


@test("ROI implementation has no proposal-by-full-map expansion")
def test_roi_memory_bound():
    sampler = ChunkedRotatedBEVROISampler(make_geometry(), 5, 4, 0.5)
    sampler(torch.randn(2, 2, 32, 32), make_boxes(19))
    source = inspect.getsource(ChunkedRotatedBEVROISampler.forward)
    assert ".expand(" not in source
    assert sampler.last_debug_stats["full_map_expanded"] is False
    assert sampler.last_debug_stats["max_grid_proposals"] <= 4
    assert sampler.last_debug_stats["max_grid_points"] <= 4 * 5 * 5


@test("rotated ROI follows 0/45/90 degree positive-yaw convention")
def test_rotated_roi_orientation():
    coordinates = torch.arange(32, dtype=torch.float32) - 15.5
    x_map = coordinates.view(1, 32).repeat(32, 1)
    y_map = coordinates.view(32, 1).repeat(1, 32)
    features = torch.stack((x_map, y_map), dim=0).unsqueeze(0)
    proposals = make_boxes(3)
    proposals[:, 4] = 2.0
    proposals[:, 5] = 10.0
    proposals[:, 6] = torch.tensor([0.0, math.pi / 4.0, math.pi / 2.0])
    rois, _, _ = ChunkedRotatedBEVROISampler(make_geometry(), 5, 3, 0.5)(
        features, proposals
    )
    zero = rois[0, 0, :, 2, :]
    diagonal = rois[1, 0, :, 2, :]
    ninety = rois[2, 0, :, 2, :]
    assert zero[0, -1] > zero[0, 0] and (zero[1].max() - zero[1].min()) < 1e-4
    assert diagonal[0, -1] > diagonal[0, 0]
    assert diagonal[1, -1] > diagonal[1, 0]
    assert ninety[1, -1] > ninety[1, 0] and (ninety[0].max() - ninety[0].min()) < 1e-4


@test("ROI coverage distinguishes full partial and outside")
def test_roi_coverage():
    proposals = make_boxes(3)
    proposals[:, 0] = torch.tensor([0.0, 15.0, 40.0])
    proposals[:, 5] = 6.0
    _, valid, coverage = ChunkedRotatedBEVROISampler(
        make_geometry(), 5, 3, 0.5
    )(torch.ones(1, 1, 32, 32), proposals)
    assert coverage[0, 0] == 1
    assert 0 < coverage[1, 0] < 1
    assert coverage[2, 0] == 0
    assert valid[:, 0].tolist() == [True, True, False]


@test("all-invalid agents fall back exactly to original proposal")
def test_no_valid_fallback():
    for yaw_mode in ("sin_cos", "sin_cos_centered"):
        host = TinyHost(
            ["m1", "m2"], mode="inference", yaw_mode=yaw_mode
        )
        features = torch.randn(2, 4, 32, 32)
        support = torch.zeros(2, 1, 32, 32)
        proposals = make_boxes(2)
        proposals[:, 6] = torch.tensor([-0.1, 2.7])
        result = predict_scene_residuals(
            host, make_scene(features, ["m1", "m2"], support), proposals
        )
        assert not bool(result["any_valid"].any())
        assert torch.equal(result["refined_boxes"], proposals)
        assert torch.isfinite(result["refined_boxes"]).all()


@test("single valid agent is the exact consensus residual")
def test_single_valid_consensus():
    residuals = torch.randn(2, 3, 8)
    valid = torch.tensor([[False, True, False], [False, False, True]])
    fused, any_valid = uniform_geometry_consensus(residuals, valid)
    assert any_valid.tolist() == [True, True]
    assert torch.equal(fused[0], residuals[0, 1])
    assert torch.equal(fused[1], residuals[1, 2])


@test("uniform geometry consensus is invariant to agent order")
def test_consensus_order():
    residuals = torch.randn(4, 5, 8)
    valid = torch.rand(4, 5) > 0.3
    order = torch.tensor([3, 0, 4, 1, 2])
    fused_a, valid_a = uniform_geometry_consensus(residuals, valid)
    fused_b, valid_b = uniform_geometry_consensus(
        residuals[:, order], valid[:, order]
    )
    assert torch.allclose(fused_a, fused_b, atol=1e-6)
    assert torch.equal(valid_a, valid_b)


@test("m1 object adapter is parameter-free identity")
def test_m1_identity():
    host = TinyHost(["m1"])
    value = torch.randn(3, 4, 5, 5)
    assert isinstance(host.dual_space_object_adapter_m1, nn.Identity)
    assert torch.equal(host.dual_space_object_adapter_m1(value), value)
    assert list(host.dual_space_object_adapter_m1.parameters()) == []


@test("m2/m3/m4 residual adapters are identity at zero init")
def test_adapter_zero_init():
    value = torch.randn(2, 4, 5, 5)
    for _ in ("m2", "m3", "m4"):
        adapter = ResidualObjectAdapter(4)
        assert torch.equal(adapter(value), value)


@test("shared refiner zero init decodes to original proposal")
def test_refiner_zero_init():
    for yaw_mode in ("sin_cos", "sin_cos_centered"):
        host = TinyHost(["m1"], yaw_mode=yaw_mode)
        roi = torch.randn(2, 4, 5, 5)
        proposals = make_boxes(2)
        z = host.dual_space_shared_object_encoder(roi)
        g = host.dual_space_shared_geometry_encoder(
            proposal_geometry_raw(proposals, host.dual_space_bev_geometry)
        )
        residuals = host.dual_space_shared_object_refiner(z, g)
        assert torch.equal(residuals, torch.zeros_like(residuals))
        assert torch.equal(
            decode_box_residual(proposals, residuals, yaw_mode=yaw_mode),
            proposals,
        )


@test("Stage1 object loss reaches shared modules and base BEV branch")
def test_stage1_gradients():
    torch.manual_seed(4)
    host = TinyHost(["m1"], mode="stage1_anchor")
    optimizer = torch.optim.SGD(host.parameters(), lr=0.1)
    proposals = make_boxes(2).detach()
    targets = proposals.clone()
    targets[:, 0] += 0.5

    def object_loss():
        features = host.base_branches["m1"](torch.randn(1, 4, 32, 32))
        result = predict_scene_residuals(
            host, make_scene(features, ["m1"]), proposals
        )
        payload = make_object_payload(
            result, targets, host.dual_space_config, "stage1_anchor"
        )
        return compute_dual_space_object_loss(payload)[0]

    object_loss().backward()
    final_layer = host.dual_space_shared_object_refiner.network[-1]
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad) > 0
    optimizer.step()
    optimizer.zero_grad()

    object_loss().backward()
    assert host.base_branches["m1"].weight.grad is not None
    assert torch.count_nonzero(host.base_branches["m1"].weight.grad) > 0
    assert any(
        parameter.grad is not None
        for parameter in host.dual_space_shared_object_encoder.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in host.dual_space_shared_object_refiner.parameters()
    )
    assert proposals.requires_grad is False


@test("Stage1 full GT-jitter payload and object-loss helper path")
def test_stage1_training_payload():
    host = TinyHost(["m1"], mode="stage1_anchor")
    context = {
        "scenes": (make_scene(torch.randn(2, 4, 32, 32), ["m1", "m1"]),)
    }
    gt_boxes = make_boxes(3).unsqueeze(0)
    gt_boxes[0, :, 0] = torch.tensor([-4.0, 0.0, 4.0])
    data = {
        "object_bbx_center": gt_boxes,
        "object_bbx_mask": torch.tensor([[True, True, False]]),
    }
    torch.manual_seed(13)
    payload = run_dual_space_training(host, context, data)
    loss, stats = compute_dual_space_object_loss(payload)
    assert len(payload["scenes"]) == 1
    assert payload["stats"]["object_roi_count"] == 4
    assert stats["dual_space_valid_agent_object_pairs"] > 0
    assert torch.isfinite(loss)


@test("Stage2-m2 gradients respect backward-adaptation freeze boundary")
def test_stage2_gradients():
    torch.manual_seed(5)
    host = TinyHost(["m1", "m2", "m3", "m4"], "stage2_adapt", "m2")
    with torch.no_grad():
        host.dual_space_shared_object_refiner.network[-1].weight.normal_(0, 0.02)
    features = host.base_branches["m2"](torch.randn(1, 4, 32, 32))
    proposals = make_boxes(2)
    targets = proposals.clone()
    targets[:, 1] += 0.4
    result = predict_scene_residuals(host, make_scene(features, ["m2"]), proposals)
    payload = make_object_payload(result, targets, host.dual_space_config, "stage2_adapt")
    loss, _ = compute_dual_space_object_loss(payload)
    loss.backward()
    adapter_parameters = list(host.dual_space_object_adapter_m2.parameters())
    assert any(parameter.grad is not None for parameter in adapter_parameters)
    assert host.base_branches["m2"].weight.grad is not None
    for module in (
        host.dual_space_shared_object_encoder,
        host.dual_space_shared_geometry_encoder,
        host.dual_space_shared_object_refiner,
    ):
        assert all(parameter.grad is None for parameter in module.parameters())
        assert all(not parameter.requires_grad for parameter in module.parameters())
    assert host.base_branches["m1"].weight.grad is None
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for modality in ("m3", "m4")
        for parameter in getattr(host, "dual_space_object_adapter_%s" % modality).parameters()
    )


@test("Stage2 single-context helper produces adaptation-only object loss")
def test_stage2_training_payload():
    host = TinyHost(["m2"], "stage2_adapt", "m2")
    feature = torch.randn(2, 4, 32, 32)
    context = build_single_dual_space_context(host, feature, "m2")
    gt_boxes = make_boxes(2).unsqueeze(1)
    data = {
        "object_bbx_center": gt_boxes,
        "object_bbx_mask": torch.ones(2, 1, dtype=torch.bool),
    }
    payload = run_dual_space_training(host, context, data)
    loss, stats = compute_dual_space_object_loss(payload)
    assert len(payload["scenes"]) == 2
    assert stats["dual_space_consensus_loss"] == 0.0
    assert torch.isfinite(loss)


@test("Stage2 m3 and m4 select only their own adapter")
def test_stage2_active_adapter_configs():
    for active in ("m3", "m4"):
        host = TinyHost(["m1", "m2", "m3", "m4"], "stage2_adapt", active)
        for modality in ("m2", "m3", "m4"):
            trainable = any(
                parameter.requires_grad
                for parameter in getattr(
                    host, "dual_space_object_adapter_%s" % modality
                ).parameters()
            )
            assert trainable == (modality == active)


@test("mixed modality routing follows labels rather than positions")
def test_modality_routing():
    class AddConstant(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, tensor):
            return tensor + self.value

    host = TinyHost(["m1", "m2", "m3", "m4"], mode="inference")
    for index, modality in enumerate(("m1", "m2", "m3", "m4"), start=1):
        setattr(host, "dual_space_object_adapter_%s" % modality, AddConstant(index))
    labels = ["m3", "m1", "m4", "m2"]
    routed = route_modality_adapters(host, torch.zeros(4, 4, 2, 2), labels)
    assert torch.equal(routed[:, 0, 0, 0], torch.tensor([3.0, 1.0, 4.0, 2.0]))


@test("object forward accepts variable CAV counts 1 through 5")
def test_variable_cav():
    host = TinyHost(["m1"], mode="inference")
    proposals = make_boxes(3)
    for agent_count in range(1, 6):
        result = predict_scene_residuals(
            host,
            make_scene(torch.randn(agent_count, 4, 32, 32), ["m1"] * agent_count),
            proposals,
        )
        assert result["per_agent_residuals"].shape == (3, agent_count, 8)


@test("inference refines top 64 and preserves remaining 36 boxes and all scores")
def test_inference_topk_preservation():
    host = TinyHost(["m1"], mode="inference")
    host.eval()
    with torch.no_grad():
        final = host.dual_space_shared_object_refiner.network[-1]
        final.bias[0] = 0.1
        final.bias[7] = 1.0
    boxes = make_boxes(100)
    boxes[:, 0] = torch.linspace(-8.0, 8.0, 100)
    corners = boxes_hwl_to_corners_3d(boxes)
    scores = torch.linspace(0.0, 1.0, 100)
    context = {"scenes": (make_scene(torch.ones(1, 4, 32, 32), ["m1"]),)}
    refined, returned_scores = refine_dual_space_detections(
        host, corners, scores, context
    )
    changed = torch.any(refined != corners, dim=(1, 2))
    assert int(changed.sum()) == 64
    assert torch.equal(refined[:36], corners[:36])
    assert refined.shape[0] == 100
    assert returned_scores.data_ptr() == scores.data_ptr()
    assert torch.equal(returned_scores, scores)


@test("remote-valid evidence refines even when ego support is invalid")
def test_remote_only():
    host = TinyHost(["m1", "m2"], mode="inference")
    host.eval()
    with torch.no_grad():
        final = host.dual_space_shared_object_refiner.network[-1]
        final.bias[1] = 0.2
        final.bias[7] = 1.0
    support = torch.ones(2, 1, 32, 32)
    support[0].zero_()
    proposal = make_boxes(1)
    result = predict_scene_residuals(
        host,
        make_scene(torch.ones(2, 4, 32, 32), ["m1", "m2"], support),
        proposal,
    )
    assert result["valid_mask"].tolist() == [[False, True]]
    assert result["any_valid"].item() is True
    assert not torch.equal(result["refined_boxes"], proposal)


@test("enabled/disabled state dict compatibility is explicit")
def test_state_dict_contract():
    enabled = TinyHost(["m1", "m2", "m3", "m4"], mode="inference")
    clone = TinyHost(["m1", "m2", "m3", "m4"], mode="inference")
    keys = enabled.state_dict().keys()
    assert any(key.startswith("dual_space_shared_object_encoder.") for key in keys)
    assert any(key.startswith("dual_space_object_adapter_m2.") for key in keys)
    clone.load_state_dict(enabled.state_dict(), strict=True)

    class Disabled(nn.Module):
        def __init__(self):
            super().__init__()
            self.modality_name_list = ["m1"]
            self.sensor_type_dict = OrderedDict(m1="lidar")
            self.weight = nn.Parameter(torch.ones(1))
    disabled = Disabled()
    install_dual_space_modules(
        disabled,
        {"lidar_range": [-1, -1, -1, 1, 1, 1], "m1": {"backbone_args": {"num_filters": [1]}}},
    )
    assert not any(key.startswith("dual_space_") for key in disabled.state_dict())


@test("checkpoint policy rejects old inference weights and accepts Stage1 warm start")
def test_checkpoint_policy():
    old_keys = {"encoder_m1.weight"}
    inference_host = TinyHost(["m1", "m2"], mode="inference")
    try:
        validate_dual_space_checkpoint_keys(inference_host, old_keys)
    except RuntimeError as error:
        assert "missing trained dual-space keys" in str(error)
    else:
        raise AssertionError("old checkpoint was silently accepted for DS inference")
    stage1_host = TinyHost(["m1"], mode="stage1_anchor")
    validate_dual_space_checkpoint_keys(stage1_host, old_keys)
    stage1_keys = set(stage1_host.state_dict().keys())
    dual_keys = {key for key in stage1_keys if key.startswith("dual_space_")}
    partial_dual = old_keys | {next(iter(dual_keys))}
    try:
        validate_dual_space_checkpoint_keys(stage1_host, partial_dual)
    except RuntimeError as error:
        assert "missing trained dual-space keys" in str(error)
    else:
        raise AssertionError("partial Stage1 dual-space checkpoint was accepted")
    validate_dual_space_checkpoint_keys(stage1_host, old_keys | stage1_keys)


@test("Stage2/inference runtime rejects unvalidated random object weights")
def test_runtime_checkpoint_guard():
    host = TinyHost(["m1", "m2"], mode="inference")
    host._dual_space_checkpoint_ready = False
    try:
        predict_scene_residuals(
            host,
            make_scene(torch.ones(2, 4, 32, 32), ["m1", "m2"]),
            make_boxes(1),
        )
    except RuntimeError as error:
        assert "load and validate" in str(error)
    else:
        raise AssertionError("unvalidated random DS inference was allowed")


@test("Stage2 checkpoint policy requires Stage1 shared object backend")
def test_stage2_checkpoint_policy():
    stage1 = TinyHost(["m1"], mode="stage1_anchor")
    stage2 = TinyHost(["m2"], mode="stage2_adapt", active_modality="m2")
    validate_dual_space_checkpoint_keys(stage2, stage1.state_dict().keys())
    try:
        validate_dual_space_checkpoint_keys(stage2, {"encoder_m1.weight"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("Stage2 accepted a checkpoint without shared DS weights")


@test("merge ownership selects m1 shared and m2/m3/m4 adapters")
def test_merge_ownership():
    stage1 = OrderedDict(
        (
            ("dual_space_shared_object_encoder.weight", torch.tensor([11.0])),
            ("dual_space_shared_geometry_encoder.weight", torch.tensor([12.0])),
            ("dual_space_shared_object_refiner.weight", torch.tensor([13.0])),
        )
    )
    stage2 = []
    for value, modality in zip((22.0, 33.0, 44.0), ("m2", "m3", "m4")):
        stage2.append(
            OrderedDict(
                (("dual_space_object_adapter_%s.weight" % modality, torch.tensor([value])),)
            )
        )
    wrong = OrderedDict((key, torch.tensor([-1.0])) for key in stage1)
    merged = apply_dual_space_merge_ownership(wrong, stage2 + [stage1])
    assert merged["dual_space_shared_object_encoder.weight"].item() == 11.0
    assert merged["dual_space_shared_geometry_encoder.weight"].item() == 12.0
    assert merged["dual_space_shared_object_refiner.weight"].item() == 13.0
    assert merged["dual_space_object_adapter_m2.weight"].item() == 22.0
    assert merged["dual_space_object_adapter_m3.weight"].item() == 33.0
    assert merged["dual_space_object_adapter_m4.weight"].item() == 44.0


@test("GT-jitter sampler is torch-seeded, capped, and detached")
def test_proposal_sampler():
    config = dual_config()["training_proposals"]
    sampler = DualSpaceTrainingProposalSampler(config, max_proposals=3)
    gt = make_boxes(2).requires_grad_(True)
    mask = torch.tensor([True, True])
    torch.manual_seed(7)
    first, targets_first = sampler(gt, mask, with_jitter=True)
    torch.manual_seed(7)
    second, targets_second = sampler(gt, mask, with_jitter=True)
    assert first.shape == targets_first.shape == (3, 7)
    assert torch.equal(first, second) and torch.equal(targets_first, targets_second)
    assert not first.requires_grad and not targets_first.requires_grad


@test("proposal geometry normalization uses configured lidar range")
def test_geometry_encoding():
    proposal = make_boxes(1)
    proposal[0, 2] = -1.0
    raw = proposal_geometry_raw(proposal, make_geometry())
    assert torch.allclose(raw[0, :3], torch.zeros(3), atol=1e-6)
    assert torch.allclose(raw[0, 3:6], torch.log(torch.tensor([4.0, 2.0, 1.5])))


def make_dtype_regression_case(device="cpu"):
    torch.manual_seed(17)
    host = TinyHost(["m1", "m2"], mode="inference").to(device).eval()
    features = torch.randn(2, 4, 32, 32, device=device, dtype=torch.float32)
    proposals = make_boxes(3, device=device)
    proposals[:, 0] = proposals.new_tensor([-4.0, 0.0, 4.0])
    proposals[:, 1] = proposals.new_tensor([2.0, -2.0, 1.0])
    with torch.no_grad():
        host.dual_space_shared_object_refiner.network[-1].weight.normal_(0, 0.01)
    return host, make_scene(features, ["m1", "m2"]), proposals


@test("float64 proposals are canonicalized at the Object-Space geometry boundary")
def test_float64_proposal_geometry_boundary():
    host, scene, proposals = make_dtype_regression_case()
    proposals = proposals.to(dtype=torch.float64)
    observed = {}

    def geometry_pre_hook(module, inputs):
        observed["geometry_input_dtype"] = inputs[0].dtype
        observed["geometry_weight_dtype"] = next(module.parameters()).dtype

    def refiner_pre_hook(module, inputs):
        observed["object_embedding_dtype"] = inputs[0].dtype
        observed["geometry_embedding_dtype"] = inputs[1].dtype

    handles = (
        host.dual_space_shared_geometry_encoder.register_forward_pre_hook(
            geometry_pre_hook
        ),
        host.dual_space_shared_object_refiner.register_forward_pre_hook(
            refiner_pre_hook
        ),
    )
    try:
        with torch.no_grad():
            result = predict_scene_residuals(host, scene, proposals)
    finally:
        for handle in handles:
            handle.remove()

    assert observed == {
        "geometry_input_dtype": torch.float32,
        "geometry_weight_dtype": torch.float32,
        "object_embedding_dtype": torch.float32,
        "geometry_embedding_dtype": torch.float32,
    }
    assert result["individual_residuals"].dtype == torch.float32
    assert result["fused_residuals"].dtype == torch.float32
    assert torch.isfinite(result["individual_residuals"]).all()
    assert torch.isfinite(result["refined_boxes"]).all()


@test("float32 proposal path remains numerically unchanged")
def test_float32_proposal_path_unchanged():
    host, scene, proposals = make_dtype_regression_case()
    with torch.no_grad():
        float32_result = predict_scene_residuals(host, scene, proposals)
        float64_result = predict_scene_residuals(
            host, scene, proposals.to(dtype=torch.float64)
        )

    assert torch.count_nonzero(float32_result["individual_residuals"]) > 0
    for key in ("individual_residuals", "per_agent_residuals", "fused_residuals"):
        assert torch.allclose(
            float64_result[key], float32_result[key], atol=1e-6, rtol=1e-6
        )
    assert float32_result["refined_boxes"].dtype == torch.float32
    assert float64_result["refined_boxes"].dtype == torch.float64
    assert torch.allclose(
        float64_result["refined_boxes"].float(),
        float32_result["refined_boxes"],
        atol=1e-6,
        rtol=1e-6,
    )


@test("same-forward context spatially warps each agent to ego and retains labels")
def test_common_bev_context_warp():
    host = TinyHost(["m1", "m2"], mode="inference", channels=1)
    features = torch.zeros(2, 1, 16, 16)
    features[:, :, 8, 8] = 1.0
    record_len = torch.tensor([2])
    affine = torch.zeros(1, 2, 2, 2, 3)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    affine[0, 0, 1, 0, 2] = 0.5
    context = build_collab_dual_space_context(
        host, features, record_len, affine, ["m1", "m2"]
    )
    scene = context["scenes"][0]
    ego_peak = torch.argmax(scene["agent_features"][0, 0]).item() % 16
    remote_peak = torch.argmax(scene["agent_features"][1, 0]).item() % 16
    assert ego_peak == 8 and remote_peak < ego_peak
    assert scene["agent_modalities"] == ("m1", "m2")
    assert context["aligned_to"] == "ego"


def main():
    torch.set_num_threads(1)
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
    if torch.cuda.is_available():
        @test("CUDA autocast preserves centered zero identity")
        def test_cuda_centered_identity():
            proposals = make_boxes(4, device="cuda")
            proposals[:, 6] = torch.tensor(
                [-3.0, -0.1, 0.1, 3.0], device="cuda"
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                residuals = encode_box_residual(
                    proposals,
                    proposals.clone(),
                    yaw_mode="sin_cos_centered",
                )
                decoded = decode_box_residual(
                    proposals,
                    residuals,
                    yaw_mode="sin_cos_centered",
                )
            assert residuals.dtype == proposals.dtype
            assert residuals.device == proposals.device
            assert torch.equal(residuals, torch.zeros_like(residuals))
            assert torch.equal(decoded, proposals)


        @test("CUDA autocast accepts float64 proposals at the Object-Space boundary")
        def test_cuda_autocast_float64_proposals():
            host, scene, proposals = make_dtype_regression_case("cuda")
            proposals = proposals.to(dtype=torch.float64)
            observed = {}

            def geometry_pre_hook(module, inputs):
                observed["geometry_input_dtype"] = inputs[0].dtype

            def refiner_pre_hook(module, inputs):
                observed["object_embedding_dtype"] = inputs[0].dtype

            handles = (
                host.dual_space_shared_geometry_encoder.register_forward_pre_hook(
                    geometry_pre_hook
                ),
                host.dual_space_shared_object_refiner.register_forward_pre_hook(
                    refiner_pre_hook
                ),
            )
            try:
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        result = predict_scene_residuals(host, scene, proposals)
            finally:
                for handle in handles:
                    handle.remove()

            assert observed["geometry_input_dtype"] == observed[
                "object_embedding_dtype"
            ]
            assert result["individual_residuals"].device.type == "cuda"
            assert torch.isfinite(result["individual_residuals"]).all()
            assert torch.isfinite(result["refined_boxes"]).all()
    else:
        print(
            "[SKIP] CUDA centered identity regression: "
            "local environment lacks CUDA"
        )
        print(
            "[SKIP] CUDA autocast float64 proposal regression: "
            "local environment lacks CUDA"
        )
    sys.exit(main())

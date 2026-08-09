"""CPU smoke tests for DS-V2 multi-scale Common Object Space."""

import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.models.sub_modules.dual_space_object import (
    SharedAdaptiveScaleGate,
    attach_collab_dual_space_pyramid_context,
    predict_scene_residuals,
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


@test("V1 constructs no multi-scale modules or keys")
def test_v1_absent():
    host = TinyDualSpaceHost(multi=False)
    assert not hasattr(host, "dual_space_shared_context_encoder")
    assert not any("context" in key for key in host.state_dict())


@test("V2 constructs shared context and concat modules")
def test_v2_modules():
    host = TinyDualSpaceHost(multi=True)
    assert hasattr(host, "dual_space_shared_context_encoder")
    assert hasattr(host, "dual_space_shared_multiscale_fusion")
    assert not hasattr(host, "dual_space_shared_scale_gate")


@test("configured detail and context ROI sizes drive separate samplers")
def test_configured_roi_sizes():
    host = TinyDualSpaceHost(
        multi=True, detail_roi_size=7, context_roi_size=3
    )
    assert host.dual_space_object_roi.output_size == (7, 7)
    assert host.dual_space_context_roi.output_size == (3, 3)


@test("V2 state keys are feature-explicit")
def test_v2_keys():
    keys = TinyDualSpaceHost(multi=True).state_dict().keys()
    assert any(key.startswith("dual_space_shared_context_encoder.") for key in keys)
    assert any(key.startswith("dual_space_shared_multiscale_fusion.") for key in keys)
    assert any(key.startswith("dual_space_context_adapter_m2.") for key in keys)


@test("m1 context adapter is parameter-free identity")
def test_context_m1_identity():
    host = TinyDualSpaceHost(multi=True)
    value = torch.randn(2, 6, 3, 3)
    assert isinstance(host.dual_space_context_adapter_m1, nn.Identity)
    assert torch.equal(host.dual_space_context_adapter_m1(value), value)


@test("m2 context residual adapter starts as exact identity")
def test_context_m2_zero_init():
    host = TinyDualSpaceHost(multi=True)
    value = torch.randn(2, 6, 3, 3)
    assert torch.equal(host.dual_space_context_adapter_m2(value), value)


@test("multi-scale output shapes support P>1 A>1")
def test_multiscale_shapes():
    host = TinyDualSpaceHost(multi=True)
    result = predict_scene_residuals(host, make_scene(host, 3), make_boxes(4))
    assert result["per_agent_residuals"].shape == (4, 3, 8)
    assert result["detail_coverage"].shape == (4, 3)
    assert result["context_coverage"].shape == (4, 3)


@test("concat projection initializes as exact detail embedding")
def test_concat_safe_init():
    host = TinyDualSpaceHost(multi=True)
    detail = torch.randn(5, 16)
    context = torch.randn(5, 16)
    output = host.dual_space_shared_multiscale_fusion(detail, context)
    assert torch.equal(output, detail)


@test("trained concat projection responds to context")
def test_concat_uses_context():
    host = TinyDualSpaceHost(multi=True)
    module = host.dual_space_shared_multiscale_fusion
    module.residual_scale.data.fill_(1.0)
    detail = torch.randn(5, 16)
    context_a = torch.zeros(5, 16)
    context_b = torch.ones(5, 16)
    assert not torch.allclose(module(detail, context_a), module(detail, context_b))


@test("adaptive scalar gate is bounded and starts detail-dominant")
def test_adaptive_gate():
    gate = SharedAdaptiveScaleGate(16)
    detail = torch.ones(3, 16)
    context = torch.zeros(3, 16)
    output, values = gate(detail, context)
    assert values.shape == (3, 1)
    assert bool(((values >= 0) & (values <= 1)).all())
    assert float(values.min()) > 0.97
    assert float(output.mean()) > 0.97


@test("multi-scale gradients reach detail and context features")
def test_multiscale_gradients():
    host = TinyDualSpaceHost(multi=True)
    host.dual_space_shared_multiscale_fusion.residual_scale.data.fill_(1.0)
    scene = make_scene(host, 2)
    scene["agent_features"].requires_grad_(True)
    scene["context_agent_features"].requires_grad_(True)
    final = host.dual_space_shared_object_refiner.network[-1]
    final.weight.data.normal_(0.0, 0.02)
    result = predict_scene_residuals(host, scene, make_boxes(2))
    result["fused_residuals"].sum().backward()
    assert scene["agent_features"].grad is not None
    assert scene["context_agent_features"].grad is not None
    assert torch.isfinite(scene["context_agent_features"].grad).all()


@test("both ROI paths remain proposal-chunk bounded")
def test_multiscale_memory_bound():
    host = TinyDualSpaceHost(multi=True)
    predict_scene_residuals(host, make_scene(host, 2), make_boxes(19))
    detail = host.dual_space_object_roi.last_debug_stats
    context = host.dual_space_context_roi.last_debug_stats
    assert detail["full_map_expanded"] is False
    assert context["full_map_expanded"] is False
    assert detail["max_grid_proposals"] <= 4
    assert context["max_grid_points"] <= 4 * 3 * 3


@test("context ROI contract covers A=1/3/5 and N=1/16/64")
def test_context_roi_contract_matrix():
    host = TinyDualSpaceHost(multi=True)
    for agent_count in (1, 3, 5):
        for proposal_count in (1, 16, 64):
            result = predict_scene_residuals(
                host,
                make_scene(host, agent_count),
                make_boxes(proposal_count),
            )
            assert result["detail_coverage"].shape == (
                proposal_count, agent_count
            )
            assert result["context_coverage"].shape == (
                proposal_count, agent_count
            )
            assert result["per_agent_residuals"].shape == (
                proposal_count, agent_count, 8
            )


@test("full A=5 N=64 profile remains bounded by runtime ROI chunks")
def test_full_profile_runtime_bound():
    host = TinyDualSpaceHost(multi=True)
    predict_scene_residuals(host, make_scene(host, 5), make_boxes(64))
    detail = host.dual_space_object_roi.last_debug_stats
    context = host.dual_space_context_roi.last_debug_stats
    assert detail["full_map_expanded"] is False
    assert context["full_map_expanded"] is False
    assert detail["max_grid_proposals"] == 4
    assert detail["max_grid_points"] == 4 * 5 * 5
    assert context["max_grid_proposals"] == 4
    assert context["max_grid_points"] == 4 * 3 * 3


@test("empty proposals preserve all multi-scale output shapes")
def test_multiscale_empty():
    host = TinyDualSpaceHost(multi=True)
    result = predict_scene_residuals(host, make_scene(host, 2), make_boxes(0))
    assert result["per_agent_residuals"].shape == (0, 2, 8)
    assert result["detail_coverage"].shape == (0, 2)
    assert result["context_coverage"].shape == (0, 2)


@test("Stage2 trains only active detail and context adapters")
def test_stage2_multiscale_boundary():
    host = TinyDualSpaceHost(
        modalities=("m1", "m2", "m3"),
        mode="stage2_adapt",
        active_modality="m2",
        multi=True,
    )
    assert any(p.requires_grad for p in host.dual_space_object_adapter_m2.parameters())
    assert any(p.requires_grad for p in host.dual_space_context_adapter_m2.parameters())
    assert all(not p.requires_grad for p in host.dual_space_shared_context_encoder.parameters())
    assert all(not p.requires_grad for p in host.dual_space_context_adapter_m3.parameters())


@test("pyramid hook captures level-1 pre-fusion tensor in ego space")
def test_pyramid_capture():
    host = TinyDualSpaceHost(multi=True)
    detail = torch.zeros(2, 4, 16, 16)
    detail[:, :, 8, 8] = 1.0
    context_level = torch.zeros(2, 6, 8, 8)
    context_level[:, :, 4, 4] = 1.0
    level2 = torch.zeros(2, 8, 4, 4)
    record_len = torch.tensor([2])
    affine = torch.zeros(1, 2, 2, 2, 3)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    context = {
        "scenes": ({"agent_modalities": ("m1", "m2")},),
        "aligned_to": "ego",
    }
    attach_collab_dual_space_pyramid_context(
        host,
        context,
        (detail, context_level, level2),
        record_len,
        affine,
        ["m1", "m2"],
    )
    captured = context["scenes"][0]["context_agent_features"]
    assert captured.shape == (2, 6, 8, 8)
    assert torch.allclose(captured, context_level, atol=1e-6, rtol=0.0)


if __name__ == "__main__":
    if torch.cuda.is_available():
        @test("CUDA AMP A=5 N=64 profile is finite and chunk-bounded")
        def test_cuda_amp_memory_profile():
            host = TinyDualSpaceHost(multi=True).cuda()
            scene = make_scene(host, 5)
            for key in (
                "agent_features",
                "agent_support",
                "context_agent_features",
                "context_agent_support",
            ):
                scene[key] = scene[key].cuda()
            proposals = make_boxes(64).cuda()
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = predict_scene_residuals(host, scene, proposals)
            torch.cuda.synchronize()
            assert torch.isfinite(result["fused_residuals"]).all()
            assert host.dual_space_object_roi.last_debug_stats[
                "full_map_expanded"
            ] is False
            assert host.dual_space_context_roi.last_debug_stats[
                "full_map_expanded"
            ] is False
            peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            print("CUDA peak allocated MiB: %.2f" % peak_mib)
    else:
        print("[SKIP] CUDA AMP/memory profile: local environment lacks CUDA")
    sys.exit(run_registered_tests(TESTS))

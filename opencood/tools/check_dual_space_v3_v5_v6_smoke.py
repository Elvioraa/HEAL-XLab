"""CPU smoke tests for optional Dual-Space diagnostics, V5, and V6."""

import copy
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.loss.dual_space_object_loss import compute_dual_space_object_loss
from opencood.models.sub_modules.dual_space_config import (
    dual_space_feature_flags,
    resolve_dual_space_diagnostics,
    resolve_v5_quality_safe_config,
    resolve_v6_residual_safe_config,
    validate_dual_space_config,
)
from opencood.models.sub_modules.dual_space_diagnostics import (
    attach_dual_space_training_diagnostics,
    close_dual_space_training_diagnostics,
    loss_gradient_norms,
)
from opencood.models.sub_modules.dual_space_extensions import (
    apply_residual_norm_cap,
    build_quality_target_mask,
    cap_quality_loss_ratio,
    deterministic_quality_ranking_loss,
)
from opencood.models.sub_modules.dual_space_object import (
    predict_scene_residuals,
    route_modality_adapters,
    run_dual_space_training,
)
from opencood.tools.dual_space_smoke_common import (
    TinyDualSpaceHost,
    make_boxes,
    make_scene,
    run_registered_tests,
)
from opencood.tools.prepare_dual_space_stage2 import preflight_stage2_configs


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


DIAGNOSTICS_DISABLED = {
    "enabled": False,
    "quality_target": {"enabled": True},
    "adapter_residual": {"enabled": True},
    "gradient_flow": {"enabled": True},
    "inference_ablation": {
        "enabled": True,
        "bypass_object_adapter": True,
        "bypass_context_adapter": True,
        "bypass_quality_weighting": True,
    },
}
V5_DISABLED = {
    "enabled": False,
    "valid_target_mask": {"enabled": True},
    "loss_balance": {"enabled": True},
    "ranking": {"enabled": True, "weight": 1.0},
}
V6_DISABLED = {
    "enabled": False,
    "object": {"enabled": True},
    "context": {"enabled": True},
}


def _host(seed, **kwargs):
    torch.manual_seed(seed)
    return TinyDualSpaceHost(
        modalities=("m2",),
        mode="stage2_adapt",
        active_modality="m2",
        multi=True,
        quality=True,
        **kwargs
    )


def _trainable_contract(model):
    return tuple(
        (name, parameter.requires_grad, parameter.numel())
        for name, parameter in model.named_parameters()
    )


@test("legacy V3 config compatibility")
def test_legacy_config():
    path = os.path.join(
        REPO_ROOT,
        "opencood",
        "hypes_yaml",
        "HEAL_XLab_v4_DUAL_SPACE",
        "DS_V3",
        "stage2_m2.yaml",
    )
    dual = load_yaml(path, None)["model"]["args"]["dual_space"]
    assert resolve_dual_space_diagnostics(dual)["enabled"] is False
    assert resolve_v5_quality_safe_config(dual)["enabled"] is False
    assert resolve_v6_residual_safe_config(dual)["enabled"] is False


@test("disabled extensions preserve state_dict keys and tensor values")
def test_disabled_state_dict():
    legacy = _host(11)
    disabled = _host(
        11,
        diagnostics=DIAGNOSTICS_DISABLED,
        v5_quality_safe=V5_DISABLED,
        v6_residual_safe=V6_DISABLED,
    )
    assert set(legacy.state_dict()) == set(disabled.state_dict())
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[key]), key


@test("disabled extensions preserve trainable parameter contract")
def test_disabled_trainable_contract():
    legacy = _host(13)
    disabled = _host(
        13,
        diagnostics=DIAGNOSTICS_DISABLED,
        v5_quality_safe=V5_DISABLED,
        v6_residual_safe=V6_DISABLED,
    )
    assert _trainable_contract(legacy) == _trainable_contract(disabled)


@test("disabled diagnostics create no recorder attribute or output directory")
def test_disabled_diagnostics_resource_noop():
    host = _host(14, diagnostics=DIAGNOSTICS_DISABLED)
    with tempfile.TemporaryDirectory() as directory:
        recorder = attach_dual_space_training_diagnostics(host, directory)
        assert recorder is None
        assert not hasattr(host, "_dual_space_training_diagnostics")
        assert not os.path.exists(os.path.join(directory, "diagnostics"))


@test("enabled V5 and V6 remain parameter-free and key-compatible")
def test_enabled_extensions_parameter_free():
    legacy = _host(15)
    extended = _host(
        15,
        v5_quality_safe={
            "enabled": True,
            "valid_target_mask": {"enabled": True},
            "loss_balance": {"enabled": True},
        },
        v6_residual_safe={
            "enabled": True,
            "object": {"enabled": True},
            "context": {"enabled": True},
        },
    )
    assert set(legacy.state_dict()) == set(extended.state_dict())
    assert _trainable_contract(legacy) == _trainable_contract(extended)


@test("disabled extensions preserve deterministic V3 forward exactly")
def test_disabled_forward_identity():
    legacy = _host(17)
    disabled = _host(
        17,
        diagnostics=DIAGNOSTICS_DISABLED,
        v5_quality_safe=V5_DISABLED,
        v6_residual_safe=V6_DISABLED,
    )
    torch.manual_seed(19)
    scene = make_scene(legacy, agent_count=1)
    proposals = make_boxes(3)
    torch.manual_seed(20)
    rng_before = torch.random.get_rng_state()
    with torch.no_grad():
        first = predict_scene_residuals(legacy, scene, proposals)
        rng_after_legacy = torch.random.get_rng_state()
        torch.random.set_rng_state(rng_before)
        second = predict_scene_residuals(disabled, scene, proposals)
        rng_after_disabled = torch.random.get_rng_state()
    assert torch.equal(rng_after_legacy, rng_after_disabled)
    assert first.keys() == second.keys()
    for key in first:
        if torch.is_tensor(first[key]):
            assert torch.equal(first[key], second[key]), key
        else:
            assert first[key] == second[key], key


@test("adapter diagnostics observer is prediction-invariant")
def test_diagnostics_observer_identity():
    diagnostics = {
        "enabled": True,
        "every_n_steps": 1,
        "adapter_residual": {"enabled": True},
    }
    baseline = _host(23)
    observed = _host(23, diagnostics=diagnostics)
    torch.manual_seed(29)
    scene = make_scene(baseline, agent_count=1)
    proposals = make_boxes(2)
    with tempfile.TemporaryDirectory() as directory:
        recorder = attach_dual_space_training_diagnostics(observed, directory)
        assert recorder is not None
        with torch.no_grad():
            first = predict_scene_residuals(baseline, scene, proposals)
            second = predict_scene_residuals(observed, scene, proposals)
        close_dual_space_training_diagnostics(observed)
    for key in first:
        if torch.is_tensor(first[key]):
            assert torch.equal(first[key], second[key]), key


@test("adapter diagnostic log exposes the complete residual summary")
def test_adapter_diagnostic_fields():
    messages = []
    diagnostics = {
        "enabled": True,
        "every_n_steps": 1,
        "adapter_residual": {"enabled": True},
    }
    host = _host(24, diagnostics=diagnostics)
    with tempfile.TemporaryDirectory() as directory:
        recorder = attach_dual_space_training_diagnostics(
            host, directory, print_fn=messages.append
        )
        recorder.begin_forward()
        inputs = torch.ones(2, 4, 2, 2)
        residual = torch.full_like(inputs, 0.25)
        recorder.record_adapter(
            "m2", "object", inputs, residual, inputs + residual, feature_dim=1
        )
        recorder.end_forward()
        close_dual_space_training_diagnostics(host)
    adapter_lines = [line for line in messages if "[DSDiag][Adapter]" in line]
    assert len(adapter_lines) == 1
    for field in (
        "input_norm=", "residual_norm=", "output_norm=", "ratio=",
        "cos_in_out=", "residual_mean=", "residual_std=",
        "residual_abs_max=",
    ):
        assert field in adapter_lines[0]


@test("training diagnostics do not collect validation eval forwards")
def test_training_diagnostics_skip_eval():
    messages = []
    diagnostics = {
        "enabled": True,
        "every_n_steps": 1,
        "quality_target": {
            "enabled": True,
            "max_records": 4,
            "dump_jsonl": True,
        },
        "adapter_residual": {"enabled": True},
    }
    host = _host(26, diagnostics=diagnostics)
    host.eval()
    scene = make_scene(host, agent_count=1)
    boxes = make_boxes(1)
    data = {
        "object_bbx_center": boxes.unsqueeze(0),
        "object_bbx_mask": torch.ones(1, 1),
    }
    with tempfile.TemporaryDirectory() as directory:
        recorder = attach_dual_space_training_diagnostics(
            host, directory, print_fn=messages.append
        )
        with torch.no_grad():
            run_dual_space_training(host, {"scenes": (scene,)}, data)
        assert not recorder._adapter_values
        assert recorder._quality_record_count == 0
        assert messages == []
        assert not os.path.exists(os.path.join(directory, "diagnostics"))
        close_dual_space_training_diagnostics(host)


@test("quality diagnostics record the real Stage2 assignment")
def test_quality_target_recorder():
    diagnostics = {
        "enabled": True,
        "every_n_steps": 1,
        "quality_target": {
            "enabled": True,
            "max_records": 4,
            "dump_jsonl": True,
        },
    }
    host = TinyDualSpaceHost(
        modalities=("m2",),
        mode="stage2_adapt",
        active_modality="m2",
        quality=True,
        diagnostics=diagnostics,
    )
    scene = make_scene(host, agent_count=1)
    boxes = make_boxes(1)
    data = {
        "object_bbx_center": boxes.unsqueeze(0),
        "object_bbx_mask": torch.ones(1, 1),
    }
    with tempfile.TemporaryDirectory() as directory:
        attach_dual_space_training_diagnostics(host, directory)
        torch.manual_seed(31)
        payload = run_dual_space_training(host, {"scenes": (scene,)}, data)
        close_dual_space_training_diagnostics(host)
        path = os.path.join(directory, "diagnostics", "quality_target.jsonl")
        assert os.path.isfile(path)
        with open(path, "r", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    assert rows
    assert rows[0]["matched"] is True
    assert rows[0]["matched_gt_id"] == 0
    assert rows[0]["quality_target"] == rows[0]["refined_iou"]
    assert payload["scenes"][0]["proposal_matched_gt_indices"].numel() > 0


@test("all four extension YAML packs preserve DS-V3 core contracts")
def test_extension_yaml_packs():
    root = os.path.join(
        REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v4_DUAL_SPACE"
    )
    profiles = {
        "DS_V3_DIAG": (
            "ds_v3_diag",
            (True, True, True, True, True),
            (False, False, False, False, False),
            (False, False, False, False, False),
        ),
        "DS_V5_QUALITY_SAFE": (
            "ds_v5_quality_safe",
            (False, False, False, False, False),
            (False, True, True, True, False),
            (False, False, False, False, False),
        ),
        "DS_V6_RESIDUAL_SAFE": (
            "ds_v6_residual_safe",
            (False, False, False, False, False),
            (False, False, False, False, False),
            (False, True, True, True, True),
        ),
        "DS_V5_V6": (
            "ds_v5_v6",
            (False, False, False, False, False),
            (False, True, True, True, False),
            (False, True, True, True, True),
        ),
    }
    filenames = (
        "stage1_m1.yaml", "stage2_m2.yaml", "stage2_m3.yaml",
        "stage2_m4.yaml", "merged_infer.yaml",
    )
    for directory, expected in profiles.items():
        profile, diagnostics, v5_lifecycle, v6_lifecycle = expected
        profile_dir = os.path.join(root, directory)
        prepared_profile, prepared_version, prepared_configs = (
            preflight_stage2_configs(profile_dir)
        )
        assert prepared_profile == profile
        assert prepared_version == "ds_v3"
        assert set(prepared_configs) == {"m2", "m3", "m4"}
        for index, filename in enumerate(filenames):
            dual = load_yaml(os.path.join(profile_dir, filename), None)[
                "model"
            ]["args"]["dual_space"]
            validate_dual_space_config(dual)
            assert dual["version"] == "ds_v3"
            assert dual["experiment_profile"] == profile
            assert resolve_dual_space_diagnostics(dual)["enabled"] is diagnostics[index]
            assert resolve_v5_quality_safe_config(dual)["enabled"] is v5_lifecycle[index]
            assert resolve_v6_residual_safe_config(dual)["enabled"] is v6_lifecycle[index]
            flags = dual_space_feature_flags(dual)
            assert flags["v5_quality_safe"] is v5_lifecycle[index]
            assert flags["v6_residual_safe"] is v6_lifecycle[index]


@test("V5 target mask excludes unmatched nonfinite and out-of-range values")
def test_v5_mask():
    targets = torch.tensor([0.2, float("nan"), float("inf"), -0.1, 1.1, 0.8])
    matched = torch.tensor([True, True, True, True, True, False])
    config = resolve_v5_quality_safe_config(
        {"v5_quality_safe": {"enabled": True}}
    )["valid_target_mask"]
    config["enabled"] = True
    mask = build_quality_target_mask(targets, matched, config)
    assert mask.tolist() == [True, False, False, False, False, False]


@test("V5 zero-valid quality loss is graph-connected zero")
def test_v5_zero_valid_graph():
    residual = torch.zeros(1, 8, requires_grad=True)
    quality = torch.tensor([0.4], requires_grad=True)
    v5 = resolve_v5_quality_safe_config(
        {
            "v5_quality_safe": {
                "enabled": True,
                "valid_target_mask": {"enabled": True},
            }
        }
    )
    payload = {
        "enabled": True,
        "mode": "stage2_adapt",
        "quality_enabled": True,
        "v5_quality_safe": v5,
        "loss_config": {
            "object_loss_weight": 1.0,
            "individual_loss_weight": 0.0,
            "consensus_loss_weight": 0.0,
            "quality_loss_weight": 1.0,
        },
        "stats": {
            "valid_object_ratio": 0.0,
            "mean_roi_coverage": 0.0,
            "object_roi_count": 1,
            "valid_agent_object_pairs": 1,
        },
        "scenes": ({
            "individual_residuals": residual,
            "individual_targets": torch.zeros_like(residual),
            "any_valid": torch.tensor([False]),
            "individual_quality": quality,
            "quality_targets": torch.tensor([float("nan")]),
            "quality_matched_valid": torch.tensor([False]),
            "quality_pair_indices": torch.tensor([[0, 0]]),
        },),
    }
    loss, stats = compute_dual_space_object_loss(payload)
    loss.backward()
    assert quality.grad is not None
    assert torch.equal(quality.grad, torch.zeros_like(quality.grad))
    assert stats["dual_space_v5_valid_quality_count"] == 0


@test("V5 disabled preserves the real Stage2 object loss exactly")
def test_v5_disabled_loss_identity():
    legacy = _host(41)
    disabled = _host(41, v5_quality_safe=V5_DISABLED)
    torch.manual_seed(43)
    scene = make_scene(legacy, agent_count=1)
    boxes = make_boxes(1)
    data = {
        "object_bbx_center": boxes.unsqueeze(0),
        "object_bbx_mask": torch.ones(1, 1),
    }
    torch.manual_seed(47)
    first_payload = run_dual_space_training(
        legacy, {"scenes": (scene,)}, data
    )
    torch.manual_seed(47)
    second_payload = run_dual_space_training(
        disabled, {"scenes": (scene,)}, data
    )
    first_loss, first_stats = compute_dual_space_object_loss(first_payload)
    second_loss, second_stats = compute_dual_space_object_loss(second_payload)
    assert torch.equal(first_loss, second_loss)
    for key in (
        "dual_space_object_loss", "dual_space_individual_loss",
        "dual_space_consensus_loss", "dual_space_quality_loss",
    ):
        assert first_stats[key] == second_stats[key]


@test("V5 integrated Stage2 loss reaches the active adapter")
def test_v5_stage2_integration():
    host = _host(
        53,
        v5_quality_safe={
            "enabled": True,
            "valid_target_mask": {"enabled": True},
            "loss_balance": {"enabled": True},
            "ranking": {"enabled": False, "weight": 0.0},
        },
    )
    scene = make_scene(host, agent_count=1)
    boxes = make_boxes(2)
    data = {
        "object_bbx_center": boxes.unsqueeze(0),
        "object_bbx_mask": torch.ones(1, 2),
    }
    torch.manual_seed(59)
    payload = run_dual_space_training(host, {"scenes": (scene,)}, data)
    loss, stats = compute_dual_space_object_loss(payload)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in host.dual_space_object_adapter_m2.parameters()
        if parameter.requires_grad
    ]
    assert stats["dual_space_v5_valid_quality_count"] > 0
    assert torch.isfinite(loss)
    assert any(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )


@test("V5 ratio cap is exact and detached from detection loss")
def test_v5_ratio_cap():
    config = resolve_v5_quality_safe_config(
        {
            "v5_quality_safe": {
                "enabled": True,
                "loss_balance": {"enabled": True},
            }
        }
    )["loss_balance"]
    detection = torch.tensor(1.0, requires_grad=True)
    quality = torch.tensor(1.0, requires_grad=True)
    balanced, scale = cap_quality_loss_ratio(detection, quality, config)
    assert torch.allclose(scale, torch.tensor(0.25), atol=2e-6)
    assert torch.allclose(balanced, torch.tensor(0.25), atol=2e-6)
    balanced.backward()
    assert detection.grad is None
    assert torch.allclose(quality.grad, scale)
    small, small_scale = cap_quality_loss_ratio(
        torch.tensor(1.0), torch.tensor(0.1), config
    )
    assert small_scale.item() == 1.0
    assert small.item() == torch.tensor(0.1).item()


@test("V5 deterministic ranking rewards correct ordering")
def test_v5_ranking():
    config = resolve_v5_quality_safe_config(
        {
            "v5_quality_safe": {
                "enabled": True,
                "ranking": {"enabled": True, "weight": 1.0},
            }
        }
    )["ranking"]
    targets = torch.tensor([0.9, 0.2])
    indices = torch.tensor([[0, 0], [1, 0]])
    valid = torch.ones(2, dtype=torch.bool)
    good = deterministic_quality_ranking_loss(
        torch.tensor([0.8, 0.3]), targets, indices, valid, config
    )
    bad = deterministic_quality_ranking_loss(
        torch.tensor([0.3, 0.8]), targets, indices, valid, config
    )
    assert good[1] == bad[1] == 1
    assert good[2].item() == 1.0
    assert bad[2].item() == 0.0
    assert good[0] < bad[0]
    none = deterministic_quality_ranking_loss(
        torch.tensor([0.5], requires_grad=True),
        torch.tensor([0.5]),
        torch.tensor([[0, 0]]),
        torch.tensor([True]),
        config,
    )
    assert none[1] == 0 and none[0].requires_grad


@test("V5 ranking aggregates every valid pair and reports cap separately")
def test_v5_ranking_global_pair_mean():
    v5_config = resolve_v5_quality_safe_config(
        {
            "v5_quality_safe": {
                "enabled": True,
                "ranking": {"enabled": True, "weight": 1.0},
            }
        }
    )

    def scene(quality_prediction, quality_target):
        count = quality_prediction.shape[0]
        return {
            "individual_residuals": torch.zeros(count, 8, requires_grad=True),
            "individual_targets": torch.zeros(count, 8),
            "any_valid": torch.ones(count, dtype=torch.bool),
            "individual_quality": quality_prediction,
            "quality_targets": quality_target,
            "quality_pair_indices": torch.stack(
                (torch.arange(count), torch.zeros(count, dtype=torch.long)),
                dim=1,
            ),
        }

    first_prediction = torch.tensor([0.8, 0.2], requires_grad=True)
    first_target = torch.tensor([0.9, 0.1])
    second_prediction = torch.tensor([0.6, 0.4, 0.2], requires_grad=True)
    second_target = torch.tensor([0.9, 0.5, 0.1])
    payload = {
        "enabled": True,
        "mode": "stage2_adapt",
        "quality_enabled": True,
        "v5_quality_safe": v5_config,
        "loss_config": {
            "object_loss_weight": 1.0,
            "individual_loss_weight": 1.0,
            "consensus_loss_weight": 0.0,
            "quality_loss_weight": 1.0,
        },
        "stats": {
            "valid_object_ratio": 1.0,
            "mean_roi_coverage": 1.0,
            "object_roi_count": 5,
            "valid_agent_object_pairs": 5,
        },
        "scenes": (
            scene(first_prediction, first_target),
            scene(second_prediction, second_target),
        ),
    }
    _, stats = compute_dual_space_object_loss(payload)
    first = torch.nn.functional.softplus(torch.tensor(-0.6))
    second = torch.nn.functional.softplus(
        -torch.tensor([0.2, 0.4, 0.2])
    ).mean()
    expected = (first + 3.0 * second) / 4.0
    assert stats["dual_space_v5_ranking_pair_count"] == 4
    assert abs(stats["dual_space_v5_ranking_loss"] - expected.item()) < 1e-6
    assert abs(
        stats["dual_space_v5_weighted_quality_loss"]
        - stats["dual_space_v5_raw_quality_loss"]
    ) < 1e-7


@test("V6 norm cap bounds residual ratio and handles zero input")
def test_v6_norm_cap():
    config = resolve_v6_residual_safe_config(
        {"v6_residual_safe": {"enabled": True, "object": {"enabled": True}}}
    )["object"]
    inputs = torch.ones(2, 4, 1, 1)
    residual = inputs * 0.5
    safe, stats = apply_residual_norm_cap(inputs, residual, config, feature_dim=1)
    assert torch.allclose(stats["raw_residual_ratio"], torch.full((2, 1, 1, 1), 0.5))
    assert torch.allclose(stats["safe_residual_ratio"], torch.full((2, 1, 1, 1), 0.25), atol=2e-6)
    small, _ = apply_residual_norm_cap(inputs, inputs * 0.1, config, 1)
    assert torch.allclose(small, inputs * 0.1)
    zero_safe, zero_stats = apply_residual_norm_cap(
        torch.zeros_like(inputs), torch.ones_like(inputs), config, 1
    )
    assert torch.isfinite(zero_safe).all()
    assert torch.isfinite(zero_stats["safe_residual_ratio"]).all()


@test("V6 object and context routes are independently gated")
def test_v6_route_gating():
    v6 = {
        "enabled": True,
        "object": {"enabled": True, "max_residual_ratio": 0.25},
        "context": {"enabled": False},
    }
    host = _host(37, v6_residual_safe=v6)
    with torch.no_grad():
        host.dual_space_object_adapter_m2.delta[-1].bias.fill_(1.0)
        host.dual_space_context_adapter_m2.delta[-1].bias.fill_(1.0)
    inputs = torch.ones(2, 4, 3, 3)
    object_output = route_modality_adapters(host, inputs, ["m2", "m2"])
    context_inputs = torch.ones(2, 6, 3, 3)
    context_output = route_modality_adapters(
        host,
        context_inputs,
        ["m2", "m2"],
        adapter_namespace="context_adapter",
    )
    raw_context = host.dual_space_context_adapter_m2(context_inputs)
    assert not torch.equal(
        object_output, host.dual_space_object_adapter_m2(inputs)
    )
    assert torch.equal(context_output, raw_context)


@test("inference bypass child switches require the diagnostic parent")
def test_inference_bypass_parent_gate():
    child = {
        "enabled": False,
        "inference_ablation": {
            "enabled": True,
            "bypass_object_adapter": True,
        },
    }
    off = TinyDualSpaceHost(
        modalities=("m2",), mode="inference", quality=True, diagnostics=child
    )
    on_config = copy.deepcopy(child)
    on_config["enabled"] = True
    on = TinyDualSpaceHost(
        modalities=("m2",), mode="inference", quality=True,
        diagnostics=on_config,
    )
    with torch.no_grad():
        off.dual_space_object_adapter_m2.delta[-1].bias.fill_(1.0)
        on.dual_space_object_adapter_m2.load_state_dict(
            off.dual_space_object_adapter_m2.state_dict()
        )
    inputs = torch.ones(1, 4, 3, 3)
    off_output = route_modality_adapters(off, inputs, ["m2"])
    on_output = route_modality_adapters(on, inputs, ["m2"])
    assert not torch.equal(off_output, inputs)
    assert torch.equal(on_output, inputs)


@test("context and quality inference bypass reuse identity and uniform paths")
def test_context_and_quality_bypass():
    diagnostics = {
        "enabled": True,
        "inference_ablation": {
            "enabled": True,
            "bypass_context_adapter": True,
            "bypass_quality_weighting": True,
        },
    }
    host = TinyDualSpaceHost(
        modalities=("m2",),
        mode="inference",
        multi=True,
        quality=True,
        diagnostics=diagnostics,
    )
    with torch.no_grad():
        host.dual_space_context_adapter_m2.delta[-1].bias.fill_(1.0)
    context_inputs = torch.ones(1, 6, 3, 3)
    bypassed = route_modality_adapters(
        host,
        context_inputs,
        ["m2"],
        adapter_namespace="context_adapter",
    )
    assert torch.equal(bypassed, context_inputs)
    scene = make_scene(host, agent_count=2)
    result = predict_scene_residuals(host, scene, make_boxes(2))
    expected = result["valid_mask"].to(dtype=result["consensus_weights"].dtype)
    expected = expected / expected.sum(dim=1, keepdim=True).clamp_min(1.0)
    assert result["quality_weighting_bypassed"] is True
    assert torch.equal(result["consensus_weights"], expected)


@test("diagnostic autograd.grad leaves parameter grads untouched")
def test_gradient_diagnostic_is_noninvasive():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.dual_space_object_adapter_m2 = nn.Linear(2, 1, bias=False)

        def forward(self, value):
            return self.dual_space_object_adapter_m2(value)

    first = Toy()
    second = copy.deepcopy(first)
    value = torch.tensor([[1.0, -2.0]])
    first_output = first(value)
    detection = first_output.square().mean()
    quality = (first_output - 1.0).square().mean()
    loss_gradient_norms(detection, first)
    loss_gradient_norms(quality, first)
    assert all(parameter.grad is None for parameter in first.parameters())
    (detection + quality).backward()
    second_output = second(value)
    (second_output.square().mean() + (second_output - 1.0).square().mean()).backward()
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters()
    ):
        assert torch.allclose(first_parameter.grad, second_parameter.grad)


def main():
    return run_registered_tests(TESTS)


if __name__ == "__main__":
    sys.exit(main())

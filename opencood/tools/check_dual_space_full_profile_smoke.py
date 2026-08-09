"""CPU combination smoke for Full Dual-Space config, proposals, and state."""

import copy
import io
import os
import sys
from collections import OrderedDict

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.loss.dual_space_object_loss import compute_dual_space_object_loss
from opencood.models.sub_modules.dual_space_config import (
    dual_space_feature_flags,
    validate_dual_space_config,
)
from opencood.models.sub_modules.dual_space_object import (
    predict_scene_residuals,
    run_dual_space_training,
    validate_dual_space_checkpoint_keys,
)
from opencood.models.sub_modules.dual_space_proposal_sampler import (
    DualSpaceTrainingProposalSampler,
)
from opencood.tools.dual_space_smoke_common import (
    TinyDualSpaceHost,
    make_boxes,
    make_dual_config,
    make_scene,
    run_registered_tests,
)
from opencood.tools.heal_tools import apply_dual_space_merge_ownership


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def mixed_sampler(**predicted_updates):
    config = make_dual_config(mixed=True)["training_proposals"]
    config["predicted"].update(predicted_updates)
    return DualSpaceTrainingProposalSampler(config, max_proposals=32)


def sample_mixed(sampler, predicted_boxes, predicted_scores, seed=7):
    torch.manual_seed(seed)
    gt = make_boxes(1)
    return sampler(
        gt,
        torch.ones(1, dtype=torch.bool),
        with_jitter=True,
        predicted_boxes=predicted_boxes,
        predicted_scores=predicted_scores,
    )


@test("gt_jitter source never requires prediction decode input")
def test_gt_jitter_no_decode():
    config = make_dual_config()["training_proposals"]
    sampler = DualSpaceTrainingProposalSampler(config, 32)
    proposals, _ = sampler(make_boxes(1), torch.ones(1, dtype=torch.bool))
    assert proposals.shape[0] == 2
    assert sampler.last_stats["predicted_input_count"] == 0


@test("mixed source always retains the exact GT")
def test_mixed_gt():
    proposals, _ = sample_mixed(
        mixed_sampler(), make_boxes(0), torch.empty(0)
    )
    assert torch.equal(proposals[0], make_boxes(1)[0])


@test("mixed source retains configured GT jitter")
def test_mixed_jitter():
    proposals, _ = sample_mixed(
        mixed_sampler(), make_boxes(0), torch.empty(0)
    )
    assert proposals.shape[0] == 2
    assert not torch.equal(proposals[1], proposals[0])


@test("matched positive predicted proposal is included")
def test_mixed_positive():
    predicted = make_boxes(1, x=0.2)
    proposals, targets = sample_mixed(
        mixed_sampler(), predicted, torch.tensor([0.9])
    )
    assert proposals.shape[0] == 3
    assert torch.equal(proposals[-1], predicted[0])
    assert torch.equal(targets[-1], make_boxes(1)[0])


@test("unmatched predicted negative is filtered")
def test_mixed_negative_filter():
    sampler = mixed_sampler()
    proposals, _ = sample_mixed(
        sampler, make_boxes(1, x=12.0), torch.tensor([0.99])
    )
    assert proposals.shape[0] == 2
    assert sampler.last_stats["predicted_positive_count"] == 0


@test("predicted proposals and matched targets are detached")
def test_mixed_detach():
    predicted = make_boxes(1, x=0.1).requires_grad_(True)
    proposals, targets = sample_mixed(
        mixed_sampler(), predicted, torch.tensor([0.9], requires_grad=True)
    )
    assert proposals.requires_grad is False
    assert targets.requires_grad is False


@test("predicted max_per_scene is enforced before matching")
def test_mixed_max_count():
    predicted = make_boxes(5)
    predicted[:, 0] = torch.linspace(-0.2, 0.2, 5)
    sampler = mixed_sampler(max_per_scene=2)
    sample_mixed(sampler, predicted, torch.linspace(0.6, 1.0, 5))
    assert sampler.last_stats["predicted_positive_count"] == 2


@test("predicted min_score is enforced")
def test_mixed_min_score():
    predicted = make_boxes(2)
    predicted[:, 0] = torch.tensor([0.1, -0.1])
    sampler = mixed_sampler(min_score=0.8)
    sample_mixed(sampler, predicted, torch.tensor([0.79, 0.81]))
    assert sampler.last_stats["predicted_positive_count"] == 1


@test("predicted positive_iou_min is enforced")
def test_mixed_iou_threshold():
    predicted = make_boxes(1, x=1.5)
    loose = mixed_sampler(positive_iou_min=0.1)
    strict = mixed_sampler(positive_iou_min=0.8)
    sample_mixed(loose, predicted, torch.tensor([0.9]))
    sample_mixed(strict, predicted, torch.tensor([0.9]))
    assert loose.last_stats["predicted_positive_count"] == 1
    assert strict.last_stats["predicted_positive_count"] == 0


@test("mixed sampling is reproducible under torch seed")
def test_mixed_seed():
    predicted = make_boxes(1, x=0.1)
    sampler = mixed_sampler()
    first = sample_mixed(sampler, predicted, torch.tensor([0.9]), seed=19)[0]
    second = sample_mixed(sampler, predicted, torch.tensor([0.9]), seed=19)[0]
    assert torch.equal(first, second)


@test("plain HEAL and explicit disabled Dual-Space are legal")
def test_config_heal():
    assert validate_dual_space_config(None) is None
    assert validate_dual_space_config({"enabled": False})["enabled"] is False


@test("V1 V2 V3 V4 and Quality-no-MS configs are legal")
def test_config_profiles():
    profiles = (
        make_dual_config(),
        make_dual_config(multi=True),
        make_dual_config(multi=True, quality=True),
        make_dual_config(multi=True, quality=True, rescue=True),
        make_dual_config(multi=False, quality=True),
    )
    for profile in profiles:
        assert validate_dual_space_config(profile) is profile


def expect_invalid(config, fragment):
    try:
        validate_dual_space_config(config)
    except (TypeError, ValueError) as error:
        assert fragment in str(error)
    else:
        raise AssertionError("invalid configuration was silently accepted")


@test("dual=false rejects enabled multi-scale")
def test_config_disabled_multi():
    expect_invalid(
        {"enabled": False, "multi_scale": {"enabled": True}},
        "invalid dependency",
    )


@test("dual=false rejects enabled quality")
def test_config_disabled_quality():
    expect_invalid(
        {"enabled": False, "quality": {"enabled": True}},
        "invalid dependency",
    )


@test("dual=false rejects enabled RPR")
def test_config_disabled_rpr():
    expect_invalid(
        {"enabled": False, "remote_proposal_rescue": {"enabled": True}},
        "invalid dependency",
    )


@test("adaptive gate rejects multi-scale disabled")
def test_config_adaptive_dependency():
    config = make_dual_config()
    config["multi_scale"]["fusion"] = "adaptive_gate"
    expect_invalid(config, "requires dual_space.multi_scale.enabled=true")


@test("quality consensus rejects quality disabled")
def test_config_quality_dependency():
    config = make_dual_config()
    config["consensus"]["mode"] = "quality_weighted"
    expect_invalid(config, "requires dual_space.quality.enabled=true")


@test("unknown mode fails fast")
def test_config_unknown_mode():
    config = make_dual_config()
    config["mode"] = "mystery"
    expect_invalid(config, "invalid dual_space.mode")


@test("unknown multi-scale fusion fails fast")
def test_config_unknown_fusion():
    config = make_dual_config(multi=True)
    config["multi_scale"]["fusion"] = "cross_attention"
    expect_invalid(config, "invalid dual_space.multi_scale.fusion")


@test("unknown proposal source fails fast")
def test_config_unknown_proposals():
    config = make_dual_config()
    config["training_proposals"]["source"] = "detector_only"
    expect_invalid(config, "invalid dual_space.training_proposals.source")


@test("profile/version labels do not activate runtime features")
def test_profile_label_inert():
    config = make_dual_config()
    config["version"] = "paper_label_only"
    config["experiment_profile"] = "ds_v4"
    validate_dual_space_config(config)
    flags = dual_space_feature_flags(config)
    assert flags["multi_scale"] is False
    assert flags["quality"] is False
    assert flags["remote_proposal_rescue"] is False


@test("adaptive-gate profile constructs and forwards")
def test_adaptive_profile_forward():
    host = TinyDualSpaceHost(multi=True, fusion="adaptive_gate")
    result = predict_scene_residuals(host, make_scene(host, 2), make_boxes(2))
    assert result["scale_gates"].shape == (4, 1)


@test("feature-aware merge applies every explicit owner")
def test_full_merge_ownership():
    stage1 = OrderedDict(
        (
            ("dual_space_shared_object_encoder.weight", torch.tensor([1.0])),
            ("dual_space_shared_geometry_encoder.weight", torch.tensor([2.0])),
            ("dual_space_shared_object_refiner.weight", torch.tensor([3.0])),
            ("dual_space_shared_context_encoder.weight", torch.tensor([4.0])),
            ("dual_space_shared_multiscale_fusion.weight", torch.tensor([5.0])),
            ("dual_space_shared_quality_head.weight", torch.tensor([6.0])),
        )
    )
    stage2 = []
    for index, modality in enumerate(("m2", "m3", "m4"), start=2):
        stage2.append(
            OrderedDict(
                (
                    (
                        "dual_space_object_adapter_%s.weight" % modality,
                        torch.tensor([float(index * 10 + 1)]),
                    ),
                    (
                        "dual_space_context_adapter_%s.weight" % modality,
                        torch.tensor([float(index * 10 + 2)]),
                    ),
                )
            )
        )
    merged = apply_dual_space_merge_ownership(
        OrderedDict(), stage2 + [stage1]
    )
    assert merged["dual_space_shared_quality_head.weight"].item() == 6.0
    assert merged["dual_space_context_adapter_m2.weight"].item() == 22.0
    assert merged["dual_space_object_adapter_m4.weight"].item() == 41.0


@test("merge rejects a missing current-profile context adapter")
def test_merge_missing_context_adapter():
    stage1 = OrderedDict(
        (
            ("dual_space_shared_object_encoder.weight", torch.tensor([1.0])),
            ("dual_space_shared_context_encoder.weight", torch.tensor([2.0])),
            ("dual_space_shared_multiscale_fusion.weight", torch.tensor([3.0])),
        )
    )
    stages = []
    for modality in ("m2", "m3", "m4"):
        stage = OrderedDict(
            (("dual_space_object_adapter_%s.weight" % modality, torch.ones(1)),)
        )
        if modality != "m3":
            stage["dual_space_context_adapter_%s.weight" % modality] = torch.ones(1)
        stages.append(stage)
    try:
        apply_dual_space_merge_ownership(OrderedDict(), stages + [stage1])
    except RuntimeError as error:
        assert "stage2/m3" in str(error)
        assert "context_adapter_m3" in str(error)
    else:
        raise AssertionError("merge accepted a missing V2 context adapter")


@test("real Stage2 shared profile must match Stage1 feature keys")
def test_merge_profile_mismatch():
    stage1 = OrderedDict(
        (
            ("dual_space_shared_object_encoder.weight", torch.tensor([1.0])),
            ("dual_space_shared_quality_head.weight", torch.tensor([2.0])),
        )
    )
    stages = []
    for modality in ("m2", "m3", "m4"):
        stages.append(
            OrderedDict(
                (
                    ("dual_space_shared_object_encoder.weight", torch.tensor([9.0])),
                    ("dual_space_object_adapter_%s.weight" % modality, torch.ones(1)),
                )
            )
        )
    try:
        apply_dual_space_merge_ownership(OrderedDict(), stages + [stage1])
    except RuntimeError as error:
        assert "profile does not match" in str(error)
        assert "quality" in str(error)
    else:
        raise AssertionError("merge accepted mismatched V3 Stage2 profile")


@test("Stage1 accepts complete lower-profile checkpoint")
def test_checkpoint_lower_profile():
    v1 = TinyDualSpaceHost(mode="stage1_anchor", multi=False)
    v2 = TinyDualSpaceHost(mode="stage1_anchor", multi=True)
    validate_dual_space_checkpoint_keys(v2, v1.state_dict().keys())


@test("Stage2 rejects missing current-profile shared modules")
def test_checkpoint_stage2_strict():
    v2 = TinyDualSpaceHost(mode="stage1_anchor", multi=True)
    v3_stage2 = TinyDualSpaceHost(
        modalities=("m2",),
        mode="stage2_adapt",
        active_modality="m2",
        multi=True,
        quality=True,
    )
    try:
        validate_dual_space_checkpoint_keys(v3_stage2, v2.state_dict().keys())
    except RuntimeError as error:
        assert "quality" in str(error)
    else:
        raise AssertionError("V3 Stage2 accepted V2 shared state")


@test("V4 inference accepts matching V3 state without RPR keys")
def test_checkpoint_v4_reuses_v3():
    v3 = TinyDualSpaceHost(
        mode="inference", multi=True, quality=True, rescue=False
    )
    v4 = TinyDualSpaceHost(multi=True, quality=True, rescue=True)
    validate_dual_space_checkpoint_keys(v4, v3.state_dict().keys())


@test("V2 synthetic train forward loss and backward are finite")
def test_v2_e2e():
    host = TinyDualSpaceHost(multi=True)
    payload = run_dual_space_training(
        host,
        {"scenes": (make_scene(host, 2),)},
        {
            "object_bbx_center": make_boxes(2).unsqueeze(0),
            "object_bbx_mask": torch.tensor([[True, True]]),
        },
    )
    loss, _ = compute_dual_space_object_loss(payload)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        for parameter in host.dual_space_shared_context_encoder.parameters()
    )


@test("V3 synthetic train forward loss and backward are finite")
def test_v3_e2e():
    host = TinyDualSpaceHost(multi=True, quality=True)
    payload = run_dual_space_training(
        host,
        {"scenes": (make_scene(host, 2),)},
        {
            "object_bbx_center": make_boxes(1).unsqueeze(0),
            "object_bbx_mask": torch.ones(1, 1, dtype=torch.bool),
        },
    )
    loss, stats = compute_dual_space_object_loss(payload)
    loss.backward()
    assert torch.isfinite(loss)
    assert "dual_space_quality_loss" in stats


@test("V3 state round-trips through torch checkpoint serialization")
def test_state_roundtrip():
    source = TinyDualSpaceHost(
        mode="inference", multi=True, quality=True
    )
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    target = TinyDualSpaceHost(
        mode="inference", multi=True, quality=True
    )
    state = torch.load(buffer, map_location="cpu")
    validate_dual_space_checkpoint_keys(target, state.keys())
    target.load_state_dict(state, strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


@test("illegal partial optional checkpoint group is rejected")
def test_partial_feature_checkpoint():
    host = TinyDualSpaceHost(mode="stage1_anchor", multi=True)
    v1 = TinyDualSpaceHost(mode="stage1_anchor", multi=False)
    supplied = set(v1.state_dict().keys())
    supplied.add(next(
        key for key in host.state_dict()
        if key.startswith("dual_space_shared_context_encoder.")
    ))
    try:
        validate_dual_space_checkpoint_keys(host, supplied)
    except RuntimeError as error:
        assert "missing trained dual-space keys" in str(error)
    else:
        raise AssertionError("partial optional group was accepted")


if __name__ == "__main__":
    sys.exit(run_registered_tests(TESTS))

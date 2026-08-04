"""CPU smoke checks for Open-DCSI Phase 3 innovation token paths."""

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from opencood.tools.audit_open_dcsi_baseline_parity import (
    _collab_input,
    _single_input,
    _tiny_model_args,
)
from opencood.models.heter_pyramid_collab_open_dcsi_stage1 import (
    HeterPyramidCollabOpenDcsiStage1,
)
from opencood.models.heter_pyramid_single_open_dcsi_stage2 import (
    HeterPyramidSingleOpenDcsiStage2,
)
from opencood.models.sub_modules.open_dcsi.innovation_tokenizer import (
    TOKEN_SCHEMA,
    transform_tokens_to_ego,
)


def _phase3_config(foreground_threshold=0.0):
    return {
        "enabled": True,
        "open_heterogeneous": {"enabled": True},
        "stage2_independent": {"enabled": True},
        "common_space": {
            "enabled": True,
            "projector": {"enabled": True},
            "decoder": {"enabled": True},
            "reconstruction": {"enabled": True},
            "common_detection_supervision": {"enabled": True},
        },
        "common_fusion": {"enabled": True},
        "innovation_tokens": {
            "enabled": True,
            "max_tokens_per_agent": 6,
            "proposal_topk": 12,
            "foreground_threshold": foreground_threshold,
            "token_dim": 12,
            "semantic_dim": 6,
            "geometry_dim": 5,
        },
        "innovation_quality": {
            "enabled": True,
            "calibration": {"enabled": True},
        },
        "innovation_aggregation": {
            "enabled": True,
            "geometric_clustering": {
                "enabled": True,
                "center_radius": 2.0,
                "yaw_threshold": 3.1415926,
            },
            "absolute_reject": {"enabled": False},
        },
        "diagnostics": {"enabled": True},
    }


def _enabled_args(foreground_threshold=0.0):
    args = _tiny_model_args("missing")
    args["open_dcsi"] = _phase3_config(foreground_threshold)
    return args


def _permute_tokens(tokens, permutation):
    token_count = int(tokens["scenario_index"].numel())
    result = {}
    for key, value in tokens.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == token_count:
            result[key] = value.index_select(0, permutation)
        else:
            result[key] = value
    return result


def _clone_tokens(tokens):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in tokens.items()
    }


def _assert_finite_token_dict(tokens):
    for key, value in tokens.items():
        if torch.is_tensor(value) and value.is_floating_point():
            assert torch.isfinite(value).all(), "non-finite token field {}".format(key)


def _check_schema_forward_and_backward():
    torch.manual_seed(11)
    model = HeterPyramidCollabOpenDcsiStage1(_enabled_args()).train()
    output = model(_collab_input())
    tokens = output["open_dcsi"]["innovation_tokens"]
    fused = output["open_dcsi"]["fused_tokens"]
    assert tokens["schema"] == TOKEN_SCHEMA
    assert tokens["coordinate_frame"] == "ego_metric"
    assert tokens["box_order"] == "x_y_z_h_w_l_yaw"
    assert 0 < int(tokens["scenario_index"].numel()) <= 12
    for agent_index in (0, 1):
        count = int((tokens["agent_global_index"] == agent_index).sum())
        assert count <= 6
    _assert_finite_token_dict(tokens)
    _assert_finite_token_dict(fused)

    loss = output["cls_preds"].square().mean()
    loss = loss + tokens["innovation_embedding"].square().mean()
    loss = loss + tokens["boxes_ego_hwl"].square().mean()
    loss = loss + output["open_dcsi"]["quality"]["token_reliability"].mean()
    loss.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("open_dcsi.innovation_") and parameter.requires_grad
    }
    finite_gradients = [gradient for gradient in gradients.values() if gradient is not None]
    assert finite_gradients
    assert all(torch.isfinite(gradient).all() for gradient in finite_gradients)
    assert any("tokenizer" in name for name, gradient in gradients.items() if gradient is not None)
    assert any("quality_router" in name for name, gradient in gradients.items() if gradient is not None)
    print("[phase3] schema, bounded extraction, and backward OK")
    return model.eval(), tokens


def _check_aggregation_invariants(model, tokens):
    aggregator = model.open_dcsi.innovation_aggregator.eval()
    router = model.open_dcsi.innovation_quality_router.eval()
    count = int(tokens["scenario_index"].numel())
    permutation = torch.arange(count - 1, -1, -1, device=tokens["scenario_index"].device)
    original = aggregator(tokens, router)
    permuted = aggregator(_permute_tokens(tokens, permutation), router)
    for key in original:
        if torch.is_tensor(original[key]):
            assert torch.equal(original[key], permuted[key]), key

    single_index = torch.tensor([0], device=tokens["scenario_index"].device)
    single = _permute_tokens(tokens, single_index)
    single_fused = aggregator(single, router)
    assert torch.equal(
        single_fused["innovation_embedding"], single["innovation_embedding"]
    )
    assert torch.equal(single_fused["centers_ego"], single["centers_ego"])

    pair_index = torch.arange(min(2, count), device=tokens["scenario_index"].device)
    pair = _permute_tokens(tokens, pair_index)
    if pair_index.numel() == 2:
        pair["scenario_index"].zero_()
        pair["centers_ego"][:] = pair["centers_ego"][0]
        pair["boxes_ego_hwl"][:, :3] = pair["centers_ego"]
        pair["boxes_ego_hwl"][:, 6].zero_()
        pair["evidence_confidence"].fill_(1.0)
        pair["general_uncertainty"].zero_()
        pair["localization_uncertainty"].zero_()
        pair["validity"].fill_(1.0)
        pair["box_quality"].fill_(1.0)
        pair["innovation_embedding"][0].fill_(1.0)
        pair["innovation_embedding"][1].fill_(3.0)
        equal_fused = aggregator(pair, router)
        assert torch.allclose(
            equal_fused["innovation_embedding"],
            torch.full_like(equal_fused["innovation_embedding"], 2.0),
            atol=0.0,
            rtol=0.0,
        )
        pair["evidence_confidence"][0] = 1e-6
        pair["evidence_confidence"][1] = 1.0
        dominant = aggregator(pair, router)["innovation_embedding"]
        assert torch.all((dominant - 3.0).abs() < 1e-3)
    print("[phase3] permutation, identity, mean, and dominant quality OK")


def _check_invalid_and_empty(model, tokens):
    aggregator = model.open_dcsi.innovation_aggregator.eval()
    router = model.open_dcsi.innovation_quality_router.eval()
    invalid = _clone_tokens(tokens)
    invalid["innovation_embedding"][0, 0] = float("nan")
    filtered = aggregator(invalid, router)
    _assert_finite_token_dict(filtered)

    all_invalid = _clone_tokens(tokens)
    all_invalid["validity"] = torch.zeros_like(all_invalid["validity"])
    empty = aggregator(all_invalid, router)
    assert int(empty["scenario_index"].numel()) == 0

    empty_model = HeterPyramidCollabOpenDcsiStage1(_enabled_args(1.1)).eval()
    with torch.no_grad():
        output = empty_model(_collab_input())
    assert int(output["open_dcsi"]["innovation_tokens"]["scenario_index"].numel()) == 0
    assert "cls_preds" in output
    print("[phase3] NaN filtering and empty-token official fallback OK")


def _check_transform_and_stage2():
    model = HeterPyramidSingleOpenDcsiStage2(_enabled_args()).eval()
    with torch.no_grad():
        output = model(_single_input())
    tokens = output["open_dcsi"]["innovation_tokens"]
    assert torch.equal(tokens["centers_local"], tokens["centers_ego"])

    if int(tokens["scenario_index"].numel()) > 0:
        one = _permute_tokens(tokens, torch.tensor([0]))
        one["scenario_index"].zero_()
        one["agent_local_index"].zero_()
        pairwise = torch.eye(4).reshape(1, 1, 1, 4, 4)
        pairwise[0, 0, 0, 0, 3] = 2.0
        transformed = transform_tokens_to_ego(one, pairwise, torch.tensor([1]))
        assert torch.allclose(
            transformed["centers_ego"][:, 0], one["centers_local"][:, 0] + 2.0
        )
    print("[phase3] metric ego transform and Stage2 local tokens OK")


def main():
    model, tokens = _check_schema_forward_and_backward()
    _check_aggregation_invariants(model, tokens)
    _check_invalid_and_empty(model, tokens)
    _check_transform_and_stage2()
    print("OPEN_DCSI_PHASE3_PASS")


if __name__ == "__main__":
    main()

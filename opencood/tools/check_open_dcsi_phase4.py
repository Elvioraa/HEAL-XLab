"""CPU smoke checks for Open-DCSI Phase 4 geometry and refinement."""

from pathlib import Path
import math
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from opencood.tools.audit_open_dcsi_baseline_parity import _collab_input, _tiny_model_args
from opencood.tools.check_open_dcsi_phase3 import _phase3_config, _clone_tokens
from opencood.models.heter_pyramid_collab_open_dcsi_stage1 import (
    HeterPyramidCollabOpenDcsiStage1,
)
from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.sub_modules.open_dcsi.geometry_refiner import GeometryRefiner


def _enabled_args():
    args = _tiny_model_args("missing")
    config = _phase3_config()
    config["cross_scale_geometry"] = {
        "enabled": True,
        "sampling_points": 4,
        "offset_limit": {"x": 2.0, "y": 1.5},
        "localization_quality_controls_offset": True,
    }
    config["geometry_refiner"] = {
        "enabled": True,
        "zero_init_output": True,
        "max_center_delta": 2.0,
        "max_yaw_delta": 0.785,
    }
    args["open_dcsi"] = config
    return args


def _check_full_geometry_forward():
    torch.manual_seed(17)
    model = HeterPyramidCollabOpenDcsiStage1(_enabled_args()).train()
    output = model(_collab_input())
    open_output = output["open_dcsi"]
    geometry = open_output["cross_scale_geometry"]
    refinement = open_output["geometry_refinement"]
    fused = open_output["fused_tokens"]
    token_count = int(fused["scenario_index"].numel())
    assert tuple(geometry["context"].shape) == (token_count, 5)
    assert len(geometry["offsets"]) == 2
    assert geometry["offsets"][0][..., 0].abs().max() <= 2.0
    assert geometry["offsets"][0][..., 1].abs().max() <= 1.5
    assert torch.equal(
        refinement["box_deltas_hwl"], torch.zeros_like(refinement["box_deltas_hwl"])
    )
    assert torch.equal(
        refinement["refined_token_boxes_preview_hwl"], fused["boxes_ego_hwl"]
    )
    for value in geometry.values():
        values = value if isinstance(value, list) else [value]
        for tensor in values:
            assert torch.isfinite(tensor).all()

    loss = output["cls_preds"].square().mean()
    loss = loss + geometry["context"].square().mean()
    loss = loss + refinement["box_deltas_hwl"].sum()
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if (
            name.startswith("open_dcsi.cross_scale_geometry")
            or name.startswith("open_dcsi.geometry_refiner")
        )
        and parameter.requires_grad
        and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    print("[phase4] sparse geometry forward/backward and zero-init identity OK")
    return model.eval(), fused, open_output


def _check_localization_control_and_oob(model, fused, open_output):
    sampler = model.open_dcsi.cross_scale_geometry.eval()
    runtime = model.open_dcsi
    innovation = open_output["innovation_features"]
    affine = torch.zeros(1, 2, 2, 2, 3)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    record_len = torch.tensor([2])

    low_uncertainty = _clone_tokens(fused)
    low_uncertainty["localization_uncertainty"].zero_()
    high_uncertainty = _clone_tokens(fused)
    high_uncertainty["localization_uncertainty"].fill_(10.0)
    with torch.no_grad():
        low = sampler(low_uncertainty, innovation, record_len, affine)
        high = sampler(high_uncertainty, innovation, record_len, affine)
    low_norm = sum(offset.abs().sum() for offset in low["offsets"])
    high_norm = sum(offset.abs().sum() for offset in high["offsets"])
    assert high_norm <= low_norm

    outside = _clone_tokens(fused)
    outside["centers_ego"].fill_(1e6)
    with torch.no_grad():
        oob = sampler(outside, innovation, record_len, affine)
    assert torch.isfinite(oob["context"]).all()
    assert torch.equal(oob["validity"], torch.zeros_like(oob["validity"]))
    print("[phase4] localization control and out-of-bounds safety OK")


def _check_refiner_clamp_wrap_and_fallback(model, fused):
    refiner = model.open_dcsi.geometry_refiner.eval()
    with torch.no_grad():
        refiner.output.bias.fill_(100.0)
        refined = refiner(fused)
    delta = refined["box_deltas_hwl"]
    assert delta[:, :3].abs().max() <= 2.0
    assert delta[:, 6].abs().max() <= 0.785

    base = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, math.pi - 0.1]])
    manual_delta = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]])
    wrapped = GeometryRefiner.apply_deltas(base, manual_delta)
    assert -math.pi <= float(wrapped[0, 6]) <= math.pi
    assert float(wrapped[0, 6]) < 0

    invalid_delta = manual_delta.clone()
    invalid_delta[0, 0] = float("inf")
    fallback = GeometryRefiner.apply_deltas(base, invalid_delta)
    assert torch.equal(fallback, base)
    print("[phase4] residual clamp, yaw wrap, and non-finite fallback OK")


def _check_refiner_only_preserves_official_raw_predictions():
    parent = HeterPyramidCollab(_tiny_model_args("missing")).eval()
    args = _enabled_args()
    args["open_dcsi"]["common_space"]["common_detection_supervision"] = {
        "enabled": False
    }
    args["open_dcsi"]["common_fusion"] = {"enabled": False}
    args["open_dcsi"]["cross_scale_geometry"] = {"enabled": False}
    wrapper = HeterPyramidCollabOpenDcsiStage1(args).eval()
    load_result = wrapper.load_state_dict(parent.state_dict(), strict=False)
    assert not load_result.unexpected_keys
    assert load_result.missing_keys
    assert all(key.startswith("open_dcsi.") for key in load_result.missing_keys)
    with torch.no_grad():
        parent_output = parent(_collab_input())
        wrapper_output = wrapper(_collab_input())
    for key in ("cls_preds", "reg_preds", "dir_preds", "occ_single_list"):
        left = parent_output[key]
        right = wrapper_output[key]
        if isinstance(left, list):
            assert all(torch.equal(a, b) for a, b in zip(left, right))
        else:
            assert torch.equal(left, right)
    print("[phase4] refiner-only sidecar preserves official raw predictions")


def main():
    model, fused, open_output = _check_full_geometry_forward()
    _check_localization_control_and_oob(model, fused, open_output)
    _check_refiner_clamp_wrap_and_fallback(model, fused)
    _check_refiner_only_preserves_official_raw_predictions()
    print("OPEN_DCSI_PHASE4_PASS")


if __name__ == "__main__":
    main()

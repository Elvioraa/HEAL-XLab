"""Unified CPU smoke suite for the complete Open-DCSI implementation."""

from copy import deepcopy
from pathlib import Path
import sys

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Import parity first because it installs lightweight stubs for optional local
# dependencies before the real model modules are imported.
from opencood.tools import audit_open_dcsi_baseline_parity as parity
from opencood.tools import audit_open_dcsi_open_heterogeneous as open_heterogeneous
from opencood.tools import check_open_dcsi_phase2 as phase2
from opencood.tools import check_open_dcsi_phase3 as phase3
from opencood.tools import check_open_dcsi_phase4 as phase4
from opencood.tools import check_open_dcsi_phase6 as phase6
from opencood.tools import check_open_dcsi_phase7 as phase7
from opencood.tools import check_open_dcsi_phase8 as phase8
from opencood.models.heter_pyramid_collab_open_dcsi_stage1 import (
    HeterPyramidCollabOpenDcsiStage1,
)
from opencood.models.heter_pyramid_single_open_dcsi_stage2 import (
    HeterPyramidSingleOpenDcsiStage2,
)
from opencood.models.sub_modules.open_dcsi.config import (
    _FEATURE_PATHS,
    normalize_open_dcsi_config,
    validate_open_dcsi_config,
)
from opencood.models.sub_modules.open_dcsi.model_bridge import (
    PHASE9_IMPLEMENTED_MODULES,
)


REQUIRED_YAMLS = {
    "stage1/m1_open_dcsi_official_parity.yaml",
    "stage1/m1_open_dcsi_common_only.yaml",
    "stage1/m1_open_dcsi_common_innovation.yaml",
    "stage1/m1_open_dcsi_full.yaml",
    "stage2/m2_alignto_m1_open_dcsi_full.yaml",
    "stage2/m3_alignto_m1_open_dcsi_full.yaml",
    "stage2/m4_alignto_m1_open_dcsi_full.yaml",
    "inference/open_dcsi_cav1_full.yaml",
    "inference/open_dcsi_cav1_2_full.yaml",
    "inference/open_dcsi_cav1_3_full.yaml",
    "inference/open_dcsi_cav1_4_full.yaml",
    "inference/open_dcsi_cav1_4_dense.yaml",
    "inference/open_dcsi_cav1_4_streaming.yaml",
    "inference/open_dcsi_cav1_4_int8_budget.yaml",
    "ablation/no_common_low_rank.yaml",
    "ablation/no_innovation_tokens.yaml",
    "ablation/no_quality_gate.yaml",
    "ablation/no_cross_scale_geometry.yaml",
    "ablation/no_geometry_refiner.yaml",
    "ablation/refiner_only.yaml",
    "ablation/dense_innovation_map.yaml",
    "ablation/no_quantization.yaml",
    "ablation/random_token_selection.yaml",
    "ablation/confidence_token_selection.yaml",
    "ablation/localization_token_selection.yaml",
    "ablation/official_bgea_selection.yaml",
}


def _expect_invalid(config, expected_text):
    try:
        validate_open_dcsi_config(config, PHASE9_IMPLEMENTED_MODULES)
    except ValueError as error:
        assert expected_text in str(error), (expected_text, str(error))
        return
    raise AssertionError("invalid config was accepted: {}".format(expected_text))


def _check_defaults_dependencies_and_yamls():
    defaults = normalize_open_dcsi_config()
    assert defaults["enabled"] is False
    for path in _FEATURE_PATHS:
        value = defaults
        for part in path.split("."):
            value = value[part]
        assert value["enabled"] is False, path

    _expect_invalid(
        {"enabled": True, "geometry_refiner": {"enabled": True}},
        "requires innovation_tokens",
    )
    _expect_invalid(
        {"enabled": True, "streaming_fusion": {"enabled": True}},
        "requires common_space or innovation_tokens",
    )
    _expect_invalid(
        {
            "enabled": True,
            "stage2_independent": {"enabled": True},
        },
        "requires open_heterogeneous",
    )
    _expect_invalid(
        {
            "enabled": True,
            "communication": {"selection": {"enabled": True}},
        },
        "selection requires innovation_tokens",
    )

    root = (
        REPO_ROOT
        / "opencood"
        / "hypes_yaml"
        / "HEAL_XLab_v3_HVP_HEAL"
        / "open_dcsi"
    )
    found = {
        path.relative_to(root).as_posix() for path in root.rglob("*.yaml")
    }
    assert REQUIRED_YAMLS <= found, sorted(REQUIRED_YAMLS - found)
    for relative_path in sorted(REQUIRED_YAMLS):
        path = root / relative_path
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("# "), relative_path
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        validate_open_dcsi_config(config, PHASE9_IMPLEMENTED_MODULES)
    print("[smoke] defaults, dependency failures, and 26 YAML templates OK")


def _check_dense_innovation_comparator():
    args = phase2._enabled_args()
    args["open_dcsi"]["dense_innovation_map"] = {
        "enabled": True,
        "residual_scale": 1.0,
    }
    model = HeterPyramidCollabOpenDcsiStage1(args).train()
    output = model(parity._collab_input())
    fused = output["open_dcsi"]["fused_dense_innovation_features"]
    assert fused is not None and len(fused) == 2
    loss = output["cls_preds"].float().square().mean()
    for parameter in model.open_dcsi.parameters():
        if parameter.requires_grad:
            loss = loss + parameter.float().sum() * 0.0
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.open_dcsi.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    print("[smoke] dense innovation comparator forward/backward OK")


def _check_full_template_model_construction():
    root = (
        REPO_ROOT
        / "opencood"
        / "hypes_yaml"
        / "HEAL_XLab_v3_HVP_HEAL"
        / "open_dcsi"
    )
    with (root / "stage1/m1_open_dcsi_full.yaml").open(encoding="utf-8") as stream:
        stage1_config = yaml.safe_load(stream)["model"]["args"]["open_dcsi"]
    stage1_args = parity._tiny_model_args("missing")
    stage1_args["open_dcsi"] = stage1_config
    stage1_model = HeterPyramidCollabOpenDcsiStage1(stage1_args)
    assert hasattr(stage1_model.open_dcsi, "geometry_refiner")
    assert hasattr(stage1_model.open_dcsi, "communication")

    for modality in ("m2", "m3", "m4"):
        path = root / "stage2/{}_alignto_m1_open_dcsi_full.yaml".format(modality)
        with path.open(encoding="utf-8") as stream:
            stage2_config = yaml.safe_load(stream)["model"]["args"]["open_dcsi"]
        stage2_args = parity._tiny_model_args("missing")
        local = stage2_args.pop("m1")
        stage2_args[modality] = local
        stage2_args["open_dcsi"] = stage2_config
        model = HeterPyramidSingleOpenDcsiStage2(stage2_args)
        assert model._open_dcsi_stage2_modality == modality
        assert all(
            name.startswith(
                (
                    "encoder_{}".format(modality),
                    "backbone_{}".format(modality),
                    "aligner_{}".format(modality),
                    "open_dcsi.common_projectors.{}".format(modality),
                    "open_dcsi.innovation_tokenizer.tokenizers.{}".format(modality),
                )
            )
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    print("[smoke] full Stage1 and m2/m3/m4 Stage2 template construction OK")


def _check_amp_and_fp32_gradients():
    args = phase4._enabled_args()
    model = HeterPyramidCollabOpenDcsiStage1(deepcopy(args)).train()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(parity._collab_input())
        amp_loss = output["cls_preds"].float().square().mean()
    assert torch.isfinite(amp_loss)

    model.zero_grad(set_to_none=True)
    output = model(parity._collab_input())
    loss = output["cls_preds"].square().mean()
    for parameter in model.open_dcsi.parameters():
        if parameter.requires_grad:
            loss = loss + parameter.sum() * 0.0
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.open_dcsi.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    print("[smoke] CPU AMP and FP32 finite-gradient checks OK")


def main():
    parity.main()
    phase2.main()
    phase3.main()
    phase4.main()
    open_heterogeneous.main()
    phase6.main()
    phase7.main()
    phase8.main()
    _check_defaults_dependencies_and_yamls()
    _check_full_template_model_construction()
    _check_dense_innovation_comparator()
    _check_amp_and_fp32_gradients()
    print("OPEN_DCSI_SMOKE_PASS")


if __name__ == "__main__":
    main()

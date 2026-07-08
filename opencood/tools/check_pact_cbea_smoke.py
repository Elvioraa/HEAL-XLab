"""Smoke test for PACT-CBEA v1 rule-based evidence routing."""

import os
import sys
import types

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if "opencood.utils.box_overlaps" not in sys.modules:
    box_overlaps_stub = types.ModuleType("opencood.utils.box_overlaps")
    box_overlaps_stub.bbox_overlaps = lambda *args, **kwargs: None
    sys.modules["opencood.utils.box_overlaps"] = box_overlaps_stub

from opencood.hypes_yaml import yaml_utils
from opencood.models.heter_pyramid_collab_pact_cbea import (
    HeterPyramidCollabPactCbea,
)
from opencood.models.sub_modules.pact_cbea_rule import PACTCBEARule


YAML_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "HEAL_XLab_v3_HVP_HEAL",
    "pact",
    "cbea_rule.yaml",
)


class _DummyEncoder(nn.Module):
    def forward(self, data_dict, modality_name):
        return data_dict[f"inputs_{modality_name}"]["feature"]


class _DummyBackbone(nn.Module):
    def forward(self, batch_dict):
        return {"spatial_features_2d": batch_dict["spatial_features"]}


class _DummyAligner(nn.Module):
    def forward(self, feature):
        return feature


class _DummyPyramid(nn.Module):
    def forward_single(self, feature):
        return feature, [feature[:, :1]]


def main():
    torch.manual_seed(53)
    hypes = yaml_utils.load_yaml(YAML_PATH)
    pact_cfg = hypes["model"]["args"]["pact_cbea"]
    assert hypes["name"] == "PACT_CBEA_v1/rule_cbea"
    assert hypes["model"]["core_method"] == "heter_pyramid_collab_pact_cbea"
    assert pact_cfg["enabled"] is True
    assert pact_cfg["trainable"] is False
    assert pact_cfg["no_joint_training"] is True
    assert pact_cfg["use_stage3_joint_training"] is False

    rule = PACTCBEARule(pact_cfg)
    assert _trainable_count(rule) == 0
    _check_fallback_aggregation(rule)
    _check_evidence_routing(rule)
    _check_modality_prior(pact_cfg)

    model = _build_dummy_model(pact_cfg)
    assert model.pact_cbea_enabled
    assert model.pact_cbea_trainable is False
    assert model.pact_no_joint_training is True
    assert model.pact_use_stage3_joint_training is False
    assert _trainable_count(model.pact_cbea_rule) == 0

    data_dict = _dummy_data()
    output_dict = model(data_dict)
    assert "pact_cbea" in output_dict
    assert output_dict["cls_preds"].shape == (1, 2, 8, 8)
    assert output_dict["reg_preds"].shape == (1, 14, 8, 8)
    assert output_dict["dir_preds"].shape == (1, 4, 8, 8)
    _assert_debug_keys(output_dict["pact_cbea"])

    loss = output_dict["cls_preds"].mean() + output_dict["reg_preds"].abs().mean()
    loss.backward()
    input_grads = [
        payload["feature"].grad
        for key, payload in data_dict.items()
        if key.startswith("inputs_")
    ]
    assert any(
        grad is not None
        and torch.isfinite(grad).all()
        and grad.detach().abs().sum() > 0
        for grad in input_grads
    )
    assert _trainable_count(model.pact_cbea_rule) == 0

    print("PACT-CBEA yaml load OK")
    print("PACT-CBEA model create OK")
    print("PACT-CBEA HEAL-compatible config OK")
    print("PACT-CBEA no-joint-training config OK")
    print("PACT-CBEA rule parameter-free OK")
    print("PACT-CBEA fallback aggregation OK")
    print("PACT-CBEA evidence routing OK")
    print("PACT-CBEA modality prior OK")
    print("PACT-CBEA forward OK")
    print("PACT-CBEA smoke OK")


def _check_fallback_aggregation(rule):
    zeros = torch.zeros(1, 4, 3, 3)
    ones = torch.ones(1, 4, 3, 3)
    enhanced, debug = rule([zeros, ones])
    assert enhanced.shape == (1, 4, 3, 3)
    assert torch.allclose(enhanced, torch.full_like(enhanced, 0.5), atol=1e-5)
    assert "missing_evidence_heatmap_confidence_ones" in debug["pact_fallbacks"]
    assert "missing_evidence_uncertainty_weight_ones" in debug["pact_fallbacks"]
    _assert_debug_keys(debug)


def _check_evidence_routing(rule):
    features = torch.stack([
        torch.zeros(4, 3, 3),
        torch.full((4, 3, 3), 10.0),
    ], dim=0)
    heatmap = torch.stack([
        torch.full((1, 3, 3), -5.0),
        torch.full((1, 3, 3), 5.0),
    ], dim=0)
    uncertainty = torch.stack([
        torch.full((1, 3, 3), 5.0),
        torch.zeros(1, 3, 3),
    ], dim=0)
    enhanced, debug = rule(
        features,
        evidence_heatmap=heatmap,
        evidence_uncertainty=uncertainty,
        modality_names=["m1", "m2"],
    )
    assert enhanced.shape == (1, 4, 3, 3)
    assert enhanced.mean() > 9.0
    alpha = debug["pact_alpha"]
    assert alpha[:, 1].mean() > alpha[:, 0].mean()


def _check_modality_prior(base_cfg):
    cfg = PACTCBEARule._normalize_cfg(base_cfg)
    cfg["weights"]["modality_prior"] = {
        "m1": 1.0,
        "m2": 3.0,
    }
    rule = PACTCBEARule(cfg)
    features = torch.stack([
        torch.zeros(4, 3, 3),
        torch.full((4, 3, 3), 10.0),
    ], dim=0)
    enhanced, debug = rule(features, modality_names=["m1", "m2"])
    assert enhanced.mean() > 7.0
    alpha = debug["pact_alpha"]
    assert alpha[:, 1].mean() > alpha[:, 0].mean()


def _build_dummy_model(pact_cfg):
    channels = 32
    model = HeterPyramidCollabPactCbea.__new__(HeterPyramidCollabPactCbea)
    nn.Module.__init__(model)
    model.supervise_single = True
    model.modality_name_list = ["m1", "m2", "m3", "m4"]
    model.sensor_type_dict = {name: "lidar" for name in model.modality_name_list}
    model.cam_crop_info = {}
    model.H = 8
    model.W = 8
    model.fake_voxel_size = 1
    model.compress = False
    model.shrink_flag = False
    for modality_name in model.modality_name_list:
        setattr(model, f"encoder_{modality_name}", _DummyEncoder())
        setattr(model, f"backbone_{modality_name}", _DummyBackbone())
        setattr(model, f"aligner_{modality_name}", _DummyAligner())
        setattr(model, f"depth_supervision_{modality_name}", False)
    model.pyramid_backbone = _DummyPyramid()
    model.cls_head = nn.Conv2d(channels, 2, kernel_size=1)
    model.reg_head = nn.Conv2d(channels, 14, kernel_size=1)
    model.dir_head = nn.Conv2d(channels, 4, kernel_size=1)
    model.pact_cbea_cfg = HeterPyramidCollabPactCbea._normalize_pact_cfg(pact_cfg)
    model.pact_cbea_enabled = bool(model.pact_cbea_cfg["enabled"])
    model.pact_cbea_trainable = bool(model.pact_cbea_cfg["trainable"])
    model.pact_no_joint_training = bool(model.pact_cbea_cfg["no_joint_training"])
    model.pact_use_stage3_joint_training = bool(
        model.pact_cbea_cfg["use_stage3_joint_training"]
    )
    model.pact_cbea_rule = PACTCBEARule(model.pact_cbea_cfg)
    model._freeze_all_model_parameters()
    return model


def _dummy_data():
    modalities = ["m1", "m2", "m3", "m4"]
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4).repeat(1, 4, 4, 1, 1)
    data = {
        "agent_modality_list": modalities,
        "pairwise_t_matrix": pairwise,
        "record_len": torch.tensor([4]),
    }
    for idx, modality_name in enumerate(modalities):
        feature = torch.randn(1, 32, 8, 8, requires_grad=True) + float(idx) * 0.1
        feature.retain_grad()
        data[f"inputs_{modality_name}"] = {"feature": feature}
    return data


def _assert_debug_keys(debug):
    for key in (
        "pact_alpha",
        "pact_reliability",
        "pact_evidence_confidence",
        "pact_uncertainty_weight",
        "pact_modality_prior",
        "pact_mode",
    ):
        assert key in debug
    assert debug["pact_mode"] == "trust_calibrated_rule"


def _trainable_count(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


if __name__ == "__main__":
    main()

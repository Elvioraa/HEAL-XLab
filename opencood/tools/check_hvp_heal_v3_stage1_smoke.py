"""Smoke test for HVP-HEAL v3 Stage1 hypothesis skeleton."""

import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.loss.hvp_cbea_aux_loss import compute_hvp_v3_stage1_loss
from opencood.models.heter_pyramid_collab_hvp_heal_v3 import HeterPyramidCollabHvpHealV3
from opencood.models.hvp_heal_v3.hypothesis_head import HvpHealV3HypothesisHead


class _DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, data_dict, modality_name):
        return data_dict[f"inputs_{modality_name}"]["feature"] * self.scale


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, batch_dict):
        return {"spatial_features_2d": batch_dict["spatial_features"] * self.scale}


class _DummyAligner(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, feature):
        return feature * self.scale


class _DummyPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward_collab(self, heter_feature_2d, record_len, affine_matrix,
                       agent_modality_list, cam_crop_info):
        rows = []
        start = 0
        for length in record_len.detach().cpu().tolist():
            rows.append(heter_feature_2d[start:start + int(length)].mean(dim=0))
            start += int(length)
        fused = torch.stack(rows, dim=0) * self.scale
        return fused, [fused[:, :1]]


def main():
    torch.manual_seed(17)
    disabled_model = _build_dummy_model(enabled=False)
    enabled_model = _build_dummy_model(enabled=True)
    data_dict = _dummy_data()

    disabled_output = disabled_model(data_dict)
    assert "hvp_v3" not in disabled_output
    assert not any(key.startswith("hvp_v3_hypothesis_head") for key in disabled_model.state_dict())

    enabled_output = enabled_model(data_dict)
    assert "hvp_v3" in enabled_output
    assert_base_frozen_for_head_only(enabled_model)
    hvp_v3 = enabled_output["hvp_v3"]
    assert hvp_v3["stage"] == "stage1_hypothesis"
    assert hvp_v3["hypothesis_heatmap_logits"].shape == (1, 1, 16, 16)
    assert hvp_v3["hypothesis_heatmap"].shape == (1, 1, 16, 16)
    assert any(key.startswith("hvp_v3_hypothesis_head") for key in enabled_model.state_dict())

    target_dict = {
        "pos_equal_one": torch.zeros(1, 16, 16, 2),
    }
    target_dict["pos_equal_one"][0, 4, 5, 0] = 1.0
    target_dict["pos_equal_one"][0, 10, 11, 1] = 1.0
    stage1_loss, stats = compute_hvp_v3_stage1_loss(hvp_v3, target_dict)
    assert torch.is_tensor(stage1_loss)
    assert torch.isfinite(stage1_loss).all()
    assert stage1_loss.detach().item() > 0.0
    assert stats["hvp_v3_stage1_hypothesis_loss"] > 0.0

    stage1_loss.backward()
    grads = [
        param.grad
        for name, param in enabled_model.named_parameters()
        if name.startswith("hvp_v3_hypothesis_head")
    ]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)
    base_grads = [
        param.grad
        for name, param in enabled_model.named_parameters()
        if not name.startswith("hvp_v3_hypothesis_head")
    ]
    assert all(grad is None for grad in base_grads)

    print("HVP-HEAL v3 Stage1 disabled forward OK")
    print("HVP-HEAL v3 Stage1 enabled forward OK")
    print("HVP-HEAL v3 Stage1 head-only freeze OK")
    print("HVP-HEAL v3 Stage1 hypothesis loss OK")
    print("HVP-HEAL v3 Stage1 backward OK")


def _build_dummy_model(enabled):
    model = HeterPyramidCollabHvpHealV3.__new__(HeterPyramidCollabHvpHealV3)
    nn.Module.__init__(model)
    model.modality_name_list = ["m1"]
    model.sensor_type_dict = {"m1": "lidar"}
    model.cam_crop_info = {}
    model.H = 16
    model.W = 16
    model.fake_voxel_size = 1
    model.compress = False
    model.shrink_flag = False
    model.encoder_m1 = _DummyEncoder()
    model.backbone_m1 = _DummyBackbone()
    model.aligner_m1 = _DummyAligner()
    model.pyramid_backbone = _DummyPyramid()
    model.cls_head = nn.Conv2d(64, 2, kernel_size=1)
    model.reg_head = nn.Conv2d(64, 14, kernel_size=1)
    model.dir_head = nn.Conv2d(64, 4, kernel_size=1)
    cfg = {
        "enabled": bool(enabled),
        "stage": "stage1_hypothesis" if enabled else "none",
        "stage1": {
            "train_mode": "hypothesis_head_only",
            "freeze_base_model": True,
            "detach_bev_for_hypothesis": True,
        },
        "feature_main": {
            "enabled": bool(enabled),
        },
        "hypothesis_head": {
            "enabled": bool(enabled),
            "in_channels": 64,
            "hidden_dim": 16,
            "out_channels": 1,
            "use_sigmoid": True,
        },
        "aux_loss": {
            "enabled": bool(enabled),
            "mode": "stage1_hypothesis",
            "hypothesis_heatmap": {
                "enabled": bool(enabled),
                "weight": 0.01,
                "pos_weight": 1.0,
            },
            "residual_reg": {"enabled": False},
            "alpha_reg": {"enabled": False},
            "residual_focus": {"enabled": False},
        },
    }
    model.hvp_v3_cfg = HeterPyramidCollabHvpHealV3._normalize_hvp_v3_cfg(cfg)
    model.hvp_v3_enabled = bool(model.hvp_v3_cfg["enabled"])
    model.hvp_v3_stage = model.hvp_v3_cfg["stage"]
    if model._is_stage1_enabled():
        head_cfg = model.hvp_v3_cfg["hypothesis_head"]
        model.hvp_v3_hypothesis_head = HvpHealV3HypothesisHead(
            in_channels=head_cfg["in_channels"],
            hidden_dim=head_cfg["hidden_dim"],
            out_channels=head_cfg["out_channels"],
            use_sigmoid=head_cfg["use_sigmoid"],
        )
        model._apply_stage1_train_mode()
        model.train()
    return model


def assert_base_frozen_for_head_only(model):
    head_params = [
        param for name, param in model.named_parameters()
        if name.startswith("hvp_v3_hypothesis_head")
    ]
    base_params = [
        param for name, param in model.named_parameters()
        if not name.startswith("hvp_v3_hypothesis_head")
    ]
    assert head_params
    assert all(param.requires_grad for param in head_params)
    assert all(not param.requires_grad for param in base_params)
    assert model.hvp_v3_hypothesis_head.training
    assert not model.encoder_m1.training
    assert not model.backbone_m1.training
    assert not model.pyramid_backbone.training


def _dummy_data():
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4)
    return {
        "agent_modality_list": ["m1"],
        "pairwise_t_matrix": pairwise,
        "record_len": torch.tensor([1]),
        "inputs_m1": {
            "feature": torch.randn(1, 64, 16, 16),
        },
    }


if __name__ == "__main__":
    main()

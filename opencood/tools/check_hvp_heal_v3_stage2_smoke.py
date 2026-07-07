"""Smoke test for HVP-HEAL v3 Stage2 evidence adaptation."""

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

from opencood.models.heter_pyramid_single_hvp_heal_v3 import (
    HeterPyramidSingleHvpHealV3,
)
from opencood.models.hvp_heal_v3.evidence_head import HvpHealV3EvidenceHead
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss


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

    def forward_single(self, feature):
        feature = feature * self.scale
        return feature, [feature[:, :1]]


def main():
    torch.manual_seed(31)
    for modality_name in ("m2", "m3", "m4"):
        disabled_model = _build_dummy_model(modality_name, enabled=False)
        disabled_output = disabled_model(_dummy_data(modality_name))
        assert "hvp_v3" not in disabled_output

        enabled_model = _build_dummy_model(modality_name, enabled=True)
        enabled_output = enabled_model(_dummy_data(modality_name))
        _assert_forward_shapes(enabled_output, modality_name)

        target_dict = _dummy_target()
        criterion = PointPillarPyramidLoss(_loss_args())
        total_loss = criterion(enabled_output, target_dict)
        assert torch.is_tensor(total_loss)
        assert torch.isfinite(total_loss).all()
        assert total_loss.detach().item() > 0.0
        assert criterion.loss_dict["hvp_v3_stage"] == "stage2_evidence"
        assert criterion.loss_dict["hvp_v3_stage2_evidence_loss"] > 0.0
        assert criterion.loss_dict["hvp_v3_stage2_evidence_heatmap_loss"] > 0.0

        total_loss.backward()
        grads = [
            param.grad
            for name, param in enabled_model.named_parameters()
            if name.startswith("hvp_v3_evidence_head")
        ]
        assert any(
            grad is not None
            and torch.isfinite(grad).all()
            and grad.abs().sum() > 0
            for grad in grads
        )
        assert any(
            name.startswith("hvp_v3_evidence_head")
            for name in enabled_model.state_dict()
        )
        _assert_checkpoint_filter(enabled_model)
        print(f"HVP-HEAL v3 Stage2 {modality_name} evidence forward/backward OK")

    print("HVP-HEAL v3 Stage2 smoke OK")


def _build_dummy_model(modality_name, enabled):
    model = HeterPyramidSingleHvpHealV3.__new__(HeterPyramidSingleHvpHealV3)
    nn.Module.__init__(model)
    model.modality_name_list = [modality_name]
    model.sensor_type_dict = {modality_name: "lidar"}
    model.fix_modules = []
    model.shrink_flag = False
    setattr(model, f"encoder_{modality_name}", _DummyEncoder())
    setattr(model, f"backbone_{modality_name}", _DummyBackbone())
    setattr(model, f"aligner_{modality_name}", _DummyAligner())
    model.pyramid_backbone = _DummyPyramid()
    model.cls_head = nn.Conv2d(64, 2, kernel_size=1)
    model.reg_head = nn.Conv2d(64, 14, kernel_size=1)
    model.dir_head = nn.Conv2d(64, 4, kernel_size=1)
    cfg = {
        "enabled": bool(enabled),
        "stage": "stage2_evidence" if enabled else "none",
        "stage2": {
            "train_mode": "evidence_adaptation",
            "use_hvp_v3_evidence": bool(enabled),
        },
        "evidence_head": {
            "enable": bool(enabled),
            "in_channels": 64,
            "hidden_dim": 16,
            "descriptor_dim": 8,
            "use_sigmoid": True,
            "normalize_descriptor": True,
        },
        "evidence_loss": {
            "enable": bool(enabled),
            "mode": "stage2_evidence",
            "evidence_heatmap": {
                "enable": bool(enabled),
                "weight": 0.01,
                "pos_weight": 1.0,
            },
            "uncertainty": {
                "enable": bool(enabled),
                "weight": 0.001,
            },
            "descriptor": {
                "enable": bool(enabled),
                "weight": 0.001,
            },
        },
    }
    model.hvp_v3_cfg = HeterPyramidSingleHvpHealV3._normalize_hvp_v3_cfg(cfg)
    model.hvp_v3_enabled = bool(model.hvp_v3_cfg["enabled"])
    model.hvp_v3_stage = model.hvp_v3_cfg["stage"]
    if model._is_stage2_enabled():
        head_cfg = model.hvp_v3_cfg["evidence_head"]
        model.hvp_v3_evidence_head = HvpHealV3EvidenceHead(
            in_channels=head_cfg["in_channels"],
            hidden_dim=head_cfg["hidden_dim"],
            descriptor_dim=head_cfg["descriptor_dim"],
            use_sigmoid=head_cfg["use_sigmoid"],
            normalize_descriptor=head_cfg["normalize_descriptor"],
        )
        model._apply_stage2_train_mode()
    model.train()
    return model


def _assert_forward_shapes(output_dict, modality_name):
    assert output_dict["pyramid"] == "single"
    assert output_dict["cls_preds"].shape == (1, 2, 16, 16)
    assert output_dict["reg_preds"].shape == (1, 14, 16, 16)
    assert output_dict["dir_preds"].shape == (1, 4, 16, 16)
    assert "hvp_v3" in output_dict
    hvp_v3 = output_dict["hvp_v3"]
    assert hvp_v3["stage"] == "stage2_evidence"
    assert hvp_v3["modality"] == modality_name
    assert hvp_v3["evidence_heatmap_logits"].shape == (1, 1, 16, 16)
    assert hvp_v3["evidence_heatmap"].shape == (1, 1, 16, 16)
    assert hvp_v3["evidence_uncertainty"].shape == (1, 1, 16, 16)
    assert hvp_v3["evidence_descriptor"].shape == (1, 8, 16, 16)
    assert torch.isfinite(hvp_v3["evidence_descriptor"]).all()


def _assert_checkpoint_filter(model):
    state_dict = model.state_dict()
    fake_stage1 = {
        "hvp_v3_hypothesis_head.heatmap_head.weight": torch.randn(1, 16, 1, 1),
        "encoder_m1.scale": torch.ones(()),
        "backbone_m1.scale": torch.ones(()),
    }
    fake_stage1.update(state_dict)
    model.load_state_dict(fake_stage1, strict=False)
    skipped = getattr(model, "_hvp_v3_last_skipped_state_keys", [])
    assert "hvp_v3_hypothesis_head.heatmap_head.weight" in skipped
    assert "encoder_m1.scale" in skipped
    assert "backbone_m1.scale" in skipped


def _dummy_data(modality_name):
    return {
        f"inputs_{modality_name}": {
            "feature": torch.randn(1, 64, 16, 16),
        },
    }


def _dummy_target():
    pos = torch.zeros(1, 16, 16, 2)
    pos[0, 4, 5, 0] = 1.0
    pos[0, 10, 11, 1] = 1.0
    neg = 1.0 - pos
    targets = torch.zeros(1, 16, 16, 2, 7)
    return {
        "pos_equal_one": pos,
        "neg_equal_one": neg,
        "targets": targets,
    }


def _loss_args():
    dir_args = {
        "dir_offset": 0.7853,
        "num_bins": 2,
        "anchor_yaw": [0, 90],
    }
    return {
        "pos_cls_weight": 2.0,
        "cls": {
            "type": "SigmoidFocalLoss",
            "alpha": 0.25,
            "gamma": 2.0,
            "weight": 1.0,
        },
        "reg": {
            "type": "WeightedSmoothL1Loss",
            "sigma": 3.0,
            "codewise": True,
            "weight": 2.0,
        },
        "dir": {
            "type": "WeightedSoftmaxClassificationLoss",
            "weight": 0.2,
            "args": dir_args,
        },
        "depth": {
            "weight": 1.0,
        },
        "pyramid": {
            "relative_downsample": [1],
            "weight": [0.1],
        },
    }


if __name__ == "__main__":
    main()

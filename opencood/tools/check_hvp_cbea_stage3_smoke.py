"""Smoke test for HVP-CBEA v3 Stage3 collaborative aggregation."""

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
from opencood.models.heter_pyramid_collab_hvp_cbea import HeterPyramidCollabHvpCbea
from opencood.models.sub_modules.bayesian_hypothesis_fusion import (
    BayesianHypothesisFusion,
)
from opencood.models.sub_modules.hypothesis_encoder import HypothesisEncoder
from opencood.models.sub_modules.hypothesis_verifier import HypothesisVerifier
from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.loss.hvp_cbea_aux_loss import normalize_hvp_aux_loss_cfg


YAML_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "HEAL_XLab_v3_HVP_HEAL",
    "stage3",
    "cbea_aggregator.yaml",
)


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

    def forward_collab(self, heter_feature_2d, record_len, affine_matrix,
                       agent_modality_list, cam_crop_info):
        rows = []
        start = 0
        for length in record_len.detach().cpu().tolist():
            length = int(length)
            rows.append(heter_feature_2d[start:start + length].mean(dim=0))
            start += length
        fused = torch.stack(rows, dim=0) * self.scale
        return fused, [fused[:, :1]]


def main():
    torch.manual_seed(41)
    hypes = yaml_utils.load_yaml(YAML_PATH)
    _assert_stage3_yaml(hypes)

    model = _build_dummy_stage3_model(hypes)
    assert model.hvp_cbea_enabled
    assert model.hvp_train_only
    assert not model.hvp_packet_enabled
    assert hasattr(model, "hypothesis_encoder")
    assert hasattr(model, "hypothesis_verifier")
    assert hasattr(model, "bayesian_hypothesis_fusion")
    model.train()

    output_dict = model(_dummy_data())
    assert "hvp_cbea_debug" in output_dict
    assert "hvp_cbea_aux" in output_dict
    assert output_dict["hvp_cbea_debug"]["hvp_cbea_enabled"]
    assert output_dict["hvp_cbea_debug"].get("packet_enabled", False) is False
    assert output_dict["hvp_cbea_debug"].get("packet_used", False) is False
    assert output_dict["cls_preds"].shape == (1, 2, 8, 8)
    assert output_dict["reg_preds"].shape == (1, 14, 8, 8)
    assert output_dict["dir_preds"].shape == (1, 4, 8, 8)

    criterion = PointPillarDepthLoss(_loss_args())
    loss = criterion(output_dict, _dummy_target())
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss).all()
    assert loss.detach().item() > 0.0
    stats = criterion.loss_dict
    assert stats["cls_loss"] > 0.0
    assert stats["reg_loss"] >= 0.0
    assert stats["hvp_cbea_loss"] > 0.0
    assert stats["hvp_aux_enabled"] is True
    assert stats["hvp_aux_total_loss"] > 0.0
    assert stats["hvp_gt_hypothesis_heatmap_loss"] > 0.0
    assert stats["hvp_gt_residual_focus_loss"] > 0.0
    assert stats["hvp_residual_reg_loss"] >= 0.0
    assert stats["hvp_alpha_reg_loss"] >= 0.0

    loss.backward()
    hvp_grads = [
        param.grad
        for name, param in model.named_parameters()
        if model._is_hvp_module_name(name)
    ]
    assert any(
        grad is not None
        and torch.isfinite(grad).all()
        and grad.detach().abs().sum() > 0
        for grad in hvp_grads
    )
    base_grads = [
        param.grad
        for name, param in model.named_parameters()
        if not model._is_hvp_module_name(name)
    ]
    assert all(grad is None for grad in base_grads)

    print("HVP-CBEA Stage3 yaml load OK")
    print("HVP-CBEA Stage3 model create OK")
    print("HVP-CBEA Stage3 packet disabled OK")
    print("HVP-CBEA Stage3 forward OK")
    print("HVP-CBEA Stage3 loss OK")
    print("HVP-CBEA Stage3 backward OK")


def _assert_stage3_yaml(hypes):
    assert hypes["name"] == "HVP_CBEA_v3/stage3/cbea_aggregator"
    assert hypes["model"]["core_method"] == "heter_pyramid_collab_hvp_cbea"
    hvp_cfg = hypes["model"]["args"]["hvp_cbea"]
    assert hvp_cfg["enabled"] is True
    assert hvp_cfg["packet"]["enabled"] is False
    assert hvp_cfg["aux_loss"]["enabled"] is True
    assert hvp_cfg["aux_loss"]["gt_guided"]["hypothesis_heatmap"]["enabled"] is True
    assert hvp_cfg["aux_loss"]["gt_guided"]["residual_focus"]["enabled"] is True
    assert hvp_cfg["aux_loss"]["alpha_reg"]["enabled"] is True
    assert hvp_cfg["aux_loss"]["residual_reg"]["enabled"] is True


def _build_dummy_stage3_model(hypes):
    channels = 32
    model = HeterPyramidCollabHvpCbea.__new__(HeterPyramidCollabHvpCbea)
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

    args = {
        "lidar_range": hypes["model"]["args"]["lidar_range"],
        "in_head": channels,
        "fusion_backbone": {"num_filters": [channels]},
    }
    cfg = model._default_hvp_cfg(args)
    cfg.update(hypes["model"]["args"]["hvp_cbea"])
    cfg.update({
        "in_channels": channels,
        "collaborator_in_channels": channels,
        "mid_channels": 16,
        "max_hypotheses": 8,
        "hyp_conf_threshold": 0.0,
        "max_novel": 4,
        "debug": False,
    })
    cfg["residual_gate"] = model._normalize_residual_gate_cfg(cfg.get("residual_gate"))
    cfg["packet"] = model._normalize_packet_cfg(cfg.get("packet"))
    cfg["aux_loss"] = normalize_hvp_aux_loss_cfg(cfg.get("aux_loss"))
    model.hvp_cbea_cfg = cfg
    model.hvp_cbea_enabled = bool(cfg["enabled"])
    model.hvp_train_only = bool(cfg["train_only_hvp"])
    model.hvp_packet_enabled = bool(cfg["packet"]["enabled"])
    model.hvp_aux_loss_enabled = bool(cfg["aux_loss"]["enabled"])
    model.hvp_trainable_summary = None

    model.hvp_collaborator_proj = nn.Identity()
    model.hypothesis_encoder = HypothesisEncoder(
        in_channels=channels,
        mid_channels=cfg["mid_channels"],
        max_hypotheses=cfg["max_hypotheses"],
        hyp_conf_threshold=cfg["hyp_conf_threshold"],
        pc_range=cfg.get("pc_range"),
    )
    model.hypothesis_verifier = HypothesisVerifier(
        in_channels=channels,
        mid_channels=cfg["mid_channels"],
        max_novel=cfg["max_novel"],
        novel_threshold=cfg["novel_threshold"],
    )
    model.bayesian_hypothesis_fusion = BayesianHypothesisFusion(
        in_channels=channels,
        mid_channels=cfg["mid_channels"],
        pc_range=cfg.get("pc_range"),
        confirm_boost=cfg["confirm_boost"],
        refute_penalty=cfg["refute_penalty"],
        refine_boost=cfg["refine_boost"],
        residual_gate=cfg["residual_gate"],
    )
    if model.hvp_train_only:
        model._freeze_non_hvp_parameters()
        model._set_frozen_heal_modules_eval()
    model.hvp_trainable_summary = model._summarize_trainable_parameters()
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
        feature = torch.randn(1, 32, 8, 8) + float(idx) * 0.1
        data[f"inputs_{modality_name}"] = {"feature": feature}
    return data


def _dummy_target():
    pos = torch.zeros(1, 8, 8, 2)
    pos[0, 2, 3, 0] = 1.0
    pos[0, 5, 4, 1] = 1.0
    neg = 1.0 - pos
    targets = torch.zeros(1, 8, 8, 2, 7)
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
    }


if __name__ == "__main__":
    main()

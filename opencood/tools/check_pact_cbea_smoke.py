"""Smoke test for PACT-CBEA v1 Feature Mode."""

import argparse
import os
import sys
import types

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if "opencood.utils.box_overlaps" not in sys.modules:
    box_overlaps_stub = types.ModuleType("opencood.utils.box_overlaps")
    box_overlaps_stub.bbox_overlaps = lambda *args, **kwargs: None
    sys.modules["opencood.utils.box_overlaps"] = box_overlaps_stub
if "icecream" not in sys.modules:
    icecream_stub = types.ModuleType("icecream")
    icecream_stub.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
    sys.modules["icecream"] = icecream_stub
if "shapely.geometry" not in sys.modules:
    shapely_stub = types.ModuleType("shapely")
    shapely_geometry_stub = types.ModuleType("shapely.geometry")

    class _Polygon:
        def __init__(self, *args, **kwargs):
            self.area = 0.0

        def intersection(self, other):
            return self

        def union(self, other):
            return self

    shapely_geometry_stub.Polygon = _Polygon
    shapely_stub.geometry = shapely_geometry_stub
    sys.modules.setdefault("shapely", shapely_stub)
    sys.modules.setdefault("shapely.geometry", shapely_geometry_stub)
if "pyquaternion" not in sys.modules:
    pyquaternion_stub = types.ModuleType("pyquaternion")

    class _Quaternion:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def transformation_matrix(self):
            return torch.eye(4).numpy()

    pyquaternion_stub.Quaternion = _Quaternion
    sys.modules.setdefault("pyquaternion", pyquaternion_stub)

from opencood.hypes_yaml import yaml_utils
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.models.heter_pyramid_collab_pact_cbea import (
    HeterPyramidCollabPactCbea,
)
from opencood.models.heter_pyramid_collab_pact_cbea_stage1 import (
    HeterPyramidCollabPactCbeaStage1,
)
from opencood.models.heter_pyramid_single_pact_cbea import (
    HeterPyramidSinglePactCbea,
)
from opencood.models.sub_modules.pact_cbea_evidence_head import (
    PACTCBEALocalEvidenceHead,
)
from opencood.models.sub_modules.pact_cbea_rule import PACTCBEARule


PACT_YAML = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "HEAL_XLab_v3_HVP_HEAL",
    "pact",
    "cbea_rule.yaml",
)
STAGE1_YAML = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "HEAL_XLab_v3_HVP_HEAL",
    "pact",
    "stage1",
    "m1_local_evidence.yaml",
)
STAGE2_LOCAL_YAMLS = {
    "m2": os.path.join(REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v3_HVP_HEAL", "pact", "stage2", "m2_local_evidence_adapt.yaml"),
    "m3": os.path.join(REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v3_HVP_HEAL", "pact", "stage2", "m3_local_evidence_adapt.yaml"),
    "m4": os.path.join(REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v3_HVP_HEAL", "pact", "stage2", "m4_local_evidence_adapt.yaml"),
}


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
        self.align_corners = False
        self.scale = nn.Parameter(torch.ones(()))

    def forward_single(self, feature):
        feature = feature * self.scale
        return feature, [feature[:, :1]]

    def forward_collab(self, feature, record_len, affine_matrix, agent_modality_list=None, cam_crop_info=None):
        rows = []
        start = 0
        for length in record_len.detach().cpu().view(-1).tolist():
            rows.append(feature[start].clone())
            start += int(length)
        return torch.stack(rows, dim=0) * self.scale, [feature[:, :1]]


def main():
    options = _parse_args()
    torch.manual_seed(71)
    pact_hypes = yaml_utils.load_yaml(PACT_YAML)
    pact_cfg = pact_hypes["model"]["args"]["pact_cbea"]
    _assert_pact_yaml(pact_hypes, pact_cfg)

    rule = PACTCBEARule(pact_cfg)
    assert _trainable_count(rule) == 0
    assert sum(param.numel() for param in rule.parameters()) == 0
    _check_fallback_aggregation(rule)
    _check_evidence_routing(rule)
    _check_uncertainty_routing(rule)
    _check_spatial_consistency(rule)
    _check_modality_prior(pact_cfg)
    _check_multi_scene_rule(rule)

    stage1_hypes = yaml_utils.load_yaml(STAGE1_YAML)
    _assert_stage1_yaml(stage1_hypes)
    stage1_model = _build_dummy_stage1_model(
        stage1_hypes["model"]["args"]["pact_cbea"]
    )
    stage1_output = stage1_model(_dummy_stage1_data())
    _assert_stage1_output(stage1_output)
    _check_stage1_trainability_and_backward(stage1_model, stage1_output)
    local_param_count = _local_head_param_count(stage1_model, "m1")
    print("PACT-CBEA Stage1 collab forward/loss/backward OK")
    _check_real_stage1_batch(STAGE1_YAML, options.require_real_stage1)

    for modality_name, yaml_path in STAGE2_LOCAL_YAMLS.items():
        hypes = yaml_utils.load_yaml(yaml_path)
        _assert_stage2_local_yaml(hypes, modality_name)
        model = _build_dummy_single_model(modality_name, hypes["model"]["args"]["pact_cbea"])
        output_dict = model(_dummy_single_data(modality_name))
        _assert_local_evidence_output(output_dict, modality_name)
        local_param_count = max(local_param_count, _local_head_param_count(model, modality_name))
        _check_local_loss_backward(model, output_dict)
        print(f"PACT-CBEA local expert {modality_name} forward/backward OK")

    collab_model = _build_dummy_collab_model(pact_cfg)
    assert _trainable_count(collab_model.pact_cbea_rule) == 0
    assert _trainable_count(collab_model) == 0
    _check_warp_helpers(collab_model)
    output_dict, data_dict = _run_collab_forward(collab_model)
    _assert_collab_output(output_dict)
    _check_collab_backward(output_dict, data_dict)
    _check_single_car_fallback(collab_model)
    _check_missing_evidence_fallback(pact_cfg)

    print("PACT-CBEA yaml load OK")
    print("PACT-CBEA Stage1 local expert create OK")
    print("PACT-CBEA Stage1 base train modules OK")
    print("PACT-CBEA Stage2 local expert create OK")
    print("PACT-CBEA local evidence output OK")
    print("PACT-CBEA model create OK")
    print("PACT-CBEA HEAL-compatible config OK")
    print("PACT-CBEA no-joint-training config OK")
    print("PACT-CBEA rule parameter-free OK")
    print("PACT-CBEA fallback aggregation OK")
    print("PACT-CBEA evidence routing OK")
    print("PACT-CBEA uncertainty routing OK")
    print("PACT-CBEA spatial consistency OK")
    print("PACT-CBEA modality prior OK")
    print("PACT-CBEA feature warp OK")
    print("PACT-CBEA auto local evidence OK")
    print("PACT-CBEA multi-scene record_len OK")
    print("PACT-CBEA forward OK")
    print("PACT-CBEA rule parameter count: 0")
    print(f"PACT-CBEA local evidence head parameter count: {local_param_count}")
    print("PACT-CBEA smoke OK")


def _parse_args():
    parser = argparse.ArgumentParser(description="PACT-CBEA v1 smoke checks")
    parser.add_argument(
        "--require-real-stage1",
        action="store_true",
        help="fail instead of skipping when the local OPV2V Stage1 dataset is unavailable",
    )
    return parser.parse_args()


def _assert_pact_yaml(hypes, pact_cfg):
    assert hypes["name"] == "PACT_CBEA_v1/rule_cbea"
    assert hypes["model"]["core_method"] == "heter_pyramid_collab_pact_cbea"
    assert pact_cfg["enabled"] is True
    assert pact_cfg["trainable"] is False
    assert pact_cfg["no_joint_training"] is True
    assert pact_cfg["use_stage3_joint_training"] is False
    assert pact_cfg["local_evidence"]["enabled"] is True
    assert pact_cfg["evidence_head"]["enabled"] is True


def _assert_stage1_yaml(hypes):
    assert hypes["name"] == "PACT_CBEA_v1/stage1/m1_base"
    assert hypes["fusion"]["core_method"] == "intermediateheter"
    assert hypes["model"]["core_method"] == "heter_pyramid_collab_pact_cbea_stage1"
    cfg = hypes["model"]["args"]["pact_cbea"]
    assert cfg["enabled"] is True
    assert cfg["stage"] == "stage1_local_evidence"
    assert cfg["trainable"] is True
    assert cfg["no_joint_training"] is True
    assert cfg["use_stage3_joint_training"] is False
    assert cfg["local_evidence"]["train_mode"] == "stage1_base_train"
    assert cfg["local_evidence"]["enabled"] is True
    assert cfg["evidence_head"]["enabled"] is True


def _assert_stage2_local_yaml(hypes, modality_name):
    assert hypes["model"]["core_method"] == "heter_pyramid_single_pact_cbea"
    assert hypes["name"].startswith("PACT_CBEA_v1/")
    cfg = hypes["model"]["args"]["pact_cbea"]
    assert cfg["enabled"] is True
    assert cfg["stage"] == "local_evidence"
    assert cfg["trainable"] is True
    assert cfg["no_joint_training"] is True
    assert cfg["use_stage3_joint_training"] is False
    assert cfg["local_evidence"]["enabled"] is True
    assert cfg["evidence_loss"]["descriptor"].get("enabled", cfg["evidence_loss"]["descriptor"].get("enable")) is False
    assert modality_name in hypes["name"]


def _build_dummy_stage1_model(pact_cfg):
    modality_name = "m1"
    model = HeterPyramidCollabPactCbeaStage1.__new__(
        HeterPyramidCollabPactCbeaStage1
    )
    nn.Module.__init__(model)
    model.modality_name_list = [modality_name]
    model.sensor_type_dict = {modality_name: "lidar"}
    model.cam_crop_info = {}
    model.H = 8
    model.W = 8
    model.fake_voxel_size = 1
    model.compress = False
    model.shrink_flag = True
    setattr(model, "encoder_m1", _DummyEncoder())
    setattr(model, "backbone_m1", _DummyBackbone())
    setattr(model, "aligner_m1", _DummyAligner())
    setattr(model, "depth_supervision_m1", False)
    model.pyramid_backbone = _DummyPyramid()
    model.shrink_conv = nn.Conv2d(64, 64, kernel_size=1)
    model.cls_head = nn.Conv2d(64, 2, kernel_size=1)
    model.reg_head = nn.Conv2d(64, 14, kernel_size=1)
    model.dir_head = nn.Conv2d(64, 4, kernel_size=1)
    model.pact_cbea_cfg = HeterPyramidCollabPactCbeaStage1._normalize_pact_stage1_cfg({
        **pact_cfg,
        "evidence_head": {
            **pact_cfg["evidence_head"],
            "in_channels": 64,
            "hidden_dim": 16,
            "descriptor_dim": 8,
        },
    })
    model.pact_cbea_enabled = bool(model.pact_cbea_cfg["enabled"])
    model.pact_cbea_rule = PACTCBEARule(model.pact_cbea_cfg)
    model._pact_stage1_modality = modality_name
    model.pact_cbea_evidence_head_m1 = PACTCBEALocalEvidenceHead(
        in_channels=64,
        hidden_dim=16,
        descriptor_dim=8,
        use_sigmoid=True,
        normalize_descriptor=True,
    )
    model.train()
    return model


def _assert_stage1_output(output_dict):
    assert output_dict["pyramid"] == "collab"
    assert output_dict["cls_preds"].shape == (1, 2, 16, 16)
    assert output_dict["reg_preds"].shape == (1, 14, 16, 16)
    assert output_dict["dir_preds"].shape == (1, 4, 16, 16)
    pact = output_dict["pact_cbea"]
    assert pact["stage"] == "local_evidence"
    assert pact["train_mode"] == "stage1_base_train"
    assert pact["global_rule_trainable"] is False
    assert pact["evidence_heatmap_logits"].shape == (1, 1, 16, 16)
    assert pact["evidence_uncertainty"].shape == (1, 1, 16, 16)


def _check_stage1_trainability_and_backward(model, output_dict):
    expected_modules = (
        "encoder_m1",
        "backbone_m1",
        "pyramid_backbone",
        "shrink_conv",
        "cls_head",
        "reg_head",
        "dir_head",
        "pact_cbea_evidence_head_m1",
    )
    for module_name in expected_modules:
        module = getattr(model, module_name)
        assert any(param.requires_grad for param in module.parameters()), module_name
    assert _trainable_count(model.pact_cbea_rule) == 0

    criterion = PointPillarPyramidLoss(_loss_args())
    total_loss = criterion(output_dict, _dummy_target())
    assert torch.isfinite(total_loss)
    assert criterion.loss_dict["pact_cbea_local_evidence_enabled"] is True
    assert criterion.loss_dict["pact_cbea_local_evidence_loss"] > 0.0
    total_loss.backward()
    for module_name in expected_modules:
        module = getattr(model, module_name)
        assert any(
            param.grad is not None
            and torch.isfinite(param.grad).all()
            and param.grad.abs().sum() > 0
            for param in module.parameters()
        ), module_name


def _check_real_stage1_batch(yaml_path, required):
    hypes = yaml_utils.load_yaml(yaml_path)
    paths = [
        _resolve_repo_path(hypes["root_dir"]),
        _resolve_repo_path(hypes["validate_dir"]),
        _resolve_repo_path(hypes["heter"].get("assignment_path")),
    ]
    missing_paths = [path for path in paths if not path or not os.path.exists(path)]
    if missing_paths:
        message = "PACT-CBEA Stage1 real DataLoader batch smoke skipped; missing: %s" % (
            ", ".join(missing_paths)
        )
        if required:
            raise RuntimeError(message)
        print(message)
        return False

    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils

    train_dataset = build_dataset(hypes, visualize=False, train=True)
    validate_dataset = build_dataset(hypes, visualize=False, train=False)
    assert len(train_dataset) == 6374, len(train_dataset)
    assert len(validate_dataset) == 1980, len(validate_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=train_dataset.collate_batch_train,
        shuffle=False,
    )
    batch_data = next(iter(train_loader))
    assert batch_data is not None and "ego" in batch_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes).to(device)
    criterion = train_utils.create_loss(hypes)
    model.train()
    model.model_train_init()
    batch_data = train_utils.to_device(batch_data, device)
    output_dict = model(batch_data["ego"])
    total_loss = criterion(output_dict, batch_data["ego"]["label_dict"])
    assert torch.isfinite(total_loss)
    total_loss.backward()
    head = model.pact_cbea_evidence_head_m1
    assert any(
        param.grad is not None
        and torch.isfinite(param.grad).all()
        and param.grad.abs().sum() > 0
        for param in head.parameters()
    )
    print(
        "PACT-CBEA Stage1 real DataLoader batch forward/loss/backward OK "
        "(train=%d validate=%d)" % (len(train_dataset), len(validate_dataset))
    )
    return True


def _resolve_repo_path(path):
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _check_fallback_aggregation(rule):
    zeros = torch.zeros(1, 4, 3, 3)
    ones = torch.ones(1, 4, 3, 3)
    enhanced, debug = rule([zeros, ones])
    assert enhanced.shape == (1, 4, 3, 3)
    assert torch.allclose(enhanced, torch.full_like(enhanced, 0.5), atol=1e-5)
    assert "missing_evidence_heatmap_confidence_ones" in debug["pact_fallbacks"]
    _assert_debug_keys(debug)


def _check_evidence_routing(rule):
    features = torch.stack([
        torch.zeros(4, 3, 3),
        torch.full((4, 3, 3), 10.0),
    ], dim=0)
    low = torch.full((1, 3, 3), -5.0)
    high = torch.full((1, 3, 3), 5.0)
    uncertainty = torch.zeros(2, 1, 3, 3)
    _, debug_a = rule(
        features,
        evidence_heatmap=torch.stack([low, high], dim=0),
        evidence_uncertainty=uncertainty,
        modality_names=["m1", "m2"],
        pairwise_t_matrix=torch.eye(4).view(1, 1, 4, 4),
    )
    _, debug_b = rule(
        features,
        evidence_heatmap=torch.stack([high, low], dim=0),
        evidence_uncertainty=uncertainty,
        modality_names=["m1", "m2"],
        pairwise_t_matrix=torch.eye(4).view(1, 1, 4, 4),
    )
    assert debug_a["pact_alpha"][:, 1].mean() > debug_a["pact_alpha"][:, 0].mean()
    assert not torch.allclose(debug_a["pact_alpha"], debug_b["pact_alpha"])


def _check_uncertainty_routing(rule):
    features = torch.stack([torch.zeros(4, 3, 3), torch.ones(4, 3, 3)], dim=0)
    heatmap = torch.ones(2, 1, 3, 3)
    uncertainty = torch.stack([
        torch.zeros(1, 3, 3),
        torch.full((1, 3, 3), 5.0),
    ], dim=0)
    _, debug = rule(
        features,
        evidence_heatmap=heatmap,
        evidence_uncertainty=uncertainty,
        modality_names=["m1", "m2"],
    )
    assert debug["pact_alpha"][:, 0].mean() > debug["pact_alpha"][:, 1].mean()


def _check_spatial_consistency(rule):
    features = torch.ones(2, 4, 3, 3)
    heatmap = torch.stack([
        torch.full((1, 3, 3), 5.0),
        torch.full((1, 3, 3), -5.0),
    ], dim=0)
    _, debug = rule(
        features,
        evidence_heatmap=heatmap,
        evidence_uncertainty=torch.zeros(2, 1, 3, 3),
        modality_names=["m1", "m2"],
    )
    spatial = debug["pact_spatial_weight"]
    assert spatial[:, 1].mean() < spatial[:, 0].mean()


def _check_modality_prior(base_cfg):
    cfg = PACTCBEARule._normalize_cfg(base_cfg)
    cfg["weights"]["modality_prior"] = {"m1": 1.0, "m2": 3.0}
    rule = PACTCBEARule(cfg)
    features = torch.stack([
        torch.zeros(4, 3, 3),
        torch.full((4, 3, 3), 10.0),
    ], dim=0)
    enhanced, debug = rule(features, modality_names=["m1", "m2"])
    assert enhanced.mean() > 7.0
    assert debug["pact_alpha"][:, 1].mean() > debug["pact_alpha"][:, 0].mean()


def _check_multi_scene_rule(rule):
    features = torch.randn(4, 4, 3, 3)
    heatmap = torch.randn(4, 1, 3, 3)
    uncertainty = torch.rand(4, 1, 3, 3)
    enhanced, debug = rule(
        features,
        evidence_heatmap=heatmap,
        evidence_uncertainty=uncertainty,
        record_len=torch.tensor([2, 2]),
        modality_names=["m1", "m2", "m3", "m4"],
    )
    assert enhanced.shape == (2, 4, 3, 3)
    assert debug["pact_batch_size"] == 2


def _build_dummy_single_model(modality_name, pact_cfg):
    model = HeterPyramidSinglePactCbea.__new__(HeterPyramidSinglePactCbea)
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
    model.pact_cbea_cfg = HeterPyramidSinglePactCbea._normalize_pact_local_cfg({
        **pact_cfg,
        "evidence_head": {
            **pact_cfg["evidence_head"],
            "in_channels": 64,
            "hidden_dim": 16,
            "descriptor_dim": 8,
        },
    })
    model.pact_cbea_enabled = bool(model.pact_cbea_cfg["enabled"])
    model.pact_cbea_stage = model.pact_cbea_cfg["stage"]
    setattr(
        model,
        f"pact_cbea_evidence_head_{modality_name}",
        PACTCBEALocalEvidenceHead(
            in_channels=64,
            hidden_dim=16,
            descriptor_dim=8,
            use_sigmoid=True,
            normalize_descriptor=True,
        ),
    )
    model._pact_local_modality = modality_name
    model.train()
    return model


def _assert_local_evidence_output(output_dict, modality_name):
    assert output_dict["pyramid"] == "single"
    assert output_dict["cls_preds"].shape == (1, 2, 16, 16)
    assert "pact_cbea" in output_dict
    pact = output_dict["pact_cbea"]
    assert pact["stage"] == "local_evidence"
    assert pact["modality"] == modality_name
    assert pact["evidence_heatmap_logits"].shape == (1, 1, 16, 16)
    assert pact["evidence_heatmap"].shape == (1, 1, 16, 16)
    assert pact["evidence_uncertainty"].shape == (1, 1, 16, 16)


def _check_local_loss_backward(model, output_dict):
    criterion = PointPillarPyramidLoss(_loss_args())
    total_loss = criterion(output_dict, _dummy_target())
    assert torch.is_tensor(total_loss)
    assert torch.isfinite(total_loss)
    assert criterion.loss_dict["pact_cbea_local_evidence_enabled"] is True
    assert criterion.loss_dict["pact_cbea_local_evidence_loss"] > 0.0
    total_loss.backward()
    grads = [
        param.grad
        for name, param in model.named_parameters()
        if name.startswith("pact_cbea_evidence_head_")
    ]
    assert any(
        grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        for grad in grads
    )


def _build_dummy_collab_model(pact_cfg):
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
    cfg = HeterPyramidCollabPactCbea._normalize_pact_cfg({
        **pact_cfg,
        "evidence_head": {
            **pact_cfg["evidence_head"],
            "in_channels": channels,
            "hidden_dim": 16,
            "descriptor_dim": 8,
        },
    })
    model.pact_cbea_cfg = cfg
    model.pact_cbea_enabled = bool(cfg["enabled"])
    model.pact_cbea_trainable = bool(cfg["trainable"])
    model.pact_no_joint_training = bool(cfg["no_joint_training"])
    model.pact_use_stage3_joint_training = bool(cfg["use_stage3_joint_training"])
    model.pact_cbea_rule = PACTCBEARule(cfg)
    model._build_local_evidence_heads({"in_head": channels})
    model._freeze_all_model_parameters()
    return model


def _check_warp_helpers(model):
    feature = torch.zeros(2, 1, 5, 5)
    feature[1, 0, 2, 2] = 1.0
    record_len = torch.tensor([2])
    affine = torch.eye(2, 3).view(1, 1, 1, 2, 3).repeat(1, 2, 2, 1, 1)
    affine[0, 0, 1, 0, 2] = 0.5
    warped_feature = model._warp_to_ego(feature, record_len, affine)
    warped_evidence = model._warp_to_ego(feature, record_len, affine)
    warped_uncertainty = model._warp_to_ego(feature, record_len, affine)
    assert not torch.allclose(warped_feature[1], feature[1])
    assert torch.allclose(warped_evidence, warped_uncertainty)


def _run_collab_forward(model):
    data_dict = _dummy_collab_data(["m1", "m2", "m3", "m4"], torch.tensor([2, 2]))
    output_dict = model(data_dict)
    return output_dict, data_dict


def _assert_collab_output(output_dict):
    assert "pact_cbea" in output_dict
    assert output_dict["cls_preds"].shape == (2, 2, 8, 8)
    debug = output_dict["pact_cbea"]
    _assert_debug_keys(debug)
    assert debug["pact_features_warped_to_ego"] is True
    assert debug["pact_evidence_warped_to_ego"] is True
    assert debug["pact_uncertainty_warped_to_ego"] is True
    assert debug["pact_used_base_heal_fallback"] is False
    assert "pact_evidence_heatmap_logits" in debug
    assert "pact_evidence_uncertainty" in debug


def _check_collab_backward(output_dict, data_dict):
    loss = output_dict["cls_preds"].mean() + output_dict["reg_preds"].abs().mean()
    loss.backward()
    input_grads = [
        payload["feature"].grad
        for key, payload in data_dict.items()
        if key.startswith("inputs_")
    ]
    assert any(
        grad is not None and torch.isfinite(grad).all() and grad.detach().abs().sum() > 0
        for grad in input_grads
    )


def _check_single_car_fallback(model):
    data_dict = _dummy_collab_data(["m1"], torch.tensor([1]))
    output_dict = model(data_dict)
    assert output_dict["cls_preds"].shape == (1, 2, 8, 8)
    debug = output_dict["pact_cbea"]
    assert debug["pact_agent_count"] == 1


def _check_missing_evidence_fallback(pact_cfg):
    cfg = HeterPyramidCollabPactCbea._normalize_pact_cfg({
        **pact_cfg,
        "local_evidence": {"enabled": False},
        "evidence_head": {"enabled": False},
    })
    model = _build_dummy_collab_model({
        **cfg,
        "local_evidence": {"enabled": False},
        "evidence_head": {"enabled": False, "in_channels": 32, "hidden_dim": 16, "descriptor_dim": 8},
    })
    data_dict = _dummy_collab_data(["m1", "m2"], torch.tensor([2]))
    output_dict = model(data_dict)
    assert output_dict["pact_cbea"]["pact_used_base_heal_fallback"] is True
    assert output_dict["cls_preds"].shape == (1, 2, 8, 8)


def _dummy_single_data(modality_name):
    return {
        f"inputs_{modality_name}": {
            "feature": torch.randn(1, 64, 16, 16),
        },
    }


def _dummy_stage1_data():
    return {
        "agent_modality_list": ["m1"],
        "record_len": torch.tensor([1]),
        "pairwise_t_matrix": torch.eye(4).view(1, 1, 1, 4, 4),
        "inputs_m1": {
            "feature": torch.randn(1, 64, 16, 16),
        },
    }


def _dummy_collab_data(modalities, record_len):
    max_cav = int(record_len.max().item())
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4).repeat(record_len.numel(), max_cav, max_cav, 1, 1)
    data = {
        "agent_modality_list": modalities,
        "pairwise_t_matrix": pairwise,
        "record_len": record_len,
    }
    counts = {name: modalities.count(name) for name in set(modalities)}
    for idx, (modality_name, count) in enumerate(counts.items()):
        feature = torch.randn(count, 32, 8, 8, requires_grad=True) + float(idx) * 0.1
        feature.retain_grad()
        data[f"inputs_{modality_name}"] = {"feature": feature}
    return data


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


def _assert_debug_keys(debug):
    for key in (
        "pact_alpha",
        "pact_reliability",
        "pact_evidence_confidence",
        "pact_uncertainty_weight",
        "pact_modality_prior",
        "pact_spatial_weight",
        "pact_mode",
    ):
        assert key in debug
    assert debug["pact_mode"] == "trust_calibrated_rule"


def _trainable_count(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _local_head_param_count(model, modality_name):
    head = getattr(model, f"pact_cbea_evidence_head_{modality_name}")
    return sum(param.numel() for param in head.parameters())


if __name__ == "__main__":
    main()

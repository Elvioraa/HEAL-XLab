"""CPU smoke checks for the isolated original-PACT rule-uniform ablation."""

from __future__ import absolute_import, division, print_function

import ast
import copy
import os
import subprocess
import sys

import torch
import torch.nn as nn
import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MODEL_PATH = os.path.join(
    ROOT,
    "opencood",
    "models",
    "heter_pyramid_collab_pact_cbea_original_rule_uniform.py",
)
ROUTER_PATH = os.path.join(
    ROOT,
    "opencood",
    "models",
    "sub_modules",
    "pact_cbea_original_rule_uniform.py",
)
YAML_PATH = os.path.join(
    ROOT,
    "opencood",
    "hypes_yaml",
    "PACT_CBEA_ORIGINAL_RULE_UNIFORM_v1",
    "rule_uniform.yaml",
)
SOURCE_YAML_PATH = os.path.join(
    ROOT,
    "opencood",
    "hypes_yaml",
    "HEAL_XLab_v3_HVP_HEAL",
    "pact",
    "cbea_rule.yaml",
)


class _ParameterlessEvidenceHead(nn.Module):
    """Raise if the ablation ever tries to consume an evidence head."""

    def forward(self, feature):
        raise AssertionError("original-rule-uniform must not call evidence heads")


class _PassBlock(nn.Module):
    def __init__(self):
        super(_PassBlock, self).__init__()
        self.token = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(2)

    def forward(self, value):
        return value


class _Backbone(_PassBlock):
    def forward(self, payload):
        return {"spatial_features_2d": payload["spatial_features"]}


class _Pyramid(nn.Module):
    def __init__(self):
        super(_Pyramid, self).__init__()
        self.token = nn.Parameter(torch.ones(1))
        self.forward_collab_calls = 0
        self.align_corners = False

    def forward_single(self, feature):
        return feature, [feature[:, :1]]

    def forward_collab(self, feature, record_len, affine_matrix,
                       agent_modality_list, cam_crop_info):
        del affine_matrix, agent_modality_list, cam_crop_info
        self.forward_collab_calls += 1
        records = [int(item) for item in record_len.detach().cpu().tolist()]
        result = []
        offset = 0
        for count in records:
            result.append(feature[offset:offset + count].mean(dim=0, keepdim=True) * 99.0)
            offset += count
        return torch.cat(result, dim=0), ["forward_collab_occ"]


class _RecordingHead(nn.Module):
    def __init__(self):
        super(_RecordingHead, self).__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.inputs = []

    def forward(self, feature):
        self.inputs.append(feature.detach().clone())
        return feature * self.scale.view(1, 1, 1, 1)


def _fake_parent_init(self, args):
    """A minimal HeterPyramidCollabPactCbea body for discovery/forward smoke."""
    del args
    nn.Module.__init__(self)
    self.modality_name_list = ["m1", "m2", "m3", "m4"]
    self.sensor_type_dict = {
        "m1": "lidar", "m2": "lidar", "m3": "lidar", "m4": "lidar",
    }
    self.H = 4
    self.W = 4
    self.fake_voxel_size = 1.0
    self.compress = False
    self.shrink_flag = False
    self.supervise_single = True
    self.cam_crop_info = None
    self.pact_cbea_enabled = True
    self.pact_cbea_trainable = False
    self.pact_no_joint_training = True
    self.pact_use_stage3_joint_training = False
    for modality in self.modality_name_list:
        encoder = _PassBlock()
        encoder.forward = lambda data, name, modality_name=modality: data["features"][modality_name]
        setattr(self, "encoder_%s" % modality, encoder)
        setattr(self, "backbone_%s" % modality, _Backbone())
        setattr(self, "aligner_%s" % modality, _PassBlock())
    self.pyramid_backbone = _Pyramid()
    self.cls_head = _RecordingHead()
    self.reg_head = _RecordingHead()
    self.dir_head = _RecordingHead()
    self.pact_cbea_evidence_head_m1 = _ParameterlessEvidenceHead()


def _args(enabled=True):
    return {
        "supervise_single": True,
        "pact_cbea": {
            "enabled": True,
            "trainable": False,
            "no_joint_training": True,
            "use_stage3_joint_training": False,
            "local_evidence": {"enabled": True},
            "evidence_head": {"enabled": True},
        },
        "pact_cbea_original_rule_uniform": {"enabled": enabled},
    }


def _pairwise_identity():
    pairwise = torch.eye(4, dtype=torch.float32).view(1, 1, 1, 4, 4)
    return pairwise.repeat(2, 5, 5, 1, 1)


def _data(external_evidence=None):
    data = {
        "agent_modality_list": ["m1", "m2", "m1", "m2", "m1"],
        "record_len": torch.tensor([2, 3], dtype=torch.long),
        "pairwise_t_matrix": _pairwise_identity(),
        "features": {
            "m1": torch.stack((
                torch.full((2, 4, 4), 1.0),
                torch.full((2, 4, 4), 3.0),
                torch.full((2, 4, 4), 5.0),
            )),
            "m2": torch.stack((
                torch.full((2, 4, 4), 2.0),
                torch.full((2, 4, 4), 4.0),
            )),
            "m3": torch.empty((0, 2, 4, 4)),
            "m4": torch.empty((0, 2, 4, 4)),
        },
    }
    if external_evidence is not None:
        data["evidence_heatmap"] = external_evidence
        data["evidence_uncertainty"] = external_evidence * 3.0
    return data


def _assert_python38_ast():
    for path in (MODEL_PATH, ROUTER_PATH, __file__):
        with open(path, "r") as handle:
            source = handle.read()
        ast.parse(source, filename=path, feature_version=(3, 8))


def _assert_yaml_contract():
    with open(SOURCE_YAML_PATH, "r") as handle:
        source = yaml.safe_load(handle)
    with open(YAML_PATH, "r") as handle:
        target = yaml.safe_load(handle)
    expected = copy.deepcopy(source)
    expected["name"] = "PACT_CBEA_ORIGINAL_RULE_UNIFORM_v1/rule_uniform"
    expected["model"]["core_method"] = (
        "heter_pyramid_collab_pact_cbea_original_rule_uniform"
    )
    expected["model"]["args"]["pact_cbea_original_rule_uniform"] = {
        "enabled": True,
    }
    assert target == expected, "new YAML must differ only by the isolated method switch"


def _assert_static_isolation():
    with open(MODEL_PATH, "r") as handle:
        source = handle.read()
    assert "PACTCBEAEvidenceGeometryAligner" not in source
    assert "pact_cbea_aligned_uniform" not in source
    assert "_compute_local_evidence(" not in source
    assert "_lookup_optional_evidence(" not in source
    assert "pact_geometry_aligner" not in source


def _assert_only_new_files_changed():
    tracked_output = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=ROOT, universal_newlines=True
    )
    untracked_output = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        universal_newlines=True,
    )
    allowed = {
        "opencood/models/heter_pyramid_collab_pact_cbea_original_rule_uniform.py",
        "opencood/models/sub_modules/pact_cbea_original_rule_uniform.py",
        "opencood/hypes_yaml/PACT_CBEA_ORIGINAL_RULE_UNIFORM_v1/rule_uniform.yaml",
        "opencood/tools/check_pact_cbea_original_rule_uniform_smoke.py",
    }
    changed = set(tracked_output.splitlines())
    changed.update(untracked_output.splitlines())
    assert changed == allowed, "unexpected workspace changes: %s" % sorted(changed)


def main():
    _assert_python38_ast()
    _assert_yaml_contract()
    _assert_static_isolation()

    from opencood.models.heter_pyramid_collab_pact_cbea import (
        HeterPyramidCollabPactCbea,
    )
    from opencood.models.heter_pyramid_collab_pact_cbea_original_rule_uniform import (
        HeterPyramidCollabPactCbeaOriginalRuleUniform,
    )
    from opencood.models.sub_modules.pact_cbea_original_rule_uniform import (
        PACTCBEAOriginalRuleUniform,
    )
    from opencood.tools import train_utils

    for invalid_args in ({}, _args(enabled=False)):
        try:
            HeterPyramidCollabPactCbeaOriginalRuleUniform(invalid_args)
        except ValueError:
            pass
        else:
            raise AssertionError("missing/false enabled must fail")

    router = PACTCBEAOriginalRuleUniform()
    assert router.parameter_count() == 0
    feature = torch.stack([
        torch.full((2, 2, 2), 1.0),
        torch.full((2, 2, 2), 3.0),
        torch.full((2, 2, 2), 6.0),
        torch.full((2, 2, 2), 9.0),
        torch.full((2, 2, 2), 12.0),
    ])
    fused, debug = router(feature, torch.tensor([2, 3]))
    assert torch.allclose(fused[0], torch.full_like(fused[0], 2.0))
    assert torch.allclose(fused[1], torch.full_like(fused[1], 9.0))
    alpha_two = router.uniform_alpha(feature[:2], 2)
    alpha_three = router.uniform_alpha(feature[2:], 3)
    assert torch.allclose(alpha_two, torch.full_like(alpha_two, 0.5))
    assert torch.allclose(alpha_three, torch.full_like(alpha_three, 1.0 / 3.0))
    assert torch.allclose(alpha_two.sum(dim=0), torch.ones_like(alpha_two[0]))
    assert torch.allclose(alpha_three.sum(dim=0), torch.ones_like(alpha_three[0]))
    assert debug["weight_sum_error"] == 0.0

    original_parent_init = HeterPyramidCollabPactCbea.__init__
    HeterPyramidCollabPactCbea.__init__ = _fake_parent_init
    try:
        hypes = {
            "model": {
                "core_method": "heter_pyramid_collab_pact_cbea_original_rule_uniform",
                "args": _args(),
            },
        }
        model = train_utils.create_model(hypes)
        reference = HeterPyramidCollabPactCbea(_args())
        assert set(model.state_dict()) == set(reference.state_dict())

        source_state = model.state_dict()
        source_state["pact_cbea_evidence_head_extra.weight"] = torch.ones(1)
        model.load_state_dict(source_state, strict=False)
        reference.load_state_dict(model.state_dict(), strict=False)
        for key in ("cls_head.scale", "reg_head.scale", "dir_head.scale"):
            assert torch.equal(model.state_dict()[key], reference.state_dict()[key])
        assert model.core_checkpoint_verified is True
        assert model.core_checkpoint_report["auxiliary_extra_key_count"] == 1
        assert sum(parameter.numel() for parameter in model.parameters()
                   if parameter.requires_grad) == 0
        assert all(not module.training for module in model.modules()
                   if isinstance(module, nn.modules.batchnorm._BatchNorm))

        output = model(_data())
        debug = output["pact_cbea"]
        assert model.pyramid_backbone.forward_collab_calls == 1
        assert debug["final_fusion_source"] == "original_rule_uniform"
        assert debug["forward_collab_executed"] is True
        assert debug["forward_collab_output_used"] is False
        assert debug["evidence_used"] is False
        assert debug["external_evidence_read"] is False
        assert debug["uncertainty_used"] is False
        assert debug["descriptor_used"] is False
        assert debug["modality_prior_used"] is False
        assert debug["aligned_uniform_geometry_used"] is False
        assert debug["per_scene_agent_count"] == [2, 3]
        assert abs(debug["uniform_weight_min"] - (1.0 / 3.0)) < 1e-6
        assert abs(debug["uniform_weight_max"] - 0.5) < 1e-6
        assert debug["weight_sum_error"] == 0.0
        assert torch.allclose(model.cls_head.inputs[-1][0],
                              torch.full_like(model.cls_head.inputs[-1][0], 1.5))
        assert torch.allclose(model.cls_head.inputs[-1][1],
                              torch.full_like(model.cls_head.inputs[-1][1], 4.0))
        assert not torch.allclose(model.cls_head.inputs[-1][0],
                                  torch.full_like(model.cls_head.inputs[-1][0], 148.5))
        for key in ("cls_preds", "reg_preds", "dir_preds"):
            assert key in output and output[key].requires_grad is False
        assert "occ_single_list" in output and "pact_cbea" in output

        output_with_external = model(_data(torch.full((5, 1, 4, 4), 99.0)))
        assert torch.equal(output["cls_preds"], output_with_external["cls_preds"])
        assert torch.equal(output["reg_preds"], output_with_external["reg_preds"])
    finally:
        HeterPyramidCollabPactCbea.__init__ = original_parent_init

    _assert_only_new_files_changed()
    print("PACT_CBEA_ORIGINAL_RULE_UNIFORM_SMOKE_PASS")


if __name__ == "__main__":
    main()

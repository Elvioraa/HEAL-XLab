"""CPU smoke for the isolated PACT-CBEA B/D geometry ablation."""

from __future__ import absolute_import, division, print_function

import ast
import copy
import os
import subprocess
import sys
import types

import torch
import torch.nn as nn
import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _install_stubs():
    if "icecream" not in sys.modules:
        module = types.ModuleType("icecream")
        module.ic = lambda *args, **kwargs: args[0] if args else None
        sys.modules["icecream"] = module
    if "pyquaternion" not in sys.modules:
        module = types.ModuleType("pyquaternion")
        module.Quaternion = object
        sys.modules["pyquaternion"] = module
    if "shapely" not in sys.modules:
        shapely = types.ModuleType("shapely")
        geometry = types.ModuleType("shapely.geometry")
        geometry.Polygon = object
        geometry.Point = object
        geometry.MultiPoint = object
        shapely.geometry = geometry
        sys.modules["shapely"] = shapely
        sys.modules["shapely.geometry"] = geometry
    if "timm" not in sys.modules:
        timm = types.ModuleType("timm")
        models = types.ModuleType("timm.models")
        layers = types.ModuleType("timm.models.layers")
        layers.DropPath = nn.Identity
        timm.models = models
        models.layers = layers
        sys.modules["timm"] = timm
        sys.modules["timm.models"] = models
        sys.modules["timm.models.layers"] = layers
    if "einops" not in sys.modules:
        module = types.ModuleType("einops")
        module.rearrange = lambda value, *args, **kwargs: value
        module.repeat = lambda value, *args, **kwargs: value
        sys.modules["einops"] = module


_install_stubs()

from opencood.models.heter_pyramid_collab_pact_cbea_aligned_uniform import (
    HeterPyramidCollabPactCbeaAlignedUniform,
)
from opencood.models.heter_pyramid_collab_pact_cbea_geometry_ablation import (
    HeterPyramidCollabPactCbeaGeometryAblation,
)
from opencood.models.sub_modules.pact_cbea_geometry_ablation import (
    PACTCBEAGeometryAblationAligner,
)
from opencood.tools import train_utils


SOURCE_YAML = os.path.join(
    ROOT, "opencood", "hypes_yaml", "PACT_CBEA_GEOMETRY_ABLATION_v1",
    "rule_source_ego.yaml"
)
IDENTITY_YAML = os.path.join(
    ROOT, "opencood", "hypes_yaml", "PACT_CBEA_GEOMETRY_ABLATION_v1",
    "rule_identity.yaml"
)
ALIGNED_YAML = os.path.join(
    ROOT, "opencood", "hypes_yaml", "PACT_CBEA_ALIGNED_UNIFORM_v1",
    "rule_aligned_uniform.yaml"
)
MODEL_PATH = os.path.join(
    ROOT, "opencood", "models", "heter_pyramid_collab_pact_cbea_geometry_ablation.py"
)
SUBMODULE_PATH = os.path.join(
    ROOT, "opencood", "models", "sub_modules", "pact_cbea_geometry_ablation.py"
)


class _Encoder(nn.Module):
    def __init__(self):
        super(_Encoder, self).__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.bn = nn.BatchNorm2d(2)

    def forward(self, data, modality):
        return self.bn(data["inputs_%s" % modality]["feature"]) * self.weight


class _Backbone(nn.Module):
    def __init__(self):
        super(_Backbone, self).__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, data):
        return {"spatial_features_2d": data["spatial_features"] * self.weight}


class _Aligner(nn.Module):
    def __init__(self):
        super(_Aligner, self).__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, feature):
        return feature * self.weight


class _Pyramid(nn.Module):
    def __init__(self):
        super(_Pyramid, self).__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.align_corners = False
        self.forward_single_calls = 0

    def forward_single(self, feature):
        self.forward_single_calls += 1
        return feature * self.weight, [feature[:, :1]]

    def forward_collab(self, *args, **kwargs):
        raise AssertionError("geometry ablation must not call forward_collab")


def _lightweight_base_init(self, args):
    nn.Module.__init__(self)
    self.cav_range = list(args["lidar_range"])
    self.modality_name_list = ["m1", "m2", "m3", "m4"]
    self.sensor_type_dict = dict((name, "lidar") for name in self.modality_name_list)
    self.H = self.cav_range[4] - self.cav_range[1]
    self.W = self.cav_range[3] - self.cav_range[0]
    self.fake_voxel_size = 1
    self.compress = False
    self.shrink_flag = False
    for name in self.modality_name_list:
        setattr(self, "encoder_%s" % name, _Encoder())
        setattr(self, "backbone_%s" % name, _Backbone())
        setattr(self, "aligner_%s" % name, _Aligner())
        setattr(self, "depth_supervision_%s" % name, False)
    self.pyramid_backbone = _Pyramid()
    self.cls_head = nn.Conv2d(2, 2, 1)
    self.reg_head = nn.Conv2d(2, 14, 1)
    self.dir_head = nn.Conv2d(2, 4, 1)


def _load_yaml(path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)


def _load_hypes():
    source = _load_yaml(SOURCE_YAML)
    identity = _load_yaml(IDENTITY_YAML)
    aligned = _load_yaml(ALIGNED_YAML)
    for hypes, convention in ((source, "B_source_ego"), (identity, "D_identity")):
        assert hypes["model"]["core_method"] == "heter_pyramid_collab_pact_cbea_geometry_ablation"
        cfg = hypes["model"]["args"]["pact_cbea_geometry_ablation"]
        assert cfg == {"enabled": True, "convention": convention}
        comparable = copy.deepcopy(hypes)
        comparable["name"] = "CONTROL"
        comparable["model"]["core_method"] = "CONTROL"
        comparable["model"]["args"].pop("pact_cbea_geometry_ablation")
        expected = copy.deepcopy(aligned)
        expected["name"] = "CONTROL"
        expected["model"]["core_method"] = "CONTROL"
        expected["model"]["args"].pop("pact_cbea_aligned_uniform")
        assert comparable == expected
    return source, identity


def _build_model(hypes):
    base = HeterPyramidCollabPactCbeaGeometryAblation.__mro__[2]
    original_init = base.__init__
    base.__init__ = _lightweight_base_init
    try:
        model = train_utils.create_model(copy.deepcopy(hypes))
    finally:
        base.__init__ = original_init
    assert isinstance(model, HeterPyramidCollabPactCbeaGeometryAblation)
    return model


def _identity_affine(batch, agents):
    affine = torch.zeros(batch, agents, agents, 2, 3)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    return affine


def _align(aligner, feature, record_len, affine):
    scratch = feature.new_zeros((feature.shape[0], 1, feature.shape[2], feature.shape[3]))
    return aligner(feature, scratch, torch.ones_like(scratch), scratch, record_len, affine)


def _check_conventions():
    affine = _identity_affine(1, 2)
    # Deliberately non-symmetric source/ego pair; B must select [1, 0].
    affine[0, 0, 1, 0, 2] = -0.4
    affine[0, 1, 0, 0, 2] = 0.7
    affine[0, 1, 0, 0, 0] = 0.0
    affine[0, 1, 0, 0, 1] = -1.0
    affine[0, 1, 0, 1, 0] = 1.0
    affine[0, 1, 0, 1, 1] = 0.0
    feature = torch.cat((
        torch.ones(1, 2, 7, 7),
        torch.full((1, 2, 7, 7), 3.0),
    ), dim=0)
    b_aligner = PACTCBEAGeometryAblationAligner("B_source_ego")
    d_aligner = PACTCBEAGeometryAblationAligner("D_identity")
    b_result = _align(b_aligner, feature, [2], affine)
    d_result = _align(d_aligner, feature, [2], affine)
    selected_b = b_aligner.last_debug["normalized_affine"][0]
    expected_b = affine[0, 1, 0].tolist()
    assert selected_b[1] == expected_b
    assert selected_b[1] != affine[0, 0, 1].tolist()
    assert b_aligner.last_debug["selected_pairwise_indices"] == [[[0, 0], [1, 0]]]
    for matrix in d_aligner.last_debug["normalized_affine"][0]:
        assert matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert torch.allclose(d_result["feature"], feature)
    assert torch.allclose(d_result["validity"], torch.ones_like(d_result["validity"]))
    from opencood.models.sub_modules.pact_cbea_aligned_uniform import PACTCBEAAlignedUniformRouter
    d_fused, unused_debug = PACTCBEAAlignedUniformRouter()(
        d_result["feature"], d_result["validity"], [2]
    )
    assert torch.allclose(d_fused, torch.full_like(d_fused, 2.0))
    assert b_result["feature"].shape == feature.shape
    assert b_result["feature"].dtype == feature.dtype
    assert b_result["feature"].device == feature.device
    assert b_aligner.parameter_count() == 0
    assert sum(parameter.numel() for parameter in b_aligner.parameters()) == 0
    print("B source->ego selection, B/A difference, D identity and dtype/device OK")


def _check_router_participation_and_records():
    from opencood.models.sub_modules.pact_cbea_aligned_uniform import PACTCBEAAlignedUniformRouter
    router = PACTCBEAAlignedUniformRouter()
    features = torch.cat((torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 3.0)), dim=0)
    fused, debug = router(features, torch.ones(2, 1, 2, 2), [2])
    assert torch.allclose(fused, torch.full_like(fused, 2.0))
    assert debug["per_scene_agent_count"] == [2]
    multi = torch.cat((features, torch.full((3, 1, 2, 2), 9.0)), dim=0)
    fused, debug = router(multi, torch.ones(5, 1, 2, 2), [2, 3])
    assert torch.allclose(fused[0], torch.full_like(fused[0], 2.0))
    assert torch.allclose(fused[1], torch.full_like(fused[1], 9.0))
    print("D keeps multiple sources in uniform aggregation and record_len isolation OK")


def _check_model_and_checkpoint(hypes):
    model = _build_model(hypes)
    assert model._core_module_names() == HeterPyramidCollabPactCbeaAlignedUniform._core_module_names(model)
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 0
    assert model.pact_geometry_aligner.parameter_count() == 0
    assert not any(parameter.requires_grad for parameter in model.parameters())
    state = dict((key, value.clone()) for key, value in model.state_dict().items())
    state["pact_cbea_evidence_head_m1.synthetic"] = torch.ones(1)
    model.load_state_dict(state, strict=False)
    assert model.core_checkpoint_verified is True
    assert model.core_checkpoint_report["auxiliary_extra_key_count"] == 1
    print("create_model discovery, frozen parameters and aligned-uniform checkpoint contract OK")


def _check_guards(hypes):
    variants = []
    missing = copy.deepcopy(hypes["model"]["args"])
    missing.pop("pact_cbea_geometry_ablation")
    variants.append(missing)
    disabled = copy.deepcopy(hypes["model"]["args"])
    disabled["pact_cbea_geometry_ablation"]["enabled"] = False
    variants.append(disabled)
    unknown = copy.deepcopy(hypes["model"]["args"])
    unknown["pact_cbea_geometry_ablation"]["convention"] = "A_ego_source"
    variants.append(unknown)
    for args in variants:
        try:
            HeterPyramidCollabPactCbeaGeometryAblation(args)
        except ValueError:
            continue
        raise AssertionError("invalid ablation config was accepted")
    print("missing/disabled/unknown convention guards OK")


def _check_static_isolation():
    for path in (MODEL_PATH, SUBMODULE_PATH, os.path.abspath(__file__)):
        source = open(path, "r", encoding="utf-8").read()
        ast.parse(source, filename=path, feature_version=8)
        if path != os.path.abspath(__file__):
            assert "forward_collab(" not in source
            assert "A_ego_source" not in source
    changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT,
                                      universal_newlines=True).splitlines()
    forbidden = [
        "heter_pyramid_collab_pact_cbea_aligned_uniform.py",
        "pact_cbea_aligned_uniform.py",
    ]
    assert not any(path.endswith(item) for path in changed for item in forbidden)
    print("Python 3.8 AST and aligned-uniform isolation OK")


def main():
    source_hypes, identity_hypes = _load_hypes()
    _check_conventions()
    _check_router_participation_and_records()
    _check_guards(source_hypes)
    _check_model_and_checkpoint(source_hypes)
    _check_model_and_checkpoint(identity_hypes)
    _check_static_isolation()
    print("PACT_CBEA_GEOMETRY_ABLATION_SMOKE_PASS")


if __name__ == "__main__":
    main()

"""CPU smoke for the isolated aligned-uniform PACT-CBEA control."""

from __future__ import absolute_import, division, print_function

import ast
import contextlib
import copy
import importlib.machinery
import io
import os
import subprocess
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO_ROOT_ADDED = REPO_ROOT not in sys.path
if REPO_ROOT_ADDED:
    sys.path.insert(0, REPO_ROOT)

MODULES_BEFORE_SMOKE = dict(sys.modules)
YAML_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "PACT_CBEA_ALIGNED_UNIFORM_v1",
    "rule_aligned_uniform.yaml",
)
ROUTED_YAML_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "PACT_CBEA_EVIDENCE_ROUTED_v2",
    "rule_evidence_routed.yaml",
)
MODEL_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "models",
    "heter_pyramid_collab_pact_cbea_aligned_uniform.py",
)
ROUTER_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "models",
    "sub_modules",
    "pact_cbea_aligned_uniform.py",
)


def _set_smoke_module(name, module):
    sys.modules[name] = module


def _install_optional_dependency_stubs():
    """Install temporary stubs only for this smoke-test process."""
    try:
        from icecream import ic as unused_ic  # noqa: F401
    except Exception:
        module = types.ModuleType("icecream")
        module.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
        _set_smoke_module("icecream", module)

    try:
        from shapely.geometry import (  # noqa: F401
            MultiPoint as unused_multi_point,
            Point as unused_point,
            Polygon as unused_polygon,
        )
    except Exception:
        shapely_module = types.ModuleType("shapely")
        geometry_module = types.ModuleType("shapely.geometry")

        class _AreaShape(object):
            def __init__(self, area):
                self.area = float(max(area, 0.0))

        class _Polygon(object):
            def __init__(self, points):
                points = np.asarray(points, dtype=np.float64)
                self.bounds = (
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    float(points[:, 0].max()),
                    float(points[:, 1].max()),
                )
                self.area = max(
                    (self.bounds[2] - self.bounds[0])
                    * (self.bounds[3] - self.bounds[1]),
                    0.0,
                )

            def intersection(self, other):
                left = max(self.bounds[0], other.bounds[0])
                bottom = max(self.bounds[1], other.bounds[1])
                right = min(self.bounds[2], other.bounds[2])
                top = min(self.bounds[3], other.bounds[3])
                return _AreaShape(
                    max(right - left, 0.0) * max(top - bottom, 0.0)
                )

            def union(self, other):
                intersection = self.intersection(other).area
                return _AreaShape(self.area + other.area - intersection)

        class _Point(object):
            def __init__(self, *coordinates):
                self.coordinates = coordinates

        class _MultiPoint(object):
            def __init__(self, points):
                self.points = points

        geometry_module.Polygon = _Polygon
        geometry_module.Point = _Point
        geometry_module.MultiPoint = _MultiPoint
        shapely_module.geometry = geometry_module
        _set_smoke_module("shapely", shapely_module)
        _set_smoke_module("shapely.geometry", geometry_module)

    try:
        from pyquaternion import Quaternion as unused_quaternion  # noqa: F401
    except Exception:
        module = types.ModuleType("pyquaternion")

        class _Quaternion(object):
            def __init__(self, *args, **kwargs):
                pass

            @property
            def transformation_matrix(self):
                return np.eye(4, dtype=np.float32)

        module.Quaternion = _Quaternion
        _set_smoke_module("pyquaternion", module)

    try:
        from timm.models.layers import DropPath as unused_drop_path  # noqa: F401
    except Exception:
        timm_module = types.ModuleType("timm")
        models_module = types.ModuleType("timm.models")
        layers_module = types.ModuleType("timm.models.layers")

        class _DropPath(nn.Identity):
            pass

        layers_module.DropPath = _DropPath
        timm_module.models = models_module
        models_module.layers = layers_module
        _set_smoke_module("timm", timm_module)
        _set_smoke_module("timm.models", models_module)
        _set_smoke_module("timm.models.layers", layers_module)

    try:
        from einops import rearrange as unused_rearrange  # noqa: F401
    except Exception:
        module = types.ModuleType("einops")

        def _missing_einops(*args, **kwargs):
            raise ImportError("einops is required by this encoder")

        module.rearrange = _missing_einops
        module.repeat = _missing_einops
        _set_smoke_module("einops", module)

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from opencood.utils.box_overlaps import (  # noqa: F401
                bbox_overlaps as unused_overlaps,
            )
    except Exception:
        module = types.ModuleType("opencood.utils.box_overlaps")
        module.__spec__ = importlib.machinery.ModuleSpec(
            "opencood.utils.box_overlaps", loader=None
        )

        def _missing_bbox_overlaps(*args, **kwargs):
            raise ImportError("compiled box_overlaps is required for this operation")

        module.bbox_overlaps = _missing_bbox_overlaps
        _set_smoke_module("opencood.utils.box_overlaps", module)


def _restore_smoke_modules():
    for name in list(sys.modules):
        if name not in MODULES_BEFORE_SMOKE:
            del sys.modules[name]
    for name, module in MODULES_BEFORE_SMOKE.items():
        sys.modules[name] = module
    if REPO_ROOT_ADDED and REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)


_install_optional_dependency_stubs()

try:
    from opencood.models.heter_pyramid_collab_pact_cbea_aligned_uniform import (
        HeterPyramidCollabPactCbeaAlignedUniform,
    )
    from opencood.models.sub_modules.pact_cbea_aligned_uniform import (
        PACTCBEAAlignedUniformRouter,
    )
    from opencood.models.sub_modules.pact_cbea_evidence_routed import (
        PACTCBEAEvidenceGeometryAligner,
    )
    from opencood.tools import train_utils
except Exception:
    _restore_smoke_modules()
    raise


class _DummyEncoder(nn.Module):
    def __init__(self, channels):
        super(_DummyEncoder, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, data_dict, modality_name):
        feature = data_dict["inputs_%s" % modality_name]["feature"]
        return self.bn(feature) * self.scale


class _DummyBackbone(nn.Module):
    def __init__(self):
        super(_DummyBackbone, self).__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, batch_dict):
        return {"spatial_features_2d": batch_dict["spatial_features"] * self.scale}


class _DummyAligner(nn.Module):
    def __init__(self):
        super(_DummyAligner, self).__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, feature):
        return feature * self.scale


class _DummyPyramid(nn.Module):
    def __init__(self):
        super(_DummyPyramid, self).__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.align_corners = False
        self.forward_single_calls = 0

    def forward_single(self, feature):
        self.forward_single_calls += 1
        feature = feature * self.scale
        return feature, [feature[:, :1]]

    def forward_collab(self, *args, **kwargs):
        raise AssertionError("aligned-uniform model must not call forward_collab")


def _lightweight_base_init(self, args):
    nn.Module.__init__(self)
    self.args = args
    self.cav_range = list(args["lidar_range"])
    self.modality_name_list = ["m1", "m2", "m3", "m4"]
    self.sensor_type_dict = {
        name: "lidar" for name in self.modality_name_list
    }
    self.cam_crop_info = {}
    self.H = self.cav_range[4] - self.cav_range[1]
    self.W = self.cav_range[3] - self.cav_range[0]
    self.fake_voxel_size = 1
    self.compress = False
    self.shrink_flag = False
    for name in self.modality_name_list:
        setattr(self, "encoder_%s" % name, _DummyEncoder(256))
        setattr(self, "backbone_%s" % name, _DummyBackbone())
        setattr(self, "aligner_%s" % name, _DummyAligner())
        setattr(self, "depth_supervision_%s" % name, False)
    self.pyramid_backbone = _DummyPyramid()
    self.cls_head = nn.Conv2d(256, 2, kernel_size=1)
    self.reg_head = nn.Conv2d(256, 14, kernel_size=1)
    self.dir_head = nn.Conv2d(256, 4, kernel_size=1)


def _load_hypes():
    with open(YAML_PATH, "r") as stream:
        hypes = yaml.safe_load(stream)
    with open(ROUTED_YAML_PATH, "r") as stream:
        routed_hypes = yaml.safe_load(stream)
    assert hypes["name"] == "PACT_CBEA_ALIGNED_UNIFORM_v1"
    assert hypes["model"]["core_method"] == (
        "heter_pyramid_collab_pact_cbea_aligned_uniform"
    )
    cfg = hypes["model"]["args"]["pact_cbea_aligned_uniform"]
    assert cfg["enabled"] is True
    assert cfg["uniform_over_valid_support"] is True
    assert cfg["packet_only"] is False

    comparable = copy.deepcopy(hypes)
    routed_comparable = copy.deepcopy(routed_hypes)
    comparable["name"] = "CONTROL"
    routed_comparable["name"] = "CONTROL"
    comparable["model"]["core_method"] = "CONTROL"
    routed_comparable["model"]["core_method"] = "CONTROL"
    comparable["model"]["args"].pop("pact_cbea_aligned_uniform")
    routed_comparable["model"]["args"].pop("pact_cbea_evidence_routed")
    assert comparable == routed_comparable
    print("YAML load and exact v2 perception/config parity OK")
    return hypes


def _build_model(hypes):
    base_class = HeterPyramidCollabPactCbeaAlignedUniform.__mro__[1]
    original_init = base_class.__init__
    base_class.__init__ = _lightweight_base_init
    try:
        model = train_utils.create_model(copy.deepcopy(hypes))
    finally:
        base_class.__init__ = original_init
    assert isinstance(model, HeterPyramidCollabPactCbeaAlignedUniform)
    return model


def _check_discovery_guard_and_freeze(hypes):
    model = _build_model(hypes)
    missing = copy.deepcopy(hypes["model"]["args"])
    missing.pop("pact_cbea_aligned_uniform")
    try:
        HeterPyramidCollabPactCbeaAlignedUniform(missing)
    except ValueError:
        pass
    else:
        raise AssertionError("missing enabled config must fail")

    disabled = copy.deepcopy(hypes["model"]["args"])
    disabled["pact_cbea_aligned_uniform"]["enabled"] = False
    try:
        HeterPyramidCollabPactCbeaAlignedUniform(disabled)
    except ValueError:
        pass
    else:
        raise AssertionError("enabled=false must fail")

    assert model.pact_geometry_aligner.__class__ is PACTCBEAEvidenceGeometryAligner
    assert isinstance(model.pact_aligned_uniform_router, PACTCBEAAlignedUniformRouter)
    assert model.pact_aligned_uniform_router.parameter_count() == 0
    assert sum(
        parameter.numel()
        for parameter in model.pact_aligned_uniform_router.parameters()
    ) == 0
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 0
    assert not any(
        name.startswith("pact_cbea_evidence_head_")
        for name, unused_module in model.named_modules()
    )
    model.train()
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(
        not module.training
        for module in model.modules()
        if isinstance(module, HeterPyramidCollabPactCbeaAlignedUniform.BN_TYPES)
    )
    print("create_model discovery, config guard, shared geometry and full freeze OK")
    return model


def _identity_affine(batch_size, max_agents):
    affine = torch.zeros(batch_size, max_agents, max_agents, 2, 3)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    return affine


def _align_feature(aligner, feature, record_len, affine):
    scratch = feature.new_zeros(
        (feature.shape[0], 1, feature.shape[2], feature.shape[3])
    )
    positive = torch.ones_like(scratch)
    return aligner(feature, scratch, positive, scratch, record_len, affine)


def _check_shared_geometry():
    aligner = PACTCBEAEvidenceGeometryAligner(align_corners=False)
    feature = torch.randn(2, 3, 5, 5)
    identity = _identity_affine(1, 2)
    aligned = _align_feature(aligner, feature, [2], identity)
    assert torch.allclose(aligned["feature"], feature, atol=1e-5, rtol=1e-5)
    assert torch.allclose(aligned["validity"], torch.ones_like(aligned["validity"]))

    ones_feature = torch.ones(2, 1, 5, 5)
    translated = _identity_affine(1, 2)
    translated[0, 0, 1, 0, 2] = 0.8
    shifted = _align_feature(aligner, ones_feature, [2], translated)
    shifted_feature_support = shifted["feature"][1, 0] > 1e-7
    shifted_valid_support = shifted["validity"][1, 0] > 1e-7
    assert torch.equal(shifted_feature_support, shifted_valid_support)
    assert not shifted_valid_support.all().item()
    invalid = ~shifted_valid_support
    assert shifted["feature"][1, 0].masked_select(invalid).abs().max().item() == 0.0
    assert shifted["validity"][1, 0].masked_select(invalid).abs().max().item() == 0.0
    print("identity/translation shared v2 geometry and validity support OK")


def _check_router_math():
    router = PACTCBEAAlignedUniformRouter()
    assert router.parameter_count() == 0
    assert sum(parameter.numel() for parameter in router.parameters()) == 0

    one_valid = torch.ones(1, 1, 2, 2)
    one_alpha = router._normalized_alpha(one_valid)
    assert torch.equal(one_alpha, torch.ones_like(one_alpha))
    one_feature = torch.randn(1, 2, 2, 2, requires_grad=True)
    one_fused, unused_debug = router(one_feature, one_valid, [1])
    assert torch.allclose(one_fused, one_feature.detach())
    assert one_fused.requires_grad is False

    two_valid = torch.ones(2, 1, 2, 2)
    two_alpha = router._normalized_alpha(two_valid)
    assert torch.allclose(two_alpha, torch.full_like(two_alpha, 0.5))
    three_valid = torch.ones(3, 1, 2, 2)
    three_alpha = router._normalized_alpha(three_valid)
    assert torch.allclose(three_alpha, torch.full_like(three_alpha, 1.0 / 3.0))

    partial = torch.ones(2, 1, 2, 2)
    partial[1, 0, 0, 0] = 0.0
    partial_alpha = router._normalized_alpha(partial)
    assert partial_alpha[0, 0, 0, 0].item() == 1.0
    assert partial_alpha[1, 0, 0, 0].item() == 0.0

    continuous = torch.tensor([1.0, 0.5, 0.25]).view(3, 1, 1, 1)
    continuous_alpha = router._normalized_alpha(continuous)
    expected = continuous / continuous.sum(dim=0, keepdim=True)
    assert torch.allclose(continuous_alpha, expected)
    assert torch.allclose(
        continuous_alpha.sum(dim=0),
        torch.ones_like(continuous_alpha[0]),
    )

    multi_feature = torch.cat([
        torch.full((1, 1, 2, 2), 1.0),
        torch.full((1, 1, 2, 2), 3.0),
        torch.full((1, 1, 2, 2), 6.0),
        torch.full((1, 1, 2, 2), 9.0),
        torch.full((1, 1, 2, 2), 12.0),
    ], dim=0)
    multi_validity = torch.ones(5, 1, 2, 2)
    multi_fused, multi_debug = router(
        multi_feature, multi_validity, torch.tensor([2, 3])
    )
    assert torch.allclose(multi_fused[0], torch.full_like(multi_fused[0], 2.0))
    assert torch.allclose(multi_fused[1], torch.full_like(multi_fused[1], 9.0))
    assert multi_debug["per_scene_agent_count"] == [2, 3]
    assert multi_debug["alpha_sum_verified"] is True

    try:
        router(torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), [1])
    except RuntimeError as error:
        assert "no valid agent" in str(error)
    else:
        raise AssertionError("zero-validity routing must fail")
    print("CAV1, 1/2, 1/3, partial, continuous and multi-scene alpha math OK")


def _dummy_data():
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4).repeat(2, 3, 3, 1, 1)
    generator = torch.Generator().manual_seed(89)
    return {
        "agent_modality_list": ["m1", "m2", "m3", "m2", "m4"],
        "record_len": torch.tensor([2, 3]),
        "pairwise_t_matrix": pairwise,
        "inputs_m1": {"feature": torch.randn(1, 256, 5, 5, generator=generator)},
        "inputs_m2": {"feature": torch.randn(2, 256, 5, 5, generator=generator)},
        "inputs_m3": {"feature": torch.randn(1, 256, 5, 5, generator=generator)},
        "inputs_m4": {"feature": torch.randn(1, 256, 5, 5, generator=generator)},
    }


def _core_tensor_key(model):
    prefixes = tuple("%s." % name for name in model._core_module_names())
    for key, value in model.state_dict().items():
        if key.startswith(prefixes) and value.dim() > 0 and value.shape[0] > 1:
            return key
    raise AssertionError("no suitable core checkpoint tensor found")


def _check_checkpoint_validation(hypes, model):
    complete = {key: value.clone() for key, value in model.state_dict().items()}
    for modality in ("m1", "m2", "m3", "m4"):
        complete[
            "pact_cbea_evidence_head_%s.synthetic" % modality
        ] = torch.ones(1)
    complete["legacy.unrelated_history"] = torch.ones(1)
    model.load_state_dict(complete, strict=False)
    assert model.core_checkpoint_verified is True
    assert model.core_checkpoint_report["auxiliary_extra_key_count"] == 4
    assert model.core_checkpoint_report["unknown_extra_key_count"] == 1
    assert model.core_checkpoint_report["unknown_extra_keys"] == [
        "legacy.unrelated_history"
    ]

    missing_model = _build_model(hypes)
    missing_state = {
        key: value.clone() for key, value in missing_model.state_dict().items()
    }
    missing_key = _core_tensor_key(missing_model)
    del missing_state[missing_key]
    try:
        missing_model.load_state_dict(missing_state, strict=False)
    except RuntimeError as error:
        assert "missing=" in str(error)
    else:
        raise AssertionError("missing core key must fail")

    mismatch_model = _build_model(hypes)
    mismatch_state = {
        key: value.clone() for key, value in mismatch_model.state_dict().items()
    }
    mismatch_key = _core_tensor_key(mismatch_model)
    mismatch_state[mismatch_key] = mismatch_state[mismatch_key][:-1]
    try:
        mismatch_model.load_state_dict(mismatch_state, strict=False)
    except RuntimeError as error:
        assert "shape_mismatch=" in str(error)
    else:
        raise AssertionError("core shape mismatch must fail")
    print("strict core checkpoint and permitted auxiliary extras validation OK")


def _check_unverified_forward(hypes):
    model = _build_model(hypes)
    try:
        model(_dummy_data())
    except RuntimeError as error:
        assert "verified core checkpoint" in str(error)
    else:
        raise AssertionError("unverified checkpoint forward must fail")
    print("unverified checkpoint forward rejection OK")


def _assert_compact_debug(value):
    if torch.is_tensor(value):
        raise AssertionError("debug must not retain dense tensors")
    if isinstance(value, dict):
        for item in value.values():
            _assert_compact_debug(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_compact_debug(item)


def _check_full_forward(model):
    output = model(_dummy_data())
    assert output["cls_preds"].shape == (2, 2, 5, 5)
    assert output["reg_preds"].shape == (2, 14, 5, 5)
    assert output["dir_preds"].shape == (2, 4, 5, 5)
    assert len(output["occ_single_list"]) == 1
    assert model.pyramid_backbone.forward_single_calls == 1
    for key in (
            "cls_preds", "reg_preds", "dir_preds",
            "cls_preds_single", "reg_preds_single", "dir_preds_single"):
        assert output[key].requires_grad is False

    debug = output["pact_cbea_aligned_uniform_debug"]
    _assert_compact_debug(debug)
    expected_flags = {
        "enabled": True,
        "dense_feature_collaboration": True,
        "weak_communication_claimed": False,
        "uniform_over_valid_support": True,
        "geometry_alignment_used": True,
        "shared_alignment_grid_used": True,
        "validity_mask_used": True,
        "evidence_used": False,
        "uncertainty_used": False,
        "descriptor_used": False,
        "modality_prior_used": False,
        "forward_collab_used": False,
        "stage3_training_required": False,
        "no_joint_training_verified": True,
        "core_checkpoint_verified": True,
        "alpha_sum_verified": True,
    }
    for key, expected in expected_flags.items():
        assert debug[key] is expected
    assert debug["routing_mode"] == "aligned_uniform"
    assert debug["trainable_total"] == 0
    assert debug["router_parameter_count"] == 0
    assert debug["per_scene_agent_count"] == [2, 3]
    assert len(debug["per_agent_valid_ratio"]) == 5
    assert debug["fallbacks"] == []
    print("full multi-scene forward, official head shapes and detached outputs OK")


def _check_static_isolation():
    with open(MODEL_PATH, "r") as stream:
        model_source = stream.read()
    with open(ROUTER_PATH, "r") as stream:
        router_source = stream.read()
    for path, source in ((MODEL_PATH, model_source), (ROUTER_PATH, router_source)):
        tree = ast.parse(source, filename=path, feature_version=(3, 8))
        unions = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
        ]
        assert not unions

    forbidden_production = (
        "import sys",
        "import types",
        "sys.modules",
        "_install_model_import_stubs",
        "evidence_heatmap",
        "evidence_uncertainty",
        "evidence_descriptor",
        "pact_cbea_rule",
        "_lookup_optional_evidence",
    )
    for token in forbidden_production:
        assert token not in model_source
        assert token not in router_source
    assert ".forward_collab(" not in model_source
    assert "nn.Parameter" not in router_source
    assert "nn.Conv" not in router_source
    assert "nn.Linear" not in router_source
    assert "BatchNorm" not in router_source

    protected = [
        "opencood/models/heter_pyramid_collab_pact_cbea.py",
        "opencood/models/heter_pyramid_collab_pact_cbea_evidence_routed.py",
        "opencood/models/sub_modules/pact_cbea_evidence_routed.py",
        "opencood/hypes_yaml/PACT_CBEA_EVIDENCE_ROUTED_v2/rule_evidence_routed.yaml",
        "opencood/models/heter_pyramid_collab_pact_cbea_packet_nojoint.py",
        "opencood/models/heter_pyramid_collab_pact_cbea_box_packet_nojoint.py",
        "opencood/tools/inference_utils.py",
    ]
    for path in protected:
        result = subprocess.run(
            ["git", "diff", "--quiet", "--", path],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, "protected file changed: %s" % path

    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=REPO_ROOT, universal_newlines=True
    )
    allowed = {
        "opencood/models/heter_pyramid_collab_pact_cbea_aligned_uniform.py",
        "opencood/models/sub_modules/pact_cbea_aligned_uniform.py",
        "opencood/hypes_yaml/PACT_CBEA_ALIGNED_UNIFORM_v1/",
        "opencood/hypes_yaml/PACT_CBEA_ALIGNED_UNIFORM_v1/rule_aligned_uniform.yaml",
        "opencood/tools/check_pact_cbea_aligned_uniform_smoke.py",
    }
    for line in status.splitlines():
        assert line[3:].replace("\\", "/") in allowed, (
            "unexpected workspace change: %s" % line
        )
    print("Python 3.8 AST, no evidence/stubs/trainable router, and isolation OK")


def main():
    torch.manual_seed(43)
    hypes = _load_hypes()
    model = _check_discovery_guard_and_freeze(hypes)
    _check_shared_geometry()
    _check_router_math()
    _check_unverified_forward(hypes)
    _check_checkpoint_validation(hypes, model)
    _check_full_forward(model)
    _check_static_isolation()
    print("PACT_CBEA_ALIGNED_UNIFORM_SMOKE_PASS")


if __name__ == "__main__":
    try:
        main()
    finally:
        _restore_smoke_modules()

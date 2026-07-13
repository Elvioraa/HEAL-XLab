"""CPU smoke for frozen evidence-routed PACT-CBEA dense collaboration."""

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
    "PACT_CBEA_EVIDENCE_ROUTED_v2",
    "rule_evidence_routed.yaml",
)
MODEL_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "models",
    "heter_pyramid_collab_pact_cbea_evidence_routed.py",
)
SUBMODULE_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "models",
    "sub_modules",
    "pact_cbea_evidence_routed.py",
)


def _set_smoke_module(name, module):
    sys.modules[name] = module


def _install_optional_dependency_stubs():
    """Install process-local stubs before importing the production model."""
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
    from opencood.models.heter_pyramid_collab_pact_cbea_evidence_routed import (
        HeterPyramidCollabPactCbeaEvidenceRouted,
    )
    from opencood.models.sub_modules.pact_cbea_evidence_routed import (
        PACTCBEAEvidenceGeometryAligner,
        PACTCBEAEvidenceRoutingValidator,
    )
    from opencood.models.sub_modules.pact_cbea_rule import PACTCBEARule
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
        raise AssertionError("evidence-routed model must not call forward_collab")


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
    assert hypes["name"] == "PACT_CBEA_EVIDENCE_ROUTED_v2"
    assert hypes["model"]["core_method"] == (
        "heter_pyramid_collab_pact_cbea_evidence_routed"
    )
    cfg = hypes["model"]["args"]["pact_cbea_evidence_routed"]
    assert cfg["enabled"] is True
    assert cfg["strict_evidence_routing"] is True
    assert cfg["aggregation"]["descriptor_consistency"] is False
    print("YAML load and isolated config guard fields OK")
    return hypes


def _build_model(hypes):
    base_class = HeterPyramidCollabPactCbeaEvidenceRouted.__mro__[1]
    original_init = base_class.__init__
    base_class.__init__ = _lightweight_base_init
    try:
        model = train_utils.create_model(copy.deepcopy(hypes))
    finally:
        base_class.__init__ = original_init
    assert isinstance(model, HeterPyramidCollabPactCbeaEvidenceRouted)
    return model


def _check_discovery_guard_and_parameters(hypes):
    model = _build_model(hypes)
    args = copy.deepcopy(hypes["model"]["args"])
    args.pop("pact_cbea_evidence_routed")
    try:
        HeterPyramidCollabPactCbeaEvidenceRouted(args)
    except ValueError:
        pass
    else:
        raise AssertionError("missing enabled config must fail")

    args = copy.deepcopy(hypes["model"]["args"])
    args["pact_cbea_evidence_routed"]["enabled"] = False
    try:
        HeterPyramidCollabPactCbeaEvidenceRouted(args)
    except ValueError:
        pass
    else:
        raise AssertionError("enabled=false must fail")

    expected_heads = [
        "pact_cbea_evidence_head_%s" % name for name in ("m1", "m2", "m3", "m4")
    ]
    assert all(hasattr(model, name) for name in expected_heads)
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 0
    assert model.pact_evidence_geometry_aligner.parameter_count() == 0
    assert model.pact_evidence_routing_validator.parameter_count() == 0
    assert sum(parameter.numel() for parameter in model.pact_cbea_rule.parameters()) == 0
    assert sum(parameter.numel() for parameter in model.pact_evidence_geometry_aligner.parameters()) == 0
    assert sum(parameter.numel() for parameter in model.pact_evidence_routing_validator.parameters()) == 0
    model.train()
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(
        not module.training
        for module in model.modules()
        if isinstance(module, HeterPyramidCollabPactCbeaEvidenceRouted.BN_TYPES)
    )
    print("create_model discovery, config guard, zero trainable parameters and BN eval OK")
    return model


def _check_evidence_heads(model):
    feature = torch.randn(2, 256, 5, 5)
    for modality in ("m1", "m2", "m3", "m4"):
        head = getattr(model, "pact_cbea_evidence_head_%s" % modality)
        output = head(feature)
        assert output["evidence_heatmap_logits"].shape == (2, 1, 5, 5)
        assert output["evidence_uncertainty"].shape == (2, 1, 5, 5)
        assert output["evidence_descriptor"].shape == (2, 16, 5, 5)
        assert (output["evidence_uncertainty"] > 0).all().item()
        norms = torch.linalg.vector_norm(output["evidence_descriptor"], dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)
        assert "evidence_feature" not in output
        assert head.training is False
    print("four real evidence heads, positive uncertainty and normalized descriptor OK")


def _evidence_key(model):
    for key, value in model.state_dict().items():
        if key.startswith("pact_cbea_evidence_head_m2.") and value.dim() > 0 and value.shape[0] > 1:
            return key
    raise AssertionError("no suitable evidence tensor key found")


def _check_checkpoint_validation(hypes, model):
    complete = {key: value.clone() for key, value in model.state_dict().items()}
    complete["legacy.unrelated_extra"] = torch.ones(1)
    model.load_state_dict(complete, strict=False)
    assert model.evidence_checkpoint_verified is True
    assert model.evidence_checkpoint_report["verified_modalities"] == ["m1", "m2", "m3", "m4"]
    assert model.evidence_checkpoint_report["unrelated_checkpoint_key_count"] == 1

    missing_model = _build_model(hypes)
    missing = {key: value.clone() for key, value in missing_model.state_dict().items()}
    missing_key = next(
        key for key in missing if key.startswith("pact_cbea_evidence_head_m3.")
    )
    del missing[missing_key]
    try:
        missing_model.load_state_dict(missing, strict=False)
    except RuntimeError as error:
        assert "missing=" in str(error)
    else:
        raise AssertionError("missing evidence checkpoint key must fail")

    mismatch_model = _build_model(hypes)
    mismatch = {key: value.clone() for key, value in mismatch_model.state_dict().items()}
    mismatch_key = _evidence_key(mismatch_model)
    mismatch[mismatch_key] = mismatch[mismatch_key][:-1]
    try:
        mismatch_model.load_state_dict(mismatch, strict=False)
    except RuntimeError as error:
        assert "shape_mismatch=" in str(error)
    else:
        raise AssertionError("evidence checkpoint shape mismatch must fail")
    print("strict four-modality evidence checkpoint validation OK")


def _identity_affine(batch_size, max_agents):
    affine = torch.zeros(batch_size, max_agents, max_agents, 2, 3)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    return affine


def _check_shared_geometry_and_validity():
    aligner = PACTCBEAEvidenceGeometryAligner(align_corners=False)
    validator = PACTCBEAEvidenceRoutingValidator()
    feature = torch.randn(2, 3, 5, 5)
    logits = torch.randn(2, 1, 5, 5)
    uncertainty = torch.rand(2, 1, 5, 5) + 0.2
    descriptor = torch.randn(2, 4, 5, 5)
    identity = _identity_affine(1, 2)
    aligned = aligner(feature, logits, uncertainty, descriptor, [2], identity)
    validator.validate_aligned_outputs(aligned)
    assert torch.allclose(aligned["feature"], feature, atol=1e-5, rtol=1e-5)
    assert torch.allclose(aligned["heatmap_logits"], logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(aligned["uncertainty"], uncertainty, atol=1e-5, rtol=1e-5)
    assert torch.allclose(aligned["descriptor"], descriptor, atol=1e-5, rtol=1e-5)
    assert torch.allclose(aligned["validity"], torch.ones_like(aligned["validity"]))

    marker_feature = torch.zeros(2, 1, 5, 5)
    marker_logits = torch.full((2, 1, 5, 5), -4.0)
    marker_uncertainty = torch.full((2, 1, 5, 5), 4.0)
    marker_descriptor = torch.zeros(2, 1, 5, 5)
    marker_feature[1, 0, 2, 2] = 1.0
    marker_logits[1, 0, 2, 2] = 4.0
    marker_uncertainty[1, 0, 2, 2] = 0.2
    marker_descriptor[1, 0, 2, 2] = 1.0
    translated = _identity_affine(1, 2)
    translated[0, 0, 1, 0, 2] = 0.8
    shifted = aligner(
        marker_feature,
        marker_logits,
        marker_uncertainty,
        marker_descriptor,
        [2],
        translated,
    )
    feature_peak = int(shifted["feature"][1, 0].argmax().item())
    confidence_peak = int(torch.sigmoid(shifted["heatmap_logits"][1, 0]).argmax().item())
    descriptor_peak = int(shifted["descriptor"][1, 0].argmax().item())
    uncertainty_min = int(shifted["uncertainty"][1, 0].argmin().item())
    assert feature_peak == confidence_peak == descriptor_peak == uncertainty_min
    assert feature_peak != 12

    outside = _identity_affine(1, 2)
    outside[0, 0, 1, 0, 2] = 2.0
    invalid = aligner(
        marker_feature,
        marker_logits,
        marker_uncertainty,
        marker_descriptor,
        [2],
        outside,
    )
    invalid_mask = invalid["validity"][1] <= torch.finfo(torch.float32).eps
    assert invalid_mask.any().item()
    assert invalid["feature"][1].masked_select(invalid_mask).abs().max().item() == 0.0
    assert invalid["descriptor"][1].masked_select(invalid_mask).abs().max().item() == 0.0
    assert torch.sigmoid(invalid["heatmap_logits"][1]).masked_select(invalid_mask).max().item() < 1e-6
    assert invalid["uncertainty"][1].masked_select(invalid_mask).min().item() >= 15.0
    print("identity/translation shared grid and invalid-region reliability suppression OK")


def _extract_rule_tensors(debug, key):
    if torch.is_tensor(debug.get(key)):
        return [debug[key]]
    tensors = []
    for item in debug.get("pact_group_debug", []):
        tensors.extend(_extract_rule_tensors(item, key))
    return tensors


def _check_rule_behavior():
    cfg = PACTCBEARule._default_cfg()
    rule = PACTCBEARule(cfg)
    validator = PACTCBEAEvidenceRoutingValidator()
    feature = torch.randn(2, 3, 4, 4)
    logits = torch.cat([
        torch.full((1, 1, 4, 4), 3.0),
        torch.full((1, 1, 4, 4), -3.0),
    ], dim=0)
    uncertainty = torch.cat([
        torch.full((1, 1, 4, 4), 0.1),
        torch.full((1, 1, 4, 4), 3.0),
    ], dim=0)
    descriptor = torch.nn.functional.normalize(
        torch.randn(2, 4, 4, 4), p=2, dim=1
    )
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    _, debug = rule(
        feature,
        evidence_heatmap=logits,
        evidence_uncertainty=uncertainty,
        evidence_descriptor=descriptor,
        record_len=torch.tensor([2]),
        pairwise_t_matrix=pairwise,
        modality_names=["m1", "m2"],
    )
    validator.validate_rule_debug(debug)
    alpha = debug["pact_alpha"]
    assert torch.allclose(alpha.sum(dim=1), torch.ones_like(alpha[:, 0]), atol=1e-4, rtol=1e-4)
    assert (alpha[:, 0] > alpha[:, 1]).all().item()
    assert not torch.allclose(alpha[:, 0], alpha[:, 1])

    _, cav1_debug = rule(
        feature[:1],
        evidence_heatmap=logits[:1],
        evidence_uncertainty=uncertainty[:1],
        evidence_descriptor=descriptor[:1],
        record_len=torch.tensor([1]),
        pairwise_t_matrix=pairwise[:, :1, :1],
        modality_names=["m1"],
    )
    cav1_alpha = cav1_debug["pact_alpha"]
    assert torch.allclose(cav1_alpha, torch.ones_like(cav1_alpha), atol=1e-4, rtol=1e-4)

    _, missing_debug = rule(
        feature,
        evidence_heatmap=None,
        evidence_uncertainty=None,
        record_len=torch.tensor([2]),
        pairwise_t_matrix=pairwise,
        modality_names=["m1", "m2"],
    )
    try:
        validator.validate_rule_debug(missing_debug)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing evidence fallback must be rejected")
    print("single-CAV alpha, normalized multi-agent alpha and nonuniform evidence routing OK")


def _dummy_data():
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4).repeat(2, 3, 3, 1, 1)
    generator = torch.Generator().manual_seed(73)
    return {
        "agent_modality_list": ["m1", "m2", "m3", "m2", "m4"],
        "record_len": torch.tensor([2, 3]),
        "pairwise_t_matrix": pairwise,
        "inputs_m1": {"feature": torch.randn(1, 256, 5, 5, generator=generator)},
        "inputs_m2": {"feature": torch.randn(2, 256, 5, 5, generator=generator)},
        "inputs_m3": {"feature": torch.randn(1, 256, 5, 5, generator=generator)},
        "inputs_m4": {"feature": torch.randn(1, 256, 5, 5, generator=generator)},
        # Deliberately bogus optional evidence must never be consumed.
        "evidence_heatmap": torch.full((5, 1, 5, 5), 99.0),
        "evidence_uncertainty": torch.full((5, 1, 5, 5), -99.0),
    }


def _assert_compact_debug(value):
    if torch.is_tensor(value):
        raise AssertionError("debug must not retain dense tensors")
    if isinstance(value, dict):
        for item in value.values():
            _assert_compact_debug(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_compact_debug(item)


def _check_unverified_forward(hypes):
    model = _build_model(hypes)
    try:
        model(_dummy_data())
    except RuntimeError as error:
        assert "verified" in str(error)
    else:
        raise AssertionError("forward before checkpoint verification must fail")
    print("unverified checkpoint forward rejection OK")


def _check_full_forward(model):
    captured_rule = {}
    head_outputs = []
    handles = []
    for modality in ("m1", "m2", "m3", "m4"):
        head = getattr(model, "pact_cbea_evidence_head_%s" % modality)

        def _capture_head(module, inputs, output, name=modality):
            head_outputs.append((name, output))

        handles.append(head.register_forward_hook(_capture_head))

    original_rule_forward = model.pact_cbea_rule.forward

    def _capture_rule(*args, **kwargs):
        captured_rule["heatmap"] = kwargs["evidence_heatmap"].clone()
        captured_rule["uncertainty"] = kwargs["evidence_uncertainty"].clone()
        captured_rule["descriptor"] = kwargs["evidence_descriptor"].clone()
        return original_rule_forward(*args, **kwargs)

    model.pact_cbea_rule.forward = _capture_rule
    try:
        output = model(_dummy_data())
    finally:
        model.pact_cbea_rule.forward = original_rule_forward
        for handle in handles:
            handle.remove()

    raw_logits = torch.cat([
        item[1]["evidence_heatmap_logits"] for item in head_outputs
    ], dim=0)
    raw_uncertainty = torch.cat([
        item[1]["evidence_uncertainty"] for item in head_outputs
    ], dim=0)
    raw_descriptor = torch.cat([
        item[1]["evidence_descriptor"] for item in head_outputs
    ], dim=0)
    assert torch.allclose(captured_rule["heatmap"], raw_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(captured_rule["uncertainty"], raw_uncertainty, atol=1e-5, rtol=1e-5)
    assert torch.allclose(captured_rule["descriptor"], raw_descriptor, atol=1e-5, rtol=1e-5)
    assert (captured_rule["uncertainty"] > 0).all().item()
    assert captured_rule["heatmap"].max().item() < 99.0

    assert output["cls_preds"].shape == (2, 2, 5, 5)
    assert output["reg_preds"].shape == (2, 14, 5, 5)
    assert output["dir_preds"].shape == (2, 4, 5, 5)
    assert len(output["occ_single_list"]) == 1
    assert model.pyramid_backbone.forward_single_calls == 1
    for key in ("cls_preds", "reg_preds", "dir_preds"):
        assert output[key].requires_grad is False

    debug = output["pact_cbea_evidence_routed_debug"]
    _assert_compact_debug(debug)
    assert debug["evidence_checkpoint_verified"] is True
    assert debug["evidence_heads_used"] == ["m1", "m2", "m3", "m4"]
    assert debug["evidence_source"] == "modality_specific_evidence_heads"
    assert debug["shared_alignment_grid_used"] is True
    assert debug["forward_collab_used"] is False
    assert debug["stage3_training_required"] is False
    assert debug["no_joint_training_verified"] is True
    assert debug["trainable_total"] == 0
    assert debug["missing_evidence_fallback_used"] is False
    assert debug["per_scene_agent_count"] == [2, 3]
    assert len(debug["per_agent_valid_ratio"]) == 5
    assert debug["fallbacks"] == []
    assert isinstance(debug["alpha_nonuniform"], bool)
    print("full multi-scene forward, live evidence routing, heads and detached outputs OK")


def _check_static_isolation():
    with open(MODEL_PATH, "r") as stream:
        model_source = stream.read()
    with open(SUBMODULE_PATH, "r") as stream:
        submodule_source = stream.read()
    for source in (model_source, submodule_source):
        ast.parse(source, feature_version=(3, 8))
    forbidden_production = (
        "import sys",
        "import types",
        "sys.modules",
        "_install_model_import_stubs",
    )
    for token in forbidden_production:
        assert token not in model_source
        assert token not in submodule_source
    assert ".forward_collab(" not in model_source
    assert "_lookup_optional_evidence" not in model_source
    assert "data_dict.get(\"evidence" not in model_source

    protected = [
        "opencood/models/heter_pyramid_collab_pact_cbea.py",
        "opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/pact/cbea_rule.yaml",
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
        "opencood/models/heter_pyramid_collab_pact_cbea_evidence_routed.py",
        "opencood/models/sub_modules/pact_cbea_evidence_routed.py",
        "opencood/hypes_yaml/PACT_CBEA_EVIDENCE_ROUTED_v2/",
        "opencood/hypes_yaml/PACT_CBEA_EVIDENCE_ROUTED_v2/rule_evidence_routed.yaml",
        "opencood/tools/check_pact_cbea_evidence_routed_smoke.py",
    }
    for line in status.splitlines():
        assert line[3:].replace("\\", "/") in allowed, "unexpected workspace change: %s" % line
    print("Python 3.8 AST, no production stubs/fallback reads, and protected-file isolation OK")


def main():
    torch.manual_seed(31)
    hypes = _load_hypes()
    model = _check_discovery_guard_and_parameters(hypes)
    _check_evidence_heads(model)
    _check_unverified_forward(hypes)
    _check_checkpoint_validation(hypes, model)
    _check_shared_geometry_and_validity()
    _check_rule_behavior()
    _check_full_forward(model)
    _check_static_isolation()
    print("PACT_CBEA_EVIDENCE_ROUTED_SMOKE_PASS")


if __name__ == "__main__":
    try:
        main()
    finally:
        _restore_smoke_modules()

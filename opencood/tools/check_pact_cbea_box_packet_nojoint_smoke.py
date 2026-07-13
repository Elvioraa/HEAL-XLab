"""CPU smoke for isolated PACT no-joint sparse box-packet inference."""

import contextlib
import io
import importlib.machinery
import os
import subprocess
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO_ROOT_ADDED = REPO_ROOT not in sys.path
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_MODULES_BEFORE_SMOKE = dict(sys.modules)


def _set_smoke_module(name, module):
    sys.modules[name] = module


def _install_optional_dependency_stubs():
    """Install process-local import stubs before loading project modules."""
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
            from sklearn.metrics import mean_squared_error as unused_mse  # noqa: F401
    except Exception:
        sklearn_module = types.ModuleType("sklearn")
        metrics_module = types.ModuleType("sklearn.metrics")
        sklearn_module.__spec__ = importlib.machinery.ModuleSpec(
            "sklearn", loader=None
        )
        metrics_module.__spec__ = importlib.machinery.ModuleSpec(
            "sklearn.metrics", loader=None
        )
        metrics_module.mean_squared_error = lambda first, second: float(
            ((torch.as_tensor(first) - torch.as_tensor(second)) ** 2).mean()
        )
        sklearn_module.metrics = metrics_module
        _set_smoke_module("sklearn", sklearn_module)
        _set_smoke_module("sklearn.metrics", metrics_module)

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from opencood.utils.box_overlaps import (  # noqa: F401
                bbox_overlaps as unused_overlaps,
            )
    except Exception:
        module = types.ModuleType("opencood.utils.box_overlaps")

        def _missing_bbox_overlaps(*args, **kwargs):
            raise ImportError(
                "compiled box_overlaps is required for this operation"
            )

        module.bbox_overlaps = _missing_bbox_overlaps
        _set_smoke_module("opencood.utils.box_overlaps", module)


def _restore_smoke_modules():
    for name in list(sys.modules):
        if name not in _MODULES_BEFORE_SMOKE:
            del sys.modules[name]
    for name, module in _MODULES_BEFORE_SMOKE.items():
        sys.modules[name] = module
    if _REPO_ROOT_ADDED and REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)


_install_optional_dependency_stubs()

try:
    from opencood.models.heter_pyramid_collab_pact_cbea_box_packet_nojoint import (
        HeterPyramidCollabPactCbeaBoxPacketNojoint,
    )
    from opencood.models.sub_modules.pact_cbea_box_packet_nojoint import (
        PACKET_SOURCE,
        PACTNoJointBoxCommunicationMeter,
        PACTNoJointBoxPacketCodec,
    )
    from opencood.tools import inference_utils, train_utils
except Exception:
    _restore_smoke_modules()
    raise


YAML_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "PACT_CBEA_BOX_PACKET_NOJOINT_v1",
    "rule_box_packet.yaml",
)


class _DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, data_dict, modality_name):
        return self.bn(data_dict["inputs_%s" % modality_name]["feature"]) * self.scale


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, batch_dict):
        return {
            "spatial_features_2d": batch_dict["spatial_features"] * self.scale
        }


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

    def forward_collab(self, *args, **kwargs):
        raise AssertionError("box-packet no-joint must not call forward_collab")


class _InferenceDataset(object):
    def __init__(self, boxes, scores, gt):
        self.boxes = boxes
        self.scores = scores
        self.gt = gt
        self.call_count = 0

    def post_process(self, batch_data, output_dict):
        self.call_count += 1
        return self.boxes, self.scores, self.gt


class _OutputModel(object):
    def __init__(self, output):
        self.output = output

    def __call__(self, cav_content):
        return self.output


def main():
    torch.manual_seed(307)
    hypes = _load_yaml()
    _assert_yaml(hypes)
    codec, meter = _check_parameter_free_packet_modules()
    prediction, anchor = _check_local_decode(codec)
    _check_raw_transform(codec, prediction)
    _check_global_fusion(codec, prediction)
    _check_communication(meter, codec, prediction)
    model = _check_model_discovery_and_guard(hypes)
    output = _check_full_forward(model, anchor)
    _check_failure_policies(model, anchor)
    _check_inference_explicit_branch(output)
    _check_source_boundary()
    _check_existing_experiments_unchanged()

    print("PACT box-packet module import OK")
    print("PACT box-packet YAML load OK")
    print("PACT box-packet train_utils discovery OK")
    print("PACT box-packet explicit enable guard OK")
    print("PACT box-packet codec parameter count: 0")
    print("PACT box-packet local decode OK")
    print("PACT box-packet raw 4x4 transform OK")
    print("PACT box-packet global rotated NMS OK")
    print("PACT box-packet empty collaborator OK")
    print("PACT box-packet communication: 54 bytes/box OK")
    print("PACT box-packet multi-scene forward OK")
    print("PACT box-packet forward_collab forbidden OK")
    print("PACT box-packet failure policies OK")
    print("PACT box-packet trainable_total: 0")
    print("PACT box-packet inference explicit branch OK")
    print("PACT box-packet existing experiments unchanged OK")
    print("BOX_PACKET_NOJOINT_SMOKE_PASS")


def _load_yaml():
    with open(YAML_PATH, "r") as stream:
        return yaml.safe_load(stream)


def _assert_yaml(hypes):
    assert hypes["name"] == "PACT_CBEA_BOX_PACKET_NOJOINT_v1"
    assert hypes["fusion"]["args"]["proj_first"] is False
    assert hypes["postprocess"]["max_num"] == 256
    assert (
        hypes["model"]["core_method"]
        == "heter_pyramid_collab_pact_cbea_box_packet_nojoint"
    )
    cfg = hypes["model"]["args"]["pact_box_packet_nojoint"]
    assert cfg["enabled"] is True
    assert cfg["no_joint_training"] is True
    assert cfg["use_stage3_joint_training"] is False
    assert cfg["trainable"] is False
    assert cfg["packet_only_strict"] is True
    assert cfg["packet_source"] == PACKET_SOURCE
    assert cfg["quantize"] == "fp16"


def _check_parameter_free_packet_modules():
    codec = PACTNoJointBoxPacketCodec(
        score_threshold=0.2,
        local_nms_thresh=0.15,
        global_nms_thresh=0.15,
        max_boxes_per_agent=100,
        gt_range=[-102.4, -102.4, -3.0, 102.4, 102.4, 1.0],
    )
    meter = PACTNoJointBoxCommunicationMeter(
        quantize="fp16",
        bytes_per_scalar=2,
        deadline_ms=100,
        bandwidth_budget_kb=64,
    )
    assert codec.parameter_count() == 0
    assert meter.parameter_count() == 0
    try:
        PACTNoJointBoxPacketCodec(quantize="int8")
    except ValueError:
        pass
    else:
        raise AssertionError("non-fp16 box packet must be rejected")
    return codec, meter


def _synthetic_anchor():
    return torch.tensor([[[
        [0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 0.0],
        [0.0, 0.0, -1.0, 1.56, 1.6, 3.9, 1.5708],
    ]]], dtype=torch.float32)


def _check_local_decode(codec):
    anchor = _synthetic_anchor()
    cls_preds = torch.tensor([[[[5.0]], [[-5.0]]]], requires_grad=True)
    reg_preds = torch.zeros((1, 14, 1, 1), requires_grad=True)
    dir_preds = torch.tensor(
        [[[[3.0]], [[0.0]], [[3.0]], [[0.0]]]],
        requires_grad=True,
    )
    prediction = codec.decode_local_predictions(
        cls_preds, reg_preds, dir_preds, anchor
    )
    assert prediction["boxes_corner"].shape == (1, 8, 3)
    assert prediction["scores"].shape == (1,)
    assert prediction["boxes_center"].shape == (1, 7)
    assert all(not tensor.requires_grad for tensor in prediction.values())

    empty = codec.decode_local_predictions(
        torch.full((1, 2, 1, 1), -20.0),
        torch.zeros((1, 14, 1, 1)),
        torch.zeros((1, 4, 1, 1)),
        anchor,
    )
    assert empty["boxes_corner"].shape == (0, 8, 3)
    assert empty["scores"].shape == (0,)
    assert empty["boxes_center"].shape == (0, 7)
    return prediction, anchor


def _check_raw_transform(codec, prediction):
    corners = prediction["boxes_corner"]
    identity = codec.transform_packet_to_ego(corners, torch.eye(4))
    assert torch.allclose(identity, corners)

    transform = torch.eye(4)
    transform[0, 3] = 7.0
    transform[1, 3] = -3.0
    shifted = codec.transform_packet_to_ego(corners, transform)
    assert torch.allclose(shifted[..., 0], corners[..., 0] + 7.0)
    assert torch.allclose(shifted[..., 1], corners[..., 1] - 3.0)


def _check_global_fusion(codec, prediction):
    ego_packet = codec.build_packet(prediction, "m1", 0, transmitted=False)
    collaborator = codec.build_packet(prediction, "m2", 1, transmitted=True)
    fused = codec.fuse_packets(ego_packet, [collaborator])
    assert fused["boxes_corner"].shape[0] == 1
    assert fused["scores"].shape == (1,)
    empty_collaborator = codec.fuse_packets(ego_packet, [])
    assert empty_collaborator["boxes_corner"].shape[0] == 1
    assert not fused["boxes_corner"].requires_grad
    assert not fused["scores"].requires_grad


def _check_communication(meter, codec, prediction):
    collaborator = codec.build_packet(prediction, "m2", 1, transmitted=True)
    stats = meter([collaborator])
    assert stats["collaborator_box_count"] == 1
    assert stats["bytes_per_box"] == 54
    assert stats["packet_bytes_per_frame"] == 54
    assert stats["packet_kb_per_frame"] > 0.0
    assert stats["estimated_mbps"] > 0.0


def _lightweight_base_init(self, args):
    nn.Module.__init__(self)
    self.args = args
    self.cav_range = list(args["lidar_range"])
    self.modality_name_list = ["m1", "m2", "m3", "m4"]
    self.sensor_type_dict = {
        name: "lidar" for name in self.modality_name_list
    }
    self.shrink_flag = False
    for name in self.modality_name_list:
        setattr(self, "encoder_%s" % name, _DummyEncoder())
        setattr(self, "backbone_%s" % name, _DummyBackbone())
        setattr(self, "aligner_%s" % name, _DummyAligner())
        setattr(self, "depth_supervision_%s" % name, False)
    self.pyramid_backbone = _DummyPyramid()
    self.cls_head = nn.Conv2d(1, 2, kernel_size=1)
    self.reg_head = nn.Conv2d(1, 14, kernel_size=1)
    self.dir_head = nn.Conv2d(1, 4, kernel_size=1)
    with torch.no_grad():
        self.cls_head.weight.zero_()
        self.cls_head.bias.copy_(torch.tensor([5.0, -5.0]))
        self.reg_head.weight.zero_()
        self.reg_head.bias.zero_()
        self.dir_head.weight.zero_()
        self.dir_head.bias.copy_(torch.tensor([3.0, 0.0, 3.0, 0.0]))


def _check_model_discovery_and_guard(hypes):
    base_class = HeterPyramidCollabPactCbeaBoxPacketNojoint.__mro__[1]
    original_base_init = base_class.__init__
    base_class.__init__ = _lightweight_base_init
    try:
        model = train_utils.create_model(hypes)
        assert isinstance(
            model, HeterPyramidCollabPactCbeaBoxPacketNojoint
        )

        missing_args = dict(hypes["model"]["args"])
        missing_args.pop("pact_box_packet_nojoint")
        try:
            HeterPyramidCollabPactCbeaBoxPacketNojoint(missing_args)
        except ValueError:
            pass
        else:
            raise AssertionError("missing enabled config must fail")

        false_args = dict(hypes["model"]["args"])
        false_args["pact_box_packet_nojoint"] = {"enabled": False}
        try:
            HeterPyramidCollabPactCbeaBoxPacketNojoint(false_args)
        except ValueError:
            pass
        else:
            raise AssertionError("enabled=false must fail")
    finally:
        base_class.__init__ = original_base_init

    model.train()
    model.model_train_init()
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.encoder_m1.bn.training is False
    assert model._packet_parameter_count() == 0
    assert model.pact_box_packet_nojoint_summary["trainable_total"] == 0
    assert "load_state_dict" not in (
        HeterPyramidCollabPactCbeaBoxPacketNojoint.__dict__
    )
    return model


def _dummy_data(anchor):
    pairwise = torch.eye(4).view(1, 1, 1, 4, 4).repeat(
        2, 2, 2, 1, 1
    )
    pairwise[0, 1, 0, 0, 3] = 10.0
    return {
        "agent_modality_list": ["m1", "m2", "m3"],
        "record_len": torch.tensor([2, 1]),
        "pairwise_t_matrix": pairwise,
        "anchor_box": anchor,
        "inputs_m1": {"feature": torch.ones(1, 1, 1, 1)},
        "inputs_m2": {"feature": torch.ones(1, 1, 1, 1) * 2.0},
        "inputs_m3": {"feature": torch.ones(1, 1, 1, 1) * 3.0},
    }


def _check_full_forward(model, anchor):
    output = model(_dummy_data(anchor))
    assert output["cls_preds"].shape == (2, 2, 1, 1)
    assert output["reg_preds"].shape == (2, 14, 1, 1)
    assert output["dir_preds"].shape == (2, 4, 1, 1)
    assert len(output["occ_single_list"]) == 1
    assert output["box_packet_nojoint_enabled"] is True
    assert output["box_packet_pred_box_tensor"].shape == (3, 8, 3)
    assert output["box_packet_pred_score"].shape == (3,)
    assert output["box_packet_pred_box_tensor"][..., 0].max() > 8.0
    for key in (
            "cls_preds", "reg_preds", "dir_preds",
            "box_packet_pred_box_tensor", "box_packet_pred_score"):
        assert output[key].requires_grad is False

    debug = output["pact_box_packet_nojoint_debug"]
    assert debug["box_packet_only_verified"] is True
    assert debug["no_joint_training_verified"] is True
    assert debug["stage3_training_required"] is False
    assert debug["dense_collab_fusion_used"] is False
    assert debug["forward_collab_used"] is False
    assert debug["collaborator_dense_after_packet_used"] is False
    assert debug["packet_source"] == PACKET_SOURCE
    assert debug["collaborator_box_count"] == 1
    assert debug["packet_bytes_per_frame"] == 54
    assert debug["bytes_per_box"] == 54
    assert debug["packet_parameter_count"] == 0
    assert debug["trainable_total"] == 0
    assert [len(scene) for scene in debug["per_agent_box_count"]] == [2, 1]
    assert debug["per_agent_box_count"][0][0]["is_ego"] is True
    assert debug["per_agent_box_count"][1][0]["is_ego"] is True
    return output


def _check_failure_policies(model, anchor):
    original_transform = model.pact_box_packet_nojoint_codec.transform_packet_to_ego

    def _forced_failure(*args, **kwargs):
        raise RuntimeError("forced collaborator packet failure")

    model.pact_box_packet_nojoint_codec.transform_packet_to_ego = _forced_failure
    original_policy = model.pact_box_packet_nojoint_cfg["failure_policy"]
    try:
        model.pact_box_packet_nojoint_cfg["failure_policy"] = "ego_only"
        output = model(_dummy_data(anchor))
        debug = output["pact_box_packet_nojoint_debug"]
        assert output["box_packet_pred_box_tensor"].shape[0] == 2
        assert debug["fallback_reason"]
        assert "forced collaborator packet failure" in debug["fallback_reason"]
        assert debug["dense_collab_fusion_used"] is False
        assert debug["collaborator_dense_after_packet_used"] is False

        model.pact_box_packet_nojoint_cfg["failure_policy"] = "error"
        try:
            model(_dummy_data(anchor))
        except RuntimeError as exc:
            assert "fusion failed" in str(exc)
        else:
            raise AssertionError("failure_policy=error must raise")
    finally:
        model.pact_box_packet_nojoint_codec.transform_packet_to_ego = (
            original_transform
        )
        model.pact_box_packet_nojoint_cfg["failure_policy"] = original_policy


def _check_inference_explicit_branch(output):
    official_boxes = torch.zeros((1, 8, 3))
    official_scores = torch.tensor([0.25])
    gt = torch.ones((1, 8, 3))

    official_dataset = _InferenceDataset(
        official_boxes, official_scores, gt
    )
    official = inference_utils.inference_early_fusion(
        {"ego": {}}, _OutputModel({"cls_preds": torch.zeros(1)}),
        official_dataset,
    )
    assert official_dataset.call_count == 1
    assert official["pred_box_tensor"] is official_boxes
    assert official["pred_score"] is official_scores

    direct_dataset = _InferenceDataset(official_boxes, official_scores, gt)
    direct = inference_utils.inference_early_fusion(
        {"ego": {}},
        _OutputModel({
            "box_packet_nojoint_enabled": True,
            "box_packet_pred_box_tensor": output[
                "box_packet_pred_box_tensor"
            ],
            "box_packet_pred_score": output["box_packet_pred_score"],
        }),
        direct_dataset,
    )
    assert direct_dataset.call_count == 1
    assert direct["pred_box_tensor"] is output[
        "box_packet_pred_box_tensor"
    ]
    assert direct["pred_score"] is output["box_packet_pred_score"]
    assert direct["gt_box_tensor"] is gt

    missing_dataset = _InferenceDataset(official_boxes, official_scores, gt)
    try:
        inference_utils.inference_early_fusion(
            {"ego": {}},
            _OutputModel({"box_packet_nojoint_enabled": True}),
            missing_dataset,
        )
    except RuntimeError as exc:
        assert "missing required fields" in str(exc)
        assert missing_dataset.call_count == 1
    else:
        raise AssertionError("missing direct box output must fail")


def _check_source_boundary():
    model_path = os.path.join(
        REPO_ROOT,
        "opencood",
        "models",
        "heter_pyramid_collab_pact_cbea_box_packet_nojoint.py",
    )
    with open(model_path, "r") as stream:
        source = stream.read()
    forbidden = (
        ".forward_collab(",
        "PACTCBEALocalEvidenceHead",
        "PACTNoJointPacketizer",
        "PACTNoJointPacketAggregator",
        "normalize_pairwise_tfm",
        "affine_grid",
        "grid_sample",
        "def load_state_dict",
    )
    for token in forbidden:
        assert token not in source
    assert "pairwise_t_matrix[" in source
    assert "batch_index, local_index, 0" in source


def _check_existing_experiments_unchanged():
    protected = [
        "opencood/models/heter_pyramid_collab_bger.py",
        "opencood/models/sub_modules/bger_box_prior.py",
        "opencood/models/sub_modules/bger_refine.py",
        "opencood/tools/check_bger_smoke.py",
        "opencood/tools/inference_bger.py",
        "opencood/tools/prepare_bger.py",
        "docs/BGER_PLAN.md",
        "opencood/models/heter_pyramid_collab_pact_cbea_packet_nojoint.py",
        "opencood/models/sub_modules/pact_cbea_packet_nojoint.py",
        "opencood/hypes_yaml/PACT_CBEA_PACKET_NOJOINT_v1",
    ]
    result = subprocess.run(
        ["git", "diff", "--quiet", "--"] + protected,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0


if __name__ == "__main__":
    try:
        main()
    finally:
        _restore_smoke_modules()

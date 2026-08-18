"""CPU regression tests for Dual-Space runtime inference mode preparation."""

import copy
import ast
import io
import os
import sys
import types
from contextlib import redirect_stdout

import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def install_test_dependency_stubs():
    """Install lightweight local stubs before importing HEAL model modules."""
    if "icecream" not in sys.modules:
        module = types.ModuleType("icecream")
        module.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
        sys.modules["icecream"] = module

    if "timm.models.layers" not in sys.modules:
        timm_module = types.ModuleType("timm")
        models_module = types.ModuleType("timm.models")
        layers_module = types.ModuleType("timm.models.layers")

        class DropPath(nn.Identity):
            pass

        layers_module.DropPath = DropPath
        timm_module.models = models_module
        models_module.layers = layers_module
        sys.modules["timm"] = timm_module
        sys.modules["timm.models"] = models_module
        sys.modules["timm.models.layers"] = layers_module

    if "einops" not in sys.modules:
        einops_module = types.ModuleType("einops")

        def rearrange(value, pattern, **axes):
            if pattern != "b (l c) h w -> b l c h w":
                raise NotImplementedError("test stub does not support %s" % pattern)
            length = axes["l"]
            batch, combined, height, width = value.shape
            return value.reshape(batch, length, combined // length, height, width)

        def repeat(value, pattern, **axes):
            raise NotImplementedError("einops.repeat is outside this smoke path")

        einops_module.rearrange = rearrange
        einops_module.repeat = repeat
        sys.modules["einops"] = einops_module

    if "shapely.geometry" not in sys.modules:
        shapely_module = types.ModuleType("shapely")
        geometry_module = types.ModuleType("shapely.geometry")

        class GeometryStub(object):
            pass

        geometry_module.Polygon = GeometryStub
        geometry_module.Point = GeometryStub
        geometry_module.MultiPoint = GeometryStub
        shapely_module.geometry = geometry_module
        sys.modules["shapely"] = shapely_module
        sys.modules["shapely.geometry"] = geometry_module

    if "pyquaternion" not in sys.modules:
        quaternion_module = types.ModuleType("pyquaternion")

        class Quaternion(object):
            pass

        quaternion_module.Quaternion = Quaternion
        sys.modules["pyquaternion"] = quaternion_module

    if "sklearn.metrics" not in sys.modules:
        sklearn_module = types.ModuleType("sklearn")
        metrics_module = types.ModuleType("sklearn.metrics")

        def mean_squared_error(*args, **kwargs):
            raise NotImplementedError("depth RMSE is outside this smoke path")

        metrics_module.mean_squared_error = mean_squared_error
        sklearn_module.metrics = metrics_module
        sys.modules["sklearn"] = sklearn_module
        sys.modules["sklearn.metrics"] = metrics_module

    if "torchvision" not in sys.modules:
        torchvision_module = types.ModuleType("torchvision")
        transforms_module = types.ModuleType("torchvision.transforms")

        class CenterCrop(object):
            def __init__(self, size):
                self.size = size

            def __call__(self, value):
                return value

        class Normalize(object):
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std

            def __call__(self, value):
                return value

        class Compose(object):
            def __init__(self, transforms):
                self.transforms = transforms

            def __call__(self, value):
                for transform in self.transforms:
                    value = transform(value)
                return value

        class IdentityTransform(object):
            def __call__(self, value):
                return value

        transforms_module.CenterCrop = CenterCrop
        transforms_module.Normalize = Normalize
        transforms_module.Compose = Compose
        transforms_module.ToPILImage = IdentityTransform
        transforms_module.ToTensor = IdentityTransform
        torchvision_module.transforms = transforms_module
        sys.modules["torchvision"] = torchvision_module
        sys.modules["torchvision.transforms"] = transforms_module

    encoder_module = types.ModuleType("opencood.models.heter_encoders")

    class TestEncoder(nn.Module):
        def __init__(self, args):
            super().__init__()

        def forward(self, data_dict, modality_name):
            return data_dict["inputs_%s" % modality_name]["spatial_features"]

    encoder_module.TestEncoder = TestEncoder
    sys.modules["opencood.models.heter_encoders"] = encoder_module


install_test_dependency_stubs()

from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.heter_pyramid_single import HeterPyramidSingle
from opencood.models.sub_modules.dual_space_config import (
    prepare_dual_space_inference_config,
    validate_dual_space_config,
)
from opencood.tools import inference_utils, train_utils
from opencood.tools.dual_space_smoke_common import make_dual_config


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


def model_args(modality="m1", dual=None):
    """Return a tiny real HEAL model contract for one lidar modality."""
    args = {
        "lidar_range": [-4.0, -4.0, -3.0, 4.0, 4.0, 1.0],
        modality: {
            "core_method": "test_encoder",
            "sensor_type": "lidar",
            "encoder_args": {},
            "backbone_args": {
                "layer_nums": [1],
                "layer_strides": [1],
                "num_filters": [64],
            },
            "aligner_args": {"core_method": "identity"},
        },
        "fusion_backbone": {
            "resnext": False,
            "layer_nums": [1],
            "layer_strides": [1],
            "num_filters": [64],
            "upsample_strides": [1],
            "num_upsample_filter": [8],
            "anchor_number": 2,
        },
        "in_head": 8,
        "anchor_number": 2,
        "dir_args": {"num_bins": 2},
    }
    if dual is not None:
        args["dual_space"] = dual
    return args


def hypes_for(args, core_method="heter_pyramid_collab"):
    return {"model": {"core_method": core_method, "args": args}}


def collab_input():
    identity = torch.eye(4, dtype=torch.float32)
    return {
        "agent_modality_list": ["m1", "m1"],
        "pairwise_t_matrix": identity.view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1),
        "record_len": torch.tensor([2], dtype=torch.long),
        "inputs_m1": {"spatial_features": torch.randn(2, 64, 8, 8)},
    }


def single_input(modality="m2"):
    return {
        "inputs_%s" % modality: {
            "spatial_features": torch.randn(1, 64, 8, 8)
        }
    }


class EmptyDataset(object):
    def post_process(self, batch_data, output_dict):
        return None, None, None


def validate_complete_checkpoint(model):
    model.validate_dual_space_checkpoint_keys(model.state_dict().keys())


@test("Stage1 config becomes inference-only in memory and passes context guard")
def test_stage1_direct_inference():
    source = hypes_for(model_args(dual=make_dual_config(mode="stage1_anchor")))
    before = copy.deepcopy(source)
    runtime = prepare_dual_space_inference_config(source)
    assert source == before
    assert runtime is not source
    assert runtime["model"]["args"]["dual_space"]["mode"] == "inference"
    assert runtime["model"]["args"]["dual_space"][
        "allow_untrained_initialization"
    ] is False

    model = HeterPyramidCollab(runtime["model"]["args"])
    assert model._dual_space_checkpoint_ready is False
    validate_complete_checkpoint(model)
    assert model._dual_space_checkpoint_ready is True
    model.eval()
    output = model(collab_input())
    assert "dual_space_context" in output
    result = inference_utils.inference_intermediate_fusion(
        {"ego": collab_input()}, model, EmptyDataset()
    )
    assert result["pred_box_tensor"] is None


@test("already-final inference config is returned unchanged")
def test_final_inference_unchanged():
    source = hypes_for(model_args(dual=make_dual_config(mode="inference")))
    runtime = prepare_dual_space_inference_config(source)
    assert runtime is source
    assert runtime["model"]["args"]["dual_space"]["mode"] == "inference"


@test("Dual-Space disabled keeps the official HEAL inference path")
def test_disabled_path():
    dual = {
        "enabled": False,
        "multi_scale": {"enabled": False},
        "quality": {"enabled": False},
        "remote_proposal_rescue": {"enabled": False},
        "diagnostics": {"enabled": False},
        "training_proposals": {"source": "gt_jitter"},
    }
    source = hypes_for(model_args(dual=dual))
    runtime = prepare_dual_space_inference_config(source)
    assert runtime is source
    model = HeterPyramidCollab(runtime["model"]["args"])
    model.eval()
    output = model(collab_input())
    assert model.dual_space_enabled is False
    assert "dual_space_context" not in output
    result = inference_utils.inference_intermediate_fusion(
        {"ego": collab_input()}, model, EmptyDataset()
    )
    assert result["pred_box_tensor"] is None


@test("Stage1 training construction retains fresh trainable semantics")
def test_stage1_training_unchanged():
    source = hypes_for(model_args(dual=make_dual_config(mode="stage1_anchor")))
    model = HeterPyramidCollab(source["model"]["args"])
    model.train()
    model.model_train_init()
    output = io.StringIO()
    with redirect_stdout(output):
        train_utils.validate_initialization_source(model, "")
    assert model.dual_space_config["mode"] == "stage1_anchor"
    assert model._dual_space_checkpoint_ready is True
    assert "[DualSpace] initialization=fresh" in output.getvalue()
    assert any(
        parameter.requires_grad
        for parameter in model.dual_space_shared_object_refiner.parameters()
    )


@test("Stage2 training and independent runtime inference contracts stay distinct")
def test_stage2_contracts():
    dual = make_dual_config(mode="stage2_adapt", active_modality="m2")
    source = hypes_for(
        model_args(modality="m2", dual=dual),
        core_method="heter_pyramid_single",
    )
    before = copy.deepcopy(source)

    training_model = HeterPyramidSingle(source["model"]["args"])
    assert training_model.dual_space_config["mode"] == "stage2_adapt"
    assert training_model.dual_space_config["active_modality"] == "m2"
    assert any(
        parameter.requires_grad
        for parameter in training_model.dual_space_object_adapter_m2.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in training_model.dual_space_shared_object_refiner.parameters()
    )

    runtime = prepare_dual_space_inference_config(source)
    assert source == before
    runtime_dual = runtime["model"]["args"]["dual_space"]
    assert runtime_dual["mode"] == "inference"
    assert "active_modality" not in runtime_dual
    runtime_model = HeterPyramidSingle(runtime["model"]["args"])
    assert runtime_model._dual_space_checkpoint_ready is False
    validate_complete_checkpoint(runtime_model)
    assert runtime_model._dual_space_checkpoint_ready is True
    runtime_model.eval()
    output = runtime_model(single_input())
    assert "dual_space_context" in output
    inference_utils.inference_intermediate_fusion(
        {"ego": single_input()}, runtime_model, EmptyDataset()
    )


@test("all Dual-Space feature profiles preserve features during override")
def test_profile_compatibility():
    profiles = (
        ("ds_v1", make_dual_config(mode="stage1_anchor")),
        ("ds_v1_1", make_dual_config(mode="stage1_anchor")),
        ("ds_v2", make_dual_config(mode="stage1_anchor", multi=True)),
        ("ds_v2_1", make_dual_config(mode="stage1_anchor", multi=True)),
        ("ds_v3", make_dual_config(mode="stage1_anchor", multi=True, quality=True)),
        ("ds_v4", make_dual_config(mode="inference", multi=True, quality=True, rescue=True)),
    )
    for profile, dual in profiles:
        dual["version"] = profile
        dual["experiment_profile"] = profile
        if profile == "ds_v1_1":
            dual["refiner"]["yaw_mode"] = "sin_cos_centered"
        validate_dual_space_config(dual)
        source = hypes_for(model_args(dual=dual))
        before = copy.deepcopy(source)
        runtime = prepare_dual_space_inference_config(source)
        assert source == before
        runtime_dual = runtime["model"]["args"]["dual_space"]
        expected = copy.deepcopy(dual)
        expected["mode"] = "inference"
        expected["allow_untrained_initialization"] = False
        expected.pop("active_modality", None)
        assert runtime_dual == expected


@test("missing same-forward context guard remains active")
def test_missing_context_guard():
    class BrokenModel(object):
        dual_space_enabled = True

        def __call__(self, data):
            return {}

    try:
        inference_utils.inference_intermediate_fusion(
            {"ego": {}}, BrokenModel(), EmptyDataset()
        )
    except RuntimeError as error:
        assert "missing same-forward Common-BEV context" in str(error)
    else:
        raise AssertionError("missing-context inference guard was bypassed")


@test("inference entry point prepares runtime mode before model construction")
def test_inference_entrypoint_order():
    inference_path = os.path.join(REPO_ROOT, "opencood", "tools", "inference.py")
    with open(inference_path, "r", encoding="utf-8") as stream:
        module = ast.parse(stream.read(), filename=inference_path)
    main_function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    prepare_line = None
    create_line = None
    for node in ast.walk(main_function):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == (
            "prepare_dual_space_inference_config"
        ):
            prepare_line = node.lineno
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "train_utils"
            and function.attr == "create_model"
        ):
            create_line = node.lineno
    assert prepare_line is not None and create_line is not None
    assert prepare_line < create_line


def main():
    torch.set_num_threads(1)
    passed = 0
    for name, function in TESTS:
        try:
            function()
        except Exception as error:
            print("[FAIL] %s: %s: %s" % (name, type(error).__name__, error))
        else:
            passed += 1
            print("[PASS] %s" % name)
    print("RESULT: %d/%d PASS" % (passed, len(TESTS)))
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Audit strict disabled-mode parity for the Phase 1 Open-DCSI wrappers."""

import argparse
from copy import deepcopy
import random
from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_audit_only_dependency_stubs():
    if "icecream" not in sys.modules:
        module = types.ModuleType("icecream")
        module.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
        sys.modules["icecream"] = module

    if "timm.models.layers" not in sys.modules:
        timm_module = types.ModuleType("timm")
        models_module = types.ModuleType("timm.models")
        layers_module = types.ModuleType("timm.models.layers")

        class DropPath(nn.Identity):
            """Identity DropPath used only when timm is unavailable locally."""

        layers_module.DropPath = DropPath
        timm_module.models = models_module
        models_module.layers = layers_module
        sys.modules["timm"] = timm_module
        sys.modules["timm.models"] = models_module
        sys.modules["timm.models.layers"] = layers_module

    encoder_module = types.ModuleType("opencood.models.heter_encoders")

    class AuditEncoder(nn.Module):
        def __init__(self, args):
            super().__init__()

        def forward(self, data_dict, modality_name):
            return data_dict["inputs_{}".format(modality_name)]["spatial_features"]

    encoder_module.AuditEncoder = AuditEncoder
    sys.modules["opencood.models.heter_encoders"] = encoder_module


_install_audit_only_dependency_stubs()

from opencood.models.heter_pyramid_collab_pact_cbea import (
    HeterPyramidCollabPactCbea,
)
from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.heter_pyramid_collab_open_dcsi import (
    HeterPyramidCollabOpenDcsi,
)
from opencood.models.heter_pyramid_collab_open_dcsi_stage1 import (
    HeterPyramidCollabOpenDcsiStage1,
)
from opencood.models.heter_pyramid_single import HeterPyramidSingle
from opencood.models.heter_pyramid_single_open_dcsi_stage2 import (
    HeterPyramidSingleOpenDcsiStage2,
)
from opencood.models.sub_modules.open_dcsi.config import (
    is_open_dcsi_enabled,
    normalize_open_dcsi_config,
    validate_open_dcsi_config,
)
import opencood.models.fuse_modules.pyramid_fuse as pyramid_fuse_module


def _tiny_model_args(open_dcsi_marker="missing"):
    args = {
        "lidar_range": [-4.0, -4.0, -3.0, 4.0, 4.0, 1.0],
        "supervise_single": True,
        "m1": {
            "core_method": "audit_encoder",
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
            "layer_nums": [1, 1],
            "layer_strides": [1, 2],
            "num_filters": [64, 128],
            "upsample_strides": [1, 2],
            "num_upsample_filter": [16, 16],
            "anchor_number": 2,
        },
        "shrink_header": {
            "kernal_size": [3],
            "stride": [1],
            "padding": [1],
            "dim": [8],
            "input_dim": 32,
        },
        "in_head": 8,
        "anchor_number": 2,
        "dir_args": {"num_bins": 2},
    }
    if open_dcsi_marker == "false":
        args["open_dcsi"] = {"enabled": False}
    return args


def _collab_input():
    identity = torch.eye(4, dtype=torch.float32)
    pairwise = identity.view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    feature = torch.linspace(
        -1.0, 1.0, steps=2 * 64 * 8 * 8, dtype=torch.float32
    ).reshape(2, 64, 8, 8)
    return {
        "agent_modality_list": ["m1", "m1"],
        "pairwise_t_matrix": pairwise,
        "record_len": torch.tensor([2], dtype=torch.long),
        "inputs_m1": {"spatial_features": feature},
    }


def _single_input():
    feature = torch.linspace(
        -0.5, 0.5, steps=2 * 64 * 8 * 8, dtype=torch.float32
    ).reshape(2, 64, 8, 8)
    return {"inputs_m1": {"spatial_features": feature}}


def _snapshot_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _assert_rng_equal(left, right, label):
    assert left["python"] == right["python"], "{} Python RNG changed".format(label)
    assert left["numpy"][0] == right["numpy"][0]
    assert np.array_equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][2:] == right["numpy"][2:]
    assert torch.equal(left["torch"], right["torch"]), "{} torch RNG changed".format(
        label
    )
    assert len(left["cuda"]) == len(right["cuda"])
    for index, (left_state, right_state) in enumerate(zip(left["cuda"], right["cuda"])):
        assert torch.equal(left_state, right_state), "{} CUDA RNG {} changed".format(
            label, index
        )


def _tensor_error(left, right):
    if left.numel() == 0:
        return 0.0, 0.0
    delta = (left.to(torch.float64) - right.to(torch.float64)).abs()
    max_abs = float(delta.max())
    denominator = right.to(torch.float64).abs().clamp_min(1e-12)
    max_rel = float((delta / denominator).max())
    return max_abs, max_rel


def _assert_tree_equal(left, right, path, metrics):
    assert type(left) is type(right), "{} type mismatch: {} != {}".format(
        path, type(left), type(right)
    )
    if torch.is_tensor(left):
        assert left.shape == right.shape, "{} shape mismatch".format(path)
        max_abs, max_rel = _tensor_error(left, right)
        metrics.append((path, max_abs, max_rel))
        assert torch.equal(left, right), (
            "{} differs: max_abs_error={}, max_relative_error={}".format(
                path, max_abs, max_rel
            )
        )
        return
    if isinstance(left, dict):
        assert list(left.keys()) == list(right.keys()), "{} key/order mismatch".format(path)
        for key in left:
            _assert_tree_equal(left[key], right[key], "{}.{}".format(path, key), metrics)
        return
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right), "{} length mismatch".format(path)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_tree_equal(
                left_item,
                right_item,
                "{}[{}]".format(path, index),
                metrics,
            )
        return
    assert left == right, "{} value mismatch".format(path)


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _capture_forward(model, data_dict):
    captured = {}
    handles = []
    for name in ("backbone_m1", "pyramid_backbone", "shrink_conv"):
        module = getattr(model, name, None)
        if module is None:
            continue

        def hook(_module, _inputs, output, capture_name=name):
            captured[capture_name] = _clone_tree(output)

        handles.append(module.register_forward_hook(hook))

    warp_outputs = []
    original_warp = pyramid_fuse_module.warp_affine_simple

    def capture_warp(*args, **kwargs):
        output = original_warp(*args, **kwargs)
        warp_outputs.append(output.detach().clone())
        return output

    pyramid_fuse_module.warp_affine_simple = capture_warp
    try:
        with torch.no_grad():
            output = model(_clone_tree(data_dict))
    finally:
        pyramid_fuse_module.warp_affine_simple = original_warp
        for handle in handles:
            handle.remove()
    captured["warp_outputs"] = warp_outputs
    return output, captured


def _parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def _buffer_count(model):
    buffers = list(model.buffers())
    return len(buffers), sum(buffer.numel() for buffer in buffers)


def _run_pair(label, parent_cls, wrapper_cls, data_factory):
    torch.manual_seed(20260801)
    parent = parent_cls(deepcopy(_tiny_model_args("missing"))).eval()
    parent_state = parent.state_dict()

    for marker in ("missing", "false"):
        torch.manual_seed(20260801)
        wrapper = wrapper_cls(deepcopy(_tiny_model_args(marker))).eval()
        load_result = wrapper.load_state_dict(parent_state, strict=True)
        assert not load_result.missing_keys, "{} {} missing keys".format(label, marker)
        assert not load_result.unexpected_keys, "{} {} unexpected keys".format(
            label, marker
        )

        assert list(parent.state_dict()) == list(wrapper.state_dict()), (
            "{} {} state-dict key/order mismatch".format(label, marker)
        )
        assert _parameter_count(parent) == _parameter_count(wrapper)
        assert _buffer_count(parent) == _buffer_count(wrapper)

        initial_rng = _snapshot_rng()
        _restore_rng(initial_rng)
        parent_output, parent_middle = _capture_forward(parent, data_factory())
        parent_after = _snapshot_rng()
        _restore_rng(initial_rng)
        wrapper_output, wrapper_middle = _capture_forward(wrapper, data_factory())
        wrapper_after = _snapshot_rng()
        _assert_rng_equal(parent_after, wrapper_after, "{} {}".format(label, marker))

        output_metrics = []
        middle_metrics = []
        _assert_tree_equal(parent_output, wrapper_output, "output", output_metrics)
        _assert_tree_equal(parent_middle, wrapper_middle, "middle", middle_metrics)
        max_abs = max((metric[1] for metric in output_metrics + middle_metrics), default=0.0)
        max_rel = max((metric[2] for metric in output_metrics + middle_metrics), default=0.0)
        print(
            "[{}:{}] keys={} params={} buffers={} max_abs_error={} "
            "max_relative_error={}".format(
                label,
                marker,
                len(parent_state),
                _parameter_count(parent),
                _buffer_count(parent),
                max_abs,
                max_rel,
            )
        )


def _audit_config():
    assert not is_open_dcsi_enabled(None)
    assert not is_open_dcsi_enabled({})
    assert not is_open_dcsi_enabled({"enabled": False})
    missing = normalize_open_dcsi_config(None)
    explicit = normalize_open_dcsi_config({"enabled": False})
    assert missing == explicit
    assert all(not _path_enabled(missing, path) for path in _feature_paths())

    expected = (
        "Open-DCSI common_space is not implemented in the current development phase"
    )
    try:
        validate_open_dcsi_config(
            {"enabled": True, "common_space": {"enabled": True}}
        )
    except ValueError as error:
        assert str(error) == expected
    else:
        raise AssertionError("unimplemented common_space did not fail")
    print("[config] missing/false defaults and phase guard OK")


def _feature_paths():
    from opencood.models.sub_modules.open_dcsi.config import _FEATURE_PATHS

    return _FEATURE_PATHS


def _path_enabled(config, path):
    value = config
    for part in path.split("."):
        value = value[part]
    return value["enabled"]


def _audit_yaml():
    baseline_path = (
        REPO_ROOT
        / "opencood/hypes_yaml/opv2v/MoreModality/HEAL/stage1/m1_pyramid.yaml"
    )
    parity_path = (
        REPO_ROOT
        / "opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/open_dcsi/stage1/"
        "m1_open_dcsi_official_parity.yaml"
    )
    with baseline_path.open("r", encoding="utf-8") as stream:
        baseline = yaml.safe_load(stream)
    with parity_path.open("r", encoding="utf-8") as stream:
        parity = yaml.safe_load(stream)

    assert parity["model"]["core_method"] == "heter_pyramid_collab_open_dcsi_stage1"
    assert parity["model"]["args"]["open_dcsi"] == {"enabled": False}
    baseline_compare = deepcopy(baseline)
    parity_compare = deepcopy(parity)
    baseline_compare.pop("name")
    parity_compare.pop("name")
    baseline_compare["model"].pop("core_method")
    parity_compare["model"].pop("core_method")
    parity_compare["model"]["args"].pop("open_dcsi")
    assert baseline_compare == parity_compare, "parity YAML changed official settings"
    print("[yaml] official training settings preserved")


def _audit_external_checkpoint(checkpoint_path):
    if checkpoint_path is None:
        print("[checkpoint] strict parent state-dict compatibility checked")
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dict")
    wrapper = HeterPyramidCollabOpenDcsiStage1(
        deepcopy(_tiny_model_args("false"))
    )
    result = wrapper.load_state_dict(checkpoint, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    print("[checkpoint] external checkpoint strict load OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint matching the audit model architecture.",
    )
    args = parser.parse_args()

    _audit_config()
    _audit_yaml()
    _run_pair(
        "stage1",
        HeterPyramidCollab,
        HeterPyramidCollabOpenDcsiStage1,
        _collab_input,
    )
    _run_pair(
        "stage2",
        HeterPyramidSingle,
        HeterPyramidSingleOpenDcsiStage2,
        _single_input,
    )
    _run_pair(
        "final-pact",
        HeterPyramidCollabPactCbea,
        HeterPyramidCollabOpenDcsi,
        _collab_input,
    )
    _audit_external_checkpoint(args.checkpoint)
    print("OPEN_DCSI_PHASE1_PASS")


if __name__ == "__main__":
    main()

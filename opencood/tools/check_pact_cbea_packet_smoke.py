"""Smoke checks for the independent PACT-CBEA packet-only experiment."""

import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml import yaml_utils
from opencood.models.heter_pyramid_collab_pact_cbea_packet import (
    HeterPyramidCollabPactCbeaPacket,
)
from opencood.models.sub_modules.pact_cbea_packet import (
    PACKET_SOURCE,
    PACTPacketAggregator,
    PACTPacketCompressor,
    PACTPacketizer,
    collate_scene_packets,
    flatten_agent_packets,
)


YAML_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "PACT_CBEA_PACKET_v1",
    "stage3",
    "packet_aggregator.yaml",
)


class _DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(64)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, data_dict, modality_name):
        return self.bn(data_dict["inputs_%s" % modality_name]["feature"]) * self.scale


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

    def forward_collab(self, *args, **kwargs):
        raise AssertionError("packet-only model must not call dense forward_collab")


def main():
    torch.manual_seed(103)
    hypes = yaml_utils.load_yaml(YAML_PATH)
    _assert_yaml(hypes)
    _check_packet_modules()

    model = _build_dummy_packet_model(hypes)
    model.train()
    _check_train_only_packet(model)
    output_dict = model(_dummy_data())
    _assert_packet_only_output(output_dict)
    _check_model_backward(model, output_dict)
    _check_multi_scene_output(output_dict)
    _check_model_empty_packet(model)
    _check_strict_failures(model)
    _check_ego_only_failure_policy(hypes)
    _check_checkpoint_compatibility(hypes)

    print("PACT-CBEA packet yaml load OK")
    print("PACT-CBEA packetizer forward OK")
    print("PACT-CBEA packet aggregator forward OK")
    print("PACT-CBEA packet backward OK")
    print("PACT-CBEA packet empty OK")
    print("PACT-CBEA packet multi-scene OK")
    print("PACT-CBEA packet dense fallback forbidden OK")
    print("PACT-CBEA packet train-only freeze OK")
    print("PACT-CBEA packet frozen BN eval OK")
    print("PACT-CBEA packet checkpoint compatibility OK")
    print("PACT-CBEA packet-only boundary OK")
    print("PACT-CBEA packet smoke OK")


def _assert_yaml(hypes):
    assert hypes["name"] == "PACT_CBEA_PACKET_v1/stage3/packet_aggregator"
    assert hypes["model"]["core_method"] == "heter_pyramid_collab_pact_cbea_packet"
    args = hypes["model"]["args"]
    packet = args["packet"]
    assert args["train_only_packet"] is True
    assert args["supervise_single"] is False
    assert packet["enabled"] is True
    assert packet["packet_only_strict"] is True
    assert packet["packet_failure_policy"] == "error"
    assert packet["source"] == PACKET_SOURCE


def _check_packet_modules():
    packetizer = PACTPacketizer(64, packet_dim=16, descriptor_dim=8, topk=20)
    raw_packet = packetizer(torch.randn(2, 64, 8, 8))
    assert raw_packet["boxes"].shape == (2, 20, 7)
    assert raw_packet["descriptor"].shape == (2, 20, 8)
    assert raw_packet["packet_feature"].shape == (2, 20, 16)
    compressor = PACTPacketCompressor(quantize="fp16", bandwidth_budget_kb=8)
    packet = compressor(raw_packet)
    assert packet["comm_stats"]["packet_num"] == 40
    assert packet["comm_stats"]["bytes_per_frame"] > 0
    scene_packet = collate_scene_packets([flatten_agent_packets(packet)])
    aggregator = PACTPacketAggregator(64, packet_dim=16, descriptor_dim=8, hidden_dim=16)
    ego = torch.randn(1, 64, 8, 8)
    delta, debug = aggregator(ego, scene_packet)
    assert delta.shape == ego.shape
    assert torch.isfinite(delta).all()
    delta.mean().backward()
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
        for param in aggregator.parameters()
    )
    empty = packetizer.empty_packet(ego.device, ego.dtype, agents=0)
    empty_scene = collate_scene_packets([flatten_agent_packets(empty)])
    empty_delta, empty_debug = aggregator(ego, empty_scene)
    assert empty_debug["empty_packet"] is True
    assert torch.allclose(empty_delta, torch.zeros_like(empty_delta))


def _build_dummy_packet_model(hypes):
    channels = 64
    model = HeterPyramidCollabPactCbeaPacket.__new__(HeterPyramidCollabPactCbeaPacket)
    nn.Module.__init__(model)
    model.modality_name_list = ["m1", "m2", "m3", "m4"]
    model.sensor_type_dict = {name: "lidar" for name in model.modality_name_list}
    model.cam_crop_info = {}
    model.H = 8
    model.W = 8
    model.fake_voxel_size = 1
    model.compress = False
    model.shrink_flag = False
    model.supervise_single = False
    for modality_name in model.modality_name_list:
        setattr(model, "encoder_%s" % modality_name, _DummyEncoder())
        setattr(model, "backbone_%s" % modality_name, _DummyBackbone())
        setattr(model, "aligner_%s" % modality_name, _DummyAligner())
        setattr(model, "depth_supervision_%s" % modality_name, False)
    model.pyramid_backbone = _DummyPyramid()
    model.cls_head = nn.Conv2d(channels, 2, kernel_size=1)
    model.reg_head = nn.Conv2d(channels, 14, kernel_size=1)
    model.dir_head = nn.Conv2d(channels, 4, kernel_size=1)
    model.packet_cfg = HeterPyramidCollabPactCbeaPacket._normalize_packet_cfg({
        **hypes["model"]["args"]["packet"],
        "in_channels": channels,
        "context_channels": channels,
        "hidden_dim": 16,
    })
    model.train_only_packet = True
    model.packet_only_strict = True
    model.packet_failure_policy = "error"
    model.pact_packetizer = PACTPacketizer(channels, packet_dim=16, descriptor_dim=8, topk=20)
    model.pact_packet_compressor = PACTPacketCompressor("fp16", bandwidth_budget_kb=8)
    model.pact_packet_aggregator = PACTPacketAggregator(channels, 16, 8, hidden_dim=16)
    model.pact_packet_residual_fusion = nn.Module()
    from opencood.models.sub_modules.pact_cbea_packet import PACTPacketResidualFusion, PACTPacketCommunicationMeter
    model.pact_packet_residual_fusion = PACTPacketResidualFusion(0.05, 0.3)
    model.pact_packet_comm_meter = PACTPacketCommunicationMeter(100)
    model._freeze_nonpacket_parameters()
    model._set_frozen_modules_eval()
    model.packet_trainable_summary = model._summarize_trainable_parameters()
    return model


def _check_train_only_packet(model):
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == model._is_packet_module_name(name), name
    assert model.encoder_m1.bn.training is False
    assert model.packet_trainable_summary["train_only_packet"] is True
    assert model.packet_trainable_summary["frozen_bn_eval_count"] >= 4
    assert model.packet_trainable_summary["packet_bn_train_count"] == 0


def _assert_packet_only_output(output_dict):
    assert output_dict["cls_preds"].shape == (2, 2, 8, 8)
    debug = output_dict["pact_packet_debug"]
    assert debug["packet_only_verified"] is True
    assert debug["dense_collab_fusion_used"] is False
    assert debug["collaborator_dense_after_packet_used"] is False
    assert debug["packet_source"] == PACKET_SOURCE
    assert debug["packet_count"] > 0
    assert debug["packet_bytes_per_frame"] > 0
    assert debug["failure_policy"] == "error"


def _check_model_backward(model, output_dict):
    loss = output_dict["cls_preds"].mean() + output_dict["reg_preds"].abs().mean()
    loss.backward()
    packet_grads = [
        parameter.grad for name, parameter in model.named_parameters()
        if model._is_packet_module_name(name)
    ]
    assert any(
        grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        for grad in packet_grads
    )
    frozen_grads = [
        parameter.grad for name, parameter in model.named_parameters()
        if not model._is_packet_module_name(name)
    ]
    assert all(grad is None for grad in frozen_grads)


def _check_multi_scene_output(output_dict):
    assert output_dict["cls_preds"].shape[0] == 2
    assert output_dict["pact_packet_debug"]["packet_count"] == 60


def _check_model_empty_packet(model):
    data = {
        "agent_modality_list": ["m1"],
        "record_len": torch.tensor([1]),
        "inputs_m1": {"feature": torch.randn(1, 64, 8, 8)},
    }
    output = model(data)
    assert output["cls_preds"].shape == (1, 2, 8, 8)
    assert output["pact_packet_debug"]["packet_count"] == 0
    assert output["pact_packet_debug"]["packet_aggregation"]["empty_packet"] is True


def _check_strict_failures(model):
    try:
        HeterPyramidCollabPactCbeaPacket._normalize_packet_cfg({
            "packet_failure_policy": "dense_feature",
        })
    except ValueError:
        pass
    else:
        raise AssertionError("dense_feature fallback must be rejected")
    try:
        model._assert_packet_boundary({
            "packet_only_verified": True,
            "dense_collab_fusion_used": True,
            "collaborator_dense_after_packet_used": False,
        })
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict boundary must reject dense fusion")


def _check_ego_only_failure_policy(hypes):
    model = _build_dummy_packet_model(hypes)
    model.packet_only_strict = False
    model.packet_failure_policy = "ego_only"
    ego = torch.randn(1, 64, 8, 8)
    output = model._handle_packet_failure("synthetic_packet_failure", ego, [])
    debug = output["pact_packet_debug"]
    assert debug["fallback_reason"] == "synthetic_packet_failure"
    assert debug["dense_collab_fusion_used"] is False
    assert debug["collaborator_dense_after_packet_used"] is False


def _check_checkpoint_compatibility(hypes):
    source = _build_dummy_packet_model(hypes)
    base_state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if not source._is_packet_module_name(key)
    }
    target = _build_dummy_packet_model(hypes)
    target.load_state_dict(base_state, strict=False)
    report = target.packet_checkpoint_report
    assert report["allowed_missing_packet_keys"]
    assert not report["unexpected_missing_keys"]
    assert not report["unexpected_checkpoint_keys"]
    bad_state = dict(base_state)
    bad_state["hvp_cbea_feature_mode.weight"] = torch.ones(1)
    try:
        target.load_state_dict(bad_state, strict=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("HVP feature-mode checkpoint key must be rejected")
    incomplete_state = dict(base_state)
    incomplete_state.pop("cls_head.weight")
    try:
        target.load_state_dict(incomplete_state, strict=False)
    except RuntimeError:
        return
    raise AssertionError("missing non-packet checkpoint key must be rejected")


def _dummy_data():
    modalities = ["m1", "m2", "m3", "m2", "m4"]
    counts = {name: modalities.count(name) for name in set(modalities)}
    data = {
        "agent_modality_list": modalities,
        "record_len": torch.tensor([3, 2]),
    }
    for offset, (modality_name, count) in enumerate(counts.items()):
        data["inputs_%s" % modality_name] = {
            "feature": torch.randn(count, 64, 8, 8) + 0.1 * offset,
        }
    return data


if __name__ == "__main__":
    main()

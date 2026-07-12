"""CPU smoke test for no-joint PACT top-K evidence packet inference."""

import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml import yaml_utils
from opencood.models.heter_pyramid_collab_pact_cbea_packet_nojoint import (
    HeterPyramidCollabPactCbeaPacketNojoint,
)
from opencood.models.sub_modules.pact_cbea_evidence_head import PACTCBEALocalEvidenceHead
from opencood.models.sub_modules.pact_cbea_packet_nojoint import (
    PACKET_SOURCE,
    PACTNoJointCommunicationMeter,
    PACTNoJointPacketAggregator,
    PACTNoJointPacketizer,
)


YAML_PATH = os.path.join(
    REPO_ROOT,
    "opencood",
    "hypes_yaml",
    "PACT_CBEA_PACKET_NOJOINT_v1",
    "rule_packet.yaml",
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
        raise AssertionError("no-joint packet model must not call forward_collab")


def main():
    torch.manual_seed(211)
    hypes = yaml_utils.load_yaml(YAML_PATH)
    _assert_yaml(hypes)
    _check_parameter_free_modules()

    model = _build_dummy_model(hypes)
    model.train()
    model.model_train_init()
    _check_frozen_model(model)
    output = model(_dummy_data())
    _assert_forward(output)
    _check_python38_packet_merge(model)
    _check_packet_failure_fallback(model)
    _check_empty_packet(model)
    _check_failure_policy()
    _check_checkpoint_loading(hypes)

    print("PACT no-joint YAML load OK")
    print("PACT no-joint packetizer parameter-free OK")
    print("PACT no-joint aggregator parameter-free OK")
    print("PACT no-joint top-K packet OK")
    print("PACT no-joint confidence uncertainty sampling OK")
    print("PACT no-joint Python 3.8 packet merge OK")
    print("PACT no-joint coordinate mapping OK")
    print("PACT no-joint multi-scene record_len OK")
    print("PACT no-joint ego-first selection OK")
    print("PACT no-joint empty packet ego-only OK")
    print("PACT no-joint packet failure ego-only fallback OK")
    print("PACT no-joint frozen BN eval OK")
    print("PACT no-joint checkpoint load OK")
    print("PACT no-joint communication boundary OK")
    print("PACT no-joint smoke OK")


def _assert_yaml(hypes):
    assert hypes["name"] == "PACT_CBEA_PACKET_NOJOINT_v1/rule_packet"
    assert hypes["model"]["core_method"] == "heter_pyramid_collab_pact_cbea_packet_nojoint"
    cfg = hypes["model"]["args"]["pact_packet_nojoint"]
    assert cfg["enabled"] is True
    assert cfg["no_joint_training"] is True
    assert cfg["use_stage3_joint_training"] is False
    assert cfg["trainable"] is False
    assert cfg["packet_only_strict"] is True
    assert cfg["failure_policy"] == "ego_only"
    assert cfg["packet_source"] == PACKET_SOURCE
    assert cfg["use_descriptor"] is False


def _check_parameter_free_modules():
    packetizer = PACTNoJointPacketizer(topk=4, confidence_threshold=0.15)
    aggregator = PACTNoJointPacketAggregator({"m1": 1.0}, fixed_gain=0.1)
    meter = PACTNoJointCommunicationMeter("fp16", 100, 8)
    assert sum(parameter.numel() for parameter in packetizer.parameters()) == 0
    assert sum(parameter.numel() for parameter in aggregator.parameters()) == 0
    assert sum(parameter.numel() for parameter in meter.parameters()) == 0
    heatmap = torch.zeros(1, 1, 4, 4)
    uncertainty = torch.ones(1, 1, 4, 4)
    heatmap[0, 0, 2, 3] = 0.9
    uncertainty[0, 0, 2, 3] = 0.25
    packet = packetizer(heatmap, uncertainty, "m2", 3, torch.eye(2, 3))
    assert packet["coordinates"].shape == (4, 2)
    assert torch.allclose(packet["confidence"][0], torch.tensor([0.9]), atol=1e-3)
    assert torch.allclose(packet["uncertainty"][0], torch.tensor([0.25]), atol=1e-3)
    ego = torch.ones(1, 2, 4, 4)
    enhanced, evidence_map, debug = aggregator(ego, [packet])
    assert debug["valid_packet_count"] > 0
    assert enhanced[0, 0, 2, 3] > ego[0, 0, 2, 3]
    assert evidence_map.shape == (1, 1, 4, 4)
    higher_uncertainty = dict(packet)
    higher_uncertainty["uncertainty"] = packet["uncertainty"].clone()
    higher_uncertainty["uncertainty"][0] = 4.0
    _, higher_uncertainty_map, _ = aggregator(ego, [higher_uncertainty])
    assert higher_uncertainty_map[0, 0, 2, 3] < evidence_map[0, 0, 2, 3]
    stats = meter([packet])
    assert stats["bytes_per_packet"] == 10


def _build_dummy_model(hypes):
    channels = 64
    model = HeterPyramidCollabPactCbeaPacketNojoint.__new__(HeterPyramidCollabPactCbeaPacketNojoint)
    nn.Module.__init__(model)
    model.modality_name_list = ["m1", "m2", "m3", "m4"]
    model.sensor_type_dict = {name: "lidar" for name in model.modality_name_list}
    model.H = 8
    model.W = 8
    model.fake_voxel_size = 1
    model.shrink_flag = False
    for name in model.modality_name_list:
        setattr(model, "encoder_%s" % name, _DummyEncoder())
        setattr(model, "backbone_%s" % name, _DummyBackbone())
        setattr(model, "aligner_%s" % name, _DummyAligner())
        setattr(model, "depth_supervision_%s" % name, False)
        setattr(model, "pact_cbea_evidence_head_%s" % name, PACTCBEALocalEvidenceHead(
            in_channels=channels,
            hidden_dim=16,
            descriptor_dim=8,
            return_feature=False,
        ))
    model.pyramid_backbone = _DummyPyramid()
    model.cls_head = nn.Conv2d(channels, 2, kernel_size=1)
    model.reg_head = nn.Conv2d(channels, 14, kernel_size=1)
    model.dir_head = nn.Conv2d(channels, 4, kernel_size=1)
    model.pact_packet_nojoint_cfg = HeterPyramidCollabPactCbeaPacketNojoint._normalize_cfg({
        **hypes["model"]["args"]["pact_packet_nojoint"],
        "topk": 6,
        "evidence_head": {
            "in_channels": channels,
            "hidden_dim": 16,
            "descriptor_dim": 8,
        },
    })
    model.pact_packet_nojoint_packetizer = PACTNoJointPacketizer(6, 0.0, "fp16")
    model.pact_packet_nojoint_aggregator = PACTNoJointPacketAggregator(
        model.pact_packet_nojoint_cfg["modality_prior"], 0.1, "max"
    )
    model.pact_packet_nojoint_comm_meter = PACTNoJointCommunicationMeter("fp16", 100, 8)
    model._freeze_all_parameters_and_eval()
    model.pact_packet_nojoint_summary = model._parameter_summary()
    return model


def _check_frozen_model(model):
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.encoder_m1.bn.training is False
    assert model.pact_cbea_evidence_head_m1.stem[1].training is False
    assert model._packet_parameter_count() == 0
    assert model.pact_packet_nojoint_summary["trainable_total"] == 0


def _assert_forward(output):
    assert output["cls_preds"].shape == (2, 2, 8, 8)
    debug = output["pact_packet_nojoint_debug"]
    for key, expected in (
        ("packet_only_verified", True),
        ("no_joint_training_verified", True),
        ("stage3_training_required", False),
        ("dense_collab_fusion_used", False),
        ("collaborator_dense_after_packet_used", False),
        ("full_evidence_map_transmitted", False),
    ):
        assert debug[key] is expected
    assert debug["packet_source"] == PACKET_SOURCE
    assert debug["packet_count"] > 0
    assert debug["packet_parameter_count"] == 0
    assert debug["trainable_total"] == 0


def _check_empty_packet(model):
    output = model({
        "agent_modality_list": ["m1"],
        "record_len": torch.tensor([1]),
        "pairwise_t_matrix": torch.eye(4).view(1, 1, 1, 4, 4),
        "inputs_m1": {"feature": torch.randn(1, 64, 8, 8)},
    })
    debug = output["pact_packet_nojoint_debug"]
    assert debug["packet_count"] == 0
    assert debug["packet_aggregation"]["empty_packet"] is True


def _check_python38_packet_merge(model):
    reference = torch.zeros(1, 64, 8, 8)
    packet = model.pact_packet_nojoint_packetizer(
        torch.ones(1, 1, 8, 8),
        torch.zeros(1, 1, 8, 8),
        "m2",
        1,
        torch.eye(2, 3),
    )
    merged = model._merge_packets([packet, packet], reference)
    assert merged["packet_source"] == PACKET_SOURCE
    assert merged["coordinates"].shape[0] == 2 * packet["coordinates"].shape[0]


def _check_packet_failure_fallback(model):
    data = _dummy_data()
    with torch.no_grad():
        raw_features = model._encode_agent_features(data, {})
        ego_feature, _ = model._extract_ego_local_features(
            raw_features, data["record_len"]
        )
        expected_cls = model.cls_head(ego_feature)

    original_merge = model._merge_packets

    def _raise_forced_merge_error(*unused_args):
        raise RuntimeError("forced packet merge failure")

    model._merge_packets = _raise_forced_merge_error
    try:
        output = model(data)
    finally:
        model._merge_packets = original_merge

    debug = output["pact_packet_nojoint_debug"]
    assert torch.allclose(output["cls_preds"], expected_cls)
    assert debug["fallback_reason"]
    assert "forced packet merge failure" in debug["fallback_reason"]
    assert "UnboundLocalError" not in debug["fallback_reason"]
    assert debug["packet_only_verified"] is True
    assert debug["dense_collab_fusion_used"] is False
    assert debug["collaborator_dense_after_packet_used"] is False


def _check_failure_policy():
    try:
        HeterPyramidCollabPactCbeaPacketNojoint._normalize_cfg({"failure_policy": "dense_feature"})
    except ValueError:
        return
    raise AssertionError("dense feature fallback must be rejected")


def _check_checkpoint_loading(hypes):
    source = _build_dummy_model(hypes)
    target = _build_dummy_model(hypes)
    target.load_state_dict({key: value.detach().clone() for key, value in source.state_dict().items()})
    report = target.pact_packet_nojoint_checkpoint_report
    assert not report["unexpected_missing_keys"]
    assert not report["unexpected_checkpoint_keys"]
    assert report["packet_parameter_count"] == 0


def _dummy_data():
    modalities = ["m1", "m2", "m3", "m1", "m4"]
    counts = {name: modalities.count(name) for name in set(modalities)}
    data = {
        "agent_modality_list": modalities,
        "record_len": torch.tensor([3, 2]),
        "pairwise_t_matrix": torch.eye(4).view(1, 1, 1, 4, 4).repeat(2, 3, 3, 1, 1),
    }
    for offset, (name, count) in enumerate(counts.items()):
        data["inputs_%s" % name] = {"feature": torch.randn(count, 64, 8, 8) + offset * 0.1}
    return data


if __name__ == "__main__":
    main()

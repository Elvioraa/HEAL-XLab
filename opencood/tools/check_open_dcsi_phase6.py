"""CPU smoke checks for Open-DCSI Phase 6 packet communication."""

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from opencood.tools.audit_open_dcsi_baseline_parity import _collab_input
from opencood.tools.check_open_dcsi_phase4 import _enabled_args as _phase4_args
from opencood.models.heter_pyramid_collab_open_dcsi_stage1 import (
    HeterPyramidCollabOpenDcsiStage1,
)
from opencood.models.sub_modules.open_dcsi.packet_codec import (
    CommunicationManager,
    TensorCodec,
    TokenCodec,
    encode_tensor_packet,
)


def _communication_args(budget_mode="dense_ratio", budget_value=1.0):
    args = _phase4_args()
    budget = {"enabled": True, "mode": budget_mode}
    if budget_mode == "dense_ratio":
        budget["ratio_of_dense"] = budget_value
    elif budget_mode == "bytes":
        budget["bytes_per_frame"] = int(budget_value)
    else:
        budget["fixed_tokens"] = int(budget_value)
    args["open_dcsi"]["communication"] = {
        "enabled": True,
        "common_codec": {
            "enabled": True,
            "precision": "int8",
            "per_channel_scale": True,
        },
        "token_codec": {"enabled": True, "precision": "int8"},
        "budget": budget,
        "selection": {
            "enabled": True,
            "score": "quality_adjusted_innovation",
            "allow_negative_reject": True,
            "deterministic_inference": True,
        },
    }
    return args


def _check_tensor_packets():
    tensor = torch.linspace(-2.0, 2.0, 2 * 4 * 3 * 3).reshape(2, 4, 3, 3)
    for precision in ("fp16", "int8"):
        codec = TensorCodec(
            {
                "precision": precision,
                "per_channel_scale": precision == "int8",
            }
        ).eval()
        decoded, packet = codec(tensor)
        assert decoded.shape == tensor.shape and torch.isfinite(decoded).all()
        serialized = packet.to_bytes()
        assert len(serialized) == packet.nbytes
        assert packet.metadata_bytes > 0 and packet.payload_bytes > 0
        if precision == "int8":
            assert (decoded - tensor).abs().max() < 0.03
    print("[phase6] FP16/INT8 tensor serialization and decode OK")


def _check_token_packet(model_output):
    tokens = model_output["open_dcsi"]["innovation_tokens"]
    collaborator = torch.nonzero(
        tokens["agent_local_index"] != 0, as_tuple=False
    ).flatten()
    if collaborator.numel() == 0:
        return
    index = collaborator[:1]
    count = int(tokens["scenario_index"].numel())
    row = {
        key: value.index_select(0, index)
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count
        else value
        for key, value in tokens.items()
    }
    decoded, packet = TokenCodec({"precision": "int8"})(row)
    assert len(packet.to_bytes()) == packet.nbytes
    assert packet.metadata_bytes > 0
    assert decoded["innovation_embedding"].shape == row["innovation_embedding"].shape
    print("[phase6] token payload, indices, quantization metadata, and header OK")


def _check_model_communication_and_backward():
    torch.manual_seed(31)
    model = HeterPyramidCollabOpenDcsiStage1(_communication_args()).train()
    output = model(_collab_input())
    stats = output["open_dcsi"]["communication_stats"]
    assert stats["dense_baseline_bytes"] > 0
    assert stats["common_payload_bytes"] > 0
    assert stats["metadata_bytes"] > 0
    assert stats["total_bytes"] == (
        stats["common_payload_bytes"]
        + stats["token_payload_bytes"]
        + stats["metadata_bytes"]
    )
    assert stats["compression_ratio"] > 0
    _check_token_packet(output)
    loss = output["cls_preds"].square().mean()
    tokens = output["open_dcsi"]["innovation_tokens"]
    if tokens["innovation_embedding"].numel() > 0:
        loss = loss + tokens["innovation_embedding"].square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("open_dcsi.") and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    print("[phase6] integrated codec/budget forward and STE backward OK")


def _one_agent_input():
    data = _collab_input()
    data["agent_modality_list"] = ["m1"]
    data["record_len"] = torch.tensor([1])
    data["pairwise_t_matrix"] = torch.eye(4).reshape(1, 1, 1, 4, 4)
    data["inputs_m1"]["spatial_features"] = data["inputs_m1"][
        "spatial_features"
    ][:1]
    return data


def _check_budget_and_cav1():
    tight = HeterPyramidCollabOpenDcsiStage1(
        _communication_args("bytes", 32)
    ).eval()
    with torch.no_grad():
        tight_output = tight(_collab_input())
    tight_stats = tight_output["open_dcsi"]["communication_stats"]
    assert tight_stats["selected_common_packets"] == 0
    assert tight_stats["selected_tokens"] == 0
    assert tight_stats["total_bytes"] == 0
    tight_tokens = tight_output["open_dcsi"]["innovation_tokens"]
    assert torch.all(tight_tokens["agent_local_index"] == 0)

    cav1 = HeterPyramidCollabOpenDcsiStage1(_communication_args()).eval()
    with torch.no_grad():
        cav1_output = cav1(_one_agent_input())
    cav1_stats = cav1_output["open_dcsi"]["communication_stats"]
    assert cav1_stats["dense_baseline_bytes"] == 0
    assert cav1_stats["total_bytes"] == 0
    assert cav1_stats["packet_count"] == 0
    print("[phase6] byte budget rejection and CAV1 zero communication OK")


def main():
    _check_tensor_packets()
    _check_model_communication_and_backward()
    _check_budget_and_cav1()
    print("OPEN_DCSI_PHASE6_PASS")


if __name__ == "__main__":
    main()

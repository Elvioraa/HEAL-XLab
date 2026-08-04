"""CPU smoke checks for Open-DCSI Phase 7 streaming fusion."""

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from opencood.tools.audit_open_dcsi_baseline_parity import _collab_input
from opencood.tools.check_open_dcsi_phase6 import (
    _communication_args,
    _one_agent_input,
)
from opencood.models.heter_pyramid_collab_open_dcsi_stage1 import (
    HeterPyramidCollabOpenDcsiStage1,
)


def _streaming_args():
    args = _communication_args("dense_ratio", 1.0)
    args["open_dcsi"]["streaming_fusion"] = {
        "enabled": True,
        "inference_only": True,
        "process_agent_sequentially": True,
        "release_packet_after_fusion": True,
        "recover_full_dense_stack": False,
        "numerical_parity_test": True,
    }
    return args


def _compare_outputs(dense, streaming, label):
    for key in ("cls_preds", "reg_preds", "dir_preds"):
        difference = (dense[key] - streaming[key]).abs()
        max_abs = float(difference.max())
        assert torch.allclose(dense[key], streaming[key], atol=1e-6, rtol=1e-6), (
            label,
            key,
            max_abs,
        )
    dense_tokens = dense["open_dcsi"]["fused_tokens"]
    stream_tokens = streaming["open_dcsi"]["fused_tokens"]
    assert torch.equal(
        dense_tokens["scenario_index"], stream_tokens["scenario_index"]
    )
    print("[phase7] {} dense/streaming numerical parity OK".format(label))


def _models():
    torch.manual_seed(37)
    dense = HeterPyramidCollabOpenDcsiStage1(_communication_args()).eval()
    streaming = HeterPyramidCollabOpenDcsiStage1(_streaming_args()).eval()
    streaming.load_state_dict(dense.state_dict(), strict=True)
    dense.open_dcsi.common_residual_gate.data.fill_(0.7)
    streaming.open_dcsi.common_residual_gate.data.fill_(0.7)
    return dense, streaming


def _multi_scene_input():
    base = _collab_input()
    feature = base["inputs_m1"]["spatial_features"]
    base["inputs_m1"]["spatial_features"] = torch.cat(
        (feature[:1], feature), dim=0
    )
    base["agent_modality_list"] = ["m1", "m1", "m1"]
    base["record_len"] = torch.tensor([1, 2])
    pairwise = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(2, 2, 2, 1, 1)
    base["pairwise_t_matrix"] = pairwise
    return base


def _check_eval_parity_and_stats():
    dense_model, stream_model = _models()
    with torch.no_grad():
        dense = dense_model(_collab_input())
        streaming = stream_model(_collab_input())
    _compare_outputs(dense, streaming, "CAV2")
    stats = streaming["open_dcsi"]["streaming_stats"]
    assert stats["processed_common_packets"] > 0
    assert stats["fusion_local_peak_bytes"] > 0
    assert stats["recovered_full_dense_stack"] is False
    assert all(
        item["recovered_full_dense_stack"] is False
        for item in stats["scale_stats"]
    )

    with torch.no_grad():
        dense_multi = dense_model(_multi_scene_input())
        stream_multi = stream_model(_multi_scene_input())
    _compare_outputs(dense_multi, stream_multi, "record_len=[1,2]")
    assert tuple(stream_multi["cls_preds"].shape[:2]) == (2, 2)
    print("[phase7] scene-local accumulators and no dense recovery audit OK")


def _check_training_dense_fallback_and_cav1():
    model = HeterPyramidCollabOpenDcsiStage1(_streaming_args()).train()
    output = model(_collab_input())
    assert output["open_dcsi"]["streaming_stats"] is None
    loss = output["cls_preds"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)

    model.eval()
    with torch.no_grad():
        cav1 = model(_one_agent_input())
    stats = cav1["open_dcsi"]["streaming_stats"]
    assert stats["processed_common_packets"] == 0
    assert cav1["open_dcsi"]["communication_stats"]["total_bytes"] == 0
    print("[phase7] training dense fallback and CAV1 streaming identity OK")


def main():
    _check_eval_parity_and_stats()
    _check_training_dense_fallback_and_cav1()
    print("OPEN_DCSI_PHASE7_PASS")


if __name__ == "__main__":
    main()

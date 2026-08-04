"""CPU smoke checks for Open-DCSI Phase 2 common-space implementation."""

from copy import deepcopy
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from opencood.tools.audit_open_dcsi_baseline_parity import (
    _collab_input,
    _single_input,
    _tiny_model_args,
)
from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.heter_pyramid_collab_open_dcsi_stage1 import (
    HeterPyramidCollabOpenDcsiStage1,
)
from opencood.models.heter_pyramid_single_open_dcsi_stage2 import (
    HeterPyramidSingleOpenDcsiStage2,
)


def _phase2_config(absolute_reject=False):
    return {
        "enabled": True,
        "open_heterogeneous": {"enabled": True},
        "stage2_independent": {"enabled": True},
        "common_space": {
            "enabled": True,
            "projector": {"enabled": True},
            "decoder": {"enabled": True},
            "reconstruction": {"enabled": True},
            "common_detection_supervision": {"enabled": True},
        },
        "common_fusion": {
            "enabled": True,
            "absolute_reject": {
                "enabled": absolute_reject,
                "threshold": 0.9,
            },
        },
        "diagnostics": {"enabled": True},
    }


def _enabled_args(absolute_reject=False):
    args = _tiny_model_args("missing")
    args["open_dcsi"] = _phase2_config(absolute_reject)
    return args


def _identity_affine(agent_count):
    affine = torch.zeros(1, agent_count, agent_count, 2, 3)
    affine[..., 0, 0] = 1.0
    affine[..., 1, 1] = 1.0
    return affine


def _check_enabled_forward_and_backward():
    torch.manual_seed(7)
    model = HeterPyramidCollabOpenDcsiStage1(_enabled_args()).train()
    output = model(_collab_input())
    assert "open_dcsi" in output
    open_output = output["open_dcsi"]
    assert open_output["config_summary"]["common_channels"] == [16, 32]
    assert [tuple(item.shape) for item in open_output["common_features"]] == [
        (2, 16, 8, 8),
        (2, 32, 4, 4),
    ]
    assert [tuple(item.shape) for item in open_output["reconstructed_features"]] == [
        (2, 64, 8, 8),
        (2, 128, 4, 4),
    ]
    assert tuple(output["cls_preds"].shape) == (1, 2, 8, 8)
    loss = output["cls_preds"].square().mean()
    for reconstruction in open_output["reconstructed_features"]:
        loss = loss + reconstruction.square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("open_dcsi.") and parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    parent = HeterPyramidCollab(deepcopy(_tiny_model_args("missing")))
    parent_keys = set(parent.state_dict())
    added_keys = set(model.state_dict()) - parent_keys
    assert added_keys and all(key.startswith("open_dcsi.") for key in added_keys)
    print("[phase2] common forward/backward and state namespace OK")


def _check_single_agent_identity():
    model = HeterPyramidCollabOpenDcsiStage1(_enabled_args()).eval()
    fusion = model.open_dcsi.common_fusions[0].eval()
    common = torch.randn(1, 16, 5, 5)
    score = torch.rand(1, 1, 5, 5)
    fused, weights = fusion(
        common,
        score,
        torch.tensor([1]),
        _identity_affine(1),
        model.open_dcsi.common_residual_gate[0],
    )
    assert torch.equal(fused, common)
    assert len(weights) == 1
    print("[phase2] CAV1 common fusion is strict ego identity")


def _check_permutation_and_absolute_reject():
    model = HeterPyramidCollabOpenDcsiStage1(_enabled_args()).eval()
    fusion = model.open_dcsi.common_fusions[0].eval()
    model.open_dcsi.common_residual_gate.data.fill_(10.0)
    common = torch.stack(
        (torch.zeros(16, 3, 3), torch.ones(16, 3, 3), torch.full((16, 3, 3), 3.0))
    )
    score = torch.ones(3, 1, 3, 3)
    record_len = torch.tensor([3])
    affine = _identity_affine(3)
    first, _ = fusion(
        common,
        score,
        record_len,
        affine,
        model.open_dcsi.common_residual_gate[0],
    )
    permuted, _ = fusion(
        common[[0, 2, 1]],
        score[[0, 2, 1]],
        record_len,
        affine,
        model.open_dcsi.common_residual_gate[0],
    )
    assert torch.equal(first, permuted)

    reject_model = HeterPyramidCollabOpenDcsiStage1(
        _enabled_args(absolute_reject=True)
    ).eval()
    reject_model.open_dcsi.common_residual_gate.data.fill_(10.0)
    reject_fusion = reject_model.open_dcsi.common_fusions[0].eval()
    rejected, _ = reject_fusion(
        common[:2],
        torch.full((2, 1, 3, 3), 0.1),
        torch.tensor([2]),
        _identity_affine(2),
        reject_model.open_dcsi.common_residual_gate[0],
    )
    assert torch.equal(rejected, common[:1])
    print("[phase2] permutation invariance and absolute rejection OK")


def _check_stage2_single():
    model = HeterPyramidSingleOpenDcsiStage2(_enabled_args()).eval()
    with torch.no_grad():
        output = model(_single_input())
    assert tuple(output["cls_preds"].shape) == (2, 2, 8, 8)
    assert len(output["open_dcsi"]["common_features"]) == 2
    print("[phase2] independent Stage2 common path forward OK")


def main():
    _check_enabled_forward_and_backward()
    _check_single_agent_identity()
    _check_permutation_and_absolute_reject()
    _check_stage2_single()
    print("OPEN_DCSI_PHASE2_PASS")


if __name__ == "__main__":
    main()

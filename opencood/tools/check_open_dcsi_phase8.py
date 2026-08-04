"""CPU smoke checks for Open-DCSI Phase 8 composable loss."""

from copy import deepcopy
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from opencood.tools.audit_open_dcsi_baseline_parity import _collab_input
from opencood.tools.check_open_dcsi_phase4 import _enabled_args
from opencood.models.heter_pyramid_collab_open_dcsi_stage1 import (
    HeterPyramidCollabOpenDcsiStage1,
)


class AuditOfficialLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, suffix=""):
        value = output_dict["cls_preds"].square().mean()
        self.loss_dict = {
            "cls_loss": float(value.detach()),
            "total_loss": float(value.detach()),
        }
        return value

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        return None


audit_loss_module = types.ModuleType("opencood.loss.audit_official_loss")
audit_loss_module.AuditOfficialLoss = AuditOfficialLoss
sys.modules["opencood.loss.audit_official_loss"] = audit_loss_module

from opencood.loss.open_dcsi_loss import OpenDcsiLoss


_TERM_NAMES = (
    "common_detection",
    "reconstruction",
    "common_innovation_decorrelation",
    "innovation_detection",
    "box_refinement",
    "quality",
    "token_sparsity",
    "budget",
)


def _anchors():
    height = width = 8
    anchors = torch.zeros(height, width, 2, 7)
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    anchors[..., 0] = x[..., None] - width / 2
    anchors[..., 1] = y[..., None] - height / 2
    anchors[..., 2] = 0.0
    anchors[..., 3] = 1.56
    anchors[..., 4] = 1.6
    anchors[..., 5] = 3.9
    anchors[..., 6] = torch.tensor((0.0, 1.5707963))
    return anchors


def _target():
    positive = torch.zeros(1, 8, 8, 2)
    negative = torch.ones_like(positive)
    positive[0, 4, 4, 0] = 1.0
    negative[0, 4, 4, 0] = 0.0
    targets = torch.zeros(1, 8, 8, 14)
    return {
        "pos_equal_one": positive,
        "neg_equal_one": negative,
        "targets": targets,
    }


def _loss_config(model_open_config, enabled_terms):
    config = deepcopy(model_open_config)
    config["losses"] = {"enabled": True}
    for name in _TERM_NAMES:
        config["losses"][name] = {
            "enabled": name in enabled_terms,
            "weight": 1.0,
        }
    return {
        "official_loss": {
            "core_method": "audit_official_loss",
            "args": {},
        },
        "open_dcsi": config,
    }


def _model_output():
    args = _enabled_args()
    model = HeterPyramidCollabOpenDcsiStage1(args).train()
    data = _collab_input()
    data["anchor_box"] = _anchors()
    return model, args, model(data)


def _check_all_terms_and_backward():
    model, args, output = _model_output()
    criterion = OpenDcsiLoss(_loss_config(args["open_dcsi"], set(_TERM_NAMES)))
    total = criterion(output, _target())
    assert torch.isfinite(total)
    for name in _TERM_NAMES:
        assert "open_dcsi_{}_loss".format(name) in criterion.loss_dict
    assert criterion.loss_dict["open_dcsi_skipped_non_finite_count"] == 0
    total.backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("open_dcsi.") and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    criterion.logging(0, 0, 1)
    print("[phase8] all auxiliary terms, logging, and backward OK")
    return args, output


def _check_independent_switches(args, output):
    target = _target()
    for name in _TERM_NAMES:
        criterion = OpenDcsiLoss(_loss_config(args["open_dcsi"], {name}))
        total = criterion(output, target)
        assert torch.isfinite(total), name
        enabled_keys = [
            key
            for key in criterion.loss_dict
            if key.startswith("open_dcsi_") and key.endswith("_loss")
        ]
        assert "open_dcsi_{}_loss".format(name) in enabled_keys
    print("[phase8] every Open-DCSI loss term is independently switchable")


def _clone_output(output):
    result = dict(output)
    result["open_dcsi"] = dict(output["open_dcsi"])
    for key in ("reconstructed_features", "innovation_features"):
        result["open_dcsi"][key] = [tensor.clone() for tensor in output["open_dcsi"][key]]
    return result


def _check_non_finite_isolation_and_disabled_parity(args, output):
    corrupted = _clone_output(output)
    corrupted["open_dcsi"]["reconstructed_features"][0][0, 0, 0, 0] = float("nan")
    criterion = OpenDcsiLoss(_loss_config(args["open_dcsi"], {"reconstruction"}))
    total = criterion(corrupted, _target())
    official_value = corrupted["cls_preds"].square().mean()
    assert torch.equal(total, official_value)
    assert criterion.loss_dict["open_dcsi_skipped_non_finite_count"] == 1

    disabled_config = _loss_config(args["open_dcsi"], set())
    disabled_config["open_dcsi"] = {"enabled": False}
    disabled = OpenDcsiLoss(disabled_config)
    disabled_total = disabled(output, _target())
    assert torch.equal(disabled_total, output["cls_preds"].square().mean())
    assert not any(key.startswith("open_dcsi_") for key in disabled.loss_dict)
    print("[phase8] non-finite isolation and disabled official-loss parity OK")


def main():
    args, output = _check_all_terms_and_backward()
    _check_independent_switches(args, output)
    _check_non_finite_isolation_and_disabled_parity(args, output)
    print("OPEN_DCSI_PHASE8_PASS")


if __name__ == "__main__":
    main()

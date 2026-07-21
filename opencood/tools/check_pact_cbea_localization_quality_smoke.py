"""CPU smoke tests for the localization-quality-aware CBEA extension.

Covers, all opt-in / default-off:
- HvpHealV3EvidenceHead's new `evidence_localization_uncertainty` output.
- PACTCBEARule's new `localization_weight` aggregation term.
- compute_pact_cbea_local_evidence_loss's new `localization_uncertainty` term.
- HeterPyramidCollabPactCbeaStage1's config normalization for the new fields.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opencood.models.hvp_heal_v3.evidence_head import HvpHealV3EvidenceHead
from opencood.models.sub_modules.pact_cbea_rule import PACTCBEARule
from opencood.models.heter_pyramid_collab_pact_cbea_stage1 import (
    HeterPyramidCollabPactCbeaStage1,
)
from opencood.loss.hvp_cbea_aux_loss import (
    compute_pact_cbea_local_evidence_loss,
    _regression_residual_map,
)


def test_evidence_head_default_unchanged():
    torch.manual_seed(0)
    head_default = HvpHealV3EvidenceHead(in_channels=8, hidden_dim=4, descriptor_dim=2)
    torch.manual_seed(0)
    head_explicit_off = HvpHealV3EvidenceHead(
        in_channels=8, hidden_dim=4, descriptor_dim=2,
        predict_localization_uncertainty=False,
    )
    feature = torch.randn(1, 8, 5, 5)
    out_default = head_default(feature)
    out_explicit_off = head_explicit_off(feature)
    for key in out_default:
        assert torch.equal(out_default[key], out_explicit_off[key])
    assert "evidence_localization_uncertainty" not in out_default
    assert not hasattr(head_default, "evidence_localization_uncertainty_head")
    print("evidence head default-off unchanged: True")


def test_evidence_head_localization_uncertainty_enabled():
    head = HvpHealV3EvidenceHead(
        in_channels=8, hidden_dim=4, descriptor_dim=2,
        predict_localization_uncertainty=True,
    )
    assert hasattr(head, "evidence_localization_uncertainty_head")
    feature = torch.randn(2, 8, 5, 6)
    out = head(feature)
    assert "evidence_localization_uncertainty" in out
    assert out["evidence_localization_uncertainty"].shape == (2, 1, 5, 6)
    assert bool((out["evidence_localization_uncertainty"] > 0).all())
    print("evidence head localization_uncertainty enabled: shape True; positive: True")


def test_rule_localization_weight_default_off_identity():
    torch.manual_seed(1)
    rule = PACTCBEARule({"aggregation": {"localization_weight": False}})
    feature = torch.randn(1, 3, 4, 5, 5)
    heatmap = torch.randn(1, 3, 1, 5, 5)
    uncertainty = torch.rand(1, 3, 1, 5, 5)
    loc_uncertainty = torch.rand(1, 3, 1, 5, 5)
    out_without, _ = rule(feature, evidence_heatmap=heatmap, evidence_uncertainty=uncertainty)
    out_with, _ = rule(
        feature, evidence_heatmap=heatmap, evidence_uncertainty=uncertainty,
        evidence_localization_uncertainty=loc_uncertainty,
    )
    assert torch.equal(out_without, out_with)
    print("rule localization_weight default-off identity: True")


def test_rule_localization_weight_missing_signal_fallback():
    rule = PACTCBEARule({"aggregation": {"localization_weight": True}})
    feature = torch.randn(1, 2, 3, 4, 4)
    heatmap = torch.randn(1, 2, 1, 4, 4)
    uncertainty = torch.rand(1, 2, 1, 4, 4)
    _, debug = rule(feature, evidence_heatmap=heatmap, evidence_uncertainty=uncertainty)
    assert "missing_evidence_localization_uncertainty_weight_ones" in debug["pact_fallbacks"]
    print("rule localization_weight missing-signal fallback: True")


def test_rule_localization_weight_shifts_alpha():
    rule = PACTCBEARule({"aggregation": {"localization_weight": True}})
    feature = torch.stack((
        torch.full((1, 3, 3), 10.0),
        torch.full((1, 3, 3), 10.0),
    )).unsqueeze(0)
    heatmap = torch.zeros(1, 2, 1, 3, 3)
    uncertainty = torch.zeros(1, 2, 1, 3, 3)
    loc_uncertainty = torch.stack((
        torch.zeros(1, 3, 3),
        torch.full((1, 3, 3), 5.0),
    ), dim=1)
    _, debug = rule(
        feature, evidence_heatmap=heatmap, evidence_uncertainty=uncertainty,
        evidence_localization_uncertainty=loc_uncertainty,
    )
    alpha = debug["pact_alpha"]
    good_mean = float(alpha[:, 0].mean())
    bad_mean = float(alpha[:, 1].mean())
    assert good_mean > bad_mean
    print(
        "rule localization_weight shifts alpha toward lower loc-uncertainty agent: True; "
        "good=%.4f bad=%.4f" % (good_mean, bad_mean)
    )


def test_regression_residual_map_shape_and_value():
    reg_preds = torch.zeros(1, 7, 2, 2)
    reg_targets_flat = torch.full((1, 4, 7), 0.1)
    ref_tensor = torch.zeros(1, 1, 2, 2)
    residual_map = _regression_residual_map(reg_preds, reg_targets_flat, ref_tensor)
    assert residual_map.shape == (1, 1, 2, 2)
    assert torch.allclose(residual_map, torch.full((1, 1, 2, 2), 0.1), atol=1e-6)
    assert not residual_map.requires_grad
    print("regression residual map shape/value/detached: True")


def test_pact_local_evidence_loss_localization_uncertainty():
    batch_size, height, width, anchors = 1, 2, 2, 1
    pos_equal_one = torch.zeros(batch_size, height, width, anchors)
    pos_equal_one[0, 0, 0, 0] = 1.0
    target_dict = {
        "pos_equal_one": pos_equal_one,
        "targets": torch.zeros(batch_size, height * width * anchors, 7),
    }
    reg_preds = torch.zeros(batch_size, anchors * 7, height, width)
    loc_uncertainty = torch.full((batch_size, 1, height, width), 0.5)
    pact_dict = {
        "enabled": True,
        "stage": "local_evidence",
        "evidence_heatmap_logits": torch.zeros(batch_size, 1, height, width),
        "evidence_uncertainty": torch.zeros(batch_size, 1, height, width),
        "evidence_localization_uncertainty": loc_uncertainty,
        "evidence_loss_cfg": {
            "enabled": True,
            "mode": "pact_local_evidence",
            "localization_uncertainty": {"enabled": True, "weight": 1.0},
        },
    }
    loss, stats = compute_pact_cbea_local_evidence_loss(
        pact_dict, target_dict=target_dict, reg_preds=reg_preds,
    )
    assert stats["pact_cbea_fallback_reason"] == ""
    assert abs(stats["pact_cbea_localization_uncertainty_loss"] - 0.5) < 1e-4
    assert torch.isfinite(loss).item()
    print(
        "pact local evidence loss localization_uncertainty term: True; loss=%.4f"
        % stats["pact_cbea_localization_uncertainty_loss"]
    )


def test_pact_local_evidence_loss_localization_uncertainty_default_off():
    batch_size, height, width = 1, 2, 2
    pact_dict = {
        "enabled": True,
        "stage": "local_evidence",
        "evidence_heatmap_logits": torch.zeros(batch_size, 1, height, width),
        "evidence_uncertainty": torch.zeros(batch_size, 1, height, width),
        "evidence_loss_cfg": {
            "enabled": True,
            "mode": "pact_local_evidence",
        },
    }
    target_dict = {
        "pos_equal_one": torch.zeros(batch_size, height, width, 1),
        "targets": torch.zeros(batch_size, height * width, 7),
    }
    loss, stats = compute_pact_cbea_local_evidence_loss(
        pact_dict, target_dict=target_dict, reg_preds=None,
    )
    assert stats["pact_cbea_localization_uncertainty_loss"] == 0.0
    assert stats["pact_cbea_fallback_reason"] == ""
    assert torch.isfinite(loss).item()
    print("pact local evidence loss localization_uncertainty default-off: True")


def test_pact_local_evidence_loss_localization_uncertainty_missing_reg_preds_fallback():
    batch_size, height, width = 1, 2, 2
    pact_dict = {
        "enabled": True,
        "stage": "local_evidence",
        "evidence_heatmap_logits": torch.zeros(batch_size, 1, height, width),
        "evidence_uncertainty": torch.zeros(batch_size, 1, height, width),
        "evidence_localization_uncertainty": torch.zeros(batch_size, 1, height, width),
        "evidence_loss_cfg": {
            "enabled": True,
            "mode": "pact_local_evidence",
            "localization_uncertainty": {"enabled": True, "weight": 1.0},
        },
    }
    target_dict = {
        "pos_equal_one": torch.zeros(batch_size, height, width, 1),
        "targets": torch.zeros(batch_size, height * width, 7),
    }
    loss, stats = compute_pact_cbea_local_evidence_loss(
        pact_dict, target_dict=target_dict, reg_preds=None, fallback_on_error=True,
    )
    assert stats["pact_cbea_fallback_reason"] != ""
    assert float(loss.item()) == 0.0
    print("pact local evidence loss missing reg_preds safe fallback: True")


def test_stage1_config_normalization_defaults_and_validation():
    default_cfg = HeterPyramidCollabPactCbeaStage1._normalize_pact_stage1_cfg({})
    assert default_cfg["evidence_head"]["localization_uncertainty"]["enabled"] is False
    assert default_cfg["evidence_loss"]["localization_uncertainty"]["enabled"] is False
    assert default_cfg["evidence_loss"]["localization_uncertainty"]["weight"] == 0.001

    configured = HeterPyramidCollabPactCbeaStage1._normalize_pact_stage1_cfg({
        "evidence_head": {"localization_uncertainty": {"enabled": True}},
        "evidence_loss": {
            "localization_uncertainty": {"enabled": True, "weight": "0.5"},
        },
    })
    assert configured["evidence_head"]["localization_uncertainty"]["enabled"] is True
    assert configured["evidence_loss"]["localization_uncertainty"]["enabled"] is True
    assert configured["evidence_loss"]["localization_uncertainty"]["weight"] == 0.5
    print("stage1 config normalization defaults/validation: True")


def main():
    test_evidence_head_default_unchanged()
    test_evidence_head_localization_uncertainty_enabled()
    test_rule_localization_weight_default_off_identity()
    test_rule_localization_weight_missing_signal_fallback()
    test_rule_localization_weight_shifts_alpha()
    test_regression_residual_map_shape_and_value()
    test_pact_local_evidence_loss_localization_uncertainty()
    test_pact_local_evidence_loss_localization_uncertainty_default_off()
    test_pact_local_evidence_loss_localization_uncertainty_missing_reg_preds_fallback()
    test_stage1_config_normalization_defaults_and_validation()
    print("PACT_CBEA_LOCALIZATION_QUALITY_SMOKE_PASS")


if __name__ == "__main__":
    main()

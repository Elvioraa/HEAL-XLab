"""Smoke test for HEAL-XLab-v2 HVP-CBEA modules."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.models.sub_modules.hypothesis_encoder import HypothesisEncoder
from opencood.models.sub_modules.hypothesis_verifier import HypothesisVerifier
from opencood.models.sub_modules.bayesian_hypothesis_fusion import BayesianHypothesisFusion
from opencood.loss.hvp_cbea_aux_loss import compute_hvp_auxiliary_loss


def main():
    torch.manual_seed(7)
    bev = torch.randn(2, 32, 20, 24, requires_grad=True)
    collab = torch.randn(3, 32, 20, 24)

    encoder = HypothesisEncoder(in_channels=32, mid_channels=16, max_hypotheses=8)
    verifier = HypothesisVerifier(in_channels=32, mid_channels=16, max_novel=4)
    fusion = BayesianHypothesisFusion(in_channels=32, mid_channels=16)

    hyps, hmap, reg = encoder(bev)
    assert hyps.shape == (2, 8, 9)
    assert hmap.shape == (2, 1, 20, 24)
    assert reg.shape == (2, 7, 20, 24)

    logits, delta, novel = verifier(collab, hyps)
    assert logits.shape == (2, 8, 3)
    assert delta.shape == (2, 8, 7)
    assert novel.shape[-1] == 9

    fused, updated = fusion(hyps, logits, delta, novel, bev)
    assert fused.shape == bev.shape
    assert updated.shape == hyps.shape
    residual_debug = fusion.get_residual_debug()
    assert residual_debug["hvp_cbea_residual_gate_enabled"]
    assert residual_debug["hvp_cbea_residual_fallback_reason"] == ""
    assert abs(residual_debug["hvp_cbea_residual_alpha"] - 0.05) < 1e-5
    assert residual_debug["hvp_cbea_residual_alpha"] <= 0.3
    assert residual_debug["hvp_cbea_delta_norm"] > 0.0
    assert isinstance(fusion.residual_alpha_logit, torch.nn.Parameter)

    loss = fused.sum()
    loss.backward()
    hvp_params = (
        list(encoder.parameters())
        + list(verifier.parameters())
        + list(fusion.parameters())
    )
    assert any(param.grad is not None for param in hvp_params)
    assert any(param.grad is not None for param in encoder.parameters())
    assert any(param.grad is not None for param in verifier.parameters())
    assert any(param.grad is not None for param in fusion.parameters())
    assert fusion.residual_alpha_logit.grad is not None
    assert torch.isfinite(fusion.residual_alpha_logit.grad).all()
    assert fusion.residual_alpha_logit.grad.detach().abs().sum() > 0

    empty_hyps = bev.new_zeros((2, 0, 9))
    fused_empty, updated_empty = fusion(empty_hyps, None, None, None, bev)
    assert fused_empty.shape == bev.shape
    assert updated_empty.shape == empty_hyps.shape
    check_collaboration_aware_gate()
    check_hvp_auxiliary_loss()

    print("HVP-CBEA smoke OK")
    print("HVP-CBEA backward OK")
    print("HVP-CBEA residual gate OK")
    print("HVP-CBEA collaboration-aware gate OK")
    print("HVP-CBEA auxiliary loss OK")


def check_collaboration_aware_gate():
    base = torch.ones(1, 4, 3, 3)
    delta = torch.full_like(base, 2.0)
    cfg = {
        "enabled": True,
        "alpha_init": 0.05,
        "alpha_max": 0.3,
        "learnable": True,
        "collaboration_aware": {
            "enabled": True,
            "no_collab_scale": 0.0,
            "collab_scale": 1.0,
            "min_cav": 2,
            "use_record_len": True,
            "fallback_scale": 1.0,
            "debug": True,
        },
    }
    fusion_ca = BayesianHypothesisFusion(in_channels=4, mid_channels=2, residual_gate=cfg)
    no_collab_scale, no_collab_debug = BayesianHypothesisFusion.compute_collaboration_scale(
        torch.tensor([1]),
        cfg["collaboration_aware"],
        device=base.device,
        dtype=base.dtype,
        batch_size=1,
    )
    no_collab_out = fusion_ca.apply_residual_delta(
        base,
        delta,
        collaboration_scale=no_collab_scale,
        collaboration_debug=no_collab_debug,
    )
    assert torch.allclose(no_collab_out, base)
    no_collab_residual_debug = fusion_ca.get_residual_debug()
    assert no_collab_residual_debug["hvp_cbea_collaboration_scale"] == 0.0
    assert no_collab_residual_debug["hvp_cbea_effective_alpha"] == 0.0

    collab_scale, collab_debug = BayesianHypothesisFusion.compute_collaboration_scale(
        torch.tensor([2]),
        cfg["collaboration_aware"],
        device=base.device,
        dtype=base.dtype,
        batch_size=1,
    )
    collab_out = fusion_ca.apply_residual_delta(
        base,
        delta,
        collaboration_scale=collab_scale,
        collaboration_debug=collab_debug,
    )
    assert torch.isfinite(collab_out).all()
    assert (collab_out - base).abs().sum() > 0
    collab_residual_debug = fusion_ca.get_residual_debug()
    assert collab_residual_debug["hvp_cbea_collaboration_scale"] == 1.0
    assert collab_residual_debug["hvp_cbea_effective_alpha"] > 0.0

    disabled_cfg = {
        "enabled": True,
        "alpha_init": 0.05,
        "alpha_max": 0.3,
        "learnable": True,
        "collaboration_aware": {
            "enabled": False,
            "no_collab_scale": 0.0,
            "collab_scale": 1.0,
            "min_cav": 2,
            "use_record_len": True,
            "fallback_scale": 1.0,
            "debug": False,
        },
    }
    fusion_disabled = BayesianHypothesisFusion(in_channels=4, mid_channels=2, residual_gate=disabled_cfg)
    disabled_scale, disabled_debug = BayesianHypothesisFusion.compute_collaboration_scale(
        torch.tensor([1]),
        disabled_cfg["collaboration_aware"],
        device=base.device,
        dtype=base.dtype,
        batch_size=1,
    )
    disabled_out = fusion_disabled.apply_residual_delta(
        base,
        delta,
        collaboration_scale=disabled_scale,
        collaboration_debug=disabled_debug,
    )
    alpha = fusion_disabled._residual_alpha().to(dtype=base.dtype)
    assert torch.allclose(disabled_out, base + alpha * delta)


def check_hvp_auxiliary_loss():
    delta_feature = torch.randn(1, 64, 32, 32, requires_grad=True)
    alpha = torch.tensor(0.05, requires_grad=True)
    effective_alpha = alpha.view(1, 1, 1, 1)
    hvp_residual = effective_alpha * delta_feature
    aux_dict = {
        "enabled": True,
        "delta_feature": delta_feature,
        "hvp_residual": hvp_residual,
        "alpha": alpha,
        "effective_alpha": effective_alpha,
    }
    aux_cfg = {
        "enabled": True,
        "residual_reg": {
            "enabled": True,
            "weight": 0.001,
            "type": "l1",
        },
        "alpha_reg": {
            "enabled": True,
            "weight": 0.001,
            "target": 0.02,
        },
        "refinement_consistency": {
            "enabled": True,
            "weight": 0.05,
            "mode": "feature_delta_l1",
        },
    }
    aux_loss, aux_stats = compute_hvp_auxiliary_loss(aux_dict, aux_cfg)
    assert torch.is_tensor(aux_loss)
    assert torch.isfinite(aux_loss).all()
    assert aux_loss.detach().item() > 0.0
    for key in (
        "hvp_residual_reg_loss",
        "hvp_alpha_reg_loss",
        "hvp_refinement_consistency_loss",
        "hvp_aux_total_loss",
    ):
        assert torch.isfinite(torch.tensor(aux_stats[key]))
    aux_loss.backward()
    assert delta_feature.grad is not None
    assert alpha.grad is not None
    assert torch.isfinite(delta_feature.grad).all()
    assert torch.isfinite(alpha.grad).all()
    assert delta_feature.grad.detach().abs().sum() > 0
    assert alpha.grad.detach().abs().sum() > 0


if __name__ == "__main__":
    main()

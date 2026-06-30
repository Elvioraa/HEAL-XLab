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

    print("HVP-CBEA smoke OK")
    print("HVP-CBEA backward OK")
    print("HVP-CBEA residual gate OK")


if __name__ == "__main__":
    main()

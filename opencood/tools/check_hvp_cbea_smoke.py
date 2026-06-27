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
    bev = torch.randn(2, 32, 20, 24)
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

    empty_hyps = bev.new_zeros((2, 0, 9))
    fused_empty, updated_empty = fusion(empty_hyps, None, None, None, bev)
    assert fused_empty.shape == bev.shape
    assert updated_empty.shape == empty_hyps.shape

    print("HVP-CBEA smoke OK")


if __name__ == "__main__":
    main()

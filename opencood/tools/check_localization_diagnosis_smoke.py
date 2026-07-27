"""CPU smoke test for the localization-signal diagnosis logic.

The diagnosis script's conclusion hinges on one claim: a reliability factor that
is uniform across agents cancels in the normalization, while one that differs
across agents does move alpha. If that claim were implemented wrongly the
diagnosis would be misleading, so it is tested here directly against the same
normalization the rule uses.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opencood.tools.pact_cbea_alpha_utils import alpha_from_reliability


def test_matches_rule_normalization():
    """_alpha_from_reliability must reproduce PACTCBEARule's alpha exactly."""
    from opencood.models.sub_modules.pact_cbea_rule import PACTCBEARule

    torch.manual_seed(0)
    rule = PACTCBEARule({})
    feature = torch.randn(1, 3, 2, 4, 4)
    heatmap = torch.randn(1, 3, 1, 4, 4)
    uncertainty = torch.rand(1, 3, 1, 4, 4)
    _, debug = rule(feature, evidence_heatmap=heatmap,
                    evidence_uncertainty=uncertainty)
    recomputed = alpha_from_reliability(debug["pact_reliability"])
    assert torch.allclose(debug["pact_alpha"], recomputed, atol=1e-6), \
        "diagnosis normalization diverges from the rule's own alpha"
    print("diagnosis normalization reproduces the rule's alpha: True")


def test_uniform_factor_cancels():
    """A factor identical across agents must leave alpha untouched."""
    torch.manual_seed(1)
    base = torch.rand(1, 4, 1, 5, 5) + 0.1
    # same L for every agent, varying over space
    uniform_L = torch.rand(1, 1, 1, 5, 5).expand(1, 4, 1, 5, 5) + 0.5

    alpha_without = alpha_from_reliability(base)
    alpha_with = alpha_from_reliability(base * uniform_L)
    delta = (alpha_with - alpha_without).abs().max()
    assert float(delta) < 1e-5, "uniform factor changed alpha by %.3e" % float(delta)
    print("agent-uniform L cancels in normalization (max delta %.2e): True" % float(delta))


def test_discriminative_factor_moves_alpha():
    """A factor that differs across agents must change alpha."""
    torch.manual_seed(2)
    base = torch.rand(1, 4, 1, 5, 5) + 0.1
    # strongly different L per agent
    per_agent = torch.tensor([1.0, 0.5, 0.2, 0.05]).view(1, 4, 1, 1, 1)
    discriminative_L = per_agent.expand(1, 4, 1, 5, 5)

    alpha_without = alpha_from_reliability(base)
    alpha_with = alpha_from_reliability(base * discriminative_L)
    delta = (alpha_with - alpha_without).abs().max()
    assert float(delta) > 0.05, \
        "discriminative factor barely moved alpha (%.3e)" % float(delta)
    print("agent-varying L moves alpha (max delta %.4f): True" % float(delta))


def main():
    test_matches_rule_normalization()
    test_uniform_factor_cancels()
    test_discriminative_factor_moves_alpha()
    print("LOCALIZATION_DIAGNOSIS_SMOKE_PASS")


if __name__ == "__main__":
    main()

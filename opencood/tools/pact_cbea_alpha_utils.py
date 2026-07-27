"""Shared alpha-normalization helper for PACT-CBEA diagnostics.

Kept in its own module (free of dataset/model imports) so diagnosis logic can
be unit-tested on any machine, including ones without the full runtime deps.
"""

import torch


def alpha_from_reliability(reliability, eps=1e-6):
    """Reproduce PACTCBEARule's normalization: reliability -> alpha.

    Mirrors the agent-dimension normalization in
    PACTCBEARule._aggregate_dense, including the uniform fallback when the
    per-pixel reliability sum is degenerate.

    reliability : torch.Tensor, [B, N, 1, H, W]
    """
    reliability = torch.nan_to_num(reliability, nan=0.0, posinf=0.0, neginf=0.0)
    reliability = torch.clamp(reliability, min=0.0)
    denom = reliability.sum(dim=1, keepdim=True)
    agent_count = reliability.shape[1]
    uniform = reliability.new_full(reliability.shape, 1.0 / max(agent_count, 1))
    return torch.where(denom > eps, reliability / (denom + eps), uniform)

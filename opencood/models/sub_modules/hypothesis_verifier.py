"""Collaborator-side hypothesis verifier for HVP-CBEA."""

import torch
import torch.nn as nn


class HypothesisVerifier(nn.Module):
    """Verify ego hypotheses with collaborator BEV features."""

    def __init__(self, in_channels=256, mid_channels=64, max_novel=20,
                 novel_threshold=0.5):
        super().__init__()
        self.max_novel = int(max_novel)
        self.novel_threshold = float(novel_threshold)
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.hyp_proj = nn.Linear(9, mid_channels)
        self.verif_head = nn.Linear(mid_channels, 3)
        self.refine_head = nn.Linear(mid_channels, 7)
        self.novel_head = nn.Linear(mid_channels, 9)

    def forward(self, collaborator_bev_feat, ego_hyps):
        if ego_hyps is None or ego_hyps.ndim != 3:
            return None, None, self._empty_novel(ego_hyps)
        bsz, num_hyp, _ = ego_hyps.shape
        if num_hyp == 0 or collaborator_bev_feat is None or collaborator_bev_feat.ndim != 4 or collaborator_bev_feat.shape[0] == 0:
            device, dtype = ego_hyps.device, ego_hyps.dtype
            return (
                torch.zeros((bsz, num_hyp, 3), device=device, dtype=dtype),
                torch.zeros((bsz, num_hyp, 7), device=device, dtype=dtype),
                self._empty_novel(ego_hyps),
            )

        ctx = self.context(collaborator_bev_feat).flatten(1).mean(dim=0, keepdim=True)
        ctx = ctx.expand(bsz, -1).unsqueeze(1)
        hyp_feat = self.hyp_proj(ego_hyps)
        feat = torch.tanh(hyp_feat + ctx)
        verif_logits = self.verif_head(feat)
        refine_delta = self.refine_head(feat)
        novel_base = self.novel_head(ctx.squeeze(1)).unsqueeze(1)
        novel_hyps = novel_base.repeat(1, min(self.max_novel, max(num_hyp, 1)), 1)
        novel_hyps[..., 7] = torch.sigmoid(novel_hyps[..., 7])
        novel_hyps[..., 8] = (novel_hyps[..., 7] >= self.novel_threshold).to(novel_hyps.dtype)
        return verif_logits, refine_delta, novel_hyps

    def compute_loss(self, *args, **kwargs):
        ref = self._find_tensor(args, kwargs)
        if ref is None:
            return torch.tensor(0.0)
        return ref.sum() * 0.0

    def assign_verification_labels(self, *args, **kwargs):
        ref = self._find_tensor(args, kwargs)
        if ref is None:
            return None
        return torch.zeros(ref.shape[:-1], device=ref.device, dtype=torch.long)

    def _empty_novel(self, ref):
        if torch.is_tensor(ref):
            return ref.new_zeros((ref.shape[0], 0, 9))
        return torch.zeros((0, 0, 9), dtype=torch.float32)

    @staticmethod
    def _find_tensor(args, kwargs):
        for item in list(args) + list(kwargs.values()):
            if torch.is_tensor(item):
                return item
        return None


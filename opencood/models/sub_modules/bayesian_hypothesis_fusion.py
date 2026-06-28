"""Bayesian hypothesis fusion for HVP-CBEA."""

import torch
import torch.nn as nn


class BayesianHypothesisFusion(nn.Module):
    """Scatter updated hypotheses into BEV feature space with gated fusion."""

    def __init__(self, in_channels=256, mid_channels=64, pc_range=None,
                 confirm_boost=1.5, refute_penalty=-2.5, refine_boost=0.8):
        super().__init__()
        self.pc_range = pc_range or [-102.4, -102.4, -3.0, 102.4, 102.4, 1.0]
        self.confirm_boost = float(confirm_boost)
        self.refute_penalty = float(refute_penalty)
        self.refine_boost = float(refine_boost)
        self.hyp_proj = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, ego_hyps, verif_logits, refine_delta, novel_hyps, ego_bev_feat):
        if ego_bev_feat is None or ego_bev_feat.ndim != 4:
            return ego_bev_feat, ego_hyps
        if ego_hyps is None or ego_hyps.ndim != 3 or ego_hyps.shape[1] == 0:
            return ego_bev_feat, ego_hyps
        try:
            updated_hyps = self._update_hypotheses(ego_hyps, verif_logits, refine_delta)
            hyp_map = self._scatter(updated_hyps, novel_hyps, ego_bev_feat)
            hyp_feat = self.hyp_proj(hyp_map)
            gate = self.gate(torch.cat([ego_bev_feat, hyp_feat], dim=1))
            fused = ego_bev_feat * (1.0 - gate) + hyp_feat * gate
            return fused, updated_hyps
        except Exception:
            return ego_bev_feat, ego_hyps

    def compute_loss(self, *args, **kwargs):
        ref = self._find_tensor(args, kwargs)
        if ref is None:
            return torch.tensor(0.0)
        return ref.sum() * 0.0

    def _update_hypotheses(self, ego_hyps, verif_logits, refine_delta):
        if verif_logits is None or refine_delta is None:
            return ego_hyps
        probs = torch.softmax(verif_logits, dim=-1)
        base_score = torch.clamp(ego_hyps[..., 7:8], 1e-4, 1.0 - 1e-4)
        log_odds = torch.logit(base_score)
        log_odds = log_odds + probs[..., 0:1] * self.confirm_boost
        log_odds = log_odds + probs[..., 1:2] * self.refute_penalty
        log_odds = log_odds + probs[..., 2:3] * self.refine_boost
        updated_score = torch.sigmoid(log_odds)
        updated_box = ego_hyps[..., :7] + refine_delta[..., :7] * probs[..., 2:3]
        updated_valid = (updated_score > 0.05).to(ego_hyps.dtype)
        return torch.cat([updated_box, updated_score, updated_valid], dim=-1)

    def _scatter(self, hyps, novel_hyps, ref_feat):
        bsz, _, height, width = ref_feat.shape
        all_hyps = hyps
        if novel_hyps is not None and torch.is_tensor(novel_hyps) and novel_hyps.numel() > 0:
            all_hyps = torch.cat([hyps, novel_hyps.to(device=hyps.device, dtype=hyps.dtype)], dim=1)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range[:6]
        x = ((all_hyps[..., 0] - x_min) / max(x_max - x_min, 1e-6) * width).long()
        y = ((all_hyps[..., 1] - y_min) / max(y_max - y_min, 1e-6) * height).long()
        valid = (all_hyps[..., 8] > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        rows = []
        for bidx in range(bsz):
            flat = ref_feat.new_zeros((height * width,))
            if not torch.any(valid[bidx]):
                rows.append(flat)
                continue
            flat_idx = y[bidx][valid[bidx]] * width + x[bidx][valid[bidx]]
            values = all_hyps[bidx, valid[bidx], 7]
            rows.append(flat.scatter_add(0, flat_idx, values))
        return torch.stack(rows, dim=0).view(bsz, 1, height, width)

    @staticmethod
    def _find_tensor(args, kwargs):
        for item in list(args) + list(kwargs.values()):
            if torch.is_tensor(item):
                return item
        return None

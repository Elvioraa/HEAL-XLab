"""Hypothesis head for HVP-HEAL v3 Stage1."""

import torch
import torch.nn as nn


class HvpHealV3HypothesisHead(nn.Module):
    """Lightweight BEV hypothesis heatmap head."""

    def __init__(self, in_channels=64, hidden_dim=64, out_channels=1,
                 use_sigmoid=True, return_feature=True):
        super().__init__()
        self.use_sigmoid = bool(use_sigmoid)
        self.return_feature = bool(return_feature)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.heatmap_head = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, bev_feature):
        if bev_feature is None or bev_feature.ndim != 4:
            raise ValueError("HVP-HEAL v3 hypothesis head expects [B,C,H,W] BEV feature")
        hypothesis_feature = self.stem(bev_feature)
        logits = self.heatmap_head(hypothesis_feature)
        heatmap = torch.sigmoid(logits) if self.use_sigmoid else logits
        output = {
            "hypothesis_heatmap_logits": logits,
            "hypothesis_heatmap": heatmap,
        }
        if self.return_feature:
            output["hypothesis_feature"] = hypothesis_feature
        return output

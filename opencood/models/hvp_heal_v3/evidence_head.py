"""Evidence head for HVP-HEAL v3 Stage2."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HvpHealV3EvidenceHead(nn.Module):
    """Lightweight BEV evidence head for modality adaptation."""

    def __init__(self, in_channels=256, hidden_dim=64, descriptor_dim=16,
                 use_sigmoid=True, normalize_descriptor=True,
                 return_feature=True, predict_localization_uncertainty=False):
        super().__init__()
        self.use_sigmoid = bool(use_sigmoid)
        self.normalize_descriptor = bool(normalize_descriptor)
        self.return_feature = bool(return_feature)
        self.predict_localization_uncertainty = bool(predict_localization_uncertainty)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.evidence_heatmap_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.evidence_uncertainty_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.evidence_descriptor_head = nn.Conv2d(
            hidden_dim,
            descriptor_dim,
            kernel_size=1,
        )
        if self.predict_localization_uncertainty:
            self.evidence_localization_uncertainty_head = nn.Conv2d(
                hidden_dim, 1, kernel_size=1,
            )

    def forward(self, bev_feature):
        if bev_feature is None or bev_feature.ndim != 4:
            raise ValueError("HVP-HEAL v3 evidence head expects [B,C,H,W] BEV feature")
        evidence_feature = self.stem(bev_feature)
        heatmap_logits = self.evidence_heatmap_head(evidence_feature)
        heatmap = torch.sigmoid(heatmap_logits) if self.use_sigmoid else heatmap_logits
        uncertainty_logits = self.evidence_uncertainty_head(evidence_feature)
        uncertainty = F.softplus(uncertainty_logits) + 1e-6
        descriptor = self.evidence_descriptor_head(evidence_feature)
        if self.normalize_descriptor:
            descriptor = F.normalize(descriptor, p=2, dim=1, eps=1e-6)

        output = {
            "evidence_heatmap_logits": heatmap_logits,
            "evidence_heatmap": heatmap,
            "evidence_uncertainty_logits": uncertainty_logits,
            "evidence_uncertainty": uncertainty,
            "evidence_descriptor": descriptor,
        }
        if self.return_feature:
            output["evidence_feature"] = evidence_feature
        if self.predict_localization_uncertainty:
            loc_uncertainty_logits = self.evidence_localization_uncertainty_head(
                evidence_feature
            )
            loc_uncertainty = F.softplus(loc_uncertainty_logits) + 1e-6
            output["evidence_localization_uncertainty_logits"] = loc_uncertainty_logits
            output["evidence_localization_uncertainty"] = loc_uncertainty
        return output

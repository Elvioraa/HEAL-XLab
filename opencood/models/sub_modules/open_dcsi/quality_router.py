"""Modality-independent evidence quality routing for Open-DCSI tokens."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceQualityRouter(nn.Module):
    """Combine PACT-compatible quality signals without fixed modality IDs."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        calibration = config["calibration"]
        if calibration["enabled"]:
            self.calibration_exponent_raw = nn.Parameter(
                torch.tensor(0.5413248546)
            )

    def forward(self, token_dict):
        evidence = token_dict["evidence_confidence"].clamp(0.0, 1.0)
        if not self.config["enabled"]:
            return torch.ones_like(evidence)
        reliability = evidence
        if self.config["use_general_uncertainty"]:
            reliability = reliability * torch.exp(
                -token_dict["general_uncertainty"].clamp_min(0.0)
            )
        if self.config["use_localization_uncertainty"]:
            reliability = reliability * torch.exp(
                -token_dict["localization_uncertainty"].clamp_min(0.0)
            )
        if self.config["predict_token_validity"]:
            reliability = reliability * token_dict["validity"].clamp(0.0, 1.0)
        if self.config["predict_box_quality"]:
            reliability = reliability * token_dict["box_quality"].clamp(0.0, 1.0)
        reliability = torch.where(
            torch.isfinite(reliability), reliability, torch.zeros_like(reliability)
        ).clamp(0.0, 1.0)
        if hasattr(self, "calibration_exponent_raw"):
            exponent = F.softplus(self.calibration_exponent_raw).clamp_min(1e-6)
            calibrated = reliability.clamp_min(1e-12).pow(exponent)
            reliability = torch.where(
                reliability > 0, calibrated, torch.zeros_like(calibrated)
            )
        return reliability


def absolute_rejection_gate(reliability, config, training):
    """Apply continuous training rejection and optional hard inference rejection."""

    if not config["enabled"]:
        return torch.ones_like(reliability)
    threshold = float(config["threshold"])
    if not training and config["hard_threshold_inference_only"]:
        return (reliability >= threshold).to(reliability.dtype)
    temperature = max(float(config["temperature"]), 1e-6)
    return torch.sigmoid((reliability - threshold) / temperature)

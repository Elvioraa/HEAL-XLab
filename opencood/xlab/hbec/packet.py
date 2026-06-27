"""HBEC hypothesis and evidence packets."""

from dataclasses import dataclass
from typing import Optional

import torch

from opencood.xlab.utils import tensor_payload_bytes


@dataclass
class HypothesisPacket:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
    source_agent: Optional[str] = None
    source_modality: Optional[str] = None
    timestamp: Optional[object] = None
    transform_status: str = "ego_frame"
    payload_bytes_est: int = 0

    @classmethod
    def from_tensors(cls, boxes, scores, cfg, **metadata):
        uncertainty = estimate_uncertainty(scores, cfg)
        payload = tensor_payload_bytes(boxes, scores, uncertainty, metadata.get("labels"))
        return cls(
            boxes=boxes,
            scores=scores,
            labels=metadata.get("labels"),
            uncertainty=uncertainty,
            source_agent=metadata.get("source_agent"),
            source_modality=metadata.get("source_modality"),
            timestamp=metadata.get("timestamp"),
            transform_status=metadata.get("transform_status", "ego_frame"),
            payload_bytes_est=payload,
        )


@dataclass
class EvidencePacket(HypothesisPacket):
    pass


def estimate_uncertainty(scores, cfg):
    hbec_cfg = cfg.get("hbec", cfg) if isinstance(cfg, dict) else {}
    base = float(hbec_cfg.get("base_uncertainty", 1.0))
    min_score = float(hbec_cfg.get("min_score_for_uncertainty", 0.05))
    if scores is None or not torch.is_tensor(scores):
        return None
    safe_score = torch.clamp(scores.detach(), min=min_score, max=1.0)
    return torch.ones_like(safe_score) * base / safe_score


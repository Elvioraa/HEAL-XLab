"""HBEC post-process engine."""

import torch

from opencood.xlab.hbec.matcher import HypothesisMatcher
from opencood.xlab.hbec.evidence import HBECEvidenceExtractor
from opencood.xlab.hbec.packet import EvidencePacket, HypothesisPacket
from opencood.xlab.hbec.refiner import BayesianRefiner
from opencood.xlab.utils import is_valid_box_tensor, is_valid_score_tensor, tensor_payload_bytes


class HBECPostProcessor:
    """Object-state Bayesian evidence communication after official post_process."""

    def __init__(self, xlab_cfg, recorder=None):
        self.xlab_cfg = xlab_cfg or {}
        self.hbec_cfg = self.xlab_cfg.get("hbec", {})
        self.recorder = recorder

    def __call__(
        self,
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        batch_data=None,
        model=None,
        hypes=None,
        infer_context=None,
    ):
        record = {
            "hbec_enabled": True,
            "payload_bytes_est": tensor_payload_bytes(pred_box_tensor, pred_score),
            "nms_status": "not_applied",
            "evidence_source": self.hbec_cfg.get("evidence_source", "none"),
            "evidence_source_requested": self.hbec_cfg.get("evidence_source", "none"),
            "evidence_source_used": "",
            "evidence_extract_func": "",
            "evidence_box_count": 0,
            "evidence_score_mean": 0.0,
            "evidence_extract_success": False,
            "evidence_extract_error": "",
            "raw_output_keys": [],
            "dataset_class": "",
            "has_post_process_no_fusion": False,
            "has_post_processor": False,
        }

        if not is_valid_box_tensor(pred_box_tensor) or not is_valid_score_tensor(pred_score):
            record["fallback_reason"] = "invalid_official_prediction_shape"
            self._write(record)
            return pred_box_tensor, pred_score, gt_box_tensor
        if pred_box_tensor.shape[0] != pred_score.shape[0]:
            record["fallback_reason"] = "box_score_count_mismatch"
            self._write(record)
            return pred_box_tensor, pred_score, gt_box_tensor

        ego_packet = HypothesisPacket.from_tensors(
            pred_box_tensor,
            pred_score,
            self.xlab_cfg,
            source_agent="ego",
            source_modality=(hypes or {}).get("ego_modality"),
        )
        evidence_packet = self._get_collaborator_evidence(
            batch_data=batch_data,
            model=model,
            hypes=hypes,
            infer_context=infer_context,
            record=record,
        )
        if evidence_packet is None:
            record["fallback_reason"] = record.get("fallback_reason") or "no_collaborator_evidence"
            self._write(record)
            return pred_box_tensor, pred_score, gt_box_tensor

        matcher = HypothesisMatcher(self.hbec_cfg.get("match", {}))
        matches = matcher.match(ego_packet.boxes, evidence_packet.boxes)
        refiner = BayesianRefiner(self.hbec_cfg)
        fused_boxes, fused_scores, refined_count = refiner.refine(ego_packet, evidence_packet, matches)
        fused_boxes, fused_scores, novel_count = refiner.insert_novel(
            fused_boxes, fused_scores, ego_packet, evidence_packet, matches
        )
        fused_scores, suppressed_count = refiner.suppress(fused_scores, matches)
        fused_boxes, fused_scores, nms_status = self._apply_optional_nms(fused_boxes, fused_scores)
        fused_boxes, fused_scores = self._cap_boxes(fused_boxes, fused_scores)

        if not self._output_is_safe(fused_boxes, fused_scores, pred_box_tensor, pred_score):
            record["fallback_reason"] = "unsafe_hbec_output"
            self._write(record)
            return pred_box_tensor, pred_score, gt_box_tensor

        record.update({
            "matched_count": len(matches),
            "refined_count": refined_count,
            "novel_count": novel_count,
            "suppressed_count": suppressed_count,
            "payload_bytes_est": ego_packet.payload_bytes_est + evidence_packet.payload_bytes_est,
            "fallback_reason": matcher.last_fallback_reason,
            "nms_status": nms_status,
        })
        self._write(record)
        return fused_boxes, fused_scores, gt_box_tensor

    def _get_collaborator_evidence(self, batch_data, model, hypes, infer_context, record):
        infer_context = dict(infer_context or {})
        extractor = HBECEvidenceExtractor(self.xlab_cfg)
        evidence_packet = extractor.extract(
            batch_data=batch_data,
            model=model,
            hypes=hypes,
            infer_context=infer_context,
        )
        record["evidence_extract_error"] = extractor.last_error
        record.update(extractor.debug_info)
        if evidence_packet is None:
            record["fallback_reason"] = extractor.last_reason or "no_collaborator_evidence"
            return None
        record["evidence_extract_success"] = True
        record["evidence_box_count"] = int(evidence_packet.boxes.shape[0])
        record["evidence_score_mean"] = float(evidence_packet.scores.detach().mean().cpu())
        record["evidence_source"] = evidence_packet.source_agent or self.hbec_cfg.get("evidence_source", "none")
        record["evidence_source_used"] = record["evidence_source"]
        return evidence_packet

    def _apply_optional_nms(self, boxes, scores):
        try:
            from opencood.utils import box_utils

            keep = box_utils.nms_rotated(boxes, scores, self.hbec_cfg.get("nms_thresh", 0.01))
            keep = torch.as_tensor(keep, device=boxes.device, dtype=torch.long)
            return boxes[keep], scores[keep], "applied_box_utils_nms_rotated"
        except Exception:
            return boxes, scores, "not_applied"

    def _cap_boxes(self, boxes, scores):
        max_boxes = int(self.hbec_cfg.get("safety", {}).get("max_boxes_after_fusion", 300))
        if len(scores) <= max_boxes:
            return boxes, scores
        order = torch.argsort(scores, descending=True)[:max_boxes]
        return boxes[order], scores[order]

    @staticmethod
    def _output_is_safe(boxes, scores, ref_boxes, ref_scores):
        return (
            is_valid_box_tensor(boxes)
            and is_valid_score_tensor(scores)
            and boxes.shape[0] == scores.shape[0]
            and boxes.device == ref_boxes.device
            and scores.device == ref_scores.device
            and boxes.dtype == ref_boxes.dtype
            and scores.dtype == ref_scores.dtype
        )

    def _write(self, record):
        if self.recorder is not None:
            self.recorder.write(record)

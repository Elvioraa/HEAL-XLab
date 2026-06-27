"""Object-level evidence extraction for HBEC."""

import warnings

import torch

from opencood.xlab.hbec.packet import EvidencePacket
from opencood.xlab.utils import is_valid_box_tensor, is_valid_score_tensor


class HBECEvidenceExtractor:
    """Extract object-level evidence through official inference utilities."""

    def __init__(self, xlab_cfg):
        self.xlab_cfg = xlab_cfg or {}
        self.hbec_cfg = self.xlab_cfg.get("hbec", {})
        self.last_reason = ""
        self.last_error = ""

    def extract(self, batch_data=None, model=None, hypes=None, infer_context=None):
        self.last_reason = ""
        self.last_error = ""
        infer_context = infer_context or {}
        source = self.hbec_cfg.get("evidence_source", "none")

        if infer_context.get("collaborator_evidence") is not None or infer_context.get("evidence_packet") is not None:
            return self._extract_explicit(infer_context)
        if source == "none":
            self.last_reason = "evidence_source_none"
            return None
        if source == "explicit":
            return self._extract_explicit(infer_context)
        if source == "late_fusion_reinfer":
            return self._extract_reinfer(
                source=source,
                fn_name="inference_late_fusion",
                batch_data=batch_data,
                model=model,
                infer_context=infer_context,
            )
        if source == "no_fusion_reinfer":
            return self._extract_reinfer(
                source=source,
                fn_name="inference_no_fusion",
                batch_data=batch_data,
                model=model,
                infer_context=infer_context,
            )

        self.last_reason = "unsupported_evidence_source:%s" % source
        return None

    def _extract_explicit(self, infer_context):
        evidence = infer_context.get("collaborator_evidence") or infer_context.get("evidence_packet")
        if isinstance(evidence, EvidencePacket):
            if self._is_empty(evidence):
                self.last_reason = "empty_evidence"
                return None
            return evidence
        if not isinstance(evidence, dict):
            self.last_reason = "missing_explicit_evidence"
            return None
        packet = self._packet_from_result(
            boxes=evidence.get("boxes"),
            scores=evidence.get("scores"),
            source_agent=evidence.get("source_agent", "explicit"),
            transform_status=evidence.get("transform_status", "provided_by_context"),
        )
        if packet is None:
            self.last_reason = self.last_reason or "invalid_explicit_evidence"
        return packet

    def _extract_reinfer(self, source, fn_name, batch_data, model, infer_context):
        if infer_context.get("xlab_internal_reinfer", False):
            self.last_reason = "recursive_reinfer_guard"
            return None
        dataset = infer_context.get("dataset") or infer_context.get("opencood_dataset")
        if batch_data is None or model is None or dataset is None:
            self.last_reason = "missing_reinfer_inputs"
            return None
        if fn_name == "no_fusion_reinfer" or source == "no_fusion_reinfer":
            if not hasattr(dataset, "post_process_no_fusion"):
                self.last_reason = "dataset_missing_post_process_no_fusion"
                return None

        was_training = getattr(model, "training", False)
        try:
            from opencood.tools import inference_utils

            fn = getattr(inference_utils, fn_name)
            model.eval()
            with torch.no_grad():
                result = fn(batch_data, model, dataset)
            if was_training:
                model.train()
            boxes = result.get("pred_box_tensor") if isinstance(result, dict) else None
            scores = result.get("pred_score") if isinstance(result, dict) else None
            return self._packet_from_result(
                boxes=boxes,
                scores=scores,
                source_agent=source,
                transform_status="official_postprocess_ego_frame",
            )
        except Exception as exc:
            if was_training:
                model.train()
            self.last_reason = "evidence_extract_error"
            self.last_error = "%s:%s" % (type(exc).__name__, exc)
            warnings.warn("HBEC evidence extraction fallback: %s" % self.last_error)
            return None

    def _packet_from_result(self, boxes, scores, source_agent, transform_status):
        if not is_valid_box_tensor(boxes) or not is_valid_score_tensor(scores):
            self.last_reason = "invalid_evidence_shape"
            return None
        if boxes.shape[0] != scores.shape[0]:
            self.last_reason = "evidence_box_score_count_mismatch"
            return None
        if boxes.shape[0] == 0:
            self.last_reason = "empty_evidence"
            return None
        return EvidencePacket.from_tensors(
            boxes=boxes,
            scores=scores,
            cfg=self.xlab_cfg,
            labels=None,
            source_agent=source_agent,
            source_modality="object_level",
            timestamp=None,
            transform_status=transform_status,
        )

    @staticmethod
    def _is_empty(packet):
        return packet.boxes is None or packet.scores is None or packet.boxes.shape[0] == 0

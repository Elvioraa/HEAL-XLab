"""Object-level evidence extraction for HBEC."""

import warnings
from copy import copy

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
        self.debug_info = {}

    def extract(self, batch_data=None, model=None, hypes=None, infer_context=None):
        self.last_reason = ""
        self.last_error = ""
        self.debug_info = {
            "evidence_source_requested": self.hbec_cfg.get("evidence_source", "none"),
            "evidence_source_used": "",
            "evidence_extract_func": "",
            "raw_output_keys": [],
            "dataset_class": "",
            "has_post_process_no_fusion": False,
            "has_post_processor": False,
            "evidence_box_count": 0,
        }
        infer_context = infer_context or {}
        source = self.hbec_cfg.get("evidence_source", "none")
        dataset = infer_context.get("dataset") or infer_context.get("opencood_dataset")
        self._record_dataset_debug(dataset)

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
        if source == "single_agent_reinfer":
            return self._extract_single_agent_reinfer(
                source=source,
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
        if source == "no_fusion_reinfer":
            if not hasattr(dataset, "post_process_no_fusion"):
                return self._extract_single_agent_reinfer(
                    source="single_agent_reinfer",
                    batch_data=batch_data,
                    model=model,
                    infer_context=infer_context,
                    fallback_from="dataset_missing_post_process_no_fusion",
                )

        was_training = getattr(model, "training", False)
        try:
            from opencood.tools import inference_utils

            fn = getattr(inference_utils, fn_name)
            self.debug_info["evidence_extract_func"] = "opencood.tools.inference_utils.%s" % fn_name
            model.eval()
            with torch.no_grad():
                result = fn(batch_data, model, dataset)
            if was_training:
                model.train()
            self.debug_info["raw_output_keys"] = sorted(result.keys()) if isinstance(result, dict) else []
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

    def _extract_single_agent_reinfer(self, source, batch_data, model, infer_context, fallback_from=""):
        if infer_context.get("xlab_internal_reinfer", False):
            self.last_reason = "recursive_reinfer_guard"
            return None
        dataset = infer_context.get("dataset") or infer_context.get("opencood_dataset")
        if batch_data is None or model is None or dataset is None:
            self.last_reason = "missing_single_agent_reinfer_inputs"
            return None
        if not hasattr(dataset, "post_process"):
            self.last_reason = "dataset_missing_post_process"
            return None
        try:
            ego_only_batch = self._build_ego_only_batch(batch_data)
        except Exception as exc:
            self.last_reason = "single_agent_batch_build_error"
            self.last_error = "%s:%s" % (type(exc).__name__, exc)
            return None

        was_training = getattr(model, "training", False)
        try:
            model.eval()
            with torch.no_grad():
                output_dict = {"ego": model(ego_only_batch["ego"])}
                pred_box_tensor, pred_score, _ = dataset.post_process(ego_only_batch, output_dict)
            if was_training:
                model.train()
            self.debug_info["evidence_extract_func"] = "single_agent_reinfer:dataset.post_process"
            self.debug_info["raw_output_keys"] = sorted(output_dict["ego"].keys())
            packet = self._packet_from_result(
                boxes=pred_box_tensor,
                scores=pred_score,
                source_agent=source,
                transform_status="official_postprocess_ego_frame",
            )
            if packet is not None and fallback_from:
                self.last_reason = ""
                self.debug_info["fallback_from"] = fallback_from
            return packet
        except Exception as exc:
            if was_training:
                model.train()
            self.last_reason = "single_agent_reinfer_error"
            self.last_error = "%s:%s" % (type(exc).__name__, exc)
            warnings.warn("HBEC single-agent evidence fallback: %s" % self.last_error)
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
        self.debug_info["evidence_source_used"] = source_agent
        self.debug_info["evidence_box_count"] = int(boxes.shape[0])
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

    def _record_dataset_debug(self, dataset):
        if dataset is None:
            return
        self.debug_info["dataset_class"] = dataset.__class__.__name__
        self.debug_info["has_post_process_no_fusion"] = hasattr(dataset, "post_process_no_fusion")
        self.debug_info["has_post_processor"] = hasattr(dataset, "post_processor")

    def _build_ego_only_batch(self, batch_data):
        if not isinstance(batch_data, dict) or "ego" not in batch_data:
            raise ValueError("batch_data must contain ego")
        ego_data = batch_data["ego"]
        agent_modality_list = list(ego_data.get("agent_modality_list", []))
        if not agent_modality_list:
            raise ValueError("missing agent_modality_list")
        ego_modality = agent_modality_list[0]
        local_modality_index = agent_modality_list[:1].count(ego_modality) - 1

        ego_only = {}
        for key, value in ego_data.items():
            if key == "agent_modality_list":
                ego_only[key] = [ego_modality]
            elif key == "record_len" and torch.is_tensor(value):
                ego_only[key] = torch.ones_like(value)
            elif key in ("lidar_pose", "lidar_pose_clean", "lidar_poses", "lidar_poses_clean") and torch.is_tensor(value):
                ego_only[key] = value[:1].clone()
            elif key.startswith("inputs_") and key == "inputs_%s" % ego_modality:
                ego_only[key] = self._slice_modality_input(value, local_modality_index)
            elif key.startswith("inputs_"):
                ego_only[key] = None
            else:
                ego_only[key] = value
        return {"ego": ego_only}

    def _slice_modality_input(self, value, local_index):
        if value is None:
            return None
        if isinstance(value, dict):
            return self._slice_modality_dict(value, local_index)
        if torch.is_tensor(value):
            return value[local_index:local_index + 1].clone()
        if isinstance(value, list):
            return value[local_index:local_index + 1]
        return copy(value)

    def _slice_modality_dict(self, value, local_index):
        out = {}
        voxel_mask = None
        if "voxel_coords" in value and torch.is_tensor(value["voxel_coords"]):
            coords = value["voxel_coords"]
            voxel_mask = coords[:, 0] == local_index
        for key, item in value.items():
            if torch.is_tensor(item):
                if key == "voxel_coords" and voxel_mask is not None:
                    sliced = item[voxel_mask].clone()
                    if sliced.numel() > 0:
                        sliced[:, 0] = 0
                    out[key] = sliced
                elif voxel_mask is not None and item.shape[0] == voxel_mask.shape[0]:
                    out[key] = item[voxel_mask].clone()
                elif item.ndim > 0 and item.shape[0] > local_index:
                    out[key] = item[local_index:local_index + 1].clone()
                else:
                    out[key] = item.clone()
            elif isinstance(item, list):
                out[key] = item[local_index:local_index + 1]
            else:
                out[key] = copy(item)
        return out

"""Frozen PACT no-joint inference with sparse detector box packets only."""

from collections import Counter

import torch
import torch.nn as nn
import torchvision

from opencood.models.sub_modules.pact_cbea_box_packet_nojoint import (
    PACKET_SOURCE,
    PACTNoJointBoxCommunicationMeter,
    PACTNoJointBoxPacketCodec,
)

from opencood.models.heter_pyramid_collab import HeterPyramidCollab


class HeterPyramidCollabPactCbeaBoxPacketNojoint(HeterPyramidCollab):
    """Frozen local detectors with parameter-free box-packet fusion."""

    BN_TYPES = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )

    def __init__(self, args):
        raw_cfg = args.get("pact_box_packet_nojoint")
        if not isinstance(raw_cfg, dict) or raw_cfg.get("enabled") is not True:
            raise ValueError(
                "box-packet no-joint model requires "
                "pact_box_packet_nojoint.enabled=true"
            )
        super().__init__(args)
        self.pact_box_packet_nojoint_cfg = self._normalize_cfg(raw_cfg)
        cfg = self.pact_box_packet_nojoint_cfg
        self.pact_box_packet_nojoint_codec = PACTNoJointBoxPacketCodec(
            score_threshold=cfg["score_threshold"],
            local_nms_thresh=cfg["local_nms_thresh"],
            global_nms_thresh=cfg["global_nms_thresh"],
            max_boxes_per_agent=cfg["max_boxes_per_agent"],
            order=cfg["order"],
            dir_offset=cfg["dir_offset"],
            num_bins=cfg["num_bins"],
            gt_range=self.cav_range,
            quantize=cfg["quantize"],
        )
        self.pact_box_packet_nojoint_comm_meter = (
            PACTNoJointBoxCommunicationMeter(
                quantize=cfg["quantize"],
                bytes_per_scalar=cfg["bytes_per_scalar"],
                deadline_ms=cfg["deadline_ms"],
                bandwidth_budget_kb=cfg["bandwidth_budget_kb"],
            )
        )
        self._freeze_all_parameters_and_eval()
        self.pact_box_packet_nojoint_summary = self._parameter_summary()

    def train(self, mode=True):
        self._freeze_all_parameters_and_eval()
        return self

    def model_train_init(self):
        self._freeze_all_parameters_and_eval()

    def forward(self, data_dict):
        with torch.no_grad():
            output_dict = {"pyramid": "collab"}
            modality_names = data_dict["agent_modality_list"]
            record_len = data_dict["record_len"]
            pairwise_t_matrix = data_dict["pairwise_t_matrix"]
            anchor_box = data_dict["anchor_box"]

            agent_features = self._encode_agent_features(data_dict, output_dict)
            self._validate_scene_layout(
                agent_features,
                record_len,
                modality_names,
                pairwise_t_matrix,
            )
            scene_records = []
            ego_cls_predictions = []
            ego_reg_predictions = []
            ego_dir_predictions = []
            ego_occ_outputs = []
            per_agent_box_count = []
            start = 0

            for batch_index, scene_length_value in enumerate(
                    record_len.detach().cpu().tolist()):
                scene_length = int(scene_length_value)
                scene_anchor = self._scene_anchor_box(
                    anchor_box, batch_index, len(record_len)
                )
                try:
                    ego_dense, ego_prediction = self._run_local_detector(
                        agent_features,
                        start,
                        scene_anchor,
                        retain_dense=True,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "PACT box-packet ego local detection failed for "
                        "scene %d: %s" % (batch_index, exc)
                    ) from exc

                ego_cls_predictions.append(ego_dense["cls_preds"])
                ego_reg_predictions.append(ego_dense["reg_preds"])
                ego_dir_predictions.append(ego_dense["dir_preds"])
                ego_occ_outputs.append(ego_dense["occ_outputs"])
                ego_packet = self.pact_box_packet_nojoint_codec.build_packet(
                    ego_prediction,
                    modality_names[start],
                    start,
                    transmitted=False,
                )
                scene_counts = [{
                    "agent_index": start,
                    "local_index": 0,
                    "modality": modality_names[start],
                    "is_ego": True,
                    "box_count": int(ego_prediction["scores"].shape[0]),
                }]
                local_collaborator_packets = []
                packet_failure = ""

                try:
                    for local_index in range(1, scene_length):
                        agent_index = start + local_index
                        _, collaborator_prediction = self._run_local_detector(
                            agent_features,
                            agent_index,
                            scene_anchor,
                            retain_dense=False,
                        )
                        packet = self.pact_box_packet_nojoint_codec.build_packet(
                            collaborator_prediction,
                            modality_names[agent_index],
                            agent_index,
                            transmitted=True,
                        )
                        local_collaborator_packets.append({
                            "local_index": local_index,
                            "packet": packet,
                        })
                        scene_counts.append({
                            "agent_index": agent_index,
                            "local_index": local_index,
                            "modality": modality_names[agent_index],
                            "is_ego": False,
                            "box_count": int(packet["scores"].shape[0]),
                        })
                        del collaborator_prediction
                except Exception as exc:
                    if self.pact_box_packet_nojoint_cfg[
                            "failure_policy"] == "error":
                        raise RuntimeError(
                            "PACT box-packet collaborator processing failed "
                            "for scene %d: %s" % (batch_index, exc)
                        ) from exc
                    packet_failure = "%s: %s" % (type(exc).__name__, exc)
                    local_collaborator_packets = []

                scene_records.append({
                    "ego_packet": ego_packet,
                    "local_collaborator_packets": local_collaborator_packets,
                    "fallback_reason": packet_failure,
                })
                per_agent_box_count.append(scene_counts)
                start += scene_length

            # Collaborator dense tensors exist only during local inference and
            # are discarded before communication and packet fusion.
            del agent_features

            final_box_parts = []
            final_score_parts = []
            transmitted_packets = []
            fallback_reasons = []
            for batch_index, record in enumerate(scene_records):
                ego_packet = record["ego_packet"]
                local_packets = record["local_collaborator_packets"]
                fallback_reason = record["fallback_reason"]
                if fallback_reason:
                    final_packet = {
                        "boxes_corner": ego_packet["boxes_corner"],
                        "scores": ego_packet["scores"],
                    }
                    fallback_reasons.append(
                        "scene_%d %s" % (batch_index, fallback_reason)
                    )
                else:
                    try:
                        collaborators = []
                        for local_packet in local_packets:
                            local_index = local_packet["local_index"]
                            packet = local_packet["packet"]
                            packet["boxes_corner"] = (
                                self.pact_box_packet_nojoint_codec
                                .transform_packet_to_ego(
                                    packet["boxes_corner"],
                                    pairwise_t_matrix[
                                        batch_index, local_index, 0
                                    ],
                                )
                            )
                            collaborators.append(packet)
                        transmitted_packets.extend(collaborators)
                        final_packet = (
                            self.pact_box_packet_nojoint_codec.fuse_packets(
                                ego_packet, collaborators
                            )
                        )
                    except Exception as exc:
                        if self.pact_box_packet_nojoint_cfg[
                                "failure_policy"] == "error":
                            raise RuntimeError(
                                "PACT box-packet fusion failed for scene "
                                "%d: %s" % (batch_index, exc)
                            ) from exc
                        final_packet = {
                            "boxes_corner": ego_packet["boxes_corner"],
                            "scores": ego_packet["scores"],
                        }
                        fallback_reasons.append(
                            "scene_%d %s: %s" % (
                                batch_index, type(exc).__name__, exc
                            )
                        )
                final_box_parts.append(final_packet["boxes_corner"])
                final_score_parts.append(final_packet["scores"])

            final_boxes = self._concatenate_or_empty(
                final_box_parts, ego_cls_predictions[0], (0, 8, 3)
            )
            final_scores = self._concatenate_or_empty(
                final_score_parts, ego_cls_predictions[0], (0,)
            )
            comm_stats = self.pact_box_packet_nojoint_comm_meter(
                transmitted_packets
            )
            debug = {
                "box_packet_only_verified": True,
                "no_joint_training_verified": True,
                "stage3_training_required": False,
                "dense_collab_fusion_used": False,
                "forward_collab_used": False,
                "collaborator_dense_after_packet_used": False,
                "packet_source": PACKET_SOURCE,
                "collaborator_box_count": comm_stats[
                    "collaborator_box_count"
                ],
                "packet_bytes_per_frame": comm_stats[
                    "packet_bytes_per_frame"
                ],
                "packet_kb_per_frame": comm_stats["packet_kb_per_frame"],
                "estimated_mbps": comm_stats["estimated_mbps"],
                "bytes_per_box": comm_stats["bytes_per_box"],
                "failure_policy": self.pact_box_packet_nojoint_cfg[
                    "failure_policy"
                ],
                "fallback_reason": "; ".join(fallback_reasons),
                "packet_parameter_count": self._packet_parameter_count(),
                "trainable_total": self.pact_box_packet_nojoint_summary[
                    "trainable_total"
                ],
                "frozen_bn_eval_count": self.pact_box_packet_nojoint_summary[
                    "frozen_bn_eval_count"
                ],
                "per_agent_box_count": per_agent_box_count,
            }
            self._assert_strict_boundary(debug)
            output_dict.update({
                "cls_preds": torch.cat(
                    ego_cls_predictions, dim=0
                ).detach(),
                "reg_preds": torch.cat(
                    ego_reg_predictions, dim=0
                ).detach(),
                "dir_preds": torch.cat(
                    ego_dir_predictions, dim=0
                ).detach(),
                "occ_single_list": self._merge_ego_occ_outputs(
                    ego_occ_outputs
                ),
                "box_packet_nojoint_enabled": True,
                "box_packet_pred_box_tensor": final_boxes.detach(),
                "box_packet_pred_score": final_scores.detach(),
                "pact_box_packet_nojoint_debug": debug,
            })
            return output_dict

    def _run_local_detector(self, agent_features, agent_index, anchor_box,
                            retain_dense=False):
        local_feature, occ_outputs = self.pyramid_backbone.forward_single(
            agent_features[agent_index:agent_index + 1]
        )
        if self.shrink_flag:
            local_feature = self.shrink_conv(local_feature)
        cls_preds = self.cls_head(local_feature)
        reg_preds = self.reg_head(local_feature)
        dir_preds = self.dir_head(local_feature)
        prediction = (
            self.pact_box_packet_nojoint_codec.decode_local_predictions(
                cls_preds, reg_preds, dir_preds, anchor_box
            )
        )
        dense_output = None
        if retain_dense:
            dense_output = {
                "cls_preds": cls_preds.detach(),
                "reg_preds": reg_preds.detach(),
                "dir_preds": dir_preds.detach(),
                "occ_outputs": [item.detach() for item in occ_outputs],
            }
        del local_feature
        del cls_preds
        del reg_preds
        del dir_preds
        del occ_outputs
        return dense_output, prediction

    def _encode_agent_features(self, data_dict, output_dict):
        modality_count = Counter(data_dict["agent_modality_list"])
        modality_features = {}
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count:
                continue
            feature = getattr(self, "encoder_%s" % modality_name)(
                data_dict, modality_name
            )
            feature = getattr(self, "backbone_%s" % modality_name)({
                "spatial_features": feature,
            })["spatial_features_2d"]
            feature = getattr(self, "aligner_%s" % modality_name)(feature)
            if self.sensor_type_dict[modality_name] == "camera":
                _, _, height, width = feature.shape
                feature = torchvision.transforms.CenterCrop((
                    int(height * getattr(
                        self, "crop_ratio_H_%s" % modality_name
                    )),
                    int(width * getattr(
                        self, "crop_ratio_W_%s" % modality_name
                    )),
                ))(feature)
                if getattr(self, "depth_supervision_%s" % modality_name):
                    output_dict["depth_items_%s" % modality_name] = getattr(
                        self, "encoder_%s" % modality_name
                    ).depth_items
            modality_features[modality_name] = feature

        offsets = {name: 0 for name in self.modality_name_list}
        assembled = []
        for modality_name in data_dict["agent_modality_list"]:
            assembled.append(
                modality_features[modality_name][offsets[modality_name]]
            )
            offsets[modality_name] += 1
        return torch.stack(assembled)

    @staticmethod
    def _scene_anchor_box(anchor_box, batch_index, batch_size):
        if torch.is_tensor(anchor_box) and anchor_box.ndim == 5:
            if anchor_box.shape[0] != batch_size:
                raise ValueError("batched anchor_box does not match record_len")
            return anchor_box[batch_index]
        return anchor_box

    @staticmethod
    def _validate_scene_layout(agent_features, record_len, modality_names,
                               pairwise_t_matrix):
        lengths = [
            int(item) for item in record_len.detach().cpu().tolist()
        ]
        if any(length < 1 for length in lengths):
            raise ValueError("record_len entries must be positive")
        if (
                sum(lengths) != agent_features.shape[0]
                or sum(lengths) != len(modality_names)):
            raise ValueError("record_len does not match encoded agent order")
        if (
                pairwise_t_matrix.ndim != 5
                or pairwise_t_matrix.shape[-2:] != (4, 4)):
            raise ValueError("pairwise_t_matrix must be [B,L,L,4,4]")
        if pairwise_t_matrix.shape[0] != len(lengths):
            raise ValueError(
                "pairwise_t_matrix batch does not match record_len"
            )

    @staticmethod
    def _merge_ego_occ_outputs(ego_occ_outputs):
        if not ego_occ_outputs:
            return []
        return [
            torch.cat(
                [scene[level] for scene in ego_occ_outputs], dim=0
            ).detach()
            for level in range(len(ego_occ_outputs[0]))
        ]

    @staticmethod
    def _concatenate_or_empty(parts, reference, empty_shape):
        nonempty = [part for part in parts if part.shape[0] > 0]
        if nonempty:
            return torch.cat(nonempty, dim=0).detach()
        return reference.new_empty(empty_shape).detach()

    def _freeze_all_parameters_and_eval(self):
        nn.Module.train(self, False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _packet_parameter_count(self):
        return (
            self.pact_box_packet_nojoint_codec.parameter_count()
            + self.pact_box_packet_nojoint_comm_meter.parameter_count()
        )

    def _parameter_summary(self):
        return {
            "trainable_total": sum(
                parameter.numel() for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "packet_parameter_count": self._packet_parameter_count(),
            "frozen_bn_eval_count": sum(
                int(not module.training) for module in self.modules()
                if isinstance(module, self.BN_TYPES)
            ),
        }

    def _assert_strict_boundary(self, debug):
        if not self.pact_box_packet_nojoint_cfg["packet_only_strict"]:
            return
        valid = (
            debug["box_packet_only_verified"]
            and debug["no_joint_training_verified"]
            and not debug["stage3_training_required"]
            and not debug["dense_collab_fusion_used"]
            and not debug["forward_collab_used"]
            and not debug["collaborator_dense_after_packet_used"]
            and debug["packet_source"] == PACKET_SOURCE
            and debug["packet_parameter_count"] == 0
            and debug["trainable_total"] == 0
        )
        if not valid:
            raise RuntimeError(
                "PACT box-packet no-joint boundary verification failed"
            )

    @staticmethod
    def _normalize_cfg(cfg):
        normalized = {
            "enabled": False,
            "no_joint_training": True,
            "use_stage3_joint_training": False,
            "trainable": False,
            "packet_only_strict": True,
            "failure_policy": "ego_only",
            "packet_source": PACKET_SOURCE,
            "score_threshold": 0.2,
            "local_nms_thresh": 0.15,
            "global_nms_thresh": 0.15,
            "max_boxes_per_agent": 100,
            "order": "hwl",
            "dir_offset": 0.7853,
            "num_bins": 2,
            "quantize": "fp16",
            "bytes_per_scalar": 2,
            "deadline_ms": 100,
            "bandwidth_budget_kb": 64,
            "debug": False,
        }
        if isinstance(cfg, dict):
            normalized.update(cfg)
        for key in (
                "enabled", "no_joint_training", "use_stage3_joint_training",
                "trainable", "packet_only_strict", "debug"):
            normalized[key] = bool(normalized[key])
        if not normalized["enabled"]:
            raise ValueError(
                "pact_box_packet_nojoint.enabled must be true"
            )
        if (
                not normalized["no_joint_training"]
                or normalized["use_stage3_joint_training"]
                or normalized["trainable"]):
            raise ValueError(
                "PACT box-packet no-joint must remain frozen without Stage3"
            )
        if normalized["failure_policy"] not in ("ego_only", "error"):
            raise ValueError("failure_policy must be ego_only or error")
        if normalized["packet_source"] != PACKET_SOURCE:
            raise ValueError(
                "packet_source must be local_detector_boxes_after_nms"
            )
        normalized["score_threshold"] = float(
            normalized["score_threshold"]
        )
        normalized["local_nms_thresh"] = float(
            normalized["local_nms_thresh"]
        )
        normalized["global_nms_thresh"] = float(
            normalized["global_nms_thresh"]
        )
        normalized["max_boxes_per_agent"] = int(
            normalized["max_boxes_per_agent"]
        )
        normalized["dir_offset"] = float(normalized["dir_offset"])
        normalized["num_bins"] = int(normalized["num_bins"])
        normalized["bytes_per_scalar"] = int(
            normalized["bytes_per_scalar"]
        )
        normalized["deadline_ms"] = float(normalized["deadline_ms"])
        normalized["bandwidth_budget_kb"] = float(
            normalized["bandwidth_budget_kb"]
        )
        return normalized

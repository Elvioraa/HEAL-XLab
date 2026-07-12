"""No-joint PACT packet inference using frozen local experts only."""

from collections import Counter
import sys
import types

import torch
import torch.nn as nn
import torchvision

if "icecream" not in sys.modules:
    icecream_stub = types.ModuleType("icecream")
    icecream_stub.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
    sys.modules["icecream"] = icecream_stub
if "timm.models.layers" not in sys.modules:
    timm_stub = types.ModuleType("timm")
    timm_models_stub = types.ModuleType("timm.models")
    timm_layers_stub = types.ModuleType("timm.models.layers")

    class _DropPath(nn.Identity):
        pass

    timm_layers_stub.DropPath = _DropPath
    timm_stub.models = timm_models_stub
    timm_models_stub.layers = timm_layers_stub
    sys.modules.setdefault("timm", timm_stub)
    sys.modules.setdefault("timm.models", timm_models_stub)
    sys.modules.setdefault("timm.models.layers", timm_layers_stub)
if "einops" not in sys.modules:
    einops_stub = types.ModuleType("einops")
    einops_stub.rearrange = lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("einops is required"))
    einops_stub.repeat = einops_stub.rearrange
    sys.modules.setdefault("einops", einops_stub)
if "shapely.geometry" not in sys.modules:
    shapely_stub = types.ModuleType("shapely")
    shapely_geometry_stub = types.ModuleType("shapely.geometry")

    class _Polygon:
        def __init__(self, *args, **kwargs):
            self.area = 0.0

        def intersection(self, other):
            return self

        def union(self, other):
            return self

    shapely_geometry_stub.Polygon = _Polygon
    shapely_stub.geometry = shapely_geometry_stub
    sys.modules.setdefault("shapely", shapely_stub)
    sys.modules.setdefault("shapely.geometry", shapely_geometry_stub)
if "pyquaternion" not in sys.modules:
    pyquaternion_stub = types.ModuleType("pyquaternion")

    class _Quaternion:
        def __init__(self, *args, **kwargs):
            pass

        @property
        def transformation_matrix(self):
            return torch.eye(4).numpy()

    pyquaternion_stub.Quaternion = _Quaternion
    sys.modules.setdefault("pyquaternion", pyquaternion_stub)

from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.sub_modules.pact_cbea_evidence_head import PACTCBEALocalEvidenceHead
from opencood.models.sub_modules.pact_cbea_packet_nojoint import (
    PACKET_SOURCE,
    PACTNoJointCommunicationMeter,
    PACTNoJointPacketAggregator,
    PACTNoJointPacketizer,
)
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeterPyramidCollabPactCbeaPacketNojoint(HeterPyramidCollab):
    """Inference-only PACT top-K evidence packets with no Stage3 training."""

    BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def __init__(self, args):
        super().__init__(args)
        self.pact_packet_nojoint_cfg = self._normalize_cfg(args.get("pact_packet_nojoint"))
        if not self.pact_packet_nojoint_cfg["enabled"]:
            raise ValueError("PACT no-joint packet model requires pact_packet_nojoint.enabled=true")
        head_cfg = self.pact_packet_nojoint_cfg["evidence_head"]
        for modality_name in self.modality_name_list:
            setattr(
                self,
                "pact_cbea_evidence_head_%s" % modality_name,
                PACTCBEALocalEvidenceHead(
                    in_channels=int(head_cfg["in_channels"]),
                    hidden_dim=int(head_cfg["hidden_dim"]),
                    descriptor_dim=int(head_cfg["descriptor_dim"]),
                    use_sigmoid=bool(head_cfg["use_sigmoid"]),
                    normalize_descriptor=bool(head_cfg["normalize_descriptor"]),
                    return_feature=False,
                ),
            )
        self.pact_packet_nojoint_packetizer = PACTNoJointPacketizer(
            topk=self.pact_packet_nojoint_cfg["topk"],
            confidence_threshold=self.pact_packet_nojoint_cfg["confidence_threshold"],
            quantize=self.pact_packet_nojoint_cfg["quantize"],
        )
        self.pact_packet_nojoint_aggregator = PACTNoJointPacketAggregator(
            modality_prior=self.pact_packet_nojoint_cfg["modality_prior"],
            fixed_gain=self.pact_packet_nojoint_cfg["fixed_gain"],
            collision_reduce=self.pact_packet_nojoint_cfg["collision_reduce"],
        )
        self.pact_packet_nojoint_comm_meter = PACTNoJointCommunicationMeter(
            quantize=self.pact_packet_nojoint_cfg["quantize"],
            deadline_ms=self.pact_packet_nojoint_cfg["deadline_ms"],
            bandwidth_budget_kb=self.pact_packet_nojoint_cfg["bandwidth_budget_kb"],
        )
        self._freeze_all_parameters_and_eval()
        self.pact_packet_nojoint_summary = self._parameter_summary()

    def train(self, mode=True):
        self._freeze_all_parameters_and_eval()
        return self

    def model_train_init(self):
        """Keep the inference-only contract when generic HEAL setup calls this hook."""
        self._freeze_all_parameters_and_eval()

    def forward(self, data_dict):
        with torch.no_grad():
            output_dict = {"pyramid": "collab"}
            agent_modality_list = data_dict["agent_modality_list"]
            record_len = data_dict["record_len"]
            affine_matrix = normalize_pairwise_tfm(
                data_dict["pairwise_t_matrix"], self.H, self.W, self.fake_voxel_size
            )
            try:
                heter_feature_2d = self._encode_agent_features(data_dict, output_dict)
                ego_feature, occ_outputs = self._extract_ego_local_features(
                    heter_feature_2d, record_len
                )
            except Exception as exc:
                raise RuntimeError(
                    "PACT no-joint ego feature generation failed; packet fallback is unavailable: %s" % exc
                ) from exc

            try:
                scene_packets = self._build_scene_packets(
                    heter_feature_2d,
                    record_len,
                    agent_modality_list,
                    affine_matrix,
                    ego_feature,
                )
                enhanced_feature, packet_map, aggregation_debug = self.pact_packet_nojoint_aggregator(
                    ego_feature, scene_packets
                )
                comm_stats = self.pact_packet_nojoint_comm_meter(scene_packets)
                fallback_reason = ""
            except Exception as exc:
                if self.pact_packet_nojoint_cfg["failure_policy"] == "error":
                    raise RuntimeError("PACT no-joint packet forward failed: %s" % exc) from exc
                enhanced_feature = ego_feature
                packet_map = ego_feature.new_zeros((ego_feature.shape[0], 1, *ego_feature.shape[-2:]))
                aggregation_debug = {"empty_packet": True, "valid_packet_count": 0}
                comm_stats = self.pact_packet_nojoint_comm_meter([])
                fallback_reason = "%s: %s" % (type(exc).__name__, exc)
            finally:
                # Packet aggregation never receives this multi-agent tensor.
                del heter_feature_2d

            debug = {
                "packet_only_verified": True,
                "no_joint_training_verified": True,
                "stage3_training_required": False,
                "dense_collab_fusion_used": False,
                "collaborator_dense_after_packet_used": False,
                "full_evidence_map_transmitted": False,
                "packet_source": PACKET_SOURCE,
                "packet_count": comm_stats["packet_count"],
                "packet_bytes_per_frame": comm_stats["packet_bytes_per_frame"],
                "packet_kb_per_frame": comm_stats["packet_kb_per_frame"],
                "estimated_mbps": comm_stats["estimated_mbps"],
                "topk": self.pact_packet_nojoint_cfg["topk"],
                "failure_policy": self.pact_packet_nojoint_cfg["failure_policy"],
                "fallback_reason": fallback_reason,
                "packet_parameter_count": self._packet_parameter_count(),
                "trainable_total": self.pact_packet_nojoint_summary["trainable_total"],
                "frozen_bn_eval_count": self.pact_packet_nojoint_summary["frozen_bn_eval_count"],
                "packet_aggregation": aggregation_debug,
                "packet_evidence_map": packet_map,
            }
            self._assert_strict_boundary(debug)
            output_dict.update({
                "cls_preds": self.cls_head(enhanced_feature),
                "reg_preds": self.reg_head(enhanced_feature),
                "dir_preds": self.dir_head(enhanced_feature),
                "occ_single_list": occ_outputs,
                "pact_packet_nojoint_debug": debug,
            })
            return output_dict

    def _encode_agent_features(self, data_dict, output_dict):
        modality_count = Counter(data_dict["agent_modality_list"])
        modal_features = {}
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count:
                continue
            feature = getattr(self, "encoder_%s" % modality_name)(data_dict, modality_name)
            feature = getattr(self, "backbone_%s" % modality_name)({"spatial_features": feature})[
                "spatial_features_2d"
            ]
            feature = getattr(self, "aligner_%s" % modality_name)(feature)
            if self.sensor_type_dict[modality_name] == "camera":
                _, _, height, width = feature.shape
                feature = torchvision.transforms.CenterCrop((
                    int(height * getattr(self, "crop_ratio_H_%s" % modality_name)),
                    int(width * getattr(self, "crop_ratio_W_%s" % modality_name)),
                ))(feature)
                if getattr(self, "depth_supervision_%s" % modality_name):
                    output_dict["depth_items_%s" % modality_name] = getattr(
                        self, "encoder_%s" % modality_name
                    ).depth_items
            modal_features[modality_name] = feature
        offsets = {name: 0 for name in self.modality_name_list}
        assembled = []
        for modality_name in data_dict["agent_modality_list"]:
            assembled.append(modal_features[modality_name][offsets[modality_name]])
            offsets[modality_name] += 1
        return torch.stack(assembled)

    def _extract_ego_local_features(self, agent_features, record_len):
        """Build the sole dense ego detection context before packet handling."""
        ego_features = []
        ego_occ_outputs = []
        start = 0
        for length in record_len.detach().cpu().tolist():
            length = int(length)
            if length < 1:
                raise ValueError("record_len entries must be positive")
            ego_feature, ego_occ = self.pyramid_backbone.forward_single(
                agent_features[start:start + 1]
            )
            if self.shrink_flag:
                ego_feature = self.shrink_conv(ego_feature)
            ego_features.append(ego_feature)
            ego_occ_outputs.append(ego_occ)
            start += length

        return torch.cat(ego_features, dim=0), self._merge_ego_occ_outputs(ego_occ_outputs)

    def _build_scene_packets(self, agent_features, record_len, modality_names, affine_matrix,
                             ego_feature):
        """Packetize collaborators after all ego features have been retained."""
        scene_packets = []
        start = 0
        for batch_index, length in enumerate(record_len.detach().cpu().tolist()):
            length = int(length)

            packet_parts = []
            for local_index in range(1, length):
                agent_index = start + local_index
                modality_name = modality_names[agent_index]
                collaborator_feature, _ = self.pyramid_backbone.forward_single(
                    agent_features[agent_index:agent_index + 1]
                )
                if self.shrink_flag:
                    collaborator_feature = self.shrink_conv(collaborator_feature)
                head_output = getattr(self, "pact_cbea_evidence_head_%s" % modality_name)(
                    collaborator_feature
                )
                packet_parts.append(self.pact_packet_nojoint_packetizer(
                    head_output["evidence_heatmap"],
                    head_output["evidence_uncertainty"],
                    modality_name=modality_name,
                    agent_index=agent_index,
                    agent_to_ego=affine_matrix[batch_index, 0, local_index],
                ))
                del collaborator_feature
                del head_output
            scene_packets.append(self._merge_packets(
                packet_parts, ego_feature[batch_index:batch_index + 1]
            ))
            start += length
        return scene_packets

    @staticmethod
    def _merge_ego_occ_outputs(ego_occ_outputs):
        if not ego_occ_outputs:
            return []
        return [
            torch.cat([scene_outputs[level] for scene_outputs in ego_occ_outputs], dim=0)
            for level in range(len(ego_occ_outputs[0]))
        ]

    @staticmethod
    def _merge_packets(packet_parts, reference):
        if not packet_parts:
            device, dtype = reference.device, reference.dtype
            return {
                "coordinates": torch.empty((0, 2), device=device, dtype=dtype),
                "confidence": torch.empty((0, 1), device=device, dtype=dtype),
                "uncertainty": torch.empty((0, 1), device=device, dtype=dtype),
                "modality_id": torch.empty((0, 1), device=device, dtype=torch.long),
                "agent_id": torch.empty((0, 1), device=device, dtype=torch.long),
                "valid_mask": torch.empty((0,), device=device, dtype=torch.bool),
                "packet_source": PACKET_SOURCE,
            }
        merged = {
            key: torch.cat([packet[key] for packet in packet_parts], dim=0)
            for key in ("coordinates", "confidence", "uncertainty", "modality_id", "agent_id", "valid_mask")
        }
        merged["packet_source"] = PACKET_SOURCE
        return merged

    def _freeze_all_parameters_and_eval(self):
        # Call nn.Module directly so this wrapper's train override cannot
        # recursively re-enter the freeze path.
        nn.Module.train(self, False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _parameter_summary(self):
        frozen_bn = sum(
            int(not module.training) for module in self.modules() if isinstance(module, self.BN_TYPES)
        )
        return {
            "trainable_total": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "packet_parameter_count": self._packet_parameter_count(),
            "frozen_bn_eval_count": frozen_bn,
        }

    def _packet_parameter_count(self):
        modules = (
            self.pact_packet_nojoint_packetizer,
            self.pact_packet_nojoint_aggregator,
            self.pact_packet_nojoint_comm_meter,
        )
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    def _assert_strict_boundary(self, debug):
        if not self.pact_packet_nojoint_cfg["packet_only_strict"]:
            return
        required = (
            debug["packet_only_verified"],
            debug["no_joint_training_verified"],
            not debug["stage3_training_required"],
            not debug["dense_collab_fusion_used"],
            not debug["collaborator_dense_after_packet_used"],
            not debug["full_evidence_map_transmitted"],
        )
        if not all(required):
            raise RuntimeError("PACT no-joint packet communication boundary failed")

    def load_state_dict(self, state_dict, strict=True):
        own_state = nn.Module.state_dict(self)
        normalized = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        compatible = {
            key: value for key, value in normalized.items()
            if key in own_state and tuple(value.shape) == tuple(own_state[key].shape)
        }
        report = {
            "unexpected_missing_keys": sorted(set(own_state) - set(compatible)),
            "unexpected_checkpoint_keys": sorted(set(normalized) - set(own_state)),
            "packet_parameter_count": self._packet_parameter_count(),
        }
        self.pact_packet_nojoint_checkpoint_report = report
        print("unexpected_missing_keys:", report["unexpected_missing_keys"])
        print("unexpected_checkpoint_keys:", report["unexpected_checkpoint_keys"])
        print("packet_parameter_count:", report["packet_parameter_count"])
        if report["unexpected_missing_keys"] or report["unexpected_checkpoint_keys"]:
            raise RuntimeError("PACT no-joint checkpoint must exactly match the four local experts")
        return nn.Module.load_state_dict(self, compatible, strict=False)

    @staticmethod
    def _normalize_cfg(cfg):
        defaults = {
            "enabled": False,
            "no_joint_training": True,
            "use_stage3_joint_training": False,
            "trainable": False,
            "packet_only_strict": True,
            "failure_policy": "ego_only",
            "packet_source": PACKET_SOURCE,
            "topk": 50,
            "use_descriptor": False,
            "confidence_threshold": 0.15,
            "quantize": "fp16",
            "fixed_gain": 0.1,
            "collision_reduce": "max",
            "modality_prior": {"m1": 1.0, "m2": 1.0, "m3": 1.0, "m4": 1.0},
            "bandwidth_budget_kb": 8,
            "deadline_ms": 100,
            "debug": False,
            "evidence_head": {
                "in_channels": 256,
                "hidden_dim": 64,
                "descriptor_dim": 16,
                "use_sigmoid": True,
                "normalize_descriptor": True,
            },
        }
        if isinstance(cfg, dict):
            for key, value in cfg.items():
                if key == "evidence_head" and isinstance(value, dict):
                    defaults[key].update(value)
                else:
                    defaults[key] = value
        for key in ("enabled", "no_joint_training", "use_stage3_joint_training", "trainable", "packet_only_strict", "use_descriptor", "debug"):
            defaults[key] = bool(defaults[key])
        if not defaults["no_joint_training"] or defaults["use_stage3_joint_training"] or defaults["trainable"]:
            raise ValueError("PACT no-joint packet inference must remain frozen with no Stage3")
        if defaults["failure_policy"] not in ("ego_only", "error"):
            raise ValueError("failure_policy must be ego_only or error")
        if defaults["packet_source"] != PACKET_SOURCE or defaults["use_descriptor"]:
            raise ValueError("PACT no-joint v1 uses topk_local_evidence without descriptors")
        defaults["topk"] = int(defaults["topk"])
        defaults["confidence_threshold"] = float(defaults["confidence_threshold"])
        defaults["fixed_gain"] = float(defaults["fixed_gain"])
        defaults["bandwidth_budget_kb"] = float(defaults["bandwidth_budget_kb"])
        defaults["deadline_ms"] = int(defaults["deadline_ms"])
        return defaults

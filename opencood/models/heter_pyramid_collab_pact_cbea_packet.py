"""Independent PACT-CBEA packet-only collaborative perception model."""

from collections import Counter
from contextlib import nullcontext
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

    def _missing_einops(*args, **kwargs):
        raise ImportError("einops is required for this optional HEAL component")

    einops_stub.rearrange = _missing_einops
    einops_stub.repeat = _missing_einops
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
from opencood.models.sub_modules.pact_cbea_packet import (
    PACKET_SOURCE,
    PACTPacketAggregator,
    PACTPacketCommunicationMeter,
    PACTPacketCompressor,
    PACTPacketResidualFusion,
    PACTPacketizer,
    collate_scene_packets,
    flatten_agent_packets,
)
from opencood.utils.model_utils import check_trainable_module


class HeterPyramidCollabPactCbeaPacket(HeterPyramidCollab):
    """Use collaborator BEV only to create packets, never for dense fusion."""

    PACKET_TRAINABLE_PREFIXES = (
        "pact_packetizer",
        "pact_packet_compressor",
        "pact_packet_aggregator",
        "pact_packet_residual_fusion",
        "pact_packet_comm_meter",
    )
    BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def __init__(self, args):
        super().__init__(args)
        self.packet_cfg = self._normalize_packet_cfg(args.get("packet"))
        self.train_only_packet = bool(args.get("train_only_packet", False))
        self.packet_only_strict = bool(self.packet_cfg["packet_only_strict"])
        self.packet_failure_policy = self.packet_cfg["packet_failure_policy"]
        self.supervise_single = bool(args.get("supervise_single", False))
        if self.supervise_single and self.packet_only_strict:
            raise ValueError("packet_only_strict forbids supervise_single dense collaborator heads")

        raw_channels = int(self.packet_cfg.get(
            "in_channels",
            args.get("fusion_backbone", {}).get("num_filters", [64])[0],
        ))
        context_channels = int(self.packet_cfg.get("context_channels", args["in_head"]))
        self.pact_packetizer = PACTPacketizer(
            in_channels=raw_channels,
            packet_dim=self.packet_cfg["packet_dim"],
            descriptor_dim=self.packet_cfg["descriptor_dim"],
            topk=self.packet_cfg["topk"],
            send_uncertainty=self.packet_cfg["send_uncertainty"],
            send_agent_quality=self.packet_cfg["send_agent_quality"],
            send_timestamp=self.packet_cfg["send_timestamp"],
        )
        self.pact_packet_compressor = PACTPacketCompressor(
            quantize=self.packet_cfg["quantize"],
            bandwidth_budget_kb=self.packet_cfg["bandwidth_budget_kb"],
            detach_packet=self.packet_cfg["detach_packet"],
        )
        self.pact_packet_aggregator = PACTPacketAggregator(
            context_channels=context_channels,
            packet_dim=self.packet_cfg["packet_dim"],
            descriptor_dim=self.packet_cfg["descriptor_dim"],
            hidden_dim=self.packet_cfg["hidden_dim"],
        )
        self.pact_packet_residual_fusion = PACTPacketResidualFusion(
            alpha_init=self.packet_cfg["residual_alpha_init"],
            alpha_max=self.packet_cfg["residual_alpha_max"],
        )
        self.pact_packet_comm_meter = PACTPacketCommunicationMeter(
            deadline_ms=self.packet_cfg["deadline_ms"],
        )
        if self.train_only_packet:
            self._freeze_nonpacket_parameters()
            self._set_frozen_modules_eval()
        self.packet_trainable_summary = self._summarize_trainable_parameters()
        check_trainable_module(self)

    def model_train_init(self):
        super().model_train_init()
        if getattr(self, "train_only_packet", False):
            self._freeze_nonpacket_parameters()
            self._set_frozen_modules_eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and self.train_only_packet:
            self._set_frozen_modules_eval()
        return self

    def forward(self, data_dict):
        if not self.packet_cfg["enabled"]:
            return self._handle_packet_failure(
                "packet.disabled",
                ego_context_feature=self._ego_feature_without_packets(data_dict),
                occ_outputs=[],
            )
        if self.compress and self.packet_only_strict:
            raise RuntimeError("packet_only_strict forbids dense compressor on collaborator features")

        output_dict = {"pyramid": "collab"}
        grad_context = nullcontext() if not self.train_only_packet else torch.no_grad()
        with grad_context:
            heter_feature_2d = self._encode_agent_features(data_dict, output_dict)
            record_len = data_dict["record_len"]
            ego_raw_feature = self._extract_ego_features(heter_feature_2d, record_len)
            ego_context_feature, occ_outputs = self.pyramid_backbone.forward_single(ego_raw_feature)
            if self.shrink_flag:
                ego_context_feature = self.shrink_conv(ego_context_feature)

        try:
            packet, comm_stats = self._packetize_collaborators(
                heter_feature_2d,
                record_len,
                data_dict,
            )
            # The packet is the only collaborator-derived value beyond this point.
            del heter_feature_2d
            residual, aggregation_debug = self.pact_packet_aggregator(
                ego_context_feature,
                packet,
            )
            fused_feature, residual_alpha = self.pact_packet_residual_fusion(
                ego_context_feature,
                residual,
            )
        except Exception as exc:
            return self._handle_packet_failure(
                "%s: %s" % (type(exc).__name__, exc),
                ego_context_feature=ego_context_feature,
                occ_outputs=occ_outputs,
                output_dict=output_dict,
            )

        debug = {
            "packet_only_verified": True,
            "dense_collab_fusion_used": False,
            "collaborator_dense_after_packet_used": False,
            "packet_source": PACKET_SOURCE,
            "packet_count": comm_stats["packet_count"],
            "packet_bytes_per_frame": comm_stats["packet_bytes_per_frame"],
            "packet_kb_per_frame": comm_stats["packet_kb_per_frame"],
            "estimated_mbps": comm_stats["estimated_mbps"],
            "bandwidth_budget_kb": comm_stats["bandwidth_budget_kb"],
            "bandwidth_saturated": comm_stats["bandwidth_saturated"],
            "trainable_summary": self.packet_trainable_summary,
            "failure_policy": self.packet_failure_policy,
            "fallback_reason": "",
            "packet_residual_alpha": residual_alpha.detach(),
            "packet_aggregation": aggregation_debug,
        }
        self._assert_packet_boundary(debug)
        output_dict.update({
            "cls_preds": self.cls_head(fused_feature),
            "reg_preds": self.reg_head(fused_feature),
            "dir_preds": self.dir_head(fused_feature),
            "occ_single_list": occ_outputs,
            "pact_packet_debug": debug,
        })
        return output_dict

    def _encode_agent_features(self, data_dict, output_dict):
        agent_modality_list = data_dict["agent_modality_list"]
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = getattr(self, "encoder_%s" % modality_name)(data_dict, modality_name)
            feature = getattr(self, "backbone_%s" % modality_name)({
                "spatial_features": feature,
            })["spatial_features_2d"]
            feature = getattr(self, "aligner_%s" % modality_name)(feature)
            modality_feature_dict[modality_name] = feature

        for modality_name in self.modality_name_list:
            if (
                modality_name not in modality_count_dict
                or self.sensor_type_dict[modality_name] != "camera"
            ):
                continue
            feature = modality_feature_dict[modality_name]
            _, _, height, width = feature.shape
            target_h = int(height * getattr(self, "crop_ratio_H_%s" % modality_name))
            target_w = int(width * getattr(self, "crop_ratio_W_%s" % modality_name))
            modality_feature_dict[modality_name] = torchvision.transforms.CenterCrop(
                (target_h, target_w)
            )(feature)
            if getattr(self, "depth_supervision_%s" % modality_name):
                output_dict["depth_items_%s" % modality_name] = getattr(
                    self,
                    "encoder_%s" % modality_name,
                ).depth_items

        indices = {name: 0 for name in self.modality_name_list}
        assembled = []
        for modality_name in agent_modality_list:
            assembled.append(modality_feature_dict[modality_name][indices[modality_name]])
            indices[modality_name] += 1
        return torch.stack(assembled)

    @staticmethod
    def _extract_ego_features(agent_feature, record_len):
        rows = []
        start = 0
        for length in record_len.detach().cpu().tolist():
            length = int(length)
            if length < 1:
                raise ValueError("record_len must contain positive scene lengths")
            rows.append(agent_feature[start])
            start += length
        return torch.stack(rows, dim=0)

    def _packetize_collaborators(self, agent_feature, record_len, data_dict):
        scene_packets = []
        scene_stats = []
        start = 0
        qualities = data_dict.get("agent_quality")
        timestamps = data_dict.get("timestamp")
        for scene_index, length in enumerate(record_len.detach().cpu().tolist()):
            length = int(length)
            collaborator_feature = agent_feature[start + 1:start + length]
            if collaborator_feature.shape[0] == 0:
                packet = self.pact_packetizer.empty_packet(
                    agent_feature.device,
                    agent_feature.dtype,
                    agents=0,
                )
                packet["comm_stats"] = {
                    "packet_num": 0,
                    "bytes_per_frame": 0,
                    "bandwidth_budget_kb": self.packet_cfg["bandwidth_budget_kb"],
                    "quantize_mode": self.packet_cfg["quantize"],
                    "budget_saturated": False,
                }
            else:
                packet = self.pact_packetizer(
                    collaborator_feature,
                    agent_quality=self._scene_collaborator_scalar(qualities, start, length, scene_index),
                    timestamp=self._scene_collaborator_scalar(timestamps, start, length, scene_index),
                )
                packet = self.pact_packet_compressor(packet)
            scene_stats.append(packet["comm_stats"])
            scene_packets.append(flatten_agent_packets(packet))
            start += length
        return collate_scene_packets(scene_packets), self.pact_packet_comm_meter(scene_stats)

    @staticmethod
    def _scene_collaborator_scalar(value, start, length, scene_index):
        if value is None:
            return None
        value = torch.as_tensor(value)
        if value.numel() == 1:
            return value
        if value.ndim == 1 and value.numel() >= start + length:
            return value[start + 1:start + length]
        if value.ndim >= 1 and value.shape[0] > scene_index:
            return value[scene_index]
        return None

    def _ego_feature_without_packets(self, data_dict):
        grad_context = nullcontext() if not self.train_only_packet else torch.no_grad()
        with grad_context:
            output_dict = {}
            agent_feature = self._encode_agent_features(data_dict, output_dict)
            ego_feature = self._extract_ego_features(agent_feature, data_dict["record_len"])
            ego_feature, _ = self.pyramid_backbone.forward_single(ego_feature)
            return self.shrink_conv(ego_feature) if self.shrink_flag else ego_feature

    def _handle_packet_failure(self, reason, ego_context_feature, occ_outputs, output_dict=None):
        if self.packet_only_strict or self.packet_failure_policy == "error":
            raise RuntimeError("PACT packet-only forward failed: %s" % reason)
        if self.packet_failure_policy != "ego_only":
            raise RuntimeError("unsupported packet failure policy: %s" % self.packet_failure_policy)
        output_dict = {} if output_dict is None else output_dict
        debug = {
            "packet_only_verified": True,
            "dense_collab_fusion_used": False,
            "collaborator_dense_after_packet_used": False,
            "packet_source": PACKET_SOURCE,
            "packet_count": 0,
            "packet_bytes_per_frame": 0,
            "packet_kb_per_frame": 0.0,
            "estimated_mbps": 0.0,
            "bandwidth_budget_kb": self.packet_cfg["bandwidth_budget_kb"],
            "bandwidth_saturated": False,
            "trainable_summary": self.packet_trainable_summary,
            "failure_policy": self.packet_failure_policy,
            "fallback_reason": reason,
        }
        self._assert_packet_boundary(debug)
        output_dict.update({
            "pyramid": "collab",
            "cls_preds": self.cls_head(ego_context_feature),
            "reg_preds": self.reg_head(ego_context_feature),
            "dir_preds": self.dir_head(ego_context_feature),
            "occ_single_list": occ_outputs,
            "pact_packet_debug": debug,
        })
        return output_dict

    def _freeze_nonpacket_parameters(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(self._is_packet_module_name(name))

    def _set_frozen_modules_eval(self):
        self.training = True
        for name, module in self.named_modules():
            if not name:
                continue
            module.train(self._is_packet_module_name(name))

    def _summarize_trainable_parameters(self):
        trainable_total = 0
        frozen_total = 0
        for parameter in self.parameters():
            if parameter.requires_grad:
                trainable_total += parameter.numel()
            else:
                frozen_total += parameter.numel()
        frozen_bn_eval_count = 0
        packet_bn_train_count = 0
        for name, module in self.named_modules():
            if isinstance(module, self.BN_TYPES):
                if self._is_packet_module_name(name):
                    packet_bn_train_count += int(module.training)
                else:
                    frozen_bn_eval_count += int(not module.training)
        return {
            "train_only_packet": self.train_only_packet,
            "trainable_total": trainable_total,
            "frozen_total": frozen_total,
            "trainable_prefix_count": len(self.PACKET_TRAINABLE_PREFIXES),
            "frozen_bn_eval_count": frozen_bn_eval_count,
            "packet_bn_train_count": packet_bn_train_count,
        }

    def _is_packet_module_name(self, name):
        return any(name.startswith(prefix) for prefix in self.PACKET_TRAINABLE_PREFIXES)

    def _assert_packet_boundary(self, debug):
        if not self.packet_only_strict:
            return
        if debug["dense_collab_fusion_used"] or debug["collaborator_dense_after_packet_used"]:
            raise RuntimeError("packet_only_strict detected collaborator dense feature ego fusion")
        if not debug["packet_only_verified"]:
            raise RuntimeError("packet_only_strict verification failed")

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
        missing = sorted(set(own_state) - set(compatible))
        extra = sorted(set(normalized) - set(own_state))
        allowed_missing = [key for key in missing if self._is_packet_module_name(key)]
        unexpected_missing = [key for key in missing if not self._is_packet_module_name(key)]
        unexpected_checkpoint = [key for key in extra if not self._is_packet_module_name(key)]
        report = {
            "allowed_missing_packet_keys": allowed_missing,
            "unexpected_missing_keys": unexpected_missing,
            "unexpected_checkpoint_keys": unexpected_checkpoint,
        }
        self.packet_checkpoint_report = report
        print("allowed_missing_packet_keys:", allowed_missing)
        print("unexpected_missing_keys:", unexpected_missing)
        print("unexpected_checkpoint_keys:", unexpected_checkpoint)
        if unexpected_missing or unexpected_checkpoint:
            raise RuntimeError("PACT packet checkpoint is incompatible with the HEAL expert composition")
        return nn.Module.load_state_dict(self, compatible, strict=False)

    @staticmethod
    def _normalize_packet_cfg(cfg):
        defaults = {
            "enabled": False,
            "packet_only_strict": True,
            "packet_failure_policy": "error",
            "mode": "packet_one_round",
            "source": PACKET_SOURCE,
            "topk": 50,
            "packet_dim": 16,
            "descriptor_dim": 8,
            "quantize": "fp16",
            "send_uncertainty": True,
            "send_agent_quality": True,
            "send_timestamp": True,
            "bandwidth_budget_kb": 8,
            "deadline_ms": 100,
            "detach_packet": False,
            "debug": False,
            "hidden_dim": 64,
            "residual_alpha_init": 0.05,
            "residual_alpha_max": 0.3,
        }
        if isinstance(cfg, dict):
            defaults.update(cfg)
        defaults["enabled"] = bool(defaults["enabled"])
        defaults["packet_only_strict"] = bool(defaults["packet_only_strict"])
        defaults["packet_failure_policy"] = str(defaults["packet_failure_policy"])
        if defaults["packet_failure_policy"] not in ("error", "ego_only"):
            raise ValueError("packet_failure_policy supports only error or ego_only")
        if defaults["source"] != PACKET_SOURCE:
            raise ValueError("PACT packet v1 only supports feature_derived_pseudo_hypothesis")
        for key in ("topk", "packet_dim", "descriptor_dim", "hidden_dim", "deadline_ms"):
            defaults[key] = int(defaults[key])
        for key in ("bandwidth_budget_kb", "residual_alpha_init", "residual_alpha_max"):
            defaults[key] = float(defaults[key])
        for key in ("send_uncertainty", "send_agent_quality", "send_timestamp", "detach_packet", "debug"):
            defaults[key] = bool(defaults[key])
        if defaults["topk"] <= 0 or defaults["packet_dim"] <= 0 or defaults["descriptor_dim"] <= 0:
            raise ValueError("packet topk and dimensions must be positive")
        if not 0.0 < defaults["residual_alpha_init"] < defaults["residual_alpha_max"]:
            raise ValueError("packet residual alpha_init must be inside (0, alpha_max)")
        return defaults

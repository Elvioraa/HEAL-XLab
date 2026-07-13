"""Frozen evidence-routed PACT-CBEA dense BEV collaboration experiment."""

from __future__ import absolute_import, division, print_function

from collections import Counter

import torch
import torch.nn as nn

from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.sub_modules.pact_cbea_evidence_head import (
    PACTCBEALocalEvidenceHead,
)
from opencood.models.sub_modules.pact_cbea_evidence_routed import (
    PACTCBEAEvidenceGeometryAligner,
    PACTCBEAEvidenceRoutingValidator,
)
from opencood.models.sub_modules.pact_cbea_rule import PACTCBEARule
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeterPyramidCollabPactCbeaEvidenceRouted(HeterPyramidCollab):
    """Run frozen local experts and route their aligned evidence to PACT."""

    EVIDENCE_MODALITIES = ("m1", "m2", "m3", "m4")
    BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def __init__(self, args):
        raw_cfg = args.get("pact_cbea_evidence_routed")
        self._validate_config_guard(raw_cfg)
        super(HeterPyramidCollabPactCbeaEvidenceRouted, self).__init__(args)

        self.pact_cbea_evidence_routed_cfg = self._normalize_cfg(raw_cfg)
        head_cfg = self.pact_cbea_evidence_routed_cfg["evidence_head"]
        for modality_name in self.EVIDENCE_MODALITIES:
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

        self.pact_cbea_rule = PACTCBEARule(self.pact_cbea_evidence_routed_cfg)
        align_corners = bool(getattr(self.pyramid_backbone, "align_corners", False))
        self.pact_evidence_geometry_aligner = PACTCBEAEvidenceGeometryAligner(
            align_corners=align_corners,
            invalid_logit=float(
                self.pact_cbea_evidence_routed_cfg.get("invalid_logit", -16.0)
            ),
            invalid_uncertainty=float(
                self.pact_cbea_evidence_routed_cfg.get("invalid_uncertainty", 16.0)
            ),
        )
        self.pact_evidence_routing_validator = PACTCBEAEvidenceRoutingValidator()
        self.evidence_checkpoint_verified = False
        self.evidence_checkpoint_report = {}
        self.supervise_single = bool(args.get("supervise_single", False))
        self._freeze_and_eval()

    @staticmethod
    def _validate_config_guard(cfg):
        if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
            raise ValueError(
                "PACT_CBEA_EVIDENCE_ROUTED_v2 requires "
                "pact_cbea_evidence_routed.enabled=true"
            )
        required_true = (
            "no_joint_training",
            "strict_evidence_routing",
            "require_evidence_checkpoint",
            "align_feature_and_evidence",
            "use_shared_alignment_grid",
            "suppress_invalid_regions",
        )
        for key in required_true:
            if cfg.get(key) is not True:
                raise ValueError(
                    "pact_cbea_evidence_routed.%s must be true" % key
                )
        if cfg.get("use_stage3_joint_training") is not False:
            raise ValueError(
                "pact_cbea_evidence_routed.use_stage3_joint_training must be false"
            )
        if cfg.get("trainable") is not False:
            raise ValueError("pact_cbea_evidence_routed.trainable must be false")

    @classmethod
    def _normalize_cfg(cls, cfg):
        normalized = PACTCBEARule._default_cfg()
        normalized.update({
            "enabled": True,
            "no_joint_training": True,
            "use_stage3_joint_training": False,
            "trainable": False,
            "strict_evidence_routing": True,
            "require_evidence_checkpoint": True,
            "align_feature_and_evidence": True,
            "use_shared_alignment_grid": True,
            "suppress_invalid_regions": True,
            "debug": True,
            "invalid_logit": -16.0,
            "invalid_uncertainty": 16.0,
            "evidence_head": {
                "enabled": True,
                "in_channels": 256,
                "hidden_dim": 64,
                "descriptor_dim": 16,
                "use_sigmoid": True,
                "normalize_descriptor": True,
                "return_feature": False,
            },
        })
        _deep_update(normalized, cfg)
        normalized = PACTCBEARule._normalize_cfg(normalized)
        head_cfg = normalized["evidence_head"]
        head_cfg["enabled"] = bool(head_cfg.get("enabled", True))
        head_cfg["in_channels"] = int(head_cfg.get("in_channels", 256))
        head_cfg["hidden_dim"] = int(head_cfg.get("hidden_dim", 64))
        head_cfg["descriptor_dim"] = int(head_cfg.get("descriptor_dim", 16))
        head_cfg["use_sigmoid"] = bool(head_cfg.get("use_sigmoid", True))
        head_cfg["normalize_descriptor"] = bool(
            head_cfg.get("normalize_descriptor", True)
        )
        head_cfg["return_feature"] = False
        if not head_cfg["enabled"]:
            raise ValueError("All modality-specific evidence heads must be enabled")
        return normalized

    def _freeze_and_eval(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        nn.Module.train(self, False)
        for module in self.modules():
            if isinstance(module, self.BN_TYPES):
                module.eval()

    def train(self, mode=True):
        self._freeze_and_eval()
        return self

    def load_state_dict(self, state_dict, strict=True):
        normalized = {}
        for key, value in state_dict.items():
            normalized_key = key[7:] if key.startswith("module.") else key
            normalized[normalized_key] = value

        model_state = self.state_dict()
        evidence_prefixes = [
            "pact_cbea_evidence_head_%s." % modality
            for modality in self.EVIDENCE_MODALITIES
        ]
        expected_evidence = {
            key: value for key, value in model_state.items()
            if any(key.startswith(prefix) for prefix in evidence_prefixes)
        }
        if not expected_evidence:
            raise RuntimeError("Model has no modality-specific evidence checkpoint keys")

        missing = []
        shape_mismatches = []
        for key, expected in expected_evidence.items():
            if key not in normalized:
                missing.append(key)
            elif tuple(normalized[key].shape) != tuple(expected.shape):
                shape_mismatches.append(
                    "%s checkpoint=%s model=%s"
                    % (key, tuple(normalized[key].shape), tuple(expected.shape))
                )
        unexpected_evidence = [
            key for key in normalized
            if any(key.startswith(prefix) for prefix in evidence_prefixes)
            and key not in expected_evidence
        ]
        if missing or shape_mismatches or unexpected_evidence:
            details = []
            if missing:
                details.append("missing=%s" % ", ".join(missing))
            if shape_mismatches:
                details.append("shape_mismatch=%s" % "; ".join(shape_mismatches))
            if unexpected_evidence:
                details.append(
                    "unexpected_evidence=%s" % ", ".join(unexpected_evidence)
                )
            raise RuntimeError(
                "Strict evidence checkpoint validation failed: %s"
                % " | ".join(details)
            )

        compatible = {}
        non_evidence_shape_mismatches = []
        for key, value in normalized.items():
            if key not in model_state:
                continue
            if tuple(value.shape) != tuple(model_state[key].shape):
                non_evidence_shape_mismatches.append(key)
                continue
            compatible[key] = value

        result = nn.Module.load_state_dict(self, compatible, strict=False)
        self.evidence_checkpoint_verified = True
        self.evidence_checkpoint_report = {
            "verified_modalities": list(self.EVIDENCE_MODALITIES),
            "verified_evidence_key_count": int(len(expected_evidence)),
            "loaded_key_count": int(len(compatible)),
            "unrelated_checkpoint_key_count": int(
                len([key for key in normalized if key not in model_state])
            ),
            "non_evidence_shape_mismatch_count": int(
                len(non_evidence_shape_mismatches)
            ),
        }
        self._freeze_and_eval()
        return result

    def forward(self, data_dict):
        if not self.evidence_checkpoint_verified:
            raise RuntimeError(
                "PACT evidence-routed forward requires a verified m1/m2/m3/m4 "
                "evidence checkpoint"
            )
        with torch.no_grad():
            return self._forward_evidence_routed(data_dict)

    def _forward_evidence_routed(self, data_dict):
        output_dict = {"pyramid": "collab"}
        agent_modality_list = list(data_dict["agent_modality_list"])
        record_len = data_dict["record_len"]
        self.pact_evidence_routing_validator.validate_heads(
            self, self.EVIDENCE_MODALITIES
        )
        for modality_name in agent_modality_list:
            if modality_name not in self.EVIDENCE_MODALITIES:
                raise RuntimeError(
                    "Unsupported evidence modality in agent order: %s" % modality_name
                )

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = getattr(self, "encoder_%s" % modality_name)(
                data_dict, modality_name
            )
            feature = getattr(self, "backbone_%s" % modality_name)({
                "spatial_features": feature,
            })["spatial_features_2d"]
            feature = getattr(self, "aligner_%s" % modality_name)(feature)
            modality_feature_dict[modality_name] = feature

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            if self.sensor_type_dict[modality_name] != "camera":
                continue
            import torchvision

            feature = modality_feature_dict[modality_name]
            height, width = feature.shape[-2:]
            target_height = int(
                height * getattr(self, "crop_ratio_H_%s" % modality_name)
            )
            target_width = int(
                width * getattr(self, "crop_ratio_W_%s" % modality_name)
            )
            crop = torchvision.transforms.CenterCrop((target_height, target_width))
            modality_feature_dict[modality_name] = crop(feature)
            if getattr(self, "depth_supervision_%s" % modality_name):
                output_dict["depth_items_%s" % modality_name] = getattr(
                    self, "encoder_%s" % modality_name
                ).depth_items

        counters = {name: 0 for name in self.modality_name_list}
        ordered_features = []
        for modality_name in agent_modality_list:
            if modality_name not in modality_feature_dict:
                raise RuntimeError(
                    "No encoded feature for modality %s" % modality_name
                )
            index = counters[modality_name]
            ordered_features.append(modality_feature_dict[modality_name][index])
            counters[modality_name] += 1
        heter_feature = torch.stack(ordered_features, dim=0)
        if self.compress:
            heter_feature = self.compressor(heter_feature)

        single_feature, single_occ_outputs = self.pyramid_backbone.forward_single(
            heter_feature
        )
        if self.shrink_flag:
            single_feature = self.shrink_conv(single_feature)

        evidence_logits = []
        evidence_uncertainty = []
        evidence_descriptor = []
        heads_used = []
        for index, modality_name in enumerate(agent_modality_list):
            head = getattr(
                self, "pact_cbea_evidence_head_%s" % modality_name
            )
            evidence = head(single_feature[index:index + 1])
            evidence_logits.append(evidence["evidence_heatmap_logits"])
            evidence_uncertainty.append(evidence["evidence_uncertainty"])
            evidence_descriptor.append(evidence["evidence_descriptor"])
            if modality_name not in heads_used:
                heads_used.append(modality_name)
        heatmap_logits = torch.cat(evidence_logits, dim=0)
        uncertainty = torch.cat(evidence_uncertainty, dim=0)
        descriptor = torch.cat(evidence_descriptor, dim=0)

        self.pact_evidence_routing_validator.validate_agent_outputs(
            single_feature,
            heatmap_logits,
            uncertainty,
            descriptor,
            agent_modality_list,
            record_len,
        )
        affine_matrix = normalize_pairwise_tfm(
            data_dict["pairwise_t_matrix"],
            self.H,
            self.W,
            self.fake_voxel_size,
        )
        aligned = self.pact_evidence_geometry_aligner(
            single_feature,
            heatmap_logits,
            uncertainty,
            descriptor,
            record_len,
            affine_matrix,
        )
        self.pact_evidence_routing_validator.validate_aligned_outputs(aligned)

        fused_feature, rule_debug = self.pact_cbea_rule(
            aligned["feature"],
            evidence_heatmap=aligned["heatmap_logits"],
            evidence_uncertainty=aligned["uncertainty"],
            evidence_descriptor=aligned["descriptor"],
            record_len=record_len,
            pairwise_t_matrix=data_dict["pairwise_t_matrix"],
            modality_names=agent_modality_list,
        )
        fallbacks = self.pact_evidence_routing_validator.validate_rule_debug(
            rule_debug
        )
        alpha_debug = self._compact_alpha_debug(rule_debug)
        if alpha_debug["uniform_fallback_detected"]:
            raise RuntimeError("PACT rule attempted a zero-reliability uniform fallback")

        trainable_total = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )
        if trainable_total != 0:
            raise RuntimeError("Evidence-routed inference requires trainable_total == 0")
        records = self._record_list(record_len)
        debug = {
            "enabled": True,
            "strict_evidence_routing": True,
            "evidence_checkpoint_verified": True,
            "evidence_heads_used": heads_used,
            "evidence_source": "modality_specific_evidence_heads",
            "feature_alignment_used": True,
            "evidence_alignment_used": True,
            "shared_alignment_grid_used": True,
            "forward_collab_used": False,
            "stage3_training_required": False,
            "no_joint_training_verified": True,
            "trainable_total": int(trainable_total),
            "missing_evidence_fallback_used": False,
            "per_scene_agent_count": records,
            "per_agent_valid_ratio": aligned["debug"]["per_agent_valid_ratio"],
            "alpha_min": alpha_debug["alpha_min"],
            "alpha_max": alpha_debug["alpha_max"],
            "alpha_mean": alpha_debug["alpha_mean"],
            "alpha_nonuniform": alpha_debug["alpha_nonuniform"],
            "alpha_sum_max_error": alpha_debug["alpha_sum_max_error"],
            "fallbacks": fallbacks,
        }

        output_dict.update({
            "cls_preds": self.cls_head(fused_feature),
            "reg_preds": self.reg_head(fused_feature),
            "dir_preds": self.dir_head(fused_feature),
            "occ_single_list": single_occ_outputs,
            "pact_cbea_evidence_routed_debug": debug,
        })
        if self.supervise_single:
            output_dict.update({
                "cls_preds_single": self.cls_head(single_feature),
                "reg_preds_single": self.reg_head(single_feature),
                "dir_preds_single": self.dir_head(single_feature),
            })
        return output_dict

    @staticmethod
    def _record_list(record_len):
        if torch.is_tensor(record_len):
            values = record_len.detach().cpu().tolist()
        else:
            values = list(record_len)
        return [int(value) for value in values]

    @classmethod
    def _alpha_tensors(cls, debug):
        if torch.is_tensor(debug.get("pact_alpha")):
            return [debug["pact_alpha"]]
        tensors = []
        for item in debug.get("pact_group_debug", []):
            tensors.extend(cls._alpha_tensors(item))
        return tensors

    @classmethod
    def _reliability_tensors(cls, debug):
        if torch.is_tensor(debug.get("pact_reliability")):
            return [debug["pact_reliability"]]
        tensors = []
        for item in debug.get("pact_group_debug", []):
            tensors.extend(cls._reliability_tensors(item))
        return tensors

    @classmethod
    def _compact_alpha_debug(cls, rule_debug):
        alpha_tensors = cls._alpha_tensors(rule_debug)
        reliability_tensors = cls._reliability_tensors(rule_debug)
        if not alpha_tensors or not reliability_tensors:
            raise RuntimeError("PACT rule debug did not expose alpha/reliability")
        alpha_values = torch.cat([tensor.reshape(-1) for tensor in alpha_tensors])
        nonuniform = False
        sum_errors = []
        for alpha in alpha_tensors:
            if alpha.shape[1] > 1:
                difference = alpha.max(dim=1).values - alpha.min(dim=1).values
                nonuniform = nonuniform or bool((difference > 1e-7).any().item())
            sums = alpha.sum(dim=1)
            sum_errors.append(float((sums - 1.0).abs().max().item()))
        uniform_fallback = False
        for reliability in reliability_tensors:
            denominator = reliability.sum(dim=1)
            if (denominator <= 1e-6).any().item():
                uniform_fallback = True
        return {
            "alpha_min": float(alpha_values.min().item()),
            "alpha_max": float(alpha_values.max().item()),
            "alpha_mean": float(alpha_values.mean().item()),
            "alpha_nonuniform": bool(nonuniform),
            "alpha_sum_max_error": float(max(sum_errors)),
            "uniform_fallback_detected": bool(uniform_fallback),
        }


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

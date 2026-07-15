"""Frozen PACT ablation that forces the original rule boundary to 1/N weights."""

from __future__ import absolute_import, division, print_function

from collections import Counter

import torch
import torch.nn as nn

from opencood.models.heter_pyramid_collab_pact_cbea import (
    HeterPyramidCollabPactCbea,
)
from opencood.models.sub_modules.pact_cbea_original_rule_uniform import (
    PACTCBEAOriginalRuleUniform,
)
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeterPyramidCollabPactCbeaOriginalRuleUniform(HeterPyramidCollabPactCbea):
    """Keep original PACT execution order while forcing its rule output to 1/N."""

    def __init__(self, args):
        raw_cfg = args.get("pact_cbea_original_rule_uniform")
        self._validate_config_guard(raw_cfg)
        super(HeterPyramidCollabPactCbeaOriginalRuleUniform, self).__init__(args)
        self.pact_cbea_original_rule_uniform_cfg = dict(raw_cfg)
        self.pact_cbea_original_rule_uniform = PACTCBEAOriginalRuleUniform()
        self.core_checkpoint_verified = False
        self.core_checkpoint_report = {}
        self._freeze_and_eval()

    @staticmethod
    def _validate_config_guard(cfg):
        if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
            raise ValueError(
                "PACT_CBEA_ORIGINAL_RULE_UNIFORM_v1 requires "
                "pact_cbea_original_rule_uniform.enabled=true"
            )

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

    def _core_module_names(self):
        names = []
        for modality in self.modality_name_list:
            names.extend((
                "encoder_%s" % modality,
                "backbone_%s" % modality,
                "aligner_%s" % modality,
            ))
        names.append("pyramid_backbone")
        if self.shrink_flag:
            names.append("shrink_conv")
        names.extend(("cls_head", "reg_head", "dir_head"))
        if self.compress:
            names.append("compressor")
        return names

    def load_state_dict(self, state_dict, strict=True):
        """Require every shared PACT perception and detection parameter."""
        self.core_checkpoint_verified = False
        normalized = {}
        for key, value in state_dict.items():
            normalized_key = key[7:] if key.startswith("module.") else key
            normalized[normalized_key] = value

        model_state = self.state_dict()
        core_modules = self._core_module_names()
        core_prefixes = tuple("%s." % name for name in core_modules)
        expected_core = {
            key: value for key, value in model_state.items()
            if key.startswith(core_prefixes)
        }
        if not expected_core:
            raise RuntimeError("Original-rule-uniform model has no core checkpoint keys")

        missing = []
        shape_mismatches = []
        for key, expected in expected_core.items():
            value = normalized.get(key)
            if value is None:
                missing.append(key)
            elif tuple(value.shape) != tuple(expected.shape):
                shape_mismatches.append(
                    "%s checkpoint=%s model=%s" % (
                        key, tuple(value.shape), tuple(expected.shape)
                    )
                )
        if missing or shape_mismatches:
            details = []
            if missing:
                details.append("missing=%s" % ", ".join(missing))
            if shape_mismatches:
                details.append("shape_mismatch=%s" % "; ".join(shape_mismatches))
            raise RuntimeError(
                "Strict original-rule-uniform core checkpoint validation failed: %s"
                % " | ".join(details)
            )

        auxiliary_prefix = "pact_cbea_evidence_head_"
        compatible = {
            key: value for key, value in normalized.items() if key in model_state
        }
        result = nn.Module.load_state_dict(self, compatible, strict=False)
        unknown_extra = [
            key for key in normalized
            if key not in model_state and not key.startswith(auxiliary_prefix)
        ]
        auxiliary_extra = [
            key for key in normalized if key.startswith(auxiliary_prefix)
        ]
        self.core_checkpoint_verified = True
        self.core_checkpoint_report = {
            "verified_core_key_count": int(len(expected_core)),
            "loaded_core_key_count": int(len(compatible)),
            "auxiliary_extra_key_count": int(len(auxiliary_extra)),
            "unknown_extra_key_count": int(len(unknown_extra)),
            "unknown_extra_keys": list(unknown_extra[:16]),
        }
        self._freeze_and_eval()
        return result

    def forward(self, data_dict):
        if not self.core_checkpoint_verified:
            raise RuntimeError(
                "Original-rule-uniform forward requires a verified core checkpoint"
            )
        with torch.no_grad():
            return self._forward_original_rule_uniform(data_dict)

    def _forward_original_rule_uniform(self, data_dict):
        output_dict = {"pyramid": "collab"}
        agent_modality_list = list(data_dict["agent_modality_list"])
        record_len = data_dict["record_len"]
        affine_matrix = normalize_pairwise_tfm(
            data_dict["pairwise_t_matrix"],
            self.H,
            self.W,
            self.fake_voxel_size,
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
            modality_feature_dict[modality_name] = getattr(
                self, "aligner_%s" % modality_name
            )(feature)

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            if self.sensor_type_dict[modality_name] != "camera":
                continue
            import torchvision

            feature = modality_feature_dict[modality_name]
            target_h = int(feature.shape[-2] * getattr(
                self, "crop_ratio_H_%s" % modality_name
            ))
            target_w = int(feature.shape[-1] * getattr(
                self, "crop_ratio_W_%s" % modality_name
            ))
            modality_feature_dict[modality_name] = torchvision.transforms.CenterCrop(
                (target_h, target_w)
            )(feature)
            if getattr(self, "depth_supervision_%s" % modality_name):
                output_dict["depth_items_%s" % modality_name] = getattr(
                    self, "encoder_%s" % modality_name
                ).depth_items

        counters = {name: 0 for name in self.modality_name_list}
        ordered_features = []
        for modality_name in agent_modality_list:
            feature_index = counters[modality_name]
            ordered_features.append(
                modality_feature_dict[modality_name][feature_index]
            )
            counters[modality_name] += 1
        heter_feature = torch.stack(ordered_features, dim=0)
        if self.compress:
            heter_feature = self.compressor(heter_feature)

        single_feature, single_occ_outputs = self.pyramid_backbone.forward_single(
            heter_feature
        )
        if self.shrink_flag:
            single_feature = self.shrink_conv(single_feature)

        # Preserve the original PACT execution order, but never select this result.
        base_fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
            heter_feature,
            record_len,
            affine_matrix,
            agent_modality_list,
            self.cam_crop_info,
        )
        if self.shrink_flag:
            base_fused_feature = self.shrink_conv(base_fused_feature)
        del base_fused_feature

        if self.supervise_single:
            output_dict.update({
                "cls_preds_single": self.cls_head(single_feature),
                "reg_preds_single": self.reg_head(single_feature),
                "dir_preds_single": self.dir_head(single_feature),
            })

        # This is the original PACT rule branch transform, not the aligned-uniform
        # geometry module. Evidence heads and data_dict evidence are never queried.
        rule_feature = self._warp_to_ego(single_feature, record_len, affine_matrix)
        pact_feature, uniform_debug = self.pact_cbea_original_rule_uniform(
            rule_feature, record_len
        )

        trainable_total = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )
        if trainable_total != 0:
            raise RuntimeError(
                "Original-rule-uniform inference requires trainable_total == 0"
            )
        pact_debug = {
            "pact_cbea_enabled": True,
            "final_fusion_source": "original_rule_uniform",
            "forward_collab_executed": True,
            "forward_collab_output_used": False,
            "original_pact_rule_warp_used": True,
            "evidence_used": False,
            "external_evidence_read": False,
            "uncertainty_used": False,
            "descriptor_used": False,
            "modality_prior_used": False,
            "aligned_uniform_geometry_used": False,
            "per_scene_agent_count": uniform_debug["per_scene_agent_count"],
            "uniform_weight_min": uniform_debug["uniform_weight_min"],
            "uniform_weight_max": uniform_debug["uniform_weight_max"],
            "uniform_weight_mean": uniform_debug["uniform_weight_mean"],
            "weight_sum_error": uniform_debug["weight_sum_error"],
            "trainable_total": int(trainable_total),
            "router_parameter_count": int(
                self.pact_cbea_original_rule_uniform.parameter_count()
            ),
            "core_checkpoint_verified": True,
            "checkpoint_auxiliary_extra_key_count": self.core_checkpoint_report.get(
                "auxiliary_extra_key_count", 0
            ),
            "checkpoint_unknown_extra_key_count": self.core_checkpoint_report.get(
                "unknown_extra_key_count", 0
            ),
            "checkpoint_unknown_extra_keys": self.core_checkpoint_report.get(
                "unknown_extra_keys", []
            ),
        }
        output_dict.update({
            "cls_preds": self.cls_head(pact_feature),
            "reg_preds": self.reg_head(pact_feature),
            "dir_preds": self.dir_head(pact_feature),
            "occ_single_list": occ_outputs,
            "pact_cbea": pact_debug,
        })
        return output_dict

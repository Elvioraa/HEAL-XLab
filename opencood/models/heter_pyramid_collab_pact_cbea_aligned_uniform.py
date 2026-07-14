"""Frozen aligned-uniform dense BEV collaboration control experiment."""

from __future__ import absolute_import, division, print_function

from collections import Counter

import torch
import torch.nn as nn

from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.sub_modules.pact_cbea_aligned_uniform import (
    PACTCBEAAlignedUniformRouter,
)
from opencood.models.sub_modules.pact_cbea_evidence_routed import (
    PACTCBEAEvidenceGeometryAligner,
)
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeterPyramidCollabPactCbeaAlignedUniform(HeterPyramidCollab):
    """Fuse frozen, geometrically aligned features uniformly over support."""

    MODALITIES = ("m1", "m2", "m3", "m4")
    BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def __init__(self, args):
        raw_cfg = args.get("pact_cbea_aligned_uniform")
        self._validate_config_guard(raw_cfg)
        super(HeterPyramidCollabPactCbeaAlignedUniform, self).__init__(args)

        self.pact_cbea_aligned_uniform_cfg = dict(raw_cfg)
        align_corners = bool(getattr(self.pyramid_backbone, "align_corners", False))
        self.pact_geometry_aligner = PACTCBEAEvidenceGeometryAligner(
            align_corners=align_corners
        )
        self.pact_aligned_uniform_router = PACTCBEAAlignedUniformRouter(
            epsilon=float(raw_cfg.get("epsilon", 1e-6))
        )
        self.supervise_single = bool(args.get("supervise_single", False))
        self.core_checkpoint_verified = False
        self.core_checkpoint_report = {}
        self._freeze_and_eval()

    @staticmethod
    def _validate_config_guard(cfg):
        if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
            raise ValueError(
                "PACT_CBEA_ALIGNED_UNIFORM_v1 requires "
                "pact_cbea_aligned_uniform.enabled=true"
            )
        required_true = (
            "no_joint_training",
            "align_features",
            "use_shared_alignment_grid",
            "uniform_over_valid_support",
            "strict_core_checkpoint",
            "debug",
        )
        for key in required_true:
            if cfg.get(key) is not True:
                raise ValueError("pact_cbea_aligned_uniform.%s must be true" % key)
        required_false = (
            "use_stage3_joint_training",
            "trainable",
            "packet_only",
        )
        for key in required_false:
            if cfg.get(key) is not False:
                raise ValueError("pact_cbea_aligned_uniform.%s must be false" % key)

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
        for modality in self.MODALITIES:
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
        self.core_checkpoint_verified = False
        self.core_checkpoint_report = {}
        normalized = {}
        for key, value in state_dict.items():
            normalized_key = key[7:] if key.startswith("module.") else key
            normalized[normalized_key] = value

        core_modules = self._core_module_names()
        absent_modules = [name for name in core_modules if not hasattr(self, name)]
        if absent_modules:
            raise RuntimeError(
                "Aligned-uniform model is missing core modules: %s"
                % ", ".join(absent_modules)
            )
        model_state = self.state_dict()
        core_prefixes = tuple("%s." % name for name in core_modules)
        expected_core = {
            key: value for key, value in model_state.items()
            if key.startswith(core_prefixes)
        }
        if not expected_core:
            raise RuntimeError("Aligned-uniform model has no core checkpoint keys")

        missing = []
        shape_mismatches = []
        for key, expected in expected_core.items():
            if key not in normalized:
                missing.append(key)
            elif tuple(normalized[key].shape) != tuple(expected.shape):
                shape_mismatches.append(
                    "%s checkpoint=%s model=%s"
                    % (key, tuple(normalized[key].shape), tuple(expected.shape))
                )
        if missing or shape_mismatches:
            details = []
            if missing:
                details.append("missing=%s" % ", ".join(missing))
            if shape_mismatches:
                details.append("shape_mismatch=%s" % "; ".join(shape_mismatches))
            raise RuntimeError(
                "Strict aligned-uniform core checkpoint validation failed: %s"
                % " | ".join(details)
            )

        permitted_auxiliary_prefix = "pact_cbea_evidence_head_"
        auxiliary_extra_keys = [
            key for key in normalized if key.startswith(permitted_auxiliary_prefix)
        ]
        unknown_extra_keys = [
            key for key in normalized
            if key not in model_state
            and not key.startswith(permitted_auxiliary_prefix)
        ]
        compatible = {
            key: normalized[key] for key in expected_core
        }
        result = nn.Module.load_state_dict(self, compatible, strict=False)
        self.core_checkpoint_verified = True
        self.core_checkpoint_report = {
            "verified_core_module_count": int(len(core_modules)),
            "verified_core_key_count": int(len(expected_core)),
            "loaded_core_key_count": int(len(compatible)),
            "auxiliary_extra_key_count": int(len(auxiliary_extra_keys)),
            "unknown_extra_key_count": int(len(unknown_extra_keys)),
            "unknown_extra_keys": list(unknown_extra_keys[:16]),
        }
        self._freeze_and_eval()
        return result

    def forward(self, data_dict):
        if not self.core_checkpoint_verified:
            raise RuntimeError(
                "Aligned-uniform forward requires a verified core checkpoint"
            )
        with torch.no_grad():
            return self._forward_aligned_uniform(data_dict)

    def _forward_aligned_uniform(self, data_dict):
        output_dict = {"pyramid": "collab"}
        agent_modality_list = list(data_dict["agent_modality_list"])
        record_len = data_dict["record_len"]
        for modality_name in agent_modality_list:
            if modality_name not in self.MODALITIES:
                raise RuntimeError(
                    "Unsupported aligned-uniform modality: %s" % modality_name
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

        affine_matrix = normalize_pairwise_tfm(
            data_dict["pairwise_t_matrix"],
            self.H,
            self.W,
            self.fake_voxel_size,
        )
        scratch_zero = single_feature.new_zeros(
            (single_feature.shape[0], 1, single_feature.shape[2], single_feature.shape[3])
        )
        scratch_positive = torch.ones_like(scratch_zero)
        aligned = self.pact_geometry_aligner(
            single_feature,
            scratch_zero,
            scratch_positive,
            scratch_zero,
            record_len,
            affine_matrix,
        )
        fused_feature, router_debug = self.pact_aligned_uniform_router(
            aligned["feature"], aligned["validity"], record_len
        )

        trainable_total = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad
        )
        if trainable_total != 0:
            raise RuntimeError("Aligned-uniform inference requires trainable_total == 0")
        debug = {
            "enabled": True,
            "routing_mode": "aligned_uniform",
            "dense_feature_collaboration": True,
            "weak_communication_claimed": False,
            "uniform_over_valid_support": True,
            "geometry_alignment_used": True,
            "shared_alignment_grid_used": True,
            "validity_mask_used": True,
            "evidence_used": False,
            "uncertainty_used": False,
            "descriptor_used": False,
            "modality_prior_used": False,
            "forward_collab_used": False,
            "stage3_training_required": False,
            "no_joint_training_verified": True,
            "core_checkpoint_verified": True,
            "trainable_total": int(trainable_total),
            "router_parameter_count": int(
                self.pact_aligned_uniform_router.parameter_count()
            ),
            "alpha_sum_verified": router_debug["alpha_sum_verified"],
            "alpha_min": router_debug["alpha_min"],
            "alpha_max": router_debug["alpha_max"],
            "alpha_mean": router_debug["alpha_mean"],
            "per_scene_agent_count": router_debug["per_scene_agent_count"],
            "per_agent_valid_ratio": router_debug["per_agent_valid_ratio"],
            "checkpoint_auxiliary_extra_key_count": self.core_checkpoint_report.get(
                "auxiliary_extra_key_count", 0
            ),
            "checkpoint_unknown_extra_key_count": self.core_checkpoint_report.get(
                "unknown_extra_key_count", 0
            ),
            "checkpoint_unknown_extra_keys": self.core_checkpoint_report.get(
                "unknown_extra_keys", []
            ),
            "fallbacks": [],
        }

        output_dict.update({
            "cls_preds": self.cls_head(fused_feature),
            "reg_preds": self.reg_head(fused_feature),
            "dir_preds": self.dir_head(fused_feature),
            "occ_single_list": single_occ_outputs,
            "pact_cbea_aligned_uniform_debug": debug,
        })
        if self.supervise_single:
            output_dict.update({
                "cls_preds_single": self.cls_head(single_feature),
                "reg_preds_single": self.reg_head(single_feature),
                "dir_preds_single": self.dir_head(single_feature),
            })
        return output_dict

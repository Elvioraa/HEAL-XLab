"""PACT-CBEA v1 Stage1 wrapper for train-from-scratch HEAL base training.

This class deliberately follows :class:`HeterPyramidCollab` for the complete
intermediate-fusion detection path.  It only appends the m1 local evidence
head, so the Stage1 detector, pyramid fusion backbone, and heads all retain
the official random-initialization training behaviour.
"""

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

    def _rearrange(tensor, pattern, **kwargs):
        if pattern == "b (l c) h w -> b l c h w" and "l" in kwargs:
            batch_size, channels, height, width = tensor.shape
            length = int(kwargs["l"])
            return tensor.view(batch_size, length, channels // length, height, width)
        raise ImportError("einops is required for rearrange pattern: %s" % pattern)

    def _repeat(tensor, pattern, **kwargs):
        if pattern == "b h w c l -> b (h new_h) (w new_w) c l":
            return tensor.repeat_interleave(int(kwargs["new_h"]), dim=1).repeat_interleave(
                int(kwargs["new_w"]), dim=2
            )
        raise ImportError("einops is required for repeat pattern: %s" % pattern)

    einops_stub.rearrange = _rearrange
    einops_stub.repeat = _repeat
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
from opencood.models.sub_modules.pact_cbea_rule import PACTCBEARule
from opencood.utils.model_utils import check_trainable_module
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeterPyramidCollabPactCbeaStage1(HeterPyramidCollab):
    """Official HEAL Stage1 base training with an m1 local evidence head."""

    def __init__(self, args):
        super().__init__(args)
        self.supervise_single = bool(args.get("supervise_single", False))
        self.pact_cbea_cfg = self._normalize_pact_stage1_cfg(args.get("pact_cbea"))
        self.pact_cbea_enabled = bool(self.pact_cbea_cfg["enabled"])
        self.pact_cbea_rule = PACTCBEARule(self.pact_cbea_cfg)
        for param in self.pact_cbea_rule.parameters():
            param.requires_grad_(False)

        if self._is_stage1_local_evidence_enabled():
            modality_name = self._configured_modality(args)
            head_cfg = self.pact_cbea_cfg["evidence_head"]
            setattr(
                self,
                "pact_cbea_evidence_head_%s" % modality_name,
                PACTCBEALocalEvidenceHead(
                    in_channels=int(head_cfg.get("in_channels", args.get("in_head", 256))),
                    hidden_dim=int(head_cfg.get("hidden_dim", 64)),
                    descriptor_dim=int(head_cfg.get("descriptor_dim", 16)),
                    use_sigmoid=bool(head_cfg.get("use_sigmoid", True)),
                    normalize_descriptor=bool(head_cfg.get("normalize_descriptor", True)),
                    return_feature=bool(head_cfg.get("return_feature", True)),
                ),
            )
            self._pact_stage1_modality = modality_name
            check_trainable_module(self)

    def forward(self, data_dict):
        if not self._is_stage1_local_evidence_enabled():
            return super().forward(data_dict)

        output_dict = {"pyramid": "collab"}
        agent_modality_list = data_dict["agent_modality_list"]
        affine_matrix = normalize_pairwise_tfm(
            data_dict["pairwise_t_matrix"],
            self.H,
            self.W,
            self.fake_voxel_size,
        )
        record_len = data_dict["record_len"]
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = getattr(self, "encoder_%s" % modality_name)(data_dict, modality_name)
            feature = getattr(self, "backbone_%s" % modality_name)({
                "spatial_features": feature,
            })["spatial_features_2d"]
            modality_feature_dict[modality_name] = getattr(
                self,
                "aligner_%s" % modality_name,
            )(feature)

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

        counting_dict = {name: 0 for name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feature_index = counting_dict[modality_name]
            heter_feature_2d_list.append(
                modality_feature_dict[modality_name][feature_index]
            )
            counting_dict[modality_name] += 1
        heter_feature_2d = torch.stack(heter_feature_2d_list)
        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        if self.supervise_single:
            single_feature, _ = self.pyramid_backbone.forward_single(heter_feature_2d)
            if self.shrink_flag:
                single_feature = self.shrink_conv(single_feature)
            output_dict.update({
                "cls_preds_single": self.cls_head(single_feature),
                "reg_preds_single": self.reg_head(single_feature),
                "dir_preds_single": self.dir_head(single_feature),
            })

        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
            heter_feature_2d,
            record_len,
            affine_matrix,
            agent_modality_list,
            self.cam_crop_info,
        )
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        output_dict.update({
            "cls_preds": self.cls_head(fused_feature),
            "reg_preds": self.reg_head(fused_feature),
            "dir_preds": self.dir_head(fused_feature),
            "occ_single_list": occ_outputs,
        })
        head = getattr(self, "pact_cbea_evidence_head_%s" % self._pact_stage1_modality)
        head_output = head(fused_feature)
        output_dict["pact_cbea"] = {
            "enabled": True,
            "stage": "local_evidence",
            "train_mode": self.pact_cbea_cfg["local_evidence"]["train_mode"],
            "modality": self._pact_stage1_modality,
            "global_rule_trainable": False,
            "use_stage3_joint_training": False,
            "evidence_heatmap_logits": head_output["evidence_heatmap_logits"],
            "evidence_heatmap": head_output["evidence_heatmap"],
            "evidence_uncertainty_logits": head_output["evidence_uncertainty_logits"],
            "evidence_uncertainty": head_output["evidence_uncertainty"],
            "evidence_loss_cfg": self.pact_cbea_cfg["evidence_loss"],
        }
        if self.pact_cbea_cfg["evidence_head"].get("return_descriptor", False):
            output_dict["pact_cbea"]["evidence_descriptor"] = head_output[
                "evidence_descriptor"
            ]
        if "evidence_feature" in head_output:
            output_dict["pact_cbea"]["evidence_feature"] = head_output["evidence_feature"]
        return output_dict

    def _is_stage1_local_evidence_enabled(self):
        return (
            self.pact_cbea_enabled
            and self.pact_cbea_cfg["stage"] == "stage1_local_evidence"
            and self.pact_cbea_cfg["local_evidence"]["enabled"]
            and self.pact_cbea_cfg["evidence_head"]["enabled"]
        )

    @staticmethod
    def _configured_modality(args):
        names = [name for name in args if name.startswith("m") and name[1:].isdigit()]
        if names != ["m1"]:
            raise ValueError("PACT-CBEA Stage1 expects exactly the m1 base modality")
        return "m1"

    @classmethod
    def _normalize_pact_stage1_cfg(cls, cfg):
        normalized = cls._default_pact_stage1_cfg()
        if isinstance(cfg, bool):
            normalized["enabled"] = cfg
        elif isinstance(cfg, dict):
            _deep_update(normalized, cfg)

        normalized["enabled"] = bool(normalized["enabled"])
        normalized["stage"] = str(normalized["stage"])
        normalized["trainable"] = bool(normalized["trainable"])
        normalized["no_joint_training"] = bool(normalized["no_joint_training"])
        normalized["use_stage3_joint_training"] = bool(
            normalized["use_stage3_joint_training"]
        )
        if not normalized["trainable"]:
            raise ValueError("PACT-CBEA Stage1 must train the HEAL base model")
        if not normalized["no_joint_training"]:
            raise ValueError("PACT-CBEA v1 Stage1 does not support joint Stage3 training")
        if normalized["use_stage3_joint_training"]:
            raise ValueError("PACT-CBEA v1 Stage1 requires use_stage3_joint_training=false")

        local = normalized["local_evidence"]
        local["enabled"] = bool(local.get("enabled", False) or local.get("enable", False))
        local["train_mode"] = str(local.get("train_mode", "stage1_base_train"))
        head = normalized["evidence_head"]
        head["enabled"] = bool(head.get("enabled", False) or head.get("enable", False))
        head["in_channels"] = int(head.get("in_channels", 256))
        head["hidden_dim"] = int(head.get("hidden_dim", 64))
        head["descriptor_dim"] = int(head.get("descriptor_dim", 16))
        head["use_sigmoid"] = bool(head.get("use_sigmoid", True))
        head["normalize_descriptor"] = bool(head.get("normalize_descriptor", True))
        head["return_feature"] = bool(head.get("return_feature", True))
        head["return_descriptor"] = bool(head.get("return_descriptor", False))
        loss = normalized["evidence_loss"]
        loss["enabled"] = bool(loss.get("enabled", False) or loss.get("enable", False))
        loss["mode"] = str(loss.get("mode", "pact_local_evidence"))
        for term, default_weight in (("evidence_heatmap", 0.01), ("uncertainty", 0.001), ("descriptor", 0.001)):
            item = loss[term]
            item["enabled"] = bool(item.get("enabled", False) or item.get("enable", False))
            item["weight"] = float(item.get("weight", default_weight))
        loss["evidence_heatmap"]["pos_weight"] = float(
            loss["evidence_heatmap"].get("pos_weight", 1.0)
        )
        return normalized

    @staticmethod
    def _default_pact_stage1_cfg():
        return {
            "enabled": False,
            "stage": "none",
            "trainable": True,
            "no_joint_training": True,
            "use_stage3_joint_training": False,
            "local_evidence": {
                "enabled": False,
                "train_mode": "stage1_base_train",
            },
            "evidence_head": {
                "enabled": False,
                "in_channels": 256,
                "hidden_dim": 64,
                "descriptor_dim": 16,
                "use_sigmoid": True,
                "normalize_descriptor": True,
                "return_feature": True,
                "return_descriptor": False,
            },
            "evidence_loss": {
                "enabled": False,
                "mode": "pact_local_evidence",
                "evidence_heatmap": {
                    "enabled": False,
                    "weight": 0.01,
                    "pos_weight": 1.0,
                },
                "uncertainty": {
                    "enabled": False,
                    "weight": 0.001,
                },
                "descriptor": {
                    "enabled": False,
                    "weight": 0.001,
                },
            },
        }


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

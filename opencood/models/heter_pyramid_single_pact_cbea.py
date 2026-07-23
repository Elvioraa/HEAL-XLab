"""PACT-CBEA v1 local-expert wrapper for HEAL single-modality training."""

import sys
import types

import torch
import torchvision

if "icecream" not in sys.modules:
    icecream_stub = types.ModuleType("icecream")
    icecream_stub.ic = lambda *args, **kwargs: args[0] if len(args) == 1 else args
    sys.modules["icecream"] = icecream_stub
if "timm.models.layers" not in sys.modules:
    timm_stub = types.ModuleType("timm")
    timm_models_stub = types.ModuleType("timm.models")
    timm_layers_stub = types.ModuleType("timm.models.layers")

    class _DropPath(torch.nn.Identity):
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
            bsz, lc, height, width = tensor.shape
            length = int(kwargs["l"])
            channels = lc // length
            return tensor.view(bsz, length, channels, height, width)
        raise ImportError("einops is required for rearrange pattern: %s" % pattern)

    def _repeat(tensor, pattern, **kwargs):
        if pattern == "b h w c l -> b (h new_h) (w new_w) c l":
            new_h = int(kwargs["new_h"])
            new_w = int(kwargs["new_w"])
            return tensor.repeat_interleave(new_h, dim=1).repeat_interleave(new_w, dim=2)
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

from opencood.models.heter_pyramid_single import HeterPyramidSingle
from opencood.models.sub_modules.pact_cbea_evidence_head import (
    PACTCBEALocalEvidenceHead,
)
from opencood.utils.model_utils import check_trainable_module


class HeterPyramidSinglePactCbea(HeterPyramidSingle):
    """Train one local expert with a PACT-named evidence head."""

    def __init__(self, args):
        super().__init__(args)
        self.pact_cbea_cfg = self._normalize_pact_local_cfg(args.get("pact_cbea"))
        self.pact_cbea_enabled = bool(self.pact_cbea_cfg.get("enabled", False))
        self.pact_cbea_stage = self.pact_cbea_cfg.get("stage", "none")
        if self._is_local_evidence_enabled():
            modality_name = self._configured_modality(args)
            head_cfg = self.pact_cbea_cfg["evidence_head"]
            setattr(
                self,
                f"pact_cbea_evidence_head_{modality_name}",
                PACTCBEALocalEvidenceHead(
                    in_channels=int(head_cfg.get("in_channels", args.get("in_head", 256))),
                    hidden_dim=int(head_cfg.get("hidden_dim", 64)),
                    descriptor_dim=int(head_cfg.get("descriptor_dim", 16)),
                    use_sigmoid=bool(head_cfg.get("use_sigmoid", True)),
                    normalize_descriptor=bool(head_cfg.get("normalize_descriptor", True)),
                    return_feature=bool(head_cfg.get("return_feature", True)),
                    predict_localization_uncertainty=bool(
                        head_cfg["localization_uncertainty"].get("enabled", False)
                    ),
                ),
            )
            self._pact_local_modality = modality_name
            self._apply_local_evidence_train_mode(modality_name)
            check_trainable_module(self)

    def forward(self, data_dict):
        if not self._is_local_evidence_enabled():
            return super().forward(data_dict)
        return self._forward_local_evidence(data_dict)

    def load_state_dict(self, state_dict, strict=True):
        """Load only compatible keys so each local expert can start from scratch or partial seeds."""
        own_state = super().state_dict()
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            target_key = key[7:] if key.startswith("module.") else key
            own_value = own_state.get(target_key)
            if own_value is not None and getattr(value, "shape", None) == own_value.shape:
                filtered[target_key] = value
            else:
                skipped.append(key)
        self._pact_cbea_last_skipped_state_keys = skipped[:80]
        return super().load_state_dict(filtered, strict=False)

    def _forward_local_evidence(self, data_dict):
        output_dict = {"pyramid": "single"}
        modality_name = self._select_single_modality(data_dict)

        feature = getattr(self, f"encoder_{modality_name}")(data_dict, modality_name)
        feature = getattr(self, f"backbone_{modality_name}")({
            "spatial_features": feature,
        })["spatial_features_2d"]
        feature = getattr(self, f"aligner_{modality_name}")(feature)

        if self.sensor_type_dict[modality_name] == "camera":
            _, _, height, width = feature.shape
            target_h = int(height * getattr(self, f"crop_ratio_H_{modality_name}"))
            target_w = int(width * getattr(self, f"crop_ratio_W_{modality_name}"))
            feature = torchvision.transforms.CenterCrop((target_h, target_w))(feature)

            if getattr(self, f"depth_supervision_{modality_name}"):
                output_dict.update({
                    f"depth_items_{modality_name}": getattr(
                        self,
                        f"encoder_{modality_name}",
                    ).depth_items
                })

        feature, occ_map_list = self.pyramid_backbone.forward_single(feature)

        if self.shrink_flag:
            feature = self.shrink_conv(feature)

        output_dict.update({
            "cls_preds": self.cls_head(feature),
            "reg_preds": self.reg_head(feature),
            "dir_preds": self.dir_head(feature),
            "occ_single_list": occ_map_list,
        })
        self._append_pact_local_evidence_output(output_dict, feature, modality_name)
        return output_dict

    @staticmethod
    def _select_single_modality(data_dict):
        modality_keys = [key for key in data_dict.keys() if key.startswith("inputs_")]
        assert len(modality_keys) == 1
        return modality_keys[0][len("inputs_"):]

    def _append_pact_local_evidence_output(self, output_dict, bev_feature, modality_name):
        head = getattr(self, f"pact_cbea_evidence_head_{modality_name}")
        head_output = head(bev_feature)
        output_dict["pact_cbea"] = {
            "enabled": True,
            "stage": "local_evidence",
            "train_mode": self.pact_cbea_cfg["local_evidence"].get(
                "train_mode",
                "local_evidence",
            ),
            "modality": modality_name,
            "global_rule_trainable": False,
            "use_stage3_joint_training": False,
            "evidence_heatmap_logits": head_output["evidence_heatmap_logits"],
            "evidence_heatmap": head_output["evidence_heatmap"],
            "evidence_uncertainty_logits": head_output["evidence_uncertainty_logits"],
            "evidence_uncertainty": head_output["evidence_uncertainty"],
            "evidence_loss_cfg": self.pact_cbea_cfg["evidence_loss"],
        }
        if bool(self.pact_cbea_cfg["evidence_head"].get("return_descriptor", False)):
            output_dict["pact_cbea"]["evidence_descriptor"] = head_output["evidence_descriptor"]
        if "evidence_feature" in head_output:
            output_dict["pact_cbea"]["evidence_feature"] = head_output["evidence_feature"]
        if "evidence_localization_uncertainty" in head_output:
            output_dict["pact_cbea"]["evidence_localization_uncertainty_logits"] = (
                head_output["evidence_localization_uncertainty_logits"]
            )
            output_dict["pact_cbea"]["evidence_localization_uncertainty"] = head_output[
                "evidence_localization_uncertainty"
            ]
        return output_dict

    def _is_local_evidence_enabled(self):
        return (
            bool(self.pact_cbea_enabled)
            and self.pact_cbea_stage == "local_evidence"
            and bool(self.pact_cbea_cfg["local_evidence"].get("enabled", False))
            and bool(self.pact_cbea_cfg["evidence_head"].get("enabled", False))
        )

    def _apply_local_evidence_train_mode(self, modality_name):
        for param in getattr(self, f"pact_cbea_evidence_head_{modality_name}").parameters():
            param.requires_grad_(True)

    @staticmethod
    def _configured_modality(args):
        names = [name for name in args.keys() if name.startswith("m") and name[1:].isdigit()]
        if len(names) != 1:
            raise ValueError("PACT local evidence expert expects exactly one modality")
        return names[0]

    @classmethod
    def _normalize_pact_local_cfg(cls, cfg):
        normalized = cls._default_pact_local_cfg()
        if isinstance(cfg, bool):
            normalized["enabled"] = cfg
        elif isinstance(cfg, dict):
            _deep_update(normalized, cfg)
        normalized["enabled"] = bool(normalized.get("enabled", False))
        normalized["stage"] = str(normalized.get("stage", "none"))
        normalized["trainable"] = bool(normalized.get("trainable", True))
        normalized["no_joint_training"] = bool(normalized.get("no_joint_training", True))
        normalized["use_stage3_joint_training"] = bool(
            normalized.get("use_stage3_joint_training", False)
        )

        local = normalized["local_evidence"]
        local["enabled"] = bool(local.get("enabled", False) or local.get("enable", False))
        local["train_mode"] = str(local.get("train_mode", "local_evidence"))

        head = normalized["evidence_head"]
        head["enabled"] = bool(head.get("enabled", False) or head.get("enable", False))
        head["in_channels"] = int(head.get("in_channels", 256))
        head["hidden_dim"] = int(head.get("hidden_dim", 64))
        head["descriptor_dim"] = int(head.get("descriptor_dim", 16))
        head["use_sigmoid"] = bool(head.get("use_sigmoid", True))
        head["normalize_descriptor"] = bool(head.get("normalize_descriptor", True))
        head["return_feature"] = bool(head.get("return_feature", True))
        head["return_descriptor"] = bool(head.get("return_descriptor", False))
        loc_unc_head = head["localization_uncertainty"]
        loc_unc_head["enabled"] = bool(
            loc_unc_head.get("enabled", False) or loc_unc_head.get("enable", False)
        )

        loss = normalized["evidence_loss"]
        loss["enabled"] = bool(loss.get("enabled", False) or loss.get("enable", False))
        loss["mode"] = str(loss.get("mode", "pact_local_evidence"))
        hmap = loss["evidence_heatmap"]
        hmap["enabled"] = bool(hmap.get("enabled", False) or hmap.get("enable", False))
        hmap["weight"] = float(hmap.get("weight", 0.01))
        hmap["pos_weight"] = float(hmap.get("pos_weight", 1.0))
        unc = loss["uncertainty"]
        unc["enabled"] = bool(unc.get("enabled", False) or unc.get("enable", False))
        unc["weight"] = float(unc.get("weight", 0.001))
        desc = loss["descriptor"]
        desc["enabled"] = bool(desc.get("enabled", False) or desc.get("enable", False))
        desc["weight"] = float(desc.get("weight", 0.001))
        loc_unc = loss["localization_uncertainty"]
        loc_unc["enabled"] = bool(
            loc_unc.get("enabled", False) or loc_unc.get("enable", False)
        )
        loc_unc["weight"] = float(loc_unc.get("weight", 0.001))
        loc_unc["max_residual"] = float(loc_unc.get("max_residual", 10.0))
        if loc_unc["max_residual"] <= 0.0:
            raise ValueError(
                "evidence_loss.localization_uncertainty.max_residual must be positive"
            )
        return normalized

    @staticmethod
    def _default_pact_local_cfg():
        return {
            "enabled": False,
            "stage": "none",
            "trainable": True,
            "no_joint_training": True,
            "use_stage3_joint_training": False,
            "local_evidence": {
                "enabled": False,
                "train_mode": "local_evidence",
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
                "localization_uncertainty": {
                    "enabled": False,
                },
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
                "localization_uncertainty": {
                    "enabled": False,
                    "weight": 0.001,
                    "max_residual": 10.0,
                },
            },
        }


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

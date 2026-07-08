"""PACT-CBEA HEAL-compatible plug-and-play collaboration wrapper."""

from collections import Counter
import sys
import types

import torch
import torch.nn as nn

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

from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.sub_modules.pact_cbea_rule import PACTCBEARule
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeterPyramidCollabPactCbea(HeterPyramidCollab):
    """HEAL collab model with optional parameter-free PACT-CBEA rule routing."""

    BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def __init__(self, args):
        super().__init__(args)
        self.supervise_single = bool(args.get("supervise_single", False))
        self.pact_cbea_cfg = self._normalize_pact_cfg(args.get("pact_cbea"))
        self.pact_cbea_enabled = bool(self.pact_cbea_cfg.get("enabled", False))
        self.pact_cbea_trainable = bool(self.pact_cbea_cfg.get("trainable", False))
        self.pact_no_joint_training = bool(self.pact_cbea_cfg.get("no_joint_training", True))
        self.pact_use_stage3_joint_training = bool(
            self.pact_cbea_cfg.get("use_stage3_joint_training", False)
        )
        if self.pact_cbea_enabled:
            self.pact_cbea_rule = PACTCBEARule(self.pact_cbea_cfg)
            for param in self.pact_cbea_rule.parameters():
                param.requires_grad_(False)
            if self.pact_no_joint_training and not self.pact_cbea_trainable:
                self._freeze_all_model_parameters()

    def forward(self, data_dict):
        if not self.pact_cbea_enabled:
            return super().forward(data_dict)

        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list']
        affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'],
            self.H,
            self.W,
            self.fake_voxel_size,
        )
        record_len = data_dict['record_len']
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = getattr(self, f"encoder_{modality_name}")(data_dict, modality_name)
            feature = getattr(self, f"backbone_{modality_name}")({
                "spatial_features": feature,
            })['spatial_features_2d']
            feature = getattr(self, f"aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict and self.sensor_type_dict[modality_name] == "camera":
                import torchvision

                feature = modality_feature_dict[modality_name]
                _, _, height, width = feature.shape
                target_h = int(height * getattr(self, f"crop_ratio_H_{modality_name}"))
                target_w = int(width * getattr(self, f"crop_ratio_W_{modality_name}"))
                crop_func = torchvision.transforms.CenterCrop((target_h, target_w))
                modality_feature_dict[modality_name] = crop_func(feature)
                if getattr(self, f"depth_supervision_{modality_name}"):
                    output_dict.update({
                        f"depth_items_{modality_name}": getattr(
                            self,
                            f"encoder_{modality_name}",
                        ).depth_items
                    })

        counting_dict = {modality_name: 0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)
        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        single_feature, occ_outputs = self.pyramid_backbone.forward_single(heter_feature_2d)
        if self.shrink_flag:
            single_feature = self.shrink_conv(single_feature)

        if self.supervise_single:
            output_dict.update({
                "cls_preds_single": self.cls_head(single_feature),
                "reg_preds_single": self.reg_head(single_feature),
                "dir_preds_single": self.dir_head(single_feature),
            })

        pact_feature, pact_debug = self.pact_cbea_rule(
            single_feature,
            evidence_heatmap=self._lookup_optional_evidence(data_dict, "evidence_heatmap"),
            evidence_uncertainty=self._lookup_optional_evidence(data_dict, "evidence_uncertainty"),
            record_len=record_len,
            pairwise_t_matrix=data_dict.get("pairwise_t_matrix"),
            modality_names=agent_modality_list,
        )
        pact_debug.update({
            "pact_cbea_enabled": True,
            "pact_trainable": bool(self.pact_cbea_trainable),
            "pact_no_joint_training": bool(self.pact_no_joint_training),
            "pact_use_stage3_joint_training": bool(self.pact_use_stage3_joint_training),
            "heal_compatible": bool(self.pact_cbea_cfg.get("heal_compatible", True)),
            "plug_and_play": bool(self.pact_cbea_cfg.get("plug_and_play", True)),
        })

        cls_preds = self.cls_head(pact_feature)
        reg_preds = self.reg_head(pact_feature)
        dir_preds = self.dir_head(pact_feature)

        output_dict.update({
            'cls_preds': cls_preds,
            'reg_preds': reg_preds,
            'dir_preds': dir_preds,
            'occ_single_list': occ_outputs,
            'pact_cbea': pact_debug,
        })
        return output_dict

    @staticmethod
    def _lookup_optional_evidence(data_dict, key):
        if key in data_dict:
            return data_dict[key]
        pact_payload = data_dict.get("pact_cbea")
        if isinstance(pact_payload, dict) and key in pact_payload:
            return pact_payload[key]
        return None

    def _freeze_all_model_parameters(self):
        for param in self.parameters():
            param.requires_grad_(False)
        self.apply(self._set_bn_eval)

    @classmethod
    def _normalize_pact_cfg(cls, cfg):
        normalized = PACTCBEARule._default_cfg()
        if isinstance(cfg, bool):
            normalized["enabled"] = bool(cfg)
        elif isinstance(cfg, dict):
            _deep_update(normalized, cfg)
        normalized = PACTCBEARule._normalize_cfg(normalized)
        if normalized["trainable"]:
            raise ValueError("PACT-CBEA v1 is parameter-free; pact_cbea.trainable must be false")
        if not normalized["no_joint_training"]:
            raise ValueError("PACT-CBEA requires no_joint_training=true")
        if normalized["use_stage3_joint_training"]:
            raise ValueError("PACT-CBEA requires use_stage3_joint_training=false")
        return normalized

    @staticmethod
    def _set_bn_eval(module):
        if isinstance(module, HeterPyramidCollabPactCbea.BN_TYPES):
            module.eval()


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

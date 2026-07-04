"""HVP-HEAL v3 feature-main wrapper for HEAL collaborative pyramid.

The wrapper is inert unless ``model.args.hvp_v3.enabled`` is explicitly true.
"""

from collections import Counter
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

from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.hvp_heal_v3.hypothesis_head import HvpHealV3HypothesisHead
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeterPyramidCollabHvpHealV3(HeterPyramidCollab):
    """HEAL pyramid collab with optional HVP-HEAL v3 Stage1 hypothesis head."""

    def __init__(self, args):
        super().__init__(args)
        self.hvp_v3_cfg = self._normalize_hvp_v3_cfg(args.get("hvp_v3"))
        self.hvp_v3_enabled = bool(self.hvp_v3_cfg.get("enabled", False))
        self.hvp_v3_stage = self.hvp_v3_cfg.get("stage", "none")
        if self._is_stage1_enabled():
            head_cfg = self.hvp_v3_cfg["hypothesis_head"]
            self.hvp_v3_hypothesis_head = HvpHealV3HypothesisHead(
                in_channels=int(head_cfg.get("in_channels", args.get("in_head", 256))),
                hidden_dim=int(head_cfg.get("hidden_dim", 64)),
                out_channels=int(head_cfg.get("out_channels", 1)),
                use_sigmoid=bool(head_cfg.get("use_sigmoid", True)),
                return_feature=bool(head_cfg.get("return_feature", True)),
            )

    def forward(self, data_dict):
        if not self._is_stage1_enabled():
            return super().forward(data_dict)
        return self._forward_stage1_hypothesis(data_dict)

    def _forward_stage1_hypothesis(self, data_dict):
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
            feature = getattr(self, f"encoder_{modality_name}")(data_dict, modality_name)
            feature = getattr(self, f"backbone_{modality_name}")({"spatial_features": feature})[
                "spatial_features_2d"
            ]
            feature = getattr(self, f"aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            if self.sensor_type_dict[modality_name] != "camera":
                continue
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
        hvp_v3_stage1_feature = self._select_ego_feature(heter_feature_2d, record_len)

        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
            heter_feature_2d,
            record_len,
            affine_matrix,
            agent_modality_list,
            self.cam_crop_info,
        )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        self._append_hvp_v3_stage1_output(output_dict, hvp_v3_stage1_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({
            "cls_preds": cls_preds,
            "reg_preds": reg_preds,
            "dir_preds": dir_preds,
            "occ_single_list": occ_outputs,
        })
        return output_dict

    @staticmethod
    def _select_ego_feature(heter_feature_2d, record_len):
        rows = []
        start = 0
        lengths = record_len.detach().cpu().tolist() if torch.is_tensor(record_len) else list(record_len)
        for length in lengths:
            rows.append(heter_feature_2d[start])
            start += int(length)
        return torch.stack(rows, dim=0)

    def _append_hvp_v3_stage1_output(self, output_dict, bev_feature):
        if not self._is_stage1_enabled():
            return output_dict
        head_output = self.hvp_v3_hypothesis_head(bev_feature)
        output_dict["hvp_v3"] = {
            "enabled": True,
            "stage": "stage1_hypothesis",
            "feature_main": bool(self.hvp_v3_cfg["feature_main"].get("enabled", False)),
            "hypothesis_heatmap_logits": head_output["hypothesis_heatmap_logits"],
            "hypothesis_heatmap": head_output["hypothesis_heatmap"],
            "aux_loss_cfg": self.hvp_v3_cfg["aux_loss"],
        }
        if "hypothesis_feature" in head_output:
            output_dict["hvp_v3"]["hypothesis_feature"] = head_output["hypothesis_feature"]
        return output_dict

    def _is_stage1_enabled(self):
        return (
            bool(self.hvp_v3_enabled)
            and self.hvp_v3_stage == "stage1_hypothesis"
            and bool(self.hvp_v3_cfg["feature_main"].get("enabled", False))
            and bool(self.hvp_v3_cfg["hypothesis_head"].get("enabled", False))
        )

    @classmethod
    def _normalize_hvp_v3_cfg(cls, cfg):
        normalized = cls._default_hvp_v3_cfg()
        if isinstance(cfg, bool):
            normalized["enabled"] = cfg
        elif isinstance(cfg, dict):
            _deep_update(normalized, cfg)
        normalized["enabled"] = bool(normalized.get("enabled", False))
        normalized["stage"] = str(normalized.get("stage", "none"))
        normalized["feature_main"]["enabled"] = bool(
            normalized["feature_main"].get("enabled", False)
        )
        head = normalized["hypothesis_head"]
        head["enabled"] = bool(head.get("enabled", False))
        head["in_channels"] = int(head.get("in_channels", 64))
        head["hidden_dim"] = int(head.get("hidden_dim", 64))
        head["out_channels"] = int(head.get("out_channels", 1))
        head["use_sigmoid"] = bool(head.get("use_sigmoid", True))
        head["return_feature"] = bool(head.get("return_feature", True))

        aux = normalized["aux_loss"]
        aux["enabled"] = bool(aux.get("enabled", False))
        aux["mode"] = str(aux.get("mode", "stage1_hypothesis"))
        heatmap = aux["hypothesis_heatmap"]
        heatmap["enabled"] = bool(heatmap.get("enabled", False))
        heatmap["weight"] = float(heatmap.get("weight", 0.01))
        heatmap["pos_weight"] = float(heatmap.get("pos_weight", 1.0))
        for key in ("residual_reg", "alpha_reg", "residual_focus"):
            aux[key]["enabled"] = bool(aux[key].get("enabled", False))
        return normalized

    @staticmethod
    def _default_hvp_v3_cfg():
        return {
            "enabled": False,
            "stage": "none",
            "feature_main": {
                "enabled": False,
            },
            "hypothesis_head": {
                "enabled": False,
                "in_channels": 64,
                "hidden_dim": 64,
                "out_channels": 1,
                "use_sigmoid": True,
                "return_feature": True,
            },
            "aux_loss": {
                "enabled": False,
                "mode": "stage1_hypothesis",
                "hypothesis_heatmap": {
                    "enabled": False,
                    "weight": 0.01,
                    "pos_weight": 1.0,
                },
                "residual_reg": {
                    "enabled": False,
                },
                "alpha_reg": {
                    "enabled": False,
                },
                "residual_focus": {
                    "enabled": False,
                },
            },
        }


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

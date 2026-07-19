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
from opencood.models.sub_modules.pact_cbea_evidence_head import (
    PACTCBEALocalEvidenceHead,
)
from opencood.models.sub_modules.pact_cbea_rule import PACTCBEARule
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
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
        self.pact_fusion_mode = self.pact_cbea_cfg["fusion_mode"]
        self.pact_multiscale_prior_cfg = self.pact_cbea_cfg["multiscale_prior"]
        if self.pact_cbea_enabled:
            self.pact_cbea_rule = PACTCBEARule(self.pact_cbea_cfg)
            if self._pact_local_evidence_enabled():
                self._build_local_evidence_heads(args)
            for param in self.pact_cbea_rule.parameters():
                param.requires_grad_(False)
            if self.pact_no_joint_training and not self.pact_cbea_trainable:
                self._freeze_all_model_parameters()

    def forward(self, data_dict):
        if not self.pact_cbea_enabled:
            return super().forward(data_dict)
        pact_fusion_mode = getattr(self, "pact_fusion_mode", "legacy_rule")
        pact_multiscale_prior_cfg = getattr(
            self,
            "pact_multiscale_prior_cfg",
            {"enabled": False, "lambda": 0.0},
        )
        if (
            pact_fusion_mode == "heal_multiscale_prior"
            and pact_multiscale_prior_cfg["enabled"]
            and self.training
        ):
            raise RuntimeError("heal_multiscale_prior is an inference-only mode")

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

        single_feature, single_occ_outputs = self.pyramid_backbone.forward_single(heter_feature_2d)
        if self.shrink_flag:
            single_feature = self.shrink_conv(single_feature)

        if (
            pact_fusion_mode == "heal_multiscale_prior"
            and pact_multiscale_prior_cfg["enabled"]
        ):
            return self._forward_heal_multiscale_prior(
                output_dict,
                heter_feature_2d,
                single_feature,
                record_len,
                affine_matrix,
                data_dict.get("pairwise_t_matrix"),
                agent_modality_list,
            )

        base_fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
            heter_feature_2d,
            record_len,
            affine_matrix,
            agent_modality_list,
            self.cam_crop_info,
        )
        if self.shrink_flag:
            base_fused_feature = self.shrink_conv(base_fused_feature)

        if self.supervise_single:
            output_dict.update({
                "cls_preds_single": self.cls_head(single_feature),
                "reg_preds_single": self.reg_head(single_feature),
                "dir_preds_single": self.dir_head(single_feature),
            })

        local_evidence = self._compute_local_evidence(single_feature, agent_modality_list)
        if local_evidence is None:
            pact_feature = base_fused_feature
            pact_debug = {
                "pact_mode": self.pact_cbea_cfg.get("mode", "trust_calibrated_rule"),
                "pact_fallbacks": ["missing_local_evidence_heads_base_heal_fallback"],
                "pact_features_warped_to_ego": False,
                "pact_evidence_warped_to_ego": False,
                "pact_used_base_heal_fallback": True,
                "pact_agent_count": int(single_feature.shape[0]),
                "pact_trainable": False,
                "pact_no_joint_training": True,
            }
        else:
            warped_feature = self._warp_to_ego(single_feature, record_len, affine_matrix)
            warped_logits = self._warp_to_ego(
                local_evidence["evidence_heatmap_logits"],
                record_len,
                affine_matrix,
            )
            warped_uncertainty = self._warp_to_ego(
                local_evidence["evidence_uncertainty"],
                record_len,
                affine_matrix,
            )
            pact_feature, pact_debug = self.pact_cbea_rule(
                warped_feature,
                evidence_heatmap=warped_logits,
                evidence_uncertainty=warped_uncertainty,
                record_len=record_len,
                pairwise_t_matrix=data_dict.get("pairwise_t_matrix"),
                modality_names=agent_modality_list,
            )
            pact_debug.update({
                "pact_features_warped_to_ego": True,
                "pact_evidence_warped_to_ego": True,
                "pact_uncertainty_warped_to_ego": True,
                "pact_used_base_heal_fallback": False,
                "pact_evidence_heatmap_logits": warped_logits,
                "pact_evidence_uncertainty": warped_uncertainty,
            })
        pact_debug.update({
            "pact_cbea_enabled": True,
            "pact_trainable": bool(self.pact_cbea_trainable),
            "pact_no_joint_training": bool(self.pact_no_joint_training),
            "pact_use_stage3_joint_training": bool(self.pact_use_stage3_joint_training),
            "heal_compatible": bool(self.pact_cbea_cfg.get("heal_compatible", True)),
            "plug_and_play": bool(self.pact_cbea_cfg.get("plug_and_play", True)),
            "pact_local_evidence_enabled": bool(self._pact_local_evidence_enabled()),
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

    def _forward_heal_multiscale_prior(
            self, output_dict, heter_feature_2d, single_feature,
            record_len, affine_matrix, pairwise_t_matrix,
            agent_modality_list):
        if self.training:
            raise RuntimeError("heal_multiscale_prior is an inference-only mode")

        cbea_lambda = self.pact_multiscale_prior_cfg["lambda"]
        cbea_alpha = None
        multiscale_used = False
        fallbacks = []

        if cbea_lambda == 0.0:
            fallbacks.append("lambda_zero_strict_heal_baseline")
        else:
            local_evidence = self._compute_local_evidence(
                single_feature,
                agent_modality_list,
            )
            if local_evidence is None:
                fallbacks.append("missing_local_evidence_heal_fallback")
            else:
                warped_feature = self._warp_to_ego(
                    single_feature,
                    record_len,
                    affine_matrix,
                )
                warped_logits = self._warp_to_ego(
                    local_evidence["evidence_heatmap_logits"],
                    record_len,
                    affine_matrix,
                )
                warped_uncertainty = self._warp_to_ego(
                    local_evidence["evidence_uncertainty"],
                    record_len,
                    affine_matrix,
                )
                _, rule_debug = self.pact_cbea_rule(
                    warped_feature,
                    evidence_heatmap=warped_logits,
                    evidence_uncertainty=warped_uncertainty,
                    record_len=record_len,
                    pairwise_t_matrix=pairwise_t_matrix,
                    modality_names=agent_modality_list,
                )
                cbea_alpha = self._flatten_pact_alpha(rule_debug, record_len)
                if cbea_alpha is None:
                    fallbacks.append("missing_pact_alpha_heal_fallback")
                else:
                    multiscale_used = True
                    fallbacks.extend(rule_debug.get("pact_fallbacks", []))

        if multiscale_used:
            fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                heter_feature_2d,
                record_len,
                affine_matrix,
                agent_modality_list,
                self.cam_crop_info,
                cbea_alpha=cbea_alpha,
                cbea_lambda=cbea_lambda,
            )
        else:
            fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                heter_feature_2d,
                record_len,
                affine_matrix,
                agent_modality_list,
                self.cam_crop_info,
            )
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        if self.supervise_single:
            output_dict.update({
                "cls_preds_single": self.cls_head(single_feature),
                "reg_preds_single": self.reg_head(single_feature),
                "dir_preds_single": self.dir_head(single_feature),
            })

        pact_debug = {
            "pact_fusion_mode": "heal_multiscale_prior",
            "pact_multiscale_prior_enabled": True,
            "pact_multiscale_used": bool(multiscale_used),
            "pact_multiscale_lambda": float(cbea_lambda),
            "pact_multiscale_fallbacks": fallbacks,
            "pact_multiscale_alpha_shape": (
                list(cbea_alpha.shape) if cbea_alpha is not None else None
            ),
            "pact_multiscale_alpha_min": (
                float(cbea_alpha.min().item()) if cbea_alpha is not None else None
            ),
            "pact_multiscale_alpha_max": (
                float(cbea_alpha.max().item()) if cbea_alpha is not None else None
            ),
            "pact_multiscale_alpha_mean": (
                float(cbea_alpha.mean().item()) if cbea_alpha is not None else None
            ),
            "pact_cbea_enabled": True,
            "pact_trainable": bool(self.pact_cbea_trainable),
            "pact_no_joint_training": bool(self.pact_no_joint_training),
            "pact_use_stage3_joint_training": bool(
                self.pact_use_stage3_joint_training
            ),
            "pact_local_evidence_enabled": bool(
                self._pact_local_evidence_enabled()
            ),
        }
        if self.pact_cbea_cfg.get("debug", False) and cbea_alpha is not None:
            pact_debug["pact_multiscale_alpha"] = cbea_alpha.detach()

        output_dict.update({
            "cls_preds": self.cls_head(fused_feature),
            "reg_preds": self.reg_head(fused_feature),
            "dir_preds": self.dir_head(fused_feature),
            "occ_single_list": occ_outputs,
            "pact_cbea": pact_debug,
        })
        return output_dict

    @staticmethod
    def _flatten_pact_alpha(pact_debug, record_len):
        if torch.is_tensor(record_len):
            records = [
                int(value) for value in record_len.detach().cpu().view(-1).tolist()
            ]
        else:
            records = [int(value) for value in record_len]
        if not records or any(value <= 0 for value in records):
            raise ValueError("record_len must contain positive scene sizes")

        group_debug = pact_debug.get("pact_group_debug")
        if group_debug is not None:
            if not isinstance(group_debug, (list, tuple)):
                raise ValueError("pact_group_debug must be a list or tuple")
            if len(group_debug) != len(records):
                raise ValueError("pact_group_debug scene count does not match record_len")
            flattened = []
            for scene_index, (item, cav_num) in enumerate(zip(group_debug, records)):
                if not isinstance(item, dict) or "pact_alpha" not in item:
                    return None
                alpha = item["pact_alpha"]
                if not torch.is_tensor(alpha):
                    raise ValueError("pact_alpha must be a tensor")
                if alpha.ndim == 5 and alpha.shape[0] == 1:
                    alpha = alpha[0]
                if alpha.ndim != 4 or int(alpha.shape[0]) != cav_num:
                    raise ValueError(
                        "pact_alpha agent layout mismatch in scene %d" % scene_index
                    )
                flattened.append(alpha)
            alpha = torch.cat(flattened, dim=0)
        else:
            alpha = pact_debug.get("pact_alpha")
            if alpha is None:
                return None
            if not torch.is_tensor(alpha):
                raise ValueError("pact_alpha must be a tensor")
            if alpha.ndim == 5:
                if len(records) == 1 and alpha.shape[0] == 1:
                    alpha = alpha[0]
                elif (
                    alpha.shape[0] == len(records)
                    and len(set(records)) == 1
                    and alpha.shape[1] == records[0]
                ):
                    alpha = alpha.reshape(
                        sum(records),
                        alpha.shape[2],
                        alpha.shape[3],
                        alpha.shape[4],
                    )
                else:
                    raise ValueError("pact_alpha batch layout does not match record_len")
            if alpha.ndim != 4:
                raise ValueError("pact_alpha must flatten to [N, 1, H, W]")

        if int(alpha.shape[0]) != sum(records):
            raise ValueError("flattened pact_alpha agent count does not match record_len")
        if int(alpha.shape[1]) != 1:
            raise ValueError("flattened pact_alpha channel dimension must be 1")
        if not torch.isfinite(alpha).all().item():
            raise ValueError("flattened pact_alpha contains NaN or Inf")
        return alpha

    def _build_local_evidence_heads(self, args):
        head_cfg = self.pact_cbea_cfg["evidence_head"]
        for modality_name in self.modality_name_list:
            setattr(
                self,
                f"pact_cbea_evidence_head_{modality_name}",
                PACTCBEALocalEvidenceHead(
                    in_channels=int(head_cfg.get("in_channels", args.get("in_head", 256))),
                    hidden_dim=int(head_cfg.get("hidden_dim", 64)),
                    descriptor_dim=int(head_cfg.get("descriptor_dim", 16)),
                    use_sigmoid=bool(head_cfg.get("use_sigmoid", True)),
                    normalize_descriptor=bool(head_cfg.get("normalize_descriptor", True)),
                    return_feature=bool(head_cfg.get("return_feature", False)),
                ),
            )

    def _compute_local_evidence(self, feature, agent_modality_list):
        if not self._pact_local_evidence_enabled():
            return None
        logits = []
        heatmaps = []
        uncertainties = []
        for idx, modality_name in enumerate(agent_modality_list):
            head_name = f"pact_cbea_evidence_head_{modality_name}"
            if not hasattr(self, head_name):
                return None
            head_output = getattr(self, head_name)(feature[idx:idx + 1])
            logits.append(head_output["evidence_heatmap_logits"])
            heatmaps.append(head_output["evidence_heatmap"])
            uncertainties.append(head_output["evidence_uncertainty"])
        return {
            "evidence_heatmap_logits": torch.cat(logits, dim=0),
            "evidence_heatmap": torch.cat(heatmaps, dim=0),
            "evidence_uncertainty": torch.cat(uncertainties, dim=0),
        }

    def _warp_to_ego(self, tensor, record_len, affine_matrix):
        _, _, height, width = tensor.shape
        split_tensor = torch.tensor_split(
            tensor,
            torch.cumsum(record_len, dim=0)[:-1].cpu(),
        )
        warped = []
        align_corners = bool(getattr(self.pyramid_backbone, "align_corners", False))
        for batch_idx, group_tensor in enumerate(split_tensor):
            cav_num = int(record_len[batch_idx])
            ego_to_agent = affine_matrix[batch_idx, 0, :cav_num, :, :]
            warped.append(
                warp_affine_simple(
                    group_tensor,
                    ego_to_agent,
                    (height, width),
                    align_corners=align_corners,
                )
            )
        return torch.cat(warped, dim=0)

    @staticmethod
    def _lookup_optional_evidence(data_dict, key):
        if key in data_dict:
            return data_dict[key]
        pact_payload = data_dict.get("pact_cbea")
        if isinstance(pact_payload, dict) and key in pact_payload:
            return pact_payload[key]
        return None

    def _pact_local_evidence_enabled(self):
        return (
            bool(self.pact_cbea_cfg.get("local_evidence", {}).get("enabled", False))
            and bool(self.pact_cbea_cfg.get("evidence_head", {}).get("enabled", False))
        )

    def _freeze_all_model_parameters(self):
        for param in self.parameters():
            param.requires_grad_(False)
        self.apply(self._set_bn_eval)

    @classmethod
    def _normalize_pact_cfg(cls, cfg):
        normalized = PACTCBEARule._default_cfg()
        normalized.update({
            "fusion_mode": "legacy_rule",
            "multiscale_prior": {
                "enabled": False,
                "lambda": 0.0,
            },
            "local_evidence": {
                "enabled": False,
            },
            "evidence_head": {
                "enabled": False,
                "in_channels": 256,
                "hidden_dim": 64,
                "descriptor_dim": 16,
                "use_sigmoid": True,
                "normalize_descriptor": True,
                "return_feature": False,
            },
        })
        if isinstance(cfg, bool):
            normalized["enabled"] = bool(cfg)
        elif isinstance(cfg, dict):
            _deep_update(normalized, cfg)
        normalized = PACTCBEARule._normalize_cfg(normalized)
        normalized["local_evidence"]["enabled"] = bool(
            normalized["local_evidence"].get("enabled", False)
            or normalized["local_evidence"].get("enable", False)
        )
        head = normalized["evidence_head"]
        head["enabled"] = bool(head.get("enabled", False) or head.get("enable", False))
        head["in_channels"] = int(head.get("in_channels", 256))
        head["hidden_dim"] = int(head.get("hidden_dim", 64))
        head["descriptor_dim"] = int(head.get("descriptor_dim", 16))
        head["use_sigmoid"] = bool(head.get("use_sigmoid", True))
        head["normalize_descriptor"] = bool(head.get("normalize_descriptor", True))
        head["return_feature"] = bool(head.get("return_feature", False))
        fusion_mode = str(normalized.get("fusion_mode", "legacy_rule"))
        if fusion_mode not in ("legacy_rule", "heal_multiscale_prior"):
            raise ValueError(
                "pact_cbea.fusion_mode must be legacy_rule or heal_multiscale_prior"
            )
        normalized["fusion_mode"] = fusion_mode
        multiscale_prior = normalized.get("multiscale_prior")
        if not isinstance(multiscale_prior, dict):
            raise ValueError("pact_cbea.multiscale_prior must be a mapping")
        multiscale_prior["enabled"] = bool(
            multiscale_prior.get("enabled", False)
        )
        try:
            multiscale_prior["lambda"] = float(
                multiscale_prior.get("lambda", 0.0)
            )
        except (TypeError, ValueError):
            raise ValueError("pact_cbea.multiscale_prior.lambda must be a float")
        if not 0.0 <= multiscale_prior["lambda"] <= 1.0:
            raise ValueError(
                "pact_cbea.multiscale_prior.lambda must be in [0.0, 1.0]"
            )
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

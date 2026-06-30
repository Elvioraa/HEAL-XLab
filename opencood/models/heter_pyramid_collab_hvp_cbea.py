"""HEAL-XLab-v2 HVP-CBEA model wrapper.

This model keeps the official HEAL heter_pyramid_collab path intact unless
model.args.hvp_cbea.enabled is explicitly true.
"""

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
from opencood.models.sub_modules.hypothesis_encoder import HypothesisEncoder
from opencood.models.sub_modules.hypothesis_verifier import HypothesisVerifier
from opencood.models.sub_modules.bayesian_hypothesis_fusion import BayesianHypothesisFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeterPyramidCollabHvpCbea(HeterPyramidCollab):
    """HeterPyramidCollab with optional HVP-CBEA before detection heads."""

    HVP_MODULE_PREFIXES = (
        "hvp_collaborator_proj",
        "hypothesis_encoder",
        "hypothesis_verifier",
        "bayesian_hypothesis_fusion",
    )
    BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def __init__(self, args):
        super().__init__(args)
        self.supervise_single = bool(args.get("supervise_single", False))
        hvp = args.get("hvp_cbea", {}) or {}
        self.hvp_cbea_cfg = self._default_hvp_cfg(args)
        self.hvp_cbea_cfg.update(hvp)
        self.hvp_cbea_cfg["residual_gate"] = self._normalize_residual_gate_cfg(
            self.hvp_cbea_cfg.get("residual_gate")
        )
        self.hvp_cbea_enabled = bool(self.hvp_cbea_cfg.get("enabled", False))
        self.hvp_train_only = bool(self.hvp_cbea_cfg.get("train_only_hvp", False))
        self.hvp_trainable_summary = None
        if self.hvp_cbea_enabled:
            in_channels = int(self.hvp_cbea_cfg.get("in_channels", args.get("in_head", 256)))
            collaborator_in_channels = int(self.hvp_cbea_cfg.get(
                "collaborator_in_channels",
                self._infer_collaborator_channels(args, in_channels),
            ))
            mid_channels = int(self.hvp_cbea_cfg.get("mid_channels", 64))
            if collaborator_in_channels == in_channels:
                self.hvp_collaborator_proj = nn.Identity()
            else:
                self.hvp_collaborator_proj = nn.Conv2d(collaborator_in_channels, in_channels, kernel_size=1)
            self.hypothesis_encoder = HypothesisEncoder(
                in_channels=in_channels,
                mid_channels=mid_channels,
                max_hypotheses=self.hvp_cbea_cfg.get("max_hypotheses", 50),
                hyp_conf_threshold=self.hvp_cbea_cfg.get("hyp_conf_threshold", 0.15),
                pc_range=self.hvp_cbea_cfg.get("pc_range"),
            )
            self.hypothesis_verifier = HypothesisVerifier(
                in_channels=in_channels,
                mid_channels=mid_channels,
                max_novel=self.hvp_cbea_cfg.get("max_novel", 20),
                novel_threshold=self.hvp_cbea_cfg.get("novel_threshold", 0.5),
            )
            self.bayesian_hypothesis_fusion = BayesianHypothesisFusion(
                in_channels=in_channels,
                mid_channels=mid_channels,
                pc_range=self.hvp_cbea_cfg.get("pc_range"),
                confirm_boost=self.hvp_cbea_cfg.get("confirm_boost", 1.5),
                refute_penalty=self.hvp_cbea_cfg.get("refute_penalty", -2.5),
                refine_boost=self.hvp_cbea_cfg.get("refine_boost", 0.8),
                residual_gate=self.hvp_cbea_cfg.get("residual_gate"),
            )
            if self.hvp_train_only:
                self._freeze_non_hvp_parameters()
                self._set_frozen_heal_modules_eval()
            self.hvp_trainable_summary = self._summarize_trainable_parameters()
            if self.hvp_cbea_cfg.get("debug", False):
                print("HVP-CBEA trainable summary:", self.hvp_trainable_summary)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.hvp_cbea_enabled and self.hvp_train_only:
            self._set_frozen_heal_modules_eval()
            self.hvp_trainable_summary = self._summarize_trainable_parameters()
        return self

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list']
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len']
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = getattr(self, f"encoder_{modality_name}")(data_dict, modality_name)
            feature = getattr(self, f"backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = getattr(self, f"aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict and self.sensor_type_dict[modality_name] == "camera":
                import torchvision

                feature = modality_feature_dict[modality_name]
                _, _, h, w = feature.shape
                target_h = int(h * getattr(self, f"crop_ratio_H_{modality_name}"))
                target_w = int(w * getattr(self, f"crop_ratio_W_{modality_name}"))
                crop_func = torchvision.transforms.CenterCrop((target_h, target_w))
                modality_feature_dict[modality_name] = crop_func(feature)
                if getattr(self, f"depth_supervision_{modality_name}"):
                    output_dict.update({
                        f"depth_items_{modality_name}": getattr(self, f"encoder_{modality_name}").depth_items
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

        if self.supervise_single:
            # Preserve the official single-supervision prediction path:
            # per-agent BEV -> pyramid forward_single -> optional shrink -> shared heads.
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

        fused_feature, hvp_debug, hvp_loss = self._apply_hvp_cbea(
            fused_feature=fused_feature,
            heter_feature_2d=heter_feature_2d,
            record_len=record_len,
        )

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        output_dict.update({'occ_single_list': occ_outputs})
        if self.hvp_cbea_enabled:
            output_dict.update({'hvp_cbea_debug': hvp_debug})
            if hvp_loss is not None:
                output_dict['hvp_cbea_loss'] = hvp_loss
        return output_dict

    def _apply_hvp_cbea(self, fused_feature, heter_feature_2d, record_len):
        bn_mode_summary = self._summarize_bn_modes()
        debug = {
            "hvp_cbea_enabled": bool(self.hvp_cbea_enabled),
            "train_only_hvp": bool(self.hvp_train_only),
            "hvp_trainable_summary": self.hvp_trainable_summary,
            "frozen_bn_eval_count": bn_mode_summary["frozen_bn_eval_count"],
            "hvp_bn_train_count": bn_mode_summary["hvp_bn_train_count"],
            "ego_hyp_count": 0,
            "collaborator_count": 0,
            "verifier_used": False,
            "novel_count": 0,
            "updated_hyp_count": 0,
            "hvp_loss": 0.0,
            "fallback_reason": "",
            "feature_shape_before": list(fused_feature.shape),
            "feature_shape_after": list(fused_feature.shape),
        }
        if self.hvp_cbea_enabled:
            debug.update(self.bayesian_hypothesis_fusion.get_residual_debug())
        if not self.hvp_cbea_enabled:
            return fused_feature, debug, None

        try:
            ego_hyps, hmap, reg = self.hypothesis_encoder(fused_feature)
            debug["ego_hyp_count"] = int((ego_hyps[..., 8] > 0).sum().detach().cpu()) if ego_hyps is not None else 0
            collaborator_feat = self._collect_collaborator_features(heter_feature_2d, record_len)
            debug["collaborator_count"] = int(collaborator_feat.shape[0]) if collaborator_feat is not None else 0
            if collaborator_feat is None:
                verif_logits, refine_delta, novel_hyps = None, None, None
                debug["fallback_reason"] = "no_compatible_collaborator_features"
            else:
                verif_logits, refine_delta, novel_hyps = self.hypothesis_verifier(collaborator_feat, ego_hyps)
                debug["verifier_used"] = True
                debug["novel_count"] = int((novel_hyps[..., 8] > 0).sum().detach().cpu()) if novel_hyps is not None else 0
            fused_feature_out, updated_hyps = self.bayesian_hypothesis_fusion(
                ego_hyps,
                verif_logits,
                refine_delta,
                novel_hyps,
                fused_feature,
            )
            debug.update(self.bayesian_hypothesis_fusion.get_residual_debug())
            debug["updated_hyp_count"] = int((updated_hyps[..., 8] > 0).sum().detach().cpu()) if updated_hyps is not None else 0
            debug["feature_shape_after"] = list(fused_feature_out.shape)
            hvp_loss = self._compute_hvp_loss(hmap, reg, verif_logits, updated_hyps)
            debug["hvp_loss"] = float(hvp_loss.detach().cpu()) if torch.is_tensor(hvp_loss) else 0.0
            return fused_feature_out, debug, hvp_loss
        except Exception as exc:
            if self.hvp_cbea_cfg.get("fallback_on_error", True):
                debug["fallback_reason"] = "exception:%s" % type(exc).__name__
                return fused_feature, debug, fused_feature.sum() * 0.0
            raise

    def _collect_collaborator_features(self, heter_feature_2d, record_len):
        if heter_feature_2d is None or heter_feature_2d.ndim != 4:
            return None
        lengths = record_len.detach().cpu().tolist() if torch.is_tensor(record_len) else list(record_len)
        collab = []
        start = 0
        for length in lengths:
            length = int(length)
            if length > 1:
                collab.append(heter_feature_2d[start + 1:start + length])
            start += length
        if not collab:
            return None
        collab_feat = torch.cat(collab, dim=0)
        return self.hvp_collaborator_proj(collab_feat)

    def _compute_hvp_loss(self, hmap, reg, verif_logits, updated_hyps):
        loss = None
        for tensor in (hmap, reg, verif_logits, updated_hyps):
            if torch.is_tensor(tensor):
                term = tensor.sum() * 0.0
                loss = term if loss is None else loss + term
        if loss is None:
            return None
        weight = float(self.hvp_cbea_cfg.get("loss_weight_encoder", 0.5))
        return loss * weight

    def _default_hvp_cfg(self, args):
        pc_range = args.get("lidar_range") or args.get("cav_lidar_range")
        return {
            "enabled": False,
            "in_channels": args.get("in_head", 256),
            "collaborator_in_channels": self._infer_collaborator_channels(args, args.get("in_head", 256)),
            "pc_range": pc_range,
            "max_hypotheses": 50,
            "hyp_conf_threshold": 0.15,
            "mid_channels": 64,
            "novel_threshold": 0.5,
            "max_novel": 20,
            "confirm_boost": 1.5,
            "refute_penalty": -2.5,
            "refine_boost": 0.8,
            "loss_weight_encoder": 0.5,
            "loss_weight_verifier": 0.3,
            "loss_weight_fusion": 0.2,
            "fallback_on_error": True,
            "debug": False,
            "train_only_hvp": False,
            "residual_gate": self._default_residual_gate_cfg(),
        }

    @staticmethod
    def _default_residual_gate_cfg():
        return {
            "enabled": True,
            "alpha_init": 0.05,
            "alpha_max": 0.3,
            "learnable": True,
        }

    @classmethod
    def _normalize_residual_gate_cfg(cls, cfg):
        normalized = cls._default_residual_gate_cfg()
        if isinstance(cfg, bool):
            normalized["enabled"] = cfg
        elif isinstance(cfg, dict):
            normalized.update(cfg)
        normalized["enabled"] = bool(normalized.get("enabled", True))
        normalized["alpha_init"] = float(normalized.get("alpha_init", 0.05))
        normalized["alpha_max"] = float(normalized.get("alpha_max", 0.3))
        normalized["learnable"] = bool(normalized.get("learnable", True))
        return normalized

    def _freeze_non_hvp_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad_(self._is_hvp_module_name(name))

    def _set_frozen_heal_modules_eval(self):
        for module_name, module in self.named_modules():
            if self._is_hvp_module_name(module_name):
                continue
            if isinstance(module, self.BN_TYPES):
                module.eval()

    def _summarize_trainable_parameters(self):
        summary = {
            "trainable_total": 0,
            "frozen_total": 0,
            "trainable_prefix_count": {},
            "train_only_hvp": bool(self.hvp_train_only),
        }
        for name, param in self.named_parameters():
            count = int(param.numel())
            if param.requires_grad:
                summary["trainable_total"] += count
                prefix = name.split(".", 1)[0]
                summary["trainable_prefix_count"][prefix] = (
                    summary["trainable_prefix_count"].get(prefix, 0) + count
                )
            else:
                summary["frozen_total"] += count
        summary.update(self._summarize_bn_modes())
        return summary

    def _summarize_bn_modes(self):
        summary = {
            "frozen_bn_eval_count": 0,
            "hvp_bn_train_count": 0,
        }
        for module_name, module in self.named_modules():
            if not isinstance(module, self.BN_TYPES):
                continue
            if self._is_hvp_module_name(module_name):
                if module.training:
                    summary["hvp_bn_train_count"] += 1
            elif not module.training:
                summary["frozen_bn_eval_count"] += 1
        return summary

    def _is_hvp_module_name(self, module_name):
        return any(module_name.startswith(prefix) for prefix in self.HVP_MODULE_PREFIXES)

    @staticmethod
    def _infer_collaborator_channels(args, fallback):
        fusion_backbone = args.get("fusion_backbone", {}) or {}
        filters = fusion_backbone.get("num_filters", [])
        if isinstance(filters, (list, tuple)) and len(filters) > 0:
            return filters[0]
        return fallback

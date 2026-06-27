"""HEAL-XLab-v2 HVP-CBEA model wrapper.

This model keeps the official HEAL heter_pyramid_collab path intact unless
model.args.hvp_cbea.enabled is explicitly true.
"""

from collections import Counter

import torch
import torch.nn as nn

from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.sub_modules.hypothesis_encoder import HypothesisEncoder
from opencood.models.sub_modules.hypothesis_verifier import HypothesisVerifier
from opencood.models.sub_modules.bayesian_hypothesis_fusion import BayesianHypothesisFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm


class HeterPyramidCollabHvpCbea(HeterPyramidCollab):
    """HeterPyramidCollab with optional HVP-CBEA before detection heads."""

    def __init__(self, args):
        super().__init__(args)
        hvp = args.get("hvp_cbea", {}) or {}
        self.hvp_cbea_cfg = self._default_hvp_cfg(args)
        self.hvp_cbea_cfg.update(hvp)
        self.hvp_cbea_enabled = bool(self.hvp_cbea_cfg.get("enabled", False))
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
            )

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
        debug = {
            "hvp_cbea_enabled": bool(self.hvp_cbea_enabled),
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
        }

    @staticmethod
    def _infer_collaborator_channels(args, fallback):
        fusion_backbone = args.get("fusion_backbone", {}) or {}
        filters = fusion_backbone.get("num_filters", [])
        if isinstance(filters, (list, tuple)) and len(filters) > 0:
            return filters[0]
        return fallback

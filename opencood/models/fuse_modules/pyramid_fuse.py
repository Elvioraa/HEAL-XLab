# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.resblock import ResNetModified, Bottleneck, BasicBlock
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.visualization.debug_plot import plot_feature


def _softmax_nan_to_zero(scores):
    weights = torch.softmax(scores, dim=0)
    return torch.where(
        torch.isnan(weights),
        torch.zeros_like(weights),
        weights,
    )


def weighted_fuse(x, score, record_len, affine_matrix, align_corners,
                  cbea_alpha=None, cbea_lambda=0.0, cbea_exclude_threshold=0.0,
                  cbea_exclude_floor_mix=0.0):
    """
    Parameters
    ----------
    x : torch.Tensor
        input data, (sum(n_cav), C, H, W)

    score : torch.Tensor
        score, (sum(n_cav), 1, H, W)

    record_len : list
        shape: (B)

    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm'
        shape: (B, L, L, 2, 3)

    cbea_exclude_threshold : float
        Relative-reliability floor tau in [0.0, 1.0). At a given pixel, an
        agent whose N * alpha_i falls below tau is excluded from the
        vehicle-dimension softmax entirely (score set to -inf) instead of
        being continuously down-weighted. tau=0.0 (default) never triggers,
        so this is bit-exact identical to the pre-existing gate-only
        behavior. If excluding would leave zero valid agents at a pixel,
        exclusion is skipped there and the pixel falls back to the
        gate-only result, so this can never empty an otherwise non-empty
        softmax.

    cbea_exclude_floor_mix : float
        Blend factor mu in [0.0, 1.0] between the hard-excluded softmax
        weights and the gate-only (no exclusion) softmax weights:
        w_final = (1 - mu) * w_hard + mu * w_gate_only. mu=0.0 (default)
        reproduces the hard-exclusion result bit-for-bit; mu=1.0
        reproduces the tau=0.0 gate-only result bit-for-bit. Only has an
        effect when cbea_exclude_threshold > 0.0.
    """

    if not isinstance(cbea_exclude_threshold, (int, float)):
        raise ValueError("cbea_exclude_threshold must be a number")
    if float(cbea_exclude_threshold) < 0.0:
        raise ValueError("cbea_exclude_threshold must be >= 0.0")
    if not isinstance(cbea_exclude_floor_mix, (int, float)):
        raise ValueError("cbea_exclude_floor_mix must be a number")
    if not 0.0 <= float(cbea_exclude_floor_mix) <= 1.0:
        raise ValueError("cbea_exclude_floor_mix must be in [0.0, 1.0]")

    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    # score = torch.sum(score, dim=1, keepdim=True)
    split_score = regroup(score, record_len)
    use_cbea_prior = cbea_alpha is not None and float(cbea_lambda) != 0.0
    split_cbea_alpha = None
    if use_cbea_prior:
        if not torch.is_tensor(cbea_alpha) or cbea_alpha.ndim != 4:
            raise ValueError("cbea_alpha must have shape [sum(record_len), 1, H, W]")
        if cbea_alpha.shape[1] != 1:
            raise ValueError("cbea_alpha channel dimension must be 1")
        if int(cbea_alpha.shape[0]) != int(torch.sum(record_len).item()):
            raise ValueError("cbea_alpha agent count does not match record_len")
        if not torch.isfinite(cbea_alpha).all().item():
            raise ValueError("cbea_alpha contains NaN or Inf")
        split_cbea_alpha = regroup(cbea_alpha, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        score = split_score[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0 # ego
        feature_in_ego = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        if use_cbea_prior:
            alpha_in_ego = split_cbea_alpha[b]
            if int(alpha_in_ego.shape[0]) != int(N):
                raise ValueError("cbea_alpha scene agent count does not match record_len")
            alpha_in_ego = alpha_in_ego.to(
                device=scores_in_ego.device,
                dtype=scores_in_ego.dtype,
            )
            if alpha_in_ego.shape[-2:] != (H, W):
                alpha_in_ego = F.interpolate(
                    alpha_in_ego,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
            valid_mask = scores_in_ego != 0
            relative_prior = int(N) * alpha_in_ego
            gate = (1.0 - float(cbea_lambda)) + float(cbea_lambda) * relative_prior
            gate = torch.clamp(gate, min=1e-6)
            if not torch.isfinite(gate).all().item():
                raise ValueError("CBEA multiscale prior gate contains NaN or Inf")
            gated_scores = scores_in_ego * gate
            gated_scores = gated_scores.masked_fill(~valid_mask, -float('inf'))

            if float(cbea_exclude_threshold) > 0.0:
                exclude_candidate = valid_mask & (
                    relative_prior < float(cbea_exclude_threshold)
                )
                remaining_valid = valid_mask & ~exclude_candidate
                all_excluded = remaining_valid.sum(dim=0, keepdim=True) == 0
                final_exclude = exclude_candidate & ~all_excluded
                hard_scores = gated_scores.masked_fill(final_exclude, -float('inf'))
                scores_in_ego = _softmax_nan_to_zero(hard_scores)

                floor_mix = float(cbea_exclude_floor_mix)
                if floor_mix > 0.0:
                    gate_only_weights = _softmax_nan_to_zero(gated_scores)
                    scores_in_ego = (
                        (1.0 - floor_mix) * scores_in_ego
                        + floor_mix * gate_only_weights
                    )
            else:
                scores_in_ego = _softmax_nan_to_zero(gated_scores)
        else:
            scores_in_ego.masked_fill_(scores_in_ego == 0, -float('inf'))
            scores_in_ego = _softmax_nan_to_zero(scores_in_ego)

        out.append(torch.sum(feature_in_ego * scores_in_ego, dim=0))
    out = torch.stack(out)
    
    return out

class PyramidFusion(ResNetBEVBackbone):
    def __init__(self, model_cfg, input_channels=64):
        """
        Do not downsample in the first layer.
        """
        super().__init__(model_cfg, input_channels)
        if model_cfg["resnext"]:
            Bottleneck.expansion = 1
            self.resnet = ResNetModified(Bottleneck, 
                                        self.model_cfg['layer_nums'],
                                        self.model_cfg['layer_strides'],
                                        self.model_cfg['num_filters'],
                                        inplanes = model_cfg.get('inplanes', 64),
                                        groups=32,
                                        width_per_group=4)
        self.align_corners = model_cfg.get('align_corners', False)
        print('Align corners: ', self.align_corners)
        
        # add single supervision head
        for i in range(self.num_levels):
            setattr(
                self,
                f"single_head_{i}",
                nn.Conv2d(self.model_cfg["num_filters"][i], 1, kernel_size=1),
            )

    def forward_single(self, spatial_features):
        """
        This is used for single agent pass.
        """
        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
        final_feature = self.decode_multiscale_feature(feature_list)

        return final_feature, occ_map_list
    
    def forward_collab(self, spatial_features, record_len, affine_matrix,
                       agent_modality_list=None, cam_crop_info=None,
                       cbea_alpha=None, cbea_lambda=0.0, cbea_exclude_threshold=0.0,
                       cbea_exclude_floor_mix=0.0):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2':
                {
                    'crop_ratio_W_m2': 0.5,
                    'crop_ratio_H_m2': 0.5,
                }
            }
        """
        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list] 
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}


        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask

            fused_feature_list.append(weighted_fuse(
                feature_list[i],
                score,
                record_len,
                affine_matrix,
                self.align_corners,
                cbea_alpha=cbea_alpha,
                cbea_lambda=cbea_lambda,
                cbea_exclude_threshold=cbea_exclude_threshold,
                cbea_exclude_floor_mix=cbea_exclude_floor_mix,
            ))
        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        
        return fused_feature, occ_map_list

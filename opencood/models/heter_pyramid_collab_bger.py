"""BGER: Box-Guided Evidence Reactivation on top of HEAL.

Decision-level communication, feature-level fusion:

- Collaborators transmit only detection-box messages (box + confidence),
  either oracle GT boxes visible from their own viewpoint (upper-bound
  experiments) or boxes decoded from their own single-agent branch
  (realistic experiments). No BEV feature is ever transmitted.
- The ego vehicle runs its own single-view pipeline, projects the received
  boxes into its BEV frame, renders them into prior maps, and re-examines
  its own features through a small trainable refinement module before the
  shared detection heads.

Everything is gated behind ``model.args.bger.enabled``. When the flag is
absent or false, this model behaves exactly like the official
``HeterPyramidCollab``.
"""

from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.sub_modules.bger_box_prior import BGERBoxPrior
from opencood.models.sub_modules.bger_refine import BGERRefine
from opencood.utils import box_utils
from opencood.utils.common_utils import limit_period


class HeterPyramidCollabBger(HeterPyramidCollab):
    """HEAL collab model with optional box-guided evidence reactivation."""

    BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def __init__(self, args):
        super(HeterPyramidCollabBger, self).__init__(args)
        self.supervise_single = bool(args.get("supervise_single", False))
        self.bger_cfg = self._normalize_bger_cfg(args.get("bger"))
        self.bger_enabled = bool(self.bger_cfg["enabled"])
        self.bger_freeze_base = bool(self.bger_cfg["freeze_base"])

        if self.bger_enabled:
            prior_cfg = dict(self.bger_cfg["prior"])
            prior_cfg["lidar_range"] = args["lidar_range"]
            self.bger_box_prior = BGERBoxPrior(prior_cfg)

            refine_cfg = dict(self.bger_cfg["refine"])
            refine_cfg["in_channels"] = int(args["in_head"])
            refine_cfg["prior_channels"] = self.bger_box_prior.num_channels
            self.bger_refine = BGERRefine(refine_cfg)

            if self.bger_freeze_base:
                self._freeze_base_parameters()

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_bger_cfg(cfg):
        normalized = {
            "enabled": False,
            "box_source": "oracle",  # oracle | single_decode
            "mode": "refine",        # refine | box_merge_only
            "freeze_base": True,
            "order": "hwl",
            "comm_log": True,
            "oracle": {
                "score": 1.0,
            },
            "single_decode": {
                "score_threshold": 0.25,
                "nms_thresh": 0.15,
                "max_boxes": 30,
            },
            "prior": {
                "gaussian": True,
                "box_mask": True,
                "yaw": False,
                "sigma_scale": 0.25,
                "min_sigma": 1.0,
            },
            "refine": {
                "hidden_dim": 128,
                "num_layers": 2,
                "norm": "bn",
                "gate_init": 1.0,
            },
        }
        if isinstance(cfg, bool):
            normalized["enabled"] = bool(cfg)
        elif isinstance(cfg, dict):
            _deep_update(normalized, cfg)
        elif cfg is not None:
            raise ValueError("model.args.bger must be a bool or a dict")

        if normalized["box_source"] not in ("oracle", "single_decode"):
            raise ValueError(
                "bger.box_source must be 'oracle' or 'single_decode', got %r"
                % (normalized["box_source"],)
            )
        if normalized["mode"] not in ("refine", "box_merge_only"):
            raise ValueError(
                "bger.mode must be 'refine' or 'box_merge_only', got %r"
                % (normalized["mode"],)
            )
        return normalized

    # ------------------------------------------------------------------
    # freezing
    # ------------------------------------------------------------------
    def _freeze_base_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("bger_refine."):
                param.requires_grad_(False)
        self._set_frozen_bn_eval()

    def _set_frozen_bn_eval(self):
        for name, module in self.named_modules():
            if isinstance(module, self.BN_TYPES) and not name.startswith("bger_refine"):
                module.eval()

    def train(self, mode=True):
        super(HeterPyramidCollabBger, self).train(mode)
        # model.train() is called every epoch by the trainer; keep the frozen
        # base BN buffers from drifting while only bger_refine is trainable.
        if mode and self.bger_enabled and self.bger_freeze_base:
            self._set_frozen_bn_eval()
        return self

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, data_dict):
        if not self.bger_enabled:
            return super(HeterPyramidCollabBger, self).forward(data_dict)

        output_dict = {"pyramid": "collab"}
        agent_modality_list = data_dict["agent_modality_list"]
        record_len = data_dict["record_len"]
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = getattr(self, f"encoder_{modality_name}")(data_dict, modality_name)
            feature = getattr(self, f"backbone_{modality_name}")({
                "spatial_features": feature,
            })["spatial_features_2d"]
            feature = getattr(self, f"aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict and \
                    self.sensor_type_dict[modality_name] == "camera":
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
                            self, f"encoder_{modality_name}"
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

        # Per-agent single-view pipeline. No cross-agent feature fusion:
        # collaborators only ever contribute box messages.
        single_feature, single_occ_outputs = \
            self.pyramid_backbone.forward_single(heter_feature_2d)
        if self.shrink_flag:
            single_feature = self.shrink_conv(single_feature)

        if self.supervise_single:
            output_dict.update({
                "cls_preds_single": self.cls_head(single_feature),
                "reg_preds_single": self.reg_head(single_feature),
                "dir_preds_single": self.dir_head(single_feature),
            })

        ego_indices = self._ego_indices(record_len)
        ego_feature = single_feature[ego_indices]
        _, _, feat_h, feat_w = ego_feature.shape

        # Collaborator box messages (never a gradient path).
        with torch.no_grad():
            if self.bger_cfg["box_source"] == "oracle":
                boxes_list, scores_list = self._collab_boxes_oracle(
                    data_dict, record_len, pairwise_t_matrix
                )
            else:
                boxes_list, scores_list = self._collab_boxes_single_decode(
                    data_dict, single_feature, record_len, pairwise_t_matrix
                )
            prior_map = self.bger_box_prior(
                boxes_list, scores_list, (feat_h, feat_w),
                ego_feature.device, ego_feature.dtype,
            )

        if self.bger_cfg["mode"] == "refine":
            refined_feature, _ = self.bger_refine(ego_feature, prior_map)
        else:
            # box_merge_only: pure ego feature; collaborator boxes are only
            # exposed in output_dict['bger'] for box-level merging at
            # inference time (late-fusion baseline).
            refined_feature = ego_feature

        cls_preds = self.cls_head(refined_feature)
        reg_preds = self.reg_head(refined_feature)
        dir_preds = self.dir_head(refined_feature)

        bger_debug = {
            "enabled": True,
            "box_source": self.bger_cfg["box_source"],
            "mode": self.bger_cfg["mode"],
            "freeze_base": self.bger_freeze_base,
            "num_collab_boxes": [int(b.shape[0]) for b in boxes_list],
            "collab_boxes_ego_frame": boxes_list,
            "collab_box_scores": scores_list,
        }
        if self.bger_cfg["comm_log"]:
            bger_debug.update(self._comm_stats(
                boxes_list, record_len, single_feature
            ))

        output_dict.update({
            "cls_preds": cls_preds,
            "reg_preds": reg_preds,
            "dir_preds": dir_preds,
            "occ_single_list": single_occ_outputs,
            "bger": bger_debug,
        })
        return output_dict

    # ------------------------------------------------------------------
    # collaborator box messages
    # ------------------------------------------------------------------
    @staticmethod
    def _ego_indices(record_len):
        cumsum = torch.cumsum(record_len, dim=0)
        starts = torch.cat([cumsum.new_zeros(1), cumsum[:-1]])
        return starts.long()

    def _collab_boxes_oracle(self, data_dict, record_len, pairwise_t_matrix):
        """Collaborator messages = GT boxes visible from their own viewpoint."""
        centers_single = data_dict["object_bbx_center_single"]
        mask_single = data_dict["object_bbx_mask_single"]
        oracle_score = float(self.bger_cfg["oracle"]["score"])

        boxes_list, scores_list = [], []
        flat_idx = 0
        for batch_idx in range(len(record_len)):
            cav_num = int(record_len[batch_idx])
            sample_boxes = []
            sample_scores = []
            for local_idx in range(cav_num):
                agent_idx = flat_idx + local_idx
                if local_idx == 0:  # ego sends nothing to itself
                    continue
                keep = mask_single[agent_idx].bool()
                boxes_own = centers_single[agent_idx][keep].float()
                if boxes_own.shape[0] == 0:
                    continue
                scores_own = torch.full(
                    (boxes_own.shape[0],), oracle_score,
                    device=boxes_own.device, dtype=boxes_own.dtype,
                )
                boxes_ego, scores_ego = self._project_boxes_to_ego(
                    boxes_own, scores_own,
                    pairwise_t_matrix[batch_idx, local_idx, 0],
                )
                if boxes_ego.shape[0] == 0:
                    continue
                sample_boxes.append(boxes_ego)
                sample_scores.append(scores_ego)
            boxes_list.append(self._cat_or_empty(sample_boxes, centers_single))
            scores_list.append(self._cat_or_empty_scores(sample_scores, centers_single))
            flat_idx += cav_num
        return boxes_list, scores_list

    def _collab_boxes_single_decode(self, data_dict, single_feature,
                                    record_len, pairwise_t_matrix):
        """Collaborator messages = boxes decoded from their single branch."""
        decode_cfg = self.bger_cfg["single_decode"]
        anchor_box = data_dict["anchor_box"]
        if not torch.is_tensor(anchor_box):
            anchor_box = torch.from_numpy(np.asarray(anchor_box))
        anchor_box = anchor_box.to(single_feature.device).float()

        cls_single = self.cls_head(single_feature)
        reg_single = self.reg_head(single_feature)
        dir_single = self.dir_head(single_feature)

        boxes_list, scores_list = [], []
        flat_idx = 0
        for batch_idx in range(len(record_len)):
            cav_num = int(record_len[batch_idx])
            sample_boxes = []
            sample_scores = []
            for local_idx in range(1, cav_num):
                agent_idx = flat_idx + local_idx
                boxes_own, scores_own = self._decode_single_agent(
                    cls_single[agent_idx:agent_idx + 1],
                    reg_single[agent_idx:agent_idx + 1],
                    dir_single[agent_idx:agent_idx + 1],
                    anchor_box, decode_cfg,
                )
                if boxes_own.shape[0] == 0:
                    continue
                boxes_ego, scores_ego = self._project_boxes_to_ego(
                    boxes_own, scores_own,
                    pairwise_t_matrix[batch_idx, local_idx, 0],
                )
                if boxes_ego.shape[0] == 0:
                    continue
                sample_boxes.append(boxes_ego)
                sample_scores.append(scores_ego)
            boxes_list.append(self._cat_or_empty(sample_boxes, single_feature))
            scores_list.append(self._cat_or_empty_scores(sample_scores, single_feature))
            flat_idx += cav_num
        return boxes_list, scores_list

    def _decode_single_agent(self, cls_pred, reg_pred, dir_pred,
                             anchor_box, decode_cfg):
        """Decode one agent's dense predictions to boxes in its own frame.

        Mirrors VoxelPostprocessor.post_process's per-cav decoding path.
        """
        from opencood.data_utils.post_processor.voxel_postprocessor import \
            VoxelPostprocessor

        prob = torch.sigmoid(cls_pred.permute(0, 2, 3, 1)).reshape(1, -1)
        batch_box3d = VoxelPostprocessor.delta_to_boxes3d(reg_pred, anchor_box)

        mask = torch.gt(prob, float(decode_cfg["score_threshold"]))
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
        boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
        scores = torch.masked_select(prob[0], mask[0])
        if boxes3d.shape[0] == 0:
            return boxes3d, scores

        dir_args = self.args.get("dir_args", {})
        dir_offset = float(dir_args.get("dir_offset", 0.7853))
        num_bins = int(dir_args.get("num_bins", 2))
        dir_cls_preds = dir_pred.permute(0, 2, 3, 1).contiguous().reshape(
            1, -1, num_bins
        )
        dir_cls_preds = dir_cls_preds[mask]
        dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
        period = (2 * np.pi / num_bins)
        dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)
        boxes3d[..., 6] = dir_rot + dir_offset + \
            period * dir_labels.to(dir_cls_preds.dtype)
        boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)

        corners = box_utils.boxes_to_corners_3d(boxes3d, order=self.bger_cfg["order"])
        keep_index = box_utils.nms_rotated(
            corners, scores, float(decode_cfg["nms_thresh"])
        )
        boxes3d = boxes3d[keep_index]
        scores = scores[keep_index]

        max_boxes = int(decode_cfg["max_boxes"])
        if boxes3d.shape[0] > max_boxes:
            top = torch.argsort(scores, descending=True)[:max_boxes]
            boxes3d = boxes3d[top]
            scores = scores[top]
        return boxes3d, scores

    def _project_boxes_to_ego(self, boxes_own, scores_own, t_matrix):
        """Project center-format boxes from a collaborator frame to ego.

        pairwise_t_matrix[i, j] is the i->j transform, so the caller passes
        pairwise_t_matrix[batch, collaborator, 0] (collaborator -> ego).
        Boxes leaving the ego perception range are dropped together with
        their scores.
        """
        t_matrix = t_matrix.to(boxes_own.device).float()
        corners = box_utils.boxes_to_corners_3d(
            boxes_own, order=self.bger_cfg["order"]
        )
        corners_ego = box_utils.project_box3d(corners, t_matrix)
        range_mask = box_utils.get_mask_for_boxes_within_range_torch(
            corners_ego, self.cav_range
        )
        corners_ego = corners_ego[range_mask]
        scores_ego = scores_own[range_mask]
        if corners_ego.shape[0] == 0:
            return corners_ego.new_zeros((0, 7)), scores_ego
        boxes_ego = box_utils.corner_to_center_torch(
            corners_ego, order=self.bger_cfg["order"]
        )
        return boxes_ego, scores_ego

    @staticmethod
    def _cat_or_empty(tensors, reference):
        if len(tensors) == 0:
            return reference.new_zeros((0, 7))
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _cat_or_empty_scores(tensors, reference):
        if len(tensors) == 0:
            return reference.new_zeros((0,))
        return torch.cat(tensors, dim=0)

    # ------------------------------------------------------------------
    # communication accounting
    # ------------------------------------------------------------------
    @staticmethod
    def _comm_stats(boxes_list, record_len, single_feature):
        bytes_per_box = (7 + 1) * 4  # box params + confidence, fp32
        _, channels, height, width = single_feature.shape
        feature_bytes_per_agent = channels * height * width * 4

        total_boxes = sum(int(b.shape[0]) for b in boxes_list)
        num_collaborators = int(sum(max(int(n) - 1, 0) for n in record_len))
        comm_bytes_boxes = total_boxes * bytes_per_box
        comm_bytes_feature_equiv = num_collaborators * feature_bytes_per_agent
        return {
            "comm_bytes_boxes": comm_bytes_boxes,
            "comm_bytes_feature_equiv": comm_bytes_feature_equiv,
            "comm_ratio": (
                comm_bytes_boxes / comm_bytes_feature_equiv
                if comm_bytes_feature_equiv > 0 else 0.0
            ),
        }


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

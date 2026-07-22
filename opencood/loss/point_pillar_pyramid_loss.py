# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.loss.point_pillar_loss import sigmoid_focal_loss
from opencood.loss.hvp_cbea_aux_loss import compute_pact_cbea_local_evidence_loss

class PointPillarPyramidLoss(PointPillarDepthLoss):
    def __init__(self, args):
        super().__init__(args)
        self.pyramid = args['pyramid']

        # relative downsampled GT cls map from fused labels.
        self.relative_downsample = self.pyramid['relative_downsample']
        self.pyramid_weight = self.pyramid['weight']
        self.num_levels = len(self.relative_downsample)
    
    def forward(self, output_dict, target_dict, suffix=""):
        if output_dict['pyramid'] == 'collab': # intermediate fusion, pyramid collab.
            return self.forward_collab(output_dict, target_dict, suffix)

        elif output_dict['pyramid'] == 'single': # late fusion, pyramid single 
            return self.forward_single(output_dict, target_dict, suffix)
        raise
        
    def forward_single(self, output_dict, target_dict, suffix):
        """
        for heter_pyramid_single
        """
        batch_size = target_dict['pos_equal_one'].shape[0]
        total_loss = super().forward(output_dict, target_dict, suffix)

        occ_single_list = output_dict['occ_single_list']    
        occ_loss = self.calc_occ_loss(occ_single_list, target_dict['pos_equal_one'], target_dict['neg_equal_one'], batch_size)
        total_loss += occ_loss
        self.loss_dict.update({
            'pyramid_loss': occ_loss.item(),
            'total_loss': total_loss.item()
        })
        return total_loss

    def forward_collab(self, output_dict, target_dict, suffix):
        """
        for heter_pyramid_collab
        """
        if suffix == "": 
            return super().forward(output_dict, target_dict)
        assert suffix == "_single"
        batch_size = target_dict['pos_equal_one'].shape[0]

        positives = target_dict['pos_equal_one']
        negatives = target_dict['neg_equal_one']

        occ_single_list = output_dict['occ_single_list']
        occ_loss = self.calc_occ_loss(occ_single_list, positives, negatives, batch_size)
        total_loss = occ_loss
        self.loss_dict = {
            'pyramid_loss': occ_loss.item(),
            'total_loss': total_loss.item()
        }

        # When the evidence head is attached to the per-agent single feature,
        # its supervision belongs here: this branch receives the per-agent
        # single target (label_dict_single) and reg_preds_single, which match
        # the per-agent evidence tensors. attach_to='fused' (default) leaves
        # this untouched and keeps the loss in the collab (suffix="") path.
        if (
            "pact_cbea" in output_dict
            and output_dict["pact_cbea"].get("attach_to", "fused") == "single"
        ):
            pact_loss, pact_stats = compute_pact_cbea_local_evidence_loss(
                output_dict["pact_cbea"],
                target_dict=target_dict,
                fallback_on_error=True,
                reg_preds=output_dict.get("reg_preds_single"),
            )
            if (
                torch.is_tensor(pact_loss)
                and pact_stats.get("pact_cbea_local_evidence_enabled", False)
            ):
                total_loss = total_loss + pact_loss
            self.loss_dict.update(pact_stats)
            self.loss_dict['total_loss'] = total_loss.item()

        return total_loss


    def calc_occ_loss(self, occ_single_list, positives, negatives, batch_size):
        total_occ_loss = 0
        occ_positives = torch.logical_or(positives[...,0], positives[...,1]).unsqueeze(-1).float() # N, H, W
        occ_negatives = torch.logical_and(negatives[...,0], negatives[...,1]).unsqueeze(-1).float() # N, H, W

        for i, occ_preds_single in enumerate(occ_single_list):
            """
            occ_preds_single: N, 1, H, W

            occ_positives: N, H, W, 1
            occ_negatives: N, H, W, 1

            """

            positives_level = F.max_pool2d(occ_positives.permute(0,3,1,2), kernel_size=self.relative_downsample[i]).permute(0,2,3,1)
            negatives_level = 1 - F.max_pool2d((1 - occ_negatives).permute(0,3,1,2), kernel_size=self.relative_downsample[i]).permute(0,2,3,1)

            occ_labls = positives_level.view(batch_size, -1, 1)
            positives_level = occ_labls
            negatives_level = negatives_level.view(batch_size, -1, 1)

            pos_normalizer = positives_level.sum(1, keepdim=True).float()

            occ_preds = occ_preds_single.permute(0, 2, 3, 1).contiguous() \
                        .view(batch_size, -1,  1)
            occ_weights = positives_level * self.pos_cls_weight + negatives_level * 1.0
            occ_weights /= torch.clamp(pos_normalizer, min=1.0)
            occ_loss = sigmoid_focal_loss(occ_preds, occ_labls, weights=occ_weights, **self.cls)
            occ_loss = occ_loss.sum() / batch_size
            occ_loss *= self.pyramid_weight[i]

            total_occ_loss += occ_loss


        return total_occ_loss
    



    def logging(self, epoch, batch_id, batch_len, writer = None, suffix=""):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict.get('total_loss', 0)
        reg_loss = self.loss_dict.get('reg_loss', 0)
        cls_loss = self.loss_dict.get('cls_loss', 0)
        dir_loss = self.loss_dict.get('dir_loss', 0)
        iou_loss = self.loss_dict.get('iou_loss', 0)
        depth_loss = self.loss_dict.get('depth_loss', 0)
        pyramid_loss = self.loss_dict.get('pyramid_loss', 0)
        hvp_v3_enabled = self.loss_dict.get('hvp_v3_enabled', False)
        hvp_v3_loss = self.loss_dict.get('hvp_v3_loss', 0)
        hvp_v3_stage = self.loss_dict.get('hvp_v3_stage', "")
        hvp_v3_stage1_loss = self.loss_dict.get('hvp_v3_stage1_hypothesis_loss', 0)
        hvp_v3_stage2_loss = self.loss_dict.get('hvp_v3_stage2_evidence_loss', 0)
        hvp_v3_stage2_heatmap_loss = self.loss_dict.get(
            'hvp_v3_stage2_evidence_heatmap_loss',
            0,
        )
        hvp_v3_stage2_uncertainty_loss = self.loss_dict.get(
            'hvp_v3_stage2_uncertainty_loss',
            0,
        )
        hvp_v3_stage2_descriptor_loss = self.loss_dict.get(
            'hvp_v3_stage2_descriptor_loss',
            0,
        )
        pact_cbea_local_evidence_enabled = self.loss_dict.get(
            'pact_cbea_local_evidence_enabled',
            False,
        )
        pact_cbea_loss = self.loss_dict.get('pact_cbea_loss', 0)
        pact_cbea_local_evidence_loss = self.loss_dict.get(
            'pact_cbea_local_evidence_loss',
            0,
        )
        pact_cbea_heatmap_loss = self.loss_dict.get(
            'pact_cbea_evidence_heatmap_loss',
            0,
        )
        pact_cbea_uncertainty_loss = self.loss_dict.get(
            'pact_cbea_uncertainty_loss',
            0,
        )
        pact_cbea_descriptor_loss = self.loss_dict.get(
            'pact_cbea_descriptor_loss',
            0,
        )

        log_msg = ("[epoch %d][%d/%d]%s || Loss: %.4f || Conf Loss: %.4f"
                   " || Loc Loss: %.4f || Dir Loss: %.4f || IoU Loss: %.4f"
                   " || Depth Loss: %.4f || Pyramid Loss: %.4f" % (
                       epoch, batch_id + 1, batch_len, suffix,
                       total_loss, cls_loss, reg_loss, dir_loss, iou_loss,
                       depth_loss, pyramid_loss))
        if hvp_v3_enabled:
            if hvp_v3_stage == "stage2_evidence":
                log_msg += (
                    " || HVP-v3 Loss: %.3e || Stage2 Evidence Loss: %.3e"
                    " || Evidence Heatmap Loss: %.3e || Evidence Unc Loss: %.3e"
                    " || Evidence Desc Loss: %.3e"
                ) % (
                    hvp_v3_loss,
                    hvp_v3_stage2_loss,
                    hvp_v3_stage2_heatmap_loss,
                    hvp_v3_stage2_uncertainty_loss,
                    hvp_v3_stage2_descriptor_loss,
                )
            else:
                log_msg += " || HVP-v3 Loss: %.3e || Stage1 Hypothesis Loss: %.3e" % (
                    hvp_v3_loss,
                    hvp_v3_stage1_loss,
                )
        if pact_cbea_local_evidence_enabled:
            log_msg += (
                " || PACT-CBEA Loss: %.3e || Local Evidence Loss: %.3e"
                " || PACT Evidence Heatmap Loss: %.3e"
                " || PACT Evidence Unc Loss: %.3e"
                " || PACT Evidence Desc Loss: %.3e"
            ) % (
                pact_cbea_loss,
                pact_cbea_local_evidence_loss,
                pact_cbea_heatmap_loss,
                pact_cbea_uncertainty_loss,
                pact_cbea_descriptor_loss,
            )
        print(log_msg)

        if not writer is None:
            writer.add_scalar('Regression_loss' + suffix, reg_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Confidence_loss' + suffix, cls_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Dir_loss' + suffix, dir_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Iou_loss' + suffix, iou_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Depth_loss' + suffix, depth_loss,
                            epoch*batch_len + batch_id)
            writer.add_scalar('Pyramid_loss' + suffix, pyramid_loss,
                epoch*batch_len + batch_id)
            if hvp_v3_enabled:
                writer.add_scalar('HVP_v3_loss' + suffix, hvp_v3_loss,
                                epoch*batch_len + batch_id)
                if hvp_v3_stage == "stage2_evidence":
                    writer.add_scalar('Stage2_evidence_loss' + suffix,
                                    hvp_v3_stage2_loss,
                                    epoch*batch_len + batch_id)
                    writer.add_scalar('Stage2_evidence_heatmap_loss' + suffix,
                                    hvp_v3_stage2_heatmap_loss,
                                    epoch*batch_len + batch_id)
                    writer.add_scalar('Stage2_evidence_uncertainty_loss' + suffix,
                                    hvp_v3_stage2_uncertainty_loss,
                                    epoch*batch_len + batch_id)
                    writer.add_scalar('Stage2_evidence_descriptor_loss' + suffix,
                                    hvp_v3_stage2_descriptor_loss,
                                    epoch*batch_len + batch_id)
                else:
                    writer.add_scalar('Stage1_hypothesis_loss' + suffix,
                                    hvp_v3_stage1_loss,
                                    epoch*batch_len + batch_id)
            if pact_cbea_local_evidence_enabled:
                writer.add_scalar('PACT_CBEA_loss' + suffix, pact_cbea_loss,
                                epoch*batch_len + batch_id)
                writer.add_scalar('PACT_CBEA_local_evidence_loss' + suffix,
                                pact_cbea_local_evidence_loss,
                                epoch*batch_len + batch_id)
                writer.add_scalar('PACT_CBEA_evidence_heatmap_loss' + suffix,
                                pact_cbea_heatmap_loss,
                                epoch*batch_len + batch_id)
                writer.add_scalar('PACT_CBEA_evidence_uncertainty_loss' + suffix,
                                pact_cbea_uncertainty_loss,
                                epoch*batch_len + batch_id)
                writer.add_scalar('PACT_CBEA_evidence_descriptor_loss' + suffix,
                                pact_cbea_descriptor_loss,
                                epoch*batch_len + batch_id)


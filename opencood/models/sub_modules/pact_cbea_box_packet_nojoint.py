"""Parameter-free sparse box packets for PACT no-joint inference."""

import math

import numpy as np
import torch
import torch.nn as nn

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils import box_utils
from opencood.utils.common_utils import limit_period


PACKET_SOURCE = "local_detector_boxes_after_nms"
MODALITY_IDS = {"m1": 1, "m2": 2, "m3": 3, "m4": 4}


class PACTNoJointBoxPacketCodec(nn.Module):
    """Decode, transform, and fuse sparse local detector boxes."""

    def __init__(self, score_threshold=0.2, local_nms_thresh=0.15,
                 global_nms_thresh=0.15, max_boxes_per_agent=100,
                 order="hwl", dir_offset=0.7853, num_bins=2,
                 gt_range=None, quantize="fp16"):
        super().__init__()
        self.score_threshold = float(score_threshold)
        self.local_nms_thresh = float(local_nms_thresh)
        self.global_nms_thresh = float(global_nms_thresh)
        self.max_boxes_per_agent = int(max_boxes_per_agent)
        self.order = str(order)
        self.dir_offset = float(dir_offset)
        self.num_bins = int(num_bins)
        self.gt_range = list(
            gt_range or [-102.4, -102.4, -3.0, 102.4, 102.4, 1.0]
        )
        self.quantize = str(quantize).lower()
        if self.quantize != "fp16":
            raise ValueError("PACT box packet v1 supports only quantize=fp16")
        if self.order != "hwl":
            raise ValueError("PACT box packet v1 requires box order hwl")
        if self.num_bins < 1 or self.max_boxes_per_agent < 1:
            raise ValueError("num_bins and max_boxes_per_agent must be positive")

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def decode_local_predictions(self, cls_preds, reg_preds, dir_preds, anchor_box):
        """Apply the official anchor decode and local filtering sequence."""
        reference = reg_preds if torch.is_tensor(reg_preds) else cls_preds
        if not torch.is_tensor(reference):
            raise TypeError("local detector predictions must be torch tensors")
        if reference.numel() == 0:
            return self.empty_prediction(reference)
        if cls_preds.ndim != 4 or reg_preds.ndim != 4 or cls_preds.shape[0] != 1:
            raise ValueError("local box decode expects batch-one dense predictions")
        if reg_preds.shape[0] != 1:
            raise ValueError("local box decode expects batch-one regression")

        anchor_box = torch.as_tensor(
            anchor_box, device=reg_preds.device, dtype=reg_preds.dtype
        )
        if anchor_box.numel() == 0:
            return self.empty_prediction(reg_preds)

        probability = torch.sigmoid(
            cls_preds.permute(0, 2, 3, 1)
        ).reshape(1, -1)
        batch_boxes = VoxelPostprocessor.delta_to_boxes3d(reg_preds, anchor_box)
        if probability.shape[1] != batch_boxes.shape[1]:
            raise ValueError("classification and anchor box counts do not match")

        score_mask = probability > self.score_threshold
        boxes_center = batch_boxes[0][score_mask[0]].reshape(-1, 7)
        scores = probability[0][score_mask[0]].reshape(-1)
        if boxes_center.shape[0] == 0:
            return self.empty_prediction(reg_preds)

        if dir_preds is not None:
            if not torch.is_tensor(dir_preds) or dir_preds.ndim != 4:
                raise ValueError("direction predictions must be [1,A*num_bins,H,W]")
            direction = dir_preds.permute(0, 2, 3, 1).contiguous()
            direction = direction.reshape(1, -1, self.num_bins)[score_mask]
            direction_labels = torch.argmax(direction, dim=-1)
            period = 2.0 * math.pi / float(self.num_bins)
            direction_rotation = limit_period(
                boxes_center[..., 6] - self.dir_offset, 0.0, period
            )
            boxes_center[..., 6] = (
                direction_rotation
                + self.dir_offset
                + period * direction_labels.to(direction.dtype)
            )
            boxes_center[..., 6] = limit_period(
                boxes_center[..., 6], 0.5, 2.0 * math.pi
            )

        boxes_corner = box_utils.boxes_to_corners_3d(
            boxes_center, order=self.order
        )
        boxes_corner, scores, boxes_center = self._filter_abnormal(
            boxes_corner, scores, boxes_center
        )
        if boxes_corner.shape[0] == 0:
            return self.empty_prediction(reg_preds)

        keep = self._torch_indices(
            box_utils.nms_rotated(
                boxes_corner, scores, self.local_nms_thresh
            ),
            boxes_corner.device,
        )
        boxes_corner = boxes_corner[keep]
        scores = scores[keep]
        boxes_center = boxes_center[keep]
        score_order = torch.argsort(scores, descending=True)[
            :self.max_boxes_per_agent
        ]
        return self._detach_prediction(
            boxes_corner[score_order],
            scores[score_order],
            boxes_center[score_order],
        )

    def build_packet(self, prediction, modality_name, agent_id, transmitted=True):
        if modality_name not in MODALITY_IDS:
            raise ValueError("unknown PACT modality: %s" % modality_name)
        boxes_corner = prediction["boxes_corner"].detach()
        scores = prediction["scores"].detach()
        if transmitted:
            boxes_corner = self._fp16_roundtrip(boxes_corner)
            scores = self._fp16_roundtrip(scores)
        count = int(scores.shape[0])
        scalar_dtype = boxes_corner.dtype
        return {
            "boxes_corner": boxes_corner,
            "scores": scores,
            "modality_id": boxes_corner.new_full(
                (count,), float(MODALITY_IDS[modality_name]),
                dtype=scalar_dtype,
            ).detach(),
            "agent_id": boxes_corner.new_full(
                (count,), float(agent_id), dtype=scalar_dtype
            ).detach(),
            "packet_source": PACKET_SOURCE,
        }

    @staticmethod
    def transform_packet_to_ego(boxes_corner, agent_to_ego):
        if boxes_corner.ndim != 3 or boxes_corner.shape[1:] != (8, 3):
            raise ValueError("box packet corners must be [N,8,3]")
        if boxes_corner.shape[0] == 0:
            return boxes_corner.detach()
        transform = torch.as_tensor(
            agent_to_ego,
            device=boxes_corner.device,
            dtype=boxes_corner.dtype,
        )
        if transform.shape != (4, 4):
            raise ValueError("agent_to_ego must be a raw [4,4] transform")
        return box_utils.project_box3d(boxes_corner, transform).detach()

    def fuse_packets(self, ego_packet, collaborator_packets):
        packets = [ego_packet] + list(collaborator_packets)
        corner_parts = [
            packet["boxes_corner"] for packet in packets
            if packet["boxes_corner"].shape[0] > 0
        ]
        score_parts = [
            packet["scores"] for packet in packets
            if packet["scores"].shape[0] > 0
        ]
        if not corner_parts:
            reference = ego_packet["boxes_corner"]
            return {
                "boxes_corner": reference.new_empty((0, 8, 3)).detach(),
                "scores": reference.new_empty((0,)).detach(),
            }

        boxes_corner = torch.cat(corner_parts, dim=0)
        scores = torch.cat(score_parts, dim=0)
        boxes_corner, scores, _ = self._filter_abnormal(
            boxes_corner, scores
        )
        if boxes_corner.shape[0] == 0:
            return {
                "boxes_corner": boxes_corner.detach(),
                "scores": scores.detach(),
            }

        keep = self._torch_indices(
            box_utils.nms_rotated(
                boxes_corner, scores, self.global_nms_thresh
            ),
            boxes_corner.device,
        )
        boxes_corner = boxes_corner[keep]
        scores = scores[keep]

        boxes_numpy = boxes_corner.detach().cpu().numpy()
        boxes_numpy, range_mask = box_utils.mask_boxes_outside_range_numpy(
            boxes_numpy,
            self.gt_range,
            order=None,
            return_mask=True,
        )
        boxes_corner = torch.from_numpy(boxes_numpy).to(
            device=scores.device, dtype=boxes_corner.dtype
        )
        range_mask = torch.as_tensor(
            range_mask, device=scores.device, dtype=torch.bool
        )
        scores = scores[range_mask]
        score_order = torch.argsort(scores, descending=True)
        return {
            "boxes_corner": boxes_corner[score_order].detach(),
            "scores": scores[score_order].detach(),
        }

    @staticmethod
    def empty_prediction(reference):
        return {
            "boxes_corner": reference.new_empty((0, 8, 3)).detach(),
            "scores": reference.new_empty((0,)).detach(),
            "boxes_center": reference.new_empty((0, 7)).detach(),
        }

    @staticmethod
    def _detach_prediction(boxes_corner, scores, boxes_center):
        return {
            "boxes_corner": boxes_corner.detach(),
            "scores": scores.detach(),
            "boxes_center": boxes_center.detach(),
        }

    @staticmethod
    def _filter_abnormal(boxes_corner, scores, boxes_center=None):
        if boxes_corner.shape[0] == 0:
            return boxes_corner, scores, boxes_center
        keep_large = box_utils.remove_large_pred_bbx(boxes_corner)
        keep_z = box_utils.remove_bbx_abnormal_z(boxes_corner)
        keep = torch.logical_and(keep_large, keep_z)
        filtered_center = boxes_center[keep] if boxes_center is not None else None
        return boxes_corner[keep], scores[keep], filtered_center

    @staticmethod
    def _torch_indices(indices, device):
        if isinstance(indices, np.ndarray):
            return torch.from_numpy(indices).to(
                device=device, dtype=torch.long
            )
        return torch.as_tensor(indices, device=device, dtype=torch.long)

    @staticmethod
    def _fp16_roundtrip(tensor):
        return tensor.to(dtype=torch.float16).to(dtype=tensor.dtype).detach()


class PACTNoJointBoxCommunicationMeter(nn.Module):
    """Count collaborator-only fp16 box packet communication."""

    SCALARS_PER_BOX = 27

    def __init__(self, quantize="fp16", bytes_per_scalar=2,
                 deadline_ms=100, bandwidth_budget_kb=64):
        super().__init__()
        self.quantize = str(quantize).lower()
        self.bytes_per_scalar = int(bytes_per_scalar)
        self.deadline_ms = float(deadline_ms)
        self.bandwidth_budget_kb = float(bandwidth_budget_kb)
        if self.quantize != "fp16" or self.bytes_per_scalar != 2:
            raise ValueError(
                "PACT box packet v1 supports fp16 with 2 bytes per scalar only"
            )

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, collaborator_packets):
        box_count = sum(
            int(packet["scores"].shape[0])
            for packet in collaborator_packets
        )
        bytes_per_box = self.SCALARS_PER_BOX * self.bytes_per_scalar
        byte_count = box_count * bytes_per_box
        seconds = max(self.deadline_ms, 1e-6) / 1000.0
        return {
            "collaborator_box_count": box_count,
            "packet_bytes_per_frame": byte_count,
            "packet_kb_per_frame": byte_count / 1024.0,
            "estimated_mbps": byte_count * 8.0 / seconds / 1e6,
            "bytes_per_box": bytes_per_box,
            "bandwidth_budget_kb": self.bandwidth_budget_kb,
            "bandwidth_saturated": bool(
                byte_count > self.bandwidth_budget_kb * 1024.0
            ),
        }

"""HBEC hypothesis/evidence matching."""

import math

import numpy as np
import torch


class HypothesisMatcher:
    def __init__(self, match_cfg):
        self.cfg = match_cfg or {}
        self.last_fallback_reason = ""

    def match(self, ego_boxes, evidence_boxes):
        self.last_fallback_reason = ""
        if ego_boxes is None or evidence_boxes is None or len(ego_boxes) == 0 or len(evidence_boxes) == 0:
            return []

        distances = self._center_distances(ego_boxes, evidence_boxes)
        ious = self._bev_ious(ego_boxes, evidence_boxes)
        if ious is None:
            ious = torch.zeros_like(distances)
            self.last_fallback_reason = "iou_unavailable_center_distance_only"

        dist_scale = max(float(self.cfg.get("dist_scale", 2.0)), 1e-6)
        iou_weight = float(self.cfg.get("iou_weight", 0.7))
        dist_weight = float(self.cfg.get("dist_weight", 0.3))
        match_score = iou_weight * ious + dist_weight * torch.exp(-distances / dist_scale)

        iou_threshold = float(self.cfg.get("iou_threshold", 0.1))
        dist_threshold = float(self.cfg.get("center_dist_threshold", 2.0))
        valid = torch.logical_or(ious >= iou_threshold, distances <= dist_threshold)

        pairs = []
        for ego_idx in range(match_score.shape[0]):
            for ev_idx in range(match_score.shape[1]):
                if valid[ego_idx, ev_idx]:
                    pairs.append((
                        float(match_score[ego_idx, ev_idx].detach().cpu()),
                        ego_idx,
                        ev_idx,
                        float(ious[ego_idx, ev_idx].detach().cpu()),
                        float(distances[ego_idx, ev_idx].detach().cpu()),
                    ))
        pairs.sort(reverse=True, key=lambda item: item[0])

        used_ego = set()
        used_ev = set()
        matches = []
        for score, ego_idx, ev_idx, iou, distance in pairs:
            if ego_idx in used_ego or ev_idx in used_ev:
                continue
            used_ego.add(ego_idx)
            used_ev.add(ev_idx)
            matches.append({
                "ego_idx": ego_idx,
                "evidence_idx": ev_idx,
                "match_score": score,
                "iou": iou,
                "distance": distance,
            })
        return matches

    @staticmethod
    def _center_distances(a_boxes, b_boxes):
        a_center = a_boxes[:, :, :2].mean(dim=1)
        b_center = b_boxes[:, :, :2].mean(dim=1)
        return torch.cdist(a_center, b_center)

    @staticmethod
    def _bev_ious(a_boxes, b_boxes):
        try:
            from opencood.utils import common_utils

            a_polys = common_utils.convert_format(a_boxes.detach().cpu().numpy())
            b_polys = common_utils.convert_format(b_boxes.detach().cpu().numpy())
            matrix = np.zeros((len(a_polys), len(b_polys)), dtype=np.float32)
            for idx, poly in enumerate(a_polys):
                matrix[idx] = common_utils.compute_iou(poly, b_polys)
            return torch.from_numpy(matrix).to(device=a_boxes.device, dtype=a_boxes.dtype)
        except Exception:
            return None

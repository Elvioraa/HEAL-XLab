"""Bayesian box refinement for HBEC."""

import torch

from opencood.xlab.utils import clamp_probability, weighted_yaw_average


class BayesianRefiner:
    def __init__(self, hbec_cfg):
        self.cfg = hbec_cfg or {}
        self.order = self.cfg.get("box_order", "lwh")

    def refine(self, ego_packet, evidence_packet, matches):
        boxes = ego_packet.boxes.clone()
        scores = ego_packet.scores.clone()
        if not matches:
            return boxes, scores, 0

        refine_cfg = self.cfg.get("refine", {})
        strength = float(refine_cfg.get("refine_strength", 0.5))
        evidence_weight = float(refine_cfg.get("evidence_weight", 0.5))
        if not refine_cfg.get("enabled", True) or strength <= 0:
            return boxes, scores, 0

        from opencood.utils import box_utils

        ego_centers = box_utils.corner_to_center_torch(ego_packet.boxes, order=self.order)
        ev_centers = box_utils.corner_to_center_torch(evidence_packet.boxes, order=self.order)

        refined_count = 0
        for match in matches:
            ego_idx = match["ego_idx"]
            ev_idx = match["evidence_idx"]
            ego_unc = self._scalar_uncertainty(ego_packet.uncertainty, ego_idx)
            ev_unc = self._scalar_uncertainty(evidence_packet.uncertainty, ev_idx)
            ego_w = 1.0 / torch.clamp(ego_unc ** 2, min=1e-6)
            ev_w = evidence_weight / torch.clamp(ev_unc ** 2, min=1e-6)

            bayes = ego_centers[ego_idx].clone()
            bayes[:6] = (ego_centers[ego_idx, :6] * ego_w + ev_centers[ev_idx, :6] * ev_w) / (ego_w + ev_w)
            bayes[6] = weighted_yaw_average(
                torch.stack([ego_centers[ego_idx, 6], ev_centers[ev_idx, 6]]),
                torch.stack([ego_w, ev_w]),
            )

            blended = ego_centers[ego_idx].clone()
            blended[:6] = (1.0 - strength) * ego_centers[ego_idx, :6] + strength * bayes[:6]
            blended[6] = weighted_yaw_average(
                torch.stack([ego_centers[ego_idx, 6], bayes[6]]),
                torch.stack([ego_w.new_tensor(1.0 - strength), ego_w.new_tensor(strength)]),
            )
            boxes[ego_idx] = box_utils.boxes_to_corners_3d(blended.view(1, 7), order=self.order)[0]

            ego_logit = torch.logit(clamp_probability(scores[ego_idx]))
            ev_logit = torch.logit(clamp_probability(evidence_packet.scores[ev_idx]))
            scores[ego_idx] = torch.sigmoid(ego_logit + evidence_weight * ev_logit)
            refined_count += 1

        return boxes, scores, refined_count

    def insert_novel(self, boxes, scores, ego_packet, evidence_packet, matches):
        novel_cfg = self.cfg.get("novel", {})
        if not novel_cfg.get("enabled", True) or evidence_packet.boxes is None or len(evidence_packet.boxes) == 0:
            return boxes, scores, 0

        matched_evidence = {match["evidence_idx"] for match in matches}
        threshold = float(novel_cfg.get("novel_score_threshold", 0.6))
        dist_threshold = float(novel_cfg.get("novel_dist_threshold", 2.0))
        max_novel = int(novel_cfg.get("max_novel", 20))

        ego_centers = boxes[:, :, :2].mean(dim=1) if len(boxes) else boxes.new_zeros((0, 2))
        ev_centers = evidence_packet.boxes[:, :, :2].mean(dim=1)
        additions = []
        add_scores = []
        for idx in range(len(evidence_packet.boxes)):
            if idx in matched_evidence or evidence_packet.scores[idx] < threshold:
                continue
            min_dist = torch.tensor(float("inf"), device=evidence_packet.boxes.device)
            if len(ego_centers):
                min_dist = torch.min(torch.norm(ego_centers - ev_centers[idx], dim=1))
            if min_dist > dist_threshold:
                additions.append(evidence_packet.boxes[idx])
                add_scores.append(evidence_packet.scores[idx])
            if len(additions) >= max_novel:
                break

        if not additions:
            return boxes, scores, 0
        return torch.cat([boxes, torch.stack(additions)], dim=0), torch.cat([scores, torch.stack(add_scores)], dim=0), len(additions)

    def suppress(self, scores, matches):
        suppress_cfg = self.cfg.get("suppress", {})
        if not suppress_cfg.get("enabled", False):
            return scores, 0
        threshold = float(suppress_cfg.get("suppress_score_threshold", 0.3))
        factor = float(suppress_cfg.get("suppress_factor", 1.0))
        matched_ego = {match["ego_idx"] for match in matches}
        count = 0
        out = scores.clone()
        for idx in range(len(out)):
            if idx not in matched_ego and out[idx] < threshold:
                out[idx] = out[idx] * factor
                count += 1
        return out, count

    @staticmethod
    def _scalar_uncertainty(uncertainty, idx):
        if uncertainty is None:
            return torch.tensor(1.0)
        value = uncertainty[idx]
        if value.ndim > 0:
            value = value.mean()
        return value

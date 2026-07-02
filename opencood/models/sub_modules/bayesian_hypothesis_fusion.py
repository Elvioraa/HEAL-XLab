"""Bayesian hypothesis fusion for HVP-CBEA."""

import math

import torch
import torch.nn as nn


class BayesianHypothesisFusion(nn.Module):
    """Scatter updated hypotheses into BEV feature space with gated fusion."""

    def __init__(self, in_channels=256, mid_channels=64, pc_range=None,
                 confirm_boost=1.5, refute_penalty=-2.5, refine_boost=0.8,
                 residual_gate=None):
        super().__init__()
        self.pc_range = pc_range or [-102.4, -102.4, -3.0, 102.4, 102.4, 1.0]
        self.confirm_boost = float(confirm_boost)
        self.refute_penalty = float(refute_penalty)
        self.refine_boost = float(refine_boost)
        if isinstance(residual_gate, bool):
            residual_gate = {"enabled": residual_gate}
        else:
            residual_gate = residual_gate or {}
        self.residual_gate_enabled = bool(residual_gate.get("enabled", True))
        self.residual_alpha_max = max(float(residual_gate.get("alpha_max", 0.3)), 1e-6)
        alpha_init = float(residual_gate.get("alpha_init", 0.05))
        alpha_init = min(max(alpha_init, 1e-6), self.residual_alpha_max * (1.0 - 1e-6))
        self.residual_alpha_learnable = bool(residual_gate.get("learnable", True))
        alpha_ratio = alpha_init / self.residual_alpha_max
        alpha_logit = math.log(alpha_ratio / (1.0 - alpha_ratio))
        alpha_logit = torch.tensor(alpha_logit, dtype=torch.float32)
        if self.residual_alpha_learnable:
            self.residual_alpha_logit = nn.Parameter(alpha_logit)
        else:
            self.register_buffer("residual_alpha_logit", alpha_logit)
        self.hyp_proj = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.last_residual_debug = self._make_residual_debug()

    def forward(self, ego_hyps, verif_logits, refine_delta, novel_hyps, ego_bev_feat):
        if ego_bev_feat is None or ego_bev_feat.ndim != 4:
            self.last_residual_debug = self._make_residual_debug(fallback_reason="invalid_ego_bev_feat")
            return ego_bev_feat, ego_hyps
        if ego_hyps is None or ego_hyps.ndim != 3 or ego_hyps.shape[1] == 0:
            self.last_residual_debug = self._make_residual_debug(fallback_reason="no_ego_hypotheses")
            return ego_bev_feat, ego_hyps
        try:
            updated_hyps = self._update_hypotheses(ego_hyps, verif_logits, refine_delta)
            hyp_map = self._scatter(updated_hyps, novel_hyps, ego_bev_feat)
            hyp_feat = self.hyp_proj(hyp_map)
            gate = self.gate(torch.cat([ego_bev_feat, hyp_feat], dim=1))
            candidate = ego_bev_feat * (1.0 - gate) + hyp_feat * gate
            delta_feature = candidate - ego_bev_feat
            fused = self.apply_residual_delta(ego_bev_feat, delta_feature)
            return fused, updated_hyps
        except Exception:
            self.last_residual_debug = self._make_residual_debug(fallback_reason="exception")
            return ego_bev_feat, ego_hyps

    def compute_loss(self, *args, **kwargs):
        ref = self._find_tensor(args, kwargs)
        if ref is None:
            return torch.tensor(0.0)
        return ref.sum() * 0.0

    def _update_hypotheses(self, ego_hyps, verif_logits, refine_delta):
        if verif_logits is None or refine_delta is None:
            return ego_hyps
        probs = torch.softmax(verif_logits, dim=-1)
        base_score = torch.clamp(ego_hyps[..., 7:8], 1e-4, 1.0 - 1e-4)
        log_odds = torch.logit(base_score)
        log_odds = log_odds + probs[..., 0:1] * self.confirm_boost
        log_odds = log_odds + probs[..., 1:2] * self.refute_penalty
        log_odds = log_odds + probs[..., 2:3] * self.refine_boost
        updated_score = torch.sigmoid(log_odds)
        updated_box = ego_hyps[..., :7] + refine_delta[..., :7] * probs[..., 2:3]
        updated_valid = (updated_score > 0.05).to(ego_hyps.dtype)
        return torch.cat([updated_box, updated_score, updated_valid], dim=-1)

    def _scatter(self, hyps, novel_hyps, ref_feat):
        bsz, _, height, width = ref_feat.shape
        all_hyps = hyps
        if novel_hyps is not None and torch.is_tensor(novel_hyps) and novel_hyps.numel() > 0:
            all_hyps = torch.cat([hyps, novel_hyps.to(device=hyps.device, dtype=hyps.dtype)], dim=1)
        x_min, y_min, _, x_max, y_max, _ = self.pc_range[:6]
        x = ((all_hyps[..., 0] - x_min) / max(x_max - x_min, 1e-6) * width).long()
        y = ((all_hyps[..., 1] - y_min) / max(y_max - y_min, 1e-6) * height).long()
        valid = (all_hyps[..., 8] > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        rows = []
        for bidx in range(bsz):
            flat = ref_feat.new_zeros((height * width,))
            if not torch.any(valid[bidx]):
                rows.append(flat)
                continue
            flat_idx = y[bidx][valid[bidx]] * width + x[bidx][valid[bidx]]
            values = all_hyps[bidx, valid[bidx], 7]
            rows.append(flat.scatter_add(0, flat_idx, values))
        return torch.stack(rows, dim=0).view(bsz, 1, height, width)

    def get_residual_debug(self):
        return dict(self.last_residual_debug)

    def apply_residual_delta(self, ego_bev_feat, delta_feature):
        alpha = self._residual_alpha().to(device=ego_bev_feat.device, dtype=ego_bev_feat.dtype)
        if self.residual_gate_enabled:
            fused = ego_bev_feat + alpha * delta_feature
        else:
            fused = ego_bev_feat + delta_feature
        self.last_residual_debug = self._make_residual_debug(alpha, delta_feature)
        return fused

    def _residual_alpha(self):
        if not self.residual_gate_enabled:
            return self.residual_alpha_logit.new_tensor(1.0)
        return self.residual_alpha_max * torch.sigmoid(self.residual_alpha_logit)

    def _make_residual_debug(self, alpha=None, delta_feature=None, fallback_reason=""):
        if alpha is None:
            alpha = self._residual_alpha()
        debug = {
            "hvp_cbea_residual_gate_enabled": bool(self.residual_gate_enabled),
            "hvp_cbea_residual_alpha": float(alpha.detach().cpu()),
            "hvp_cbea_residual_alpha_max": float(self.residual_alpha_max),
            "hvp_cbea_residual_alpha_learnable": bool(self.residual_alpha_learnable),
            "hvp_cbea_delta_norm": 0.0,
            "hvp_cbea_delta_mean": 0.0,
            "hvp_cbea_delta_std": 0.0,
            "hvp_cbea_residual_fallback_reason": fallback_reason,
        }
        if torch.is_tensor(delta_feature) and delta_feature.numel() > 0:
            detached = delta_feature.detach().float()
            debug["hvp_cbea_delta_norm"] = float(detached.norm().cpu())
            debug["hvp_cbea_delta_mean"] = float(detached.mean().cpu())
            debug["hvp_cbea_delta_std"] = float(detached.std(unbiased=False).cpu())
        return debug

    @staticmethod
    def _find_tensor(args, kwargs):
        for item in list(args) + list(kwargs.values()):
            if torch.is_tensor(item):
                return item
        return None

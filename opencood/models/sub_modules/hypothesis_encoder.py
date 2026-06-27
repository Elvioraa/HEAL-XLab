"""Hypothesis encoder for HEAL-XLab HVP-CBEA."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypothesisEncoder(nn.Module):
    """Generate sparse object hypotheses from BEV features."""

    def __init__(self, in_channels=256, mid_channels=64, max_hypotheses=50,
                 hyp_conf_threshold=0.15, pc_range=None):
        super().__init__()
        self.max_hypotheses = int(max_hypotheses)
        self.hyp_conf_threshold = float(hyp_conf_threshold)
        self.pc_range = pc_range or [-102.4, -102.4, -3.0, 102.4, 102.4, 1.0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.hmap_head = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.reg_head = nn.Conv2d(mid_channels, 7, kernel_size=1)

    def forward(self, bev_feat):
        if bev_feat is None or bev_feat.ndim != 4 or bev_feat.shape[0] == 0:
            return self._empty(bev_feat), None, None
        feat = self.stem(bev_feat)
        hmap = torch.sigmoid(self.hmap_head(feat))
        reg = self.reg_head(feat)
        hyps = self._decode(hmap, reg)
        return hyps, hmap, reg

    def compute_loss(self, *args, **kwargs):
        ref = self._find_tensor(args, kwargs)
        if ref is None:
            return torch.tensor(0.0)
        return ref.sum() * 0.0

    def _decode(self, hmap, reg):
        bsz, _, height, width = hmap.shape
        k = min(self.max_hypotheses, height * width)
        scores, inds = torch.topk(hmap.view(bsz, -1), k=k, dim=1)
        valid = scores >= self.hyp_conf_threshold

        ys = torch.div(inds, width, rounding_mode="floor")
        xs = inds % width
        gather_inds = inds.unsqueeze(1).expand(-1, reg.shape[1], -1)
        reg_flat = reg.view(bsz, reg.shape[1], -1)
        reg_vals = torch.gather(reg_flat, 2, gather_inds).transpose(1, 2)

        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range[:6]
        x_scale = (x_max - x_min) / max(width, 1)
        y_scale = (y_max - y_min) / max(height, 1)
        x = x_min + (xs.to(reg.dtype) + 0.5 + torch.tanh(reg_vals[..., 0]) * 0.5) * x_scale
        y = y_min + (ys.to(reg.dtype) + 0.5 + torch.tanh(reg_vals[..., 1]) * 0.5) * y_scale
        z = torch.tanh(reg_vals[..., 2]) * max((z_max - z_min) * 0.5, 1e-3)
        w = F.softplus(reg_vals[..., 3]) + 1.0
        length = F.softplus(reg_vals[..., 4]) + 1.0
        h = F.softplus(reg_vals[..., 5]) + 1.0
        yaw = torch.atan2(torch.sin(reg_vals[..., 6]), torch.cos(reg_vals[..., 6]))

        return torch.stack(
            [x, y, z, w, length, h, yaw, scores, valid.to(reg.dtype)],
            dim=-1,
        )

    def _empty(self, ref):
        device = ref.device if torch.is_tensor(ref) else torch.device("cpu")
        dtype = ref.dtype if torch.is_tensor(ref) else torch.float32
        batch = ref.shape[0] if torch.is_tensor(ref) and ref.ndim > 0 else 0
        return torch.zeros((batch, 0, 9), device=device, dtype=dtype)

    @staticmethod
    def _find_tensor(args, kwargs):
        for item in list(args) + list(kwargs.values()):
            if torch.is_tensor(item):
                return item
        return None


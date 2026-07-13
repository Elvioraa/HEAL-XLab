"""BGER prior-conditioned feature reactivation module.

The only trainable component of the BGER line. It takes the ego BEV feature
and the rendered box prior maps, and produces a residual update so the ego
detector can re-examine ("reactivate") weak evidence in regions highlighted
by collaborator boxes.

The final projection conv is zero-initialized (ControlNet-style zero conv),
so at initialization the module is an exact identity mapping over the ego
feature; an untrained BGER model therefore behaves like the ego-only
baseline.
"""

import torch
import torch.nn as nn


def _make_norm(norm_type, channels):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    if norm_type == "gn":
        return nn.GroupNorm(min(32, channels), channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError("unsupported bger refine norm type: %s" % norm_type)


class BGERRefine(nn.Module):
    def __init__(self, args):
        super(BGERRefine, self).__init__()
        in_channels = int(args["in_channels"])
        prior_channels = int(args["prior_channels"])
        hidden_dim = int(args.get("hidden_dim", 128))
        num_layers = int(args.get("num_layers", 2))
        norm_type = str(args.get("norm", "bn"))
        gate_init = float(args.get("gate_init", 1.0))
        if num_layers < 1:
            raise ValueError("bger refine num_layers must be >= 1")

        layers = []
        current = in_channels + prior_channels
        for _ in range(num_layers):
            layers.append(nn.Conv2d(current, hidden_dim, kernel_size=3, padding=1))
            layers.append(_make_norm(norm_type, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current = hidden_dim
        self.body = nn.Sequential(*layers)

        self.out_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        self.gate = nn.Parameter(torch.full((1,), gate_init))

    def forward(self, ego_feature, prior_map):
        """
        Parameters
        ----------
        ego_feature : torch.Tensor
            (B, C, H, W) ego BEV feature.
        prior_map : torch.Tensor
            (B, P, H, W) rendered box prior.

        Returns
        -------
        refined : torch.Tensor
            (B, C, H, W) reactivated ego feature.
        delta : torch.Tensor
            (B, C, H, W) residual update (before gating), for debugging.
        """
        hidden = self.body(torch.cat([ego_feature, prior_map], dim=1))
        delta = self.out_conv(hidden)
        refined = ego_feature + self.gate.view(1, -1, 1, 1) * delta
        return refined, delta

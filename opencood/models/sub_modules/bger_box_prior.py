"""BGER box-to-BEV prior map renderer.

Renders collaborator detection boxes (already projected to the ego frame)
into dense BEV prior maps aligned with the ego feature grid. The renderer is
parameter-free; it only performs tensor rasterization and is always executed
under no_grad by the caller (boxes are messages, not gradient paths).

Box format follows the repo convention used by ``delta_to_boxes3d`` and
``boxes_to_corners_3d`` with order 'hwl': [x, y, z, h, w, l, yaw].
"""

import math

import torch
import torch.nn as nn


class BGERBoxPrior(nn.Module):
    """Parameter-free rasterizer from ego-frame boxes to BEV prior maps.

    Channels (in order, controlled by config flags):
      0. gaussian  - confidence-weighted gaussian bump at each box center,
                     sigma proportional to box footprint (max over boxes)
      1. box_mask  - confidence inside the rotated box footprint
                     (max over boxes)
      2. yaw_cos   - cos(yaw) of the highest-confidence covering box (optional)
      3. yaw_sin   - sin(yaw) of the highest-confidence covering box (optional)
    """

    def __init__(self, args):
        super(BGERBoxPrior, self).__init__()
        self.lidar_range = list(args["lidar_range"])
        self.use_gaussian = bool(args.get("gaussian", True))
        self.use_box_mask = bool(args.get("box_mask", True))
        self.use_yaw = bool(args.get("yaw", False))
        self.sigma_scale = float(args.get("sigma_scale", 0.25))
        self.min_sigma = float(args.get("min_sigma", 1.0))
        if not (self.use_gaussian or self.use_box_mask or self.use_yaw):
            raise ValueError("BGERBoxPrior needs at least one channel enabled")
        self._grid_key = None
        self._grid_x = None
        self._grid_y = None

    @property
    def num_channels(self):
        channels = 0
        if self.use_gaussian:
            channels += 1
        if self.use_box_mask:
            channels += 1
        if self.use_yaw:
            channels += 2
        return channels

    def _get_grid(self, height, width, device, dtype):
        key = (height, width, device, dtype)
        if self._grid_key != key:
            xmin, ymin = self.lidar_range[0], self.lidar_range[1]
            xmax, ymax = self.lidar_range[3], self.lidar_range[4]
            xs = torch.linspace(
                xmin + 0.5 * (xmax - xmin) / width,
                xmax - 0.5 * (xmax - xmin) / width,
                width, device=device, dtype=dtype,
            )
            ys = torch.linspace(
                ymin + 0.5 * (ymax - ymin) / height,
                ymax - 0.5 * (ymax - ymin) / height,
                height, device=device, dtype=dtype,
            )
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            self._grid_key = key
            self._grid_x = grid_x
            self._grid_y = grid_y
        return self._grid_x, self._grid_y

    def forward(self, boxes_list, scores_list, spatial_size, device, dtype):
        """
        Parameters
        ----------
        boxes_list : list[torch.Tensor]
            One (N_i, 7) tensor per sample, ego-frame, order 'hwl'.
        scores_list : list[torch.Tensor]
            One (N_i,) confidence tensor per sample.
        spatial_size : tuple(int, int)
            (H, W) of the ego BEV feature map.
        device, dtype : target device / dtype of the prior map.

        Returns
        -------
        prior : torch.Tensor
            (B, num_channels, H, W)
        """
        height, width = spatial_size
        batch_size = len(boxes_list)
        prior = torch.zeros(
            batch_size, self.num_channels, height, width,
            device=device, dtype=dtype,
        )
        grid_x, grid_y = self._get_grid(height, width, device, dtype)

        for batch_idx in range(batch_size):
            boxes = boxes_list[batch_idx]
            scores = scores_list[batch_idx]
            if boxes is None or boxes.numel() == 0:
                continue
            boxes = boxes.to(device=device, dtype=dtype)
            scores = scores.to(device=device, dtype=dtype)

            center_x = boxes[:, 0].view(-1, 1, 1)
            center_y = boxes[:, 1].view(-1, 1, 1)
            box_h = boxes[:, 3]
            box_w = boxes[:, 4].view(-1, 1, 1)
            box_l = boxes[:, 5].view(-1, 1, 1)
            yaw = boxes[:, 6]

            dx = grid_x.unsqueeze(0) - center_x
            dy = grid_y.unsqueeze(0) - center_y

            channel_idx = 0
            if self.use_gaussian:
                footprint = (boxes[:, 4] * boxes[:, 5]).clamp(min=1e-6)
                sigma = (self.sigma_scale * footprint.sqrt()).clamp(min=self.min_sigma)
                sigma = sigma.view(-1, 1, 1)
                gaussian = scores.view(-1, 1, 1) * torch.exp(
                    -(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2)
                )
                prior[batch_idx, channel_idx] = gaussian.max(dim=0).values
                channel_idx += 1

            if self.use_box_mask or self.use_yaw:
                cos_t = torch.cos(yaw).view(-1, 1, 1)
                sin_t = torch.sin(yaw).view(-1, 1, 1)
                local_x = dx * cos_t + dy * sin_t
                local_y = -dx * sin_t + dy * cos_t
                inside = (
                    (local_x.abs() <= box_l / 2.0)
                    & (local_y.abs() <= box_w / 2.0)
                ).to(dtype)
                masked_scores = inside * scores.view(-1, 1, 1)
                mask_map, best_idx = masked_scores.max(dim=0)

                if self.use_box_mask:
                    prior[batch_idx, channel_idx] = mask_map
                    channel_idx += 1

                if self.use_yaw:
                    covered = mask_map > 0
                    yaw_of_best = yaw[best_idx]
                    prior[batch_idx, channel_idx] = torch.where(
                        covered, torch.cos(yaw_of_best),
                        torch.zeros_like(mask_map),
                    )
                    prior[batch_idx, channel_idx + 1] = torch.where(
                        covered, torch.sin(yaw_of_best),
                        torch.zeros_like(mask_map),
                    )
                    channel_idx += 2

        return prior

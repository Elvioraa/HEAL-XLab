"""Dense innovation-map comparator for the sparse-token ablation."""

import torch.nn as nn

from opencood.models.fuse_modules.pyramid_fuse import weighted_fuse


class DenseInnovationFusion(nn.Module):
    """Fuse full residual maps using the same geometry as official HEAL."""

    def __init__(self, config):
        super().__init__()
        self.residual_scale = float(config["residual_scale"])

    def forward(
        self,
        innovations,
        scores,
        record_len,
        affine_matrix,
        align_corners,
        **fusion_kwargs
    ):
        fused = []
        for innovation, score in zip(innovations, scores):
            fused.append(
                weighted_fuse(
                    innovation,
                    score,
                    record_len,
                    affine_matrix,
                    align_corners,
                    **fusion_kwargs
                )
            )
        return fused

    def add_to_detection_features(self, detection_features, fused_innovations):
        return [
            feature + self.residual_scale * innovation
            for feature, innovation in zip(detection_features, fused_innovations)
        ]

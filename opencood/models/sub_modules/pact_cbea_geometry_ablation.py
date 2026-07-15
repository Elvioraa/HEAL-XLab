"""Parameter-free geometry selections for the PACT aligned-uniform ablation."""

from __future__ import absolute_import, division, print_function

import torch

from opencood.models.sub_modules.pact_cbea_evidence_routed import (
    PACTCBEAEvidenceGeometryAligner,
)


class PACTCBEAGeometryAblationAligner(PACTCBEAEvidenceGeometryAligner):
    """Reuse the production sampler while selecting an explicit ablation row."""

    SUPPORTED_CONVENTIONS = ("B_source_ego", "D_identity")

    def __init__(self, convention, align_corners=False):
        if convention not in self.SUPPORTED_CONVENTIONS:
            raise ValueError(
                "Unsupported geometry ablation convention: %s" % convention
            )
        super(PACTCBEAGeometryAblationAligner, self).__init__(
            align_corners=align_corners
        )
        self.convention = convention
        self.last_debug = {}

    @staticmethod
    def parameter_count():
        return 0

    @staticmethod
    def _identity_affine_like(affine_matrix):
        identity = affine_matrix.new_zeros(affine_matrix.shape)
        identity[..., 0, 0] = 1.0
        identity[..., 1, 1] = 1.0
        return identity

    def _select_affine(self, affine_matrix, records):
        if self.convention == "D_identity":
            selected = self._identity_affine_like(affine_matrix)
            indices = [["identity" for _ in range(count)] for count in records]
            return selected, indices

        # pairwise[s, e] is the source-to-ego physical transform. Feeding it
        # to grid_sample is intentionally a controlled, non-production test of
        # the opposite sampling convention, not a correctness claim.
        selected = affine_matrix.clone()
        indices = []
        for scene_index, count in enumerate(records):
            selected[scene_index, 0, :count] = affine_matrix[
                scene_index, :count, 0
            ]
            indices.append([[source_index, 0] for source_index in range(count)])
        return selected, indices

    def forward(self, feature, heatmap_logits, uncertainty, descriptor,
                record_len, affine_matrix):
        records = self._validate_inputs(
            feature, heatmap_logits, uncertainty, descriptor,
            record_len, affine_matrix
        )
        selected, indices = self._select_affine(affine_matrix, records)
        result = super(PACTCBEAGeometryAblationAligner, self).forward(
            feature, heatmap_logits, uncertainty, descriptor,
            record_len, selected
        )
        selected_affine = []
        for scene_index, count in enumerate(records):
            selected_affine.append(
                selected[scene_index, 0, :count].detach().cpu().tolist()
            )
        debug = dict(result["debug"])
        debug.update({
            "convention": self.convention,
            "selected_pairwise_indices": indices,
            "normalized_affine": selected_affine,
            "validity_ratio": list(debug["per_agent_valid_ratio"]),
        })
        result["debug"] = debug
        self.last_debug = dict(debug)
        return result

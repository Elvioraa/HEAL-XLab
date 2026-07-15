"""Frozen B/D geometry ablations sharing the aligned-uniform PACT pipeline."""

from __future__ import absolute_import, division, print_function

import copy

from opencood.models.heter_pyramid_collab_pact_cbea_aligned_uniform import (
    HeterPyramidCollabPactCbeaAlignedUniform,
)
from opencood.models.sub_modules.pact_cbea_geometry_ablation import (
    PACTCBEAGeometryAblationAligner,
)


class HeterPyramidCollabPactCbeaGeometryAblation(
        HeterPyramidCollabPactCbeaAlignedUniform):
    """Inference-only geometry convention ablation with no trainable state."""

    _BASE_GUARD = {
        "enabled": True,
        "no_joint_training": True,
        "use_stage3_joint_training": False,
        "trainable": False,
        "align_features": True,
        "use_shared_alignment_grid": True,
        "uniform_over_valid_support": True,
        "strict_core_checkpoint": True,
        "packet_only": False,
        "debug": True,
    }

    def __init__(self, args):
        raw_cfg = args.get("pact_cbea_geometry_ablation")
        self._validate_ablation_config(raw_cfg)
        base_args = copy.deepcopy(args)
        base_cfg = dict(self._BASE_GUARD)
        base_cfg["epsilon"] = float(raw_cfg.get("epsilon", 1e-6))
        base_args["pact_cbea_aligned_uniform"] = base_cfg
        super(HeterPyramidCollabPactCbeaGeometryAblation, self).__init__(
            base_args
        )
        self.pact_cbea_geometry_ablation_cfg = dict(raw_cfg)
        self.pact_geometry_aligner = PACTCBEAGeometryAblationAligner(
            convention=raw_cfg["convention"],
            align_corners=bool(getattr(self.pyramid_backbone, "align_corners", False)),
        )
        self._freeze_and_eval()

    @staticmethod
    def _validate_ablation_config(cfg):
        if not isinstance(cfg, dict) or cfg.get("enabled") is not True:
            raise ValueError(
                "PACT geometry ablation requires "
                "pact_cbea_geometry_ablation.enabled=true"
            )
        convention = cfg.get("convention")
        if convention not in PACTCBEAGeometryAblationAligner.SUPPORTED_CONVENTIONS:
            raise ValueError(
                "pact_cbea_geometry_ablation.convention must be one of %s"
                % ", ".join(PACTCBEAGeometryAblationAligner.SUPPORTED_CONVENTIONS)
            )

    def _forward_aligned_uniform(self, data_dict):
        output_dict = super(HeterPyramidCollabPactCbeaGeometryAblation, self)._forward_aligned_uniform(
            data_dict
        )
        debug = output_dict["pact_cbea_aligned_uniform_debug"]
        debug.update({
            "geometry_ablation": True,
            "convention": self.pact_cbea_geometry_ablation_cfg["convention"],
            "selected_pairwise_indices": self.pact_geometry_aligner.last_debug.get(
                "selected_pairwise_indices", []
            ),
            "normalized_affine": self.pact_geometry_aligner.last_debug.get(
                "normalized_affine", []
            ),
            "geometry_validity_ratio": self.pact_geometry_aligner.last_debug.get(
                "validity_ratio", []
            ),
        })
        return output_dict

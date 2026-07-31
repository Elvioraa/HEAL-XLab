"""Frozen PACT-CBEA base with trainable object-level Stage 3 refinement."""

import copy
import math

import torch
import torch.nn as nn

from opencood.models.heter_pyramid_collab_pact_cbea import (
    HeterPyramidCollabPactCbea,
)
from opencood.models.sub_modules.pact_cbea_object_refiner import (
    ObjectResidualCoder,
    SharedObjectRefiner,
    precision_fuse,
    repository_hwl_to_sampler_lwh,
)
from opencood.models.sub_modules.pact_cbea_object_roi import (
    BEVGeometry,
    RotatedBEVROISampler,
)
from opencood.models.sub_modules.pact_cbea_object_stage3_utils import (
    CollaborativeProposalDecoder,
    OBJECT_STAGE3_VERSION,
    freeze_except_object_stage3,
    match_proposals_to_gt,
    rotated_nms_sampler_boxes,
    stage3_parameter_summary,
)


class HeterPyramidCollabPactCbeaObjectStage3(HeterPyramidCollabPactCbea):
    """Independent object-level Stage 3 over a frozen four-modality base.

    The object path is built only when
    ``pact_cbea.object_level_stage3.enabled`` is explicitly true. Otherwise
    this class registers no Stage 3 parameters and delegates directly to the
    current PACT-CBEA forward path.
    """

    def __init__(self, args):
        raw_pact_cfg = args.get("pact_cbea")
        raw_object_cfg = (
            raw_pact_cfg.get("object_level_stage3")
            if isinstance(raw_pact_cfg, dict) else None
        )
        object_cfg = self._normalize_object_stage3_cfg(raw_object_cfg)
        base_mode = "configured_pact_cbea"
        if (
                object_cfg["enabled"]
                and object_cfg["require_strict_heal_base"]):
            base_mode = self._validate_strict_heal_base_config(
                args, object_cfg
            )
        super(HeterPyramidCollabPactCbeaObjectStage3, self).__init__(args)
        self.object_stage3_cfg = object_cfg
        self.object_stage3_enabled = bool(object_cfg["enabled"])
        self.object_stage3_runtime_enabled = self.object_stage3_enabled
        self.object_stage3_base_mode = base_mode

        if not self.object_stage3_enabled:
            return

        bev_cfg = object_cfg["bev_geometry"]
        lidar_range = args["lidar_range"]
        self.object_stage3_geometry = BEVGeometry(
            x_min=float(bev_cfg.get("x_min", lidar_range[0])),
            x_max=float(bev_cfg.get("x_max", lidar_range[3])),
            y_min=float(bev_cfg.get("y_min", lidar_range[1])),
            y_max=float(bev_cfg.get("y_max", lidar_range[4])),
            resolution_x=float(bev_cfg["resolution_x"]),
            resolution_y=float(bev_cfg["resolution_y"]),
            feature_stride_x=float(bev_cfg["feature_stride_x"]),
            feature_stride_y=float(bev_cfg["feature_stride_y"]),
        )
        self.object_stage3_roi_sampler = RotatedBEVROISampler(
            self.object_stage3_geometry,
            roi_size=tuple(object_cfg["roi_size"]),
            min_coverage=object_cfg["min_coverage"],
        )
        self.object_stage3_refiner = SharedObjectRefiner(
            in_channels=int(args["in_head"]),
            hidden_dim=object_cfg["hidden_dim"],
            min_log_variance=object_cfg["min_log_variance"],
            max_log_variance=object_cfg["max_log_variance"],
            initial_log_variance=object_cfg["initial_log_variance"],
        )
        self.object_stage3_coder = ObjectResidualCoder(
            min_size=object_cfg["min_box_size"],
            max_log_scale=object_cfg["max_log_scale"],
            strict_sizes=True,
        )
        self.object_stage3_proposal_decoder = CollaborativeProposalDecoder(
            object_cfg, args["dir_args"], lidar_range
        )
        freeze_except_object_stage3(self)
        self.train(True)

    def forward(self, data_dict):
        if not self.object_stage3_enabled or not self.object_stage3_runtime_enabled:
            return super(HeterPyramidCollabPactCbeaObjectStage3, self).forward(
                data_dict
            )

        # The current multiscale PACT base is inference-only. Stage 3 training
        # therefore executes the fully frozen base in eval/no-grad mode, then
        # restores the outer training flag for matching and the trainable head.
        outer_training = self.training
        self.training = False
        try:
            with torch.no_grad():
                output_dict = super(
                    HeterPyramidCollabPactCbeaObjectStage3, self
                ).forward(data_dict)
        finally:
            self.training = outer_training

        context = output_dict.pop("_pact_cbea_object_context", None)
        if context is None:
            raise RuntimeError(
                "object Stage 3 requested per-agent context but base did not provide it"
            )
        if context.get("single_feature_frame") != "per_agent_local":
            raise RuntimeError("unexpected single_feature coordinate frame")
        if context.get("pairwise_direction") != "source_i_to_target_j":
            raise RuntimeError("unexpected pairwise transform direction")

        single_feature = context["single_feature"].detach()
        record_len = context["record_len"]
        pairwise_t_matrix = context["pairwise_t_matrix"]
        if pairwise_t_matrix is None:
            raise KeyError("data_dict must contain metric pairwise_t_matrix")
        with torch.no_grad():
            proposal_list, score_list = self.object_stage3_proposal_decoder.decode(
                output_dict["cls_preds"],
                output_dict["reg_preds"],
                output_dict["dir_preds"],
                data_dict["anchor_box"],
            )

        compute_targets = bool(
            data_dict.get("object_stage3_compute_targets", outer_training)
        )
        scenes = []
        feature_offset = 0
        record_values = [int(value) for value in record_len.detach().cpu().tolist()]
        for batch_idx, agent_count in enumerate(record_values):
            scene_features = single_feature[
                feature_offset:feature_offset + agent_count
            ].detach()
            feature_offset += agent_count
            proposals = proposal_list[batch_idx].detach()
            proposal_scores = score_list[batch_idx].detach()
            ego_to_agent = pairwise_t_matrix[
                batch_idx, 0, :agent_count
            ].to(device=scene_features.device).detach()

            roi_features, valid_mask, coverage = self.object_stage3_roi_sampler(
                scene_features,
                proposals,
                ego_to_agent,
            )
            roi_features = roi_features.detach()
            agent_residuals, agent_log_variances = self.object_stage3_refiner(
                roi_features, valid_mask
            )
            fused_residual, normalized_weights, fallback_mask = precision_fuse(
                agent_residuals,
                agent_log_variances,
                valid_mask,
                eps=self.object_stage3_cfg["precision_eps"],
                coverage=coverage,
                use_coverage_weight=self.object_stage3_cfg[
                    "use_coverage_weight"
                ],
            )
            refined_boxes = self.object_stage3_coder.decode(
                proposals.to(fused_residual.dtype), fused_residual
            )
            final_boxes, final_scores, final_keep = rotated_nms_sampler_boxes(
                refined_boxes.detach(),
                proposal_scores,
                threshold=self.object_stage3_cfg["refined_nms_threshold"],
                max_boxes=self.object_stage3_cfg["proposal_post_nms_topk"],
            )

            matching = self._empty_matching(proposals)
            target_residuals = proposals.new_zeros(proposals.shape)
            positive_count = 0
            if compute_targets:
                gt_boxes = self._scene_gt_boxes(
                    data_dict, batch_idx, proposals.device, proposals.dtype
                )
                matching = match_proposals_to_gt(
                    proposals,
                    gt_boxes,
                    positive_iou_threshold=self.object_stage3_cfg[
                        "positive_iou_threshold"
                    ],
                    ignore_iou_threshold=self.object_stage3_cfg[
                        "ignore_iou_threshold"
                    ],
                )
                positive = matching["positive_mask"]
                positive_count = int(positive.sum().item())
                if positive_count:
                    target_residuals[positive] = self.object_stage3_coder.encode(
                        proposals[positive],
                        matching["matched_gt_boxes"][positive],
                    )

            scenes.append({
                "proposals": proposals,
                "proposal_scores": proposal_scores,
                "agent_residuals": agent_residuals,
                "agent_log_variances": agent_log_variances,
                "fused_residual": fused_residual,
                "normalized_agent_weights": normalized_weights,
                "fallback_mask": fallback_mask,
                "refined_boxes": refined_boxes,
                "final_boxes": final_boxes,
                "final_scores": final_scores,
                "final_keep": final_keep,
                "valid_mask": valid_mask,
                "coverage": coverage,
                "target_residuals": target_residuals,
                "positive_mask": matching["positive_mask"],
                "ignore_mask": matching["ignore_mask"],
                "matched_ious": matching["matched_ious"],
                "proposal_count": int(proposals.shape[0]),
                "positive_count": positive_count,
                "agent_count": agent_count,
                "roi_feature_shape": tuple(roi_features.shape),
            })

        if feature_offset != single_feature.shape[0]:
            raise RuntimeError("record_len does not consume all single_feature agents")
        zero_loss_anchor = sum(
            parameter.sum() * 0.0
            for parameter in self.object_stage3_refiner.parameters()
        )
        output_dict["object_stage3"] = {
            "enabled": True,
            "version": OBJECT_STAGE3_VERSION,
            "object_stage3_base_mode": self.object_stage3_base_mode,
            "scenes": scenes,
            "zero_loss_anchor": zero_loss_anchor,
            "single_feature_shape": tuple(single_feature.shape),
            "single_feature_frame": "per_agent_local",
            "pairwise_direction": "source_i_to_target_j",
            "record_len": tuple(record_values),
            "agent_modality_list_diagnostic": context[
                "agent_modality_list"
            ],
        }
        return output_dict

    def train(self, mode=True):
        if not getattr(self, "object_stage3_enabled", False):
            return super(
                HeterPyramidCollabPactCbeaObjectStage3, self
            ).train(mode)
        nn.Module.train(self, False)
        self.training = bool(mode)
        self.object_stage3_refiner.train(mode)
        freeze_except_object_stage3(self)
        return self

    def set_object_stage3_runtime_enabled(self, enabled):
        """Enable or bypass an already constructed Stage 3 for A/B inference."""
        enabled = bool(enabled)
        if enabled and not self.object_stage3_enabled:
            raise RuntimeError("cannot enable an unconstructed object Stage 3")
        self.object_stage3_runtime_enabled = enabled

    def object_stage3_parameter_summary(self):
        """Return strict trainable/frozen parameter diagnostics."""
        if not self.object_stage3_enabled:
            return {
                "trainable_names": [],
                "trainable_count": 0,
                "frozen_count": sum(p.numel() for p in self.parameters()),
            }
        return stage3_parameter_summary(self)

    def _object_stage3_context_requested(self):
        return bool(
            getattr(self, "object_stage3_enabled", False)
            and getattr(self, "object_stage3_runtime_enabled", False)
        )

    @staticmethod
    def _scene_gt_boxes(data_dict, batch_idx, device, dtype):
        if "object_bbx_center" not in data_dict or "object_bbx_mask" not in data_dict:
            raise KeyError(
                "training targets require object_bbx_center and object_bbx_mask"
            )
        mask = data_dict["object_bbx_mask"][batch_idx].bool()
        repository_boxes = data_dict["object_bbx_center"][batch_idx][mask]
        repository_boxes = repository_boxes.to(device=device, dtype=dtype)
        return repository_hwl_to_sampler_lwh(repository_boxes).detach()

    @staticmethod
    def _empty_matching(proposals):
        count = proposals.shape[0]
        return {
            "positive_mask": torch.zeros(
                (count,), dtype=torch.bool, device=proposals.device
            ),
            "ignore_mask": torch.zeros(
                (count,), dtype=torch.bool, device=proposals.device
            ),
            "matched_ious": proposals.new_zeros((count,)),
        }

    @staticmethod
    def _validate_strict_heal_base_config(args, object_cfg):
        """Reject every setting that could alter the frozen HEAL base path.

        This validation intentionally reads the raw configuration. Running the
        general PACT normalizer first could fill in a missing strict-mode field
        or raise on one field before all violations have been collected.
        """
        del object_cfg
        missing = "<missing>"

        def lookup(path):
            value = args
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    return missing
                value = value[key]
            return value

        expected = [
            (("pact_cbea", "enabled"), True),
            (("pact_cbea", "trainable"), False),
            (("pact_cbea", "no_joint_training"), True),
            (("pact_cbea", "use_stage3_joint_training"), False),
            (("pact_cbea", "fusion_mode"), "heal_multiscale_prior"),
            (("pact_cbea", "multiscale_prior", "enabled"), True),
            (("pact_cbea", "multiscale_prior", "lambda"), 0.0),
            (
                ("pact_cbea", "multiscale_prior", "injection_strength"),
                0.0,
            ),
            (("pact_cbea", "local_evidence", "enabled"), False),
            (("pact_cbea", "evidence_head", "enabled"), False),
            (
                (
                    "pact_cbea",
                    "evidence_head",
                    "localization_uncertainty",
                    "enabled",
                ),
                False,
            ),
            (("supervise_single",), False),
        ]
        for key in (
                "evidence_confidence",
                "uncertainty_weight",
                "descriptor_consistency",
                "spatial_consistency",
                "modality_prior",
                "localization_weight"):
            expected.append((("pact_cbea", "aggregation", key), False))

        violations = []
        for path, required in expected:
            actual = lookup(path)
            matches = actual == required
            if isinstance(required, bool):
                matches = matches and isinstance(actual, bool)
            if not matches:
                diagnostic_path = ".".join(path)
                if path == ("supervise_single",):
                    diagnostic_path = "model.args.supervise_single"
                violations.append(
                    "%s expected %r, got %r"
                    % (diagnostic_path, required, actual)
                )
        if violations:
            raise ValueError(
                "strict HEAL base configuration violations:\n - %s"
                % "\n - ".join(violations)
            )
        return "strict_heal_lambda_zero"

    @staticmethod
    def _normalize_object_stage3_cfg(cfg):
        defaults = {
            "enabled": False,
            "require_strict_heal_base": False,
            "version": OBJECT_STAGE3_VERSION,
            "start_from_scratch": False,
            "roi_size": [7, 7],
            "min_coverage": 0.5,
            "proposal_score_threshold": 0.2,
            "proposal_pre_nms_topk": 512,
            "proposal_post_nms_topk": 100,
            "proposal_nms_threshold": 0.15,
            "positive_iou_threshold": 0.55,
            "ignore_iou_threshold": 0.35,
            "hidden_dim": 128,
            "min_log_variance": -4.0,
            "max_log_variance": 4.0,
            "initial_log_variance": 0.0,
            "precision_eps": 1e-6,
            "use_coverage_weight": False,
            "min_box_size": 1e-3,
            "max_log_scale": 5.0,
            "refined_nms_threshold": 0.15,
            "bev_geometry": {
                "resolution_x": 0.4,
                "resolution_y": 0.4,
                "feature_stride_x": 2.0,
                "feature_stride_y": 2.0,
            },
            "base_checkpoint": None,
            "stage3_checkpoint": None,
        }
        if cfg is None:
            normalized = defaults
        elif isinstance(cfg, dict):
            normalized = copy.deepcopy(defaults)
            _deep_update(normalized, cfg)
        else:
            raise TypeError("object_level_stage3 must be a mapping")
        normalized["enabled"] = bool(normalized["enabled"])
        normalized["require_strict_heal_base"] = bool(
            normalized["require_strict_heal_base"]
        )
        if int(normalized["version"]) != OBJECT_STAGE3_VERSION:
            raise ValueError("unsupported object Stage 3 version")
        if len(normalized["roi_size"]) != 2:
            raise ValueError("roi_size must contain [Rh,Rw]")
        if not 0.0 < float(normalized["min_coverage"]) <= 1.0:
            raise ValueError("min_coverage must lie in (0,1]")
        if not (
                0.0 <= float(normalized["ignore_iou_threshold"])
                <= float(normalized["positive_iou_threshold"]) <= 1.0):
            raise ValueError("invalid matching IoU thresholds")
        for key in ("precision_eps", "min_box_size", "max_log_scale"):
            if not math.isfinite(float(normalized[key])) or float(normalized[key]) <= 0:
                raise ValueError("%s must be finite and positive" % key)
        return normalized


def _deep_update(destination, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _deep_update(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)

"""Shared CPU fixtures for Full Dual-Space synthetic smoke tests."""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from opencood.models.sub_modules.dual_space_object import (
    configure_dual_space_trainability,
    install_dual_space_modules,
)


def make_dual_config(
    mode="stage1_anchor",
    active_modality=None,
    multi=False,
    quality=False,
    rescue=False,
    mixed=False,
    fusion="concat_projection",
    report_stats=False,
    detail_roi_size=5,
    context_roi_size=3,
    quality_use_coverage=True,
    quality_use_distance=True,
    diagnostics=None,
    v5_quality_safe=None,
    v6_residual_safe=None,
):
    """Return one complete explicit synthetic Dual-Space configuration."""
    if rescue:
        mode = "inference"
    version = "ds_v3" if quality else ("ds_v2" if multi else "ds_v1")
    if rescue:
        version = "ds_v4"
    config = {
        "enabled": True,
        "version": version,
        "experiment_profile": version,
        "mode": mode,
        "allow_untrained_initialization": mode == "stage1_anchor",
        "roi": {
            "output_size": 5,
            "max_train_proposals": 32,
            "max_infer_proposals": 16,
            "chunk_size": 4,
            "min_coverage": 0.5,
        },
        "adapter": {"type": "residual_1x1", "zero_init": True},
        "object_encoder": {
            "embedding_dim": 16,
            "hidden_channels": 8,
            "pooled_size": 2,
        },
        "geometry_encoder": {"enabled": True, "hidden_dim": 8},
        "refiner": {
            "hidden_dim": 16,
            "yaw_mode": "sin_cos",
            "zero_init_output": True,
        },
        "multi_scale": {"enabled": bool(multi)},
        "quality": {"enabled": bool(quality)},
        "consensus": {
            "mode": "quality_weighted" if quality else "uniform_geometry_mean",
            "fallback_to_original": True,
        },
        "training_proposals": {
            "source": "mixed" if mixed else "gt_jitter",
            "include_gt": True,
            "jitters_per_gt": 1,
            "center_xy_std_rel": 0.10,
            "center_z_std_rel": 0.05,
            "log_size_std": 0.05,
            "yaw_std_deg": 5.0,
            "max_proposals": 32,
        },
        "remote_proposal_rescue": {"enabled": bool(rescue)},
        "loss": {
            "object_loss_weight": 0.2,
            "individual_loss_weight": 1.0,
            "consensus_loss_weight": 1.0,
            "iou_loss_weight": 0.0,
        },
        "report_stats": bool(report_stats),
    }
    if multi:
        config["multi_scale"].update(
            {
                "detail": {"roi_size": detail_roi_size},
                "context": {"roi_size": context_roi_size},
                "fusion": fusion,
            }
        )
    if quality:
        config["quality"].update(
            {
                "target": "refined_iou",
                "hidden_dim": 8,
                "use_roi_coverage": bool(quality_use_coverage),
                "use_agent_distance": bool(quality_use_distance),
                "detach_target": True,
                "detach_weight_for_consensus": True,
            }
        )
        config["consensus"].update(
            {"min_quality_sum": 1e-6, "low_quality_fallback": "uniform"}
        )
        config["loss"]["quality_loss_weight"] = 0.05
    if mixed:
        config["training_proposals"]["predicted"] = {
            "enabled": True,
            "max_per_scene": 8,
            "min_score": 0.2,
            "positive_iou_min": 0.3,
        }
    if rescue:
        config["remote_proposal_rescue"].update(
            {
                "include_ego": False,
                "min_score": 0.5,
                "dedup_iou": 0.5,
                "max_per_agent": 4,
                "max_total_added": 4,
            }
        )
    if active_modality is not None:
        config["active_modality"] = active_modality
    if diagnostics is not None:
        config["diagnostics"] = copy.deepcopy(diagnostics)
    if v5_quality_safe is not None:
        config["v5_quality_safe"] = copy.deepcopy(v5_quality_safe)
    if v6_residual_safe is not None:
        config["v6_residual_safe"] = copy.deepcopy(v6_residual_safe)
    return config


class TinyDualSpaceHost(nn.Module):
    """Small host exposing the same attributes used by Dual-Space plumbing."""

    def __init__(
        self,
        modalities=("m1", "m2"),
        mode="stage1_anchor",
        active_modality=None,
        multi=False,
        quality=False,
        rescue=False,
        mixed=False,
        fusion="concat_projection",
        report_stats=False,
        detail_roi_size=5,
        context_roi_size=3,
        quality_use_coverage=True,
        quality_use_distance=True,
        detail_channels=4,
        context_channels=6,
        diagnostics=None,
        v5_quality_safe=None,
        v6_residual_safe=None,
    ):
        super().__init__()
        self.modality_name_list = list(modalities)
        self.sensor_type_dict = OrderedDict(
            (name, "lidar") for name in modalities
        )
        self.base_branches = nn.ModuleDict(
            (name, nn.Conv2d(detail_channels, detail_channels, 1, bias=False))
            for name in modalities
        )
        args = {
            "lidar_range": [-16.0, -16.0, -3.0, 16.0, 16.0, 1.0],
            "fusion_backbone": {
                "num_filters": [detail_channels, context_channels, 8]
            },
            "dual_space": make_dual_config(
                mode=mode,
                active_modality=active_modality,
                multi=multi,
                quality=quality,
                rescue=rescue,
                mixed=mixed,
                fusion=fusion,
                report_stats=report_stats,
                detail_roi_size=detail_roi_size,
                context_roi_size=context_roi_size,
                quality_use_coverage=quality_use_coverage,
                quality_use_distance=quality_use_distance,
                diagnostics=diagnostics,
                v5_quality_safe=v5_quality_safe,
                v6_residual_safe=v6_residual_safe,
            ),
        }
        for name in modalities:
            args[name] = {
                "backbone_args": {"num_filters": [detail_channels]}
            }
        install_dual_space_modules(self, args)
        configure_dual_space_trainability(self)
        self._dual_space_checkpoint_ready = True


def make_boxes(count=1, x=0.0, y=0.0):
    """Return valid repository-order synthetic boxes."""
    boxes = torch.zeros(count, 7)
    boxes[:, 0] = x
    boxes[:, 1] = y
    boxes[:, 3] = 1.5
    boxes[:, 4] = 2.0
    boxes[:, 5] = 4.0
    return boxes


def make_scene(host, agent_count=2, height=32, width=32):
    """Return one complete synthetic scene for the host's enabled features."""
    modalities = tuple(
        host.modality_name_list[index % len(host.modality_name_list)]
        for index in range(agent_count)
    )
    features = torch.randn(
        agent_count, host.dual_space_common_bev_channels, height, width
    )
    scene = {
        "agent_features": features,
        "agent_support": features.new_ones((agent_count, 1, height, width)),
        "agent_modalities": modalities,
    }
    if host.dual_space_flags["multi_scale"]:
        context = torch.randn(
            agent_count,
            host.dual_space_context_bev_channels,
            height // 2,
            width // 2,
        )
        scene["context_agent_features"] = context
        scene["context_agent_support"] = context.new_ones(
            (agent_count, 1, height // 2, width // 2)
        )
    if host.dual_space_flags["quality"]:
        scene["agent_positions"] = features.new_zeros((agent_count, 2))
        if agent_count > 1:
            scene["agent_positions"][1:, 0] = torch.arange(
                1, agent_count, dtype=features.dtype
            ) * 4.0
    return scene


def run_registered_tests(tests):
    """Run a flat smoke registry and return a process exit code."""
    torch.set_num_threads(1)
    passed = 0
    for name, function in tests:
        try:
            function()
        except Exception as error:
            print("[FAIL] %s: %s: %s" % (name, type(error).__name__, error))
        else:
            passed += 1
            print("[PASS] %s" % name)
    print("RESULT: %d/%d PASS" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1

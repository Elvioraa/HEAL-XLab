"""Central configuration normalization and validation for Open-DCSI.

Every optional feature is disabled by default. This module is intentionally the
only source of Open-DCSI defaults so model, loss, codec, and audit paths cannot
silently diverge as later development phases add their implementations.
"""

from copy import deepcopy
from collections.abc import Mapping


OPEN_DCSI_CONFIG_DEFAULTS = {
    "enabled": False,
    "compatibility": {
        "strict_baseline_bypass": True,
        "old_checkpoint_compatible": True,
        "missing_module_fallback": "baseline",
        "non_finite_fallback": "ego",
        "preserve_agent_permutation": True,
    },
    "open_heterogeneous": {
        "enabled": False,
        "enforce_stage1_homogeneous_m1": True,
        "enforce_stage2_single_modality": True,
        "freeze_shared_modules_stage2": True,
        "forbid_pair_specific_modules": True,
        "forbid_fixed_modality_router": True,
    },
    "stage2_independent": {
        "enabled": False,
    },
    "common_space": {
        "enabled": False,
        "scales": "all",
        "channel_ratio": 0.25,
        "min_channels": 8,
        "projector": {
            "enabled": False,
            "type": "pointwise_depthwise",
            "norm": "none",
            "activation": "silu",
            "residual": False,
        },
        "decoder": {
            "enabled": False,
            "zero_init_residual": True,
        },
        "reconstruction": {
            "enabled": False,
            "detach_target": True,
        },
        "common_detection_supervision": {
            "enabled": False,
        },
    },
    "common_fusion": {
        "enabled": False,
        "type": "bgea_residual",
        "use_evidence": True,
        "use_uncertainty": True,
        "use_localization_quality": True,
        "use_descriptor_conflict": False,
        "scale_specific_gate": True,
        "absolute_reject": {
            "enabled": False,
            "threshold": 0.05,
            "temperature": 0.05,
            "hard_threshold_inference_only": True,
        },
        "ego_identity": True,
        "fallback_to_official": True,
    },
    "innovation_tokens": {
        "enabled": False,
        "source": "common_proposals",
        "token_dim": 64,
        "semantic_dim": 32,
        "geometry_dim": 16,
        "max_tokens_per_agent": 64,
        "proposal_topk": 128,
        "foreground_threshold": 0.1,
        "nms_threshold": 0.1,
        "roi_pool_size": 3,
        "standardized_schema": True,
        "residual_definition": "feature_reconstruction_residual",
        "detach_common_for_residual": False,
    },
    "innovation_quality": {
        "enabled": False,
        "use_evidence": True,
        "use_general_uncertainty": True,
        "use_localization_uncertainty": True,
        "predict_token_validity": True,
        "predict_box_quality": True,
        "calibration": {
            "enabled": False,
            "monotonic": True,
        },
    },
    "innovation_aggregation": {
        "enabled": False,
        "type": "permutation_invariant_gated_set",
        "pair_specific_parameters": False,
        "fixed_modality_embeddings": False,
        "geometric_clustering": {
            "enabled": False,
            "center_radius": 2.0,
            "yaw_threshold": 0.785,
        },
        "absolute_reject": {
            "enabled": False,
            "threshold": 0.05,
            "temperature": 0.05,
            "hard_threshold_inference_only": True,
        },
        "ego_token_identity": True,
    },
    "dense_innovation_map": {
        "enabled": False,
        "residual_scale": 1.0,
    },
    "cross_scale_geometry": {
        "enabled": False,
        "scales": "all",
        "sampling_points": 4,
        "offset_limit": {"x": 2.0, "y": 2.0},
        "localization_quality_controls_offset": True,
        "shared_across_modalities": True,
        "shared_across_scales": False,
    },
    "geometry_refiner": {
        "enabled": False,
        "predict_center": True,
        "predict_size": True,
        "predict_height": True,
        "predict_yaw": True,
        "predict_confidence_delta": False,
        "zero_init_output": True,
        "residual_on_official_boxes": True,
        "max_center_delta": 2.0,
        "max_yaw_delta": 0.785,
    },
    "communication": {
        "enabled": False,
        "common_codec": {
            "enabled": False,
            "precision": "int8",
            "count_metadata": True,
            "per_channel_scale": False,
        },
        "token_codec": {
            "enabled": False,
            "precision": "int8",
            "count_metadata": True,
        },
        "budget": {
            "enabled": False,
            "mode": "bytes",
            "bytes_per_frame": None,
            "ratio_of_dense": 0.125,
            "fixed_tokens": None,
            "include_headers": True,
            "include_indices": True,
            "include_quant_params": True,
        },
        "selection": {
            "enabled": False,
            "score": "marginal_quality",
            "allow_negative_reject": True,
            "deterministic_inference": True,
        },
    },
    "streaming_fusion": {
        "enabled": False,
        "inference_only": True,
        "process_agent_sequentially": True,
        "release_packet_after_fusion": True,
        "recover_full_dense_stack": False,
        "numerical_parity_test": True,
    },
    "losses": {
        "enabled": False,
        "common_detection": {"enabled": False, "weight": 1.0},
        "reconstruction": {"enabled": False, "weight": 0.05},
        "common_innovation_decorrelation": {
            "enabled": False,
            "weight": 0.01,
        },
        "innovation_detection": {"enabled": False, "weight": 1.0},
        "box_refinement": {"enabled": False, "weight": 1.0},
        "quality": {"enabled": False, "weight": 0.01},
        "token_sparsity": {"enabled": False, "weight": 0.0001},
        "budget": {"enabled": False, "weight": 0.001},
    },
    "diagnostics": {
        "enabled": False,
        "log_every_n_steps": 100,
        "save_token_statistics": False,
        "save_weight_statistics": False,
        "measure_bandwidth": True,
        "measure_parameters": True,
        "measure_vram": False,
        "measure_latency": False,
    },
}


_FEATURE_PATHS = (
    "open_heterogeneous",
    "stage2_independent",
    "common_space",
    "common_space.projector",
    "common_space.decoder",
    "common_space.reconstruction",
    "common_space.common_detection_supervision",
    "common_fusion",
    "common_fusion.absolute_reject",
    "innovation_tokens",
    "innovation_quality",
    "innovation_quality.calibration",
    "innovation_aggregation",
    "innovation_aggregation.geometric_clustering",
    "innovation_aggregation.absolute_reject",
    "dense_innovation_map",
    "cross_scale_geometry",
    "geometry_refiner",
    "communication",
    "communication.common_codec",
    "communication.token_codec",
    "communication.budget",
    "communication.selection",
    "streaming_fusion",
    "losses",
    "losses.common_detection",
    "losses.reconstruction",
    "losses.common_innovation_decorrelation",
    "losses.innovation_detection",
    "losses.box_refinement",
    "losses.quality",
    "losses.token_sparsity",
    "losses.budget",
    "diagnostics",
)


def _extract_open_dcsi_config(config):
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError("Open-DCSI config must be a mapping or None")
    if "model" in config:
        model = config.get("model")
        if not isinstance(model, Mapping):
            raise TypeError("model must be a mapping")
        args = model.get("args", {})
        if not isinstance(args, Mapping):
            raise TypeError("model.args must be a mapping")
        return args.get("open_dcsi")
    if "open_dcsi" in config and "enabled" not in config:
        return config.get("open_dcsi")
    return config


def _merge_known(defaults, supplied, path="open_dcsi"):
    if supplied is None:
        return deepcopy(defaults)
    if not isinstance(supplied, Mapping):
        raise TypeError("{} must be a mapping".format(path))

    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        raise ValueError(
            "Unknown Open-DCSI config field(s) at {}: {}".format(
                path, ", ".join(unknown)
            )
        )

    merged = deepcopy(defaults)
    for key, value in supplied.items():
        default_value = defaults[key]
        current_path = "{}.{}".format(path, key)
        if isinstance(default_value, Mapping):
            merged[key] = _merge_known(default_value, value, current_path)
        else:
            merged[key] = deepcopy(value)
    return merged


def _get_path(config, path):
    value = config
    for part in path.split("."):
        value = value[part]
    return value


def _require_bool(config, path):
    value = _get_path(config, path)
    if not isinstance(value, bool):
        raise TypeError("Open-DCSI {} must be a boolean".format(path))


def _require_positive(config, path, allow_zero=False):
    value = _get_path(config, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Open-DCSI {} must be numeric".format(path))
    valid = value >= 0 if allow_zero else value > 0
    if not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError("Open-DCSI {} must be {}".format(path, relation))


def normalize_open_dcsi_config(config=None):
    """Return a deep, fully populated Open-DCSI configuration."""

    supplied = _extract_open_dcsi_config(config)
    return _merge_known(OPEN_DCSI_CONFIG_DEFAULTS, supplied)


def is_open_dcsi_enabled(config=None):
    """Return whether the top-level Open-DCSI switch is explicitly enabled."""

    supplied = _extract_open_dcsi_config(config)
    if supplied is None:
        return False
    if not isinstance(supplied, Mapping):
        raise TypeError("Open-DCSI config must be a mapping or None")
    enabled = supplied.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("Open-DCSI enabled must be a boolean")
    return enabled


def validate_open_dcsi_config(config, implemented_modules=()):
    """Validate normalized values, dependencies, and phase availability.

    Parameters
    ----------
    config : mapping
        Raw or normalized Open-DCSI configuration.
    implemented_modules : iterable[str]
        Feature paths implemented by the current development phase.
    """

    normalized = normalize_open_dcsi_config(config)
    for path in ("enabled",) + tuple(
        "{}.enabled".format(path) for path in _FEATURE_PATHS
    ):
        _require_bool(normalized, path)

    if not normalized["enabled"]:
        return normalized

    implemented = set(implemented_modules)
    for path in _FEATURE_PATHS:
        if _get_path(normalized, path)["enabled"] and path not in implemented:
            raise ValueError(
                "Open-DCSI {} is not implemented in the current development phase".format(
                    path
                )
            )

    _require_positive(normalized, "common_space.channel_ratio")
    _require_positive(normalized, "common_space.min_channels")
    _require_positive(normalized, "innovation_tokens.token_dim")
    _require_positive(normalized, "innovation_tokens.semantic_dim")
    _require_positive(normalized, "innovation_tokens.geometry_dim")
    _require_positive(normalized, "innovation_tokens.max_tokens_per_agent")
    _require_positive(normalized, "innovation_tokens.proposal_topk")
    _require_positive(normalized, "innovation_tokens.roi_pool_size")
    _require_positive(normalized, "dense_innovation_map.residual_scale", allow_zero=True)
    _require_positive(normalized, "cross_scale_geometry.sampling_points")
    _require_positive(normalized, "geometry_refiner.max_center_delta")
    _require_positive(normalized, "geometry_refiner.max_yaw_delta")

    if normalized["geometry_refiner"]["enabled"] and not normalized[
        "innovation_tokens"
    ]["enabled"]:
        raise ValueError("Open-DCSI geometry_refiner requires innovation_tokens")
    if normalized["cross_scale_geometry"]["enabled"] and not normalized[
        "innovation_tokens"
    ]["enabled"]:
        raise ValueError("Open-DCSI cross_scale_geometry requires innovation_tokens")
    if normalized["streaming_fusion"]["enabled"] and not (
        normalized["common_space"]["enabled"]
        or normalized["innovation_tokens"]["enabled"]
    ):
        raise ValueError(
            "Open-DCSI streaming_fusion requires common_space or innovation_tokens"
        )
    if normalized["streaming_fusion"]["enabled"] and not normalized[
        "common_fusion"
    ]["enabled"]:
        raise ValueError("Open-DCSI streaming_fusion requires common_fusion")
    if normalized["streaming_fusion"]["enabled"] and normalized[
        "streaming_fusion"
    ]["recover_full_dense_stack"]:
        raise ValueError(
            "Open-DCSI streaming_fusion forbids recover_full_dense_stack"
        )
    if normalized["communication"]["common_codec"]["enabled"] and not normalized[
        "common_space"
    ]["enabled"]:
        raise ValueError("Open-DCSI common_codec requires common_space")
    if normalized["communication"]["token_codec"]["enabled"] and not normalized[
        "innovation_tokens"
    ]["enabled"]:
        raise ValueError("Open-DCSI token_codec requires innovation_tokens")
    if normalized["communication"]["common_codec"]["enabled"] and not normalized[
        "communication"
    ]["enabled"]:
        raise ValueError("Open-DCSI common_codec requires communication")
    if normalized["communication"]["token_codec"]["enabled"] and not normalized[
        "communication"
    ]["enabled"]:
        raise ValueError("Open-DCSI token_codec requires communication")
    if normalized["communication"]["budget"]["enabled"]:
        mode = normalized["communication"]["budget"]["mode"]
        if mode not in ("bytes", "dense_ratio", "fixed_tokens"):
            raise ValueError(
                "Open-DCSI communication budget mode must be bytes, dense_ratio, or fixed_tokens"
            )
        if mode == "bytes" and normalized["communication"]["budget"][
            "bytes_per_frame"
        ] is None:
            raise ValueError(
                "Open-DCSI bytes budget requires bytes_per_frame"
            )
        if mode == "fixed_tokens" and normalized["communication"]["budget"][
            "fixed_tokens"
        ] is None:
            raise ValueError("Open-DCSI fixed_tokens budget requires fixed_tokens")
    if normalized["common_space"]["enabled"] and not normalized["common_space"][
        "projector"
    ]["enabled"]:
        raise ValueError("Open-DCSI common_space requires common_space.projector")
    if normalized["common_space"]["enabled"] and not normalized["common_space"][
        "decoder"
    ]["enabled"]:
        raise ValueError("Open-DCSI common_space requires common_space.decoder")
    if normalized["common_fusion"]["enabled"] and not normalized["common_space"][
        "enabled"
    ]:
        raise ValueError("Open-DCSI common_fusion requires common_space")
    if normalized["innovation_tokens"]["enabled"] and not normalized[
        "common_space"
    ]["enabled"]:
        raise ValueError("Open-DCSI innovation_tokens requires common_space")
    if normalized["innovation_quality"]["enabled"] and not normalized[
        "innovation_tokens"
    ]["enabled"]:
        raise ValueError("Open-DCSI innovation_quality requires innovation_tokens")
    if normalized["innovation_aggregation"]["enabled"] and not normalized[
        "innovation_tokens"
    ]["enabled"]:
        raise ValueError(
            "Open-DCSI innovation_aggregation requires innovation_tokens"
        )
    if normalized["innovation_aggregation"]["geometric_clustering"][
        "enabled"
    ] and not normalized["innovation_aggregation"]["enabled"]:
        raise ValueError(
            "Open-DCSI geometric_clustering requires innovation_aggregation"
        )
    if normalized["dense_innovation_map"]["enabled"] and not normalized[
        "common_space"
    ]["enabled"]:
        raise ValueError("Open-DCSI dense_innovation_map requires common_space")
    if normalized["dense_innovation_map"]["enabled"] and normalized[
        "innovation_tokens"
    ]["enabled"]:
        raise ValueError(
            "Open-DCSI dense_innovation_map and innovation_tokens are alternative representations"
        )
    if normalized["dense_innovation_map"]["enabled"] and normalized[
        "streaming_fusion"
    ]["enabled"]:
        raise ValueError(
            "Open-DCSI dense_innovation_map is the non-streaming dense comparator"
        )
    if normalized["stage2_independent"]["enabled"] and not normalized[
        "open_heterogeneous"
    ]["enabled"]:
        raise ValueError(
            "Open-DCSI stage2_independent requires open_heterogeneous"
        )
    if normalized["communication"]["selection"]["enabled"] and not normalized[
        "innovation_tokens"
    ]["enabled"]:
        raise ValueError("Open-DCSI token selection requires innovation_tokens")

    loss_dependencies = {
        "common_detection": "common_space.common_detection_supervision",
        "reconstruction": "common_space.reconstruction",
        "common_innovation_decorrelation": "common_space",
        "innovation_detection": "innovation_tokens",
        "box_refinement": "geometry_refiner",
        "quality": "innovation_quality",
        "token_sparsity": "innovation_tokens",
        "budget": "innovation_tokens",
    }
    if normalized["losses"]["enabled"]:
        for loss_name, dependency in loss_dependencies.items():
            if normalized["losses"][loss_name]["enabled"] and not _get_path(
                normalized, dependency
            )["enabled"]:
                raise ValueError(
                    "Open-DCSI loss {} requires {}".format(loss_name, dependency)
                )

    if normalized["common_fusion"]["use_descriptor_conflict"]:
        raise ValueError(
            "Open-DCSI descriptor conflict requires independent supervision and ablation"
        )
    for path in (
        "common_fusion.absolute_reject.threshold",
        "innovation_aggregation.absolute_reject.threshold",
        "innovation_tokens.foreground_threshold",
        "innovation_tokens.nms_threshold",
        "common_fusion.absolute_reject.temperature",
        "innovation_aggregation.absolute_reject.temperature",
    ):
        _require_positive(normalized, path, allow_zero=True)

    for codec in ("common_codec", "token_codec"):
        precision = normalized["communication"][codec]["precision"].lower()
        if precision not in ("fp16", "int8"):
            raise ValueError(
                "Open-DCSI communication.{}.precision must be fp16 or int8".format(
                    codec
                )
            )
        normalized["communication"][codec]["precision"] = precision

    return normalized

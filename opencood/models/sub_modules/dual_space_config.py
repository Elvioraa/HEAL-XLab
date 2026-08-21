"""Strict configuration validation for the full Dual-Space framework.

Experiment profiles are labels only. Runtime behavior is controlled solely by
explicit feature switches and modes inside ``model.args.dual_space``.
"""

import copy
import math

VALID_MODES = ("stage1_anchor", "stage2_adapt", "inference")
VALID_CONSENSUS_MODES = ("uniform_geometry_mean", "quality_weighted")
VALID_MULTISCALE_FUSIONS = ("concat_projection", "adaptive_gate")
VALID_PROPOSAL_SOURCES = ("gt_jitter", "mixed")
VALID_YAW_MODES = ("sin_cos", "sin_cos_centered")
DEFAULT_DUAL_SPACE_DIAGNOSTICS = {
    "enabled": False,
    "every_n_steps": 100,
    "max_records": 2000,
    "match_iou_min": 0.3,
    "thresholds": (0.3, 0.5, 0.7),
    "improvement_epsilon": 1.0e-4,
    "save_per_object": False,
    "quality_target": {
        "enabled": False,
        "max_records": 500,
        "dump_jsonl": True,
    },
    "adapter_residual": {"enabled": False},
    "gradient_flow": {
        "enabled": False,
        "every_n_steps": 200,
    },
    "merge_ownership": {"enabled": False},
    "inference_ablation": {
        "enabled": False,
        "bypass_object_adapter": False,
        "bypass_context_adapter": False,
        "bypass_quality_weighting": False,
    },
}
DEFAULT_V5_QUALITY_SAFE = {
    "enabled": False,
    "valid_target_mask": {
        "enabled": False,
        "require_matched": True,
        "require_finite": True,
        "min_target": 0.0,
        "max_target": 1.0,
    },
    "loss_balance": {
        "enabled": False,
        "mode": "ratio_cap",
        "max_quality_to_detection_ratio": 0.25,
        "eps": 1.0e-6,
    },
    "ranking": {
        "enabled": False,
        "weight": 0.0,
        "min_target_gap": 0.10,
        "max_pairs_per_sample": 128,
        "loss": "logistic",
    },
}
DEFAULT_V6_RESIDUAL_SAFE = {
    "enabled": False,
    "object": {
        "enabled": False,
        "mode": "norm_cap",
        "max_residual_ratio": 0.25,
        "residual_scale": 1.0,
        "eps": 1.0e-6,
    },
    "context": {
        "enabled": False,
        "mode": "norm_cap",
        "max_residual_ratio": 0.25,
        "residual_scale": 1.0,
        "eps": 1.0e-6,
    },
}


def prepare_dual_space_inference_config(hypes):
    """Return inference-ready runtime hypes without mutating training config.

    Dual-Space Stage1 and Stage2 directories intentionally retain their
    training modes on disk. The inference entry point calls this function
    before model construction so both module trainability and checkpoint
    validation are derived from ``mode=inference``. Plain HEAL, explicitly
    disabled Dual-Space, and already-inference configurations are returned
    unchanged.
    """
    if not isinstance(hypes, dict):
        raise TypeError("hypes must be a mapping")
    model = hypes.get("model")
    if not isinstance(model, dict):
        return hypes
    args = model.get("args")
    if not isinstance(args, dict):
        return hypes
    config = args.get("dual_space")
    if config is None:
        return hypes
    validate_dual_space_config(config)
    if config.get("enabled") is not True or config["mode"] == "inference":
        return hypes

    runtime_hypes = copy.deepcopy(hypes)
    runtime_config = runtime_hypes["model"]["args"]["dual_space"]
    runtime_config["mode"] = "inference"
    runtime_config["allow_untrained_initialization"] = False
    runtime_config.pop("active_modality", None)
    validate_dual_space_config(runtime_config)
    return runtime_hypes


def validate_dual_space_config(config):
    """Validate a Dual-Space mapping without mutating or filling it in.

    ``None`` represents a legacy HEAL configuration and is legal. Optional
    sections may be omitted only while their feature is disabled; every
    enabled feature must provide its complete explicit contract.
    """
    if config is None:
        return None
    if not isinstance(config, dict):
        raise TypeError("model.args.dual_space must be a mapping")
    enabled = _optional_bool(config, "enabled", False, "dual_space.enabled")

    multi = _optional_mapping(config, "multi_scale")
    quality = _optional_mapping(config, "quality")
    rescue = _optional_mapping(config, "remote_proposal_rescue")
    diagnostics = _optional_mapping(config, "diagnostics")
    v5_quality_safe = _optional_mapping(config, "v5_quality_safe")
    v6_residual_safe = _optional_mapping(config, "v6_residual_safe")
    multi_enabled = _optional_bool(
        multi, "enabled", False, "dual_space.multi_scale.enabled"
    )
    quality_enabled = _optional_bool(
        quality, "enabled", False, "dual_space.quality.enabled"
    )
    rescue_enabled = _optional_bool(
        rescue, "enabled", False,
        "dual_space.remote_proposal_rescue.enabled",
    )
    diagnostics_enabled = _optional_bool(
        diagnostics, "enabled", False, "dual_space.diagnostics.enabled"
    )
    v5_enabled = _optional_bool(
        v5_quality_safe, "enabled", False,
        "dual_space.v5_quality_safe.enabled",
    )
    v6_enabled = _optional_bool(
        v6_residual_safe, "enabled", False,
        "dual_space.v6_residual_safe.enabled",
    )

    if not enabled:
        active = []
        if multi_enabled:
            active.append("multi_scale.enabled")
        if quality_enabled:
            active.append("quality.enabled")
        if rescue_enabled:
            active.append("remote_proposal_rescue.enabled")
        if diagnostics_enabled:
            active.append("diagnostics.enabled")
        if v5_enabled:
            active.append("v5_quality_safe.enabled")
        if v6_enabled:
            active.append("v6_residual_safe.enabled")
        fusion = multi.get("fusion")
        if fusion == "adaptive_gate":
            active.append("multi_scale.fusion=adaptive_gate")
        proposal_source = _nested_get(config, "training_proposals", "source")
        if proposal_source == "mixed":
            active.append("training_proposals.source=mixed")
        if active:
            raise ValueError(
                "invalid dependency: dual_space.enabled=false cannot enable %s; "
                "expected all Dual-Space features disabled"
                % ", ".join(active)
            )
        return config

    version = config.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("dual_space.version must be a non-empty string")
    profile = config.get("experiment_profile", version)
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError(
            "dual_space.experiment_profile must be a non-empty string"
        )
    mode = config.get("mode")
    if mode not in VALID_MODES:
        raise ValueError(
            "invalid dual_space.mode %r; expected one of %s"
            % (mode, VALID_MODES)
        )
    _required_bool(
        config, "allow_untrained_initialization",
        "dual_space.allow_untrained_initialization",
    )
    if mode == "stage2_adapt":
        if config.get("active_modality") not in ("m2", "m3", "m4"):
            raise ValueError(
                "dual_space.active_modality must be m2, m3, or m4 for Stage2"
            )
    elif config.get("active_modality") is not None:
        raise ValueError(
            "dual_space.active_modality is only valid for stage2_adapt"
        )
    roi = _required_mapping(config, "roi", "dual_space.roi")
    for key in (
        "output_size", "max_train_proposals", "max_infer_proposals",
        "chunk_size",
    ):
        _positive_int(roi.get(key), "dual_space.roi.%s" % key)
    _unit_interval(roi.get("min_coverage"), "dual_space.roi.min_coverage")

    adapter = _required_mapping(config, "adapter", "dual_space.adapter")
    if adapter.get("type") != "residual_1x1":
        raise ValueError(
            "invalid dual_space.adapter.type; expected residual_1x1"
        )
    if adapter.get("zero_init") is not True:
        raise ValueError("dual_space.adapter.zero_init must be true")

    encoder = _required_mapping(
        config, "object_encoder", "dual_space.object_encoder"
    )
    for key in ("embedding_dim", "hidden_channels", "pooled_size"):
        _positive_int(
            encoder.get(key), "dual_space.object_encoder.%s" % key
        )

    geometry = _required_mapping(
        config, "geometry_encoder", "dual_space.geometry_encoder"
    )
    if geometry.get("enabled") is not True:
        raise ValueError("dual_space.geometry_encoder.enabled must be true")
    _positive_int(
        geometry.get("hidden_dim"), "dual_space.geometry_encoder.hidden_dim"
    )

    refiner = _required_mapping(config, "refiner", "dual_space.refiner")
    _positive_int(refiner.get("hidden_dim"), "dual_space.refiner.hidden_dim")
    yaw_mode = refiner.get("yaw_mode")
    if yaw_mode not in VALID_YAW_MODES:
        raise ValueError(
            "invalid dual_space.refiner.yaw_mode %r; expected one of %s"
            % (yaw_mode, VALID_YAW_MODES)
        )
    if refiner.get("zero_init_output") is not True:
        raise ValueError("dual_space.refiner.zero_init_output must be true")

    if multi_enabled:
        detail = _required_mapping(
            multi, "detail", "dual_space.multi_scale.detail"
        )
        context = _required_mapping(
            multi, "context", "dual_space.multi_scale.context"
        )
        _positive_int(
            detail.get("roi_size"),
            "dual_space.multi_scale.detail.roi_size",
        )
        _positive_int(
            context.get("roi_size"),
            "dual_space.multi_scale.context.roi_size",
        )
        fusion = multi.get("fusion")
        if fusion not in VALID_MULTISCALE_FUSIONS:
            raise ValueError(
                "invalid dual_space.multi_scale.fusion %r; expected one of %s"
                % (fusion, VALID_MULTISCALE_FUSIONS)
            )
    elif "fusion" in multi:
        fusion = multi.get("fusion")
        if fusion not in VALID_MULTISCALE_FUSIONS:
            raise ValueError(
                "invalid dual_space.multi_scale.fusion %r; expected one of %s"
                % (fusion, VALID_MULTISCALE_FUSIONS)
            )
        if fusion == "adaptive_gate":
            raise ValueError(
                "invalid dependency: adaptive_gate requires "
                "dual_space.multi_scale.enabled=true"
            )

    if quality_enabled:
        if quality.get("target") != "refined_iou":
            raise ValueError(
                "dual_space.quality.target must be refined_iou"
            )
        _positive_int(
            quality.get("hidden_dim"), "dual_space.quality.hidden_dim"
        )
        for key in (
            "use_roi_coverage", "use_agent_distance", "detach_target",
            "detach_weight_for_consensus",
        ):
            _required_bool(
                quality, key, "dual_space.quality.%s" % key
            )
        if quality.get("detach_target") is not True:
            raise ValueError(
                "dual_space.quality.detach_target must be true for refined_iou"
            )

    _validate_diagnostics(diagnostics, mode, quality_enabled)
    _validate_v5_quality_safe(v5_quality_safe, quality_enabled)
    _validate_v6_residual_safe(v6_residual_safe)

    consensus = _required_mapping(
        config, "consensus", "dual_space.consensus"
    )
    consensus_mode = consensus.get("mode")
    if consensus_mode not in VALID_CONSENSUS_MODES:
        raise ValueError(
            "invalid dual_space.consensus.mode %r; expected one of %s"
            % (consensus_mode, VALID_CONSENSUS_MODES)
        )
    if consensus.get("fallback_to_original") is not True:
        raise ValueError(
            "dual_space.consensus.fallback_to_original must be true"
        )
    if consensus_mode == "quality_weighted":
        if not quality_enabled:
            raise ValueError(
                "invalid dependency: quality_weighted consensus requires "
                "dual_space.quality.enabled=true"
            )
        _positive_real(
            consensus.get("min_quality_sum"),
            "dual_space.consensus.min_quality_sum",
        )
        if consensus.get("low_quality_fallback") != "uniform":
            raise ValueError(
                "dual_space.consensus.low_quality_fallback must be uniform"
            )

    proposals = _required_mapping(
        config, "training_proposals", "dual_space.training_proposals"
    )
    source = proposals.get("source")
    if source not in VALID_PROPOSAL_SOURCES:
        raise ValueError(
            "invalid dual_space.training_proposals.source %r; expected one of %s"
            % (source, VALID_PROPOSAL_SOURCES)
        )
    _required_bool(
        proposals, "include_gt",
        "dual_space.training_proposals.include_gt",
    )
    _nonnegative_int(
        proposals.get("jitters_per_gt"),
        "dual_space.training_proposals.jitters_per_gt",
    )
    for key in (
        "center_xy_std_rel", "center_z_std_rel", "log_size_std",
        "yaw_std_deg",
    ):
        _nonnegative_real(
            proposals.get(key),
            "dual_space.training_proposals.%s" % key,
        )
    _positive_int(
        proposals.get("max_proposals"),
        "dual_space.training_proposals.max_proposals",
    )
    if proposals["max_proposals"] != roi["max_train_proposals"]:
        raise ValueError(
            "dual_space.training_proposals.max_proposals must equal "
            "dual_space.roi.max_train_proposals"
        )
    predicted = _optional_mapping(proposals, "predicted")
    predicted_enabled = _optional_bool(
        predicted, "enabled", False,
        "dual_space.training_proposals.predicted.enabled",
    )
    if source == "mixed":
        if not predicted_enabled:
            raise ValueError(
                "invalid dependency: training_proposals.source=mixed requires "
                "training_proposals.predicted.enabled=true"
            )
        _positive_int(
            predicted.get("max_per_scene"),
            "dual_space.training_proposals.predicted.max_per_scene",
        )
        min_score = predicted.get("min_score")
        if min_score is not None:
            _unit_interval(
                min_score,
                "dual_space.training_proposals.predicted.min_score",
                allow_zero=True,
            )
        _unit_interval(
            predicted.get("positive_iou_min"),
            "dual_space.training_proposals.predicted.positive_iou_min",
            allow_zero=True,
        )
    elif predicted_enabled:
        raise ValueError(
            "invalid dependency: training_proposals.predicted.enabled=true "
            "requires training_proposals.source=mixed"
        )

    if rescue_enabled:
        if mode != "inference":
            raise ValueError(
                "invalid dependency: remote_proposal_rescue is an inference-only "
                "candidate policy; expected dual_space.mode=inference"
            )
        _required_bool(
            rescue, "include_ego",
            "dual_space.remote_proposal_rescue.include_ego",
        )
        for key in ("min_score", "dedup_iou"):
            _unit_interval(
                rescue.get(key),
                "dual_space.remote_proposal_rescue.%s" % key,
                allow_zero=True,
            )
        for key in ("max_per_agent", "max_total_added"):
            _positive_int(
                rescue.get(key),
                "dual_space.remote_proposal_rescue.%s" % key,
            )

    loss = _required_mapping(config, "loss", "dual_space.loss")
    for key in (
        "object_loss_weight", "individual_loss_weight",
        "consensus_loss_weight", "iou_loss_weight",
    ):
        _nonnegative_real(loss.get(key), "dual_space.loss.%s" % key)
    if float(loss["iou_loss_weight"]) != 0.0:
        raise ValueError("dual_space.loss.iou_loss_weight must remain 0.0")
    if quality_enabled or "quality_loss_weight" in loss:
        _nonnegative_real(
            loss.get("quality_loss_weight"),
            "dual_space.loss.quality_loss_weight",
        )
    _optional_bool(
        config, "report_stats", False, "dual_space.report_stats"
    )
    return config


def dual_space_feature_flags(config):
    """Return explicit runtime feature flags for a validated configuration."""
    if config is None or config.get("enabled") is not True:
        return {
            "enabled": False,
            "multi_scale": False,
            "quality": False,
            "remote_proposal_rescue": False,
            "mixed_proposals": False,
            "report_stats": False,
            "diagnostics": False,
            "v5_quality_safe": False,
            "v6_residual_safe": False,
        }
    mode = config["mode"]
    return {
        "enabled": True,
        "multi_scale": bool(config.get("multi_scale", {}).get("enabled", False)),
        "quality": bool(config.get("quality", {}).get("enabled", False)),
        "remote_proposal_rescue": bool(
            config.get("remote_proposal_rescue", {}).get("enabled", False)
        ),
        "mixed_proposals": config["training_proposals"]["source"] == "mixed",
        "report_stats": bool(config.get("report_stats", False)),
        "diagnostics": bool(
            config.get("diagnostics", {}).get("enabled", False)
        ),
        "v5_quality_safe": bool(
            mode == "stage2_adapt"
            and config.get("v5_quality_safe", {}).get("enabled", False)
        ),
        "v6_residual_safe": bool(
            mode in ("stage2_adapt", "inference")
            and config.get("v6_residual_safe", {}).get("enabled", False)
        ),
    }


def resolve_dual_space_diagnostics(config):
    """Return the complete observer config without mutating the input mapping."""
    resolved = copy.deepcopy(DEFAULT_DUAL_SPACE_DIAGNOSTICS)
    if config is None:
        resolved["thresholds"] = list(resolved["thresholds"])
        return resolved
    diagnostics = config.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise TypeError("dual_space.diagnostics must be a mapping")
    _merge_nested_defaults(resolved, diagnostics)
    resolved["thresholds"] = list(resolved["thresholds"])
    return resolved


def resolve_v5_quality_safe_config(config):
    """Return complete parameter-free Stage2 quality-safety settings."""
    resolved = copy.deepcopy(DEFAULT_V5_QUALITY_SAFE)
    if config is not None:
        value = config.get("v5_quality_safe", {})
        if not isinstance(value, dict):
            raise TypeError("dual_space.v5_quality_safe must be a mapping")
        _merge_nested_defaults(resolved, value)
    return resolved


def resolve_v6_residual_safe_config(config):
    """Return complete parameter-free adapter residual-safety settings."""
    resolved = copy.deepcopy(DEFAULT_V6_RESIDUAL_SAFE)
    if config is not None:
        value = config.get("v6_residual_safe", {})
        if not isinstance(value, dict):
            raise TypeError("dual_space.v6_residual_safe must be a mapping")
        _merge_nested_defaults(resolved, value)
    return resolved


def _validate_diagnostics(diagnostics, mode, quality_enabled):
    enabled = _optional_bool(
        diagnostics, "enabled", False, "dual_space.diagnostics.enabled"
    )
    _positive_int(
        diagnostics.get(
            "every_n_steps", DEFAULT_DUAL_SPACE_DIAGNOSTICS["every_n_steps"]
        ),
        "dual_space.diagnostics.every_n_steps",
    )
    _positive_int(
        diagnostics.get(
            "max_records", DEFAULT_DUAL_SPACE_DIAGNOSTICS["max_records"]
        ),
        "dual_space.diagnostics.max_records",
    )
    match_iou_min = diagnostics.get(
        "match_iou_min", DEFAULT_DUAL_SPACE_DIAGNOSTICS["match_iou_min"]
    )
    _unit_interval(
        match_iou_min, "dual_space.diagnostics.match_iou_min", allow_zero=True
    )
    thresholds = diagnostics.get(
        "thresholds", DEFAULT_DUAL_SPACE_DIAGNOSTICS["thresholds"]
    )
    if not isinstance(thresholds, (list, tuple)) or not thresholds:
        raise ValueError(
            "dual_space.diagnostics.thresholds must be a non-empty sequence"
        )
    normalized = []
    for index, threshold in enumerate(thresholds):
        normalized.append(
            _unit_interval(
                threshold,
                "dual_space.diagnostics.thresholds[%d]" % index,
                allow_zero=True,
            )
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("dual_space.diagnostics.thresholds must be unique")
    if normalized != sorted(normalized):
        raise ValueError("dual_space.diagnostics.thresholds must be sorted")
    _nonnegative_real(
        diagnostics.get(
            "improvement_epsilon",
            DEFAULT_DUAL_SPACE_DIAGNOSTICS["improvement_epsilon"],
        ),
        "dual_space.diagnostics.improvement_epsilon",
    )
    save_per_object = diagnostics.get(
        "save_per_object", DEFAULT_DUAL_SPACE_DIAGNOSTICS["save_per_object"]
    )
    if type(save_per_object) is not bool:
        raise TypeError("dual_space.diagnostics.save_per_object must be bool")

    quality_target = _optional_mapping(diagnostics, "quality_target")
    quality_target_enabled = _optional_bool(
        quality_target, "enabled", False,
        "dual_space.diagnostics.quality_target.enabled",
    )
    _positive_int(
        quality_target.get(
            "max_records",
            DEFAULT_DUAL_SPACE_DIAGNOSTICS["quality_target"]["max_records"],
        ),
        "dual_space.diagnostics.quality_target.max_records",
    )
    _optional_bool(
        quality_target,
        "dump_jsonl",
        DEFAULT_DUAL_SPACE_DIAGNOSTICS["quality_target"]["dump_jsonl"],
        "dual_space.diagnostics.quality_target.dump_jsonl",
    )
    if enabled and quality_target_enabled and not quality_enabled:
        raise ValueError(
            "quality-target diagnostics require dual_space.quality.enabled=true"
        )

    adapter = _optional_mapping(diagnostics, "adapter_residual")
    _optional_bool(
        adapter, "enabled", False,
        "dual_space.diagnostics.adapter_residual.enabled",
    )
    gradient = _optional_mapping(diagnostics, "gradient_flow")
    _optional_bool(
        gradient, "enabled", False,
        "dual_space.diagnostics.gradient_flow.enabled",
    )
    _positive_int(
        gradient.get(
            "every_n_steps",
            DEFAULT_DUAL_SPACE_DIAGNOSTICS["gradient_flow"]["every_n_steps"],
        ),
        "dual_space.diagnostics.gradient_flow.every_n_steps",
    )
    ownership = _optional_mapping(diagnostics, "merge_ownership")
    _optional_bool(
        ownership, "enabled", False,
        "dual_space.diagnostics.merge_ownership.enabled",
    )
    ablation = _optional_mapping(diagnostics, "inference_ablation")
    ablation_enabled = _optional_bool(
        ablation, "enabled", False,
        "dual_space.diagnostics.inference_ablation.enabled",
    )
    for key in (
        "bypass_object_adapter", "bypass_context_adapter",
        "bypass_quality_weighting",
    ):
        _optional_bool(
            ablation,
            key,
            False,
            "dual_space.diagnostics.inference_ablation.%s" % key,
        )
    if enabled and ablation_enabled and mode != "inference":
        raise ValueError(
            "diagnostic inference ablation requires dual_space.mode=inference"
        )


def _validate_v5_quality_safe(config, quality_enabled):
    enabled = _optional_bool(
        config, "enabled", False, "dual_space.v5_quality_safe.enabled"
    )
    if enabled and not quality_enabled:
        raise ValueError(
            "v5_quality_safe requires dual_space.quality.enabled=true"
        )
    valid = _optional_mapping(config, "valid_target_mask")
    _optional_bool(
        valid, "enabled", False,
        "dual_space.v5_quality_safe.valid_target_mask.enabled",
    )
    for key, default in (("require_matched", True), ("require_finite", True)):
        _optional_bool(
            valid, key, default,
            "dual_space.v5_quality_safe.valid_target_mask.%s" % key,
        )
    minimum = valid.get("min_target", 0.0)
    maximum = valid.get("max_target", 1.0)
    _finite_real(
        minimum,
        "dual_space.v5_quality_safe.valid_target_mask.min_target",
    )
    _finite_real(
        maximum,
        "dual_space.v5_quality_safe.valid_target_mask.max_target",
    )
    if float(minimum) > float(maximum):
        raise ValueError("v5 valid_target_mask min_target must not exceed max_target")

    balance = _optional_mapping(config, "loss_balance")
    balance_enabled = _optional_bool(
        balance, "enabled", False,
        "dual_space.v5_quality_safe.loss_balance.enabled",
    )
    mode = balance.get("mode", "ratio_cap")
    if balance_enabled and mode != "ratio_cap":
        raise ValueError("v5 loss_balance.mode must be ratio_cap")
    _positive_real(
        balance.get("max_quality_to_detection_ratio", 0.25),
        "dual_space.v5_quality_safe.loss_balance.max_quality_to_detection_ratio",
    )
    _positive_real(
        balance.get("eps", 1.0e-6),
        "dual_space.v5_quality_safe.loss_balance.eps",
    )

    ranking = _optional_mapping(config, "ranking")
    ranking_enabled = _optional_bool(
        ranking, "enabled", False,
        "dual_space.v5_quality_safe.ranking.enabled",
    )
    _nonnegative_real(
        ranking.get("weight", 0.0),
        "dual_space.v5_quality_safe.ranking.weight",
    )
    _nonnegative_real(
        ranking.get("min_target_gap", 0.10),
        "dual_space.v5_quality_safe.ranking.min_target_gap",
    )
    _positive_int(
        ranking.get("max_pairs_per_sample", 128),
        "dual_space.v5_quality_safe.ranking.max_pairs_per_sample",
    )
    loss = ranking.get("loss", "logistic")
    if ranking_enabled and loss != "logistic":
        raise ValueError("v5 ranking.loss must be logistic")


def _validate_v6_residual_safe(config):
    _optional_bool(
        config, "enabled", False, "dual_space.v6_residual_safe.enabled"
    )
    for branch in ("object", "context"):
        value = _optional_mapping(config, branch)
        branch_enabled = _optional_bool(
            value, "enabled", False,
            "dual_space.v6_residual_safe.%s.enabled" % branch,
        )
        mode = value.get("mode", "norm_cap")
        if branch_enabled and mode != "norm_cap":
            raise ValueError("v6 %s.mode must be norm_cap" % branch)
        _positive_real(
            value.get("max_residual_ratio", 0.25),
            "dual_space.v6_residual_safe.%s.max_residual_ratio" % branch,
        )
        _nonnegative_real(
            value.get("residual_scale", 1.0),
            "dual_space.v6_residual_safe.%s.residual_scale" % branch,
        )
        _positive_real(
            value.get("eps", 1.0e-6),
            "dual_space.v6_residual_safe.%s.eps" % branch,
        )


def _merge_nested_defaults(destination, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _merge_nested_defaults(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)
    return destination


def _required_mapping(config, key, name):
    value = config.get(key)
    if not isinstance(value, dict):
        raise TypeError("%s must be a mapping" % name)
    return value


def _optional_mapping(config, key):
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise TypeError("dual_space.%s must be a mapping" % key)
    return value


def _nested_get(config, outer, inner):
    value = config.get(outer)
    return value.get(inner) if isinstance(value, dict) else None


def _required_bool(config, key, name):
    if key not in config or type(config[key]) is not bool:
        raise TypeError("%s must be bool" % name)
    return config[key]


def _optional_bool(config, key, default, name):
    value = config.get(key, default)
    if type(value) is not bool:
        raise TypeError("%s must be bool" % name)
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return value


def _positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a real number" % name)
    if float(value) <= 0.0:
        raise ValueError("%s must be positive" % name)
    return float(value)


def _nonnegative_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a real number" % name)
    if not math.isfinite(float(value)):
        raise ValueError("%s must be finite" % name)
    if float(value) < 0.0:
        raise ValueError("%s must be non-negative" % name)
    return float(value)


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a real number" % name)
    if not math.isfinite(float(value)):
        raise ValueError("%s must be finite" % name)
    return float(value)


def _unit_interval(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a real number" % name)
    value = float(value)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value > 1.0:
        interval = "[0,1]" if allow_zero else "(0,1]"
        raise ValueError("%s must be in %s" % (name, interval))
    return value

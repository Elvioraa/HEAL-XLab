"""Safe configuration helpers for HEAL-XLab."""

import copy


_DEFAULT_XLAB_CFG = {
    "enabled": False,
    "method": "hbec",
    "debug": True,
    "debug_dir": "xlab_debug",
    "hbec": {
        "enabled": False,
        "apply_stage": "final_infer_postprocess",
        "evidence_source": "official_or_fallback",
        "target_modalities": ["m2", "m4", "camera"],
        "base_uncertainty": 1.0,
        "min_score_for_uncertainty": 0.05,
        "match": {
            "iou_threshold": 0.1,
            "center_dist_threshold": 2.0,
            "iou_weight": 0.7,
            "dist_weight": 0.3,
            "dist_scale": 2.0,
        },
        "refine": {
            "enabled": True,
            "refine_strength": 0.5,
            "evidence_weight": 0.5,
        },
        "novel": {
            "enabled": True,
            "novel_score_threshold": 0.6,
            "novel_dist_threshold": 2.0,
            "max_novel": 20,
        },
        "suppress": {
            "enabled": False,
            "suppress_score_threshold": 0.3,
            "suppress_factor": 1.0,
        },
        "safety": {
            "fallback_on_error": True,
            "max_boxes_after_fusion": 300,
            "require_no_gt_for_fusion": True,
        },
    },
}


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def safe_get_xlab_cfg(hypes):
    """Return a complete XLab config with conservative defaults."""
    cfg = copy.deepcopy(_DEFAULT_XLAB_CFG)
    if not isinstance(hypes, dict):
        return cfg
    user_cfg = hypes.get("xlab", {})
    if not isinstance(user_cfg, dict):
        return cfg
    _deep_update(cfg, user_cfg)
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["method"] = cfg.get("method") or "hbec"
    hbec_cfg = cfg.setdefault("hbec", {})
    hbec_cfg["enabled"] = bool(hbec_cfg.get("enabled", False))
    return cfg


def hbec_is_enabled(xlab_cfg):
    """True only when both global XLab and HBEC switches are enabled."""
    return bool(
        xlab_cfg.get("enabled", False)
        and xlab_cfg.get("method", "hbec") == "hbec"
        and xlab_cfg.get("hbec", {}).get("enabled", False)
    )


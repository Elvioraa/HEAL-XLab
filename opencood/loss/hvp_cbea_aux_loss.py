"""Auxiliary loss plumbing for HVP-CBEA."""

import torch
import torch.nn.functional as F


def default_hvp_aux_loss_cfg():
    return {
        "enabled": False,
        "debug": False,
        "residual_reg": {
            "enabled": False,
            "weight": 0.001,
            "type": "l1",
        },
        "alpha_reg": {
            "enabled": False,
            "weight": 0.001,
            "target": 0.05,
        },
        "refinement_consistency": {
            "enabled": False,
            "weight": 0.05,
            "mode": "feature_delta_l1",
        },
        "gt_guided": {
            "enabled": False,
            "debug": False,
            "hypothesis_heatmap": {
                "enabled": False,
                "weight": 0.05,
                "source": "anchor_pos",
                "loss": "bce",
                "pos_weight": 2.0,
            },
            "residual_focus": {
                "enabled": False,
                "weight": 0.01,
                "source": "anchor_pos",
                "bg_weight": 1.0,
                "fg_weight": 0.25,
            },
            "residual_fg_boost": {
                "enabled": False,
                "weight": 0.005,
                "source": "anchor_pos",
                "target": 0.0,
            },
        },
    }


def normalize_hvp_aux_loss_cfg(cfg):
    normalized = default_hvp_aux_loss_cfg()
    if isinstance(cfg, bool):
        normalized["enabled"] = cfg
    elif isinstance(cfg, dict):
        _deep_update(normalized, cfg)
    normalized["enabled"] = bool(
        normalized.get("enabled", False) or normalized.get("enable", False)
    )
    normalized["debug"] = bool(normalized.get("debug", False))

    residual = normalized["residual_reg"]
    residual["enabled"] = bool(residual.get("enabled", False))
    residual["weight"] = float(residual.get("weight", 0.001))
    residual["type"] = str(residual.get("type", "l1")).lower()

    alpha = normalized["alpha_reg"]
    alpha["enabled"] = bool(alpha.get("enabled", False))
    alpha["weight"] = float(alpha.get("weight", 0.001))
    alpha["target"] = float(alpha.get("target", 0.05))

    consistency = normalized["refinement_consistency"]
    consistency["enabled"] = bool(consistency.get("enabled", False))
    consistency["weight"] = float(consistency.get("weight", 0.05))
    consistency["mode"] = str(consistency.get("mode", "feature_delta_l1")).lower()

    if isinstance(normalized.get("gt_guided"), bool):
        enabled = bool(normalized["gt_guided"])
        normalized["gt_guided"] = default_hvp_aux_loss_cfg()["gt_guided"]
        normalized["gt_guided"]["enabled"] = enabled
    elif not isinstance(normalized.get("gt_guided"), dict):
        normalized["gt_guided"] = default_hvp_aux_loss_cfg()["gt_guided"]

    gt_guided = normalized["gt_guided"]
    gt_guided["enabled"] = bool(gt_guided.get("enabled", False))
    gt_guided["debug"] = bool(gt_guided.get("debug", False))

    for key in ("hypothesis_heatmap", "residual_focus", "residual_fg_boost"):
        if isinstance(gt_guided.get(key), bool):
            enabled = bool(gt_guided[key])
            gt_guided[key] = default_hvp_aux_loss_cfg()["gt_guided"][key]
            gt_guided[key]["enabled"] = enabled
        elif not isinstance(gt_guided.get(key), dict):
            gt_guided[key] = default_hvp_aux_loss_cfg()["gt_guided"][key]

    heatmap = gt_guided["hypothesis_heatmap"]
    heatmap["enabled"] = bool(
        heatmap.get("enabled", False) or heatmap.get("enable", False)
    )
    heatmap["weight"] = float(heatmap.get("weight", 0.05))
    heatmap["source"] = str(heatmap.get("source", "anchor_pos"))
    heatmap["loss"] = str(heatmap.get("loss", "bce")).lower()
    heatmap["pos_weight"] = float(heatmap.get("pos_weight", 2.0))

    focus = gt_guided["residual_focus"]
    focus["enabled"] = bool(focus.get("enabled", False))
    focus["weight"] = float(focus.get("weight", 0.01))
    focus["source"] = str(focus.get("source", "anchor_pos"))
    focus["bg_weight"] = float(focus.get("bg_weight", 1.0))
    focus["fg_weight"] = float(focus.get("fg_weight", 0.25))

    fg_boost = gt_guided["residual_fg_boost"]
    fg_boost["enabled"] = bool(fg_boost.get("enabled", False))
    fg_boost["weight"] = float(fg_boost.get("weight", 0.005))
    fg_boost["source"] = str(fg_boost.get("source", "anchor_pos"))
    fg_boost["target"] = float(fg_boost.get("target", 0.0))
    return normalized


def compute_hvp_auxiliary_loss(aux_dict, aux_cfg=None, target_dict=None,
                               fallback_on_error=True):
    cfg = normalize_hvp_aux_loss_cfg(aux_cfg or (aux_dict or {}).get("config"))
    ref_tensor = _find_ref_tensor(aux_dict)
    zero = _zero_like(ref_tensor)
    stats = _zero_stats()
    stats["hvp_aux_enabled"] = bool(cfg["enabled"])
    if not cfg["enabled"]:
        return zero, stats

    try:
        if not isinstance(aux_dict, dict):
            raise ValueError("hvp_cbea_aux is missing")
        losses = []

        residual_cfg = cfg["residual_reg"]
        if residual_cfg["enabled"]:
            hvp_residual = _require_tensor(aux_dict, "hvp_residual")
            residual_loss = _feature_penalty(hvp_residual, residual_cfg["type"])
            weighted = residual_loss * residual_cfg["weight"]
            losses.append(weighted)
            stats["hvp_residual_reg_loss"] = _item(weighted)

        alpha_cfg = cfg["alpha_reg"]
        if alpha_cfg["enabled"]:
            alpha = aux_dict.get("alpha")
            if not torch.is_tensor(alpha):
                alpha = _require_tensor(aux_dict, "effective_alpha")
            alpha_loss = torch.mean((alpha.float() - alpha_cfg["target"]) ** 2)
            weighted = alpha_loss * alpha_cfg["weight"]
            losses.append(weighted)
            stats["hvp_alpha_reg_loss"] = _item(weighted)

        consistency_cfg = cfg["refinement_consistency"]
        if consistency_cfg["enabled"]:
            delta_feature = _require_tensor(aux_dict, "delta_feature")
            consistency_loss = _feature_penalty(delta_feature, "l1")
            weighted = consistency_loss * consistency_cfg["weight"]
            losses.append(weighted)
            stats["hvp_refinement_consistency_loss"] = _item(weighted)

        gt_loss, gt_stats = _compute_gt_guided_loss(
            aux_dict,
            target_dict,
            cfg["gt_guided"],
            fallback_on_error=fallback_on_error,
        )
        if torch.is_tensor(gt_loss):
            losses.append(gt_loss)
        stats.update(gt_stats)

        total = sum(losses, zero)
        if not torch.isfinite(total):
            raise ValueError("hvp auxiliary loss is not finite")
        stats["hvp_aux_total_loss"] = _item(total)
        return total, stats
    except Exception as exc:
        if not fallback_on_error:
            raise
        stats["hvp_aux_fallback_reason"] = type(exc).__name__
        return zero, stats


def default_hvp_v3_stage1_aux_loss_cfg():
    return {
        "enabled": False,
        "mode": "stage1_hypothesis",
        "hypothesis_heatmap": {
            "enabled": False,
            "weight": 0.01,
            "pos_weight": 1.0,
        },
        "residual_reg": {
            "enabled": False,
        },
        "alpha_reg": {
            "enabled": False,
        },
        "residual_focus": {
            "enabled": False,
        },
    }


def normalize_hvp_v3_stage1_aux_loss_cfg(cfg):
    normalized = default_hvp_v3_stage1_aux_loss_cfg()
    if isinstance(cfg, bool):
        normalized["enabled"] = cfg
    elif isinstance(cfg, dict):
        _deep_update(normalized, cfg)
    normalized["enabled"] = bool(normalized.get("enabled", False))
    normalized["mode"] = str(normalized.get("mode", "stage1_hypothesis"))
    heatmap = normalized["hypothesis_heatmap"]
    heatmap["enabled"] = bool(heatmap.get("enabled", False))
    heatmap["weight"] = float(heatmap.get("weight", 0.01))
    heatmap["pos_weight"] = float(heatmap.get("pos_weight", 1.0))
    for key in ("residual_reg", "alpha_reg", "residual_focus"):
        if isinstance(normalized.get(key), bool):
            enabled = bool(normalized[key])
            normalized[key] = {"enabled": enabled}
        elif not isinstance(normalized.get(key), dict):
            normalized[key] = {"enabled": False}
        normalized[key]["enabled"] = bool(normalized[key].get("enabled", False))
    return normalized


def compute_hvp_v3_stage1_loss(hvp_v3_dict, target_dict=None,
                               fallback_on_error=True):
    cfg = normalize_hvp_v3_stage1_aux_loss_cfg(
        (hvp_v3_dict or {}).get("aux_loss_cfg")
    )
    ref_tensor = None
    if isinstance(hvp_v3_dict, dict):
        ref_tensor = hvp_v3_dict.get("hypothesis_heatmap_logits")
        if not torch.is_tensor(ref_tensor):
            ref_tensor = hvp_v3_dict.get("hypothesis_heatmap")
    zero = _zero_like(ref_tensor)
    stats = _zero_hvp_v3_stats()
    stats["hvp_v3_enabled"] = bool(isinstance(hvp_v3_dict, dict) and hvp_v3_dict.get("enabled", False))
    stats["hvp_v3_stage"] = (hvp_v3_dict or {}).get("stage", "")
    if not cfg["enabled"]:
        return zero, stats
    if cfg["mode"] != "stage1_hypothesis":
        stats["hvp_v3_fallback_reason"] = "unsupported_mode:%s" % cfg["mode"]
        return zero, stats
    if not cfg["hypothesis_heatmap"].get("enabled", False):
        return zero, stats

    try:
        if not isinstance(hvp_v3_dict, dict):
            raise ValueError("hvp_v3 output is missing")
        logits = hvp_v3_dict.get("hypothesis_heatmap_logits")
        if not torch.is_tensor(logits):
            raise ValueError("hvp_v3 hypothesis_heatmap_logits is missing")
        pos_map, source = _extract_anchor_positive_map(target_dict, logits)
        target = _resize_positive_map(pos_map, logits)
        heatmap_cfg = cfg["hypothesis_heatmap"]
        pos_weight = logits.new_tensor(heatmap_cfg.get("pos_weight", 1.0))
        loss = F.binary_cross_entropy_with_logits(
            logits.float(),
            target.float(),
            pos_weight=pos_weight,
        )
        weighted = loss * heatmap_cfg["weight"]
        if not torch.isfinite(weighted):
            raise ValueError("HVP-v3 Stage1 hypothesis loss is not finite")
        stats.update({
            "hvp_v3_loss": _item(weighted),
            "hvp_v3_stage1_hypothesis_loss": _item(weighted),
            "hvp_v3_target_source": source,
            "hvp_v3_fg_ratio": _item(target.float().mean()),
            "hvp_v3_fallback_reason": "",
        })
        return weighted, stats
    except Exception as exc:
        if not fallback_on_error:
            raise ValueError("HVP-v3 Stage1 hypothesis loss failed: %s" % exc) from exc
        stats["hvp_v3_fallback_reason"] = "%s:%s" % (type(exc).__name__, str(exc))
        return zero, stats


def default_hvp_v3_stage2_evidence_loss_cfg():
    return {
        "enabled": False,
        "mode": "stage2_evidence",
        "evidence_heatmap": {
            "enabled": False,
            "weight": 0.01,
            "pos_weight": 1.0,
        },
        "uncertainty": {
            "enabled": False,
            "weight": 0.001,
        },
        "descriptor": {
            "enabled": False,
            "weight": 0.001,
        },
        "localization_uncertainty": {
            "enabled": False,
            "weight": 0.001,
        },
    }


def normalize_hvp_v3_stage2_evidence_loss_cfg(cfg):
    normalized = default_hvp_v3_stage2_evidence_loss_cfg()
    if isinstance(cfg, bool):
        normalized["enabled"] = cfg
    elif isinstance(cfg, dict):
        _deep_update(normalized, cfg)
    normalized["enabled"] = bool(normalized.get("enabled", False))
    normalized["mode"] = str(normalized.get("mode", "stage2_evidence"))
    heatmap = normalized["evidence_heatmap"]
    heatmap["enabled"] = bool(heatmap.get("enabled", False))
    heatmap["weight"] = float(heatmap.get("weight", 0.01))
    heatmap["pos_weight"] = float(heatmap.get("pos_weight", 1.0))
    uncertainty = normalized["uncertainty"]
    uncertainty["enabled"] = bool(
        uncertainty.get("enabled", False) or uncertainty.get("enable", False)
    )
    uncertainty["weight"] = float(uncertainty.get("weight", 0.001))
    descriptor = normalized["descriptor"]
    descriptor["enabled"] = bool(
        descriptor.get("enabled", False) or descriptor.get("enable", False)
    )
    descriptor["weight"] = float(descriptor.get("weight", 0.001))
    loc_unc = normalized["localization_uncertainty"]
    loc_unc["enabled"] = bool(
        loc_unc.get("enabled", False) or loc_unc.get("enable", False)
    )
    loc_unc["weight"] = float(loc_unc.get("weight", 0.001))
    return normalized


def compute_hvp_v3_stage2_evidence_loss(hvp_v3_dict, target_dict=None,
                                        fallback_on_error=True):
    cfg = normalize_hvp_v3_stage2_evidence_loss_cfg(
        (hvp_v3_dict or {}).get("evidence_loss_cfg")
    )
    ref_tensor = None
    if isinstance(hvp_v3_dict, dict):
        ref_tensor = hvp_v3_dict.get("evidence_heatmap_logits")
        if not torch.is_tensor(ref_tensor):
            ref_tensor = hvp_v3_dict.get("evidence_heatmap")
    zero = _zero_like(ref_tensor)
    stats = _zero_hvp_v3_stats()
    stats["hvp_v3_enabled"] = bool(
        isinstance(hvp_v3_dict, dict) and hvp_v3_dict.get("enabled", False)
    )
    stats["hvp_v3_stage"] = (hvp_v3_dict or {}).get("stage", "")
    if not cfg["enabled"]:
        return zero, stats
    if cfg["mode"] != "stage2_evidence":
        stats["hvp_v3_fallback_reason"] = "unsupported_mode:%s" % cfg["mode"]
        return zero, stats

    try:
        if not isinstance(hvp_v3_dict, dict):
            raise ValueError("hvp_v3 output is missing")
        logits = hvp_v3_dict.get("evidence_heatmap_logits")
        if not torch.is_tensor(logits):
            raise ValueError("hvp_v3 evidence_heatmap_logits is missing")
        pos_map, source = _extract_anchor_positive_map(target_dict, logits)
        target = _resize_positive_map(pos_map, logits)
        losses = []

        heatmap_cfg = cfg["evidence_heatmap"]
        if heatmap_cfg.get("enabled", False):
            pos_weight = logits.new_tensor(heatmap_cfg.get("pos_weight", 1.0))
            heatmap_loss = F.binary_cross_entropy_with_logits(
                logits.float(),
                target.float(),
                pos_weight=pos_weight,
            )
            weighted = heatmap_loss * heatmap_cfg["weight"]
            losses.append(weighted)
            stats["hvp_v3_stage2_evidence_heatmap_loss"] = _item(weighted)

        uncertainty_cfg = cfg["uncertainty"]
        if uncertainty_cfg.get("enabled", False):
            uncertainty = _require_tensor(hvp_v3_dict, "evidence_uncertainty")
            fg_mask = _resize_positive_map(target, uncertainty)
            denom = torch.clamp(fg_mask.sum(), min=1.0)
            uncertainty_loss = (uncertainty.float() * fg_mask.float()).sum() / denom
            weighted = uncertainty_loss * uncertainty_cfg["weight"]
            losses.append(weighted)
            stats["hvp_v3_stage2_uncertainty_loss"] = _item(weighted)

        descriptor_cfg = cfg["descriptor"]
        if descriptor_cfg.get("enabled", False):
            descriptor = _require_tensor(hvp_v3_dict, "evidence_descriptor")
            descriptor_loss = _descriptor_smoothness_loss(descriptor)
            weighted = descriptor_loss * descriptor_cfg["weight"]
            losses.append(weighted)
            stats["hvp_v3_stage2_descriptor_loss"] = _item(weighted)

        total = sum(losses, zero)
        if not torch.isfinite(total):
            raise ValueError("HVP-v3 Stage2 evidence loss is not finite")
        stats.update({
            "hvp_v3_loss": _item(total),
            "hvp_v3_stage2_evidence_loss": _item(total),
            "hvp_v3_target_source": source,
            "hvp_v3_fg_ratio": _item(target.float().mean()),
            "hvp_v3_fallback_reason": "",
        })
        return total, stats
    except Exception as exc:
        if not fallback_on_error:
            raise ValueError("HVP-v3 Stage2 evidence loss failed: %s" % exc) from exc
        stats["hvp_v3_fallback_reason"] = "%s:%s" % (type(exc).__name__, str(exc))
        return zero, stats


def compute_hvp_v3_loss(hvp_v3_dict, target_dict=None, fallback_on_error=True):
    stage = (hvp_v3_dict or {}).get("stage", "")
    if stage == "stage2_evidence":
        return compute_hvp_v3_stage2_evidence_loss(
            hvp_v3_dict,
            target_dict=target_dict,
            fallback_on_error=fallback_on_error,
        )
    return compute_hvp_v3_stage1_loss(
        hvp_v3_dict,
        target_dict=target_dict,
        fallback_on_error=fallback_on_error,
    )


def compute_pact_cbea_local_evidence_loss(pact_dict, target_dict=None,
                                          fallback_on_error=True, reg_preds=None):
    cfg = normalize_hvp_v3_stage2_evidence_loss_cfg(
        (pact_dict or {}).get("evidence_loss_cfg")
    )
    ref_tensor = None
    if isinstance(pact_dict, dict):
        ref_tensor = pact_dict.get("evidence_heatmap_logits")
        if not torch.is_tensor(ref_tensor):
            ref_tensor = pact_dict.get("evidence_heatmap")
    zero = _zero_like(ref_tensor)
    stats = _zero_pact_local_evidence_stats()
    stats["pact_cbea_local_evidence_enabled"] = bool(
        isinstance(pact_dict, dict)
        and pact_dict.get("enabled", False)
        and pact_dict.get("stage", "") == "local_evidence"
    )
    stats["pact_cbea_stage"] = (pact_dict or {}).get("stage", "")
    if not stats["pact_cbea_local_evidence_enabled"] or not cfg["enabled"]:
        return zero, stats
    if cfg["mode"] not in ("pact_local_evidence", "local_evidence", "stage2_evidence"):
        stats["pact_cbea_fallback_reason"] = "unsupported_mode:%s" % cfg["mode"]
        return zero, stats

    try:
        if not isinstance(pact_dict, dict):
            raise ValueError("pact_cbea output is missing")
        logits = pact_dict.get("evidence_heatmap_logits")
        if not torch.is_tensor(logits):
            raise ValueError("PACT-CBEA evidence_heatmap_logits is missing")
        pos_map, source = _extract_anchor_positive_map(target_dict, logits)
        target = _resize_positive_map(pos_map, logits)
        losses = []

        heatmap_cfg = cfg["evidence_heatmap"]
        if heatmap_cfg.get("enabled", False):
            pos_weight = logits.new_tensor(heatmap_cfg.get("pos_weight", 1.0))
            heatmap_loss = F.binary_cross_entropy_with_logits(
                logits.float(),
                target.float(),
                pos_weight=pos_weight,
            )
            weighted = heatmap_loss * heatmap_cfg["weight"]
            losses.append(weighted)
            stats["pact_cbea_evidence_heatmap_loss"] = _item(weighted)

        uncertainty_cfg = cfg["uncertainty"]
        if uncertainty_cfg.get("enabled", False):
            uncertainty = _require_tensor(pact_dict, "evidence_uncertainty")
            fg_mask = _resize_positive_map(target, uncertainty)
            denom = torch.clamp(fg_mask.sum(), min=1.0)
            uncertainty_loss = (uncertainty.float() * fg_mask.float()).sum() / denom
            weighted = uncertainty_loss * uncertainty_cfg["weight"]
            losses.append(weighted)
            stats["pact_cbea_uncertainty_loss"] = _item(weighted)

        descriptor_cfg = cfg["descriptor"]
        if descriptor_cfg.get("enabled", False):
            descriptor = _require_tensor(pact_dict, "evidence_descriptor")
            descriptor_loss = _descriptor_smoothness_loss(descriptor)
            weighted = descriptor_loss * descriptor_cfg["weight"]
            losses.append(weighted)
            stats["pact_cbea_descriptor_loss"] = _item(weighted)

        loc_unc_cfg = cfg["localization_uncertainty"]
        if loc_unc_cfg.get("enabled", False):
            loc_uncertainty = _require_tensor(
                pact_dict, "evidence_localization_uncertainty",
            )
            if not torch.is_tensor(reg_preds):
                raise ValueError("reg_preds is required for localization_uncertainty loss")
            reg_targets_flat = target_dict["targets"].view(
                reg_preds.shape[0], -1, 7,
            )
            residual_map = _regression_residual_map(
                reg_preds, reg_targets_flat, loc_uncertainty,
            )
            fg_mask = _resize_positive_map(target, loc_uncertainty)
            denom = torch.clamp(fg_mask.sum(), min=1.0)
            loc_unc_loss = (
                torch.abs(loc_uncertainty.float() - residual_map) * fg_mask.float()
            ).sum() / denom
            weighted = loc_unc_loss * loc_unc_cfg["weight"]
            losses.append(weighted)
            stats["pact_cbea_localization_uncertainty_loss"] = _item(weighted)

        total = sum(losses, zero)
        if not torch.isfinite(total):
            raise ValueError("PACT-CBEA local evidence loss is not finite")
        stats.update({
            "pact_cbea_loss": _item(total),
            "pact_cbea_local_evidence_loss": _item(total),
            "pact_cbea_target_source": source,
            "pact_cbea_fg_ratio": _item(target.float().mean()),
            "pact_cbea_fallback_reason": "",
        })
        return total, stats
    except Exception as exc:
        if not fallback_on_error:
            raise ValueError("PACT-CBEA local evidence loss failed: %s" % exc) from exc
        stats["pact_cbea_fallback_reason"] = "%s:%s" % (type(exc).__name__, str(exc))
        return zero, stats


def _compute_gt_guided_loss(aux_dict, target_dict, gt_cfg, fallback_on_error=True):
    zero = _zero_like(_find_ref_tensor(aux_dict))
    stats = _zero_gt_stats()
    stats["hvp_gt_guided_enabled"] = bool(gt_cfg.get("enabled", False))
    if not gt_cfg.get("enabled", False):
        return zero, stats

    enabled_terms = (
        gt_cfg["hypothesis_heatmap"].get("enabled", False)
        or gt_cfg["residual_focus"].get("enabled", False)
        or gt_cfg["residual_fg_boost"].get("enabled", False)
    )
    if not enabled_terms:
        return zero, stats

    try:
        ref_tensor = _select_gt_ref_tensor(aux_dict, gt_cfg)
        pos_map, source = _extract_anchor_positive_map(target_dict, ref_tensor)
        stats["hvp_gt_target_source"] = source
        stats["hvp_gt_fg_ratio"] = _item(pos_map.float().mean())
        losses = []

        heatmap_cfg = gt_cfg["hypothesis_heatmap"]
        if heatmap_cfg["enabled"]:
            hmap = _require_any_tensor(aux_dict, ("hypothesis_hmap", "hmap"))
            hmap_target = _resize_positive_map(pos_map, hmap)
            heatmap_loss = _hypothesis_heatmap_loss(hmap, hmap_target, heatmap_cfg)
            weighted = heatmap_loss * heatmap_cfg["weight"]
            losses.append(weighted)
            stats["hvp_gt_hypothesis_heatmap_loss"] = _item(weighted)

        focus_cfg = gt_cfg["residual_focus"]
        if focus_cfg["enabled"]:
            hvp_residual = _require_tensor(aux_dict, "hvp_residual")
            fg_mask = _resize_positive_map(pos_map, hvp_residual)
            residual_abs = torch.abs(hvp_residual.float())
            bg_mask = 1.0 - fg_mask
            focus_loss = (
                focus_cfg["bg_weight"] * torch.mean(residual_abs * bg_mask)
                + focus_cfg["fg_weight"] * torch.mean(residual_abs * fg_mask)
            )
            weighted = focus_loss * focus_cfg["weight"]
            losses.append(weighted)
            stats["hvp_gt_residual_focus_loss"] = _item(weighted)

        fg_boost_cfg = gt_cfg["residual_fg_boost"]
        if fg_boost_cfg["enabled"]:
            hvp_residual = _require_tensor(aux_dict, "hvp_residual")
            fg_mask = _resize_positive_map(pos_map, hvp_residual)
            target = hvp_residual.new_tensor(fg_boost_cfg["target"])
            fg_boost_loss = torch.mean(torch.abs(hvp_residual.float() - target) * fg_mask)
            weighted = fg_boost_loss * fg_boost_cfg["weight"]
            losses.append(weighted)
            stats["hvp_gt_residual_fg_boost_loss"] = _item(weighted)

        total = sum(losses, zero)
        if not torch.isfinite(total):
            raise ValueError("HVP GT-guided auxiliary loss is not finite")
        return total, stats
    except Exception as exc:
        if not fallback_on_error:
            raise ValueError("HVP GT-guided auxiliary loss failed: %s" % exc) from exc
        stats["hvp_gt_fallback_reason"] = "%s:%s" % (type(exc).__name__, str(exc))
        return zero, stats


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _feature_penalty(tensor, loss_type):
    tensor = tensor.float()
    if loss_type in ("l2", "mse"):
        return torch.mean(tensor ** 2)
    return torch.mean(torch.abs(tensor))


def _select_gt_ref_tensor(aux_dict, gt_cfg):
    if gt_cfg["hypothesis_heatmap"].get("enabled", False):
        value = aux_dict.get("hypothesis_hmap")
        if not torch.is_tensor(value):
            value = aux_dict.get("hmap")
        if torch.is_tensor(value):
            return value
    if (
        gt_cfg["residual_focus"].get("enabled", False)
        or gt_cfg["residual_fg_boost"].get("enabled", False)
    ):
        value = aux_dict.get("hvp_residual")
        if torch.is_tensor(value):
            return value
    ref = _find_ref_tensor(aux_dict)
    if ref is None:
        raise ValueError("no HVP tensor is available for GT-guided loss")
    return ref


def _extract_anchor_positive_map(target_dict, ref_tensor):
    if not isinstance(target_dict, dict):
        raise ValueError("target_dict is missing")
    pos = target_dict.get("pos_equal_one")
    if not torch.is_tensor(pos):
        raise ValueError("target_dict['pos_equal_one'] is missing")
    if not torch.is_tensor(ref_tensor) or ref_tensor.ndim < 3:
        raise ValueError("reference tensor for target map is invalid")
    pos = pos.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    target_hw = tuple(ref_tensor.shape[-2:]) if ref_tensor.ndim >= 4 else None
    pos_map, layout = _positive_map_from_tensor(pos, target_hw)
    return pos_map.clamp(0.0, 1.0), "anchor_pos:%s" % layout


def _positive_map_from_tensor(pos, target_hw=None):
    if pos.ndim == 4:
        if pos.shape[1] == 1:
            return pos, "bchw"
        if pos.shape[-1] == 1:
            return pos.max(dim=-1).values.unsqueeze(1), "bhwc"
        if _looks_like_anchor_dim(pos.shape[-1], pos.shape[1], pos.shape[2]):
            return pos.max(dim=-1).values.unsqueeze(1), "bhwa"
        if _looks_like_anchor_dim(pos.shape[1], pos.shape[2], pos.shape[3]):
            return pos.max(dim=1, keepdim=True).values, "bahw"
        if target_hw is not None and pos.shape[1] == target_hw[0] and pos.shape[2] == target_hw[1]:
            return pos.max(dim=-1).values.unsqueeze(1), "bhwa"
        if target_hw is not None and pos.shape[2] == target_hw[0] and pos.shape[3] == target_hw[1]:
            return pos.max(dim=1, keepdim=True).values, "bahw"
    if pos.ndim == 3:
        if pos.shape[-1] == 1 and target_hw is not None:
            flat = pos.squeeze(-1)
            return _positive_map_from_flat(flat, target_hw), "flat_anchor"
        return pos.unsqueeze(1), "bhw"
    if pos.ndim == 2 and target_hw is not None:
        return _positive_map_from_flat(pos, target_hw), "flat_anchor"
    raise ValueError("unsupported pos_equal_one shape: %s" % (list(pos.shape),))


def _positive_map_from_flat(pos_flat, target_hw):
    height, width = int(target_hw[0]), int(target_hw[1])
    spatial = height * width
    if spatial <= 0 or pos_flat.shape[1] % spatial != 0:
        raise ValueError("cannot reshape flat pos_equal_one to target spatial size")
    anchors = pos_flat.shape[1] // spatial
    pos = pos_flat.view(pos_flat.shape[0], height, width, anchors)
    return pos.max(dim=-1).values.unsqueeze(1)


def _looks_like_anchor_dim(candidate, dim_a, dim_b):
    return int(candidate) <= 32 and int(candidate) <= int(dim_a) and int(candidate) <= int(dim_b)


def _resize_positive_map(pos_map, ref_tensor):
    pos_map = pos_map.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    if pos_map.shape[0] != ref_tensor.shape[0]:
        if pos_map.shape[0] == 1:
            pos_map = pos_map.expand(ref_tensor.shape[0], -1, -1, -1)
        else:
            raise ValueError("target batch size does not match HVP tensor batch size")
    if pos_map.shape[-2:] != ref_tensor.shape[-2:]:
        pos_map = F.interpolate(pos_map, size=ref_tensor.shape[-2:], mode="nearest")
    return pos_map.clamp(0.0, 1.0)


def _regression_residual_map(reg_preds, reg_targets_flat, ref_tensor):
    """Build a detached [B,1,H,W] per-pixel regression-residual magnitude map.

    reg_preds : torch.Tensor, [B, anchors*7, H, W]
    reg_targets_flat : torch.Tensor, [B, H*W*anchors, 7]
    ref_tensor : reference tensor whose spatial size the output should match.
    """
    if not torch.is_tensor(reg_preds) or reg_preds.ndim != 4:
        raise ValueError("reg_preds must be a [B,anchors*7,H,W] tensor")
    if not torch.is_tensor(reg_targets_flat) or reg_targets_flat.ndim != 3:
        raise ValueError("reg_targets must be a [B,H*W*anchors,7] tensor")
    batch_size, channels, height, width = reg_preds.shape
    if channels % 7 != 0:
        raise ValueError("reg_preds channel dimension must be a multiple of 7")
    anchors = channels // 7
    reg_preds_flat = reg_preds.permute(0, 2, 3, 1).contiguous().view(
        batch_size, height * width * anchors, 7,
    )
    if reg_targets_flat.shape[:2] != reg_preds_flat.shape[:2]:
        raise ValueError("reg_targets shape does not match reg_preds anchor layout")
    residual = torch.abs(
        reg_preds_flat.float().detach() - reg_targets_flat.float().detach()
    ).mean(dim=-1)
    residual_map = residual.view(batch_size, height, width, anchors).mean(dim=-1)
    residual_map = residual_map.unsqueeze(1)
    if residual_map.shape[-2:] != ref_tensor.shape[-2:]:
        residual_map = F.interpolate(
            residual_map, size=ref_tensor.shape[-2:], mode="nearest",
        )
    return residual_map.detach()


def _hypothesis_heatmap_loss(hmap, target, cfg):
    hmap = hmap.float()
    target = target.float()
    pos_weight = hmap.new_tensor(cfg.get("pos_weight", 2.0))
    loss_type = cfg.get("loss", "bce")
    if loss_type in ("bce_logits", "bcewithlogits", "with_logits"):
        return F.binary_cross_entropy_with_logits(hmap, target, pos_weight=pos_weight)
    if _is_probability_tensor(hmap):
        eps = torch.finfo(hmap.dtype).eps
        prob = hmap.clamp(min=eps, max=1.0 - eps)
        loss = -(pos_weight * target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob))
        return torch.mean(loss)
    return F.binary_cross_entropy_with_logits(hmap, target, pos_weight=pos_weight)


def _is_probability_tensor(tensor):
    detached = tensor.detach()
    return bool((detached.min() >= -1e-4).item() and (detached.max() <= 1.0 + 1e-4).item())


def _descriptor_smoothness_loss(descriptor):
    descriptor = descriptor.float()
    losses = []
    if descriptor.shape[-1] > 1:
        losses.append(torch.mean(torch.abs(descriptor[..., 1:] - descriptor[..., :-1])))
    if descriptor.shape[-2] > 1:
        losses.append(torch.mean(torch.abs(descriptor[..., 1:, :] - descriptor[..., :-1, :])))
    if not losses:
        return torch.mean(descriptor ** 2)
    return sum(losses) / len(losses)


def _find_ref_tensor(aux_dict):
    if not isinstance(aux_dict, dict):
        return None
    for key in (
        "hvp_residual",
        "delta_feature",
        "hypothesis_hmap",
        "hmap",
        "evidence_heatmap_logits",
        "evidence_heatmap",
        "evidence_uncertainty",
        "evidence_descriptor",
        "alpha",
        "effective_alpha",
        "loss_ref_tensor",
    ):
        value = aux_dict.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _zero_like(ref_tensor):
    if torch.is_tensor(ref_tensor):
        return ref_tensor.sum() * 0.0
    return torch.tensor(0.0)


def _zero_stats():
    stats = {
        "hvp_aux_enabled": False,
        "hvp_residual_reg_loss": 0.0,
        "hvp_alpha_reg_loss": 0.0,
        "hvp_refinement_consistency_loss": 0.0,
        "hvp_aux_total_loss": 0.0,
        "hvp_aux_fallback_reason": "",
    }
    stats.update(_zero_gt_stats())
    return stats


def _zero_gt_stats():
    return {
        "hvp_gt_guided_enabled": False,
        "hvp_gt_hypothesis_heatmap_loss": 0.0,
        "hvp_gt_residual_focus_loss": 0.0,
        "hvp_gt_residual_fg_boost_loss": 0.0,
        "hvp_gt_fg_ratio": 0.0,
        "hvp_gt_target_source": "",
        "hvp_gt_fallback_reason": "",
    }


def _zero_hvp_v3_stats():
    return {
        "hvp_v3_enabled": False,
        "hvp_v3_stage": "",
        "hvp_v3_loss": 0.0,
        "hvp_v3_stage1_hypothesis_loss": 0.0,
        "hvp_v3_stage2_evidence_loss": 0.0,
        "hvp_v3_stage2_evidence_heatmap_loss": 0.0,
        "hvp_v3_stage2_uncertainty_loss": 0.0,
        "hvp_v3_stage2_descriptor_loss": 0.0,
        "hvp_v3_fg_ratio": 0.0,
        "hvp_v3_target_source": "",
        "hvp_v3_fallback_reason": "",
    }


def _zero_pact_local_evidence_stats():
    return {
        "pact_cbea_local_evidence_enabled": False,
        "pact_cbea_stage": "",
        "pact_cbea_loss": 0.0,
        "pact_cbea_local_evidence_loss": 0.0,
        "pact_cbea_evidence_heatmap_loss": 0.0,
        "pact_cbea_uncertainty_loss": 0.0,
        "pact_cbea_descriptor_loss": 0.0,
        "pact_cbea_localization_uncertainty_loss": 0.0,
        "pact_cbea_fg_ratio": 0.0,
        "pact_cbea_target_source": "",
        "pact_cbea_fallback_reason": "",
    }


def _require_tensor(aux_dict, key):
    value = aux_dict.get(key)
    if not torch.is_tensor(value):
        raise ValueError("%s is missing" % key)
    return value


def _require_any_tensor(aux_dict, keys):
    for key in keys:
        value = aux_dict.get(key)
        if torch.is_tensor(value):
            return value
    raise ValueError("%s is missing" % "/".join(keys))


def _item(tensor):
    return float(tensor.detach().cpu())

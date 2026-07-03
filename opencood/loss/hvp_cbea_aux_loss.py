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
    normalized["enabled"] = bool(normalized.get("enabled", False))
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
    heatmap["enabled"] = bool(heatmap.get("enabled", False))
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


def _find_ref_tensor(aux_dict):
    if not isinstance(aux_dict, dict):
        return None
    for key in (
        "hvp_residual",
        "delta_feature",
        "hypothesis_hmap",
        "hmap",
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

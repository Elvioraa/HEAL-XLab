"""Auxiliary loss plumbing for HVP-CBEA."""

import torch


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
    return normalized


def compute_hvp_auxiliary_loss(aux_dict, aux_cfg=None, fallback_on_error=True):
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


def _find_ref_tensor(aux_dict):
    if not isinstance(aux_dict, dict):
        return None
    for key in ("hvp_residual", "delta_feature", "alpha", "effective_alpha", "loss_ref_tensor"):
        value = aux_dict.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _zero_like(ref_tensor):
    if torch.is_tensor(ref_tensor):
        return ref_tensor.sum() * 0.0
    return torch.tensor(0.0)


def _zero_stats():
    return {
        "hvp_aux_enabled": False,
        "hvp_residual_reg_loss": 0.0,
        "hvp_alpha_reg_loss": 0.0,
        "hvp_refinement_consistency_loss": 0.0,
        "hvp_aux_total_loss": 0.0,
        "hvp_aux_fallback_reason": "",
    }


def _require_tensor(aux_dict, key):
    value = aux_dict.get(key)
    if not torch.is_tensor(value):
        raise ValueError("%s is missing" % key)
    return value


def _item(tensor):
    return float(tensor.detach().cpu())

"""Parameter-free PACT-CBEA trust-calibrated evidence routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PACTCBEARule(nn.Module):
    """Rule-based evidence aggregation with no trainable parameters."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = self._normalize_cfg(cfg)
        self.mode = self.cfg["mode"]
        self.eps = float(self.cfg.get("eps", 1e-6))
        clamp = self.cfg.get("uncertainty_clamp", [-10.0, 10.0])
        self.uncertainty_min = float(clamp[0])
        self.uncertainty_max = float(clamp[1])

    def forward(self, features, evidence_heatmap=None, evidence_uncertainty=None,
                record_len=None, pairwise_t_matrix=None, modality_names=None,
                modality_ids=None):
        feature_tensor, input_layout = self._feature_tensor(features)
        if feature_tensor.ndim == 4:
            return self._forward_flat(
                feature_tensor,
                evidence_heatmap,
                evidence_uncertainty,
                record_len,
                pairwise_t_matrix,
                modality_names,
                modality_ids,
                input_layout,
            )
        return self._aggregate_dense(
            feature_tensor,
            evidence_heatmap,
            evidence_uncertainty,
            pairwise_t_matrix,
            modality_names,
            modality_ids,
            input_layout,
        )

    def _forward_flat(self, feature_tensor, evidence_heatmap, evidence_uncertainty,
                      record_len, pairwise_t_matrix, modality_names, modality_ids,
                      input_layout):
        record_list = self._record_list(record_len, int(feature_tensor.shape[0]))
        if len(record_list) == 1:
            return self._aggregate_dense(
                feature_tensor.unsqueeze(0),
                evidence_heatmap,
                evidence_uncertainty,
                pairwise_t_matrix,
                modality_names,
                modality_ids,
                input_layout,
            )

        outputs = []
        debug_items = []
        start = 0
        for batch_idx, length in enumerate(record_list):
            end = start + int(length)
            group_modalities = self._slice_names(modality_names, start, end)
            group_ids = self._slice_names(modality_ids, start, end)
            group_hmap = self._slice_evidence(evidence_heatmap, start, end)
            group_unc = self._slice_evidence(evidence_uncertainty, start, end)
            enhanced, debug = self._aggregate_dense(
                feature_tensor[start:end].unsqueeze(0),
                group_hmap,
                group_unc,
                pairwise_t_matrix,
                group_modalities,
                group_ids,
                "%s_group%d" % (input_layout, batch_idx),
            )
            outputs.append(enhanced[0])
            debug_items.append(debug)
            start = end
        return torch.stack(outputs, dim=0), self._merge_group_debug(debug_items)

    def _aggregate_dense(self, feature_tensor, evidence_heatmap, evidence_uncertainty,
                         pairwise_t_matrix, modality_names, modality_ids,
                         input_layout):
        if feature_tensor.ndim != 5:
            raise ValueError("PACT-CBEA features must be [N,C,H,W] or [B,N,C,H,W]")
        batch_size, agent_count, _, height, width = feature_tensor.shape
        device = feature_tensor.device
        dtype = feature_tensor.dtype
        fallbacks = []

        confidence = self._confidence(
            evidence_heatmap,
            batch_size,
            agent_count,
            height,
            width,
            device,
            dtype,
            fallbacks,
        )
        uncertainty_weight = self._uncertainty_weight(
            evidence_uncertainty,
            batch_size,
            agent_count,
            height,
            width,
            device,
            dtype,
            fallbacks,
        )
        modality_prior = self._modality_prior(
            modality_names,
            modality_ids,
            batch_size,
            agent_count,
            device,
            dtype,
            fallbacks,
        )
        spatial_weight = self._spatial_weight(
            pairwise_t_matrix,
            batch_size,
            agent_count,
            height,
            width,
            device,
            dtype,
            fallbacks,
        )
        descriptor_weight = self._descriptor_weight(
            batch_size,
            agent_count,
            height,
            width,
            device,
            dtype,
            fallbacks,
        )

        reliability = confidence * uncertainty_weight * modality_prior * spatial_weight * descriptor_weight
        reliability = torch.nan_to_num(reliability, nan=0.0, posinf=0.0, neginf=0.0)
        reliability = torch.clamp(reliability, min=0.0)
        denom = reliability.sum(dim=1, keepdim=True)
        uniform = feature_tensor.new_full((batch_size, agent_count, 1, height, width),
                                          1.0 / max(agent_count, 1))
        alpha = torch.where(
            denom > self.eps,
            reliability / (denom + self.eps),
            uniform,
        )
        if not torch.isfinite(alpha).all():
            alpha = uniform
            fallbacks.append("non_finite_alpha_uniform")

        enhanced = torch.sum(alpha * feature_tensor, dim=1)
        debug = {
            "pact_mode": self.mode,
            "pact_input_layout": input_layout,
            "pact_alpha": alpha,
            "pact_reliability": reliability,
            "pact_evidence_confidence": confidence,
            "pact_uncertainty_weight": uncertainty_weight,
            "pact_modality_prior": modality_prior,
            "pact_spatial_weight": spatial_weight,
            "pact_descriptor_weight": descriptor_weight,
            "pact_fallbacks": fallbacks,
            "pact_agent_count": int(agent_count),
            "pact_trainable": False,
            "pact_no_joint_training": True,
        }
        return enhanced, debug

    def _confidence(self, evidence_heatmap, batch_size, agent_count, height, width,
                    device, dtype, fallbacks):
        if not self.cfg["aggregation"].get("evidence_confidence", True):
            fallbacks.append("evidence_confidence_disabled")
            return torch.ones(batch_size, agent_count, 1, height, width, device=device, dtype=dtype)
        if evidence_heatmap is None:
            fallbacks.append("missing_evidence_heatmap_confidence_ones")
            return torch.ones(batch_size, agent_count, 1, height, width, device=device, dtype=dtype)
        heatmap = self._evidence_tensor(evidence_heatmap, batch_size, agent_count, height, width,
                                        device, dtype, fallbacks, "evidence_heatmap")
        return torch.sigmoid(heatmap) ** float(self.cfg["weights"].get("evidence", 1.0))

    def _uncertainty_weight(self, evidence_uncertainty, batch_size, agent_count,
                            height, width, device, dtype, fallbacks):
        if not self.cfg["aggregation"].get("uncertainty_weight", True):
            fallbacks.append("uncertainty_weight_disabled")
            return torch.ones(batch_size, agent_count, 1, height, width, device=device, dtype=dtype)
        if evidence_uncertainty is None:
            fallbacks.append("missing_evidence_uncertainty_weight_ones")
            return torch.ones(batch_size, agent_count, 1, height, width, device=device, dtype=dtype)
        uncertainty = self._evidence_tensor(evidence_uncertainty, batch_size, agent_count, height, width,
                                            device, dtype, fallbacks, "evidence_uncertainty")
        uncertainty = torch.clamp(uncertainty, min=self.uncertainty_min, max=self.uncertainty_max)
        weight = torch.exp(-uncertainty)
        return weight ** float(self.cfg["weights"].get("uncertainty", 1.0))

    def _spatial_weight(self, pairwise_t_matrix, batch_size, agent_count, height,
                        width, device, dtype, fallbacks):
        if not self.cfg["aggregation"].get("spatial_consistency", True):
            fallbacks.append("spatial_consistency_disabled")
            return torch.ones(batch_size, agent_count, 1, height, width, device=device, dtype=dtype)
        # The first increment keeps a safe HEAL-compatible fallback. Pose-aware
        # spatial evidence can plug in here without changing the public API.
        if pairwise_t_matrix is None:
            fallbacks.append("missing_pairwise_t_matrix_spatial_ones")
        else:
            fallbacks.append("spatial_consistency_basic_ones")
        return torch.ones(batch_size, agent_count, 1, height, width, device=device, dtype=dtype)

    def _descriptor_weight(self, batch_size, agent_count, height, width, device,
                           dtype, fallbacks):
        if self.cfg["aggregation"].get("descriptor_consistency", False):
            fallbacks.append("descriptor_consistency_fallback_ones")
        else:
            fallbacks.append("descriptor_consistency_disabled")
        return torch.ones(batch_size, agent_count, 1, height, width, device=device, dtype=dtype)

    def _modality_prior(self, modality_names, modality_ids, batch_size, agent_count,
                        device, dtype, fallbacks):
        if not self.cfg["aggregation"].get("modality_prior", True):
            fallbacks.append("modality_prior_disabled")
            return torch.ones(batch_size, agent_count, 1, 1, 1, device=device, dtype=dtype)
        names = modality_names if modality_names is not None else modality_ids
        if names is None:
            fallbacks.append("missing_modality_names_prior_ones")
            return torch.ones(batch_size, agent_count, 1, 1, 1, device=device, dtype=dtype)
        flat = self._flatten_names(names)
        if len(flat) == agent_count:
            flat = flat * batch_size
        values = []
        prior_cfg = self.cfg["weights"].get("modality_prior", {})
        for idx in range(batch_size * agent_count):
            name = str(flat[idx]) if idx < len(flat) else ""
            values.append(float(prior_cfg.get(name, 1.0)))
        prior = torch.tensor(values, device=device, dtype=dtype).view(batch_size, agent_count, 1, 1, 1)
        return torch.clamp(prior, min=0.0)

    def _evidence_tensor(self, value, batch_size, agent_count, height, width,
                         device, dtype, fallbacks, name):
        tensor, layout = self._to_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 2:
            tensor = tensor.view(1, tensor.shape[0], 1, 1, 1)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(2)
        elif tensor.ndim == 4:
            if tensor.shape[0] == agent_count:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[0] == batch_size and tensor.shape[1] == agent_count:
                tensor = tensor.unsqueeze(2)
            elif tensor.shape[1] == agent_count:
                tensor = tensor.unsqueeze(2)
            elif tensor.shape[0] == batch_size:
                tensor = tensor.unsqueeze(1)
            else:
                tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 5:
            fallbacks.append("%s_invalid_layout_%s_ones" % (name, layout))
            return torch.ones(batch_size, agent_count, 1, height, width, device=device, dtype=dtype)

        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1, -1, -1, -1)
        if tensor.shape[1] == 1 and agent_count > 1:
            tensor = tensor.expand(-1, agent_count, -1, -1, -1)
        if tensor.shape[2] != 1:
            tensor = tensor.mean(dim=2, keepdim=True)
        if tensor.shape[-2:] != (height, width):
            tensor = F.interpolate(
                tensor.reshape(tensor.shape[0] * tensor.shape[1], 1, tensor.shape[-2], tensor.shape[-1]),
                size=(height, width),
                mode="nearest",
            ).view(tensor.shape[0], tensor.shape[1], 1, height, width)
        if tensor.shape[:2] != (batch_size, agent_count):
            fallbacks.append("%s_shape_mismatch_ones" % name)
            return torch.ones(batch_size, agent_count, 1, height, width, device=device, dtype=dtype)
        return tensor

    def _feature_tensor(self, features):
        tensor, layout = self._to_tensor(features)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim not in (4, 5):
            raise ValueError("PACT-CBEA features must be [N,C,H,W], [B,N,C,H,W], or a compatible list")
        return tensor, layout

    @staticmethod
    def _to_tensor(value, device=None, dtype=None):
        if isinstance(value, (list, tuple)):
            tensors = [item if torch.is_tensor(item) else torch.as_tensor(item) for item in value]
            if not tensors:
                raise ValueError("PACT-CBEA received an empty tensor list")
            if tensors[0].ndim == 3:
                tensor = torch.stack(tensors, dim=0)
                layout = "list_chw"
            elif tensors[0].ndim == 4:
                tensor = torch.stack(tensors, dim=1)
                layout = "list_bchw"
            else:
                tensor = torch.stack(tensors, dim=0)
                layout = "list"
        elif torch.is_tensor(value):
            tensor = value
            layout = "tensor_%dd" % value.ndim
        else:
            tensor = torch.as_tensor(value)
            layout = "array"
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device or tensor.device, dtype=dtype or tensor.dtype)
        return tensor, layout

    @staticmethod
    def _slice_evidence(value, start, end):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value[start:end]
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] >= end:
            return value[start:end]
        return value

    @staticmethod
    def _slice_names(names, start, end):
        if names is None:
            return None
        if isinstance(names, (list, tuple)):
            return list(names[start:end])
        return names

    @staticmethod
    def _flatten_names(names):
        if isinstance(names, (list, tuple)):
            flat = []
            for item in names:
                if isinstance(item, (list, tuple)):
                    flat.extend(str(x) for x in item)
                else:
                    flat.append(str(item))
            return flat
        return [str(names)]

    @staticmethod
    def _record_list(record_len, total_agents):
        if record_len is None:
            return [int(total_agents)]
        if torch.is_tensor(record_len):
            values = [int(x) for x in record_len.detach().cpu().view(-1).tolist()]
        else:
            values = [int(x) for x in record_len]
        if sum(values) != int(total_agents):
            return [int(total_agents)]
        return values

    @staticmethod
    def _merge_group_debug(debug_items):
        if not debug_items:
            return {}
        merged = {
            "pact_mode": debug_items[0].get("pact_mode", "trust_calibrated_rule"),
            "pact_group_debug": debug_items,
            "pact_batch_size": len(debug_items),
            "pact_fallbacks": [],
            "pact_trainable": False,
            "pact_no_joint_training": True,
        }
        for item in debug_items:
            merged["pact_fallbacks"].extend(item.get("pact_fallbacks", []))
        if len(debug_items) == 1:
            merged.update(debug_items[0])
        return merged

    @classmethod
    def _normalize_cfg(cls, cfg):
        normalized = cls._default_cfg()
        if isinstance(cfg, dict):
            _deep_update(normalized, cfg)
        elif isinstance(cfg, bool):
            normalized["enabled"] = bool(cfg)
        normalized["enabled"] = bool(normalized.get("enabled", True))
        normalized["mode"] = str(normalized.get("mode", "trust_calibrated_rule"))
        normalized["trainable"] = bool(normalized.get("trainable", False))
        normalized["no_joint_training"] = bool(normalized.get("no_joint_training", True))
        normalized["use_stage3_joint_training"] = bool(normalized.get("use_stage3_joint_training", False))
        normalized["heal_compatible"] = bool(normalized.get("heal_compatible", True))
        normalized["plug_and_play"] = bool(normalized.get("plug_and_play", True))
        for key, value in list(normalized["aggregation"].items()):
            normalized["aggregation"][key] = bool(value)
        for key in ("evidence", "uncertainty", "spatial", "descriptor"):
            normalized["weights"][key] = float(normalized["weights"].get(key, 1.0))
        if not isinstance(normalized["weights"].get("modality_prior"), dict):
            normalized["weights"]["modality_prior"] = {}
        return normalized

    @staticmethod
    def _default_cfg():
        return {
            "enabled": True,
            "mode": "trust_calibrated_rule",
            "trainable": False,
            "no_joint_training": True,
            "use_stage3_joint_training": False,
            "heal_compatible": True,
            "plug_and_play": True,
            "eps": 1e-6,
            "uncertainty_clamp": [-10.0, 10.0],
            "aggregation": {
                "evidence_confidence": True,
                "uncertainty_weight": True,
                "descriptor_consistency": False,
                "spatial_consistency": True,
                "modality_prior": True,
            },
            "weights": {
                "evidence": 1.0,
                "uncertainty": 1.0,
                "spatial": 1.0,
                "descriptor": 0.0,
                "modality_prior": {
                    "m1": 1.0,
                    "m2": 1.0,
                    "m3": 1.0,
                    "m4": 1.0,
                },
            },
        }


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value

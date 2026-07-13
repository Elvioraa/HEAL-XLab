"""Parameter-free geometry and validation for evidence-routed PACT-CBEA."""

from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PACTCBEAEvidenceGeometryAligner(nn.Module):
    """Warp dense features and all evidence fields with one shared grid."""

    def __init__(self, align_corners=False, invalid_logit=-16.0,
                 invalid_uncertainty=16.0):
        super(PACTCBEAEvidenceGeometryAligner, self).__init__()
        self.align_corners = bool(align_corners)
        self.invalid_logit = float(invalid_logit)
        self.invalid_uncertainty = float(invalid_uncertainty)

    @staticmethod
    def parameter_count():
        return 0

    @staticmethod
    def _record_list(record_len):
        if torch.is_tensor(record_len):
            values = record_len.detach().cpu().tolist()
        else:
            values = list(record_len)
        return [int(value) for value in values]

    @staticmethod
    def _require_4d(name, tensor):
        if not torch.is_tensor(tensor) or tensor.dim() != 4:
            raise RuntimeError("%s must be a 4D tensor" % name)

    def _validate_inputs(self, feature, heatmap_logits, uncertainty,
                         descriptor, record_len, affine_matrix):
        tensors = {
            "feature": feature,
            "heatmap_logits": heatmap_logits,
            "uncertainty": uncertainty,
            "descriptor": descriptor,
        }
        for name, tensor in tensors.items():
            self._require_4d(name, tensor)
            if not torch.isfinite(tensor).all().item():
                raise RuntimeError("%s contains NaN or Inf" % name)

        agent_count = feature.shape[0]
        spatial_shape = tuple(feature.shape[-2:])
        for name, tensor in tensors.items():
            if tensor.shape[0] != agent_count:
                raise RuntimeError("%s agent count does not match feature" % name)
            if tuple(tensor.shape[-2:]) != spatial_shape:
                raise RuntimeError("%s spatial shape does not match feature" % name)
        if heatmap_logits.shape[1] != 1 or uncertainty.shape[1] != 1:
            raise RuntimeError("heatmap logits and uncertainty must be single-channel")
        if not (uncertainty > 0).all().item():
            raise RuntimeError("uncertainty must contain positive values")

        records = self._record_list(record_len)
        if not records or any(value <= 0 for value in records):
            raise RuntimeError("record_len must contain positive scene sizes")
        if sum(records) != agent_count:
            raise RuntimeError("record_len sum does not match the agent count")
        if not torch.is_tensor(affine_matrix) or affine_matrix.dim() != 5:
            raise RuntimeError("affine_matrix must have shape [B, L, L, 2, 3]")
        if affine_matrix.shape[0] != len(records):
            raise RuntimeError("affine_matrix batch does not match record_len")
        if tuple(affine_matrix.shape[-2:]) != (2, 3):
            raise RuntimeError("affine_matrix must end with shape [2, 3]")
        for scene_index, count in enumerate(records):
            if affine_matrix.shape[1] < count or affine_matrix.shape[2] < count:
                raise RuntimeError(
                    "affine_matrix cannot represent scene %d with %d agents"
                    % (scene_index, count)
                )
        return records

    def forward(self, feature, heatmap_logits, uncertainty, descriptor,
                record_len, affine_matrix):
        records = self._validate_inputs(
            feature, heatmap_logits, uncertainty, descriptor,
            record_len, affine_matrix
        )
        height, width = feature.shape[-2:]
        feature_channels = feature.shape[1]
        descriptor_channels = descriptor.shape[1]
        aligned_chunks = []
        valid_ratios = []
        offset = 0

        for scene_index, count in enumerate(records):
            scene_feature = feature[offset:offset + count]
            scene_logits = heatmap_logits[offset:offset + count]
            scene_uncertainty = uncertainty[offset:offset + count]
            scene_descriptor = descriptor[offset:offset + count]
            scene_validity = torch.ones_like(scene_logits)
            packed = torch.cat(
                [scene_feature, scene_logits, scene_uncertainty,
                 scene_descriptor, scene_validity], dim=1
            )

            # The repository's fusion path uses [ego, source] transforms with
            # grid_sample, so row zero aligns every source into the ego grid.
            transforms = affine_matrix[scene_index, 0, :count].to(
                device=packed.device, dtype=packed.dtype
            )
            grid = F.affine_grid(
                transforms,
                torch.Size((count, packed.shape[1], height, width)),
                align_corners=self.align_corners,
            )
            warped = F.grid_sample(
                packed,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )

            feature_end = feature_channels
            logits_end = feature_end + 1
            uncertainty_end = logits_end + 1
            descriptor_end = uncertainty_end + descriptor_channels
            warped_feature = warped[:, :feature_end]
            warped_logits = warped[:, feature_end:logits_end]
            warped_uncertainty = warped[:, logits_end:uncertainty_end]
            warped_descriptor = warped[:, uncertainty_end:descriptor_end]
            validity = warped[:, descriptor_end:descriptor_end + 1].clamp(0.0, 1.0)

            warped_feature = warped_feature * validity
            warped_descriptor = warped_descriptor * validity

            # Preserve logits in valid cells while making confidence approach
            # zero outside warped support. The rule performs the only sigmoid.
            confidence = torch.sigmoid(warped_logits) * validity
            eps = max(torch.finfo(confidence.dtype).eps, math.exp(self.invalid_logit))
            confidence = confidence.clamp(min=eps, max=1.0 - eps)
            suppressed_logits = torch.log(confidence) - torch.log1p(-confidence)
            full_support = validity >= (1.0 - torch.finfo(validity.dtype).eps)
            warped_logits = torch.where(
                full_support, warped_logits, suppressed_logits
            )
            invalid_logits = torch.full_like(warped_logits, self.invalid_logit)
            warped_logits = torch.where(validity > eps, warped_logits, invalid_logits)

            validity_penalty = -torch.log(validity.clamp(min=eps))
            warped_uncertainty = warped_uncertainty + validity_penalty
            invalid_uncertainty = torch.full_like(
                warped_uncertainty, self.invalid_uncertainty
            )
            warped_uncertainty = torch.where(
                validity > eps, warped_uncertainty, invalid_uncertainty
            )

            aligned_chunks.append((
                warped_feature,
                warped_logits,
                warped_uncertainty,
                warped_descriptor,
                validity,
            ))
            valid_ratios.extend(
                validity.flatten(1).mean(dim=1).detach().cpu().tolist()
            )
            offset += count

        return {
            "feature": torch.cat([chunk[0] for chunk in aligned_chunks], dim=0),
            "heatmap_logits": torch.cat([chunk[1] for chunk in aligned_chunks], dim=0),
            "uncertainty": torch.cat([chunk[2] for chunk in aligned_chunks], dim=0),
            "descriptor": torch.cat([chunk[3] for chunk in aligned_chunks], dim=0),
            "validity": torch.cat([chunk[4] for chunk in aligned_chunks], dim=0),
            "debug": {
                "shared_alignment_grid_used": True,
                "per_agent_valid_ratio": [float(value) for value in valid_ratios],
            },
        }


class PACTCBEAEvidenceRoutingValidator(nn.Module):
    """Strict, parameter-free guards for the evidence-routed inference path."""

    _FORBIDDEN_FALLBACK_TOKENS = (
        "missing_evidence",
        "confidence_ones",
        "uncertainty_weight_ones",
        "non_finite_alpha_uniform",
        "shape_mismatch",
        "invalid_layout",
    )

    def __init__(self):
        super(PACTCBEAEvidenceRoutingValidator, self).__init__()

    @staticmethod
    def parameter_count():
        return 0

    @staticmethod
    def _record_list(record_len):
        if torch.is_tensor(record_len):
            values = record_len.detach().cpu().tolist()
        else:
            values = list(record_len)
        return [int(value) for value in values]

    def validate_heads(self, model, modalities):
        missing = []
        for modality in modalities:
            name = "pact_cbea_evidence_head_%s" % modality
            if not hasattr(model, name):
                missing.append(name)
        if missing:
            raise RuntimeError("Missing evidence heads: %s" % ", ".join(missing))

    def validate_agent_outputs(self, feature, heatmap_logits, uncertainty,
                               descriptor, modalities, record_len):
        if len(modalities) != feature.shape[0]:
            raise RuntimeError("agent_modality_list does not match agent feature count")
        records = self._record_list(record_len)
        if not records or sum(records) != feature.shape[0]:
            raise RuntimeError("record_len does not match agent feature count")
        expected_hw = tuple(feature.shape[-2:])
        expected_agents = feature.shape[0]
        fields = {
            "heatmap_logits": heatmap_logits,
            "uncertainty": uncertainty,
            "descriptor": descriptor,
        }
        for name, tensor in fields.items():
            if not torch.is_tensor(tensor) or tensor.dim() != 4:
                raise RuntimeError("%s must be a 4D tensor" % name)
            if tensor.shape[0] != expected_agents:
                raise RuntimeError("%s agent count mismatch" % name)
            if tuple(tensor.shape[-2:]) != expected_hw:
                raise RuntimeError("%s spatial shape mismatch" % name)
            if not torch.isfinite(tensor).all().item():
                raise RuntimeError("%s contains NaN or Inf" % name)
        if heatmap_logits.shape[1] != 1 or uncertainty.shape[1] != 1:
            raise RuntimeError("heatmap logits and uncertainty must be single-channel")
        if not (uncertainty > 0).all().item():
            raise RuntimeError("uncertainty must be positive")

    def validate_aligned_outputs(self, aligned):
        required = ("feature", "heatmap_logits", "uncertainty", "descriptor", "validity")
        for name in required:
            if name not in aligned:
                raise RuntimeError("Aligned output is missing %s" % name)
            tensor = aligned[name]
            if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all().item():
                raise RuntimeError("Aligned %s is invalid" % name)
        validity = aligned["validity"]
        if (validity < 0).any().item() or (validity > 1).any().item():
            raise RuntimeError("Aligned validity must be in [0, 1]")
        invalid = validity <= torch.finfo(validity.dtype).eps
        if invalid.any().item():
            if (aligned["feature"] * invalid.to(aligned["feature"].dtype)).abs().max().item() != 0:
                raise RuntimeError("Invalid aligned feature support is not zero")
            if (aligned["descriptor"] * invalid.to(aligned["descriptor"].dtype)).abs().max().item() != 0:
                raise RuntimeError("Invalid aligned descriptor support is not zero")

    @staticmethod
    def _collect_strings(value):
        strings = []
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                strings.extend(PACTCBEAEvidenceRoutingValidator._collect_strings(key))
                strings.extend(PACTCBEAEvidenceRoutingValidator._collect_strings(item))
        elif isinstance(value, (list, tuple)):
            for item in value:
                strings.extend(PACTCBEAEvidenceRoutingValidator._collect_strings(item))
        return strings

    def validate_rule_debug(self, rule_debug):
        strings = [value.lower() for value in self._collect_strings(rule_debug)]
        violations = []
        for value in strings:
            for token in self._FORBIDDEN_FALLBACK_TOKENS:
                if token in value:
                    violations.append(value)
                    break
        if violations:
            raise RuntimeError(
                "PACT evidence routing rejected rule fallback(s): %s"
                % ", ".join(sorted(set(violations)))
            )
        return []

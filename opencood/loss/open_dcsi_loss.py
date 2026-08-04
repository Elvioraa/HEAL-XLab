"""Composable Open-DCSI auxiliary loss around an unchanged official loss."""

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.open_dcsi.config import normalize_open_dcsi_config


_TERM_LOG_NAMES = {
    "common_detection": "Common Detection Loss",
    "reconstruction": "Common Reconstruction Loss",
    "common_innovation_decorrelation": "Common-Innovation Decorrelation Loss",
    "innovation_detection": "Innovation Token Loss",
    "box_refinement": "Geometry Refinement Loss",
    "quality": "Innovation Quality Loss",
    "token_sparsity": "Token Sparsity Loss",
    "budget": "Budget Loss",
}


def _build_official_loss(config):
    method = config["core_method"]
    module = importlib.import_module("opencood.loss.{}".format(method))
    target = method.replace("_", "").lower()
    loss_class = None
    for name, candidate in module.__dict__.items():
        if name.lower() == target:
            loss_class = candidate
            break
    if loss_class is None:
        raise ValueError("Official loss class not found for {}".format(method))
    return loss_class(config["args"])


def _zero(reference):
    return reference.sum() * 0.0


def _delta_to_boxes3d(deltas, anchors):
    """Decode xyzhwlyaw deltas with the official VoxelPostprocessor formula."""

    batch_size = deltas.shape[0]
    deltas = deltas.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 7)
    anchors = anchors.to(device=deltas.device, dtype=deltas.dtype).view(-1, 7)
    anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
    diagonal = torch.sqrt(anchors[..., 4].square() + anchors[..., 5].square())
    boxes = torch.zeros_like(deltas)
    boxes[..., 0] = deltas[..., 0] * diagonal + anchors[..., 0]
    boxes[..., 1] = deltas[..., 1] * diagonal + anchors[..., 1]
    boxes[..., 2] = deltas[..., 2] * anchors[..., 3] + anchors[..., 2]
    boxes[..., 3:6] = torch.exp(deltas[..., 3:6]) * anchors[..., 3:6]
    boxes[..., 6] = deltas[..., 6] + anchors[..., 6]
    return boxes


class OpenDcsiLoss(nn.Module):
    """Keep official loss authoritative and isolate each optional new term."""

    def __init__(self, args):
        super().__init__()
        if "official_loss" not in args:
            raise ValueError("Open-DCSI loss requires official_loss configuration")
        self.official_loss = _build_official_loss(args["official_loss"])
        self.open_config = normalize_open_dcsi_config(args.get("open_dcsi"))
        self.enabled = bool(
            self.open_config["enabled"] and self.open_config["losses"]["enabled"]
        )
        self.loss_config = self.open_config["losses"]
        self.loss_dict = {}
        self.skipped_non_finite_count = 0

    def _common_detection(self, output, target):
        logits = output["cls_preds"].amax(dim=1, keepdim=True)
        occupancy = target["pos_equal_one"].amax(dim=-1).unsqueeze(1).to(logits.dtype)
        if occupancy.shape[-2:] != logits.shape[-2:]:
            occupancy = F.interpolate(occupancy, size=logits.shape[-2:], mode="nearest")
        return F.binary_cross_entropy_with_logits(logits, occupancy)

    def _reconstruction(self, output, target):
        open_output = output["open_dcsi"]
        losses = []
        detach_target = self.open_config["common_space"]["reconstruction"][
            "detach_target"
        ]
        for reconstructed, innovation in zip(
            open_output["reconstructed_features"],
            open_output["innovation_features"],
        ):
            original = reconstructed + innovation
            if detach_target:
                original = original.detach()
            losses.append(F.smooth_l1_loss(reconstructed, original))
        return torch.stack(losses).mean() if losses else _zero(output["cls_preds"])

    def _decorrelation(self, output, target):
        losses = []
        open_output = output["open_dcsi"]
        for reconstructed, innovation in zip(
            open_output["reconstructed_features"],
            open_output["innovation_features"],
        ):
            common_norm = F.normalize(reconstructed, dim=1, eps=1e-6)
            innovation_norm = F.normalize(innovation, dim=1, eps=1e-6)
            losses.append((common_norm * innovation_norm).sum(dim=1).square().mean())
        return torch.stack(losses).mean() if losses else _zero(output["cls_preds"])

    @staticmethod
    def _token_targets(tokens, target):
        count = int(tokens["scenario_index"].numel())
        if count == 0:
            return tokens["objectness"].new_empty(0)
        occupancy = target["pos_equal_one"].amax(dim=-1)
        height, width = occupancy.shape[-2:]
        x_min, y_min, _, x_max, y_max, _ = tokens["lidar_range"]
        x = ((tokens["centers_ego"][:, 0] - x_min) / (x_max - x_min) * width).long()
        y = ((tokens["centers_ego"][:, 1] - y_min) / (y_max - y_min) * height).long()
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        result = tokens["objectness"].new_zeros(count)
        if valid.any():
            result[valid] = occupancy[
                tokens["scenario_index"][valid].long(), y[valid], x[valid]
            ].to(result.dtype)
        return result.detach()

    def _innovation_detection(self, output, target):
        tokens = output["open_dcsi"]["innovation_tokens"]
        labels = self._token_targets(tokens, target)
        if labels.numel() == 0:
            return _zero(output["cls_preds"])
        return F.binary_cross_entropy_with_logits(tokens["objectness_logits"], labels)

    def _box_refinement(self, output, target):
        open_output = output["open_dcsi"]
        refinement = open_output.get("geometry_refinement")
        anchor = open_output.get("anchor_box")
        if refinement is None or anchor is None:
            return _zero(output["cls_preds"])
        tokens = open_output["fused_tokens"]
        if tokens["scenario_index"].numel() == 0:
            return _zero(output["cls_preds"])
        predicted_boxes = _delta_to_boxes3d(output["reg_preds"], anchor)
        target_delta = target["targets"].permute(0, 3, 1, 2).contiguous()
        target_boxes = _delta_to_boxes3d(target_delta, anchor)
        positive = target["pos_equal_one"].reshape(target_boxes.shape[0], -1) > 0
        residual_targets = []
        predicted_residuals = []
        for token_index in range(tokens["scenario_index"].numel()):
            scene = int(tokens["scenario_index"][token_index].item())
            positive_index = torch.nonzero(positive[scene], as_tuple=False).flatten()
            if positive_index.numel() == 0:
                continue
            token_center = tokens["centers_ego"][token_index, :2]
            prediction_distance = torch.linalg.vector_norm(
                predicted_boxes[scene, :, :2] - token_center, dim=-1
            )
            prediction_index = prediction_distance.argmin()
            prediction = predicted_boxes[scene, prediction_index]
            gt_candidates = target_boxes[scene].index_select(0, positive_index)
            gt_distance = torch.linalg.vector_norm(
                gt_candidates[:, :2] - prediction[:2], dim=-1
            )
            gt = gt_candidates[gt_distance.argmin()]
            residual = torch.zeros_like(prediction)
            residual[:3] = gt[:3] - prediction[:3]
            residual[3:6] = torch.log(
                gt[3:6].clamp_min(1e-6) / prediction[3:6].clamp_min(1e-6)
            )
            yaw = gt[6] - prediction[6]
            residual[6] = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            residual_targets.append(residual.detach())
            predicted_residuals.append(refinement["box_deltas_hwl"][token_index])
        if not residual_targets:
            return _zero(output["cls_preds"])
        return F.smooth_l1_loss(
            torch.stack(predicted_residuals), torch.stack(residual_targets)
        )

    def _quality(self, output, target):
        tokens = output["open_dcsi"]["innovation_tokens"]
        labels = self._token_targets(tokens, target)
        if labels.numel() == 0:
            return _zero(output["cls_preds"])
        quality_loss = F.binary_cross_entropy(
            tokens["box_quality"].clamp(1e-6, 1 - 1e-6), labels
        )
        uncertainty_target = 1.0 - labels
        quality_loss = quality_loss + F.smooth_l1_loss(
            torch.tanh(tokens["general_uncertainty"]), uncertainty_target
        )
        quality_loss = quality_loss + F.smooth_l1_loss(
            torch.tanh(tokens["localization_uncertainty"]), uncertainty_target
        )
        return quality_loss

    def _token_sparsity(self, output, target):
        tokens = output["open_dcsi"]["innovation_tokens"]
        if tokens["objectness"].numel() == 0:
            return _zero(output["cls_preds"])
        return tokens["objectness"].mean()

    def _budget(self, output, target):
        tokens = output["open_dcsi"]["innovation_tokens"]
        if tokens["objectness"].numel() == 0:
            return _zero(output["cls_preds"])
        maximum = float(
            max(1, self.open_config["innovation_tokens"]["max_tokens_per_agent"])
        )
        return (tokens["objectness"].sum() / maximum).square()

    def _compute_term(self, name, output, target):
        functions = {
            "common_detection": self._common_detection,
            "reconstruction": self._reconstruction,
            "common_innovation_decorrelation": self._decorrelation,
            "innovation_detection": self._innovation_detection,
            "box_refinement": self._box_refinement,
            "quality": self._quality,
            "token_sparsity": self._token_sparsity,
            "budget": self._budget,
        }
        return functions[name](output, target)

    def forward(self, output_dict, target_dict, suffix=""):
        official_total = self.official_loss(output_dict, target_dict, suffix)
        self.loss_dict = dict(self.official_loss.loss_dict)
        if not self.enabled or suffix or "open_dcsi" not in output_dict:
            return official_total
        open_total = _zero(official_total)
        skipped_this_call = 0
        for name in _TERM_LOG_NAMES:
            term_config = self.loss_config[name]
            if not term_config["enabled"]:
                continue
            term = self._compute_term(name, output_dict, target_dict)
            if term.ndim != 0:
                term = term.mean()
            if not torch.isfinite(term):
                skipped_this_call += 1
                self.loss_dict["open_dcsi_{}_skipped".format(name)] = True
                continue
            weighted = term * float(term_config["weight"])
            open_total = open_total + weighted
            self.loss_dict["open_dcsi_{}_loss".format(name)] = float(term.detach())
            self.loss_dict["open_dcsi_{}_weighted".format(name)] = float(
                weighted.detach()
            )
        self.skipped_non_finite_count += skipped_this_call
        total = official_total + open_total
        self.loss_dict["open_dcsi_total_loss"] = float(open_total.detach())
        self.loss_dict["open_dcsi_skipped_non_finite_count"] = (
            self.skipped_non_finite_count
        )
        self.loss_dict["total_loss"] = float(total.detach())
        return total

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        self.official_loss.logging(epoch, batch_id, batch_len, writer, suffix)
        if not self.enabled or suffix:
            return
        parts = [
            "Open-DCSI Total Loss: {:.4f}".format(
                self.loss_dict.get("open_dcsi_total_loss", 0.0)
            )
        ]
        for name, display_name in _TERM_LOG_NAMES.items():
            key = "open_dcsi_{}_loss".format(name)
            if key in self.loss_dict:
                parts.append("{}: {:.4e}".format(display_name, self.loss_dict[key]))
                if writer is not None:
                    writer.add_scalar(
                        display_name,
                        self.loss_dict[key],
                        epoch * batch_len + batch_id,
                    )
        parts.append(
            "Open-DCSI Skipped Non-Finite Count: {}".format(
                self.skipped_non_finite_count
            )
        )
        print(" || ".join(parts))

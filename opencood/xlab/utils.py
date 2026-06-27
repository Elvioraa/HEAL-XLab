"""Tensor safety helpers for XLab post-processing."""

import math

import torch


def is_valid_box_tensor(boxes):
    return boxes is not None and torch.is_tensor(boxes) and boxes.ndim == 3 and boxes.shape[1:] == (8, 3)


def is_valid_score_tensor(scores):
    return scores is not None and torch.is_tensor(scores) and scores.ndim == 1


def align_tensor_like(tensor, ref_tensor):
    if tensor is None or ref_tensor is None or not torch.is_tensor(tensor) or not torch.is_tensor(ref_tensor):
        return tensor
    return tensor.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def weighted_yaw_average(yaws, weights):
    sin_sum = torch.sum(torch.sin(yaws) * weights)
    cos_sum = torch.sum(torch.cos(yaws) * weights)
    return torch.atan2(sin_sum, cos_sum)


def empty_boxes_like(ref_tensor):
    if ref_tensor is not None and torch.is_tensor(ref_tensor):
        return ref_tensor.new_zeros((0, 8, 3))
    return torch.zeros((0, 8, 3), dtype=torch.float32)


def empty_scores_like(ref_tensor):
    if ref_tensor is not None and torch.is_tensor(ref_tensor):
        return ref_tensor.new_zeros((0,))
    return torch.zeros((0,), dtype=torch.float32)


def clamp_probability(scores, eps=1e-4):
    return torch.clamp(scores, min=eps, max=1.0 - eps)


def tensor_payload_bytes(*tensors):
    total = 0
    for tensor in tensors:
        if tensor is not None and torch.is_tensor(tensor):
            total += tensor.nelement() * tensor.element_size()
    return int(total)


def safe_float(value, default):
    try:
        if value is None or math.isnan(float(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


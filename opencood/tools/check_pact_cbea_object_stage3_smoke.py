"""Artificial smoke tests for the complete PACT-CBEA object Stage 3 path."""

import copy
import math
import sys
import tempfile
import traceback
import types
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencood.loss.pact_cbea_object_stage3_loss import (  # noqa: E402
    PactCbeaObjectStage3Loss,
)
from opencood.models.heter_pyramid_collab_pact_cbea import (  # noqa: E402
    HeterPyramidCollabPactCbea,
)
from opencood.models.heter_pyramid_collab import (  # noqa: E402
    HeterPyramidCollab,
)
from opencood.models.heter_pyramid_collab_pact_cbea_object_stage3 import (  # noqa: E402
    HeterPyramidCollabPactCbeaObjectStage3,
)
from opencood.models.sub_modules.pact_cbea_object_refiner import (  # noqa: E402
    ObjectResidualCoder,
    SharedObjectRefiner,
    precision_fuse,
    repository_hwl_to_sampler_lwh,
    sampler_lwh_to_repository_hwl,
    wrap_to_pi,
)
from opencood.models.sub_modules.pact_cbea_object_roi import (  # noqa: E402
    BEVGeometry,
    RotatedBEVROISampler,
)
from opencood.models.sub_modules.pact_cbea_object_stage3_utils import (  # noqa: E402
    OBJECT_STAGE3_VERSION,
    _pairwise_rotated_iou_torch,
    build_stage3_checkpoint,
    freeze_except_object_stage3,
    load_base_checkpoint_compatible,
    match_proposals_to_gt,
    rotated_nms_sampler_boxes,
    stage3_named_parameters,
    strict_load_stage3_checkpoint,
)


class TinyFrozenBase(nn.Module):
    """Small model exposing the same Stage-3 parameter namespace."""

    def __init__(self, channels=4):
        super(TinyFrozenBase, self).__init__()
        self.base = nn.Conv2d(channels, channels, kernel_size=1)
        self.object_stage3_refiner = SharedObjectRefiner(
            in_channels=channels,
            hidden_dim=8,
            min_log_variance=-3.0,
            max_log_variance=3.0,
        )


class _IdentityEncoder(nn.Module):
    def forward(self, data_dict, modality_name):
        del modality_name
        return data_dict["input_features"]


class _IdentityBackbone(nn.Module):
    def forward(self, payload):
        return {"spatial_features_2d": payload["spatial_features"]}


class _StrictHealPyramid(nn.Module):
    def __init__(self):
        super(_StrictHealPyramid, self).__init__()
        self.forward_collab_calls = []
        self.align_corners = False

    def forward_single(self, features):
        return features + 3.0, [features[:, :1]]

    def forward_collab(
            self, features, record_len, affine_matrix,
            agent_modality_list=None, cam_crop_info=None, **kwargs):
        del affine_matrix, agent_modality_list, cam_crop_info
        self.forward_collab_calls.append(dict(kwargs))
        scenes = torch.tensor_split(
            features,
            torch.cumsum(record_len, dim=0)[:-1].cpu(),
        )
        return torch.stack([scene.mean(dim=0) for scene in scenes]), [features[:, :1]]


class _CountingRule(nn.Module):
    def __init__(self):
        super(_CountingRule, self).__init__()
        self.call_count = 0

    def forward(self, *args, **kwargs):
        del args, kwargs
        self.call_count += 1
        raise AssertionError("PACT rule must not execute in strict HEAL mode")


class _FixedProposalDecoder(object):
    def __init__(self, proposals, scores):
        self.proposals = proposals
        self.scores = scores

    def decode(self, cls_preds, reg_preds, dir_preds, anchor_box):
        del cls_preds, reg_preds, dir_preds, anchor_box
        return [self.proposals], [self.scores]


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _assert_close(actual, expected, message, atol=1e-5, rtol=1e-5):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(
            "%s\nactual=%s\nexpected=%s" % (message, actual, expected)
        )


def _zero_anchor(module):
    return sum(parameter.sum() * 0.0 for parameter in module.parameters())


def _scene(
        refiner,
        agent_residuals,
        log_variances,
        valid_mask,
        positive_mask,
        target_residuals,
        fused_residual=None,
        normalized_weights=None,
        fallback_mask=None):
    proposal_count, agent_count, _ = agent_residuals.shape
    if fused_residual is None or normalized_weights is None or fallback_mask is None:
        fused_residual, normalized_weights, fallback_mask = precision_fuse(
            agent_residuals, log_variances, valid_mask
        )
    return {
        "object_stage3": {
            "enabled": True,
            "zero_loss_anchor": _zero_anchor(refiner),
            "scenes": [{
                "agent_residuals": agent_residuals,
                "agent_log_variances": log_variances,
                "valid_mask": valid_mask,
                "positive_mask": positive_mask,
                "target_residuals": target_residuals,
                "fused_residual": fused_residual,
                "normalized_agent_weights": normalized_weights,
                "fallback_mask": fallback_mask,
                "matched_ious": target_residuals.new_zeros((proposal_count,)),
                "proposal_count": proposal_count,
                "agent_count": agent_count,
            }],
        }
    }


def _strict_model_args():
    return {
        "lidar_range": [-10.0, -10.0, -3.0, 10.0, 10.0, 1.0],
        "in_head": 2,
        "anchor_number": 2,
        "dir_args": {"dir_offset": 0.7853, "num_bins": 2},
        "supervise_single": False,
        "pact_cbea": {
            "enabled": True,
            "trainable": False,
            "no_joint_training": True,
            "use_stage3_joint_training": False,
            "fusion_mode": "heal_multiscale_prior",
            "multiscale_prior": {
                "enabled": True,
                "lambda": 0.0,
                "injection_strength": 0.0,
            },
            "local_evidence": {"enabled": False},
            "evidence_head": {
                "enabled": False,
                "localization_uncertainty": {"enabled": False},
            },
            "aggregation": {
                "evidence_confidence": False,
                "uncertainty_weight": False,
                "descriptor_consistency": False,
                "spatial_consistency": False,
                "modality_prior": False,
                "localization_weight": False,
            },
            "object_level_stage3": {
                "enabled": True,
                "require_strict_heal_base": True,
                "hidden_dim": 8,
                "roi_size": [3, 3],
                "bev_geometry": {
                    "resolution_x": 1.0,
                    "resolution_y": 1.0,
                    "feature_stride_x": 1.0,
                    "feature_stride_y": 1.0,
                },
            },
        },
    }


def _set_nested(mapping, path, value):
    target = mapping
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _construct_strict_stage3_without_heal_modules():
    original_init = HeterPyramidCollab.__init__
    try:
        HeterPyramidCollab.__init__ = lambda self, args: nn.Module.__init__(self)
        return HeterPyramidCollabPactCbeaObjectStage3(_strict_model_args())
    finally:
        HeterPyramidCollab.__init__ = original_init


def test_box_coder_round_trip():
    coder = ObjectResidualCoder()
    proposals = torch.tensor([
        [1.0, -2.0, 0.2, 4.0, 2.0, 1.5, 0.3],
        [-3.0, 4.0, -0.1, 1.2, 0.8, 1.0, -1.0],
    ])
    targets = torch.tensor([
        [1.3, -1.8, 0.4, 4.4, 1.9, 1.6, 0.5],
        [-2.8, 3.6, 0.0, 1.0, 0.9, 1.1, -0.7],
    ])
    decoded = coder.decode(proposals, coder.encode(proposals, targets))
    _assert_close(decoded, targets, "coder round trip differs")


def test_yaw_wrap_near_pi():
    coder = ObjectResidualCoder()
    proposals = torch.tensor([[0., 0., 0., 2., 1., 1., math.pi - 0.01]])
    targets = torch.tensor([[0., 0., 0., 2., 1., 1., -math.pi + 0.02]])
    residual = coder.encode(proposals, targets)
    _assert_close(residual[:, 6], torch.tensor([0.03]), "yaw residual did not wrap")
    decoded = coder.decode(proposals, residual)
    _assert_close(
        wrap_to_pi(decoded[:, 6] - targets[:, 6]),
        torch.zeros(1),
        "decoded yaw differs modulo 2pi",
    )


def test_box_order_conversion():
    repository = torch.tensor([[1., 2., 3., 4., 5., 6., 0.7]])
    sampler = repository_hwl_to_sampler_lwh(repository)
    _assert_close(
        sampler,
        torch.tensor([[1., 2., 3., 6., 5., 4., 0.7]]),
        "hwl to lwh conversion is wrong",
    )
    _assert_close(
        sampler_lwh_to_repository_hwl(sampler),
        repository,
        "box conversion is not reversible",
    )


def test_shared_head_output_shape():
    head = SharedObjectRefiner(4, hidden_dim=8)
    residuals, logvars = head(
        torch.randn(3, 2, 4, 5, 7), torch.ones(3, 2, dtype=torch.bool)
    )
    _assert(residuals.shape == (3, 2, 7), "residual shape mismatch")
    _assert(logvars.shape == (3, 2, 7), "log-variance shape mismatch")


def test_variable_proposals():
    head = SharedObjectRefiner(3, hidden_dim=8)
    for count in (1, 5):
        output, _ = head(
            torch.randn(count, 2, 3, 3, 4),
            torch.ones(count, 2, dtype=torch.bool),
        )
        _assert(output.shape[0] == count, "variable P was not preserved")


def test_variable_agents():
    head = SharedObjectRefiner(3, hidden_dim=8)
    for count in (2, 4):
        output, _ = head(
            torch.randn(2, count, 3, 3, 4),
            torch.ones(2, count, dtype=torch.bool),
        )
        _assert(output.shape[1] == count, "variable A was not preserved")


def test_single_agent():
    head = SharedObjectRefiner(2, hidden_dim=8)
    output, logvars = head(
        torch.randn(2, 1, 2, 3, 3), torch.ones(2, 1, dtype=torch.bool)
    )
    _assert(output.shape == (2, 1, 7), "A=1 residual shape mismatch")
    _assert(logvars.shape == output.shape, "A=1 log-variance shape mismatch")


def test_empty_proposals():
    head = SharedObjectRefiner(2, hidden_dim=8)
    output, logvars = head(
        torch.empty(0, 3, 2, 3, 5), torch.empty(0, 3, dtype=torch.bool)
    )
    _assert(output.shape == (0, 3, 7), "P=0 residual shape mismatch")
    _assert(logvars.shape == (0, 3, 7), "P=0 log-variance shape mismatch")


def test_invalid_agent_masking():
    residuals = torch.tensor([[[1.] * 7, [100.] * 7]])
    logvars = torch.zeros_like(residuals)
    valid = torch.tensor([[True, False]])
    fused, weights, fallback = precision_fuse(residuals, logvars, valid)
    _assert_close(fused, torch.ones(1, 7), "invalid agent affected fusion")
    _assert_close(weights[:, 1], torch.zeros(1, 7), "invalid weight is nonzero")
    _assert(not bool(fallback.item()), "valid proposal incorrectly fell back")


def test_fully_invalid_fallback():
    residuals = torch.randn(2, 3, 7)
    fused, weights, fallback = precision_fuse(
        residuals, torch.zeros_like(residuals), torch.zeros(2, 3, dtype=torch.bool)
    )
    _assert_close(fused, torch.zeros_like(fused), "fallback residual must be zero")
    _assert_close(weights, torch.zeros_like(weights), "fallback weights must be zero")
    _assert(bool(fallback.all()), "fully invalid proposals must fall back")


def test_lower_variance_higher_weight():
    residuals = torch.zeros(1, 2, 7)
    logvars = torch.tensor([[[ -2.] * 7, [2.] * 7]])
    _, weights, _ = precision_fuse(
        residuals, logvars, torch.ones(1, 2, dtype=torch.bool)
    )
    _assert(bool((weights[:, 0] > weights[:, 1]).all()), "precision ordering is wrong")


def test_agent_permutation_invariance():
    residuals = torch.randn(3, 4, 7)
    logvars = torch.randn(3, 4, 7).clamp(-2, 2)
    valid = torch.tensor([
        [True, True, False, True],
        [False, True, True, True],
        [True, True, True, True],
    ])
    fused, _, fallback = precision_fuse(residuals, logvars, valid)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted, _, permuted_fallback = precision_fuse(
        residuals[:, permutation], logvars[:, permutation], valid[:, permutation]
    )
    _assert_close(permuted, fused, "fusion depends on agent order")
    _assert(torch.equal(permuted_fallback, fallback), "fallback depends on order")


def test_agent_feature_isolation():
    torch.manual_seed(7)
    head = SharedObjectRefiner(2, hidden_dim=8)
    nn.init.normal_(head.residual_head.weight, std=0.1)
    features = torch.randn(2, 3, 2, 4, 5)
    valid = torch.ones(2, 3, dtype=torch.bool)
    output, logvars = head(features, valid)
    permutation = torch.tensor([2, 0, 1])
    permuted_output, permuted_logvars = head(features[:, permutation], valid[:, permutation])
    _assert_close(permuted_output, output[:, permutation], "agent features crossed slots")
    _assert_close(permuted_logvars, logvars[:, permutation], "agent logvars crossed slots")
    _assert(not torch.allclose(output[:, 0], output[:, 1]), "distinct features collapsed")


def test_residual_decode_components():
    coder = ObjectResidualCoder()
    proposal = torch.tensor([[0., 0., 1., 4., 2., 2., 0.1]])
    residual = torch.tensor([[0.1, -0.2, 0.5, math.log(1.5), math.log(0.5), 0., 0.4]])
    decoded = coder.decode(proposal, residual)
    diagonal = math.sqrt(20.0)
    expected = torch.tensor([[
        0.1 * diagonal, -0.2 * diagonal, 2.0, 6.0, 1.0, 2.0, 0.5
    ]])
    _assert_close(decoded, expected, "decode did not update all box components")


def test_no_positive_safe_zero_loss():
    head = SharedObjectRefiner(2, hidden_dim=8)
    residuals = torch.zeros(2, 2, 7, requires_grad=True)
    output = _scene(
        head,
        residuals,
        torch.zeros_like(residuals),
        torch.ones(2, 2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, 7),
    )
    loss = PactCbeaObjectStage3Loss({})(output)
    _assert_close(loss.detach(), torch.tensor(0.), "no-positive loss is not zero")
    loss.backward()
    _assert(any(p.grad is not None for p in head.parameters()), "zero loss is disconnected")


def test_positive_matching():
    proposals = torch.tensor([
        [0., 0., 0., 4., 2., 1., 0.],
        [20., 0., 0., 4., 2., 1., 0.],
    ])
    gt = torch.tensor([[0., 0., 0., 4., 2., 1., 0.]])
    matched = match_proposals_to_gt(proposals, gt, 0.5, 0.2)
    _assert(torch.equal(matched["positive_mask"], torch.tensor([True, False])), "matching mask is wrong")
    _assert_close(matched["matched_ious"], torch.tensor([1., 0.]), "matching IoU is wrong")
    shifted = torch.tensor([[2., 0., 0., 4., 2., 1., 0.]])
    fallback_iou = _pairwise_rotated_iou_torch(gt, shifted)
    _assert_close(
        fallback_iou,
        torch.tensor([[1.0 / 3.0]]),
        "pure PyTorch rotated IoU fallback is wrong",
    )


def test_per_agent_loss_mask():
    head = SharedObjectRefiner(2, hidden_dim=8)
    base = torch.zeros(1, 2, 7)
    altered = base.clone()
    altered[:, 1] = 1000.0
    valid = torch.tensor([[True, False]])
    positive = torch.tensor([True])
    target = torch.ones(1, 7)
    loss_fn = PactCbeaObjectStage3Loss({
        "fused_loss_weight": 0.0,
        "agent_loss_weight": 1.0,
        "variance_reg_weight": 0.0,
        "heteroscedastic_logvar_weight": 0.0,
    })
    first = loss_fn(_scene(head, base, torch.zeros_like(base), valid, positive, target))
    second = loss_fn(_scene(head, altered, torch.zeros_like(base), valid, positive, target))
    _assert_close(first, second, "invalid agent changed per-agent loss")


def test_fused_loss():
    head = SharedObjectRefiner(2, hidden_dim=8)
    residuals = torch.zeros(1, 1, 7)
    output = _scene(
        head,
        residuals,
        torch.zeros_like(residuals),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([True]),
        torch.zeros(1, 7),
        fused_residual=torch.ones(1, 7),
        normalized_weights=torch.ones(1, 1, 7),
        fallback_mask=torch.tensor([False]),
    )
    loss_fn = PactCbeaObjectStage3Loss({
        "fused_loss_weight": 1.0,
        "agent_loss_weight": 0.0,
        "variance_reg_weight": 0.0,
    })
    _assert(float(loss_fn(output)) > 0.0, "nonzero fused error produced zero loss")


def test_variance_clamp():
    head = SharedObjectRefiner(2, hidden_dim=8, min_log_variance=-2., max_log_variance=2.)
    features = torch.randn(1, 1, 2, 3, 3)
    valid = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        head.log_variance_head.bias.fill_(100.)
    _, upper = head(features, valid)
    _assert_close(upper, torch.full_like(upper, 2.), "upper logvar clamp failed")
    with torch.no_grad():
        head.log_variance_head.bias.fill_(-100.)
    _, lower = head(features, valid)
    _assert_close(lower, torch.full_like(lower, -2.), "lower logvar clamp failed")


def test_nan_inf_guard():
    residuals = torch.zeros(1, 1, 7)
    residuals[0, 0, 0] = float("nan")
    try:
        precision_fuse(
            residuals,
            torch.zeros_like(residuals),
            torch.ones(1, 1, dtype=torch.bool),
        )
    except FloatingPointError:
        pass
    else:
        raise AssertionError("NaN residual was silently accepted")
    head = SharedObjectRefiner(1, hidden_dim=8)
    features = torch.full((1, 1, 1, 3, 3), float("inf"))
    try:
        head(features, torch.ones(1, 1, dtype=torch.bool))
    except FloatingPointError:
        return
    raise AssertionError("Inf feature was silently accepted")


def test_base_parameters_frozen():
    model = TinyFrozenBase()
    freeze_except_object_stage3(model)
    _assert(all(not p.requires_grad for p in model.base.parameters()), "base is trainable")
    _assert(all(p.requires_grad for p in model.object_stage3_refiner.parameters()), "Stage 3 is frozen")


def test_optimizer_stage3_only():
    model = TinyFrozenBase()
    freeze_except_object_stage3(model)
    named = list(stage3_named_parameters(model))
    optimizer = torch.optim.Adam([parameter for _, parameter in named], lr=1e-3)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected_ids = {id(p) for _, p in named}
    _assert(optimizer_ids == expected_ids, "optimizer parameter set is wrong")
    _assert(all(name.startswith("object_stage3_refiner.") for name, _ in named), "base name reached optimizer")


def test_stage3_gradients_exist():
    torch.manual_seed(11)
    model = TinyFrozenBase()
    freeze_except_object_stage3(model)
    with torch.no_grad():
        nn.init.normal_(model.object_stage3_refiner.residual_head.weight, std=0.1)
        nn.init.normal_(model.object_stage3_refiner.log_variance_head.weight, std=0.1)
    residuals, logvars = model.object_stage3_refiner(
        torch.randn(2, 2, 4, 5, 5), torch.ones(2, 2, dtype=torch.bool)
    )
    (residuals.square().mean() + logvars.square().mean()).backward()
    gradients = [p.grad for p in model.object_stage3_refiner.parameters()]
    _assert(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients), "Stage 3 gradient missing/nonfinite")
    _assert(any(bool((g != 0).any()) for g in gradients), "all Stage 3 gradients are zero")


def test_base_gradients_absent():
    model = TinyFrozenBase()
    freeze_except_object_stage3(model)
    residuals, logvars = model.object_stage3_refiner(
        torch.randn(1, 2, 4, 3, 3), torch.ones(1, 2, dtype=torch.bool)
    )
    (residuals.sum() + logvars.sum()).backward()
    _assert(all(p.grad is None for p in model.base.parameters()), "base received gradients")


def test_strict_checkpoint_pass():
    source = TinyFrozenBase()
    freeze_except_object_stage3(source)
    optimizer = torch.optim.Adam(
        [parameter for _, parameter in stage3_named_parameters(source)], lr=1e-3
    )
    checkpoint = build_stage3_checkpoint(
        source, optimizer, None, 2, 9, {"test": True}, "base.pth"
    )
    target = TinyFrozenBase()
    freeze_except_object_stage3(target)
    result = strict_load_stage3_checkpoint(target, checkpoint)
    _assert(result["version"] == OBJECT_STAGE3_VERSION, "checkpoint version changed")
    for key, value in source.object_stage3_refiner.state_dict().items():
        _assert_close(value, target.object_stage3_refiner.state_dict()[key], "strict load differs at %s" % key)


def test_missing_checkpoint_key_fails():
    model = TinyFrozenBase()
    freeze_except_object_stage3(model)
    state = OrderedDict(model.object_stage3_refiner.state_dict())
    state.pop(next(iter(state)))
    checkpoint = {
        "object_stage3_version": OBJECT_STAGE3_VERSION,
        "stage3_state_dict": state,
    }
    try:
        strict_load_stage3_checkpoint(model, checkpoint)
    except RuntimeError as error:
        _assert("missing=" in str(error), "failure did not identify missing keys")
        return
    raise AssertionError("incomplete Stage 3 checkpoint was accepted")


def test_disabled_config_has_no_stage3_keys():
    original_init = HeterPyramidCollabPactCbea.__init__
    try:
        HeterPyramidCollabPactCbea.__init__ = lambda self, args: nn.Module.__init__(self)
        model = HeterPyramidCollabPactCbeaObjectStage3({
            "pact_cbea": {"object_level_stage3": {"enabled": False}}
        })
    finally:
        HeterPyramidCollabPactCbea.__init__ = original_init
    keys = list(model.state_dict())
    _assert(not model.object_stage3_enabled, "disabled flag became enabled")
    _assert(not any("object_stage3" in key for key in keys), "disabled model registered Stage 3 keys")
    missing_cfg = HeterPyramidCollabPactCbeaObjectStage3._normalize_object_stage3_cfg(None)
    _assert(missing_cfg["enabled"] is False, "missing config must default false")


def test_disabled_path_delegates_to_base():
    model = HeterPyramidCollabPactCbeaObjectStage3.__new__(
        HeterPyramidCollabPactCbeaObjectStage3
    )
    nn.Module.__init__(model)
    model.object_stage3_enabled = False
    model.object_stage3_runtime_enabled = False
    original_forward = HeterPyramidCollabPactCbea.forward
    sentinel = object()
    try:
        HeterPyramidCollabPactCbea.forward = lambda self, data: data["sentinel"]
        result = model({"sentinel": sentinel})
    finally:
        HeterPyramidCollabPactCbea.forward = original_forward
    _assert(result is sentinel, "disabled path did not delegate directly")


def test_empty_scene_and_artificial_chain():
    geometry = BEVGeometry(-10., 10., -10., 10., 1., 1.)
    sampler = RotatedBEVROISampler(geometry, roi_size=(3, 5), min_coverage=0.5)
    features = torch.stack((
        torch.ones(2, 20, 20),
        torch.full((2, 20, 20), 2.0),
    ))
    transforms = torch.eye(4).repeat(2, 1, 1)
    empty_roi, empty_valid, _ = sampler(
        features, torch.empty(0, 7), transforms
    )
    _assert(empty_roi.shape == (0, 2, 2, 3, 5), "empty scene ROI shape mismatch")
    _assert(empty_valid.shape == (0, 2), "empty scene mask shape mismatch")

    proposals = torch.tensor([[0., 0., 0., 4., 2., 1., 0.]])
    roi, valid, coverage = sampler(features, proposals, transforms)
    head = SharedObjectRefiner(2, hidden_dim=8)
    residuals, logvars = head(roi, valid)
    fused, weights, fallback = precision_fuse(residuals, logvars, valid, coverage=coverage)
    refined = ObjectResidualCoder().decode(proposals, fused)
    duplicate_boxes = refined.repeat(2, 1)
    final, scores, keep = rotated_nms_sampler_boxes(
        duplicate_boxes, torch.tensor([0.9, 0.8])
    )
    _assert(final.shape == (1, 7) and scores.shape == (1,), "artificial chain output shape mismatch")
    _assert(keep.tolist() == [0], "artificial chain NMS removed the only box")
    _assert(weights.shape == (1, 2, 7) and not bool(fallback.item()), "artificial fusion diagnostics wrong")


def test_strict_heal_yaml_and_config_pass():
    config_path = REPO_ROOT / (
        "opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/pact/object_stage3/"
        "heter_pyramid_collab_pact_cbea_object_stage3.yaml"
    )
    hypes = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args = hypes["model"]["args"]
    object_cfg = (
        HeterPyramidCollabPactCbeaObjectStage3._normalize_object_stage3_cfg(
            args["pact_cbea"]["object_level_stage3"]
        )
    )
    validator = HeterPyramidCollabPactCbeaObjectStage3._validate_strict_heal_base_config
    mode = validator(args, object_cfg)
    _assert(mode == "strict_heal_lambda_zero", "strict base mode changed")
    _assert(
        object_cfg["base_checkpoint"]
        == "opencood/logs/HEAL_m1_based/final_infer/net_epoch1.pth",
        "YAML does not select the official HEAL merge_final checkpoint",
    )
    _assert(object_cfg["stage3_checkpoint"] is None, "Stage 3 checkpoint is not null")
    _assert(object_cfg["start_from_scratch"] is True, "scratch mode was disabled")


def test_strict_heal_invalid_configs_rejected():
    variants = [
        (("pact_cbea", "enabled"), False, "pact_cbea.enabled"),
        (("pact_cbea", "trainable"), True, "pact_cbea.trainable"),
        (
            ("pact_cbea", "no_joint_training"),
            False,
            "pact_cbea.no_joint_training",
        ),
        (
            ("pact_cbea", "use_stage3_joint_training"),
            True,
            "pact_cbea.use_stage3_joint_training",
        ),
        (
            ("pact_cbea", "fusion_mode"),
            "legacy_rule",
            "pact_cbea.fusion_mode",
        ),
        (
            ("pact_cbea", "multiscale_prior", "enabled"),
            False,
            "pact_cbea.multiscale_prior.enabled",
        ),
        (
            ("pact_cbea", "multiscale_prior", "lambda"),
            0.25,
            "pact_cbea.multiscale_prior.lambda",
        ),
        (
            ("pact_cbea", "multiscale_prior", "injection_strength"),
            1.0,
            "pact_cbea.multiscale_prior.injection_strength",
        ),
        (
            ("pact_cbea", "local_evidence", "enabled"),
            True,
            "pact_cbea.local_evidence.enabled",
        ),
        (
            ("pact_cbea", "evidence_head", "enabled"),
            True,
            "pact_cbea.evidence_head.enabled",
        ),
        (
            (
                "pact_cbea",
                "evidence_head",
                "localization_uncertainty",
                "enabled",
            ),
            True,
            "pact_cbea.evidence_head.localization_uncertainty.enabled",
        ),
        (
            ("pact_cbea", "aggregation", "evidence_confidence"),
            True,
            "pact_cbea.aggregation.evidence_confidence",
        ),
        (
            ("pact_cbea", "aggregation", "uncertainty_weight"),
            True,
            "pact_cbea.aggregation.uncertainty_weight",
        ),
        (
            ("pact_cbea", "aggregation", "descriptor_consistency"),
            True,
            "pact_cbea.aggregation.descriptor_consistency",
        ),
        (
            ("pact_cbea", "aggregation", "spatial_consistency"),
            True,
            "pact_cbea.aggregation.spatial_consistency",
        ),
        (
            ("pact_cbea", "aggregation", "modality_prior"),
            True,
            "pact_cbea.aggregation.modality_prior",
        ),
        (
            ("pact_cbea", "aggregation", "localization_weight"),
            True,
            "pact_cbea.aggregation.localization_weight",
        ),
        (("supervise_single",), True, "model.args.supervise_single"),
    ]
    object_cfg = (
        HeterPyramidCollabPactCbeaObjectStage3._normalize_object_stage3_cfg(
            _strict_model_args()["pact_cbea"]["object_level_stage3"]
        )
    )
    validator = HeterPyramidCollabPactCbeaObjectStage3._validate_strict_heal_base_config
    for path, value, diagnostic in variants:
        args = _strict_model_args()
        _set_nested(args, path, value)
        try:
            validator(args, object_cfg)
        except ValueError as error:
            _assert(diagnostic in str(error), "wrong strict-config diagnostic")
        else:
            raise AssertionError("strict mode accepted %s" % diagnostic)

    args = _strict_model_args()
    _set_nested(args, ("pact_cbea", "multiscale_prior", "lambda"), 0.5)
    _set_nested(args, ("pact_cbea", "local_evidence", "enabled"), True)
    _set_nested(args, ("supervise_single",), True)
    try:
        validator(args, object_cfg)
    except ValueError as error:
        message = str(error)
        for diagnostic in (
                "pact_cbea.multiscale_prior.lambda",
                "pact_cbea.local_evidence.enabled",
                "model.args.supervise_single"):
            _assert(diagnostic in message, "validator stopped before all violations")
    else:
        raise AssertionError("multiple strict-mode violations were accepted")


def test_lambda_zero_forward_isolation_and_heal_identity():
    model = HeterPyramidCollabPactCbeaObjectStage3.__new__(
        HeterPyramidCollabPactCbeaObjectStage3
    )
    nn.Module.__init__(model)
    model.modality_name_list = ["m1"]
    model.sensor_type_dict = {"m1": "lidar"}
    model.encoder_m1 = _IdentityEncoder()
    model.backbone_m1 = _IdentityBackbone()
    model.aligner_m1 = nn.Identity()
    model.pyramid_backbone = _StrictHealPyramid()
    model.cls_head = nn.Identity()
    model.reg_head = nn.Identity()
    model.dir_head = nn.Identity()
    model.cam_crop_info = {}
    model.compress = False
    model.shrink_flag = False
    model.supervise_single = False
    model.H = 4
    model.W = 4
    model.fake_voxel_size = 1.0
    model.object_stage3_enabled = True
    model.object_stage3_runtime_enabled = True
    model.training = False
    model.pact_cbea_cfg = HeterPyramidCollabPactCbea._normalize_pact_cfg(
        _strict_model_args()["pact_cbea"]
    )
    model.pact_cbea_enabled = True
    model.pact_cbea_trainable = False
    model.pact_no_joint_training = True
    model.pact_use_stage3_joint_training = False
    model.pact_fusion_mode = "heal_multiscale_prior"
    model.pact_multiscale_prior_cfg = model.pact_cbea_cfg["multiscale_prior"]
    model.pact_cbea_rule = _CountingRule()
    model.evidence_call_count = 0

    def forbidden_evidence(instance, *args, **kwargs):
        del args, kwargs
        instance.evidence_call_count += 1
        raise AssertionError("local evidence must not run at lambda=0")

    model._compute_local_evidence = types.MethodType(forbidden_evidence, model)
    torch.manual_seed(37)
    features = torch.randn(2, 2, 4, 4)
    record_len = torch.tensor([2])
    pairwise = torch.eye(4).repeat(1, 2, 2, 1, 1)
    output = HeterPyramidCollabPactCbea.forward(model, {
        "input_features": features,
        "agent_modality_list": ["m1", "m1"],
        "record_len": record_len,
        "pairwise_t_matrix": pairwise,
    })

    reference = _StrictHealPyramid()
    heal_fused, _ = reference.forward_collab(
        features, record_len, torch.empty(0), ["m1", "m1"], {}
    )
    _assert(torch.equal(output["cls_preds"], heal_fused), "lambda=0 changed HEAL cls tensor")
    _assert(torch.equal(output["reg_preds"], heal_fused), "lambda=0 changed HEAL reg tensor")
    _assert(torch.equal(output["dir_preds"], heal_fused), "lambda=0 changed HEAL dir tensor")
    _assert(model.evidence_call_count == 0, "lambda=0 evaluated local evidence")
    _assert(model.pact_cbea_rule.call_count == 0, "lambda=0 evaluated PACT rule")
    _assert(model.pyramid_backbone.forward_collab_calls == [{}], "CBEA kwargs reached PyramidFusion")
    _assert(output["pact_cbea"]["pact_multiscale_used"] is False, "lambda=0 used multiscale prior")
    context = output.get("_pact_cbea_object_context")
    _assert(isinstance(context, dict), "object context hook was not called")
    _assert(torch.equal(context["single_feature"], features + 3.0), "context single feature changed")
    _assert(torch.equal(context["record_len"], record_len), "context record_len changed")
    _assert(torch.equal(context["pairwise_t_matrix"], pairwise), "context transform changed")


def test_strict_state_dict_and_heal_checkpoint_compatibility():
    model = _construct_strict_stage3_without_heal_modules()
    model.register_parameter(
        "base_probe", nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    )
    state_keys = list(model.state_dict())
    _assert(
        not any("pact_cbea_evidence_head" in key for key in state_keys),
        "strict state_dict contains evidence-head parameters",
    )
    _assert(
        not any("localization" in key for key in state_keys),
        "strict state_dict contains localization parameters",
    )
    _assert(
        model.object_stage3_base_mode == "strict_heal_lambda_zero",
        "strict constructor did not record base mode",
    )

    base_state = OrderedDict(
        (key, value.detach().clone())
        for key, value in model.state_dict().items()
        if not key.startswith("object_stage3_refiner.")
    )
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "heal_base.pth"
        torch.save({"state_dict": base_state}, checkpoint_path)
        result = load_base_checkpoint_compatible(
            model, str(checkpoint_path), require_complete=True
        )
    _assert(result["missing"] == [], "complete HEAL base reported missing keys")
    _assert(result["unexpected"] == [], "complete HEAL base reported unexpected keys")
    _assert(result["loaded"] == len(base_state), "not all HEAL base keys loaded")


def test_context_detach_gradient_boundary():
    model = HeterPyramidCollabPactCbeaObjectStage3.__new__(
        HeterPyramidCollabPactCbeaObjectStage3
    )
    nn.Module.__init__(model)
    model.object_stage3_enabled = True
    model.object_stage3_runtime_enabled = True
    model.object_stage3_base_mode = "strict_heal_lambda_zero"
    model.object_stage3_cfg = {
        "precision_eps": 1e-6,
        "use_coverage_weight": False,
        "refined_nms_threshold": 0.15,
        "proposal_post_nms_topk": 10,
        "positive_iou_threshold": 0.55,
        "ignore_iou_threshold": 0.35,
    }
    model.object_stage3_roi_sampler = RotatedBEVROISampler(
        BEVGeometry(-10.0, 10.0, -10.0, 10.0, 1.0, 1.0),
        roi_size=(3, 3),
        min_coverage=0.5,
    )
    model.object_stage3_refiner = SharedObjectRefiner(2, hidden_dim=8)
    model.object_stage3_coder = ObjectResidualCoder()
    source_proposal = torch.tensor(
        [[0.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0]], requires_grad=True
    )
    model.object_stage3_proposal_decoder = _FixedProposalDecoder(
        source_proposal, torch.tensor([0.9], requires_grad=True)
    )
    model.register_parameter(
        "base_probe", nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    )
    single_feature = torch.randn(1, 2, 20, 20, requires_grad=True)
    pairwise = torch.eye(4).reshape(1, 1, 1, 4, 4).requires_grad_(True)
    context = {
        "single_feature": single_feature,
        "record_len": torch.tensor([1]),
        "pairwise_t_matrix": pairwise,
        "agent_modality_list": ("m1",),
        "single_feature_frame": "per_agent_local",
        "pairwise_direction": "source_i_to_target_j",
    }

    def fake_base_forward(instance, data_dict):
        del instance, data_dict
        prediction = torch.zeros(1, 1, 1, 1)
        return {
            "cls_preds": prediction,
            "reg_preds": prediction,
            "dir_preds": prediction,
            "_pact_cbea_object_context": context,
        }

    original_forward = HeterPyramidCollabPactCbea.forward
    try:
        HeterPyramidCollabPactCbea.forward = fake_base_forward
        output = model({
            "anchor_box": torch.zeros(1, 7),
            "object_stage3_compute_targets": False,
        })
    finally:
        HeterPyramidCollabPactCbea.forward = original_forward

    scene = output["object_stage3"]["scenes"][0]
    _assert(not scene["proposals"].requires_grad, "proposal was not detached")
    loss = scene["agent_residuals"].sum() + scene["agent_log_variances"].sum()
    loss.backward()
    _assert(single_feature.grad is None, "single_feature received gradients")
    _assert(source_proposal.grad is None, "proposal source received gradients")
    _assert(pairwise.grad is None, "transform received gradients")
    _assert(model.base_probe.grad is None, "frozen base parameter received gradients")
    gradients = [parameter.grad for parameter in model.object_stage3_refiner.parameters()]
    _assert(
        all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients),
        "Stage 3 parameter gradient is missing or nonfinite",
    )
    _assert(any(bool((gradient != 0).any()) for gradient in gradients), "all Stage 3 gradients are zero")
    _assert(
        output["object_stage3"]["object_stage3_base_mode"]
        == "strict_heal_lambda_zero",
        "normal diagnostics omitted strict base mode",
    )


def test_train_mode_real_parent_guard_and_restore():
    model = HeterPyramidCollabPactCbeaObjectStage3.__new__(
        HeterPyramidCollabPactCbeaObjectStage3
    )
    nn.Module.__init__(model)
    model.modality_name_list = ["m1"]
    model.sensor_type_dict = {"m1": "lidar"}
    model.encoder_m1 = _IdentityEncoder()
    model.backbone_m1 = _IdentityBackbone()
    model.aligner_m1 = nn.Conv2d(2, 2, kernel_size=1, bias=False)
    with torch.no_grad():
        model.aligner_m1.weight.zero_()
        model.aligner_m1.weight[0, 0, 0, 0] = 1.0
        model.aligner_m1.weight[1, 1, 0, 0] = 1.0
    model.pyramid_backbone = _StrictHealPyramid()
    model.cls_head = nn.Identity()
    model.reg_head = nn.Identity()
    model.dir_head = nn.Identity()
    model.cam_crop_info = {}
    model.compress = False
    model.shrink_flag = False
    model.supervise_single = False
    model.H = 20
    model.W = 20
    model.fake_voxel_size = 1.0

    model.pact_cbea_cfg = HeterPyramidCollabPactCbea._normalize_pact_cfg(
        _strict_model_args()["pact_cbea"]
    )
    model.pact_cbea_enabled = True
    model.pact_cbea_trainable = False
    model.pact_no_joint_training = True
    model.pact_use_stage3_joint_training = False
    model.pact_fusion_mode = "heal_multiscale_prior"
    model.pact_multiscale_prior_cfg = model.pact_cbea_cfg["multiscale_prior"]
    model.pact_cbea_rule = _CountingRule()

    model.object_stage3_enabled = True
    model.object_stage3_runtime_enabled = True
    model.object_stage3_base_mode = "strict_heal_lambda_zero"
    model.object_stage3_cfg = {
        "precision_eps": 1e-6,
        "use_coverage_weight": False,
        "refined_nms_threshold": 0.15,
        "proposal_post_nms_topk": 10,
        "positive_iou_threshold": 0.55,
        "ignore_iou_threshold": 0.35,
    }
    model.object_stage3_roi_sampler = RotatedBEVROISampler(
        BEVGeometry(-10.0, 10.0, -10.0, 10.0, 1.0, 1.0),
        roi_size=(3, 3),
        min_coverage=0.5,
    )
    model.object_stage3_refiner = SharedObjectRefiner(2, hidden_dim=8)
    model.object_stage3_coder = ObjectResidualCoder()
    model.object_stage3_proposal_decoder = _FixedProposalDecoder(
        torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0]]),
        torch.tensor([0.9]),
    )

    base_modules = [
        model.encoder_m1,
        model.backbone_m1,
        model.aligner_m1,
        model.pyramid_backbone,
        model.cls_head,
        model.reg_head,
        model.dir_head,
        model.pact_cbea_rule,
    ]
    model.train(True)
    _assert(model.training is True, "top-level model did not enter train mode")
    _assert(
        model.object_stage3_refiner.training is True,
        "object Stage 3 refiner did not enter train mode",
    )
    _assert(
        all(module.training is False for module in base_modules),
        "a frozen base module entered train mode",
    )

    parent_entry_training = []
    multiscale_entry_training = []
    original_parent_forward = HeterPyramidCollabPactCbea.forward
    original_multiscale_forward = (
        HeterPyramidCollabPactCbea._forward_heal_multiscale_prior
    )

    def parent_forward_spy(instance, data_dict):
        parent_entry_training.append(instance.training)
        return original_parent_forward(instance, data_dict)

    def multiscale_forward_spy(instance, *args, **kwargs):
        multiscale_entry_training.append(instance.training)
        return original_multiscale_forward(instance, *args, **kwargs)

    HeterPyramidCollabPactCbea.forward = parent_forward_spy
    model._forward_heal_multiscale_prior = types.MethodType(
        multiscale_forward_spy, model
    )
    try:
        output = model({
            "input_features": torch.randn(1, 2, 20, 20),
            "agent_modality_list": ["m1"],
            "record_len": torch.tensor([1]),
            "pairwise_t_matrix": torch.eye(4).reshape(1, 1, 1, 4, 4),
            "anchor_box": torch.zeros(1, 7),
            "object_stage3_compute_targets": False,
        })
    finally:
        HeterPyramidCollabPactCbea.forward = original_parent_forward
        del model._forward_heal_multiscale_prior

    _assert(
        parent_entry_training == [False],
        "real parent forward did not observe training=False",
    )
    _assert(
        multiscale_entry_training == [False],
        "real multiscale guard did not observe training=False",
    )
    _assert(model.training is True, "top-level train state was not restored")
    _assert(
        model.object_stage3_refiner.training is True,
        "refiner train state changed across base forward",
    )
    _assert(
        all(module.training is False for module in base_modules),
        "a frozen base module left eval mode after forward",
    )
    _assert(
        output["object_stage3"]["scenes"][0]["proposal_count"] == 1,
        "complete Stage 3 output was not produced",
    )

    scene = output["object_stage3"]["scenes"][0]
    loss = scene["agent_residuals"].sum() + scene["agent_log_variances"].sum()
    loss.backward()
    stage3_gradients = [
        parameter.grad for parameter in model.object_stage3_refiner.parameters()
    ]
    _assert(
        all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in stage3_gradients
        ),
        "Stage 3 gradient is missing or nonfinite",
    )
    _assert(
        any(bool((gradient != 0).any()) for gradient in stage3_gradients),
        "all Stage 3 gradients are zero",
    )
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("object_stage3_refiner.")
    ]
    _assert(base_parameters, "train-mode regression lacks a base parameter")
    _assert(
        all(parameter.grad is None for parameter in base_parameters),
        "a frozen base parameter received gradients",
    )

    model.eval()
    _assert(model.training is False, "top-level model did not enter eval mode")
    _assert(
        model.object_stage3_refiner.training is False,
        "object Stage 3 refiner did not enter eval mode",
    )
    _assert(
        all(module.training is False for module in base_modules),
        "a frozen base module left eval mode during model.eval()",
    )


def test_cpu_float32():
    head = SharedObjectRefiner(3, hidden_dim=8).cpu()
    features = torch.randn(2, 2, 3, 3, 4, dtype=torch.float32)
    residuals, logvars = head(features, torch.ones(2, 2, dtype=torch.bool))
    fused, weights, _ = precision_fuse(
        residuals, logvars, torch.ones(2, 2, dtype=torch.bool)
    )
    for tensor in (residuals, logvars, fused, weights):
        _assert(tensor.device.type == "cpu", "output left CPU")
        _assert(tensor.dtype == torch.float32, "output changed float32 dtype")


TESTS = [
    ("box coder encode/decode round trip", test_box_coder_round_trip),
    ("yaw wrap near +/-pi", test_yaw_wrap_near_pi),
    ("repository box order to sampler conversion", test_box_order_conversion),
    ("shared head output shape", test_shared_head_output_shape),
    ("variable P", test_variable_proposals),
    ("variable A", test_variable_agents),
    ("A=1", test_single_agent),
    ("P=0", test_empty_proposals),
    ("invalid agent masking", test_invalid_agent_masking),
    ("fully invalid fallback", test_fully_invalid_fallback),
    ("lower variance receives higher weight", test_lower_variance_higher_weight),
    ("agent permutation invariance", test_agent_permutation_invariance),
    ("agent feature isolation", test_agent_feature_isolation),
    ("residual decode updates center/size/yaw", test_residual_decode_components),
    ("no positive proposal safe zero loss", test_no_positive_safe_zero_loss),
    ("positive matching", test_positive_matching),
    ("per-agent loss mask", test_per_agent_loss_mask),
    ("fused loss", test_fused_loss),
    ("variance clamp", test_variance_clamp),
    ("NaN/Inf guard", test_nan_inf_guard),
    ("base parameters frozen", test_base_parameters_frozen),
    ("optimizer contains only Stage 3", test_optimizer_stage3_only),
    ("gradients exist on Stage 3", test_stage3_gradients_exist),
    ("gradients absent on base", test_base_gradients_absent),
    ("Stage 3 checkpoint strict-load", test_strict_checkpoint_pass),
    ("missing Stage 3 key expected failure", test_missing_checkpoint_key_fails),
    ("disabled config has no Stage 3 keys", test_disabled_config_has_no_stage3_keys),
    ("disabled path delegates to current base", test_disabled_path_delegates_to_base),
    ("empty scene/proposal and artificial chain", test_empty_scene_and_artificial_chain),
    ("strict HEAL YAML/config accepted", test_strict_heal_yaml_and_config_pass),
    ("strict HEAL invalid configs rejected", test_strict_heal_invalid_configs_rejected),
    (
        "lambda=0 forward isolation and exact HEAL identity",
        test_lambda_zero_forward_isolation_and_heal_identity,
    ),
    (
        "strict state_dict and HEAL checkpoint compatibility",
        test_strict_state_dict_and_heal_checkpoint_compatibility,
    ),
    ("context detach gradient boundary", test_context_detach_gradient_boundary),
    (
        "train mode crosses real parent guard and restores state",
        test_train_mode_real_parent_guard_and_restore,
    ),
    ("CPU float32", test_cpu_float32),
]


def run_cuda_checks():
    """Run optional CUDA float32 and autocast checks without treating skips as pass."""
    if not torch.cuda.is_available():
        print("SKIPPED: CUDA float32 (CUDA unavailable)")
        print("SKIPPED: CUDA autocast/AMP forward (CUDA unavailable)")
        print("SKIPPED: CUDA finite gradients (CUDA unavailable)")
        return

    device = torch.device("cuda")
    model = SharedObjectRefiner(4, hidden_dim=8).to(device)
    features = torch.randn(2, 3, 4, 5, 5, device=device, requires_grad=True)
    valid = torch.ones(2, 3, dtype=torch.bool, device=device)
    residuals, logvars = model(features, valid)
    _assert(residuals.dtype == torch.float32, "CUDA float32 dtype changed")
    print("PASS: CUDA float32")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_residuals, amp_logvars = model(features, valid)
        amp_loss = amp_residuals.square().mean() + amp_logvars.square().mean()
    _assert(bool(torch.isfinite(amp_loss)), "AMP loss is nonfinite")
    print("PASS: CUDA autocast/AMP forward")
    amp_loss.backward()
    gradients = [p.grad for p in model.parameters()]
    _assert(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients), "CUDA gradient missing/nonfinite")
    print("PASS: CUDA finite gradients")


def main():
    torch.manual_seed(0)
    passed = 0
    for index, (name, test) in enumerate(TESTS, start=1):
        try:
            test()
        except Exception:
            print("FAIL: %02d %s" % (index, name))
            traceback.print_exc()
        else:
            passed += 1
            print("PASS: %02d %s" % (index, name))

    try:
        run_cuda_checks()
    except Exception:
        print("FAIL: optional CUDA/AMP checks")
        traceback.print_exc()
        return 1

    print("RESULT: %d/%d PASS" % (passed, len(TESTS)))
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())

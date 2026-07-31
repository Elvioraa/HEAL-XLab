"""CPU smoke tests for the PACT-CBEA rotated BEV ROI sampler.

The tests use metric-coordinate feature channels, so sampled values expose
x/y swaps, yaw sign errors, and transform-direction errors directly.
No dataset, checkpoint, model forward, NumPy, or geometry dependency is used.
"""

import math
import os
import sys
import traceback

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.models.sub_modules.pact_cbea_object_roi import (  # noqa: E402
    BEVGeometry,
    RotatedBEVROISampler,
    rotated_bev_roi_sample,
)


HEIGHT = 9
WIDTH = 11
GEOMETRY = BEVGeometry(
    x_min=-5.5,
    x_max=5.5,
    y_min=-4.5,
    y_max=4.5,
    resolution_x=0.5,
    resolution_y=0.5,
    feature_stride_x=2.0,
    feature_stride_y=2.0,
)


def _identity_transforms(agent_count):
    return torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(
        agent_count, 1, 1
    )


def _proposal(x=0.0, y=0.0, length=1.0, width=1.0, yaw=0.0):
    return torch.tensor(
        [x, y, 0.0, length, width, 1.5, yaw], dtype=torch.float32
    )


def _xy_features(agent_count=1):
    x = torch.arange(WIDTH, dtype=torch.float32) - 5.0
    y = torch.arange(HEIGHT, dtype=torch.float32) - 4.0
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base = torch.stack((grid_x, grid_y), dim=0)
    return base.unsqueeze(0).repeat(agent_count, 1, 1, 1)


def _assert_close(actual, expected, message, atol=1e-5):
    expected = torch.as_tensor(
        expected, dtype=actual.dtype, device=actual.device
    )
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(
            "%s\nactual:\n%s\nexpected:\n%s" % (message, actual, expected)
        )


def _check_zero_angle_xy_mapping():
    features = _xy_features()
    proposals = _proposal(length=5.0, width=3.0).unsqueeze(0)
    roi, valid, coverage = rotated_bev_roi_sample(
        features,
        proposals,
        _identity_transforms(1),
        GEOMETRY,
        roi_size=(3, 5),
    )

    expected_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).view(
        1, 5
    ).expand(3, 5)
    expected_y = torch.tensor([-1.0, 0.0, 1.0]).view(
        3, 1
    ).expand(3, 5)
    _assert_close(
        roi[0, 0, 0], expected_x,
        "zero-yaw ROI columns must increase with world x",
    )
    _assert_close(
        roi[0, 0, 1], expected_y,
        "zero-yaw ROI rows must increase with world y",
    )
    _assert_close(coverage, [[1.0]], "center ROI must have full coverage")
    if not bool(valid[0, 0]):
        raise AssertionError("center zero-yaw ROI must be valid")


def _check_positive_ninety_degree_yaw():
    features = _xy_features()
    zero = _proposal(length=5.0, width=3.0, yaw=0.0)
    positive_90 = _proposal(
        length=5.0, width=3.0, yaw=math.pi / 2.0
    )
    roi, _, _ = rotated_bev_roi_sample(
        features,
        torch.stack((zero, positive_90)),
        _identity_transforms(1),
        GEOMETRY,
        roi_size=(3, 5),
    )

    expected_rotated_x = torch.tensor([1.0, 0.0, -1.0]).view(
        3, 1
    ).expand(3, 5)
    expected_rotated_y = torch.tensor(
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    ).view(1, 5).expand(3, 5)
    _assert_close(
        roi[1, 0, 0], expected_rotated_x,
        "+90-degree yaw must map increasing ROI rows toward world -x",
    )
    _assert_close(
        roi[1, 0, 1], expected_rotated_y,
        "+90-degree yaw must map increasing ROI columns toward world +y",
    )
    if torch.allclose(roi[0], roi[1]):
        raise AssertionError("asymmetric 0- and 90-degree ROIs must differ")


def _check_agent_isolation():
    features = torch.empty(3, 2, HEIGHT, WIDTH, dtype=torch.float32)
    for agent_idx in range(3):
        features[agent_idx, 0].fill_(float(agent_idx + 1))
        features[agent_idx, 1].fill_(float((agent_idx + 1) * 10))

    roi, _, _ = rotated_bev_roi_sample(
        features,
        _proposal(length=3.0, width=3.0).unsqueeze(0),
        _identity_transforms(3),
        GEOMETRY,
        roi_size=(3, 3),
    )
    for agent_idx in range(3):
        _assert_close(
            roi[0, agent_idx, 0],
            torch.full((3, 3), float(agent_idx + 1)),
            "agent channel 0 crossed agent boundaries",
        )
        _assert_close(
            roi[0, agent_idx, 1],
            torch.full((3, 3), float((agent_idx + 1) * 10)),
            "agent channel 1 crossed agent boundaries",
        )


def _check_identity_transform():
    roi, valid, coverage = rotated_bev_roi_sample(
        _xy_features(),
        _proposal(x=2.0, y=-1.0).unsqueeze(0),
        _identity_transforms(1),
        GEOMETRY,
        roi_size=(1, 1),
    )
    _assert_close(
        roi[0, 0, :, 0, 0], [2.0, -1.0],
        "identity ego_to_agent must preserve the proposal center",
    )
    _assert_close(coverage, [[1.0]], "identity center coverage changed")
    if not bool(valid[0, 0]):
        raise AssertionError("identity center proposal must be valid")


def _check_ego_to_agent_translation():
    transform = _identity_transforms(1)
    transform[0, 0, 3] = 2.0
    transform[0, 1, 3] = -1.0
    roi, _, _ = rotated_bev_roi_sample(
        _xy_features(),
        _proposal(x=0.0, y=0.0).unsqueeze(0),
        transform,
        GEOMETRY,
        roi_size=(1, 1),
    )
    _assert_close(
        roi[0, 0, :, 0, 0], [2.0, -1.0],
        "ego-to-agent translation must sample (2,-1), not its inverse",
    )


def _check_partial_out_of_bounds():
    features = _xy_features()
    proposal = _proposal(
        x=4.5, y=0.0, length=4.0, width=1.0
    ).unsqueeze(0)
    _, valid_high, coverage = rotated_bev_roi_sample(
        features,
        proposal,
        _identity_transforms(1),
        GEOMETRY,
        roi_size=(1, 4),
        min_coverage=0.8,
    )
    _assert_close(
        coverage, [[0.75]],
        "three of four sample centers must be in normalized range",
    )
    if bool(valid_high[0, 0]):
        raise AssertionError("coverage 0.75 must fail threshold 0.8")

    _, valid_low, coverage_low = rotated_bev_roi_sample(
        features,
        proposal,
        _identity_transforms(1),
        GEOMETRY,
        roi_size=(1, 4),
        min_coverage=0.7,
    )
    _assert_close(coverage_low, coverage, "coverage changed with threshold")
    if not bool(valid_low[0, 0]):
        raise AssertionError("coverage 0.75 must pass threshold 0.7")


def _check_fully_out_of_bounds():
    roi, valid, coverage = rotated_bev_roi_sample(
        _xy_features(),
        _proposal(x=20.0, y=20.0, length=2.0, width=2.0).unsqueeze(0),
        _identity_transforms(1),
        GEOMETRY,
        roi_size=(2, 2),
    )
    _assert_close(coverage, [[0.0]], "fully outside ROI coverage must be zero")
    _assert_close(roi, torch.zeros_like(roi), "fully outside ROI must be zero")
    if bool(valid[0, 0]):
        raise AssertionError("fully outside ROI must be invalid")


def _check_empty_proposals():
    features = _xy_features(agent_count=2)
    roi, valid, coverage = rotated_bev_roi_sample(
        features,
        torch.empty(0, 7, dtype=torch.float32),
        _identity_transforms(2),
        GEOMETRY,
        roi_size=(3, 5),
    )
    if roi.shape != (0, 2, 2, 3, 5):
        raise AssertionError("empty ROI shape mismatch: %r" % (roi.shape,))
    if valid.shape != (0, 2):
        raise AssertionError("empty valid shape mismatch: %r" % (valid.shape,))
    if coverage.shape != (0, 2):
        raise AssertionError(
            "empty coverage shape mismatch: %r" % (coverage.shape,)
        )


def _check_dtype_and_device():
    features = _xy_features()
    sampler = RotatedBEVROISampler(
        GEOMETRY, roi_size=(2, 3), min_coverage=0.5
    )
    roi, valid, coverage = sampler(
        features,
        _proposal(length=3.0, width=2.0).unsqueeze(0),
        _identity_transforms(1),
    )
    if features.device.type != "cpu" or features.dtype != torch.float32:
        raise AssertionError("smoke input must be CPU float32")
    if roi.device != features.device or roi.dtype != features.dtype:
        raise AssertionError("ROI feature dtype/device was not preserved")
    if coverage.device != features.device or coverage.dtype != features.dtype:
        raise AssertionError("coverage dtype/device was not preserved")
    if valid.device != features.device or valid.dtype != torch.bool:
        raise AssertionError("valid mask must be bool on the feature device")


def _check_backward_gradient():
    features = _xy_features().clone().requires_grad_(True)
    roi, _, _ = rotated_bev_roi_sample(
        features,
        _proposal(length=3.0, width=3.0).unsqueeze(0),
        _identity_transforms(1),
        GEOMETRY,
        roi_size=(3, 3),
    )
    roi.sum().backward()
    if features.grad is None:
        raise AssertionError("agent_features gradient is missing")
    if not bool(torch.isfinite(features.grad).all()):
        raise AssertionError("agent_features gradient contains non-finite values")
    if float(features.grad.abs().sum()) <= 0.0:
        raise AssertionError("agent_features gradient must be nonzero")


def _check_multi_proposal_agent_channel_shape():
    torch.manual_seed(7)
    features = torch.randn(3, 4, HEIGHT, WIDTH, dtype=torch.float32)
    proposals = torch.stack((
        _proposal(x=-1.0, y=0.0, length=3.0, width=2.0),
        _proposal(
            x=1.0, y=1.0, length=4.0, width=2.0, yaw=math.pi / 4.0
        ),
    ))
    roi, valid, coverage = rotated_bev_roi_sample(
        features,
        proposals,
        _identity_transforms(3),
        GEOMETRY,
        roi_size=(2, 5),
    )
    if roi.shape != (2, 3, 4, 2, 5):
        raise AssertionError("multi-dimensional ROI shape mismatch")
    if valid.shape != (2, 3) or coverage.shape != (2, 3):
        raise AssertionError("multi-dimensional metadata shape mismatch")
    if not bool(valid.all()):
        raise AssertionError("interior multi-proposal test must be valid")


def _check_input_validation():
    try:
        rotated_bev_roi_sample(
            _xy_features(),
            torch.zeros(1, 6, dtype=torch.float32),
            _identity_transforms(1),
            GEOMETRY,
        )
    except ValueError as error:
        if "proposals must have shape [P,7]" not in str(error):
            raise AssertionError("proposal shape error is not explicit")
    else:
        raise AssertionError("invalid proposal shape was accepted")

    try:
        rotated_bev_roi_sample(
            _xy_features(),
            _proposal(length=-1.0).unsqueeze(0),
            _identity_transforms(1),
            GEOMETRY,
        )
    except ValueError as error:
        if "length, width, and height" not in str(error):
            raise AssertionError("proposal size error is not explicit")
    else:
        raise AssertionError("negative proposal length was accepted")


TESTS = (
    ("zero-angle x/y mapping", _check_zero_angle_xy_mapping),
    ("positive 90-degree yaw direction", _check_positive_ninety_degree_yaw),
    ("agent isolation", _check_agent_isolation),
    ("identity transformation", _check_identity_transform),
    ("ego-to-agent translation", _check_ego_to_agent_translation),
    ("partial out-of-bounds coverage", _check_partial_out_of_bounds),
    ("fully out-of-bounds zero padding", _check_fully_out_of_bounds),
    ("empty proposals", _check_empty_proposals),
    ("CPU float32 dtype/device", _check_dtype_and_device),
    ("backward gradient", _check_backward_gradient),
    ("P>1 A>1 C>1 non-square shape", _check_multi_proposal_agent_channel_shape),
    ("input validation", _check_input_validation),
)


def main():
    passed = 0
    print("PACT-CBEA object ROI smoke: CPU float32")
    for name, test in TESTS:
        try:
            test()
        except Exception:
            print("[FAIL] %s" % name)
            traceback.print_exc()
            return 1
        print("[PASS] %s" % name)
        passed += 1
    print("RESULT: %d/%d PASS" % (passed, len(TESTS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

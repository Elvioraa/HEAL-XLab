"""CPU regression tests for PACT-CBEA priors inside HEAL multiscale fusion."""

from __future__ import absolute_import, division, print_function

import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opencood.models.heter_pyramid_collab_pact_cbea import (
    HeterPyramidCollabPactCbea,
)
from opencood.models.fuse_modules.pyramid_fuse import weighted_fuse


def identity_affine(batch_size, max_cav, dtype=torch.float32):
    identity = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
    )
    return identity.view(1, 1, 1, 2, 3).repeat(
        batch_size, max_cav, max_cav, 1, 1
    )


def assert_finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise AssertionError("%s contains NaN or Inf" % name)


def max_abs_diff(left, right):
    return float(torch.max(torch.abs(left - right)).item())


def test_lambda_zero_strict_identity():
    torch.manual_seed(7)
    record_len = torch.tensor([3], dtype=torch.long)
    feature = torch.randn(3, 4, 5, 6)
    score = torch.sigmoid(torch.randn(3, 1, 5, 6)) + 1e-4
    alpha = torch.rand(3, 1, 3, 4)
    affine = identity_affine(1, 3)
    old_output = weighted_fuse(feature, score, record_len, affine, False)
    new_output = weighted_fuse(
        feature,
        score,
        record_len,
        affine,
        False,
        cbea_alpha=alpha,
        cbea_lambda=0.0,
    )
    difference = max_abs_diff(old_output, new_output)
    assert torch.equal(old_output, new_output)
    assert difference == 0.0
    print("lambda=0 torch.equal: True; max_abs_diff: %.9f" % difference)


def test_uniform_alpha_identity():
    torch.manual_seed(11)
    record_len = torch.tensor([3], dtype=torch.long)
    feature = torch.randn(3, 3, 4, 5)
    score = torch.sigmoid(torch.randn(3, 1, 4, 5)) + 1e-4
    affine = identity_affine(1, 3)
    alpha = torch.full((3, 1, 2, 3), 1.0 / 3.0)
    baseline = weighted_fuse(feature, score, record_len, affine, False)
    differences = []
    for cbea_lambda in (0.5, 1.0):
        output = weighted_fuse(
            feature,
            score,
            record_len,
            affine,
            False,
            cbea_alpha=alpha,
            cbea_lambda=cbea_lambda,
        )
        assert_finite(output, "uniform alpha output")
        difference = max_abs_diff(baseline, output)
        assert difference < 1e-7
        differences.append(difference)
    print(
        "uniform alpha max_abs_diff: lambda=0.5 %.9g; lambda=1 %.9g"
        % tuple(differences)
    )


def test_single_cav_identity():
    torch.manual_seed(13)
    record_len = torch.tensor([1], dtype=torch.long)
    feature = torch.randn(1, 2, 4, 4)
    score = torch.sigmoid(torch.randn(1, 1, 4, 4)) + 1e-4
    affine = identity_affine(1, 1)
    alpha = torch.ones(1, 1, 1, 1)
    baseline = weighted_fuse(feature, score, record_len, affine, False)
    output = weighted_fuse(
        feature,
        score,
        record_len,
        affine,
        False,
        cbea_alpha=alpha,
        cbea_lambda=1.0,
    )
    difference = max_abs_diff(baseline, output)
    assert torch.equal(baseline, output)
    print("single CAV torch.equal: True; max_abs_diff: %.9f" % difference)


def test_nonuniform_alpha_changes_output():
    record_len = torch.tensor([2], dtype=torch.long)
    feature = torch.stack((
        torch.full((1, 3, 3), 10.0),
        torch.zeros(1, 3, 3),
    ))
    score = torch.ones(2, 1, 3, 3)
    affine = identity_affine(1, 2)
    uniform = torch.full((2, 1, 1, 1), 0.5)
    nonuniform = torch.tensor([0.9, 0.1]).view(2, 1, 1, 1)
    uniform_output = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=uniform, cbea_lambda=1.0,
    )
    nonuniform_output = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=nonuniform, cbea_lambda=1.0,
    )
    assert uniform_output.shape == nonuniform_output.shape
    assert_finite(nonuniform_output, "nonuniform alpha output")
    assert not torch.equal(uniform_output, nonuniform_output)
    assert float(nonuniform_output.mean().item()) > float(uniform_output.mean().item())
    print(
        "nonuniform alpha changed output: True; uniform_mean: %.6f; "
        "nonuniform_mean: %.6f"
        % (uniform_output.mean().item(), nonuniform_output.mean().item())
    )


def test_multiple_scene_layout():
    record_len = torch.tensor([2, 3], dtype=torch.long)
    feature = torch.stack([
        torch.full((1, 2, 2), value)
        for value in (2.0, 20.0, 3.0, 30.0, 60.0)
    ])
    score = torch.ones(5, 1, 2, 2)
    alpha = torch.tensor([0.8, 0.2, 0.6, 0.3, 0.1]).view(5, 1, 1, 1)
    combined = weighted_fuse(
        feature,
        score,
        record_len,
        identity_affine(2, 3),
        False,
        cbea_alpha=alpha,
        cbea_lambda=1.0,
    )
    first = weighted_fuse(
        feature[:2],
        score[:2],
        torch.tensor([2]),
        identity_affine(1, 2),
        False,
        cbea_alpha=alpha[:2],
        cbea_lambda=1.0,
    )
    second = weighted_fuse(
        feature[2:],
        score[2:],
        torch.tensor([3]),
        identity_affine(1, 3),
        False,
        cbea_alpha=alpha[2:],
        cbea_lambda=1.0,
    )
    expected = torch.cat((first, second), dim=0)
    assert combined.shape[0] == 2
    assert torch.equal(combined, expected)
    assert_finite(combined, "multiple scene output")
    print("record_len=[2,3] scene isolation: True; output_batch: 2")


def test_exclude_threshold_default_disabled_identity():
    torch.manual_seed(17)
    record_len = torch.tensor([3], dtype=torch.long)
    feature = torch.randn(3, 3, 4, 5)
    score = torch.sigmoid(torch.randn(3, 1, 4, 5)) + 1e-4
    affine = identity_affine(1, 3)
    alpha = torch.softmax(torch.randn(3, 1, 2, 3), dim=0)
    without_threshold = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0,
    )
    with_zero_threshold = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0, cbea_exclude_threshold=0.0,
    )
    assert torch.equal(without_threshold, with_zero_threshold)
    print("exclude_threshold=0.0 (default) torch.equal to gate-only: True")


def test_exclude_threshold_excludes_low_reliability_agent():
    record_len = torch.tensor([3], dtype=torch.long)
    feature = torch.stack((
        torch.full((1, 2, 2), 100.0),
        torch.full((1, 2, 2), 10.0),
        torch.zeros(1, 2, 2),
    ))
    score = torch.ones(3, 1, 2, 2)
    affine = identity_affine(1, 3)
    # agent 2 (index 2) gets a near-zero reliability share everywhere.
    alpha = torch.tensor([0.49, 0.49, 0.02]).view(3, 1, 1, 1).expand(3, 1, 2, 2)
    excluded_output = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0, cbea_exclude_threshold=0.5,
    )
    baseline_without_weak_agent = weighted_fuse(
        feature[:2], score[:2], torch.tensor([2]), identity_affine(1, 2), False,
    )
    difference = max_abs_diff(excluded_output, baseline_without_weak_agent)
    assert difference < 1e-5
    print(
        "excluded weak agent matches (N-1)-agent baseline: True; "
        "max_abs_diff: %.9g" % difference
    )


def test_exclude_threshold_never_empties_valid_set():
    record_len = torch.tensor([2], dtype=torch.long)
    feature = torch.stack((
        torch.full((1, 2, 2), 5.0),
        torch.full((1, 2, 2), 50.0),
    ))
    score = torch.ones(2, 1, 2, 2)
    affine = identity_affine(1, 2)
    # both agents fall below a high threshold everywhere.
    alpha = torch.tensor([0.1, 0.05]).view(2, 1, 1, 1).expand(2, 1, 2, 2)
    guarded_output = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0, cbea_exclude_threshold=0.9,
    )
    gate_only_output = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0,
    )
    assert_finite(guarded_output, "all-below-threshold output")
    assert torch.equal(guarded_output, gate_only_output)
    print(
        "all-agents-below-threshold safety fallback (no empty softmax): True"
    )


def test_exclude_threshold_rejects_negative():
    record_len = torch.tensor([2], dtype=torch.long)
    feature = torch.randn(2, 2, 3, 3)
    score = torch.sigmoid(torch.randn(2, 1, 3, 3)) + 1e-4
    affine = identity_affine(1, 2)
    alpha = torch.full((2, 1, 1, 1), 0.5)
    try:
        weighted_fuse(
            feature, score, record_len, affine, False,
            cbea_alpha=alpha, cbea_lambda=1.0, cbea_exclude_threshold=-0.1,
        )
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected
    print("negative exclude_threshold rejected: True")


def test_exclude_floor_mix_boundary_identities():
    record_len = torch.tensor([3], dtype=torch.long)
    feature = torch.stack((
        torch.full((1, 2, 2), 100.0),
        torch.full((1, 2, 2), 10.0),
        torch.zeros(1, 2, 2),
    ))
    score = torch.ones(3, 1, 2, 2)
    affine = identity_affine(1, 3)
    alpha = torch.tensor([0.49, 0.49, 0.02]).view(3, 1, 1, 1).expand(3, 1, 2, 2)

    hard_only = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0, cbea_exclude_threshold=0.5,
    )
    floor_zero = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0, cbea_exclude_threshold=0.5,
        cbea_exclude_floor_mix=0.0,
    )
    assert torch.equal(hard_only, floor_zero)

    gate_only = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0,
    )
    floor_one = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0, cbea_exclude_threshold=0.5,
        cbea_exclude_floor_mix=1.0,
    )
    diff = max_abs_diff(gate_only, floor_one)
    assert diff < 1e-5
    print(
        "exclude_floor_mix boundary identities (mu=0 -> hard, mu=1 -> gate-only): "
        "True; mu=1 max_abs_diff: %.9g" % diff
    )


def test_exclude_floor_mix_monotonic_blend():
    record_len = torch.tensor([3], dtype=torch.long)
    feature = torch.stack((
        torch.full((1, 2, 2), 100.0),
        torch.full((1, 2, 2), 10.0),
        torch.zeros(1, 2, 2),
    ))
    score = torch.ones(3, 1, 2, 2)
    affine = identity_affine(1, 3)
    alpha = torch.tensor([0.49, 0.49, 0.02]).view(3, 1, 1, 1).expand(3, 1, 2, 2)

    values = []
    for mu in (0.0, 0.25, 0.5, 0.75, 1.0):
        output = weighted_fuse(
            feature, score, record_len, affine, False,
            cbea_alpha=alpha, cbea_lambda=1.0, cbea_exclude_threshold=0.5,
            cbea_exclude_floor_mix=mu,
        )
        values.append(float(output.mean().item()))
    for earlier, later in zip(values, values[1:]):
        assert later < earlier
    print("exclude_floor_mix monotonic blend across mu=0..1: True; means: %s" % values)


def test_exclude_floor_mix_inert_when_threshold_zero():
    record_len = torch.tensor([2], dtype=torch.long)
    feature = torch.randn(2, 2, 3, 3)
    score = torch.sigmoid(torch.randn(2, 1, 3, 3)) + 1e-4
    affine = identity_affine(1, 2)
    alpha = torch.full((2, 1, 1, 1), 0.5)
    baseline = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0,
    )
    with_floor_mix = weighted_fuse(
        feature, score, record_len, affine, False,
        cbea_alpha=alpha, cbea_lambda=1.0, cbea_exclude_floor_mix=0.7,
    )
    assert torch.equal(baseline, with_floor_mix)
    print("exclude_floor_mix inert when exclude_threshold=0.0: True")


def test_exclude_floor_mix_rejects_out_of_range():
    record_len = torch.tensor([2], dtype=torch.long)
    feature = torch.randn(2, 2, 3, 3)
    score = torch.sigmoid(torch.randn(2, 1, 3, 3)) + 1e-4
    affine = identity_affine(1, 2)
    alpha = torch.full((2, 1, 1, 1), 0.5)
    for bad_value in (-0.1, 1.1):
        try:
            weighted_fuse(
                feature, score, record_len, affine, False,
                cbea_alpha=alpha, cbea_lambda=1.0, cbea_exclude_threshold=0.5,
                cbea_exclude_floor_mix=bad_value,
            )
        except ValueError:
            continue
        raise AssertionError("out-of-range exclude_floor_mix was accepted")
    print("exclude_floor_mix out-of-range rejected: True")


def test_alpha_flattening():
    direct = torch.rand(1, 2, 1, 4, 5)
    flattened_direct = HeterPyramidCollabPactCbea._flatten_pact_alpha(
        {"pact_alpha": direct},
        torch.tensor([2]),
    )
    assert flattened_direct.shape == (2, 1, 4, 5)

    first = torch.rand(1, 2, 1, 3, 4)
    second = torch.rand(1, 3, 1, 3, 4)
    flattened_groups = HeterPyramidCollabPactCbea._flatten_pact_alpha(
        {
            "pact_group_debug": [
                {"pact_alpha": first},
                {"pact_alpha": second},
            ],
        },
        torch.tensor([2, 3]),
    )
    assert flattened_groups.shape == (5, 1, 3, 4)
    assert torch.equal(flattened_groups[:2], first[0])
    assert torch.equal(flattened_groups[2:], second[0])

    try:
        HeterPyramidCollabPactCbea._flatten_pact_alpha(
            {"pact_group_debug": [{"pact_alpha": first}]},
            torch.tensor([2, 3]),
        )
    except ValueError:
        mismatch_failed = True
    else:
        mismatch_failed = False
    assert mismatch_failed
    print("alpha flatten batch=1 and record_len=[2,3]: True; mismatch rejected: True")


def test_config_normalization():
    default_cfg = HeterPyramidCollabPactCbea._normalize_pact_cfg({})
    assert default_cfg["fusion_mode"] == "legacy_rule"
    assert default_cfg["multiscale_prior"] == {
        "enabled": False,
        "lambda": 0.0,
        "exclude_threshold": 0.0,
        "exclude_floor_mix": 0.0,
    }
    configured = HeterPyramidCollabPactCbea._normalize_pact_cfg({
        "fusion_mode": "heal_multiscale_prior",
        "multiscale_prior": {
            "enabled": 1, "lambda": "0.5", "exclude_threshold": "0.3",
            "exclude_floor_mix": "0.4",
        },
    })
    assert configured["multiscale_prior"]["enabled"] is True
    assert configured["multiscale_prior"]["lambda"] == 0.5
    assert configured["multiscale_prior"]["exclude_threshold"] == 0.3
    assert configured["multiscale_prior"]["exclude_floor_mix"] == 0.4
    for invalid in (
        {"fusion_mode": "invalid"},
        {"multiscale_prior": {"lambda": -0.1}},
        {"multiscale_prior": {"lambda": 1.1}},
        {"multiscale_prior": {"exclude_threshold": -0.1}},
        {"multiscale_prior": {"exclude_threshold": 1.0}},
        {"multiscale_prior": {"exclude_floor_mix": -0.1}},
        {"multiscale_prior": {"exclude_floor_mix": 1.1}},
    ):
        try:
            HeterPyramidCollabPactCbea._normalize_pact_cfg(invalid)
        except ValueError:
            continue
        raise AssertionError("invalid PACT multiscale configuration was accepted")
    print("config defaults and validation: True")


class _FakePyramidBackbone(object):
    def __init__(self):
        self.last_kwargs = None

    def forward_collab(self, spatial_features, record_len, affine_matrix,
                       agent_modality_list, cam_crop_info, **kwargs):
        del affine_matrix, agent_modality_list, cam_crop_info
        self.last_kwargs = dict(kwargs)
        scene_count = int(record_len.numel())
        return spatial_features[:scene_count], ["occ"]


class _BranchHarness(object):
    def __init__(self, cbea_lambda, local_evidence, exclude_threshold=0.0,
                 exclude_floor_mix=0.0):
        self.training = False
        self.supervise_single = False
        self.shrink_flag = False
        self.cam_crop_info = None
        self.pact_multiscale_prior_cfg = {
            "enabled": True,
            "lambda": float(cbea_lambda),
            "exclude_threshold": float(exclude_threshold),
            "exclude_floor_mix": float(exclude_floor_mix),
        }
        self.pact_cbea_cfg = {"debug": False}
        self.pact_cbea_trainable = False
        self.pact_no_joint_training = True
        self.pact_use_stage3_joint_training = False
        self.pyramid_backbone = _FakePyramidBackbone()
        self.local_evidence = local_evidence
        self.evidence_calls = 0
        self.rule_calls = 0
        self.cls_head = lambda feature: feature
        self.reg_head = lambda feature: feature
        self.dir_head = lambda feature: feature

    def _compute_local_evidence(self, feature, agent_modality_list):
        del feature, agent_modality_list
        self.evidence_calls += 1
        return self.local_evidence

    @staticmethod
    def _warp_to_ego(tensor, record_len, affine_matrix):
        del record_len, affine_matrix
        return tensor

    def pact_cbea_rule(self, feature, **kwargs):
        del kwargs
        self.rule_calls += 1
        agent_count = feature.shape[0]
        alpha = feature.new_full(
            (1, agent_count, 1, feature.shape[-2], feature.shape[-1]),
            1.0 / float(agent_count),
        )
        return feature[:1], {"pact_alpha": alpha, "pact_fallbacks": []}

    @staticmethod
    def _flatten_pact_alpha(debug, record_len):
        return HeterPyramidCollabPactCbea._flatten_pact_alpha(debug, record_len)

    @staticmethod
    def _pact_local_evidence_enabled():
        return True


def _run_model_branch(harness):
    feature = torch.randn(2, 2, 3, 3)
    evidence = feature[:, :1]
    if harness.local_evidence is not None:
        harness.local_evidence = {
            "evidence_heatmap_logits": evidence,
            "evidence_uncertainty": evidence.abs(),
        }
    output = HeterPyramidCollabPactCbea._forward_heal_multiscale_prior(
        harness,
        {},
        feature,
        feature,
        torch.tensor([2]),
        identity_affine(1, 2),
        None,
        ["m1", "m2"],
    )
    return output


def test_model_branch_selection():
    lambda_zero = _BranchHarness(0.0, local_evidence={})
    zero_output = _run_model_branch(lambda_zero)
    assert lambda_zero.evidence_calls == 0
    assert lambda_zero.rule_calls == 0
    assert lambda_zero.pyramid_backbone.last_kwargs == {}
    assert zero_output["pact_cbea"]["pact_multiscale_used"] is False
    assert zero_output["pact_cbea"]["pact_multiscale_fallbacks"] == [
        "lambda_zero_strict_heal_baseline"
    ]

    missing = _BranchHarness(0.7, local_evidence=None)
    missing_output = _run_model_branch(missing)
    assert missing.evidence_calls == 1
    assert missing.rule_calls == 0
    assert missing.pyramid_backbone.last_kwargs == {}
    assert missing_output["pact_cbea"]["pact_multiscale_used"] is False
    assert "missing_local_evidence_heal_fallback" in (
        missing_output["pact_cbea"]["pact_multiscale_fallbacks"]
    )

    active = _BranchHarness(
        1.0, local_evidence={}, exclude_threshold=0.4, exclude_floor_mix=0.6,
    )
    active_output = _run_model_branch(active)
    assert active.evidence_calls == 1
    assert active.rule_calls == 1
    assert active.pyramid_backbone.last_kwargs["cbea_lambda"] == 1.0
    assert active.pyramid_backbone.last_kwargs["cbea_exclude_threshold"] == 0.4
    assert active.pyramid_backbone.last_kwargs["cbea_exclude_floor_mix"] == 0.6
    assert active.pyramid_backbone.last_kwargs["cbea_alpha"].shape == (2, 1, 3, 3)
    assert active_output["pact_cbea"]["pact_multiscale_used"] is True
    assert active_output["pact_cbea"]["pact_multiscale_exclude_threshold"] == 0.4
    assert active_output["pact_cbea"]["pact_multiscale_exclude_floor_mix"] == 0.6
    print("model branch lambda=0/evidence fallback/active prior routing: True")


def main():
    test_lambda_zero_strict_identity()
    test_uniform_alpha_identity()
    test_single_cav_identity()
    test_nonuniform_alpha_changes_output()
    test_multiple_scene_layout()
    test_exclude_threshold_default_disabled_identity()
    test_exclude_threshold_excludes_low_reliability_agent()
    test_exclude_threshold_never_empties_valid_set()
    test_exclude_threshold_rejects_negative()
    test_exclude_floor_mix_boundary_identities()
    test_exclude_floor_mix_monotonic_blend()
    test_exclude_floor_mix_inert_when_threshold_zero()
    test_exclude_floor_mix_rejects_out_of_range()
    test_alpha_flattening()
    test_config_normalization()
    test_model_branch_selection()
    print("PACT_CBEA_MULTISCALE_PRIOR_TEST_PASS")


if __name__ == "__main__":
    main()

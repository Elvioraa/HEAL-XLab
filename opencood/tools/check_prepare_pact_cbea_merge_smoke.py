"""CPU smoke test for PACT-CBEA checkpoint merging.

Regression guard for a real bug: the official heal_tools.merge_dict() drops any
key containing 'head_m' (intended for cls_head_mX / shrink_conv_mX), which also
matches 'pact_cbea_evidence_head_mX.*'. That silently discarded every stage2
evidence head, so composed checkpoints carried only m1's evidence head and the
CBEA rule routed on randomly initialized evidence for m2/m3/m4.

These tests assert the bug exists in the official helper (so we notice if it is
ever fixed upstream) and that prepare_pact_cbea's restore + validation keep the
merged checkpoint complete.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
from collections import OrderedDict

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opencood.tools.heal_tools import merge_dict
from opencood.tools.prepare_pact_cbea import (
    _restore_evidence_heads,
    _validate_merged_evidence_heads,
)


def _fake_branch(modality):
    """Minimal state_dict shaped like a real PACT stage2 branch."""
    return OrderedDict([
        ("encoder_%s.conv.weight" % modality, torch.randn(2, 2)),
        ("backbone_%s.conv.weight" % modality, torch.randn(2, 2)),
        ("pyramid_backbone.conv.weight", torch.randn(2, 2)),
        ("cls_head.weight", torch.randn(2, 2)),
        ("pact_cbea_evidence_head_%s.stem.0.weight" % modality, torch.randn(2, 2)),
        ("pact_cbea_evidence_head_%s.evidence_heatmap_head.weight" % modality, torch.randn(1, 2)),
        ("pact_cbea_evidence_head_%s.evidence_localization_uncertainty_head.weight" % modality, torch.randn(1, 2)),
        ("pact_cbea_evidence_head_%s.evidence_localization_uncertainty_head.bias" % modality, torch.randn(1)),
    ])


def test_official_merge_loses_all_but_last_modality():
    """Reproduce the real bug using prepare_pact_cbea's exact call pattern.

    The accumulator is passed as merge_dict's FIRST argument every round, and
    that argument is the one filtered, so each round drops the evidence head
    accumulated in the previous round. Only the last merged modality (m1)
    survives.
    """
    branches = {m: _fake_branch(m) for m in ("m1", "m2", "m3", "m4")}
    merged = OrderedDict()
    for modality in ("m2", "m3", "m4", "m1"):
        merged = merge_dict(merged, branches[modality])

    survivors = [
        m for m in ("m1", "m2", "m3", "m4")
        if any(k.startswith("pact_cbea_evidence_head_%s" % m) for k in merged)
    ]
    assert survivors == ["m1"], (
        "expected only m1's evidence head to survive the unpatched merge, got %s; "
        "if upstream merge_dict changed, the restore step must be re-reviewed"
        % survivors
    )
    print("unpatched merge keeps only m1's evidence head (bug reproduced): True")


def test_restore_brings_back_all_keys():
    branch = _fake_branch("m2")
    merged = merge_dict(OrderedDict(), branch)
    backup = OrderedDict(
        (k, v) for k, v in branch.items()
        if k.startswith("pact_cbea_evidence_head_m2")
    )
    merged = _restore_evidence_heads(merged, backup)
    for key, value in backup.items():
        assert key in merged, "missing restored key %s" % key
        assert torch.equal(merged[key], value), "restored value differs for %s" % key
    loc = [k for k in merged if "localization" in k]
    assert len(loc) == 2, "expected 2 localization keys, got %d" % len(loc)
    print("restore recovers all evidence-head keys incl. localization: True")


def test_full_four_modality_merge():
    """The patched flow: back up during the loop, restore once afterwards."""
    branches = {m: _fake_branch(m) for m in ("m1", "m2", "m3", "m4")}
    merged = OrderedDict()
    backup = OrderedDict()
    for modality in ("m2", "m3", "m4", "m1"):
        sd = branches[modality]
        merged = merge_dict(merged, sd)
        prefix = "pact_cbea_evidence_head_%s" % modality
        for key, value in sd.items():
            if key.startswith(prefix):
                backup[key] = value
    merged = _restore_evidence_heads(merged, backup)

    for modality in ("m1", "m2", "m3", "m4"):
        keys = [k for k in merged if k.startswith("pact_cbea_evidence_head_%s" % modality)]
        assert len(keys) == 4, "%s has %d evidence keys, expected 4" % (modality, len(keys))
        loc = [k for k in keys if "localization" in k]
        assert len(loc) == 2, "%s has %d localization keys, expected 2" % (modality, len(loc))
        for key in keys:
            assert torch.equal(merged[key], branches[modality][key]), (
                "restored value differs from source for %s" % key
            )
    print("patched four-modality merge keeps all 4 evidence heads: True")


def test_validation_rejects_missing_head(tmp_path=None):
    """_validate_merged_evidence_heads must raise when a head is missing."""
    import tempfile

    tmpdir = tempfile.mkdtemp()
    paths = {}
    for modality in ("m1", "m2"):
        path = os.path.join(tmpdir, "%s.pth" % modality)
        torch.save(_fake_branch(modality), path)
        paths[modality] = path

    # Merged without the restore step -> m1/m2 evidence heads are missing.
    broken = OrderedDict()
    for modality in ("m2", "m1"):
        broken = merge_dict(broken, torch.load(paths[modality], map_location="cpu"))
    try:
        _validate_merged_evidence_heads(broken, paths)
    except RuntimeError as exc:
        assert "missing evidence-head parameters" in str(exc)
        print("validation rejects a checkpoint with dropped evidence heads: True")
    else:
        raise AssertionError("validation accepted a checkpoint with missing evidence heads")

    # With restore applied it must pass.
    fixed = OrderedDict()
    backup = OrderedDict()
    for modality in ("m2", "m1"):
        sd = torch.load(paths[modality], map_location="cpu")
        fixed = merge_dict(fixed, sd)
        prefix = "pact_cbea_evidence_head_%s" % modality
        for key, value in sd.items():
            if key.startswith(prefix):
                backup[key] = value
    fixed = _restore_evidence_heads(fixed, backup)
    _validate_merged_evidence_heads(fixed, paths)
    print("validation accepts a fully restored checkpoint: True")


def main():
    test_official_merge_loses_all_but_last_modality()
    test_restore_brings_back_all_keys()
    test_full_four_modality_merge()
    test_validation_rejects_missing_head()
    print("PREPARE_PACT_CBEA_MERGE_SMOKE_PASS")


if __name__ == "__main__":
    main()

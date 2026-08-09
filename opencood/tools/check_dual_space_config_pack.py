"""Repository-parser validation for the formal Dual-Space YAML pack."""

import copy
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.dual_space_config import (
    validate_dual_space_config,
)


PACK_ROOT = os.path.join(
    REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v4_DUAL_SPACE"
)
STAGED_FILES = (
    "stage1_m1.yaml",
    "stage2_m2.yaml",
    "stage2_m3.yaml",
    "stage2_m4.yaml",
    "merged_infer.yaml",
)


def normalized_non_dual(config):
    """Return a deep copy excluding profile label and Dual-Space block."""
    result = copy.deepcopy(config)
    result.pop("name", None)
    result["model"]["args"].pop("dual_space", None)
    return result


def main():
    passed = 0
    expected = []
    for profile in ("DS_V1", "DS_V2", "DS_V3"):
        expected.extend((profile, name) for name in STAGED_FILES)
    expected.append(("DS_V4", "merged_infer.yaml"))

    loaded = {}
    for profile, filename in expected:
        path = os.path.join(PACK_ROOT, profile, filename)
        try:
            config = load_yaml(path, None)
            dual = config["model"]["args"]["dual_space"]
            validate_dual_space_config(dual)
            expected_version = profile.lower()
            assert dual["version"] == expected_version
            assert dual["experiment_profile"] == expected_version
            if profile == "DS_V1":
                assert dual["multi_scale"]["enabled"] is False
                assert dual["quality"]["enabled"] is False
            if profile in ("DS_V2", "DS_V3", "DS_V4"):
                assert dual["multi_scale"]["enabled"] is True
                assert dual["multi_scale"]["fusion"] == "concat_projection"
            if profile == "DS_V2":
                assert dual["quality"]["enabled"] is False
            if profile in ("DS_V3", "DS_V4"):
                assert dual["quality"]["enabled"] is True
                assert dual["loss"]["quality_loss_weight"] == 0.05
            assert dual["remote_proposal_rescue"]["enabled"] == (
                profile == "DS_V4"
            )
            assert dual["training_proposals"]["source"] == "gt_jitter"
            loaded[(profile, filename)] = config
        except Exception as error:
            print(
                "[FAIL] %s/%s: %s: %s"
                % (profile, filename, type(error).__name__, error)
            )
        else:
            passed += 1
            print("[PASS] %s/%s" % (profile, filename))

    try:
        for filename in STAGED_FILES:
            reference = normalized_non_dual(loaded[("DS_V1", filename)])
            assert normalized_non_dual(loaded[("DS_V2", filename)]) == reference
            assert normalized_non_dual(loaded[("DS_V3", filename)]) == reference
        assert normalized_non_dual(
            loaded[("DS_V4", "merged_infer.yaml")]
        ) == normalized_non_dual(loaded[("DS_V3", "merged_infer.yaml")])
    except Exception as error:
        print("[FAIL] non-dual settings invariant: %s" % error)
    else:
        passed += 1
        print("[PASS] non-dual settings invariant across profiles")

    total = len(expected) + 1
    print("RESULT: %d/%d PASS" % (passed, total))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

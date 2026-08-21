"""Repository-parser lifecycle checks for Dual-Space extension YAML packs."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.dual_space_config import (
    resolve_v5_quality_safe_config,
    resolve_v6_residual_safe_config,
    validate_dual_space_config,
)


PACK_ROOT = os.path.join(
    REPO_ROOT, "opencood", "hypes_yaml", "HEAL_XLab_v4_DUAL_SPACE"
)
FILES = (
    "stage1_m1.yaml",
    "stage2_m2.yaml",
    "stage2_m3.yaml",
    "stage2_m4.yaml",
    "merged_infer.yaml",
)
EXPECTED = {
    "DS_V3_DIAG": {
        "diagnostics": (True, True, True, True, True),
        "v5_quality_safe": (False, False, False, False, False),
        "v6_residual_safe": (False, False, False, False, False),
    },
    "DS_V5_QUALITY_SAFE": {
        "diagnostics": (False, False, False, False, False),
        "v5_quality_safe": (False, True, True, True, False),
        "v6_residual_safe": (False, False, False, False, False),
    },
    "DS_V6_RESIDUAL_SAFE": {
        "diagnostics": (False, False, False, False, False),
        "v5_quality_safe": (False, False, False, False, False),
        "v6_residual_safe": (False, True, True, True, True),
    },
    "DS_V5_V6": {
        "diagnostics": (False, False, False, False, False),
        "v5_quality_safe": (False, True, True, True, False),
        "v6_residual_safe": (False, True, True, True, True),
    },
}


def load_pack(name):
    values = []
    for filename in FILES:
        path = os.path.join(PACK_ROOT, name, filename)
        config = load_yaml(path, None)
        dual = config["model"]["args"]["dual_space"]
        validate_dual_space_config(dual)
        values.append(dual)
    return values


def enabled(dual, extension):
    return bool(dual.get(extension, {}).get("enabled", False))


def check_pack(name):
    configs = load_pack(name)
    print(name)
    for filename, dual in zip(FILES, configs):
        print(
            "  %-18s diagnostics=%-5s v5=%-5s v6=%-5s"
            % (
                filename,
                enabled(dual, "diagnostics"),
                enabled(dual, "v5_quality_safe"),
                enabled(dual, "v6_residual_safe"),
            )
        )
    for extension, expected in EXPECTED[name].items():
        actual = tuple(enabled(dual, extension) for dual in configs)
        assert actual == expected, "%s lifecycle: %s != %s" % (
            extension, actual, expected
        )

    stage2 = configs[1:4]
    if any(enabled(dual, "v5_quality_safe") for dual in stage2):
        canonical = resolve_v5_quality_safe_config(stage2[0])
        assert all(resolve_v5_quality_safe_config(value) == canonical for value in stage2)
    if any(enabled(dual, "v6_residual_safe") for dual in stage2):
        canonical = resolve_v6_residual_safe_config(stage2[0])
        assert all(resolve_v6_residual_safe_config(value) == canonical for value in stage2)
        assert resolve_v6_residual_safe_config(configs[4]) == canonical


def main():
    passed = 0
    for name in EXPECTED:
        try:
            check_pack(name)
        except Exception as error:
            print("[FAIL] %s: %s: %s" % (name, type(error).__name__, error))
        else:
            passed += 1
            print("[PASS] %s" % name)
    print("RESULT: %d/%d PASS" % (passed, len(EXPECTED)))
    return 0 if passed == len(EXPECTED) else 1


if __name__ == "__main__":
    sys.exit(main())

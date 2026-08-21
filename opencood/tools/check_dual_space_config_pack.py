"""Repository-parser validation for the formal Dual-Space YAML pack."""

import copy
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.dual_space_config import (
    resolve_dual_space_diagnostics,
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


def architecture_signature(dual):
    """Return state-architecture settings without stage/runtime observer fields."""
    ignored = {
        "version", "experiment_profile", "mode", "active_modality",
        "allow_untrained_initialization", "remote_proposal_rescue",
        "diagnostics", "ablation", "report_stats",
    }
    return {key: copy.deepcopy(value) for key, value in dual.items() if key not in ignored}


def normalized_v1_1_single_variable(config):
    """Remove only the fields allowed to distinguish DS-V1.1 from DS-V1."""
    result = copy.deepcopy(config)
    result.pop("name", None)
    dual = result["model"]["args"]["dual_space"]
    dual.pop("version", None)
    dual.pop("experiment_profile", None)
    dual["refiner"].pop("yaw_mode", None)
    return result


def expected_stage_contract(filename):
    if filename == "stage1_m1.yaml":
        return "stage1_anchor", None, {"m1"}, True
    if filename.startswith("stage2_"):
        modality = filename[len("stage2_"):len("stage2_m2")]
        return "stage2_adapt", modality, {modality}, False
    if filename == "merged_infer.yaml":
        return "inference", None, {"m1", "m2", "m3", "m4"}, False
    raise ValueError("unknown staged config filename %s" % filename)


def print_training_settings(profile, filename, config):
    train = config.get("train_params", {})
    micro = train.get("batch_size")
    if isinstance(micro, bool) or not isinstance(micro, int) or micro < 1:
        raise ValueError("train_params.batch_size must be a positive integer")
    accumulation_explicit = "accumulate_grad_batches" in train
    amp_explicit = "amp" in train
    accumulation = train.get("accumulate_grad_batches", 1)
    amp = train.get("amp", False)
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError("train_params.accumulate_grad_batches must be a positive integer")
    if type(amp) is not bool:
        raise TypeError("train_params.amp must be bool")
    print(
        "[INFO] %s/%s micro_batch=%d accumulation=%d%s AMP=%s%s "
        "effective_batch_single_gpu=%d"
        % (
            profile, filename, micro, accumulation,
            "" if accumulation_explicit else " (implicit default)",
            amp,
            "" if amp_explicit else " (implicit default)",
            micro * accumulation,
        )
    )


def main():
    passed = 0
    expected = []
    for profile in ("DS_V1", "DS_V1_1", "DS_V2", "DS_V3"):
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
            expected_mode, expected_active, expected_modalities, allow = (
                expected_stage_contract(filename)
            )
            assert dual["mode"] == expected_mode
            assert dual.get("active_modality") == expected_active
            assert dual["allow_untrained_initialization"] is allow
            modalities = {
                key for key in config["model"]["args"]
                if key in ("m1", "m2", "m3", "m4")
            }
            assert modalities == expected_modalities
            assert dual["object_encoder"]["embedding_dim"] == 128
            if profile in ("DS_V1", "DS_V1_1"):
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
            diagnostics = resolve_dual_space_diagnostics(dual)
            if filename == "merged_infer.yaml":
                assert dual["diagnostics"]["enabled"] is False
                assert diagnostics == {
                    "enabled": False,
                    "every_n_steps": 100,
                    "max_records": 2000,
                    "match_iou_min": 0.3,
                    "thresholds": [0.3, 0.5, 0.7],
                    "improvement_epsilon": 1.0e-4,
                    "save_per_object": False,
                    "quality_target": {
                        "enabled": False,
                        "max_records": 500,
                        "dump_jsonl": True,
                    },
                    "adapter_residual": {"enabled": False},
                    "gradient_flow": {
                        "enabled": False,
                        "every_n_steps": 200,
                    },
                    "merge_ownership": {"enabled": False},
                    "inference_ablation": {
                        "enabled": False,
                        "bypass_object_adapter": False,
                        "bypass_context_adapter": False,
                        "bypass_quality_weighting": False,
                    },
                }
            print_training_settings(profile, filename, config)
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
        for profile in ("DS_V1", "DS_V1_1", "DS_V2", "DS_V3"):
            reference = architecture_signature(
                loaded[(profile, "stage1_m1.yaml")]["model"]["args"]["dual_space"]
            )
            for filename in STAGED_FILES[1:]:
                dual = loaded[(profile, filename)]["model"]["args"]["dual_space"]
                assert architecture_signature(dual) == reference
    except Exception as error:
        print("[FAIL] per-profile state architecture invariant: %s" % error)
    else:
        passed += 1
        print("[PASS] Stage1/Stage2/inference state architecture invariant")

    try:
        for filename in STAGED_FILES:
            reference = normalized_non_dual(loaded[("DS_V1", filename)])
            assert normalized_non_dual(loaded[("DS_V1_1", filename)]) == reference
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

    try:
        for filename in STAGED_FILES:
            legacy = loaded[("DS_V1", filename)]
            centered = loaded[("DS_V1_1", filename)]
            assert normalized_v1_1_single_variable(
                centered
            ) == normalized_v1_1_single_variable(legacy)
            legacy_dual = legacy["model"]["args"]["dual_space"]
            centered_dual = centered["model"]["args"]["dual_space"]
            assert legacy_dual["refiner"]["yaw_mode"] == "sin_cos"
            assert centered_dual["refiner"]["yaw_mode"] == "sin_cos_centered"
    except Exception as error:
        print("[FAIL] DS-V1.1 single-variable invariant: %s" % error)
    else:
        passed += 1
        print("[PASS] DS-V1.1 differs from DS-V1 only in experiment identity and yaw mode")

    total = len(expected) + 3
    print("RESULT: %d/%d PASS" % (passed, total))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

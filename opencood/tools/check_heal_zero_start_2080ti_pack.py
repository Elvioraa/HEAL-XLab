"""Validate the self-contained HEAL zero-start 2080 Ti config pack."""

import hashlib
from pathlib import Path
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencood.hypes_yaml import yaml_utils


PACK_ROOT = (
    REPO_ROOT
    / "opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL"
    / "zero_start_2080ti_official_equivalent"
)
MANIFEST_PATH = PACK_ROOT / "experiment_manifest.yaml"
EXPECTED_LOG_ROOT = "opencood/logs/HEAL_ZERO_START_2080TI"
MISSING = object()


def _load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _canonical_sha256(path):
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_path(data, dotted_path):
    value = data
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def _strict_equal(actual, expected):
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual == expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) \
            and actual == expected
    return actual == expected


def _difference_paths(source, target, prefix=()):
    if isinstance(source, dict) and isinstance(target, dict):
        differences = []
        for key in sorted(set(source) | set(target)):
            path = prefix + (str(key),)
            if key not in source or key not in target:
                differences.append(".".join(path))
            else:
                differences.extend(
                    _difference_paths(source[key], target[key], path)
                )
        return differences
    if source != target:
        return [".".join(prefix)]
    return []


def _iter_strings(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_strings(item, prefix + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_strings(item, prefix + (str(index),))
    elif isinstance(value, str):
        yield ".".join(prefix), value


def _manifest():
    manifest = _load_yaml(MANIFEST_PATH)
    if not isinstance(manifest, dict):
        raise AssertionError("manifest must be a mapping")
    return manifest


def test_pack_tree():
    manifest = _manifest()
    expected = {"experiment_manifest.yaml"}
    for entry in manifest["configs"].values():
        target = REPO_ROOT / entry["target"]
        try:
            relative = target.resolve().relative_to(PACK_ROOT.resolve())
        except ValueError:
            raise AssertionError("target escapes config pack: %s" % target)
        expected.add(relative.as_posix())
    actual = {
        path.relative_to(PACK_ROOT).as_posix()
        for path in PACK_ROOT.rglob("*.yaml")
    }
    if actual != expected:
        raise AssertionError(
            "unexpected YAML tree: missing=%r extra=%r"
            % (sorted(expected - actual), sorted(actual - expected))
        )
    if not (PACK_ROOT / "README.md").is_file():
        raise AssertionError("README.md is missing")


def test_repository_yaml_parser_loads_all_configs():
    manifest = _manifest()
    paths = [MANIFEST_PATH]
    paths.extend(REPO_ROOT / item["target"] for item in manifest["configs"].values())
    for path in paths:
        loaded = yaml_utils.load_yaml(str(path))
        if not isinstance(loaded, dict):
            raise AssertionError("repository parser did not return mapping: %s" % path)


def test_source_hashes_are_unchanged():
    for name, entry in _manifest()["configs"].items():
        source = REPO_ROOT / entry["source"]
        actual = _canonical_sha256(source)
        if actual != entry["source_sha256"]:
            raise AssertionError(
                "%s source hash changed: expected %s, got %s"
                % (name, entry["source_sha256"], actual)
            )


def test_only_whitelisted_config_differences():
    for name, entry in _manifest()["configs"].items():
        source = _load_yaml(REPO_ROOT / entry["source"])
        target = _load_yaml(REPO_ROOT / entry["target"])
        allowed = entry["allowed_differences"]
        actual_paths = set(_difference_paths(source, target))
        expected_paths = set(allowed)
        if actual_paths != expected_paths:
            raise AssertionError(
                "%s diff mismatch: unexpected=%r missing=%r"
                % (
                    name,
                    sorted(actual_paths - expected_paths),
                    sorted(expected_paths - actual_paths),
                )
            )
        for path, expected in allowed.items():
            actual = _get_path(target, path)
            if actual is MISSING or not _strict_equal(actual, expected):
                raise AssertionError(
                    "%s %s expected %r, got %r" % (name, path, expected, actual)
                )


def test_batch_equivalence():
    manifest = _manifest()
    for name in ("m1", "m2", "m3", "m4", "object_stage3"):
        batch = manifest["batch_equivalence"][name]
        config = _load_yaml(REPO_ROOT / manifest["configs"][name]["target"])
        micro = config["train_params"]["batch_size"]
        accumulation = config["train_params"]["accumulate_grad_batches"]
        resulting = micro * accumulation * batch["target_gpu_count"]
        if micro != batch["micro_batch"] or accumulation != batch["accumulation_steps"]:
            raise AssertionError("%s runtime batch fields disagree with manifest" % name)
        if resulting != batch["resulting_effective_batch"]:
            raise AssertionError("%s effective batch calculation failed" % name)
        if name != "object_stage3":
            official = batch["official_per_gpu_batch"] * batch["official_gpu_count"]
            if official != batch["official_global_batch"] or resulting != official:
                raise AssertionError("%s does not preserve official global batch" % name)


def test_official_training_settings_unchanged():
    manifest = _manifest()
    for name in ("m1", "m2", "m3", "m4"):
        entry = manifest["configs"][name]
        source = _load_yaml(REPO_ROOT / entry["source"])
        target = _load_yaml(REPO_ROOT / entry["target"])
        for path in (
            "train_params.epoches",
            "train_params.max_cav",
            "optimizer",
            "lr_scheduler",
            "model",
        ):
            if _get_path(source, path) != _get_path(target, path):
                raise AssertionError("%s changed official %s" % (name, path))


def test_final_inference_config():
    manifest = _manifest()
    config = _load_yaml(REPO_ROOT / manifest["configs"]["final_infer"]["target"])
    if config["train_params"]["batch_size"] != 1:
        raise AssertionError("final inference batch_size must be 1")


def test_stage3_strict_heal_config():
    manifest = _manifest()
    config = _load_yaml(REPO_ROOT / manifest["configs"]["object_stage3"]["target"])
    args = config["model"]["args"]
    pact = args["pact_cbea"]
    stage3 = pact["object_level_stage3"]
    expected = {
        "enabled": True,
        "trainable": False,
        "no_joint_training": True,
        "use_stage3_joint_training": False,
        "fusion_mode": "heal_multiscale_prior",
    }
    for key, value in expected.items():
        if not _strict_equal(pact.get(key), value):
            raise AssertionError("Stage 3 strict HEAL field failed: pact_cbea.%s" % key)
    if pact["multiscale_prior"]["enabled"] is not True:
        raise AssertionError("multiscale prior must remain enabled")
    if float(pact["multiscale_prior"]["lambda"]) != 0.0:
        raise AssertionError("strict HEAL lambda must be zero")
    if float(pact["multiscale_prior"]["injection_strength"]) != 0.0:
        raise AssertionError("strict HEAL injection strength must be zero")
    if pact["local_evidence"]["enabled"] is not False:
        raise AssertionError("local evidence must be disabled")
    if pact["evidence_head"]["enabled"] is not False:
        raise AssertionError("evidence head must be disabled")
    if pact["evidence_head"]["localization_uncertainty"]["enabled"] is not False:
        raise AssertionError("localization head must be disabled")
    if any(bool(value) for value in pact["aggregation"].values()):
        raise AssertionError("all CBEA aggregation switches must be disabled")
    if args["supervise_single"] is not False:
        raise AssertionError("supervise_single must be false")
    if stage3["enabled"] is not True or stage3["require_strict_heal_base"] is not True:
        raise AssertionError("strict object Stage 3 must remain enabled")
    if stage3["start_from_scratch"] is not True or stage3["stage3_checkpoint"] is not None:
        raise AssertionError("object refiner must start from scratch")
    expected_base = EXPECTED_LOG_ROOT + "/final_infer/net_epoch1.pth"
    if stage3["base_checkpoint"] != expected_base:
        raise AssertionError("Stage 3 base checkpoint lineage is incorrect")
    train = config["train_params"]
    if train["batch_size"] != 1 or train["accumulate_grad_batches"] != 1:
        raise AssertionError("Stage 3 batch plan must remain 1 x 1")
    if train["amp"] is not False:
        raise AssertionError("Stage 3 AMP default must remain false")


def test_no_historical_checkpoint_paths():
    manifest = _manifest()
    values = [("manifest", manifest)]
    values.extend(
        (name, _load_yaml(REPO_ROOT / entry["target"]))
        for name, entry in manifest["configs"].items()
    )
    for name, value in values:
        for path, text in _iter_strings(value):
            belongs_to_run = (
                text == EXPECTED_LOG_ROOT
                or text.startswith(EXPECTED_LOG_ROOT + "/")
            )
            is_checkpoint_path = ".pth" in text.lower()
            if is_checkpoint_path and "opencood/logs/" in text and not belongs_to_run:
                raise AssertionError(
                    "%s.%s references another log root: %s" % (name, path, text)
                )


def test_stage2_epoch_semantics_documented():
    semantics = _manifest()["stage2_epoch_semantics"]
    expected = {
        "yaml_epoches": 25,
        "loaded_init_epoch": 1,
        "actual_new_training_epochs": 24,
        "scheduler_initial_steps": 1,
        "official_readme_equivalent": True,
    }
    for key, value in expected.items():
        if not _strict_equal(semantics.get(key), value):
            raise AssertionError("Stage 2 epoch semantic mismatch: %s" % key)
    readme = (PACK_ROOT / "README.md").read_text(encoding="utf-8")
    for token in ("range(1, 25)", "24 new-modality", "pre-advanced once"):
        if token not in readme:
            raise AssertionError("README lacks Stage 2 epoch note: %s" % token)


def test_readme_commands_use_real_cli_options():
    readme = (PACK_ROOT / "README.md").read_text(encoding="utf-8")
    required_commands = (
        "opencood/tools/train.py",
        "--amp",
        "opencood/tools/heal_tools.py merge_final",
        "opencood/tools/inference_heter_in_order.py",
        "opencood/tools/check_pact_cbea_object_stage3_realdata.py",
        "--allow-scratch-stage3 --backward",
        "opencood/tools/train_pact_cbea_object_stage3.py",
        "opencood/tools/inference_pact_cbea_object_stage3.py",
    )
    for token in required_commands:
        if token not in readme:
            raise AssertionError("README command token missing: %s" % token)
    order = [
        readme.index('"$M2_MODEL_DIR"', readme.index("merge_final")),
        readme.index('"$M3_MODEL_DIR"', readme.index("merge_final")),
        readme.index('"$M4_MODEL_DIR"', readme.index("merge_final")),
        readme.index('"$M1_MODEL_DIR"', readme.index("merge_final")),
        readme.index('"$FINAL_MODEL_DIR"', readme.index("merge_final")),
    ]
    if order != sorted(order):
        raise AssertionError("README merge order is not m2, m3, m4, m1, output")
    cli_sources = {
        "train": (REPO_ROOT / "opencood/tools/train.py").read_text(encoding="utf-8"),
        "preflight": (REPO_ROOT / "opencood/tools/check_pact_cbea_object_stage3_realdata.py").read_text(encoding="utf-8"),
        "stage3_train": (REPO_ROOT / "opencood/tools/train_pact_cbea_object_stage3.py").read_text(encoding="utf-8"),
        "stage3_infer": (REPO_ROOT / "opencood/tools/inference_pact_cbea_object_stage3.py").read_text(encoding="utf-8"),
    }
    expected_options = {
        "train": ("--model_dir", "--amp", "--accumulation-steps"),
        "preflight": ("--base-checkpoint", "--allow-scratch-stage3", "--backward"),
        "stage3_train": ("--base-checkpoint", "--output-dir", "--device"),
        "stage3_infer": ("--base-checkpoint", "--stage3-checkpoint", "--output-dir"),
    }
    for script, options in expected_options.items():
        for option in options:
            if option not in cli_sources[script]:
                raise AssertionError("%s CLI option does not exist: %s" % (script, option))


TESTS = (
    ("config pack tree", test_pack_tree),
    ("repository YAML parser", test_repository_yaml_parser_loads_all_configs),
    ("source config hashes", test_source_hashes_are_unchanged),
    ("whitelisted config differences", test_only_whitelisted_config_differences),
    ("global batch equivalence", test_batch_equivalence),
    ("official training settings", test_official_training_settings_unchanged),
    ("final inference config", test_final_inference_config),
    ("strict HEAL object Stage 3", test_stage3_strict_heal_config),
    ("no historical checkpoint paths", test_no_historical_checkpoint_paths),
    ("Stage 2 epoch semantics", test_stage2_epoch_semantics_documented),
    ("README CLI commands", test_readme_commands_use_real_cli_options),
)


def main():
    passed = 0
    for index, (name, test) in enumerate(TESTS, 1):
        try:
            test()
        except Exception as exc:
            print("FAIL: %02d %s: %s" % (index, name, exc))
            return 1
        passed += 1
        print("PASS: %02d %s" % (index, name))
    print("RESULT: %d/%d PASS" % (passed, len(TESTS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

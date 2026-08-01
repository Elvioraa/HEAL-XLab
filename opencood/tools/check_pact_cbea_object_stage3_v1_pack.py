"""Validate the fixed-root PACT-CBEA Object Stage 3 v1 config pack."""

import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencood.hypes_yaml import yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.tools.inference_pact_cbea_object_stage3 import (  # noqa: E402
    _resolve_output_dir,
    parse_args as parse_inference_args,
)


CONFIG_ROOT = (
    "opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/"
    "PACT_CBEA_OBJECT_STAGE3_v1"
)
OLD_CONFIG_ROOT = (
    "opencood/hypes_yaml/HEAL_XLab_v3_HVP_HEAL/"
    "zero_start_2080ti_official_equivalent"
)
EXPERIMENT_ROOT = "opencood/logs/PACT_CBEA_OBJECT_STAGE3_v1"
PACK_ROOT = REPO_ROOT / CONFIG_ROOT
OLD_PACK_ROOT = REPO_ROOT / OLD_CONFIG_ROOT
MANIFEST_PATH = PACK_ROOT / "experiment_manifest.yaml"
README_PATH = PACK_ROOT / "README.md"
MISSING = object()

EXPECTED_TREE = {
    "experiment_manifest.yaml",
    "README.md",
    "stage1/m1_pyramid.yaml",
    "stage2/m2_single_pyramid.yaml",
    "stage2/m3_single_pyramid.yaml",
    "stage2/m4_single_pyramid.yaml",
    "merged_base/m1m2m3m4.yaml",
    "stage3/heter_pyramid_collab_pact_cbea_object_stage3.yaml",
    "final_infer/heter_pyramid_collab_pact_cbea_object_stage3.yaml",
}

EXPECTED_LOG_DIRS = {
    "m1": EXPERIMENT_ROOT + "/stage1/m1_base",
    "m2": EXPERIMENT_ROOT + "/stage2/m2_alignto_m1",
    "m3": EXPERIMENT_ROOT + "/stage2/m3_alignto_m1",
    "m4": EXPERIMENT_ROOT + "/stage2/m4_alignto_m1",
    "merge": EXPERIMENT_ROOT + "/stage2/merged_m1m2m3m4",
    "object_stage3": EXPERIMENT_ROOT + "/stage3/object_refiner",
    "final_infer": EXPERIMENT_ROOT + "/final_infer",
}

EXPECTED_NAMES = {
    "m1": "PACT_CBEA_OBJECT_STAGE3_v1/stage1/m1_base",
    "m2": "PACT_CBEA_OBJECT_STAGE3_v1/stage2/m2_alignto_m1",
    "m3": "PACT_CBEA_OBJECT_STAGE3_v1/stage2/m3_alignto_m1",
    "m4": "PACT_CBEA_OBJECT_STAGE3_v1/stage2/m4_alignto_m1",
    "merged_base": "PACT_CBEA_OBJECT_STAGE3_v1/stage2/merged_m1m2m3m4",
    "object_stage3": "PACT_CBEA_OBJECT_STAGE3_v1/stage3/object_refiner",
    "final_infer": "PACT_CBEA_OBJECT_STAGE3_v1/final_infer",
}


def _load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _manifest():
    manifest = _load_yaml(MANIFEST_PATH)
    if not isinstance(manifest, dict):
        raise AssertionError("manifest must be a mapping")
    return manifest


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
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
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


def _object_cfg(config):
    return config["model"]["args"]["pact_cbea"]["object_level_stage3"]


def test_config_tree_and_old_path_removed():
    if not PACK_ROOT.is_dir():
        raise AssertionError("new config root is missing")
    if OLD_PACK_ROOT.exists():
        raise AssertionError("old hardware-named config root still exists")
    actual = {
        path.relative_to(PACK_ROOT).as_posix()
        for path in PACK_ROOT.rglob("*")
        if path.is_file()
    }
    if actual != EXPECTED_TREE:
        raise AssertionError(
            "unexpected pack tree: missing=%r extra=%r"
            % (sorted(EXPECTED_TREE - actual), sorted(actual - EXPECTED_TREE))
        )
    final_dirs = [
        path for path in PACK_ROOT.rglob("*")
        if path.is_dir() and path.name == "final_infer"
    ]
    if final_dirs != [PACK_ROOT / "final_infer"]:
        raise AssertionError("config pack must contain exactly one final_infer")


def test_repository_yaml_parser_loads_all_configs():
    manifest = _manifest()
    paths = [MANIFEST_PATH]
    paths.extend(
        REPO_ROOT / entry["target"]
        for entry in manifest["configs"].values()
    )
    for path in paths:
        loaded = yaml_utils.load_yaml(str(path))
        if not isinstance(loaded, dict):
            raise AssertionError("repository parser failed: %s" % path)


def test_source_hashes_and_whitelisted_differences():
    for name, entry in _manifest()["configs"].items():
        source_path = REPO_ROOT / entry["source"]
        if _canonical_sha256(source_path) != entry["source_sha256"]:
            raise AssertionError("%s official source hash changed" % name)
        source = _load_yaml(source_path)
        target = _load_yaml(REPO_ROOT / entry["target"])
        actual_paths = set(_difference_paths(source, target))
        expected_paths = set(entry["allowed_differences"])
        if actual_paths != expected_paths:
            raise AssertionError(
                "%s diff mismatch: unexpected=%r missing=%r"
                % (
                    name,
                    sorted(actual_paths - expected_paths),
                    sorted(expected_paths - actual_paths),
                )
            )
        for path, expected in entry["allowed_differences"].items():
            actual = _get_path(target, path)
            if actual is MISSING or not _strict_equal(actual, expected):
                raise AssertionError(
                    "%s %s expected %r, got %r"
                    % (name, path, expected, actual)
                )


def test_names_map_to_fixed_log_directories():
    manifest = _manifest()
    for name, expected_name in EXPECTED_NAMES.items():
        config = _load_yaml(
            REPO_ROOT / manifest["configs"][name]["target"]
        )
        if config["name"] != expected_name:
            raise AssertionError("%s name is not fixed: %s" % (name, config["name"]))
        mapped = "opencood/logs/" + config["name"]
        output_key = "merge" if name == "merged_base" else name
        if mapped != EXPECTED_LOG_DIRS[output_key]:
            raise AssertionError("%s maps to %s" % (name, mapped))


def test_manifest_root_directories_and_logs():
    manifest = _manifest()
    if manifest["experiment_name"] != "PACT_CBEA_OBJECT_STAGE3_v1":
        raise AssertionError("experiment_name is incorrect")
    if manifest["experiment_root"] != EXPERIMENT_ROOT:
        raise AssertionError("experiment_root is incorrect")
    if manifest["config_root"] != CONFIG_ROOT:
        raise AssertionError("config_root is incorrect")
    if manifest["hardware"]["gpu_model_binding"] != "none":
        raise AssertionError("experiment must not bind a GPU model")

    outputs = manifest["stage_outputs"]
    for name in ("m1", "m2", "m3", "m4"):
        if outputs[name]["log_dir"] != EXPECTED_LOG_DIRS[name]:
            raise AssertionError("%s log_dir is incorrect" % name)
        if outputs[name]["train_log"] != EXPECTED_LOG_DIRS[name] + "/train.log":
            raise AssertionError("%s train_log is incorrect" % name)
    if outputs["merge"]["log_dir"] != EXPECTED_LOG_DIRS["merge"]:
        raise AssertionError("merge log_dir is incorrect")
    if outputs["merge"]["merge_log"] != EXPECTED_LOG_DIRS["merge"] + "/merge.log":
        raise AssertionError("merge_log is incorrect")
    stage3 = outputs["object_stage3"]
    if stage3["log_dir"] != EXPECTED_LOG_DIRS["object_stage3"]:
        raise AssertionError("Stage 3 log_dir is incorrect")
    if stage3["train_log"] != stage3["log_dir"] + "/train.log":
        raise AssertionError("Stage 3 train_log is incorrect")
    final = outputs["final_infer"]
    if final["output_dir"] != EXPECTED_LOG_DIRS["final_infer"]:
        raise AssertionError("final inference output_dir is incorrect")
    if final["infer_log"] != final["output_dir"] + "/infer.log":
        raise AssertionError("infer_log is incorrect")


def test_checkpoint_lineage():
    lineage = _manifest()["checkpoint_lineage"]
    m1_best = EXPECTED_LOG_DIRS["m1"] + "/net_epoch_bestval_at*.pth"
    for name in ("m2", "m3", "m4"):
        if lineage[name]["source_checkpoint"] != m1_best:
            raise AssertionError("%s does not originate from this run's m1" % name)
        expected_seed = EXPECTED_LOG_DIRS[name] + "/net_epoch1.pth"
        if lineage[name]["installed_seed"] != expected_seed:
            raise AssertionError("%s seed path is incorrect" % name)

    expected_order = [
        EXPECTED_LOG_DIRS["m2"],
        EXPECTED_LOG_DIRS["m3"],
        EXPECTED_LOG_DIRS["m4"],
        EXPECTED_LOG_DIRS["m1"],
    ]
    if lineage["merge_input_order"] != expected_order:
        raise AssertionError("merge order must be m2, m3, m4, m1")
    merged = EXPECTED_LOG_DIRS["merge"] + "/net_epoch1.pth"
    stage3 = EXPECTED_LOG_DIRS["object_stage3"] + "/stage3_best.pth"
    if lineage["merged_checkpoint"] != merged:
        raise AssertionError("merged checkpoint is not under stage2")
    if lineage["object_stage3_base"] != merged:
        raise AssertionError("Stage 3 base is not the merged checkpoint")
    if lineage["object_stage3_checkpoint"] != stage3:
        raise AssertionError("Stage 3 checkpoint path is incorrect")
    final = lineage["final_inference"]
    if final != {
            "base_checkpoint": merged,
            "stage3_checkpoint": stage3,
            "output_dir": EXPECTED_LOG_DIRS["final_infer"]}:
        raise AssertionError("final inference lineage is incorrect")


def test_batch_equivalence_and_official_settings():
    manifest = _manifest()
    for name in ("m1", "m2", "m3", "m4", "object_stage3"):
        batch = manifest["batch_equivalence"][name]
        config = _load_yaml(REPO_ROOT / manifest["configs"][name]["target"])
        micro = config["train_params"]["batch_size"]
        accumulation = config["train_params"]["accumulate_grad_batches"]
        effective = micro * accumulation * batch["execution_gpu_count"]
        if micro != batch["micro_batch"]:
            raise AssertionError("%s micro batch changed" % name)
        if accumulation != batch["accumulation_steps"]:
            raise AssertionError("%s accumulation changed" % name)
        if effective != batch["resulting_effective_batch"]:
            raise AssertionError("%s effective batch is incorrect" % name)
        if name != "object_stage3":
            official = (
                batch["official_gpu_count"] * batch["official_per_gpu_batch"]
            )
            if official != batch["official_global_batch"] or effective != official:
                raise AssertionError("%s official global batch changed" % name)

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
                "loss"):
            if _get_path(source, path) != _get_path(target, path):
                raise AssertionError("%s changed official %s" % (name, path))


def test_strict_heal_stage3_train_and_inference_configs():
    manifest = _manifest()
    train_cfg = _load_yaml(
        REPO_ROOT / manifest["configs"]["object_stage3"]["target"]
    )
    infer_cfg = _load_yaml(
        REPO_ROOT / manifest["configs"]["final_infer"]["target"]
    )
    merged = EXPECTED_LOG_DIRS["merge"] + "/net_epoch1.pth"
    stage3_checkpoint = (
        EXPECTED_LOG_DIRS["object_stage3"] + "/stage3_best.pth"
    )
    for label, config in (("train", train_cfg), ("infer", infer_cfg)):
        args = config["model"]["args"]
        pact = args["pact_cbea"]
        stage3 = _object_cfg(config)
        expected = {
            "enabled": True,
            "trainable": False,
            "no_joint_training": True,
            "use_stage3_joint_training": False,
            "fusion_mode": "heal_multiscale_prior",
        }
        for key, value in expected.items():
            if not _strict_equal(pact.get(key), value):
                raise AssertionError("%s strict HEAL field failed: %s" % (label, key))
        if float(pact["multiscale_prior"]["lambda"]) != 0.0:
            raise AssertionError("%s strict HEAL lambda changed" % label)
        if float(pact["multiscale_prior"]["injection_strength"]) != 0.0:
            raise AssertionError("%s injection strength changed" % label)
        if pact["local_evidence"]["enabled"] is not False:
            raise AssertionError("%s local evidence must be disabled" % label)
        if pact["evidence_head"]["enabled"] is not False:
            raise AssertionError("%s evidence head must be disabled" % label)
        if any(bool(value) for value in pact["aggregation"].values()):
            raise AssertionError("%s aggregation switches must be disabled" % label)
        if args["supervise_single"] is not False:
            raise AssertionError("%s supervise_single changed" % label)
        if stage3["base_checkpoint"] != merged:
            raise AssertionError("%s base checkpoint is incorrect" % label)
        train = config["train_params"]
        if train["batch_size"] != 1 or train["accumulate_grad_batches"] != 1:
            raise AssertionError("%s Stage 3 batch plan changed" % label)
    if _object_cfg(train_cfg)["stage3_checkpoint"] is not None:
        raise AssertionError("training config must start Stage 3 without a checkpoint")
    if _object_cfg(infer_cfg)["stage3_checkpoint"] != stage3_checkpoint:
        raise AssertionError("inference config Stage 3 checkpoint is incorrect")


def test_no_historical_checkpoint_or_hardware_pack_references():
    forbidden = (
        "HEAL_ZERO_START_2080TI",
        "zero_start_2080ti_official_equivalent",
        "/final_infer/net_epoch1.pth",
        "localization-quality",
        "localization_quality",
    )
    for path in PACK_ROOT.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                raise AssertionError(
                    "%s contains forbidden historical token %r"
                    % (path.relative_to(PACK_ROOT), token)
                )


def test_stage2_epoch_semantics():
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
    readme = README_PATH.read_text(encoding="utf-8")
    for token in ("range(1, 25)", "24", "pre-advanced once"):
        if token not in readme:
            raise AssertionError("README lacks Stage 2 note: %s" % token)


def test_readme_commands_logging_and_merge_order():
    readme = README_PATH.read_text(encoding="utf-8")
    bash_blocks = re.findall(r"```bash\n(.*?)```", readme, flags=re.DOTALL)
    python_blocks = [block for block in bash_blocks if "python " in block]
    if len(python_blocks) != 7:
        raise AssertionError("expected seven formal Python command blocks")
    for block in python_blocks:
        for token in ("set -o pipefail", "2>&1 | tee -a"):
            if token not in block:
                raise AssertionError("formal command lacks %s" % token)
    expected_logs = (
        "$M1_MODEL_DIR/train.log",
        "$M2_MODEL_DIR/train.log",
        "$M3_MODEL_DIR/train.log",
        "$M4_MODEL_DIR/train.log",
        "$MERGED_MODEL_DIR/merge.log",
        "$STAGE3_MODEL_DIR/train.log",
        "$FINAL_INFER_DIR/infer.log",
    )
    for token in expected_logs:
        if token not in readme:
            raise AssertionError("README log target missing: %s" % token)
    if "touch " in readme:
        raise AssertionError("README must not pre-create logs")
    merge_start = readme.index("heal_tools.py merge_final")
    order = [
        readme.index('"$M2_MODEL_DIR"', merge_start),
        readme.index('"$M3_MODEL_DIR"', merge_start),
        readme.index('"$M4_MODEL_DIR"', merge_start),
        readme.index('"$M1_MODEL_DIR"', merge_start),
        readme.index('"$MERGED_MODEL_DIR"', merge_start),
    ]
    if order != sorted(order):
        raise AssertionError("README merge order is not m2, m3, m4, m1, output")
    for path in PACK_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".log", ".pth"):
            raise AssertionError("pack must not contain runtime output: %s" % path)


def test_precreated_directory_compatibility():
    readme = README_PATH.read_text(encoding="utf-8")
    for token in (
            "test -d",
            "assert_stage_dir_safe",
            "Refusing unknown content",
            "install_config_once",
            "Refusing to overwrite a different config"):
        if token not in readme:
            raise AssertionError("README safety contract missing: %s" % token)

    train_source = (
        REPO_ROOT / "opencood/tools/train.py"
    ).read_text(encoding="utf-8")
    stage3_source = (
        REPO_ROOT / "opencood/tools/train_pact_cbea_object_stage3.py"
    ).read_text(encoding="utf-8")
    if "if opt.model_dir:" not in train_source or "saved_path = opt.model_dir" not in train_source:
        raise AssertionError("train.py does not preserve explicit model_dir")
    if "args.output_dir or _default_output_dir" not in stage3_source:
        raise AssertionError("Stage 3 trainer does not honor explicit output_dir")
    if "os.makedirs(output_dir, exist_ok=True)" not in stage3_source:
        raise AssertionError("Stage 3 trainer rejects a precreated directory")

    with tempfile.TemporaryDirectory() as directory:
        model = torch.nn.Linear(1, 1)
        epoch, returned = train_utils.load_saved_model(directory, model)
        if epoch != 0 or returned is not model:
            raise AssertionError("empty precreated model_dir is not a fresh run")


def test_inference_output_dir_and_backward_compatibility():
    explicit = os.path.join("fixed", "final_infer")
    stage3_checkpoint = os.path.join("run", "stage3", "stage3_best.pth")
    base_checkpoint = os.path.join("run", "merged", "net_epoch1.pth")
    if _resolve_output_dir(
            explicit, stage3_checkpoint, base_checkpoint) != explicit:
        raise AssertionError("explicit inference output_dir was not preserved")
    expected_stage3_fallback = os.path.dirname(
        os.path.abspath(stage3_checkpoint)
    )
    if _resolve_output_dir(
            None, stage3_checkpoint, base_checkpoint
    ) != expected_stage3_fallback:
        raise AssertionError("Stage 3 checkpoint fallback is incorrect")
    expected_base_fallback = os.path.dirname(os.path.abspath(base_checkpoint))
    if _resolve_output_dir(None, None, base_checkpoint) != expected_base_fallback:
        raise AssertionError("base checkpoint fallback is incorrect")
    args = parse_inference_args(["-y", "config.yaml"])
    if args.output_dir is not None:
        raise AssertionError("output_dir default must preserve fallback behavior")
    explicit_args = parse_inference_args(
        ["-y", "config.yaml", "--output-dir", explicit]
    )
    if explicit_args.output_dir != explicit:
        raise AssertionError("--output-dir parsing failed")

    source = (
        REPO_ROOT / "opencood/tools/inference_pact_cbea_object_stage3.py"
    ).read_text(encoding="utf-8")
    if source.count("eval_final_results(") != 2:
        raise AssertionError("unexpected evaluation output call count")
    if "baseline_stat, output_dir" not in source:
        raise AssertionError("baseline AP does not use resolved output_dir")
    if "refined_stat,\n        output_dir," not in source:
        raise AssertionError("refined AP does not use resolved output_dir")


TESTS = (
    ("config tree and old path removal", test_config_tree_and_old_path_removed),
    ("repository YAML parser", test_repository_yaml_parser_loads_all_configs),
    ("source hashes and allowed differences", test_source_hashes_and_whitelisted_differences),
    ("fixed name mappings", test_names_map_to_fixed_log_directories),
    ("manifest directories and logs", test_manifest_root_directories_and_logs),
    ("checkpoint lineage", test_checkpoint_lineage),
    ("batch and official settings", test_batch_equivalence_and_official_settings),
    ("strict HEAL Stage 3 configs", test_strict_heal_stage3_train_and_inference_configs),
    ("no historical references", test_no_historical_checkpoint_or_hardware_pack_references),
    ("Stage 2 epoch semantics", test_stage2_epoch_semantics),
    ("README logging and merge order", test_readme_commands_logging_and_merge_order),
    ("precreated directory compatibility", test_precreated_directory_compatibility),
    ("inference output directory", test_inference_output_dir_and_backward_compatibility),
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

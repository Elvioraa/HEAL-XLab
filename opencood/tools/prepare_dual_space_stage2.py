"""Prepare independent Dual-Space Stage2 experiment directories safely."""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.sub_modules.dual_space_config import (
    validate_dual_space_config,
)


STAGE2_MODALITIES = ("m2", "m3", "m4")


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the lowercase SHA256 digest of a file without loading it at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_unique_stage1_best(stage1_dir):
    """Require exactly one Stage1 ``net_epoch_bestval_at*.pth`` checkpoint."""
    stage1_dir = os.path.abspath(stage1_dir)
    matches = sorted(glob.glob(os.path.join(stage1_dir, "net_epoch_bestval_at*.pth")))
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one Stage1 bestval checkpoint in %s; found %d"
            % (stage1_dir, len(matches))
        )
    if not os.path.isfile(matches[0]):
        raise RuntimeError("Stage1 bestval checkpoint is not a regular file")
    return matches[0]


def preflight_stage2_configs(profile_dir):
    """Validate profile/version and Stage2 modality ownership before writing."""
    profile_dir = os.path.abspath(profile_dir)
    result = {}
    profile = version = None
    for modality in STAGE2_MODALITIES:
        path = os.path.join(profile_dir, "stage2_%s.yaml" % modality)
        if not os.path.isfile(path):
            raise FileNotFoundError("missing Stage2 config: %s" % path)
        config = load_yaml(path, None)
        dual = config["model"]["args"]["dual_space"]
        validate_dual_space_config(dual)
        if dual["mode"] != "stage2_adapt":
            raise ValueError("%s must use dual_space.mode=stage2_adapt" % path)
        if dual.get("active_modality") != modality:
            raise ValueError("%s must use active_modality=%s" % (path, modality))
        if dual["allow_untrained_initialization"] is not False:
            raise ValueError("%s must require a trained initialization" % path)
        current_profile = dual.get("experiment_profile", dual["version"])
        if profile is None:
            profile = current_profile
            version = dual["version"]
        if current_profile != profile or dual["version"] != version:
            raise ValueError("Stage2 configs do not share one profile/version")
        result[modality] = {
            "path": path,
            "config": config,
            "dual_space": dual,
        }
    yaw_modes = {
        item["dual_space"]["refiner"]["yaw_mode"] for item in result.values()
    }
    if len(yaw_modes) != 1:
        raise ValueError("Stage2 configs do not share one refiner.yaw_mode")
    return profile, version, result


def preflight_stage1_seed_config(
    stage1_dir, expected_profile, expected_version, expected_yaw_mode
):
    """Validate the semantic identity of the Stage1 checkpoint directory."""
    config_path = os.path.join(os.path.abspath(stage1_dir), "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            "Stage1 seed directory is missing config.yaml: %s" % config_path
        )
    config = load_yaml(config_path, None)
    try:
        dual = config["model"]["args"]["dual_space"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Stage1 seed config is missing model.args.dual_space"
        ) from error
    validate_dual_space_config(dual)
    if dual["mode"] != "stage1_anchor":
        raise ValueError("Stage1 seed config must use dual_space.mode=stage1_anchor")
    if dual["allow_untrained_initialization"] is not True:
        raise ValueError(
            "Stage1 seed config must preserve its stage1 initialization contract"
        )
    profile = dual.get("experiment_profile", dual["version"])
    yaw_mode = dual["refiner"]["yaw_mode"]
    if profile != expected_profile or dual["version"] != expected_version:
        raise ValueError(
            "Stage1 seed profile/version mismatch: expected %s/%s, got %s/%s"
            % (expected_profile, expected_version, profile, dual["version"])
        )
    if yaw_mode != expected_yaw_mode:
        raise ValueError(
            "Stage1 seed refiner.yaw_mode mismatch: expected %s, got %s"
            % (expected_yaw_mode, yaw_mode)
        )
    return {
        "path": config_path,
        "profile": profile,
        "version": dual["version"],
        "yaw_mode": yaw_mode,
    }


def prepare_dual_space_stage2(profile_dir, stage1_dir, stage2_dir):
    """Create a verified Stage2 seed and independent m2/m3/m4 directories.

    The destination must not contain any pre-existing entry.  This intentionally
    refuses partial reruns and all existing training output rather than exposing
    an unsafe force-overwrite mode.
    """
    profile_dir = os.path.abspath(profile_dir)
    stage1_dir = os.path.abspath(stage1_dir)
    stage2_dir = os.path.abspath(stage2_dir)
    stage1_best = find_unique_stage1_best(stage1_dir)
    profile, version, configs = preflight_stage2_configs(profile_dir)
    target_yaw_mode = configs[STAGE2_MODALITIES[0]]["dual_space"]["refiner"][
        "yaw_mode"
    ]
    stage1_contract = preflight_stage1_seed_config(
        stage1_dir, profile, version, target_yaw_mode
    )
    if os.path.exists(stage2_dir):
        entries = sorted(os.listdir(stage2_dir)) if os.path.isdir(stage2_dir) else [stage2_dir]
        if entries:
            raise FileExistsError(
                "Stage2 destination is not empty; refusing to overwrite: %s"
                % stage2_dir
            )
    else:
        os.makedirs(stage2_dir)

    source_hash = sha256_file(stage1_best)
    seed_path = os.path.join(stage2_dir, "net_epoch1.pth")
    shutil.copy2(stage1_best, seed_path)
    if sha256_file(seed_path) != source_hash:
        raise RuntimeError("Stage2 seed SHA256 does not match Stage1 best")

    summary = {
        "profile": profile,
        "version": version,
        "yaw_mode": target_yaw_mode,
        "mode": "stage2_adapt",
        "stage1_seed": stage1_best,
        "stage1_config": stage1_contract["path"],
        "stage2_seed": seed_path,
        "sha256": source_hash,
        "modalities": {},
    }
    for modality in STAGE2_MODALITIES:
        directory = os.path.join(stage2_dir, "%s_alignto_m1" % modality)
        os.makedirs(directory)
        config_path = os.path.join(directory, "config.yaml")
        shutil.copy2(configs[modality]["path"], config_path)
        checkpoint_path = os.path.join(directory, "net_epoch1.pth")
        relative_target = os.path.join("..", "net_epoch1.pth")
        try:
            os.symlink(relative_target, checkpoint_path)
        except (OSError, NotImplementedError) as error:
            shutil.copy2(seed_path, checkpoint_path)
            method = "copy"
            print(
                "[Stage2 Prepare] %s checkpoint: copy fallback (%s)"
                % (modality, type(error).__name__)
            )
        else:
            method = "symlink"
            print("[Stage2 Prepare] %s checkpoint: symlink" % modality)
        if sha256_file(checkpoint_path) != source_hash:
            raise RuntimeError("%s seed checkpoint SHA256 mismatch" % modality)
        copied_config = load_yaml(config_path, None)
        copied_dual = copied_config["model"]["args"]["dual_space"]
        if copied_dual["active_modality"] != modality:
            raise RuntimeError("copied %s config failed active-modality check" % modality)
        summary["modalities"][modality] = {
            "active_modality": modality,
            "config": config_path,
            "checkpoint": checkpoint_path,
            "checkpoint_method": method,
        }

    print("STAGE2_PREP_SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare verified m2/m3/m4 Dual-Space Stage2 directories"
    )
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--stage2-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    prepare_dual_space_stage2(
        args.profile_dir,
        args.stage1_dir,
        args.stage2_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

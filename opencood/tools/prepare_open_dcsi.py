"""Extract local adapters and compose strict Open-DCSI inference checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from opencood.models.sub_modules.open_dcsi.checkpoint import (
    compose_open_heterogeneous_checkpoints,
    extract_stage2_local_state,
)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_checkpoint(path):
    return torch.load(str(path), map_location="cpu")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()


def _require_input(path):
    if not path.is_file():
        raise FileNotFoundError(path)


def _require_writable_output(path, force):
    if path.exists() and not force:
        raise FileExistsError(
            "{} already exists; use --force only after checking the manifest".format(
                path
            )
        )


def _write_checkpoint_and_manifest(output, state, manifest, force):
    _require_writable_output(output, force)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(output))
    manifest["output"] = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("WROTE CHECKPOINT: {}".format(output))
    print("WROTE MANIFEST: {}".format(manifest_path))


def _extract(args):
    source = _resolve(args.checkpoint)
    output = _resolve(args.output)
    _require_input(source)
    state = extract_stage2_local_state(_load_checkpoint(source), args.modality)
    manifest = {
        "operation": "extract_stage2_local",
        "git_commit": _git_commit(),
        "modality": args.modality,
        "state_dict_keys": len(state),
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        },
    }
    print("EXTRACTED {} LOCAL KEYS: {}".format(args.modality, len(state)))
    if args.dry_run:
        print("DRY RUN: no checkpoint was written")
        return
    _write_checkpoint_and_manifest(output, state, manifest, args.force)


def _build_model(config_path):
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["model"]["core_method"] != "heter_pyramid_collab_open_dcsi":
        raise ValueError(
            "Final config must use heter_pyramid_collab_open_dcsi"
        )
    from opencood.tools.train_utils import create_model

    return create_model(config)


def _compose(args):
    config = _resolve(args.config)
    stage1 = _resolve(args.stage1)
    local_paths = {
        "m2": _resolve(args.m2),
        "m3": _resolve(args.m3),
        "m4": _resolve(args.m4),
    }
    for path in (config, stage1, *local_paths.values()):
        _require_input(path)
    model = _build_model(config)
    local_checkpoints = {
        modality: _load_checkpoint(path)
        for modality, path in local_paths.items()
    }
    result = compose_open_heterogeneous_checkpoints(
        model,
        _load_checkpoint(stage1),
        local_checkpoints,
    )
    sources = {"stage1": stage1, **local_paths}
    manifest = {
        "operation": "compose_open_heterogeneous",
        "git_commit": _git_commit(),
        "config": str(config),
        "loaded_keys": result["loaded_keys"],
        "modalities": result["modalities"],
        "sources": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in sources.items()
        },
    }
    print("STRICT COMPOSITION OK: {} keys".format(result["loaded_keys"]))
    if args.dry_run:
        print("DRY RUN: no checkpoint was written")
        return
    _write_checkpoint_and_manifest(
        _resolve(args.output), model.state_dict(), manifest, args.force
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare independently trained Open-DCSI checkpoints."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract", help="write only one Stage2 modality-local adapter state"
    )
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--modality", choices=("m2", "m3", "m4"), required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--dry-run", action="store_true")
    extract.add_argument("--force", action="store_true")
    extract.set_defaults(run=_extract)

    compose = subparsers.add_parser(
        "compose", help="strictly compose Stage1 shared and m2/m3/m4 local states"
    )
    compose.add_argument("--config", required=True)
    compose.add_argument("--stage1", required=True)
    compose.add_argument("--m2", required=True)
    compose.add_argument("--m3", required=True)
    compose.add_argument("--m4", required=True)
    compose.add_argument("--output", required=True)
    compose.add_argument("--dry-run", action="store_true")
    compose.add_argument("--force", action="store_true")
    compose.set_defaults(run=_compose)
    return parser.parse_args()


def main():
    args = _parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

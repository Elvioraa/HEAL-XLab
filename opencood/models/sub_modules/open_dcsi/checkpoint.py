"""Strict checkpoint compatibility and open-heterogeneous composition helpers."""

from collections import OrderedDict

import torch


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Open-DCSI checkpoint must contain a state dict")
    return OrderedDict(
        (
            key[len("module."):] if key.startswith("module.") else key,
            value,
        )
        for key, value in checkpoint.items()
    )


def load_official_checkpoint_compatible(model, checkpoint):
    """Load an official checkpoint, allowing only new open_dcsi.* keys to be absent."""

    checkpoint = _state_dict(checkpoint)
    model_state = model.state_dict()
    unexpected = sorted(set(checkpoint) - set(model_state))
    missing = sorted(set(model_state) - set(checkpoint))
    shape_mismatches = [
        key
        for key in set(model_state).intersection(checkpoint)
        if tuple(model_state[key].shape) != tuple(checkpoint[key].shape)
    ]
    if unexpected:
        raise RuntimeError(
            "Official checkpoint has unexpected keys: {}".format(
                ", ".join(unexpected)
            )
        )
    forbidden_missing = [key for key in missing if not key.startswith("open_dcsi.")]
    if forbidden_missing:
        raise RuntimeError(
            "Official checkpoint is missing non-Open-DCSI keys: {}".format(
                ", ".join(forbidden_missing)
            )
        )
    if shape_mismatches:
        raise RuntimeError(
            "Official checkpoint shape mismatches: {}".format(
                ", ".join(shape_mismatches)
            )
        )
    result = model.load_state_dict(checkpoint, strict=False)
    if sorted(result.missing_keys) != missing or result.unexpected_keys:
        raise RuntimeError("Open-DCSI checkpoint loader audit result changed")
    return {"missing_keys": missing, "unexpected_keys": [], "shape_mismatches": []}


def stage2_local_prefixes(modality):
    return (
        "encoder_{}".format(modality),
        "backbone_{}".format(modality),
        "aligner_{}".format(modality),
        "open_dcsi.common_projectors.{}".format(modality),
        "open_dcsi.innovation_tokenizer.tokenizers.{}".format(modality),
    )


def extract_stage2_local_state(checkpoint, modality):
    """Extract only one modality's independently trainable state."""

    checkpoint = _state_dict(checkpoint)
    prefixes = stage2_local_prefixes(modality)
    local = OrderedDict(
        (key, value) for key, value in checkpoint.items() if key.startswith(prefixes)
    )
    if not local:
        raise RuntimeError(
            "No Open-DCSI Stage2 local state found for {}".format(modality)
        )
    return local


def compose_open_heterogeneous_checkpoints(model, stage1_checkpoint, local_checkpoints):
    """Compose Stage1 shared/m1 state with independent modality-local states."""

    model_state = model.state_dict()
    composed = OrderedDict()
    stage1 = _state_dict(stage1_checkpoint)
    for key, value in stage1.items():
        if key not in model_state:
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            raise RuntimeError("Stage1 checkpoint shape mismatch at {}".format(key))
        composed[key] = value

    for modality, checkpoint in sorted(local_checkpoints.items()):
        local = extract_stage2_local_state(checkpoint, modality)
        for key, value in local.items():
            if key not in model_state:
                raise RuntimeError(
                    "Local checkpoint key is absent from final model: {}".format(key)
                )
            if key in composed:
                raise RuntimeError("Checkpoint composition overlap at {}".format(key))
            if tuple(value.shape) != tuple(model_state[key].shape):
                raise RuntimeError("Local checkpoint shape mismatch at {}".format(key))
            composed[key] = value

    missing = sorted(set(model_state) - set(composed))
    if missing:
        raise RuntimeError(
            "Composed Open-DCSI checkpoint is incomplete: {}".format(
                ", ".join(missing)
            )
        )
    result = model.load_state_dict(composed, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Strict composed checkpoint load failed")
    return {
        "loaded_keys": len(composed),
        "modalities": sorted(local_checkpoints),
        "missing_keys": [],
        "unexpected_keys": [],
    }

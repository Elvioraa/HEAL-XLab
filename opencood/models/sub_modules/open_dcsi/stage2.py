"""Independent Stage2 freezing, batch, and optimizer audits."""

import torch.nn as nn


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _configured_modalities(owner):
    return [
        name
        for name in owner.modality_name_list
        if name.startswith("m") and name[1:].isdigit()
    ]


def configure_stage2_independent(owner):
    """Freeze shared Open-DCSI modules and expose only one modality-local path."""

    if not owner.open_dcsi_enabled:
        owner._open_dcsi_filter_optimizer_parameters = False
        return
    if not owner.open_dcsi_config["stage2_independent"]["enabled"]:
        raise ValueError(
            "Open-DCSI Stage2 wrapper requires stage2_independent.enabled=true"
        )
    modalities = _configured_modalities(owner)
    if len(modalities) != 1:
        raise ValueError(
            "Open-DCSI Stage2 must configure exactly one modality, found {}".format(
                modalities
            )
        )
    modality = modalities[0]
    owner._open_dcsi_stage2_modality = modality
    owner._open_dcsi_filter_optimizer_parameters = True
    if not hasattr(owner, "open_dcsi"):
        return

    for parameter in owner.open_dcsi.parameters():
        parameter.requires_grad_(False)
    local_modules = [owner.open_dcsi.common_projectors[modality]]
    if hasattr(owner.open_dcsi, "innovation_tokenizer"):
        local_modules.append(
            owner.open_dcsi.innovation_tokenizer.tokenizers[modality]
        )
    for module in local_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    enforce_stage2_shared_eval(owner)
    audit_stage2_trainable_parameters(owner)


def stage2_allowed_parameter_prefixes(owner):
    modality = owner._open_dcsi_stage2_modality
    return (
        "encoder_{}".format(modality),
        "backbone_{}".format(modality),
        "aligner_{}".format(modality),
        "open_dcsi.common_projectors.{}".format(modality),
        "open_dcsi.innovation_tokenizer.tokenizers.{}".format(modality),
    )


def audit_stage2_trainable_parameters(owner):
    """Raise if any shared or other-modality parameter remains trainable."""

    allowed = stage2_allowed_parameter_prefixes(owner)
    trainable = []
    violations = []
    for name, parameter in owner.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable.append(name)
        if not name.startswith(allowed):
            violations.append(name)
    if violations:
        raise RuntimeError(
            "Open-DCSI Stage2 has forbidden trainable parameters: {}".format(
                ", ".join(violations)
            )
        )
    if not trainable:
        raise RuntimeError("Open-DCSI Stage2 has no trainable parameters")
    return trainable


def _shared_open_modules(owner):
    if not hasattr(owner, "open_dcsi"):
        return []
    local_names = {"common_projectors", "innovation_tokenizer"}
    return [
        module
        for name, module in owner.open_dcsi.named_children()
        if name not in local_names
    ] + [owner.open_dcsi.common_decoders]


def enforce_stage2_shared_eval(owner):
    """Keep shared BatchNorm and shared modules in eval after model.train()."""

    for module in _shared_open_modules(owner):
        module.eval()
        for child in module.modules():
            if isinstance(child, _BN_TYPES):
                child.eval()


def audit_stage2_batch(owner, data_dict):
    input_keys = [key for key in data_dict if key.startswith("inputs_")]
    expected = "inputs_{}".format(owner._open_dcsi_stage2_modality)
    if input_keys != [expected]:
        raise ValueError(
            "Open-DCSI Stage2 accepts only {}, found {}".format(
                expected, sorted(input_keys)
            )
        )
    agent_modalities = data_dict.get("agent_modality_list")
    if agent_modalities is not None and any(
        modality != owner._open_dcsi_stage2_modality
        for modality in agent_modalities
    ):
        raise ValueError("Open-DCSI Stage2 forbids heterogeneous mixed batches")


def audit_stage2_optimizer(owner, optimizer):
    allowed_ids = {
        id(parameter)
        for _, parameter in owner.named_parameters()
        if parameter.requires_grad
    }
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != allowed_ids:
        raise RuntimeError("Open-DCSI Stage2 optimizer parameter set is not exact")
    return len(optimizer_ids)

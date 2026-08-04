"""Audit Open-DCSI independent Stage2 and checkpoint composition constraints."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from opencood.tools.audit_open_dcsi_baseline_parity import (
    _collab_input,
    _single_input,
    _tiny_model_args,
)
from opencood.tools.check_open_dcsi_phase4 import _enabled_args as _phase4_args
from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.heter_pyramid_collab_open_dcsi import HeterPyramidCollabOpenDcsi
from opencood.models.heter_pyramid_collab_open_dcsi_stage1 import (
    HeterPyramidCollabOpenDcsiStage1,
)
from opencood.models.heter_pyramid_single_open_dcsi_stage2 import (
    HeterPyramidSingleOpenDcsiStage2,
)
from opencood.models.sub_modules.open_dcsi.checkpoint import (
    compose_open_heterogeneous_checkpoints,
    extract_stage2_local_state,
    load_official_checkpoint_compatible,
    stage2_local_prefixes,
)
from opencood.models.sub_modules.open_dcsi.stage2 import (
    audit_stage2_optimizer,
    audit_stage2_trainable_parameters,
)


def _args_for_modalities(modalities, stage2=False):
    args = _phase4_args()
    setting = deepcopy(args.pop("m1"))
    for modality in modalities:
        args[modality] = deepcopy(setting)
    args["open_dcsi"]["open_heterogeneous"] = {"enabled": True}
    args["open_dcsi"]["stage2_independent"] = {"enabled": bool(stage2)}
    return args


def _single_input_for(modality, batch_size=2):
    data = _single_input()
    value = data.pop("inputs_m1")
    if batch_size != 2:
        value["spatial_features"] = value["spatial_features"][:batch_size]
    return {"inputs_{}".format(modality): value}


def _hash_state(state, prefixes):
    digest = hashlib.sha256()
    for key in sorted(state):
        if key.startswith(prefixes):
            digest.update(key.encode("utf-8"))
            digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _shared_prefixes(modality):
    local = stage2_local_prefixes(modality)
    return tuple(
        prefix
        for prefix in (
            "pyramid_backbone",
            "shrink_conv",
            "cls_head",
            "reg_head",
            "dir_head",
            "open_dcsi.common_decoders",
            "open_dcsi.common_fusions",
            "open_dcsi.common_residual_gate",
            "open_dcsi.innovation_quality_router",
            "open_dcsi.innovation_aggregator",
            "open_dcsi.cross_scale_geometry",
            "open_dcsi.geometry_refiner",
        )
        if not prefix.startswith(local)
    )


def _check_stage2_freeze_optimizer_and_batch():
    torch.manual_seed(23)
    model = HeterPyramidSingleOpenDcsiStage2(
        _args_for_modalities(["m2"], stage2=True)
    )
    trainable = audit_stage2_trainable_parameters(model)
    assert trainable
    assert all(
        name.startswith(
            (
                "encoder_m2",
                "backbone_m2",
                "aligner_m2",
                "open_dcsi.common_projectors.m2",
                "open_dcsi.innovation_tokenizer.tokenizers.m2",
            )
        )
        for name in trainable
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    audit_stage2_optimizer(model, optimizer)
    model.train()
    for module_name in model.fix_modules:
        assert not getattr(model, module_name).training
    for name, module in model.open_dcsi.named_children():
        if name not in ("common_projectors", "innovation_tokenizer"):
            assert not module.training

    shared_prefixes = _shared_prefixes("m2")
    before_hash = _hash_state(model.state_dict(), shared_prefixes)
    output = model(_single_input_for("m2"))
    loss = output["cls_preds"].square().mean()
    tokens = output["open_dcsi"]["innovation_tokens"]
    loss = loss + tokens["innovation_embedding"].square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after_hash = _hash_state(model.state_dict(), shared_prefixes)
    assert before_hash == after_hash

    mixed = _single_input_for("m2")
    mixed["inputs_m1"] = {
        "spatial_features": mixed["inputs_m2"]["spatial_features"].clone()
    }
    try:
        model(mixed)
    except ValueError as error:
        assert "accepts only inputs_m2" in str(error)
    else:
        raise AssertionError("Stage2 accepted a heterogeneous mixed batch")
    print("[open-hetero] Stage2 freeze, BN, optimizer, and batch guard OK")
    return model


def _check_stage1_guard_and_dummy_modality():
    stage1_args = _args_for_modalities(["m1"], stage2=False)
    stage1 = HeterPyramidCollabOpenDcsiStage1(stage1_args).eval()
    invalid = _collab_input()
    invalid["agent_modality_list"] = ["m1", "m2"]
    try:
        stage1(invalid)
    except ValueError as error:
        assert "homogeneous m1" in str(error)
    else:
        raise AssertionError("Stage1 accepted a non-m1 agent")

    dummy = HeterPyramidSingleOpenDcsiStage2(
        _args_for_modalities(["m5"], stage2=True)
    ).eval()
    with torch.no_grad():
        output = dummy(_single_input_for("m5"))
    assert "open_dcsi" in output and "cls_preds" in output
    print("[open-hetero] Stage1 homogeneous guard and dummy m5 interface OK")


def _check_arbitrary_final_combo_and_composition(stage2_model):
    stage1 = HeterPyramidCollabOpenDcsiStage1(
        _args_for_modalities(["m1"], stage2=False)
    ).eval()
    final = HeterPyramidCollabOpenDcsi(
        _args_for_modalities(["m1", "m2"], stage2=False)
    ).eval()
    composition = compose_open_heterogeneous_checkpoints(
        final,
        stage1.state_dict(),
        {"m2": stage2_model.state_dict()},
    )
    assert composition["modalities"] == ["m2"]
    assert not composition["missing_keys"]
    local = extract_stage2_local_state(stage2_model.state_dict(), "m2")
    assert local and all(key.startswith(stage2_local_prefixes("m2")) for key in local)

    data = _collab_input()
    all_features = data["inputs_m1"]["spatial_features"]
    data["inputs_m1"] = {"spatial_features": all_features[:1]}
    data["inputs_m2"] = {"spatial_features": all_features[1:]}
    data["agent_modality_list"] = ["m1", "m2"]
    with torch.no_grad():
        output = final(data)
    assert tuple(output["cls_preds"].shape[:2]) == (1, 2)
    assert "fused_tokens" in output["open_dcsi"]
    print("[open-hetero] strict Stage1+local composition and m1/m2 final forward OK")


def _check_official_checkpoint_compatibility():
    parent = HeterPyramidCollab(_tiny_model_args("missing"))
    wrapper = HeterPyramidCollabOpenDcsiStage1(
        _args_for_modalities(["m1"], stage2=False)
    )
    audit = load_official_checkpoint_compatible(wrapper, parent.state_dict())
    assert audit["missing_keys"]
    assert all(key.startswith("open_dcsi.") for key in audit["missing_keys"])
    assert not audit["unexpected_keys"] and not audit["shape_mismatches"]
    print("[open-hetero] official checkpoint compatibility audit OK")


def main():
    stage2_model = _check_stage2_freeze_optimizer_and_batch()
    _check_stage1_guard_and_dummy_modality()
    _check_arbitrary_final_combo_and_composition(stage2_model)
    _check_official_checkpoint_compatibility()
    print("OPEN_DCSI_OPEN_HETEROGENEOUS_AUDIT_PASS")


if __name__ == "__main__":
    main()

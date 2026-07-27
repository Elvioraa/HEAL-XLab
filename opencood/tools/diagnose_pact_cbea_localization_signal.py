"""Diagnose whether the CBEA localization-quality signal actually discriminates.

Collaborative AP was identical with aggregation.localization_weight on and off
(differences ~1e-4). The suspected reason is structural: L_i = exp(-u_i^loc)
enters the reliability product and is then normalized away,

    alpha_i = (R'_i * L_i) / sum_j (R'_j * L_j)

so if L_i is close to uniform across agents it cancels exactly and alpha is
unchanged, no matter what the AP says.

This script runs a few frames through the collaborative model, captures the
rule's internal factors, and reports:

  1. Per-modality statistics of the raw localization uncertainty u^loc and of
     L = exp(-clamp(u^loc)) - does the signal differ across modalities at all?
  2. The cross-agent spread of L at each BEV location - does it discriminate
     between agents, which is the only thing normalization preserves?
  3. The decisive metric: alpha recomputed WITHOUT the L factor, compared
     against the alpha the rule actually produced. If the max/mean difference
     is ~0, the localization term is being normalized away and no amount of
     inference-side tuning will change AP; the fix has to be on the training
     side (larger loss weight, per-modality calibration) or in how the signal
     is injected.

Usage (server, needs GPU + dataset):

    python opencood/tools/diagnose_pact_cbea_localization_signal.py \
        --model_dir opencood/logs/PACT_CBEA_LOCALIZATION_QUALITY_v2/infer_loc_on \
        --frames 20

The model_dir must contain config.yaml + the composed net_epoch*.pth, and its
config must have aggregation.localization_weight enabled (otherwise the rule
returns ones and there is nothing to diagnose - the script checks this).
"""

from __future__ import absolute_import, division, print_function

import argparse
import importlib
import os
import statistics
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.tools.pact_cbea_alpha_utils import alpha_from_reliability
from opencood.utils.common_utils import update_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose the CBEA localization-quality signal.",
    )
    parser.add_argument("--model_dir", required=True,
                        help="dir with config.yaml and the composed checkpoint")
    parser.add_argument("--frames", type=int, default=20,
                        help="number of frames to accumulate statistics over")
    parser.add_argument("--range", type=str, default="204.8,102.4",
                        help="evaluation range, matching inference_heter_in_order")
    return parser.parse_args()


def _to_stats(tensor):
    """mean/std/min/max of a tensor as plain floats."""
    flat = tensor.detach().float().reshape(-1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()) if flat.numel() > 1 else 0.0,
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


def main():
    opt = parse_args()

    hypes = load_yaml(None, opt)

    # Mirror inference_heter_in_order's range / mapping handling so the
    # diagnosis runs on the same setup the AP numbers came from.
    if "heter" in hypes:
        x_max = eval(opt.range.split(",")[0])
        y_max = eval(opt.range.split(",")[1])
        new_cav_range = [
            -x_max, -y_max,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max, y_max,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range,
        })
        hypes = update_dict(hypes, {
            "mapping_dict": {"m1": "m1", "m2": "m2", "m3": "m3", "m4": "m4"},
        })
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                hypes = func(hypes)
                break

    hypes["validate_dir"] = hypes["test_dir"]
    hypes["comm_range"] = 180
    hypes["heter"]["assignment_path"] = \
        hypes["heter"]["assignment_path"].replace(".json", "_in_order.json")
    hypes = update_dict(hypes, {"ego_modality": "m1"})
    if hypes["fusion"]["core_method"].endswith("infer") is False:
        hypes["fusion"]["core_method"] += "infer"

    print("Creating Model")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rule = getattr(model, "pact_cbea_rule", None)
    if rule is None:
        raise RuntimeError("model has no pact_cbea_rule; is pact_cbea enabled?")
    if not rule.cfg["aggregation"].get("localization_weight", False):
        raise RuntimeError(
            "aggregation.localization_weight is disabled in this config, so the "
            "rule returns ones for L and there is nothing to diagnose. Point "
            "--model_dir at the 'on' config."
        )

    print("Loading Model from checkpoint")
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    print("resume from %d epoch." % resume_epoch)
    model.to(device)
    model.eval()

    print("Dataset Building")
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=4,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)

    # Capture the rule's internal debug dict without touching production code.
    captured = {}
    original_forward = rule.forward

    def recording_forward(*args, **kwargs):
        enhanced, debug = original_forward(*args, **kwargs)
        captured["debug"] = debug
        captured["modality_names"] = kwargs.get("modality_names")
        return enhanced, debug

    rule.forward = recording_forward

    # modality -> list of per-frame stats
    loc_unc_by_modality = {}
    loc_weight_by_modality = {}
    alpha_abs_deltas = []
    alpha_rel_deltas = []
    spread_of_L = []
    frames_used = 0

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if frames_used >= opt.frames:
                break
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            model(batch_data["ego"])

            debug = captured.get("debug")
            if not debug:
                continue
            # Multi-scene batches merge per-group debug; take the flat fields
            # when present, otherwise the first group.
            if "pact_localization_weight" not in debug:
                groups = debug.get("pact_group_debug") or []
                if not groups:
                    continue
                debug = groups[0]
            if "pact_localization_weight" not in debug:
                continue

            loc_weight = debug["pact_localization_weight"]      # [B,N,1,H,W]
            reliability = debug["pact_reliability"]
            alpha_actual = debug["pact_alpha"]
            names = captured.get("modality_names") or []

            # 1/2. per-modality stats and cross-agent spread of L
            agent_count = loc_weight.shape[1]
            for agent_idx in range(agent_count):
                name = str(names[agent_idx]) if agent_idx < len(names) else "agent%d" % agent_idx
                L = loc_weight[:, agent_idx]
                # invert L = exp(-u) to recover u for readability
                u = -torch.log(torch.clamp(L, min=1e-12))
                loc_weight_by_modality.setdefault(name, []).append(_to_stats(L))
                loc_unc_by_modality.setdefault(name, []).append(_to_stats(u))
            if agent_count > 1:
                # how much L varies BETWEEN agents at the same pixel; this is
                # the only component normalization can preserve.
                spread = (loc_weight.max(dim=1).values - loc_weight.min(dim=1).values)
                spread_of_L.append(float(spread.mean()))

            # 3. decisive: alpha with vs without the L factor
            safe_L = torch.clamp(loc_weight, min=1e-12)
            reliability_without_L = reliability / safe_L
            alpha_without_L = alpha_from_reliability(reliability_without_L)
            delta = (alpha_actual - alpha_without_L).abs()
            alpha_abs_deltas.append(float(delta.max()))
            alpha_rel_deltas.append(float(delta.mean()))

            frames_used += 1

    rule.forward = original_forward

    if frames_used == 0:
        raise RuntimeError(
            "no frames produced a localization weight; check that the "
            "checkpoint contains the localization heads and that the rule is "
            "receiving evidence_localization_uncertainty"
        )

    def _avg(items, key):
        return statistics.mean(item[key] for item in items)

    print("\n================ localization signal diagnosis ================")
    print("frames analyzed: %d\n" % frames_used)

    print("--- raw localization uncertainty u^loc, per modality ---")
    for name in sorted(loc_unc_by_modality):
        items = loc_unc_by_modality[name]
        print("  %-4s mean=%.6f  std=%.6f  min=%.6f  max=%.6f"
              % (name, _avg(items, "mean"), _avg(items, "std"),
                 _avg(items, "min"), _avg(items, "max")))

    print("\n--- reliability factor L = exp(-u^loc), per modality ---")
    for name in sorted(loc_weight_by_modality):
        items = loc_weight_by_modality[name]
        print("  %-4s mean=%.6f  std=%.6f  min=%.6f  max=%.6f"
              % (name, _avg(items, "mean"), _avg(items, "std"),
                 _avg(items, "min"), _avg(items, "max")))

    if spread_of_L:
        print("\n--- cross-agent spread of L at the same pixel ---")
        print("  mean(max_agent L - min_agent L) = %.6f" % statistics.mean(spread_of_L))
        print("  (near 0 => L is the same for every agent => it cancels in")
        print("   the normalization and cannot change alpha)")

    print("\n--- DECISIVE: does L change alpha at all? ---")
    print("  max |alpha_with_L - alpha_without_L|  = %.8f" % max(alpha_abs_deltas))
    print("  mean|alpha_with_L - alpha_without_L|  = %.8f" % statistics.mean(alpha_rel_deltas))
    print("""
  Interpretation:
    ~0            -> L is normalized away. Inference-side tuning cannot help;
                     fix on the training side (raise the localization loss
                     weight, calibrate per modality) or inject the signal
                     outside the normalized product.
    clearly > 0   -> L does move alpha, so the flat AP means alpha itself has
                     little influence on the fused output; revisit how alpha
                     is used (lambda / tau / mu) rather than the signal.
""")
    print("PACT_CBEA_LOCALIZATION_DIAGNOSIS_DONE")


if __name__ == "__main__":
    main()

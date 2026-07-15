"""Read-only geometry audit for PACT-CBEA aligned BEV fusion.

The tool deliberately does not select a transform convention.  It records the
production convention and diagnoses three alternatives on the same real batch.
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from opencood.utils.transformation_utils import normalize_pairwise_tfm


CSV_FIELDS = [
    "scene_index", "agent_index", "modality_name", "record_len",
    "original_agent_index", "restored_feature_index", "pairwise_matrix_index",
    "order_consistent", "candidate", "raw_pairwise_ego_source_json", "normalized_affine_json",
    "raw_translation_x", "raw_translation_y",
    "raw_rotation_deg", "ego_identity_max_error", "feature_h", "feature_w",
    "feature_channels", "pixel_translation_x", "pixel_translation_y",
    "validity_ratio", "feature_nonzero_ratio", "feature_mean", "feature_std",
    "cosine_similarity_to_ego", "normalized_correlation_to_ego",
    "round_trip_mae", "inverse_consistency_error", "has_nan", "has_inf"
]


def _finite_scalar(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _matrix_record(matrix):
    return [[_finite_scalar(item) for item in row] for row in matrix.detach().cpu().tolist()]


def _matrix_angle_deg(matrix):
    return _finite_scalar(torch.atan2(matrix[1, 0], matrix[0, 0]) * 180.0 / math.pi)


def _safe_cosine(left, right):
    left = left.reshape(-1)
    right = right.reshape(-1)
    denom = torch.norm(left) * torch.norm(right)
    if float(denom) <= 1e-12:
        return None
    return _finite_scalar(torch.dot(left, right) / denom)


def _safe_correlation(left, right):
    left = left.reshape(-1)
    right = right.reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    return _safe_cosine(left, right)


def _to_homogeneous(affine):
    """Convert [N, 2, 3] affine matrices to [N, 3, 3]."""
    bottom = affine.new_zeros((affine.shape[0], 1, 3))
    bottom[:, 0, 2] = 1.0
    return torch.cat((affine, bottom), dim=1)


def _inverse_affine(affine):
    return torch.linalg.inv(_to_homogeneous(affine))[:, :2, :]


def _prepare_warp_affine(feature, affine):
    """Use the captured feature as the dtype/device authority for sampling."""
    if not torch.is_tensor(feature) or feature.ndim != 4:
        raise ValueError("feature must be a 4D Tensor")
    if not feature.is_floating_point():
        raise ValueError("feature must use a floating point dtype")
    if not torch.is_tensor(affine) or affine.shape != (feature.shape[0], 2, 3):
        raise ValueError("affine must have shape [N,2,3] matching feature")
    if not affine.is_floating_point():
        raise ValueError("affine must use a floating point dtype")
    # Pairwise matrices originate as NumPy float64. Convert only the sampling
    # affine at this boundary; never promote the captured model feature.
    affine = affine.to(device=feature.device, dtype=feature.dtype)
    if affine.device != feature.device or affine.dtype != feature.dtype:
        raise RuntimeError("warp affine dtype/device does not match feature")
    return affine


def warp_feature_and_validity(feature, affine, align_corners=False):
    """Exactly mirror v2's affine_grid/grid_sample settings for diagnostics."""
    affine = _prepare_warp_affine(feature, affine)
    validity = torch.ones((feature.shape[0], 1, feature.shape[2], feature.shape[3]),
                          device=feature.device, dtype=feature.dtype)
    packed = torch.cat((feature, validity), dim=1)
    grid = F.affine_grid(affine, packed.size(), align_corners=align_corners)
    if grid.device != packed.device or grid.dtype != packed.dtype:
        raise RuntimeError("affine_grid output dtype/device does not match packed feature")
    warped = F.grid_sample(packed, grid, mode="bilinear", padding_mode="zeros",
                           align_corners=align_corners)
    warped_validity = warped[:, -1:].clamp(0.0, 1.0)
    return warped[:, :-1] * warped_validity, warped_validity, grid


def build_candidate_affine(pairwise_scene, convention, physical_h, physical_w,
                           voxel_size, downsample_rate=1):
    """Normalize one explicitly named source/grid convention; never rank it."""
    if pairwise_scene.ndim != 4 or pairwise_scene.shape[-2:] != (4, 4):
        raise ValueError("pairwise_scene must be [N,N,4,4]")
    count = pairwise_scene.shape[0]
    matrix = torch.eye(4, dtype=pairwise_scene.dtype, device=pairwise_scene.device)
    pairwise = matrix.view(1, 1, 1, 4, 4).repeat(1, count, count, 1, 1)
    for agent_index in range(count):
        if convention == "A_ego_source":
            transform = pairwise_scene[0, agent_index]
        elif convention == "B_source_ego":
            transform = pairwise_scene[agent_index, 0]
        elif convention == "C_inverse_ego_source":
            transform = torch.linalg.inv(pairwise_scene[0, agent_index])
        elif convention == "D_identity":
            transform = matrix
        else:
            raise ValueError("Unknown convention: {}".format(convention))
        pairwise[0, 0, agent_index] = transform
    affine = normalize_pairwise_tfm(pairwise, physical_h, physical_w,
                                    voxel_size, downsample_rate)
    return affine[0, 0]


def _round_trip_error(feature, affine, align_corners):
    warped, validity, _ = warp_feature_and_validity(feature, affine, align_corners)
    restored, restored_validity, _ = warp_feature_and_validity(
        warped, _inverse_affine(affine), align_corners)
    support = (validity * restored_validity) > 0.5
    support = support.expand_as(feature)
    if not bool(support.any()):
        return None
    return _finite_scalar((restored[support] - feature[support]).abs().mean())


def diagnose_candidate(feature, pairwise_scene, convention, physical_h, physical_w,
                       voxel_size, align_corners):
    affine = build_candidate_affine(pairwise_scene, convention, physical_h,
                                    physical_w, voxel_size)
    warped_feature, validity, _ = warp_feature_and_validity(feature, affine, align_corners)
    inverse_error = []
    for agent_index in range(pairwise_scene.shape[0]):
        source_to_ego = pairwise_scene[agent_index, 0]
        ego_to_source = pairwise_scene[0, agent_index]
        identity = torch.eye(4, dtype=pairwise_scene.dtype, device=pairwise_scene.device)
        inverse_error.append(_finite_scalar((source_to_ego.matmul(ego_to_source) - identity).abs().max()))
    rows = []
    round_trip = _round_trip_error(feature, affine, align_corners)
    for agent_index in range(feature.shape[0]):
        item = warped_feature[agent_index]
        rows.append({
            "candidate": convention,
            "normalized_affine": _matrix_record(affine[agent_index]),
            "validity_ratio": _finite_scalar(validity[agent_index].mean()),
            "feature_nonzero_ratio": _finite_scalar((item.abs() > 1e-8).float().mean()),
            "feature_mean": _finite_scalar(item.mean()),
            "feature_std": _finite_scalar(item.std(unbiased=False)),
            "cosine_similarity_to_ego": _safe_cosine(item, warped_feature[0]),
            "normalized_correlation_to_ego": _safe_correlation(item, warped_feature[0]),
            "round_trip_mae": round_trip,
            "inverse_consistency_error": inverse_error[agent_index],
        })
    return rows


def synthetic_agent_order_check():
    """Mirror the modal batching/counter restoration used by the PACT models."""
    modalities = ["m2", "m1", "m2", "m3"]
    original = [10.0, 20.0, 30.0, 40.0]
    grouped = {}
    for index, modality in enumerate(modalities):
        grouped.setdefault(modality, []).append(torch.full((1, 1, 1), original[index]))
    grouped = dict((name, torch.stack(items, dim=0)) for name, items in grouped.items())
    counters = dict((name, 0) for name in grouped)
    restored = []
    for modality in modalities:
        restored.append(grouped[modality][counters[modality]])
        counters[modality] += 1
    values = [float(item.reshape(-1)[0]) for item in restored]
    return {
        "modalities": modalities,
        "input_values": original,
        "restored_values": values,
        "restored_matches_input": values == original,
    }


def _alignment_result_summary(result):
    if isinstance(result, dict):
        return "dict keys={}".format(sorted(result.keys()))
    if isinstance(result, (tuple, list)):
        return "{} length={}".format(type(result).__name__, len(result))
    if torch.is_tensor(result):
        return "Tensor shape={}".format(tuple(result.shape))
    return type(result).__name__


def extract_aligned_geometry_fields(result):
    """Extract the documented PACT geometry fields without changing result."""
    # PACTCBEAEvidenceGeometryAligner.forward returns this dict.  Keep this
    # contract strict so a future aligner API change cannot silently corrupt an
    # audit report. Tuple/list/tensor are intentionally rejected: none is a
    # valid return type for this production aligner.
    if not isinstance(result, dict):
        raise RuntimeError(
            "PACT geometry aligner returned {}; expected dict with feature and validity".format(
                _alignment_result_summary(result)
            )
        )
    required = ("feature", "validity")
    missing = [name for name in required if name not in result]
    if missing:
        raise RuntimeError(
            "PACT geometry aligner result missing {} (available keys: {})".format(
                ", ".join(missing), sorted(result.keys())
            )
        )
    fields = {}
    for name in required:
        value = result[name]
        if not torch.is_tensor(value):
            raise RuntimeError(
                "PACT geometry aligner field {} must be a Tensor; got {} (available keys: {})".format(
                    name, type(value).__name__, sorted(result.keys())
                )
            )
        fields[name] = value
    return fields


def _capture_model_geometry(model):
    """Install process-local wrappers; no production module is modified."""
    capture = {}
    original_single = model.pyramid_backbone.forward_single
    original_geometry = model.pact_geometry_aligner.forward

    def forward_single_wrapper(features):
        result = original_single(features)
        capture["forward_single_input"] = features.detach()
        capture["forward_single_output"] = result[0].detach()
        return result

    def geometry_wrapper(feature, logits, uncertainty, descriptor, record_len, affine_matrix):
        capture["single_feature"] = feature.detach()
        capture["record_len"] = record_len.detach()
        capture["affine"] = affine_matrix.detach()
        result = original_geometry(feature, logits, uncertainty, descriptor, record_len, affine_matrix)
        fields = extract_aligned_geometry_fields(result)
        capture["aligned_feature"] = fields["feature"].detach()
        capture["aligned_validity"] = fields["validity"].detach()
        return result

    model.pyramid_backbone.forward_single = forward_single_wrapper
    model.pact_geometry_aligner.forward = geometry_wrapper
    return capture, original_single, original_geometry


def _restore_model_geometry(model, originals):
    model.pyramid_backbone.forward_single = originals[0]
    model.pact_geometry_aligner.forward = originals[1]


def _batch_to_device(batch, device):
    from opencood.tools import train_utils
    return train_utils.to_device(batch, device)


def _scene_modalities(agent_modalities, start, count):
    return list(agent_modalities[start:start + count])


def _write_outputs(output_dir, report, csv_rows):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    json_path = os.path.join(output_dir, "geometry_audit.json")
    csv_path = os.path.join(output_dir, "geometry_audit.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    return json_path, csv_path


def _gt_diagnostic(batch):
    ego = batch.get("ego", {})
    centers = ego.get("object_bbx_center")
    if centers is None:
        return {"available": False, "computed": False, "reason": "object_bbx_center is absent"}
    return {
        "available": True,
        "computed": False,
        "reason": "GT centers are available, but no verified feature-grid origin/axis/stride contract is "
                  "exposed by the production model. The audit refuses to invent a GT-to-feature mapping.",
        "box_tensor_shape": list(centers.shape),
    }


def resolve_checkpoint_path(model_dir, checkpoint=None):
    """Resolve an explicit checkpoint once, relative to model_dir when needed."""
    model_dir_abs = os.path.abspath(model_dir)
    if checkpoint is None:
        return os.path.join(model_dir_abs, "net_epoch1.pth")
    if os.path.isabs(checkpoint):
        return os.path.abspath(checkpoint)

    checkpoint_from_cwd = os.path.abspath(checkpoint)
    try:
        already_under_model_dir = os.path.commonpath(
            [model_dir_abs, checkpoint_from_cwd]
        ) == model_dir_abs
    except ValueError:
        already_under_model_dir = False
    if already_under_model_dir:
        return checkpoint_from_cwd
    return os.path.abspath(os.path.join(model_dir_abs, checkpoint))


def audit_real_geometry(args):
    # Dataset/model imports are intentionally lazy so the CPU-only geometry smoke
    # can exercise the math without requiring every optional dataset dependency.
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    config_path = os.path.join(args.model_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise RuntimeError("Expected saved config at {}".format(config_path))
    hypes = yaml_utils.load_yaml(config_path)
    if args.split == "test":
        hypes["validate_dir"] = hypes["test_dir"]
    if args.in_order:
        fusion = hypes["fusion"]
        fusion["core_method"] = fusion["core_method"] + "infer"
        hypes["comm_range"] = 180
        hypes["heter"]["assignment_path"] = hypes["heter"]["assignment_path"].replace(".json", "_in_order.json")
        hypes["heter"]["ego_modality"] = "m1"

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = train_utils.create_model(hypes)
    checkpoint_path = resolve_checkpoint_path(args.model_dir, args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(
            "Expected net_epoch1.pth or an explicit --checkpoint; not found: {}".format(checkpoint_path)
        )
    # The aligned-uniform model performs its own strict core-key verification here.
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    model.to(device)
    model.eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers,
                        collate_fn=dataset.collate_batch_test, shuffle=False)
    capture, original_single, original_geometry = _capture_model_geometry(model)
    report = {
        "tool": "audit_pact_cbea_real_geometry",
        "model_dir": os.path.abspath(args.model_dir),
        "checkpoint": os.path.abspath(checkpoint_path),
        "device": str(device),
        "num_scenes_requested": args.num_scenes,
        "proj_first": bool(hypes["fusion"]["args"].get("proj_first", False)),
        "production_convention": "pairwise_t_matrix[b, 0, source]",
        "grid_sample": {"mode": "bilinear", "padding_mode": "zeros",
                        "align_corners": bool(getattr(model.pyramid_backbone, "align_corners", False))},
        "normalization": {"physical_h": model.H, "physical_w": model.W,
                          "fake_voxel_size": model.fake_voxel_size, "downsample_rate": 1},
        "code_chain_audit": {
            "pairwise_definition": "get_pairwise_transformation: pairwise[i,j]=T_j_i maps coordinates i to j",
            "grid_sample_requirement": "output ego coordinates sample source coordinates; production uses [ego,source]",
            "proj_first_semantics": "false keeps lidar inputs in each local frame; true projects lidar to ego and returns identity pairwise",
            "forward_single_pairwise_warp": False,
            "pre_geometry_warp_conclusion": "no prior inter-agent BEV warp in the audited intermediateheter path when proj_first=false",
            "camera_note": "camera Lift-Splat uses each vehicle's camera-to-local-lidar extrinsics, not pairwise inter-agent transforms",
        },
        "synthetic_agent_order": synthetic_agent_order_check(),
        "scenes": [],
    }
    csv_rows = []
    offset = 0
    try:
        with torch.no_grad():
            for scene_index, batch in enumerate(loader):
                if scene_index >= args.num_scenes:
                    break
                batch = _batch_to_device(batch, device)
                capture.clear()
                model(batch["ego"])
                if not all(key in capture for key in ("single_feature", "aligned_feature", "aligned_validity", "affine")):
                    raise RuntimeError("Model geometry hooks did not observe the expected aligned-uniform path")
                record_len = capture["record_len"].detach().cpu().tolist()
                feature = capture["single_feature"]
                raw_pairwise = batch["ego"]["pairwise_t_matrix"]
                agent_modalities = list(batch["ego"]["agent_modality_list"])
                cav_ids = list(batch["ego"].get("cav_id_list", []))
                scene_report = {"scene_index": scene_index, "record_len": record_len,
                                "gt_diagnostic": _gt_diagnostic(batch), "agents": []}
                local_start = 0
                for batch_index, count in enumerate(record_len):
                    count = int(count)
                    pairwise_scene = raw_pairwise[batch_index, :count, :count]
                    local_feature = feature[local_start:local_start + count]
                    modalities = _scene_modalities(agent_modalities, local_start, count)
                    local_cav_ids = cav_ids[local_start:local_start + count] if cav_ids else []
                    candidate_rows = {}
                    for convention in ("A_ego_source", "B_source_ego", "C_inverse_ego_source", "D_identity"):
                        candidate_rows[convention] = diagnose_candidate(
                            local_feature, pairwise_scene, convention, model.H, model.W,
                            model.fake_voxel_size, bool(getattr(model.pyramid_backbone, "align_corners", False)))
                    for agent_index in range(count):
                        raw = pairwise_scene[0, agent_index]
                        actual_validity = capture["aligned_validity"][local_start + agent_index]
                        actual_feature = capture["aligned_feature"][local_start + agent_index]
                        identity_error = _finite_scalar((pairwise_scene[0, 0] - torch.eye(4, device=device,
                            dtype=pairwise_scene.dtype)).abs().max())
                        agent = {
                            "agent_index": agent_index,
                            "modality_name": modalities[agent_index],
                            "cav_id": local_cav_ids[agent_index] if agent_index < len(local_cav_ids) else None,
                            "agent_order_evidence": "collate_batch_test cav_id_list / agent_modality_list index",
                            "original_agent_index": agent_index,
                            "restored_feature_index": local_start + agent_index,
                            "pairwise_matrix_index": agent_index,
                            "order_consistent": True,
                            "raw_pairwise_ego_source": _matrix_record(raw),
                            "ego_identity_max_error": identity_error,
                            "translation_x": _finite_scalar(raw[0, 3]),
                            "translation_y": _finite_scalar(raw[1, 3]),
                            "rotation_deg": _matrix_angle_deg(raw),
                            "single_feature_shape": list(local_feature[agent_index].shape),
                            "normalized_affine_production": _matrix_record(capture["affine"][batch_index, 0, agent_index]),
                            "pixel_translation_x": _finite_scalar(raw[0, 3] * local_feature.shape[-1] / float(model.W)),
                            "pixel_translation_y": _finite_scalar(raw[1, 3] * local_feature.shape[-2] / float(model.H)),
                            "actual_validity_ratio": _finite_scalar(actual_validity.mean()),
                            "actual_feature_nonzero_ratio": _finite_scalar((actual_feature.abs() > 1e-8).float().mean()),
                            "actual_feature_mean": _finite_scalar(actual_feature.mean()),
                            "actual_feature_std": _finite_scalar(actual_feature.std(unbiased=False)),
                            "has_nan": bool(torch.isnan(actual_feature).any()),
                            "has_inf": bool(torch.isinf(actual_feature).any()),
                            "candidates": candidate_rows,
                        }
                        scene_report["agents"].append(agent)
                        for convention, values in candidate_rows.items():
                            diagnostic = values[agent_index]
                            csv_rows.append({
                                "scene_index": scene_index, "agent_index": agent_index,
                                "modality_name": modalities[agent_index], "record_len": count,
                                "original_agent_index": agent_index,
                                "restored_feature_index": local_start + agent_index,
                                "pairwise_matrix_index": agent_index, "order_consistent": True,
                                "candidate": convention, "raw_translation_x": _finite_scalar(raw[0, 3]),
                                "raw_translation_y": _finite_scalar(raw[1, 3]),
                                "raw_pairwise_ego_source_json": json.dumps(_matrix_record(raw)),
                                "normalized_affine_json": json.dumps(diagnostic["normalized_affine"]),
                                "raw_rotation_deg": _matrix_angle_deg(raw),
                                "ego_identity_max_error": identity_error,
                                "feature_h": local_feature.shape[-2], "feature_w": local_feature.shape[-1],
                                "feature_channels": local_feature.shape[1],
                                "pixel_translation_x": _finite_scalar(raw[0, 3] * local_feature.shape[-1] / float(model.W)),
                                "pixel_translation_y": _finite_scalar(raw[1, 3] * local_feature.shape[-2] / float(model.H)),
                                "has_nan": bool(torch.isnan(actual_feature).any()),
                                "has_inf": bool(torch.isinf(actual_feature).any()),
                                **diagnostic
                            })
                    local_start += count
                report["scenes"].append(scene_report)
    finally:
        _restore_model_geometry(model, (original_single, original_geometry))
    report["num_scenes_completed"] = len(report["scenes"])
    output_dir = args.output_dir or os.path.join(args.model_dir, "geometry_audit")
    json_path, csv_path = _write_outputs(output_dir, report, csv_rows)
    print("geometry_audit.json: {}".format(json_path))
    print("geometry_audit.csv: {}".format(csv_path))
    print("REAL_GEOMETRY_AUDIT_COMPLETE scenes={}".format(report["num_scenes_completed"]))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="Read-only PACT-CBEA real geometry audit")
    parser.add_argument("--model_dir", required=True, help="Directory containing config.yaml and net_epoch*.pth")
    parser.add_argument("--num_scenes", type=int, default=8)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Absolute path or path relative to --model_dir; default net_epoch1.pth")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Default: cuda when available, else cpu")
    parser.add_argument("--split", choices=("validate", "test"), default="test")
    parser.add_argument("--in_order", action="store_true", help="Mirror inference_heter_in_order.py setup")
    return parser


if __name__ == "__main__":
    audit_real_geometry(build_parser().parse_args())

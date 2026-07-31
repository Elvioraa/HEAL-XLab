"""One-batch real-data preflight for PACT-CBEA object-level Stage 3."""

import argparse
from collections import Counter
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.loss.pact_cbea_object_stage3_loss import (
    PactCbeaObjectStage3Loss,
)
from opencood.models.sub_modules.pact_cbea_object_refiner import (
    sampler_lwh_to_repository_hwl,
)
from opencood.models.sub_modules.pact_cbea_object_stage3_utils import (
    load_base_checkpoint_compatible,
    strict_load_stage3_checkpoint,
)
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-data preflight for PACT-CBEA object Stage 3"
    )
    parser.add_argument("--hypes_yaml", "-y", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--stage3-checkpoint", default=None)
    parser.add_argument("--dataset-split", choices=("train", "validate", "test"), default="validate")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--allow-scratch-stage3", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_count <= 0:
        raise ValueError("batch-count must be positive")
    if not os.path.isfile(args.hypes_yaml):
        return _skip("YAML unavailable: %s" % args.hypes_yaml)
    hypes = yaml_utils.load_yaml(args.hypes_yaml)
    object_cfg = hypes["model"]["args"]["pact_cbea"].get(
        "object_level_stage3", {}
    )
    base_checkpoint = args.base_checkpoint or object_cfg.get("base_checkpoint")
    stage3_checkpoint = args.stage3_checkpoint or object_cfg.get("stage3_checkpoint")
    if not base_checkpoint or not os.path.isfile(base_checkpoint):
        return _skip("required base checkpoint unavailable: %s" % base_checkpoint)
    if not stage3_checkpoint and not args.allow_scratch_stage3:
        return _skip("required Stage 3 checkpoint unavailable")
    if stage3_checkpoint and not os.path.isfile(stage3_checkpoint):
        return _skip(
            "required Stage 3 checkpoint unavailable: %s" % stage3_checkpoint
        )

    dataset_path = _dataset_path(hypes, args.dataset_split)
    if not dataset_path or not os.path.exists(dataset_path):
        return _skip("required local dataset unavailable: %s" % dataset_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        return _skip("requested CUDA device unavailable")
    device = torch.device(args.device)
    from opencood.data_utils.datasets import build_dataset

    model = train_utils.create_model(hypes)
    base_report = load_base_checkpoint_compatible(
        model, base_checkpoint, require_complete=True
    )
    print(
        "base checkpoint: loaded=%d missing=%d unexpected=%d"
        % (base_report["loaded"], len(base_report["missing"]),
           len(base_report["unexpected"]))
    )
    if stage3_checkpoint:
        strict_load_stage3_checkpoint(model, stage3_checkpoint)
        print("stage3 checkpoint: strict load PASS")
    else:
        print("stage3 checkpoint: scratch parameters explicitly allowed")
    model.to(device)
    model.train(args.backward)

    train_split = args.dataset_split == "train"
    if args.dataset_split == "test":
        hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=train_split)
    collate = (
        dataset.collate_batch_train if train_split
        else dataset.collate_batch_test
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate,
        shuffle=False,
        drop_last=False,
    )
    criterion = PactCbeaObjectStage3Loss(hypes["loss"].get("args", {}))
    processed = 0
    for batch_idx, batch_data in enumerate(loader):
        if processed >= args.batch_count:
            break
        if batch_data is None:
            continue
        batch_data = train_utils.to_device(batch_data, device)
        batch_data["ego"]["object_stage3_compute_targets"] = True
        if args.backward:
            model.zero_grad(set_to_none=True)
            output = model(batch_data["ego"])
            loss = criterion(output, None)
            loss.backward()
            print("backward loss: %.6f" % float(loss.detach().cpu().item()))
        else:
            with torch.no_grad():
                output = model(batch_data["ego"])
        _print_batch(batch_idx, batch_data["ego"], output["object_stage3"])
        processed += 1
    if processed == 0:
        return _skip("dataset yielded no usable batch")
    print("REALDATA_PREFLIGHT: PASS (%d batch)" % processed)
    return 0


def _print_batch(batch_idx, data, stage3):
    modalities = list(data["agent_modality_list"])
    counts = Counter(modalities)
    print("batch %d" % batch_idx)
    print(" - record_len: %s" % (stage3["record_len"],))
    print(" - agent modality list (diagnostic only): %s" % modalities)
    print(
        " - m1/m2/m3/m4 counts: %s"
        % {name: counts.get(name, 0) for name in ("m1", "m2", "m3", "m4")}
    )
    print(" - single_feature shape: %s" % (stage3["single_feature_shape"],))
    print(" - single_feature frame: %s" % stage3["single_feature_frame"])
    print(" - pairwise transform shape: %s" % (tuple(data["pairwise_t_matrix"].shape),))
    print(" - transform direction: %s; sampler uses [batch,0,agent]" % stage3["pairwise_direction"])
    for scene_idx, scene in enumerate(stage3["scenes"]):
        scores = scene["proposal_scores"]
        score_range = (
            (float(scores.min().item()), float(scores.max().item()))
            if scores.numel() else None
        )
        repository_boxes = sampler_lwh_to_repository_hwl(scene["proposals"])
        conversion = (
            {
                "sampler_xyzlwhr": scene["proposals"][0].detach().cpu().tolist(),
                "repository_xyzhwlr": repository_boxes[0].detach().cpu().tolist(),
            }
            if scene["proposals"].shape[0] else None
        )
        coverage = scene["coverage"]
        matched = scene["matched_ious"][scene["positive_mask"]]
        log_variance = scene["agent_log_variances"]
        print(" - scene %d proposal count: %d" % (scene_idx, scene["proposal_count"]))
        print(" - scene %d proposal score range: %s" % (scene_idx, score_range))
        print(" - scene %d box order conversion: %s" % (scene_idx, conversion))
        print(" - scene %d ROI feature shape: %s" % (scene_idx, scene["roi_feature_shape"]))
        print(" - scene %d coverage min/mean/max: %s" % (scene_idx, _stats(coverage)))
        print(" - scene %d valid ratio: %.6f" % (
            scene_idx,
            float(scene["valid_mask"].float().mean().item())
            if scene["valid_mask"].numel() else 0.0,
        ))
        print(" - scene %d positive proposals: %d" % (scene_idx, scene["positive_count"]))
        print(" - scene %d matched IoU min/mean/max: %s" % (scene_idx, _stats(matched)))
        print(" - scene %d fallback ratio: %.6f" % (
            scene_idx,
            float(scene["fallback_mask"].float().mean().item())
            if scene["fallback_mask"].numel() else 0.0,
        ))
        print(" - scene %d residual shape: %s" % (scene_idx, tuple(scene["agent_residuals"].shape)))
        print(" - scene %d log-variance shape/stats: %s / %s" % (
            scene_idx, tuple(log_variance.shape), _stats(log_variance)
        ))
        print(" - scene %d refined/final shapes: %s / %s" % (
            scene_idx, tuple(scene["refined_boxes"].shape),
            tuple(scene["final_boxes"].shape)
        ))
    if torch.cuda.is_available() and next(iter(stage3["scenes"]), None) is not None:
        print(" - GPU allocated MB: %.2f" % (
            torch.cuda.memory_allocated() / (1024.0 ** 2)
        ))


def _stats(tensor):
    if tensor.numel() == 0:
        return None
    tensor = tensor.detach().float()
    return (
        float(tensor.min().cpu().item()),
        float(tensor.mean().cpu().item()),
        float(tensor.max().cpu().item()),
    )


def _dataset_path(hypes, split):
    if split == "train":
        return hypes.get("root_dir")
    if split == "test":
        return hypes.get("test_dir")
    return hypes.get("validate_dir")


def _skip(reason):
    print("SKIPPED: required local data/checkpoint unavailable")
    print("reason: %s" % reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

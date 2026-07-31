"""Inference and AP evaluation for PACT-CBEA object-level Stage 3."""

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.models.sub_modules.pact_cbea_object_refiner import (
    sampler_lwh_to_repository_hwl,
    wrap_to_pi,
)
from opencood.models.sub_modules.pact_cbea_object_stage3_utils import (
    load_base_checkpoint_compatible,
    strict_load_stage3_checkpoint,
)
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PACT-CBEA object-level Stage 3"
    )
    parser.add_argument("--hypes_yaml", "-y", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--stage3-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-split", choices=("validate", "test"), default="test")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--disable-stage3", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils import eval_utils

    hypes = yaml_utils.load_yaml(args.hypes_yaml)
    object_cfg = _object_cfg(hypes)
    if not object_cfg.get("enabled", False):
        raise RuntimeError("inference YAML must construct object Stage 3")
    base_checkpoint = args.base_checkpoint or object_cfg.get("base_checkpoint")
    stage3_checkpoint = args.stage3_checkpoint or object_cfg.get("stage3_checkpoint")
    if not base_checkpoint:
        raise ValueError("a base checkpoint must be supplied")
    if not args.disable_stage3 and not stage3_checkpoint:
        raise ValueError("enabled Stage 3 inference requires a checkpoint")

    if args.dataset_split == "test":
        hypes["validate_dir"] = hypes["test_dir"]
    device = _resolve_device(args.device)
    model = train_utils.create_model(hypes)
    base_report = load_base_checkpoint_compatible(
        model, base_checkpoint, require_complete=True
    )
    print(
        "Base checkpoint loaded=%d missing=%d unexpected=%d"
        % (base_report["loaded"], len(base_report["missing"]),
           len(base_report["unexpected"]))
    )
    if not args.disable_stage3:
        strict_report = strict_load_stage3_checkpoint(model, stage3_checkpoint)
        print(
            "Strict Stage 3 checkpoint loaded: version=%d epoch=%d"
            % (strict_report["version"], strict_report["epoch"])
        )
    model.set_object_stage3_runtime_enabled(not args.disable_stage3)
    model.to(device)
    model.eval()

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    baseline_stat = _empty_result_stat()
    refined_stat = _empty_result_stat()
    diagnostics = _Diagnostics()

    for batch_idx, batch_data in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        if batch_data is None:
            continue
        batch_data = train_utils.to_device(batch_data, device)
        with torch.no_grad():
            output = model(batch_data["ego"])
            baseline_boxes, baseline_scores, gt_boxes = dataset.post_process(
                batch_data, {"ego": output}
            )
        _accumulate_ap(
            baseline_boxes, baseline_scores, gt_boxes, baseline_stat, eval_utils
        )

        if args.disable_stage3:
            refined_boxes, refined_scores = baseline_boxes, baseline_scores
        else:
            scenes = output["object_stage3"]["scenes"]
            if len(scenes) != 1:
                raise RuntimeError("inference requires batch_size=1")
            refined_boxes, refined_scores = _scene_final_corners(scenes[0])
            diagnostics.update(scenes[0])
        _accumulate_ap(
            refined_boxes, refined_scores, gt_boxes, refined_stat, eval_utils
        )

    os.makedirs(args.output_dir, exist_ok=True)
    print("Baseline collaborative detector:")
    baseline_ap = eval_utils.eval_final_results(
        baseline_stat, args.output_dir, "object_stage3_baseline"
    )
    print("Object Stage 3 %s:" % ("OFF" if args.disable_stage3 else "ON"))
    refined_ap = eval_utils.eval_final_results(
        refined_stat,
        args.output_dir,
        "object_stage3_off" if args.disable_stage3 else "object_stage3_on",
    )
    diagnostics.print_summary()
    print(
        "AP summary baseline=(%.4f, %.4f, %.4f) refined=(%.4f, %.4f, %.4f)"
        % tuple(baseline_ap + refined_ap)
    )


def _scene_final_corners(scene):
    boxes = scene["final_boxes"]
    scores = scene["final_scores"]
    if boxes.shape[0] == 0:
        return None, None
    from opencood.utils import box_utils

    repository_boxes = sampler_lwh_to_repository_hwl(boxes)
    corners = box_utils.boxes_to_corners_3d(repository_boxes, order="hwl")
    return corners, scores


def _accumulate_ap(boxes, scores, gt_boxes, result_stat, eval_utils):
    for threshold in (0.3, 0.5, 0.7):
        eval_utils.caluclate_tp_fp(
            boxes, scores, gt_boxes, result_stat, threshold
        )


def _empty_result_stat():
    return {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in (0.3, 0.5, 0.7)
    }


class _Diagnostics(object):
    def __init__(self):
        self.frames = 0
        self.proposals = 0
        self.refined = 0
        self.fallbacks = 0
        self.center_shift = []
        self.size_shift = []
        self.yaw_shift = []
        self.coverage = []
        self.variance = []

    def update(self, scene):
        self.frames += 1
        proposals = scene["proposals"]
        refined = scene["refined_boxes"]
        self.proposals += int(proposals.shape[0])
        self.refined += int(scene["final_boxes"].shape[0])
        self.fallbacks += int(scene["fallback_mask"].sum().item())
        if proposals.shape[0]:
            self.center_shift.append(
                torch.norm(refined[:, 0:3] - proposals[:, 0:3], dim=-1)
            )
            self.size_shift.append(
                (refined[:, 3:6] - proposals[:, 3:6]).abs().mean(dim=-1)
            )
            self.yaw_shift.append(
                wrap_to_pi(refined[:, 6] - proposals[:, 6]).abs()
            )
        if scene["coverage"].numel():
            self.coverage.append(scene["coverage"].reshape(-1))
        if scene["agent_log_variances"].numel():
            self.variance.append(
                torch.exp(scene["agent_log_variances"]).reshape(-1)
            )

    def print_summary(self):
        print("Object Stage 3 diagnostics:")
        print(" - frames: %d" % self.frames)
        print(" - proposals: %d" % self.proposals)
        print(" - refined after NMS: %d" % self.refined)
        print(" - fallback proposals: %d" % self.fallbacks)
        print(" - mean center shift: %.6f" % _mean(self.center_shift))
        print(" - mean size shift: %.6f" % _mean(self.size_shift))
        print(" - mean yaw shift: %.6f" % _mean(self.yaw_shift))
        print(" - mean coverage: %.6f" % _mean(self.coverage))
        print(" - mean predicted variance: %.6f" % _mean(self.variance))


def _mean(values):
    if not values:
        return 0.0
    return float(torch.cat(values).float().mean().detach().cpu().item())


def _object_cfg(hypes):
    return hypes["model"]["args"]["pact_cbea"].get(
        "object_level_stage3", {}
    )


def _resolve_device(requested):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


if __name__ == "__main__":
    main()

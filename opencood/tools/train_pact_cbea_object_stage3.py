"""Standalone trainer for frozen-base PACT-CBEA object-level Stage 3."""

import argparse
import copy
from datetime import datetime
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.loss.pact_cbea_object_stage3_loss import (
    PactCbeaObjectStage3Loss,
)
from opencood.models.sub_modules.pact_cbea_object_stage3_utils import (
    build_stage3_checkpoint,
    load_base_checkpoint_compatible,
    stage3_named_parameters,
    strict_load_stage3_checkpoint,
)
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PACT-CBEA object-level Stage 3 only"
    )
    parser.add_argument("--hypes_yaml", "-y", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--resume-stage3", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from opencood.data_utils.datasets import build_dataset

    distributed, rank, world_size, local_rank = _init_distributed(args)
    device = _resolve_device(args.device, distributed, local_rank)
    hypes = yaml_utils.load_yaml(args.hypes_yaml)
    object_cfg = _object_cfg(hypes)
    if not object_cfg.get("enabled", False):
        raise RuntimeError("object_level_stage3.enabled must be true for training")

    base_checkpoint = args.base_checkpoint or object_cfg.get("base_checkpoint")
    resume_checkpoint = args.resume_stage3 or object_cfg.get("stage3_checkpoint")
    if not base_checkpoint:
        raise ValueError("a base checkpoint must be supplied")
    if not resume_checkpoint and not bool(object_cfg.get("start_from_scratch", False)):
        raise RuntimeError(
            "new Stage 3 training requires explicit start_from_scratch: true"
        )

    model = train_utils.create_model(hypes)
    base_report = load_base_checkpoint_compatible(
        model, base_checkpoint, require_complete=True
    )
    model.to(device)
    model.train(True)
    summary = model.object_stage3_parameter_summary()
    if rank == 0:
        _print_parameter_summary(summary)
        _print_base_report(base_report)

    named_stage3 = list(stage3_named_parameters(model))
    optimizer = _build_optimizer(hypes, named_stage3)
    _assert_optimizer_exact(optimizer, named_stage3)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=0)
    criterion = PactCbeaObjectStage3Loss(hypes["loss"].get("args", {})).to(device)

    start_epoch = 0
    global_step = 0
    if resume_checkpoint:
        resume = strict_load_stage3_checkpoint(
            model, resume_checkpoint, optimizer=optimizer, scheduler=scheduler
        )
        _validate_resume_base(resume.get("base_checkpoint"), base_checkpoint)
        start_epoch = resume["epoch"]
        global_step = resume["global_step"]
        if rank == 0:
            print("Resumed strict Stage 3 checkpoint: %s" % resume_checkpoint)

    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset = build_dataset(hypes, visualize=False, train=False)
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if distributed else None
    )
    batch_size = int(hypes["train_params"]["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_batch_train,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_dataset.collate_batch_train,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    output_dir = args.output_dir or _default_output_dir(hypes["name"])
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        yaml_utils.save_yaml(hypes, os.path.join(output_dir, "config.yaml"))
        print("Stage 3 output directory: %s" % output_dir)
    _barrier(distributed)

    amp_enabled = bool(hypes["train_params"].get("amp", False))
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and device.type == "cuda"
    )
    epochs = int(hypes["train_params"]["epoches"])
    save_freq = int(hypes["train_params"].get("save_freq", 1))
    eval_freq = int(hypes["train_params"].get("eval_freq", 1))
    best_val = float("inf")

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            batch_data["ego"]["object_stage3_compute_targets"] = True
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                    enabled=amp_enabled and device.type == "cuda"):
                output = model(batch_data["ego"])
                loss = criterion(output, None)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if rank == 0:
                criterion.logging(epoch, batch_idx, len(train_loader))

        scheduler.step()
        completed_epoch = epoch + 1
        if completed_epoch % eval_freq == 0:
            val_loss = _validate(
                model,
                criterion,
                val_loader,
                device,
                args.max_val_batches,
                distributed,
                amp_enabled,
            )
            if rank == 0:
                print("Validation epoch %d: %.6f" % (completed_epoch, val_loss))
                if val_loss < best_val:
                    best_val = val_loss
                    _save_checkpoint(
                        os.path.join(output_dir, "stage3_best.pth"),
                        model_without_ddp,
                        optimizer,
                        scheduler,
                        completed_epoch,
                        global_step,
                        hypes,
                        base_checkpoint,
                    )
        if rank == 0 and completed_epoch % save_freq == 0:
            _save_checkpoint(
                os.path.join(
                    output_dir, "stage3_epoch%03d.pth" % completed_epoch
                ),
                model_without_ddp,
                optimizer,
                scheduler,
                completed_epoch,
                global_step,
                hypes,
                base_checkpoint,
            )
        if hasattr(train_dataset, "reinitialize"):
            train_dataset.reinitialize()
        _barrier(distributed)

    if rank == 0:
        print("Object Stage 3 training complete")
    if distributed:
        dist.destroy_process_group()


@torch.no_grad()
def _validate(
        model, criterion, loader, device, max_batches, distributed,
        amp_enabled):
    model.eval()
    total = torch.zeros(2, device=device, dtype=torch.float64)
    for batch_idx, batch_data in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if batch_data is None:
            continue
        batch_data = train_utils.to_device(batch_data, device)
        batch_data["ego"]["object_stage3_compute_targets"] = True
        with torch.cuda.amp.autocast(
                enabled=amp_enabled and device.type == "cuda"):
            output = model(batch_data["ego"])
            loss = criterion(output, None)
        total[0] += loss.detach().double()
        total[1] += 1.0
    if distributed:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float((total[0] / total[1].clamp_min(1.0)).item())


def _build_optimizer(hypes, named_stage3):
    optimizer_cfg = hypes["optimizer"]
    optimizer_class = getattr(torch.optim, optimizer_cfg["core_method"], None)
    if optimizer_class is None:
        raise ValueError("unsupported optimizer %s" % optimizer_cfg["core_method"])
    parameters = [parameter for _, parameter in named_stage3]
    return optimizer_class(
        parameters,
        lr=float(optimizer_cfg["lr"]),
        **optimizer_cfg.get("args", {})
    )


def _assert_optimizer_exact(optimizer, named_stage3):
    expected = {id(parameter) for _, parameter in named_stage3}
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if actual != expected:
        raise RuntimeError("optimizer parameters are not exactly object Stage 3")


def _save_checkpoint(
        path, model, optimizer, scheduler, epoch, global_step, hypes,
        base_checkpoint):
    payload = build_stage3_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        global_step,
        copy.deepcopy(hypes),
        base_checkpoint,
    )
    torch.save(payload, path)
    print("Saved Stage-3-only checkpoint: %s" % path)


def _object_cfg(hypes):
    pact_cfg = hypes.get("model", {}).get("args", {}).get("pact_cbea", {})
    cfg = pact_cfg.get("object_level_stage3", {})
    if not isinstance(cfg, dict):
        raise TypeError("object_level_stage3 config must be a mapping")
    return cfg


def _init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return distributed, rank, world_size, local_rank


def _resolve_device(requested, distributed, local_rank):
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = local_rank if distributed else (
            int(requested.split(":", 1)[1]) if ":" in requested else 0
        )
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return torch.device(requested)


def _validate_resume_base(recorded, current):
    if recorded and os.path.abspath(str(recorded)) != os.path.abspath(str(current)):
        raise RuntimeError(
            "resume checkpoint base identifier differs from configured base"
        )


def _print_parameter_summary(summary):
    print("Object Stage 3 trainable parameters:")
    for name in summary["trainable_names"]:
        print(" - %s" % name)
    print("Trainable parameter count: %d" % summary["trainable_count"])
    print("Frozen parameter count: %d" % summary["frozen_count"])


def _print_base_report(report):
    print(
        "Base checkpoint loaded=%d missing=%d unexpected=%d"
        % (report["loaded"], len(report["missing"]), len(report["unexpected"]))
    )


def _default_output_dir(name):
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    safe_name = str(name).replace("/", "_").replace("\\", "_")
    return os.path.join("opencood", "logs", "%s_%s" % (safe_name, stamp))


def _barrier(distributed):
    if distributed:
        dist.barrier()


if __name__ == "__main__":
    main()

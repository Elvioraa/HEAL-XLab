"""Profile Open-DCSI parameters, packets, memory, MACs, and latency."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from opencood.tools.audit_open_dcsi_baseline_parity import (
    _collab_input,
    _tiny_model_args,
)
from opencood.tools.check_open_dcsi_phase6 import _communication_args
from opencood.tools.check_open_dcsi_phase7 import _streaming_args
from opencood.models.heter_pyramid_collab import HeterPyramidCollab
from opencood.models.heter_pyramid_collab_open_dcsi_stage1 import (
    HeterPyramidCollabOpenDcsiStage1,
)


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameter_stats(model):
    total = sum(parameter.numel() for _, parameter in model.named_parameters())
    trainable = sum(
        parameter.numel()
        for _, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "checkpoint_bytes": buffer.tell(),
    }


def _macs(model, data):
    total = {"macs": 0}
    handles = []

    def hook(module, inputs, output):
        output_tensor = output[0] if isinstance(output, (list, tuple)) else output
        if not torch.is_tensor(output_tensor):
            return
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            per_output = module.in_channels // module.groups * kernel
            total["macs"] += output_tensor.numel() * per_output
        elif isinstance(module, nn.Linear):
            total["macs"] += output_tensor.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(data)
    for handle in handles:
        handle.remove()
    return total["macs"]


def _module_timer(model, device):
    totals = {"codec": 0.0, "fusion": 0.0, "refinement": 0.0}
    starts = {}
    handles = []
    for name, module in model.named_modules():
        category = None
        if "codec" in name or name.endswith("communication"):
            category = "codec"
        elif "common_fusion" in name or "streaming_common_fusion" in name:
            category = "fusion"
        elif "geometry_refiner" in name:
            category = "refinement"
        if category is None:
            continue

        def pre_hook(current, _inputs, key=(category, id(module))):
            _synchronize(device)
            starts[key] = time.perf_counter()

        def post_hook(current, _inputs, _output, key=(category, id(module))):
            _synchronize(device)
            totals[key[0]] += time.perf_counter() - starts.pop(key)

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    return totals, handles


def _profile(model, data, device, warmup, iterations):
    model = model.to(device).eval()
    data = _move(data, device)
    for _ in range(warmup):
        with torch.no_grad():
            model(data)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    module_times, handles = _module_timer(model, device)
    latencies = []
    output = None
    for _ in range(iterations):
        _synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model(data)
        _synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
    for handle in handles:
        handle.remove()
    macs = _macs(model, data)
    memory = {
        "max_memory_allocated": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "max_memory_reserved": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else 0,
    }
    report = {
        **_parameter_stats(model),
        **memory,
        "module_macs": int(macs),
        "module_flops_estimate": int(2 * macs),
        "warmup_count": warmup,
        "timed_iterations": iterations,
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "codec_time_ms": module_times["codec"] * 1000.0 / iterations,
        "fusion_time_ms": module_times["fusion"] * 1000.0 / iterations,
        "refinement_time_ms": module_times["refinement"] * 1000.0 / iterations,
    }
    if output is not None and "open_dcsi" in output:
        open_output = output["open_dcsi"]
        report["communication"] = open_output.get("communication_stats")
        streaming = open_output.get("streaming_stats")
        report["fusion_local_peak_bytes"] = (
            streaming["fusion_local_peak_bytes"] if streaming else 0
        )
        tokens = open_output.get("innovation_tokens")
        if tokens is not None:
            report["tokens_total"] = int(tokens["scenario_index"].numel())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    device = torch.device(args.device)
    data = _collab_input()
    official = _profile(
        HeterPyramidCollab(_tiny_model_args("missing")),
        data,
        device,
        args.warmup,
        args.iterations,
    )
    dense_model = HeterPyramidCollabOpenDcsiStage1(_communication_args())
    streaming_model = HeterPyramidCollabOpenDcsiStage1(_streaming_args())
    streaming_model.load_state_dict(dense_model.state_dict(), strict=True)
    dense = _profile(
        dense_model,
        data,
        device,
        args.warmup,
        args.iterations,
    )
    streaming = _profile(
        streaming_model,
        data,
        device,
        args.warmup,
        args.iterations,
    )
    report = {
        "device": str(device),
        "batch_size": 1,
        "cav_count": 2,
        "input_spatial_shape": [64, 8, 8],
        "official": official,
        "open_dcsi_dense": dense,
        "open_dcsi_streaming": streaming,
        "added_parameters": dense["total_parameters"] - official["total_parameters"],
        "added_parameter_ratio": (
            (dense["total_parameters"] - official["total_parameters"])
            / official["total_parameters"]
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print("OPEN_DCSI_RESOURCE_PROFILE_PASS")


if __name__ == "__main__":
    main()

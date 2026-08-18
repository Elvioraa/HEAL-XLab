"""Safe, model-aware launcher for optional post-training inference."""

from dataclasses import dataclass
import os
import subprocess
import sys


_DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


@dataclass(frozen=True)
class PostTrainInferencePlan:
    """Immutable post-training action decided while the model still exists."""

    action: str
    reason: str
    command: tuple = ()


@dataclass(frozen=True)
class PostTrainInferenceResult:
    """Describe whether post-training inference ran, skipped, or failed."""

    status: str
    reason: str
    command: tuple = ()
    returncode: object = None


def build_post_train_inference_command(model_dir, fusion_method):
    """Build a shell-free inference command using the active interpreter."""
    if not isinstance(model_dir, str) or not model_dir:
        raise ValueError("model_dir must be a non-empty string")
    if not isinstance(fusion_method, str) or not fusion_method:
        raise ValueError("fusion_method must be a non-empty string")
    return (
        sys.executable,
        "-m",
        "opencood.tools.inference",
        "--model_dir",
        model_dir,
        "--fusion_method",
        fusion_method,
    )


def prepare_post_train_inference(
    model,
    model_dir,
    fusion_method,
    rank=0,
    print_fn=print,
):
    """Create a RUN/SKIP plan without starting a subprocess.

    Stage1 is supported by the inference entry point's validated in-memory
    runtime-mode conversion. Stage2 still requires the official merge followed
    by heterogeneous ordered inference.
    """
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an integer")
    if rank != 0:
        return PostTrainInferencePlan(
            "skip", "post-training action is rank0-only"
        )

    skip_reason = _dual_space_skip_reason(model)
    if skip_reason is not None:
        print_fn("[PostTrainInference] SKIP: %s" % skip_reason)
        return PostTrainInferencePlan("skip", skip_reason)

    command = build_post_train_inference_command(model_dir, fusion_method)
    print_fn("[PostTrainInference] Prepared: %s" % " ".join(command))
    return PostTrainInferencePlan("run", "", command)


def execute_post_train_inference(
    plan,
    runner=subprocess.run,
    print_fn=print,
    environ=None,
):
    """Execute a prepared plan after all training resources are released."""
    if not isinstance(plan, PostTrainInferencePlan):
        raise TypeError("plan must be PostTrainInferencePlan")
    if plan.action == "skip":
        return PostTrainInferenceResult("skipped", plan.reason)
    if plan.action != "run" or not plan.command:
        raise ValueError("post-training plan must be RUN or SKIP")

    child_env = build_standalone_inference_environment(environ)
    print_fn("[PostTrainInference] Running: %s" % " ".join(plan.command))
    try:
        completed = runner(list(plan.command), check=False, env=child_env)
    except Exception as error:
        reason = "%s: %s" % (type(error).__name__, error)
        print_fn("[PostTrainInference] FAILED: %s" % reason)
        return PostTrainInferenceResult("failed", reason, plan.command)

    returncode = int(completed.returncode)
    if returncode != 0:
        reason = "inference subprocess exited with code %d" % returncode
        print_fn("[PostTrainInference] FAILED: %s" % reason)
        return PostTrainInferenceResult(
            "failed", reason, plan.command, returncode
        )
    print_fn("[PostTrainInference] SUCCEEDED")
    return PostTrainInferenceResult("succeeded", "", plan.command, returncode)


def build_standalone_inference_environment(environ=None):
    """Copy the parent environment without distributed-worker identity."""
    source = os.environ if environ is None else environ
    child_env = dict(source)
    for key in _DISTRIBUTED_ENV_KEYS:
        child_env.pop(key, None)
    return child_env


def _dual_space_skip_reason(model):
    if not getattr(model, "dual_space_enabled", False):
        return None
    config = getattr(model, "dual_space_config", None)
    if not isinstance(config, dict):
        return "DualSpace is enabled but dual_space_config is unavailable"
    mode = config.get("mode")
    if mode == "stage2_adapt":
        return (
            "DualSpace mode=stage2_adapt. Standalone inference is not a valid "
            "final evaluation. Run merge_final followed by "
            "inference_heter_in_order."
        )
    if mode == "stage1_anchor":
        return None
    if mode != "inference":
        return "unsupported DualSpace post-training mode=%r" % mode
    return None

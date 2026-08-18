"""CPU-only smoke tests for model-aware post-training inference launch."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from opencood.tools.post_train_inference import (
    PostTrainInferencePlan,
    build_standalone_inference_environment,
    execute_post_train_inference,
    prepare_post_train_inference,
)


TESTS = []


def test(name):
    def register(function):
        TESTS.append((name, function))
        return function
    return register


class FakeModel(object):
    def __init__(self, mode=None):
        self.dual_space_enabled = mode is not None
        self.dual_space_config = {} if mode is None else {"mode": mode}


class Completed(object):
    def __init__(self, returncode):
        self.returncode = returncode


def recording_runner(returncode=0):
    calls = []

    def run(command, check, env):
        calls.append((command, check, env))
        return Completed(returncode)

    return calls, run


@test("prepare creates a RUN plan without invoking subprocess")
def test_prepare_does_not_execute():
    calls, runner = recording_runner()
    plan = prepare_post_train_inference(
        FakeModel(), "logs/model", "intermediate",
        print_fn=lambda message: None,
    )
    assert plan.action == "run"
    assert calls == []


@test("DualSpace stage2 skips standalone inference with merge guidance")
def test_stage2_skip():
    plan = prepare_post_train_inference(
        FakeModel("stage2_adapt"), "logs/stage2", "intermediate",
        print_fn=lambda message: None,
    )
    assert plan.action == "skip"
    assert "merge_final" in plan.reason
    assert "inference_heter_in_order" in plan.reason


@test("DualSpace stage1 uses validated runtime inference without mutating mode")
def test_stage1_runtime_inference():
    model = FakeModel("stage1_anchor")
    plan = prepare_post_train_inference(
        model, "logs/stage1", "intermediate",
        print_fn=lambda message: None,
    )
    assert plan.action == "run"
    assert model.dual_space_config["mode"] == "stage1_anchor"


@test("non-rank0 performs no post-training action or logging")
def test_non_rank0_skip():
    calls, runner = recording_runner()
    messages = []
    plan = prepare_post_train_inference(
        FakeModel(), "logs/model", "intermediate", rank=1,
        print_fn=messages.append,
    )
    assert plan.action == "skip"
    assert calls == []
    assert messages == []


@test("execute RUN uses module command and sanitized standalone environment")
def test_execute_run():
    calls, runner = recording_runner()
    plan = prepare_post_train_inference(
        FakeModel(), "logs/model", "intermediate",
        print_fn=lambda message: None,
    )
    parent_env = {
        "RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "2", "GROUP_RANK": "0", "ROLE_RANK": "0",
        "MASTER_ADDR": "localhost", "MASTER_PORT": "12345",
        "CUDA_VISIBLE_DEVICES": "0,2", "KEEP_ME": "yes",
    }
    result = execute_post_train_inference(
        plan, runner=runner, environ=parent_env,
        print_fn=lambda message: None,
    )
    assert result.status == "succeeded"
    command, check, child_env = calls[0]
    assert command[:3] == [sys.executable, "-m", "opencood.tools.inference"]
    assert command[-4:] == [
        "--model_dir", "logs/model", "--fusion_method", "intermediate"
    ]
    assert check is False
    for key in (
        "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "ROLE_RANK", "MASTER_ADDR", "MASTER_PORT",
    ):
        assert key not in child_env
    assert child_env["CUDA_VISIBLE_DEVICES"] == "0,2"
    assert child_env["KEEP_ME"] == "yes"


@test("execute SKIP never invokes subprocess")
def test_execute_skip():
    calls, runner = recording_runner()
    result = execute_post_train_inference(
        PostTrainInferencePlan("skip", "expected"), runner=runner,
        print_fn=lambda message: None,
    )
    assert result.status == "skipped"
    assert calls == []


@test("standalone environment copy does not mutate parent mapping")
def test_environment_copy():
    parent = {"RANK": "0", "CUDA_VISIBLE_DEVICES": "1", "OTHER": "x"}
    child = build_standalone_inference_environment(parent)
    assert parent["RANK"] == "0"
    assert "RANK" not in child
    assert child["CUDA_VISIBLE_DEVICES"] == "1"
    assert child["OTHER"] == "x"


@test("subprocess failure is reported separately from completed training")
def test_failure_result():
    calls, runner = recording_runner(returncode=7)
    messages = []
    plan = prepare_post_train_inference(
        FakeModel(), "logs/model", "intermediate", print_fn=lambda message: None,
    )
    result = execute_post_train_inference(
        plan, runner=runner, print_fn=messages.append, environ={},
    )
    assert result.status == "failed"
    assert result.returncode == 7
    assert len(calls) == 1
    assert messages[-1] == (
        "[PostTrainInference] FAILED: inference subprocess exited with code 7"
    )


@test("all training entry points avoid direct inference script commands")
def test_entry_point_sources():
    for name in ("train.py", "train_ddp.py", "train_w_kd.py"):
        path = os.path.join(os.path.dirname(__file__), name)
        with open(path, "r", encoding="utf-8") as stream:
            source = stream.read()
        assert "opencood/tools/inference.py" not in source, name
        assert "os.system(" not in source, name
        assert "prepare_post_train_inference" in source, name
        assert "execute_post_train_inference" in source, name


@test("DDP teardown orders barriers cleanup and execution safely")
def test_ddp_teardown_order():
    path = os.path.join(os.path.dirname(__file__), "train_ddp.py")
    with open(path, "r", encoding="utf-8") as stream:
        source = stream.read()
    first_barrier = source.find("torch.distributed.barrier()")
    second_barrier = source.find("torch.distributed.barrier()", first_barrier + 1)
    destroy = source.find("torch.distributed.destroy_process_group()")
    release = source.find("del model, model_without_ddp")
    execute = source.find("execute_post_train_inference(post_train_plan)")
    assert -1 not in (first_barrier, second_barrier, destroy, release, execute)
    assert first_barrier < second_barrier < destroy < release < execute


def main():
    passed = 0
    for name, function in TESTS:
        try:
            function()
        except Exception as error:
            print("[FAIL] %s: %s: %s" % (name, type(error).__name__, error))
        else:
            passed += 1
            print("[PASS] %s" % name)
    print("RESULT: %d/%d PASS" % (passed, len(TESTS)))
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())

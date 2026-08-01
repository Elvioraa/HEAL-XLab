"""CPU smoke checks for the shared gradient-accumulation controller."""

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencood.tools.gradient_accumulation import GradientAccumulator


class _RecordingScaler:
    def __init__(self):
        self.scale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, loss):
        self.scale_calls += 1
        return loss

    def step(self, optimizer):
        self.step_calls += 1
        return optimizer.step()

    def update(self):
        self.update_calls += 1


def _run(accumulation_steps, entries):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = _RecordingScaler()
    accumulator = GradientAccumulator(accumulation_steps)
    accumulator.start_epoch(optimizer)
    for value in entries:
        if value is None or value == "invalid":
            continue
        loss = parameter * float(value)
        if accumulator.backward(loss, scaler):
            accumulator.step(optimizer, scaler)
    summary = accumulator.finish_epoch(optimizer)
    return parameter, summary, scaler


def _assert_close(actual, expected, message):
    if not torch.isclose(torch.as_tensor(actual), torch.as_tensor(expected)):
        raise AssertionError("%s: expected %r, got %r" % (message, expected, actual))


def _gradient_is_clear(parameter):
    return parameter.grad is None or bool(torch.count_nonzero(parameter.grad) == 0)


def test_eight_valid_one_step():
    parameter, summary, _ = _run(8, [1.0] * 8)
    assert summary.valid_micro_batches == 8
    assert summary.optimizer_updates == 1
    assert summary.dropped_tail_micro_batches == 0
    _assert_close(parameter.detach(), 0.9, "one averaged update")


def test_sixteen_valid_two_steps():
    parameter, summary, _ = _run(8, [1.0] * 16)
    assert summary.optimizer_updates == 2
    _assert_close(parameter.detach(), 0.8, "two averaged updates")


def test_invalid_entries_do_not_count():
    entries = [None, 1.0, "invalid", 1.0, None, 1.0, 1.0,
               "invalid", 1.0, 1.0, None, 1.0, 1.0]
    _, summary, _ = _run(8, entries)
    assert summary.valid_micro_batches == 8
    assert summary.optimizer_updates == 1
    assert summary.dropped_tail_micro_batches == 0


def test_tails_one_through_seven_are_dropped():
    for tail in range(1, 8):
        parameter, summary, _ = _run(8, [1.0] * tail)
        assert summary.optimizer_updates == 0
        assert summary.dropped_tail_micro_batches == tail
        _assert_close(parameter.detach(), 1.0, "tail must not update")
        assert _gradient_is_clear(parameter)


def test_accumulation_sixteen_tail():
    _, summary, _ = _run(16, [1.0] * 31)
    assert summary.valid_micro_batches == 31
    assert summary.optimizer_updates == 1
    assert summary.dropped_tail_micro_batches == 15


def test_dropped_tail_clears_gradient():
    parameter, summary, _ = _run(8, [2.0] * 3)
    assert summary.dropped_tail_micro_batches == 3
    assert _gradient_is_clear(parameter)


def test_accumulation_one_matches_ordinary_training():
    values = [1.0, 2.0, 3.0]
    accumulated, summary, _ = _run(1, values)

    ordinary = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([ordinary], lr=0.1)
    for value in values:
        optimizer.zero_grad()
        (ordinary * value).backward()
        optimizer.step()
    assert summary.optimizer_updates == len(values)
    _assert_close(accumulated.detach(), ordinary.detach(), "accumulation=1")


def test_summary_update_count():
    _, summary, _ = _run(8, [1.0] * 17)
    assert summary.valid_micro_batches == 17
    assert summary.optimizer_updates == 2
    assert summary.dropped_tail_micro_batches == 1


def test_scaler_control_path():
    _, summary, scaler = _run(8, [1.0] * 10)
    assert summary.optimizer_updates == 1
    assert scaler.scale_calls == 10
    assert scaler.step_calls == 1
    assert scaler.update_calls == 1


def test_constructor_rejects_non_integer_values():
    invalid_values = (True, False, 1.5, 8.0, "8", "abc", None, 0, -1)
    for value in invalid_values:
        try:
            GradientAccumulator(value)
        except (TypeError, ValueError):
            continue
        raise AssertionError("invalid accumulation value accepted: %r" % value)


def test_disabled_grad_scaler_fp32_path():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    accumulator = GradientAccumulator(2)
    accumulator.start_epoch(optimizer)
    for _ in range(2):
        loss = parameter * 1.0
        if accumulator.backward(loss, scaler):
            accumulator.step(optimizer, scaler)
    summary = accumulator.finish_epoch(optimizer)
    assert summary.optimizer_updates == 1
    assert scaler.is_enabled() is False
    assert torch.isclose(parameter.detach(), torch.tensor(0.9))


def _run_cuda_amp_check():
    if not torch.cuda.is_available():
        print("SKIPPED: CUDA AMP scaler path (CUDA unavailable)")
        return None
    parameter = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    accumulator = GradientAccumulator(2)
    accumulator.start_epoch(optimizer)
    for _ in range(2):
        with torch.cuda.amp.autocast(enabled=True):
            loss = parameter * 1.0
        if accumulator.backward(loss, scaler):
            accumulator.step(optimizer, scaler)
    summary = accumulator.finish_epoch(optimizer)
    if summary.optimizer_updates != 1 or not torch.isfinite(parameter):
        raise AssertionError("CUDA AMP accumulation failed")
    print("PASS: CUDA AMP scaler path")
    return True


TESTS = (
    ("8 valid micro-batches -> 1 step", test_eight_valid_one_step),
    ("16 valid micro-batches -> 2 steps", test_sixteen_valid_two_steps),
    ("invalid/None entries do not count", test_invalid_entries_do_not_count),
    ("tails 1-7 are dropped", test_tails_one_through_seven_are_dropped),
    ("accumulation=16 tail", test_accumulation_sixteen_tail),
    ("dropped tail clears gradients", test_dropped_tail_clears_gradient),
    ("accumulation=1 ordinary equivalence", test_accumulation_one_matches_ordinary_training),
    ("summary update count", test_summary_update_count),
    ("scaler control path", test_scaler_control_path),
    ("strict constructor types", test_constructor_rejects_non_integer_values),
    ("disabled GradScaler FP32 path", test_disabled_grad_scaler_fp32_path),
)


def main():
    passed = 0
    for index, (name, test) in enumerate(TESTS, 1):
        try:
            test()
        except Exception as exc:
            print("FAIL: %02d %s: %s" % (index, name, exc))
            return 1
        passed += 1
        print("PASS: %02d %s" % (index, name))
    try:
        cuda_result = _run_cuda_amp_check()
    except Exception as exc:
        print("FAIL: CUDA AMP scaler path: %s" % exc)
        return 1
    print("CUDA_AMP: %s" % ("PASS" if cuda_result else "SKIPPED"))
    print("RESULT: %d/%d PASS" % (passed, len(TESTS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

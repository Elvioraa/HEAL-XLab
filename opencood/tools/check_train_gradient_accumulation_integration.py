"""CPU integration checks for HEAL train.py accumulation control semantics."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
from unittest import mock

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencood.tools.gradient_accumulation import (  # noqa: E402
    GradientAccumulator,
    calculate_effective_global_batch,
    finish_accumulation_epoch,
    get_runtime_world_size,
    resolve_accumulation_steps,
    resolve_amp_setting,
)
from opencood.tools.train import train_parser  # noqa: E402


TRAIN_PATH = REPO_ROOT / "opencood/tools/train.py"


class _RecordingScaler:
    def __init__(self):
        self.scaled_values = []
        self.step_calls = 0
        self.update_calls = 0

    def scale(self, loss):
        self.scaled_values.append(float(loss.detach()))
        return loss

    def step(self, optimizer):
        self.step_calls += 1
        optimizer.step()

    def update(self):
        self.update_calls += 1


def _assert_rejected(value, resolver=resolve_accumulation_steps):
    try:
        resolver(value)
    except (TypeError, ValueError):
        return
    raise AssertionError("invalid value was accepted: %r" % value)


def _run_valid_entries(accumulation_steps, entries):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = _RecordingScaler()
    accumulator = GradientAccumulator(accumulation_steps)
    accumulator.start_epoch(optimizer)
    for value in entries:
        if value is None:
            continue
        loss = parameter * float(value)
        if accumulator.backward(loss, scaler):
            accumulator.step(optimizer, scaler)
    return parameter, optimizer, scaler, accumulator


def _gradient_is_clear(parameter):
    return parameter.grad is None or bool(torch.count_nonzero(parameter.grad) == 0)


def test_yaml_accumulation_eight():
    value = yaml.safe_load("value: 8")["value"]
    assert resolve_accumulation_steps(value) == 8


def test_cli_accumulation_overrides_yaml():
    opt = train_parser(["-y", "dummy.yaml", "--accumulation-steps", "4"])
    assert resolve_accumulation_steps(8, opt.accumulation_steps) == 4
    assert resolve_accumulation_steps(True, opt.accumulation_steps) == 4


def test_yaml_true_rejected():
    _assert_rejected(yaml.safe_load("value: true")["value"])


def test_yaml_false_rejected():
    _assert_rejected(yaml.safe_load("value: false")["value"])


def test_yaml_floats_rejected():
    _assert_rejected(yaml.safe_load("value: 1.5")["value"])
    _assert_rejected(yaml.safe_load("value: 8.0")["value"])


def test_other_invalid_accumulation_values_rejected():
    for text in ("value: 0", "value: -1", "value: abc", "value: null"):
        _assert_rejected(yaml.safe_load(text)["value"])


def test_yaml_amp_false_and_string_rejection():
    assert resolve_amp_setting(yaml.safe_load("value: false")["value"]) is False
    _assert_rejected(
        yaml.safe_load('value: "false"')["value"],
        resolver=resolve_amp_setting,
    )


def test_cli_amp_overrides_yaml_false():
    opt = train_parser(["-y", "dummy.yaml", "--amp"])
    assert resolve_amp_setting(False, opt.amp_override) is True


def test_cli_no_amp_overrides_yaml_true():
    opt = train_parser(["-y", "dummy.yaml", "--no-amp"])
    assert resolve_amp_setting(True, opt.amp_override) is False


def test_amp_cli_flags_are_mutually_exclusive():
    with redirect_stderr(io.StringIO()):
        try:
            train_parser(["-y", "dummy.yaml", "--amp", "--no-amp"])
        except SystemExit as exc:
            assert exc.code != 0
            return
    raise AssertionError("--amp and --no-amp were accepted together")


def test_effective_global_batch_formula():
    assert calculate_effective_global_batch(2, 4, 3) == 24
    assert get_runtime_world_size() == 1
    with mock.patch("torch.distributed.is_available", return_value=True), \
            mock.patch("torch.distributed.is_initialized", return_value=True), \
            mock.patch("torch.distributed.get_world_size", return_value=3):
        assert get_runtime_world_size() == 3


def test_eight_valid_micro_batches_make_one_update():
    _, optimizer, scaler, accumulator = _run_valid_entries(8, [1.0] * 8)
    summary = finish_accumulation_epoch(accumulator, optimizer, 8)
    assert summary.optimizer_updates == 1
    assert scaler.step_calls == 1
    assert scaler.update_calls == 1


def test_skipped_batches_do_not_break_window():
    entries = [1.0, None, 1.0, None, 1.0, 1.0, None, 1.0, 1.0, 1.0, 1.0]
    _, optimizer, _, accumulator = _run_valid_entries(8, entries)
    summary = finish_accumulation_epoch(accumulator, optimizer, len(entries))
    assert summary.valid_micro_batches == 8
    assert summary.optimizer_updates == 1


def test_tail_after_update_is_dropped_and_cleared():
    parameter, optimizer, _, accumulator = _run_valid_entries(8, [1.0] * 10)
    summary = finish_accumulation_epoch(accumulator, optimizer, 10)
    assert summary.optimizer_updates == 1
    assert summary.dropped_tail_micro_batches == 2
    assert _gradient_is_clear(parameter)


def test_zero_update_epoch_raises_with_context():
    parameter, optimizer, _, accumulator = _run_valid_entries(8, [1.0] * 3)
    try:
        finish_accumulation_epoch(accumulator, optimizer, 12)
    except RuntimeError as exc:
        message = str(exc)
        for token in (
                "valid_micro_batches=3",
                "accumulation_steps=8",
                "dropped_tail_micro_batches=3",
                "train_loader_length=12",
                "Reduce accumulation_steps",
                "invalid/skipped batches"):
            if token not in message:
                raise AssertionError("zero-update error lacks %r" % token)
        assert _gradient_is_clear(parameter)
        return
    raise AssertionError("zero-update epoch did not raise RuntimeError")


def test_zero_update_guard_precedes_all_post_epoch_actions():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    guard = source.index("accumulation_summary = finish_accumulation_epoch(")
    checkpoint = source.index("if epoch % hypes['train_params']['save_freq']")
    validation = source.index("if epoch % hypes['train_params']['eval_freq']")
    scheduler = source.index("scheduler.step(epoch)")
    if not guard < checkpoint < validation < scheduler:
        raise AssertionError("zero-update guard is after a post-epoch action")
    if source.count("scheduler.step(epoch)") != 1:
        raise AssertionError("scheduler.step(epoch) must occur exactly once")


def test_accumulation_one_matches_ordinary_updates():
    values = [1.0, 2.0, 3.0]
    accumulated, optimizer, _, accumulator = _run_valid_entries(1, values)
    summary = finish_accumulation_epoch(accumulator, optimizer, len(values))

    ordinary = torch.nn.Parameter(torch.tensor(1.0))
    ordinary_optimizer = torch.optim.SGD([ordinary], lr=0.1)
    for value in values:
        ordinary_optimizer.zero_grad()
        (ordinary * value).backward()
        ordinary_optimizer.step()
    assert summary.optimizer_updates == len(values)
    assert torch.isclose(accumulated.detach(), ordinary.detach())


def test_logging_observes_unscaled_loss():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = _RecordingScaler()
    accumulator = GradientAccumulator(8)
    accumulator.start_epoch(optimizer)
    original_loss = parameter * 8.0
    logged_value = float(original_loss.detach())
    accumulator.backward(original_loss, scaler)
    accumulator.finish_epoch(optimizer)
    assert logged_value == 8.0
    assert scaler.scaled_values == [1.0]

    source = TRAIN_PATH.read_text(encoding="utf-8")
    logging_position = source.index("criterion.logging(epoch, i, len(train_loader), writer)")
    backward_position = source.index("accumulator.backward(final_loss, scaler)")
    assert logging_position < backward_position


TESTS = (
    ("YAML accumulation=8", test_yaml_accumulation_eight),
    ("CLI accumulation overrides YAML", test_cli_accumulation_overrides_yaml),
    ("YAML true rejected", test_yaml_true_rejected),
    ("YAML false rejected", test_yaml_false_rejected),
    ("YAML floats rejected", test_yaml_floats_rejected),
    ("YAML zero/negative/string/null rejected", test_other_invalid_accumulation_values_rejected),
    ("strict YAML AMP", test_yaml_amp_false_and_string_rejection),
    ("CLI --amp override", test_cli_amp_overrides_yaml_false),
    ("CLI --no-amp override", test_cli_no_amp_overrides_yaml_true),
    ("AMP CLI mutual exclusion", test_amp_cli_flags_are_mutually_exclusive),
    ("effective global batch formula", test_effective_global_batch_formula),
    ("8 valid micro-batches -> 1 update", test_eight_valid_micro_batches_make_one_update),
    ("skipped batches preserve window", test_skipped_batches_do_not_break_window),
    ("updated epoch drops tail", test_tail_after_update_is_dropped_and_cleared),
    ("zero-update epoch raises", test_zero_update_epoch_raises_with_context),
    ("zero-update guard ordering", test_zero_update_guard_precedes_all_post_epoch_actions),
    ("accumulation=1 equivalence", test_accumulation_one_matches_ordinary_updates),
    ("logging uses original loss", test_logging_observes_unscaled_loss),
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
    print("RESULT: %d/%d PASS" % (passed, len(TESTS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

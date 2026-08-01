"""Shared training controls for strict gradient accumulation semantics."""

from dataclasses import dataclass


def _positive_python_integer(value, name):
    """Validate and return a positive Python integer without coercion."""
    if type(value) is not int:
        raise TypeError("%s must be a positive integer" % name)
    if value < 1:
        raise ValueError("%s must be at least 1" % name)
    return value


def resolve_accumulation_steps(yaml_value, cli_value=None):
    """Resolve strict accumulation steps, with an explicit CLI value winning.

    YAML booleans, floats, strings, and null values are intentionally rejected
    instead of being coerced to integers.
    """
    selected = cli_value if cli_value is not None else yaml_value
    return _positive_python_integer(selected, "accumulation_steps")


def resolve_amp_setting(yaml_value, cli_value=None):
    """Resolve a strict boolean AMP setting, with CLI taking precedence."""
    selected = cli_value if cli_value is not None else yaml_value
    if type(selected) is not bool:
        raise TypeError("amp must be a boolean")
    return selected


def get_runtime_world_size():
    """Return the initialized distributed world size, otherwise one."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def calculate_effective_global_batch(
        micro_batch, accumulation_steps, world_size):
    """Calculate micro-batch x accumulation x distributed world size."""
    micro_batch = _positive_python_integer(micro_batch, "micro_batch")
    accumulation_steps = _positive_python_integer(
        accumulation_steps, "accumulation_steps"
    )
    world_size = _positive_python_integer(world_size, "world_size")
    return micro_batch * accumulation_steps * world_size


@dataclass(frozen=True)
class AccumulationEpochSummary:
    """Counters produced when one accumulation epoch finishes."""

    valid_micro_batches: int
    optimizer_updates: int
    dropped_tail_micro_batches: int


class GradientAccumulator:
    """Coordinate backward, optimizer updates, and drop-last tail handling.

    Callers must invoke :meth:`backward` only after a batch has completed a
    valid forward pass and produced a loss. Skipped batches therefore never
    occupy a position in the accumulation window.
    """

    def __init__(self, accumulation_steps):
        self.accumulation_steps = _positive_python_integer(
            accumulation_steps, "accumulation_steps"
        )
        self._active = False
        self._window_micro_batches = 0
        self._valid_micro_batches = 0
        self._optimizer_updates = 0

    @property
    def window_micro_batches(self):
        """Number of valid micro-batches currently holding gradients."""
        return self._window_micro_batches

    @property
    def ready_to_step(self):
        """Whether the current window has reached its configured size."""
        return self._window_micro_batches == self.accumulation_steps

    def start_epoch(self, optimizer):
        """Clear gradients and reset all per-epoch counters."""
        optimizer.zero_grad()
        self._active = True
        self._window_micro_batches = 0
        self._valid_micro_batches = 0
        self._optimizer_updates = 0

    def backward(self, original_loss, scaler):
        """Backpropagate one valid loss and return whether a step is due."""
        self._require_active()
        if self.ready_to_step:
            raise RuntimeError("optimizer step required before another backward")
        scaled_loss = original_loss / self.accumulation_steps
        scaler.scale(scaled_loss).backward()
        self._window_micro_batches += 1
        self._valid_micro_batches += 1
        return self.ready_to_step

    def step(self, optimizer, scaler):
        """Apply one complete window and clear its gradients."""
        self._require_active()
        if not self.ready_to_step:
            raise RuntimeError("cannot step an incomplete accumulation window")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        self._window_micro_batches = 0
        self._optimizer_updates += 1

    def finish_epoch(self, optimizer):
        """Drop an incomplete tail, clear gradients, and return counters."""
        self._require_active()
        dropped_tail = self._window_micro_batches
        if dropped_tail:
            optimizer.zero_grad()
        summary = AccumulationEpochSummary(
            valid_micro_batches=self._valid_micro_batches,
            optimizer_updates=self._optimizer_updates,
            dropped_tail_micro_batches=dropped_tail,
        )
        self._window_micro_batches = 0
        self._active = False
        return summary

    def _require_active(self):
        if not self._active:
            raise RuntimeError("start_epoch must be called before accumulation")


def finish_accumulation_epoch(accumulator, optimizer, train_loader_length):
    """Finish an epoch and reject it before post-training work if no step ran.

    Any incomplete gradient window is cleared by ``finish_epoch`` before this
    function raises, so failed epochs cannot leak gradients into a later epoch.
    """
    summary = accumulator.finish_epoch(optimizer)
    if summary.optimizer_updates == 0:
        raise RuntimeError(
            "Training epoch produced zero optimizer updates: "
            "valid_micro_batches=%d, accumulation_steps=%d, "
            "dropped_tail_micro_batches=%d, train_loader_length=%d. "
            "Reduce accumulation_steps or inspect invalid/skipped batches."
            % (
                summary.valid_micro_batches,
                accumulator.accumulation_steps,
                summary.dropped_tail_micro_batches,
                train_loader_length,
            )
        )
    return summary

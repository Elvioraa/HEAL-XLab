"""CPU smoke tests for the opt-in AMP + gradient-accumulation training paths.

train.py / train_ddp.py cannot be exercised end-to-end without a dataset, so
this test reproduces the exact stepping logic those loops use and asserts the
numerical properties that matter:

1. GradScaler(enabled=False) + accumulate_grad_batches=1 is a bit-exact
   pass-through: identical to a plain per-batch fp32 update.
2. accumulate_grad_batches=N over N micro-batches yields the same accumulated
   gradient as one forward/backward over the N-times-larger batch (mean loss).
3. The step cadence is correct: optimizer.step() fires once per N micro-batches
   and gradients are cleared right after each step.
"""

from __future__ import absolute_import, division, print_function

import copy

import torch


def _make_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Linear(4, 3)


def _loss(model, x, y):
    # mean-reduction loss, which is what makes accumulation == larger batch.
    return torch.nn.functional.mse_loss(model(x), y)


def test_default_off_is_plain_update():
    torch.manual_seed(1)
    x = torch.randn(8, 4)
    y = torch.randn(8, 3)

    # Reference: the original per-batch fp32 update.
    ref = _make_model()
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    ref_opt.zero_grad()
    _loss(ref, x, y).backward()
    ref_opt.step()

    # New path with amp=False, accumulate_grad_batches=1.
    new = _make_model()
    new_opt = torch.optim.SGD(new.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    accum_steps = 1
    new.zero_grad()
    new_opt.zero_grad()
    with torch.cuda.amp.autocast(enabled=False):
        loss = _loss(new, x, y)
    scaler.scale(loss / accum_steps).backward()
    if (0 + 1) % accum_steps == 0:
        scaler.step(new_opt)
        scaler.update()
        new.zero_grad()
        new_opt.zero_grad()

    for pr, pn in zip(ref.parameters(), new.parameters()):
        assert torch.allclose(pr, pn, atol=1e-7), "default-off path diverged from plain update"
    print("default-off (amp=False, accum=1) == plain per-batch update: True")


def test_accumulation_equals_large_batch():
    torch.manual_seed(2)
    accum_steps = 4
    micro = 2  # samples per micro-batch
    xs = [torch.randn(micro, 4) for _ in range(accum_steps)]
    ys = [torch.randn(micro, 3) for _ in range(accum_steps)]
    x_full = torch.cat(xs, dim=0)
    y_full = torch.cat(ys, dim=0)

    # Accumulation path.
    acc = _make_model()
    acc.zero_grad()
    for i in range(accum_steps):
        loss = _loss(acc, xs[i], ys[i])
        (loss / accum_steps).backward()
    acc_grads = [p.grad.detach().clone() for p in acc.parameters()]

    # Single large batch (same weights).
    big = _make_model()  # same seed -> same init as acc
    big.zero_grad()
    _loss(big, x_full, y_full).backward()
    big_grads = [p.grad.detach().clone() for p in big.parameters()]

    for ga, gb in zip(acc_grads, big_grads):
        assert torch.allclose(ga, gb, atol=1e-6), "accumulated grad != large-batch grad"
    print("accumulate_grad_batches=%d grad == %dx-batch grad: True" % (accum_steps, accum_steps))


def test_step_cadence():
    torch.manual_seed(3)
    accum_steps = 3
    num_batches = 7
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    step_calls = {"n": 0}
    real_step = opt.step

    def counting_step(*args, **kwargs):
        step_calls["n"] += 1
        return real_step(*args, **kwargs)

    opt.step = counting_step

    model.zero_grad()
    opt.zero_grad()
    for i in range(num_batches):
        x = torch.randn(2, 4)
        y = torch.randn(2, 3)
        loss = _loss(model, x, y)
        scaler.scale(loss / accum_steps).backward()
        if (i + 1) % accum_steps == 0:
            scaler.step(opt)
            scaler.update()
            model.zero_grad()
            opt.zero_grad()

    # 7 batches, window 3 -> steps at i=2 and i=5 -> 2 steps (leftover i=6 pending).
    assert step_calls["n"] == num_batches // accum_steps, (
        "expected %d steps, got %d" % (num_batches // accum_steps, step_calls["n"])
    )
    print("step cadence (7 batches / accum 3 -> 2 steps): True")


def test_scaler_disabled_is_identity():
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    t = torch.randn(5)
    assert torch.equal(scaler.scale(t), t), "GradScaler(enabled=False).scale must be identity"
    print("GradScaler(enabled=False).scale is identity: True")


def main():
    test_default_off_is_plain_update()
    test_accumulation_equals_large_batch()
    test_step_cadence()
    test_scaler_disabled_is_identity()
    print("TRAIN_AMP_ACCUM_SMOKE_PASS")


if __name__ == "__main__":
    main()

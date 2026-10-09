"""Learning-rate schedules as plain functions so they can be unit-tested."""

from __future__ import annotations

import math


def cosine_with_warmup(warmup_steps: int, total_steps: int, base_lr: float, min_lr: float):
    """Linear warmup from 0 to ``base_lr`` over ``warmup_steps``, then cosine decay to ``min_lr``.

    Returns a multiplier function suitable for ``torch.optim.lr_scheduler.LambdaLR``.
    Matches the SmolVLA recipe (1k warmup, cosine to 2.5e-6) when called with its defaults.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= min_lr <= base_lr:
        raise ValueError("need 0 <= min_lr <= base_lr")
    floor = min_lr / base_lr
    decay_steps = max(1, total_steps - warmup_steps)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return multiplier

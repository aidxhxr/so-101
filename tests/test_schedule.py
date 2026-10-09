import math

import pytest

from so101.schedule import cosine_with_warmup


def test_warmup_is_linear_then_hits_base_lr():
    f = cosine_with_warmup(warmup_steps=100, total_steps=1000, base_lr=1e-4, min_lr=1e-6)
    assert f(0) == 0.0
    assert f(50) == pytest.approx(0.5)
    assert f(100) == pytest.approx(1.0)


def test_cosine_decays_to_min_lr_and_stays_there():
    f = cosine_with_warmup(warmup_steps=100, total_steps=1000, base_lr=1e-4, min_lr=2.5e-6)
    assert f(1000) == pytest.approx(2.5e-6 / 1e-4)
    assert f(5000) == pytest.approx(2.5e-6 / 1e-4)
    mid = f(550)
    assert 2.5e-6 / 1e-4 < mid < 1.0
    assert mid == pytest.approx(0.025 + 0.975 * 0.5 * (1 + math.cos(math.pi * 0.5)))


def test_monotone_after_warmup():
    f = cosine_with_warmup(10, 200, 1.0, 0.0)
    vals = [f(s) for s in range(10, 201)]
    assert all(a >= b for a, b in zip(vals[:-1], vals[1:], strict=True))


def test_no_warmup():
    f = cosine_with_warmup(0, 10, 1.0, 0.0)
    assert f(0) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [dict(warmup_steps=0, total_steps=0), dict(min_lr=2.0)])
def test_invalid_arguments(bad):
    kwargs = dict(warmup_steps=1, total_steps=10, base_lr=1.0, min_lr=0.1)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        cosine_with_warmup(**kwargs)

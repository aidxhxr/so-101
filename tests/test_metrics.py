import math

import torch

from so101.metrics import RunningMean, hold_baseline, masked_chunk_errors, mean_action_baseline


def test_errors_without_padding():
    pred = torch.zeros(2, 3, 2)
    target = torch.tensor([[[1.0, 2.0]] * 3] * 2)
    r = masked_chunk_errors(pred, target)
    assert torch.allclose(r["mse"], torch.tensor([1.0, 4.0]))
    assert torch.allclose(r["mae"], torch.tensor([1.0, 2.0]))
    assert r["mse_mean"].item() == 2.5
    assert torch.allclose(r["mse_per_step"], torch.full((3,), 2.5))


def test_padding_is_ignored():
    pred = torch.zeros(1, 4, 1)
    target = torch.tensor([[[1.0], [1.0], [100.0], [100.0]]])
    is_pad = torch.tensor([[False, False, True, True]])
    r = masked_chunk_errors(pred, target, is_pad)
    assert r["mse"].item() == 1.0
    assert math.isnan(r["mse_per_step"][2].item())
    assert r["mse_per_step"][0].item() == 1.0


def test_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        masked_chunk_errors(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))


def test_running_mean_weights_by_batch_size():
    agg = RunningMean()
    agg.update({"x": torch.tensor(1.0)}, weight=1)
    agg.update({"x": torch.tensor(4.0)}, weight=3)
    assert agg.result()["x"].item() == 3.25


def test_running_mean_skips_nan_entries():
    agg = RunningMean()
    agg.update({"v": torch.tensor([1.0, float("nan")])}, weight=2)
    agg.update({"v": torch.tensor([3.0, 5.0])}, weight=2)
    out = agg.result()["v"]
    assert out[0].item() == 2.0
    assert out[1].item() == 2.5  # nan contributed 0 but the weight was still counted


def test_baselines_shapes():
    state = torch.arange(6.0).view(1, 6)
    h = hold_baseline(state, 50)
    assert h.shape == (1, 50, 6) and torch.equal(h[0, 49], state[0])
    m = mean_action_baseline(torch.ones(6), 4, 50)
    assert m.shape == (4, 50, 6) and m.sum().item() == 4 * 50 * 6

import pytest
import torch

from so101.data import images_to_float, split_episodes


def test_split_is_deterministic_and_disjoint():
    a = split_episodes(50, 0.1, seed=0)
    b = split_episodes(50, 0.1, seed=0)
    assert a == b
    train, val = a
    assert len(val) == 5 and len(train) == 45
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(50))


def test_split_changes_with_seed():
    assert split_episodes(50, 0.1, seed=0)[1] != split_episodes(50, 0.1, seed=1)[1]


def test_split_holds_out_at_least_one_episode():
    train, val = split_episodes(3, 0.1)
    assert len(val) == 1 and len(train) == 2


def test_split_no_validation():
    train, val = split_episodes(7, 0.0)
    assert val == [] and train == list(range(7))


@pytest.mark.parametrize("n, frac", [(0, 0.1), (10, 1.0), (10, -0.1)])
def test_split_rejects_bad_inputs(n, frac):
    with pytest.raises(ValueError):
        split_episodes(n, frac)


def test_images_to_float_scales_uint8_only():
    batch = {
        "observation.images.front": torch.full((2, 3, 4, 4), 255, dtype=torch.uint8),
        "observation.state": torch.ones(2, 6),
    }
    out = images_to_float(batch, ["observation.images.front", "missing"])
    assert out["observation.images.front"].dtype == torch.float32
    assert torch.allclose(out["observation.images.front"], torch.ones(2, 3, 4, 4))
    assert out["observation.state"].dtype == torch.float32

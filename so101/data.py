"""Dataset split and loading helpers around LeRobotDataset."""

from __future__ import annotations

import random
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader


def split_episodes(num_episodes: int, val_fraction: float, seed: int = 0) -> tuple[list[int], list[int]]:
    """Deterministically hold out ``round(num_episodes * val_fraction)`` episodes (at least one if
    val_fraction > 0). Returns sorted (train, val) episode indices."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    n_val = round(num_episodes * val_fraction)
    if val_fraction > 0:
        n_val = max(1, n_val)
    order = list(range(num_episodes))
    random.Random(seed).shuffle(order)
    val = sorted(order[:n_val])
    train = sorted(order[n_val:])
    return train, val


def images_to_float(batch: dict, camera_keys: list[str]) -> dict:
    """LeRobotDataset returns uint8 video frames when ``return_uint8=True``; policies expect [0, 1] floats."""
    for key in camera_keys:
        if key in batch and batch[key].dtype == torch.uint8:
            batch[key] = batch[key].to(torch.float32) / 255.0
    return batch


def make_dataset(repo_id: str, episodes: list[int] | None, policy_cfg, meta):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    return LeRobotDataset(repo_id, episodes=episodes, delta_timestamps=delta_timestamps, return_uint8=True)


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, drop_last: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def infinite(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader

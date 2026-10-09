"""Action-chunk metrics. Pure torch, no lerobot imports, so they are unit-testable on CPU."""

from __future__ import annotations

import torch


def masked_chunk_errors(
    pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Per-joint MSE and MAE over an action chunk, ignoring padded timesteps.

    Args:
        pred, target: (B, T, D) action chunks in the same units.
        is_pad: optional (B, T) bool mask, True where the target is padding past the episode end.
    Returns:
        dict with ``mse`` and ``mae`` of shape (D,), plus scalar ``mse_mean`` / ``mae_mean`` and
        ``mse_per_step`` of shape (T,) (nan where every sample is padded).
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
    err = pred.float() - target.float()
    if is_pad is None:
        valid = torch.ones(err.shape[:2], dtype=torch.bool, device=err.device)
    else:
        valid = ~is_pad.bool()
    w = valid.unsqueeze(-1).float()
    n_valid = w.sum(dim=(0, 1)).clamp_min(1.0)  # (D,)
    sq = (err**2) * w
    ab = err.abs() * w
    mse = sq.sum(dim=(0, 1)) / n_valid
    mae = ab.sum(dim=(0, 1)) / n_valid
    per_step_n = valid.float().sum(dim=0).clamp_min(1.0)  # (T,)
    mse_per_step = sq.mean(dim=-1).sum(dim=0) / per_step_n
    mse_per_step[valid.sum(dim=0) == 0] = float("nan")
    return {
        "mse": mse,
        "mae": mae,
        "mse_mean": mse.mean(),
        "mae_mean": mae.mean(),
        "mse_per_step": mse_per_step,
    }


class RunningMean:
    """Accumulates weighted means of tensors across batches."""

    def __init__(self):
        self.sum: dict[str, torch.Tensor] = {}
        self.n: dict[str, float] = {}

    def update(self, values: dict[str, torch.Tensor], weight: float) -> None:
        for k, v in values.items():
            v = v.detach().float().cpu()
            if k not in self.sum:
                self.sum[k] = torch.zeros_like(v)
                self.n[k] = 0.0
            if torch.isnan(v).any():
                mask = ~torch.isnan(v)
                self.sum[k][mask] += v[mask] * weight
            else:
                self.sum[k] += v * weight
            self.n[k] += weight

    def result(self) -> dict[str, torch.Tensor]:
        return {k: self.sum[k] / self.n[k] for k in self.sum}


def hold_baseline(state: torch.Tensor, chunk_len: int) -> torch.Tensor:
    """Predict that the arm holds its current position: repeat the (B, D) state over T steps."""
    return state.unsqueeze(1).expand(-1, chunk_len, -1)


def mean_action_baseline(mean: torch.Tensor, batch_size: int, chunk_len: int) -> torch.Tensor:
    """Predict the dataset-mean action everywhere. ``mean`` is (D,)."""
    return mean.view(1, 1, -1).expand(batch_size, chunk_len, -1)

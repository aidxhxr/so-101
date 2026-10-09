"""Score checkpoints on held-out episodes without touching the robot.

For every held-out frame the policy predicts a full 50-step action chunk from the two camera images,
the joint state, and the task string. Predictions are compared with the teleoperator's recorded
actions both in normalized units (what the loss sees) and in raw joint units (what the motors see),
against two baselines that need no model:

* ``hold``: keep the current joint position for the whole chunk,
* ``mean``: output the dataset-mean action.

Results are written as JSON with per-joint MAE, per-horizon MSE, and a per-checkpoint summary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from so101.data import images_to_float, make_dataset, make_loader, split_episodes
from so101.metrics import RunningMean, hold_baseline, masked_chunk_errors, mean_action_baseline
from so101.policy import load_checkpoint


def _stats_tensors(meta, key: str) -> tuple[torch.Tensor, torch.Tensor]:
    st = meta.stats[key]
    return torch.as_tensor(st["mean"]).float().flatten(), torch.as_tensor(st["std"]).float().flatten()


@torch.no_grad()
def score_checkpoint(
    checkpoint: str, repo_id: str, episodes: list[int], device: str, batch_size: int, num_workers: int, seed: int
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    policy, preprocessor, _ = load_checkpoint(checkpoint, device)
    norm_mode = str(policy.config.normalization_mapping.get("ACTION", ""))
    if "MEAN_STD" not in norm_mode:
        raise RuntimeError(f"expected MEAN_STD action normalization, got {norm_mode}")
    a_mean, a_std = _stats_tensors(meta, "action")
    s_mean, s_std = _stats_tensors(meta, "observation.state")
    ds = make_dataset(repo_id, episodes, policy.config, meta)
    loader = make_loader(ds, batch_size, num_workers, shuffle=False, drop_last=False)
    joint_names = meta.features["action"]["names"]
    dim = len(joint_names)

    torch.manual_seed(seed)
    agg = {name: RunningMean() for name in ("policy", "hold", "mean")}
    t0 = time.perf_counter()
    n_frames = 0
    for raw in loader:
        raw_action = raw["action"][..., :dim].clone()  # (B, T, D) raw units
        raw_state = raw["observation.state"][:, -1, :dim].clone()
        is_pad = raw.get("action_is_pad")
        batch = preprocessor(images_to_float(raw, meta.camera_keys))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pred_norm = policy.predict_action_chunk({k: v for k, v in batch.items() if k != "action"})
        pred_norm = pred_norm[..., :dim].float().cpu()
        pred_raw = pred_norm * a_std + a_mean
        target_norm = batch["action"][..., :dim].float().cpu()
        bsz, chunk = raw_action.shape[:2]
        preds = {
            "policy": (pred_norm, pred_raw),
            "hold": ((hold_baseline(raw_state, chunk) - a_mean) / a_std, hold_baseline(raw_state, chunk)),
            "mean": (torch.zeros_like(pred_norm), mean_action_baseline(a_mean, bsz, chunk)),
        }
        for name, (pn, pr) in preds.items():
            en = masked_chunk_errors(pn, target_norm, is_pad)
            er = masked_chunk_errors(pr, raw_action, is_pad)
            agg[name].update(
                {
                    "norm_mse": en["mse_mean"], "norm_mae": en["mae_mean"],
                    "raw_mse": er["mse_mean"], "raw_mae": er["mae_mean"],
                    "raw_mae_per_joint": er["mae"], "raw_mse_per_step": er["mse_per_step"],
                    "norm_mse_per_step": en["mse_per_step"],
                },
                weight=bsz,
            )
        n_frames += bsz
    out = {
        "checkpoint": checkpoint,
        "repo_id": repo_id,
        "episodes": episodes,
        "frames": n_frames,
        "chunk_size": int(policy.config.chunk_size),
        "joint_names": joint_names,
        "state_mean": s_mean.tolist(),
        "eval_s": round(time.perf_counter() - t0, 1),
    }
    for name, a in agg.items():
        r = a.result()
        out[name] = {
            "norm_mse": float(r["norm_mse"]), "norm_mae": float(r["norm_mae"]),
            "raw_mse": float(r["raw_mse"]), "raw_mae": float(r["raw_mae"]),
            "raw_mae_per_joint": dict(zip(joint_names, r["raw_mae_per_joint"].tolist(), strict=True)),
            "raw_mse_per_step": r["raw_mse_per_step"].tolist(),
            "norm_mse_per_step": r["norm_mse_per_step"].tolist(),
        }
    return out


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="checkpoint dirs or Hub ids to score")
    parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--episodes", type=str, default=None, help="comma-separated episode ids (default: held-out split)"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/eval_offline.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    args = build_parser().parse_args(argv)
    if args.episodes:
        episodes = [int(x) for x in args.episodes.split(",")]
    else:
        meta = LeRobotDatasetMetadata(args.repo_id)
        _, episodes = split_episodes(meta.total_episodes, args.val_fraction, args.split_seed)
    results = []
    for ckpt in args.checkpoints:
        res = score_checkpoint(ckpt, args.repo_id, episodes, args.device, args.batch_size, args.num_workers, args.seed)
        results.append(res)
        p, h, m = res["policy"], res["hold"], res["mean"]
        print(f"{ckpt}: norm_mse policy {p['norm_mse']:.4f} | hold {h['norm_mse']:.4f} | mean {m['norm_mse']:.4f}"
              f" || raw_mae policy {p['raw_mae']:.3f} | hold {h['raw_mae']:.3f} | mean {m['raw_mae']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

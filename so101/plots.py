"""Figures from the JSON/JSONL artifacts: training curves, horizon error, per-joint error, open-loop trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ema(values: list[float], alpha: float = 0.1) -> list[float]:
    out, m = [], None
    for v in values:
        m = v if m is None else alpha * v + (1 - alpha) * m
        out.append(m)
    return out


def plot_training(run_dir: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = _read_jsonl(run_dir / "train_log.jsonl")
    val = _read_jsonl(run_dir / "val_log.jsonl")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = [r["step"] for r in train]
    axes[0].plot(steps, [r["loss"] for r in train], alpha=0.25, label="train (raw)")
    axes[0].plot(steps, _ema([r["loss"] for r in train]), label="train (EMA)")
    axes[0].plot([r["step"] for r in val], [r["loss"] for r in val], "o-", label="held-out")
    axes[0].set_title("flow-matching loss")
    axes[0].set_xlabel("step")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].plot([r["step"] for r in val], [r["mse"] for r in val], "o-", color="C3")
    axes[1].set_title("held-out action-chunk MSE (normalized)")
    axes[1].set_xlabel("step")
    axes[1].set_yscale("log")
    axes[2].plot(steps, [r["lr"] for r in train])
    axes[2].set_title("learning rate")
    axes[2].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def plot_eval(eval_json: Path, out_dir: Path, label_map: dict[str, str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads(eval_json.read_text())
    label_map = label_map or {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for res in results:
        name = label_map.get(res["checkpoint"], Path(res["checkpoint"]).name)
        ax.plot(res["policy"]["raw_mse_per_step"], label=f"SmolVLA ({name})")
    ax.plot(results[0]["hold"]["raw_mse_per_step"], "--", color="gray", label="hold position")
    ax.plot(results[0]["mean"]["raw_mse_per_step"], ":", color="black", label="mean action")
    ax.set_xlabel("steps ahead in the predicted chunk")
    ax.set_ylabel("MSE (raw joint units)")
    ax.set_yscale("log")
    ax.set_title("prediction error vs horizon on held-out episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "eval_horizon.png", dpi=130)

    res = results[-1]
    joints = res["joint_names"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(joints))
    w = 0.27
    ax.bar([i - w for i in x], [res["hold"]["raw_mae_per_joint"][j] for j in joints], w, label="hold", color="gray")
    ax.bar(list(x), [res["mean"]["raw_mae_per_joint"][j] for j in joints], w, label="mean", color="black")
    ax.bar([i + w for i in x], [res["policy"]["raw_mae_per_joint"][j] for j in joints], w, label="SmolVLA")
    ax.set_xticks(list(x))
    ax.set_xticklabels([j.replace(".pos", "") for j in joints], rotation=20)
    ax.set_ylabel("MAE (raw joint units)")
    ax.set_title("per-joint error over the 50-step chunk, held-out episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "eval_per_joint.png", dpi=130)


def plot_trajectory(traj_json: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(traj_json.read_text())
    joints = d["joint_names"]
    fig, axes = plt.subplots(len(joints), 1, figsize=(10, 1.6 * len(joints)), sharex=True)
    t = [i / d["fps"] for i in range(len(d["target"]))]
    for j, ax in enumerate(axes):
        ax.plot(t, [row[j] for row in d["target"]], color="black", lw=1.2, label="teleoperator")
        for start, chunk in d["chunks"]:
            tt = [(start + i) / d["fps"] for i in range(len(chunk))]
            ax.plot(tt, [row[j] for row in chunk], color="C0", lw=0.9, alpha=0.8)
        ax.set_ylabel(joints[j].replace(".pos", ""), fontsize=8)
    axes[0].plot([], [], color="C0", label="SmolVLA open-loop chunks")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"held-out episode {d['episode']}: 50-step predictions issued every {d['stride']} frames")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None, help="training output dir with train_log.jsonl / val_log.jsonl")
    parser.add_argument("--eval-json", default=None)
    parser.add_argument("--traj-json", default=None)
    parser.add_argument("--out-dir", default="docs/figures")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        plot_training(Path(args.run_dir), out_dir / "training_curves.png")
    if args.eval_json:
        plot_eval(Path(args.eval_json), out_dir)
    if args.traj_json:
        plot_trajectory(Path(args.traj_json), out_dir / "trajectory.png")


if __name__ == "__main__":
    main()

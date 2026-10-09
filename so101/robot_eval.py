"""Closed-loop policy rollouts on the real arm with per-trial success logging.

Each trial runs the policy at the camera rate for up to ``episode_time_s`` seconds, then asks the
operator to mark success or failure. Trials are appended to a JSONL file so a run can be stopped
and continued, and ``summarize`` turns that file into a success rate with a Wilson interval.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from so101.config import DatasetConfig, EvalConfig, HardwareConfig, add_dataclass_args, from_args
from so101.hardware import make_follower
from so101.policy import load_checkpoint


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(trials_path: str | Path) -> dict:
    rows = [json.loads(line) for line in Path(trials_path).read_text().splitlines() if line.strip()]
    n = len(rows)
    k = sum(1 for r in rows if r.get("success"))
    lo, hi = wilson_interval(k, n)
    durations = [r["duration_s"] for r in rows if r.get("success") and "duration_s" in r]
    return {
        "trials": n,
        "successes": k,
        "success_rate": k / n if n else None,
        "wilson95": [lo, hi],
        "mean_success_duration_s": sum(durations) / len(durations) if durations else None,
        "checkpoints": sorted({r.get("checkpoint", "?") for r in rows}),
    }


def run_trials(hw: HardwareConfig, data: DatasetConfig, ev: EvalConfig, trials_path: Path) -> dict:
    from lerobot.common.control_utils import predict_action
    from lerobot.utils.robot_utils import precise_sleep

    device = torch.device(ev.device)
    policy, preprocessor, postprocessor = load_checkpoint(ev.checkpoint, ev.device)
    robot = make_follower(hw)
    robot.connect()
    period = 1.0 / hw.fps
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for trial in range(ev.episodes):
            policy.reset()
            input(f"trial {trial + 1}/{ev.episodes}: reset the scene, then press Enter")
            start = time.perf_counter()
            steps = 0
            while time.perf_counter() - start < ev.episode_time_s:
                t = time.perf_counter()
                obs = robot.get_observation()
                action = predict_action(
                    obs, policy, device, preprocessor, postprocessor,
                    use_amp=device.type == "cuda", task=data.task, robot_type=robot.robot_type,
                )
                robot.send_action({k: action[i].item() for i, k in enumerate(robot.action_features)})
                steps += 1
                precise_sleep(max(0.0, period - (time.perf_counter() - t)))
            duration = time.perf_counter() - start
            verdict = input("success? [y/N] ").strip().lower()
            row = {
                "trial": trial,
                "success": verdict in {"y", "yes"},
                "duration_s": round(duration, 2),
                "control_steps": steps,
                "checkpoint": ev.checkpoint,
                "task": data.task,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with trials_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
    finally:
        robot.disconnect()
    return summarize(trials_path)


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    add_dataclass_args(parser, HardwareConfig)
    add_dataclass_args(parser, DatasetConfig, prefix="data.")
    add_dataclass_args(parser, EvalConfig)
    parser.add_argument("--trials-path", default="results/robot_trials.jsonl")
    parser.add_argument("--summarize-only", action="store_true", help="only summarize an existing trials file")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.summarize_only:
        print(json.dumps(summarize(args.trials_path), indent=2))
        return
    result = run_trials(
        from_args(HardwareConfig, args),
        from_args(DatasetConfig, args, prefix="data."),
        from_args(EvalConfig, args),
        Path(args.trials_path),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

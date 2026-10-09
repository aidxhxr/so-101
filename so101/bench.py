"""Inference latency of a checkpoint: one observation in, one 50-step action chunk out.

Measures the full path the robot loop takes (preprocessor + flow-matching denoising + postprocessor)
at batch size 1, so the numbers map directly onto control-rate headroom.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from so101.data import images_to_float, make_dataset
from so101.policy import gpu_name, load_checkpoint


def bench(checkpoint: str, repo_id: str, device: str, iters: int, warmup: int, amp: bool) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    policy, preprocessor, postprocessor = load_checkpoint(checkpoint, device)
    meta = LeRobotDatasetMetadata(repo_id)
    ds = make_dataset(repo_id, [0], policy.config, meta)
    sample = ds[0]
    obs = {k: v.unsqueeze(0) for k, v in sample.items() if hasattr(v, "unsqueeze") and k != "action"}
    obs["task"] = [sample["task"]]
    obs = images_to_float(obs, meta.camera_keys)

    def once() -> float:
        policy.reset()
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            batch = preprocessor(dict(obs))
            chunk = policy.predict_action_chunk(batch)
            postprocessor(chunk[:, 0])
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t

    for _ in range(warmup):
        once()
    times = [once() * 1000 for _ in range(iters)]
    times.sort()
    chunk_len = policy.config.chunk_size
    p50 = statistics.median(times)
    return {
        "checkpoint": checkpoint,
        "device": gpu_name() if device == "cuda" else device,
        "autocast_bf16": amp,
        "iters": iters,
        "chunk_size": chunk_len,
        "n_action_steps": policy.config.n_action_steps,
        "denoise_steps": policy.config.num_steps,
        "latency_ms_p50": round(p50, 2),
        "latency_ms_p95": round(times[int(0.95 * len(times)) - 1], 2),
        "latency_ms_max": round(times[-1], 2),
        "chunks_per_s": round(1000 / p50, 2),
        "actions_per_s_if_full_chunk_executed": round(1000 / p50 * policy.config.n_action_steps, 1),
    }


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/smolvla_so101/final")
    parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--out", default=None, help="write JSON here")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    res = bench(args.checkpoint, args.repo_id, args.device, args.iters, args.warmup, not args.no_amp)
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

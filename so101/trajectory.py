"""Dump open-loop predictions for one held-out episode: a 50-step chunk every ``stride`` frames, in raw joint units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from so101.data import images_to_float, make_dataset
from so101.policy import load_for_eval


@torch.no_grad()
def dump_trajectory(checkpoint: str, repo_id: str, episode: int, stride: int, device: str, seed: int) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    policy, preprocessor, _ = load_for_eval(checkpoint, meta, device)
    a_mean = torch.as_tensor(meta.stats["action"]["mean"]).float().flatten()
    a_std = torch.as_tensor(meta.stats["action"]["std"]).float().flatten()
    ds = make_dataset(repo_id, [episode], policy.config, meta)
    dim = len(meta.features["action"]["names"])
    torch.manual_seed(seed)
    target = [ds[i]["action"][0, :dim].tolist() for i in range(len(ds))]
    chunks = []
    for start in range(0, len(ds), stride):
        sample = ds[start]
        obs = {k: v.unsqueeze(0) for k, v in sample.items() if hasattr(v, "unsqueeze") and k != "action"}
        obs["task"] = [sample["task"]]
        batch = preprocessor(images_to_float(obs, meta.camera_keys))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = policy.predict_action_chunk(batch)[0, :, :dim].float().cpu()
        pred = pred * a_std + a_mean
        n = min(len(pred), len(ds) - start)
        chunks.append([start, pred[:n].tolist()])
    return {
        "checkpoint": checkpoint,
        "repo_id": repo_id,
        "episode": episode,
        "fps": meta.fps,
        "stride": stride,
        "joint_names": meta.features["action"]["names"],
        "target": target,
        "chunks": chunks,
    }


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/smolvla_so101/final")
    parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/trajectory.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    d = dump_trajectory(args.checkpoint, args.repo_id, args.episode, args.stride, args.device, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(d))
    print(f"wrote {args.out}: {len(d['target'])} frames, {len(d['chunks'])} chunks")


if __name__ == "__main__":
    main()

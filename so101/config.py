"""Dataclass configs with .env / environment overrides.

Secrets (HF_TOKEN) and machine-specific values (serial ports, HF user) come from
a .env file next to the repo or from the environment; everything else has a
default here and can be overridden from the CLI.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field, fields
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class HardwareConfig:
    follower_port: str = field(default_factory=lambda: env("SO101_FOLLOWER_PORT", "/dev/ttyACM0"))
    leader_port: str = field(default_factory=lambda: env("SO101_LEADER_PORT", "/dev/ttyACM1"))
    follower_id: str = "so101_follower"
    leader_id: str = "so101_leader"
    fps: int = 30
    camera_width: int = 640
    camera_height: int = 480
    front_camera: int = 0
    wrist_camera: int = 2

    def cameras(self) -> dict[str, dict[str, Any]]:
        common = {"width": self.camera_width, "height": self.camera_height, "fps": self.fps}
        return {
            "front": {"index_or_path": self.front_camera, **common},
            "wrist": {"index_or_path": self.wrist_camera, **common},
        }


@dataclass
class DatasetConfig:
    repo_id: str = field(default_factory=lambda: f"{env('HF_USER', 'amirkhanaidarkhan')}/so101_pick_place")
    task: str = "Pick up the cube and place it in the box."
    num_episodes: int = 50
    episode_time_s: float = 30.0
    reset_time_s: float = 10.0
    fps: int = 30
    val_fraction: float = 0.1
    split_seed: int = 0
    max_train_episodes: int | None = None
    push_to_hub: bool = True


@dataclass
class TrainConfig:
    pretrained: str = "lerobot/smolvla_base"
    output_dir: str = "outputs/smolvla_so101"
    steps: int = 20_000
    batch_size: int = 64
    lr: float = 1e-4
    min_lr: float = 2.5e-6
    warmup_steps: int = 1_000
    grad_clip: float = 10.0
    save_every: int = 2_000
    eval_every: int = 1_000
    log_every: int = 50
    num_workers: int = 12
    seed: int = 1000
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    device: str = "cuda"
    push_to_hub: str | None = None
    resume: str | None = None


@dataclass
class EvalConfig:
    checkpoint: str = "outputs/smolvla_so101/final"
    batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda"
    seed: int = 0
    episodes: int = 10
    episode_time_s: float = 30.0
    results_path: str | None = None


def add_dataclass_args(parser: argparse.ArgumentParser, cls: type, prefix: str = "") -> None:
    """Expose every field of a dataclass as ``--field-name`` with its default."""
    for f in fields(cls):
        name = f"--{prefix}{f.name.replace('_', '-')}"
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()  # type: ignore[misc]
        if f.type in ("bool", bool):
            parser.add_argument(name, type=_str2bool, default=default, metavar="{true,false}")
        elif f.type in ("int", int, "int | None"):
            parser.add_argument(name, type=int, default=default)
        elif f.type in ("float", float):
            parser.add_argument(name, type=float, default=default)
        else:
            parser.add_argument(name, type=str, default=default)


def from_args(cls: type, args: argparse.Namespace, prefix: str = ""):
    kwargs = {f.name: getattr(args, f"{prefix}{f.name}") for f in fields(cls)}
    return cls(**kwargs)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in {"1", "true", "yes", "y"}:
        return True
    if v.lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def asdict(cfg) -> dict[str, Any]:
    return dataclasses.asdict(cfg)

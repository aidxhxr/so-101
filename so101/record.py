"""Record a teleoperated LeRobot dataset (joint states, actions, both cameras) and optionally push it to the Hub."""

from __future__ import annotations

import argparse

from so101.config import DatasetConfig, HardwareConfig, add_dataclass_args, from_args
from so101.hardware import follower_config, leader_config


def record_episodes(hw: HardwareConfig, data: DatasetConfig, resume: bool = False, display: bool = True):
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.scripts.lerobot_record import RecordConfig, record

    cfg = RecordConfig(
        robot=follower_config(hw),
        teleop=leader_config(hw),
        dataset=DatasetRecordConfig(
            repo_id=data.repo_id,
            single_task=data.task,
            fps=data.fps,
            episode_time_s=data.episode_time_s,
            reset_time_s=data.reset_time_s,
            num_episodes=data.num_episodes,
            push_to_hub=data.push_to_hub,
        ),
        display_data=display,
        resume=resume,
    )
    return record(cfg)


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    add_dataclass_args(parser, HardwareConfig)
    add_dataclass_args(parser, DatasetConfig, prefix="data.")
    parser.add_argument("--resume", action="store_true", help="append episodes to an existing dataset")
    parser.add_argument("--no-display", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    record_episodes(
        from_args(HardwareConfig, args),
        from_args(DatasetConfig, args, prefix="data."),
        resume=args.resume,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()

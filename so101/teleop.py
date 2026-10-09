"""Mirror the leader arm onto the follower at a fixed rate; use it to check calibration and latency."""

from __future__ import annotations

import argparse
import time

from so101.config import HardwareConfig, add_dataclass_args, from_args
from so101.hardware import make_follower, make_leader


def teleoperate(hw: HardwareConfig, seconds: float | None = None) -> dict[str, float]:
    from lerobot.utils.robot_utils import precise_sleep

    follower = make_follower(hw, with_cameras=False)
    leader = make_leader(hw)
    follower.connect()
    leader.connect()
    period = 1.0 / hw.fps
    n, late = 0, 0
    start = time.perf_counter()
    try:
        while seconds is None or time.perf_counter() - start < seconds:
            t = time.perf_counter()
            follower.send_action(leader.get_action())
            dt = time.perf_counter() - t
            late += dt > period
            n += 1
            precise_sleep(max(0.0, period - dt))
    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()
        leader.disconnect()
    return {"loops": n, "late_fraction": late / max(1, n)}


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    add_dataclass_args(parser, HardwareConfig)
    parser.add_argument("--seconds", type=float, default=None, help="stop after this long (default: Ctrl-C)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(teleoperate(from_args(HardwareConfig, args), args.seconds))


if __name__ == "__main__":
    main()

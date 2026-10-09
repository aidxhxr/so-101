"""SO-101 follower/leader construction (lerobot >= 0.6 module layout).

Imports are deferred so the rest of the package (training, offline eval, tests) works on a
machine without the feetech extra or a serial port.
"""

from __future__ import annotations

from so101.config import HardwareConfig


def camera_configs(hw: HardwareConfig) -> dict:
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    return {name: OpenCVCameraConfig(**kw) for name, kw in hw.cameras().items()}


def follower_config(hw: HardwareConfig, with_cameras: bool = True):
    from lerobot.robots.so_follower import SO101FollowerConfig

    return SO101FollowerConfig(
        port=hw.follower_port,
        id=hw.follower_id,
        cameras=camera_configs(hw) if with_cameras else {},
    )


def leader_config(hw: HardwareConfig):
    from lerobot.teleoperators.so_leader import SO101LeaderConfig

    return SO101LeaderConfig(port=hw.leader_port, id=hw.leader_id)


def make_follower(hw: HardwareConfig, with_cameras: bool = True):
    from lerobot.robots.so_follower import SO101Follower

    return SO101Follower(follower_config(hw, with_cameras))


def make_leader(hw: HardwareConfig):
    from lerobot.teleoperators.so_leader import SO101Leader

    return SO101Leader(leader_config(hw))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

import config


def camera_configs():
    return {name: OpenCVCameraConfig(**kw) for name, kw in config.CAMERAS.items()}


def follower_config(with_cameras=True):
    return SO101FollowerConfig(
        port=config.FOLLOWER_PORT,
        id=config.FOLLOWER_ID,
        cameras=camera_configs() if with_cameras else {},
    )


def leader_config():
    return SO101LeaderConfig(port=config.LEADER_PORT, id=config.LEADER_ID)


def make_follower(with_cameras=True):
    return SO101Follower(follower_config(with_cameras))


def make_leader():
    return SO101Leader(leader_config())

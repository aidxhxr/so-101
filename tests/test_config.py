import argparse

from so101.config import DatasetConfig, HardwareConfig, TrainConfig, add_dataclass_args, from_args


def test_cli_roundtrip_with_overrides():
    p = argparse.ArgumentParser()
    add_dataclass_args(p, TrainConfig)
    add_dataclass_args(p, DatasetConfig, prefix="data.")
    args = p.parse_args(["--steps", "5", "--freeze-vision-encoder", "false", "--data.val-fraction", "0.2"])
    cfg = from_args(TrainConfig, args)
    data = from_args(DatasetConfig, args, prefix="data.")
    assert cfg.steps == 5 and cfg.freeze_vision_encoder is False and cfg.batch_size == 64
    assert data.val_fraction == 0.2


def test_hardware_cameras_use_shared_resolution():
    hw = HardwareConfig(front_camera=4, wrist_camera=6)
    cams = hw.cameras()
    assert cams["front"]["index_or_path"] == 4 and cams["wrist"]["index_or_path"] == 6
    assert cams["front"]["width"] == cams["wrist"]["width"] == 640


def test_env_overrides_ports(monkeypatch):
    monkeypatch.setenv("SO101_FOLLOWER_PORT", "/dev/ttyUSB9")
    assert HardwareConfig().follower_port == "/dev/ttyUSB9"

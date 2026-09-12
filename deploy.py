import argparse
import time

import torch

from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait

import config
from robot import make_follower


def load_policy(checkpoint, device=config.DEVICE):
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def run_policy(checkpoint, task=config.TASK, episodes=5, episode_time_s=config.EPISODE_TIME_S, fps=config.FPS):
    device = torch.device(config.DEVICE)
    policy, preprocessor, postprocessor = load_policy(checkpoint, config.DEVICE)
    robot = make_follower()
    robot.connect()
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")

    try:
        for ep in range(episodes):
            policy.reset()
            print(f"episode {ep + 1}/{episodes}")
            start = time.perf_counter()
            while time.perf_counter() - start < episode_time_s:
                t = time.perf_counter()
                obs = robot.get_observation()
                frame = build_dataset_frame(obs_features, obs, prefix="observation")
                values = predict_action(
                    frame,
                    policy,
                    device,
                    preprocessor,
                    postprocessor,
                    use_amp=True,
                    task=task,
                    robot_type=robot.robot_type,
                )
                action = {k: values[i].item() for i, k in enumerate(robot.action_features)}
                robot.send_action(action)
                busy_wait(1 / fps - (time.perf_counter() - t))
            input("reset the scene and press enter")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=f"{config.OUTPUT_DIR}/final")
    p.add_argument("--task", default=config.TASK)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--episode-time", type=float, default=config.EPISODE_TIME_S)
    a = p.parse_args()
    run_policy(a.checkpoint, a.task, a.episodes, a.episode_time)

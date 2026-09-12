import argparse

from lerobot.record import DatasetRecordConfig, RecordConfig, record

import config
from robot import follower_config, leader_config


def record_episodes(repo_id, num_episodes, task, push_to_hub=True, resume=False):
    cfg = RecordConfig(
        robot=follower_config(),
        teleop=leader_config(),
        dataset=DatasetRecordConfig(
            repo_id=repo_id,
            single_task=task,
            fps=config.FPS,
            episode_time_s=config.EPISODE_TIME_S,
            reset_time_s=config.RESET_TIME_S,
            num_episodes=num_episodes,
            push_to_hub=push_to_hub,
        ),
        display_data=True,
        resume=resume,
    )
    return record(cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=config.DATASET_REPO_ID)
    p.add_argument("--episodes", type=int, default=config.NUM_EPISODES)
    p.add_argument("--task", default=config.TASK)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    record_episodes(a.repo_id, a.episodes, a.task, not a.no_push, a.resume)

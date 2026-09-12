import os

FOLLOWER_PORT = os.environ.get("SO101_FOLLOWER_PORT", "/dev/ttyACM0")
LEADER_PORT = os.environ.get("SO101_LEADER_PORT", "/dev/ttyACM1")
FOLLOWER_ID = "so101_follower"
LEADER_ID = "so101_leader"

CAMERAS = {
    "front": {"index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    "wrist": {"index_or_path": 2, "width": 640, "height": 480, "fps": 30},
}

HF_USER = os.environ.get("HF_USER", "amirkhanaidarkhan")
DATASET_REPO_ID = f"{HF_USER}/so101_pick_place"
TASK = "Pick up the cube and place it in the box."
FPS = 30
NUM_EPISODES = 50
EPISODE_TIME_S = 30
RESET_TIME_S = 10

PRETRAINED = "lerobot/smolvla_base"
OUTPUT_DIR = "outputs/smolvla_so101"
DEVICE = "cuda"

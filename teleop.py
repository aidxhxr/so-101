import time

from lerobot.utils.robot_utils import busy_wait

import config
from robot import make_follower, make_leader


def teleoperate(fps=config.FPS):
    follower = make_follower(with_cameras=False)
    leader = make_leader()
    follower.connect()
    leader.connect()
    try:
        while True:
            t = time.perf_counter()
            follower.send_action(leader.get_action())
            busy_wait(1 / fps - (time.perf_counter() - t))
    except KeyboardInterrupt:
        pass
    finally:
        follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    teleoperate()

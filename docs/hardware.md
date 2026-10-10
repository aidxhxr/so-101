# Hardware setup

This covers the physical side of the pipeline: wiring, calibration, cameras, and the two
things that most often silently break a deployed policy (joint units and camera order).

## Bill of materials

| Part | Notes |
|---|---|
| SO-101 follower arm | 6 Feetech STS3215 servos, 12 V supply |
| SO-101 leader arm | same kinematics, 7.4 V servos, no payload |
| 2 USB cameras | one fixed "front" view of the workspace, one on the wrist; 640x480 at 30 fps is enough |
| USB serial adapters | one per arm; they enumerate as `/dev/ttyACM*` on Linux |

## Install

```bash
python -m pip install -e ".[robot,plots,dev]"   # adds lerobot[feetech]
cp .env.example .env                           # then set the ports and HF_TOKEN
```

Find the serial ports with `lerobot-find-port` (unplug and replug one arm at a time) and
write them into `.env` as `SO101_FOLLOWER_PORT` and `SO101_LEADER_PORT`. Every `so101`
command also accepts `--follower-port` and `--leader-port` on the command line.

## Calibration

Both arms need a one-time calibration that maps raw servo ticks to joint positions:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=$SO101_FOLLOWER_PORT --robot.id=so101_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=$SO101_LEADER_PORT  --teleop.id=so101_leader
```

The ids must match `HardwareConfig.follower_id` / `leader_id` (defaults `so101_follower` and
`so101_leader`); lerobot stores the calibration under `~/.cache/huggingface/lerobot/calibration/`.

## Joint units: read this before recording

lerobot 0.6 changed the SO-101 default to **degrees** (`use_degrees=True`). Older releases, and
public datasets recorded with them such as `lerobot/svla_so101_pickplace`, use a **−100…100
range** per joint (gripper 0…100). The policy learns whatever units it was trained on, so:

- Recording your own data: keep the default `use_degrees=True` everywhere (record, train, deploy).
- Deploying a checkpoint trained on the public dataset: pass `--use-degrees false` to
  `so101 deploy`, otherwise the arm will interpret a "45" as 45° instead of 45% of range.

`HardwareConfig.use_degrees` is the single switch; it is passed to both the follower and the
leader config.

## Cameras

```bash
lerobot-find-cameras opencv
```

The front camera index goes in `--front-camera`, the wrist camera in `--wrist-camera`
(defaults 0 and 2; index 1 is often the laptop webcam). Camera names in the dataset are
`observation.images.front` and `observation.images.wrist`. The policy's input keys are fixed at
training time, so deploy with the same two names and the same resolution you recorded with.

## Check the loop before recording

```bash
so101 teleop --seconds 30
# {'loops': 900, 'late_fraction': 0.0}
```

`late_fraction` is the share of control ticks that took longer than one period (33.3 ms at
30 fps). It should be 0 on a wired USB connection; a non-zero value usually means a camera is
sharing a USB hub with a servo bus.

## Recording

```bash
so101 record --data.num-episodes 50 --data.episode-time-s 30 --data.reset-time-s 10
so101 record --resume           # continue an interrupted session
```

Recording goes through `lerobot_record`, so the usual keyboard controls apply (right arrow ends
an episode early, left arrow re-records it, Escape stops). With `push_to_hub=true` (the default)
the dataset is uploaded to `HF_USER/so101_pick_place` using `HF_TOKEN` from `.env`.

## Closed-loop trials

```bash
so101 deploy --checkpoint outputs/main/final --episodes 20 --episode-time-s 30
so101 deploy --summarize-only   # success rate with a 95% Wilson interval
```

Each trial waits for Enter, runs the policy at the camera rate for `episode_time_s`, then asks
for a y/N verdict and appends one JSON line to `results/robot_trials.jsonl`, so a session can be
stopped and continued and the file is the audit trail for any success rate you quote.

# SO-101 × SmolVLA

End-to-end imitation learning on a real robot arm: teleoperated data collection, fine-tuning a vision-language-action model on a single GPU, and closed-loop deployment at 30 Hz.

## What this is

- **Robot:** SO-101 6-DoF arm (leader/follower pair, Feetech servos) with front and wrist cameras
- **Model:** [SmolVLA](https://huggingface.co/lerobot/smolvla_base), a 450M-parameter vision-language-action policy, fine-tuned from the pretrained checkpoint
- **Compute:** single NVIDIA RTX PRO 6000 (96 GB), bf16 mixed precision, batch size 128
- **Framework:** PyTorch + Hugging Face LeRobot

## Pipeline

```
teleop.py  ->  record.py  ->  finetune.py  ->  deploy.py
 verify        LeRobot        SmolVLA          real-time
 hardware      dataset        fine-tune        inference
```

1. **Teleoperation** mirrors the leader arm onto the follower to validate calibration and latency.
2. **Recording** logs synchronized joint states, actions, and dual camera streams into a LeRobot dataset and pushes it to the Hub.
3. **Fine-tuning** loads `smolvla_base`, freezes the vision encoder, trains the action expert with AdamW and a cosine schedule with warmup, and checkpoints policy plus pre/post-processors.
4. **Deployment** runs the trained policy on the arm from a language instruction, with action chunking and normalization handled by the saved processors.

## Quick start

```bash
pip install -r requirements.txt

python teleop.py
python record.py --episodes 50
python finetune.py --steps 20000 --batch-size 128
python deploy.py --checkpoint outputs/smolvla_so101/final
```

Ports, camera indices, dataset name, and task prompt live in `config.py`. The fine-tuning entrypoint is also importable:

```python
from finetune import finetune_smolvla

finetune_smolvla(repo_id="user/so101_pick_place", steps=20000, batch_size=128)
```

## Files

| File | Purpose |
|---|---|
| `config.py` | Hardware ports, cameras, dataset and training defaults |
| `robot.py` | SO-101 follower and leader construction |
| `teleop.py` | Leader-to-follower teleoperation loop |
| `record.py` | Dataset recording with Hub upload |
| `finetune.py` | SmolVLA fine-tuning function and CLI |
| `deploy.py` | Real-time policy inference on the arm |

## Skills demonstrated

Robot hardware integration, dataset engineering for imitation learning, fine-tuning multimodal transformer policies, GPU training optimization, and real-time control loops.

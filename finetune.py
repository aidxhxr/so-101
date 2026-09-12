import argparse
import json
import math
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi
from torch.utils.data import DataLoader

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

import config


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def cosine_with_warmup(warmup_steps, total_steps, floor=0.1):
    def f(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


def save_checkpoint(policy, preprocessor, postprocessor, path):
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)


def finetune_smolvla(
    repo_id=config.DATASET_REPO_ID,
    output_dir=config.OUTPUT_DIR,
    pretrained=config.PRETRAINED,
    steps=20000,
    batch_size=128,
    lr=1e-4,
    warmup_steps=1000,
    grad_clip=10.0,
    save_every=2000,
    log_every=50,
    num_workers=16,
    device=config.DEVICE,
    freeze_vision_encoder=True,
    train_expert_only=True,
    seed=1000,
    push_to_hub=None,
):
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = LeRobotDatasetMetadata(repo_id)
    policy_cfg = SmolVLAConfig(
        device=device,
        freeze_vision_encoder=freeze_vision_encoder,
        train_expert_only=train_expert_only,
    )
    policy_cfg.pretrained_path = pretrained

    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg, dataset_stats=dataset.meta.stats
    )
    policy.train()

    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_with_warmup(warmup_steps, steps)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    batches = cycle(loader)

    log_file = (out / "train_log.jsonl").open("a")
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in policy.parameters())
    print(f"dataset={repo_id} episodes={meta.total_episodes} frames={meta.total_frames}")
    print(f"params trainable={trainable/1e6:.1f}M total={total/1e6:.1f}M")

    t0 = time.time()
    for step in range(1, steps + 1):
        batch = preprocessor(next(batches))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % log_every == 0:
            rec = {
                "step": step,
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - t0, 1),
            }
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            print(
                f"step {step}/{steps} loss {rec['loss']:.4f} "
                f"grad {rec['grad_norm']:.2f} lr {rec['lr']:.2e}"
            )

        if step % save_every == 0 or step == steps:
            save_checkpoint(policy, preprocessor, postprocessor, out / f"step_{step:06d}")

    final = out / "final"
    save_checkpoint(policy, preprocessor, postprocessor, final)
    log_file.close()

    if push_to_hub:
        api = HfApi()
        api.create_repo(push_to_hub, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(final), repo_id=push_to_hub, repo_type="model")

    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=config.DATASET_REPO_ID)
    p.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p.add_argument("--pretrained", default=config.PRETRAINED)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--unfreeze-vision", action="store_true")
    p.add_argument("--train-full-vlm", action="store_true")
    p.add_argument("--push-to-hub", default=None)
    a = p.parse_args()
    finetune_smolvla(
        repo_id=a.repo_id,
        output_dir=a.output_dir,
        pretrained=a.pretrained,
        steps=a.steps,
        batch_size=a.batch_size,
        lr=a.lr,
        warmup_steps=a.warmup_steps,
        save_every=a.save_every,
        num_workers=a.num_workers,
        freeze_vision_encoder=not a.unfreeze_vision,
        train_expert_only=not a.train_full_vlm,
        push_to_hub=a.push_to_hub,
    )

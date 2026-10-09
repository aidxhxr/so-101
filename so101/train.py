"""Fine-tune SmolVLA on a LeRobot dataset with a held-out validation split.

Logs train loss, held-out flow-matching loss, and held-out action-chunk MSE to JSONL,
checkpoints policy + processors, keeps the best checkpoint by validation MSE, and can resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import time
from pathlib import Path

import torch

from so101 import __version__
from so101.config import DatasetConfig, TrainConfig, add_dataclass_args, asdict, from_args
from so101.data import images_to_float, infinite, make_dataset, make_loader, split_episodes
from so101.metrics import RunningMean, masked_chunk_errors
from so101.policy import build_from_pretrained, gpu_name, param_counts, save_checkpoint
from so101.schedule import cosine_with_warmup

log = logging.getLogger("so101.train")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


@torch.no_grad()
def evaluate(policy, preprocessor, loader, camera_keys, device: str, seed: int) -> dict[str, float]:
    """Held-out flow-matching loss (fixed noise seed, so runs are comparable) and
    action-chunk MSE/MAE in normalized units from full denoised predictions."""
    policy.eval()
    gen_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    agg = RunningMean()
    for batch in loader:
        batch = preprocessor(images_to_float(batch, camera_keys))
        bsz = batch["action"].shape[0]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            loss, _ = policy.forward(batch)
            pred = policy.predict_action_chunk({k: v for k, v in batch.items() if k != "action"})
        dim = batch["action"].shape[-1]
        errs = masked_chunk_errors(pred[..., :dim], batch["action"], batch.get("action_is_pad"))
        agg.update({"loss": loss, "mse": errs["mse_mean"], "mae": errs["mae_mean"]}, weight=bsz)
    torch.random.set_rng_state(gen_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    policy.train()
    return {k: float(v) for k, v in agg.result().items()}


def train(cfg: TrainConfig, data: DatasetConfig) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = LeRobotDatasetMetadata(data.repo_id)
    train_eps, val_eps = split_episodes(meta.total_episodes, data.val_fraction, data.split_seed)
    policy, preprocessor, postprocessor = build_from_pretrained(
        cfg.pretrained, meta, cfg.device, cfg.freeze_vision_encoder, cfg.train_expert_only
    )
    policy.train()
    train_ds = make_dataset(data.repo_id, train_eps, policy.config, meta)
    val_ds = make_dataset(data.repo_id, val_eps, policy.config, meta) if val_eps else None
    camera_keys = meta.camera_keys

    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_with_warmup(cfg.warmup_steps, cfg.steps, cfg.lr, cfg.min_lr)
    )
    start_step = 0
    if cfg.resume:
        state = torch.load(Path(cfg.resume) / "training_state.pt", map_location="cpu")
        policy.load_state_dict(torch.load(Path(cfg.resume) / "policy_state.pt", map_location="cpu"))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        log.info("resumed from %s at step %d", cfg.resume, start_step)

    train_loader = make_loader(train_ds, cfg.batch_size, cfg.num_workers, shuffle=True, drop_last=True)
    val_loader = (
        make_loader(val_ds, cfg.batch_size, max(1, cfg.num_workers // 2), shuffle=False, drop_last=False)
        if val_ds
        else None
    )
    batches = infinite(train_loader)

    counts = param_counts(policy)
    run_meta = {
        "so101_version": __version__,
        "git_sha": git_sha(),
        "torch": torch.__version__,
        "gpu": gpu_name(),
        "dataset": {
            "repo_id": data.repo_id,
            "episodes": meta.total_episodes,
            "frames": meta.total_frames,
            "fps": meta.fps,
            "train_episodes": train_eps,
            "val_episodes": val_eps,
            "train_frames": len(train_ds),
            "val_frames": len(val_ds) if val_ds else 0,
            "camera_keys": camera_keys,
        },
        "params": counts,
        "train": asdict(cfg),
        "data": asdict(data),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    log.info(
        "dataset=%s train_eps=%d val_eps=%d train_frames=%d | params trainable=%.1fM total=%.1fM | %s",
        data.repo_id, len(train_eps), len(val_eps), len(train_ds),
        counts["trainable"] / 1e6, counts["total"] / 1e6, run_meta["gpu"],
    )

    train_log = (out / "train_log.jsonl").open("a")
    val_log = (out / "val_log.jsonl").open("a")
    best_mse = math.inf
    best_path = out / "best"
    if (out / "best_mse.txt").exists():
        best_mse = float((out / "best_mse.txt").read_text())

    def run_eval(step: int) -> None:
        nonlocal best_mse
        if val_loader is None:
            return
        t = time.perf_counter()
        res = evaluate(policy, preprocessor, val_loader, camera_keys, cfg.device, cfg.seed)
        res.update({"step": step, "eval_s": round(time.perf_counter() - t, 1)})
        val_log.write(json.dumps(res) + "\n")
        val_log.flush()
        log.info(
            "val step %d loss %.4f mse %.5f mae %.4f (%.0fs)",
            step, res["loss"], res["mse"], res["mae"], res["eval_s"],
        )
        if res["mse"] < best_mse:
            best_mse = res["mse"]
            save_checkpoint(policy, preprocessor, postprocessor, best_path)
            (out / "best_mse.txt").write_text(f"{best_mse:.8f}\n")
            (best_path / "step.txt").write_text(f"{step}\n")

    if start_step == 0:
        run_eval(0)

    t0 = time.perf_counter()
    t_data = 0.0
    t_eval = 0.0
    for step in range(start_step + 1, cfg.steps + 1):
        td = time.perf_counter()
        batch = preprocessor(images_to_float(next(batches), camera_keys))
        t_data += time.perf_counter() - td
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.device == "cuda"):
            loss, _ = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            elapsed = time.perf_counter() - t0 - t_eval
            rec = {
                "step": step,
                "loss": round(loss.item(), 6),
                "grad_norm": round(grad_norm.item(), 4),
                "lr": scheduler.get_last_lr()[0],
                "step_s": round(elapsed / (step - start_step), 4),
                "data_s": round(t_data / (step - start_step), 4),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cfg.device == "cuda" else 0,
                "elapsed_s": round(elapsed, 1),
            }
            train_log.write(json.dumps(rec) + "\n")
            train_log.flush()
            log.info(
                "step %d/%d loss %.4f grad %.2f lr %.2e %.3fs/step",
                step, cfg.steps, rec["loss"], rec["grad_norm"], rec["lr"], rec["step_s"],
            )
        if step % cfg.eval_every == 0:
            te = time.perf_counter()
            run_eval(step)
            t_eval += time.perf_counter() - te
        if step % cfg.save_every == 0 or step == cfg.steps:
            ckpt = save_checkpoint(policy, preprocessor, postprocessor, out / f"step_{step:06d}")
            torch.save(policy.state_dict(), ckpt / "policy_state.pt")
            torch.save(
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step},
                ckpt / "training_state.pt",
            )

    final = save_checkpoint(policy, preprocessor, postprocessor, out / "final")
    train_log.close()
    val_log.close()
    run_meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_meta["best_val_mse"] = best_mse if math.isfinite(best_mse) else None
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    if cfg.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(cfg.push_to_hub, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(final), repo_id=cfg.push_to_hub, repo_type="model")
        log.info("pushed %s to %s", final, cfg.push_to_hub)
    return final


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    add_dataclass_args(parser, TrainConfig)
    add_dataclass_args(parser, DatasetConfig, prefix="data.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    train(from_args(TrainConfig, args), from_args(DatasetConfig, args, prefix="data."))


if __name__ == "__main__":
    main()

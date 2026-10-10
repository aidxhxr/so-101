"""Build, save, and load SmolVLA policies together with their processor pipelines."""

from __future__ import annotations

from pathlib import Path

import torch


def build_from_pretrained(pretrained: str, ds_meta, device: str, freeze_vision_encoder: bool, train_expert_only: bool):
    """Instantiate SmolVLA from a pretrained checkpoint with features inferred from the dataset.

    Normalization statistics come from the dataset, so a base checkpoint trained on other robots
    is re-normalized for this arm's joint ranges.
    """
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    cfg = SmolVLAConfig(
        device=device,
        freeze_vision_encoder=freeze_vision_encoder,
        train_expert_only=train_expert_only,
    )
    cfg.pretrained_path = pretrained
    policy = make_policy(cfg, ds_meta=ds_meta)
    features = {**policy.config.input_features, **policy.config.output_features}
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=pretrained,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "features": features,
                "norm_map": policy.config.normalization_mapping,
                "stats": ds_meta.stats,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": ds_meta.stats,
            },
        },
    )
    return policy, preprocessor, postprocessor


def load_checkpoint(path: str | Path, device: str):
    """Load a policy saved by :func:`save_checkpoint` (weights + processor stats)."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    path = str(path)
    policy = SmolVLAPolicy.from_pretrained(path)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def save_checkpoint(policy, preprocessor, postprocessor, path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    return path


def load_for_eval(checkpoint: str | Path, ds_meta, device: str):
    """Load a fine-tuned checkpoint directory, or build an un-fine-tuned Hub model (e.g. ``lerobot/smolvla_base``)
    normalized with this dataset's statistics so zero-shot and fine-tuned policies are scored identically."""
    if Path(checkpoint).is_dir():
        return load_checkpoint(checkpoint, device)
    policy, pre, post = build_from_pretrained(str(checkpoint), ds_meta, device, True, True)
    policy.eval()
    return policy, pre, post


def param_counts(policy) -> dict[str, int]:
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    return {"trainable": trainable, "total": total}


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

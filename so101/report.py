"""Markdown tables from the artifacts in ``results/`` so the README numbers are regenerated, never typed by hand.

    python -m so101.report --results-dir results --label outputs/main/final="main, 20k steps"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def eval_table(results: list[dict], labels: dict[str, str] | None = None) -> str:
    """Held-out metrics for every scored checkpoint plus the two baselines (taken from the first result)."""
    labels = labels or {}
    header = ["Policy", "Norm. MSE", "Norm. MAE", "Raw MAE", "Raw MSE @1", "Raw MSE @50"]
    rows = []
    base = results[0]
    for name, key in (("Hold last state", "hold"), ("Dataset mean action", "mean")):
        m = base[key]
        rows.append([name, _fmt(m["norm_mse"]), _fmt(m["norm_mae"]), _fmt(m["raw_mae"], 2),
                     _fmt(m["raw_mse_per_step"][0], 1), _fmt(m["raw_mse_per_step"][-1], 1)])
    for res in results:
        m = res["policy"]
        label = labels.get(res["checkpoint"], res["checkpoint"])
        rows.append([label, _fmt(m["norm_mse"]), _fmt(m["norm_mae"]), _fmt(m["raw_mae"], 2),
                     _fmt(m["raw_mse_per_step"][0], 1), _fmt(m["raw_mse_per_step"][-1], 1)])
    return _table(header, rows)


def run_summary(run_dir: Path, max_step: int | None = None) -> dict:
    """Best and last held-out MSE, mean step time and peak memory for one training run directory."""
    meta = json.loads((run_dir / "run_meta.json").read_text())
    val = _read_jsonl(run_dir / "val_log.jsonl")
    train = _read_jsonl(run_dir / "train_log.jsonl")
    if max_step is not None:
        val = [r for r in val if r["step"] <= max_step]
        train = [r for r in train if r["step"] <= max_step]
    scored = [r for r in val if r["step"] > 0]
    best = min(scored, key=lambda r: r["mse"]) if scored else None
    step_s = [r["step_s"] for r in train if r["step"] > 0]
    return {
        "trainable_m": meta["params"]["trainable"] / 1e6,
        "train_episodes": len(meta["dataset"]["train_episodes"]),
        "best_mse": best["mse"] if best else None,
        "best_step": best["step"] if best else None,
        "last_mse": scored[-1]["mse"] if scored else None,
        "last_step": scored[-1]["step"] if scored else None,
        "step_s": sum(step_s) / len(step_s) if step_s else None,
        "peak_mem_gb": max((r["peak_mem_gb"] for r in train), default=None),
    }


def ablation_table(runs: list[tuple[str, Path]], max_step: int | None = None) -> str:
    header = ["Run", "Trainable params", "Train episodes", "Best held-out MSE (step)", "Step time", "Peak VRAM"]
    rows = []
    for label, run_dir in runs:
        s = run_summary(run_dir, max_step)
        rows.append([
            label, f"{s['trainable_m']:.1f}M", str(s["train_episodes"]),
            f"{_fmt(s['best_mse'])} ({s['best_step']})" if s["best_mse"] is not None else "n/a",
            f"{s['step_s']:.2f} s" if s["step_s"] else "n/a",
            f"{s['peak_mem_gb']:.1f} GB" if s["peak_mem_gb"] else "n/a",
        ])
    return _table(header, rows)


def latency_table(benches: list[dict]) -> str:
    header = ["Precision", "p50", "p95", "max", "Chunks / s"]
    rows = []
    for b in benches:
        prec = "bf16 autocast" if b.get("autocast_bf16") else "fp32"
        rows.append([prec, f"{b['latency_ms_p50']:.1f} ms", f"{b['latency_ms_p95']:.1f} ms",
                     f"{b['latency_ms_max']:.1f} ms", f"{b['chunks_per_s']:.1f}"])
    return _table(header, rows)


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--label", action="append", default=[], metavar="CHECKPOINT=LABEL")
    parser.add_argument("--ablation-max-step", type=int, default=3000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.results_dir)
    labels = dict(item.split("=", 1) for item in args.label)
    sections = []
    if (root / "eval_offline.json").exists():
        results = json.loads((root / "eval_offline.json").read_text())
        sections.append("### Held-out episodes\n\n" + eval_table(results, labels))
    runs = []
    if (root / "main").exists():
        runs.append((f"main (first {args.ablation_max_step} steps)", root / "main"))
    runs += sorted((p.name.removeprefix("abl_"), p) for p in (root / "ablations").glob("abl_*") if p.is_dir())
    if runs:
        sections.append("### Ablations\n\n" + ablation_table(runs, args.ablation_max_step))
    benches = [json.loads(p.read_text()) for p in sorted(root.glob("bench_*.json"))]
    if benches:
        sections.append("### Inference latency\n\n" + latency_table(benches))
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()

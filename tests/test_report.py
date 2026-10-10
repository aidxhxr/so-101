import json

from so101.report import ablation_table, eval_table, latency_table, run_summary


def _eval_result(name, mse):
    per_step = [mse * (1 + i / 10) for i in range(50)]
    m = {"norm_mse": mse, "norm_mae": mse / 2, "raw_mse": mse * 100, "raw_mae": mse * 10,
         "raw_mae_per_joint": {}, "raw_mse_per_step": per_step, "norm_mse_per_step": per_step}
    return {"checkpoint": name, "policy": m, "hold": dict(m, norm_mse=0.99), "mean": dict(m, norm_mse=0.97)}


def test_eval_table_has_baselines_and_every_checkpoint():
    table = eval_table([_eval_result("a", 0.9), _eval_result("b", 0.2)], {"b": "fine-tuned"})
    lines = table.splitlines()
    assert lines[0].startswith("| Policy |")
    assert len(lines) == 2 + 2 + 2
    assert "| Hold last state | 0.990 |" in table
    assert "| fine-tuned | 0.200 |" in table
    assert "| a | 0.900 |" in table


def _write_run(tmp_path, name, mses, step_s=0.4):
    d = tmp_path / name
    d.mkdir()
    (d / "run_meta.json").write_text(json.dumps({"params": {"trainable": 1.5e6, "total": 2e6},
                                                 "dataset": {"train_episodes": [0, 1, 2]}}))
    (d / "val_log.jsonl").write_text("\n".join(json.dumps({"step": 500 * i, "mse": m, "mae": m, "loss": m})
                                               for i, m in enumerate(mses)))
    (d / "train_log.jsonl").write_text("\n".join(json.dumps({"step": 50 * i, "loss": 1.0, "step_s": step_s,
                                                             "peak_mem_gb": 10 + i}) for i in range(5)))
    return d


def test_run_summary_ignores_step_zero_and_respects_max_step(tmp_path):
    d = _write_run(tmp_path, "run", [0.05, 0.5, 0.3, 0.2])
    s = run_summary(d)
    assert s["best_mse"] == 0.2 and s["best_step"] == 1500 and s["last_step"] == 1500
    s = run_summary(d, max_step=1000)
    assert s["best_mse"] == 0.3 and s["last_step"] == 1000
    assert s["trainable_m"] == 1.5 and s["train_episodes"] == 3
    assert s["peak_mem_gb"] == 14 and abs(s["step_s"] - 0.4) < 1e-9


def test_ablation_and_latency_tables(tmp_path):
    d = _write_run(tmp_path, "abl", [1.0, 0.4])
    table = ablation_table([("frozen", d)])
    assert "| frozen | 1.5M | 3 | 0.400 (500) | 0.40 s | 14.0 GB |" in table
    lat = latency_table([{"autocast_bf16": True, "latency_ms_p50": 50.04, "latency_ms_p95": 60,
                          "latency_ms_max": 70, "chunks_per_s": 19.98}])
    assert "| bf16 autocast | 50.0 ms | 60.0 ms | 70.0 ms | 20.0 |" in lat

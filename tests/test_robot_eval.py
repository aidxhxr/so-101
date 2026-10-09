import json

from so101.robot_eval import summarize, wilson_interval


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = wilson_interval(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.96
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and lo > 0.69


def test_summarize_counts_successes(tmp_path):
    p = tmp_path / "trials.jsonl"
    rows = [
        {"trial": 0, "success": True, "duration_s": 12.0, "checkpoint": "a"},
        {"trial": 1, "success": False, "duration_s": 30.0, "checkpoint": "a"},
        {"trial": 2, "success": True, "duration_s": 14.0, "checkpoint": "a"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = summarize(p)
    assert s["trials"] == 3 and s["successes"] == 2
    assert s["success_rate"] == 2 / 3
    assert s["mean_success_duration_s"] == 13.0
    assert s["checkpoints"] == ["a"]

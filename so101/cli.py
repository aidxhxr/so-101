"""``so101`` command line: one entry point for the whole pipeline."""

from __future__ import annotations

import argparse
import sys

COMMANDS = {
    "teleop": ("so101.teleop", "Mirror the leader arm onto the follower"),
    "record": ("so101.record", "Record a teleoperated LeRobot dataset"),
    "train": ("so101.train", "Fine-tune SmolVLA with a held-out validation split"),
    "eval": ("so101.eval_offline", "Score checkpoints on held-out episodes against baselines"),
    "bench": ("so101.bench", "Measure inference latency of a checkpoint"),
    "deploy": ("so101.robot_eval", "Run closed-loop trials on the arm and log successes"),
    "trajectory": ("so101.trajectory", "Dump open-loop chunk predictions for one episode"),
    "plot": ("so101.plots", "Render training curves and prediction figures"),
    "report": ("so101.report", "Print the README result tables from results/"),
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="so101", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        sub.add_parser(name, help=help_text, add_help=False)
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    ns, rest = parser.parse_known_args(argv[:1])
    module_name = COMMANDS[ns.command][0]
    module = __import__(module_name, fromlist=["main"])
    module.main(argv[1:])


if __name__ == "__main__":
    main()

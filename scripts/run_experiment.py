"""Run one experiment from a YAML config and write runs/<run_id>/rounds.jsonl.

python scripts/run_experiment.py --config configs/smoke.yaml
python scripts/run_experiment.py --config configs/pathmnist_dev.yaml --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from privatefair.runlog import read_round_logs
from privatefair.sim.fedavg_loop import run_from_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--seed", type=int, default=None, help="overrides the config's seed")
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--run-id", default=None, help="default: <config name>-s<seed> (no timestamp, for reproducibility)")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    run_id = args.run_id or f"{args.config.stem}-s{seed}"

    path = run_from_config(cfg, seed, args.out, run_id)
    last = read_round_logs(path)[-1]
    print(f"wrote {path}")
    print(f"final round {last.epoch}: {last.global_metrics}")


if __name__ == "__main__":
    main()

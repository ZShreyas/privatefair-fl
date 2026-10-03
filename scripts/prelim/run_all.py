"""Run the 5 preliminary configs one after another into runs/prelim/, skipping complete runs.

python scripts/prelim/run_all.py                 # resumable: re-run to continue after an interruption
python scripts/prelim/run_all.py --only random   # just one config (repeatable)
python scripts/prelim/run_all.py --smoke --out /tmp/x   # seconds: tiny override, for checking the pipeline
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import yaml

from privatefair.sim.fedavg_loop import run_from_config

CONFIG_DIR = Path("configs/prelim")
NAMES = ["fedavg_all", "random", "coverage_random", "privatefair_rr", "privatefair_nodp"]
SEED = 0

# Same values as configs/smoke_privatefair.yaml (3 sites, 2 rounds); device stays whatever the config says.
SMOKE = {
    "data": {"subsample": 0.02, "n_sites": 3, "n_shifted": 1, "min_site_size": 20},
    "train": {"local_steps": 2, "batch_size": 16, "input_size": 32},
    "fl": {"rounds": 2, "eval_every": 1},
    "policy": {"capacity": 2, "max_age": 3},
    "telemetry": {"val_max_samples": 64, "reference_size": 200},
}


def is_complete(path: Path, rounds: int) -> bool:
    """True if rounds.jsonl exists and its last line is the final round."""
    if not path.exists():
        return False
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return bool(lines) and json.loads(lines[-1])["epoch"] == rounds - 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("runs/prelim"))
    p.add_argument("--only", action="append", choices=NAMES, help="run only these configs")
    p.add_argument("--smoke", action="store_true", help="tiny subsample override for pipeline checks")
    args = p.parse_args()

    t0 = time.perf_counter()
    for name in args.only or NAMES:
        cfg = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
        if args.smoke:
            cfg = copy.deepcopy(cfg)
            for section, over in SMOKE.items():
                cfg[section] = {**cfg.get(section, {}), **over}
        run_id = f"{name}-s{SEED}"
        path = args.out / run_id / "rounds.jsonl"
        if is_complete(path, cfg["fl"]["rounds"]):
            print(f"[skip] {run_id}: already complete", flush=True)
            continue
        print(f"[run ] {run_id}", flush=True)
        t1 = time.perf_counter()
        run_from_config(cfg, SEED, args.out, run_id, verbose=True)
        print(f"[done] {run_id} in {(time.perf_counter() - t1) / 60:.1f} min", flush=True)
    print(f"all done in {(time.perf_counter() - t0) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()

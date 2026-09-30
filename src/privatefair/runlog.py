"""RoundLog JSONL writer and reader (task A3). No torch.

Layout of one run:
    <out_dir>/<run_id>/config.json    the exact config used (incl. resolved seed)
    <out_dir>/<run_id>/rounds.jsonl   one RoundLog per line
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from privatefair.interfaces import RoundLog


class RunLogger:
    def __init__(self, out_dir: str | Path, run_id: str, config: Mapping[str, Any] | None = None) -> None:
        self.run_id = run_id
        self.dir = Path(out_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "rounds.jsonl"
        self.path.write_text("", encoding="utf-8")  # a rerun with the same run_id starts clean
        if config is not None:
            (self.dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    def log(self, record: RoundLog) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(record.to_json() + "\n")


def read_round_logs(path: str | Path) -> list[RoundLog]:
    with Path(path).open(encoding="utf-8") as f:
        return [RoundLog.from_json(line) for line in f if line.strip()]


# Real wall-clock measurements (the report asks for controller runtime). They differ run to run,
# so reproducibility checks (Gate G1) compare everything except these.
WALLCLOCK_FIELDS = ("controller_ms",)


def read_comparable(path: str | Path) -> list[dict[str, Any]]:
    """Rounds as plain dicts without wall-clock fields: equal lists mean two runs reproduced exactly."""
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                for key in WALLCLOCK_FIELDS:
                    d.pop(key, None)
                out.append(d)
    return out

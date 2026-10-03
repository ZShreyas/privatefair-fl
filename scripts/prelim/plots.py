"""Plots and summary table from runs/prelim/*/rounds.jsonl -> docs/figures/prelim/.

python scripts/prelim/plots.py

Deliberately simple: Member C's analysis toolkit (C3) will replace this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from privatefair.runlog import read_round_logs

MARK = "Preliminary — single seed, PathMNIST"
METRIC = "balanced_acc"
LABELS = {
    "fedavg_all": "FedAvg (all sites)",
    "random": "Random",
    "coverage_random": "Coverage-random",
    "privatefair_rr": "PrivateFair (RR, ε=1)",
    "privatefair_nodp": "PrivateFair (non-private)",
}
POLICIES = ["random", "coverage_random", "privatefair_rr", "privatefair_nodp"]  # the 4 selection policies
COLORS = {
    "random": "#888888",
    "coverage_random": "#E69F00",
    "privatefair_rr": "#0072B2",
    "privatefair_nodp": "#009E73",
}
BASE, ACCENT = "#0072B2", "#D55E00"


def load_runs(runs_dir: Path) -> dict[str, list]:
    runs = {}
    for name in LABELS:
        path = runs_dir / f"{name}-s0" / "rounds.jsonl"
        if path.exists():
            runs[name] = read_round_logs(path)
        else:
            print(f"warning: no run for {name} ({path}), skipping")
    return runs


def final_eval(logs: list):
    """The last round that has per-site metrics (always the final round in a complete run)."""
    return next(log for log in reversed(logs) if log.per_site_metrics)


def finish(fig, ax, title: str, path: Path) -> None:
    ax.set_title(f"{title}\n{MARK}", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_per_site(runs: dict, out: Path) -> None:
    if "fedavg_all" not in runs:
        return
    last = final_eval(runs["fedavg_all"])
    shifted = set(last.sim_only["shifted_sites"])
    sites = sorted(last.per_site_metrics)
    accs = [last.per_site_metrics[s][METRIC] for s in sites]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(s) for s in sites], accs, color=[ACCENT if s in shifted else BASE for s in sites])
    ax.axhline(sum(accs) / len(accs), color="black", ls=":", lw=1)
    ax.set_xlabel("site")
    ax.set_ylabel("test balanced accuracy")
    ax.set_ylim(0, 1)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=BASE, label="typical site"),
            plt.Rectangle((0, 0), 1, 1, color=ACCENT, label="shifted site"),
            plt.Line2D([], [], color="black", ls=":", label="mean"),
        ],
        frameon=False,
        loc="upper right",
    )
    finish(fig, ax, f"FedAvg (all sites): per-site accuracy, round {last.epoch + 1}", out / "a_fedavg_all_per_site.png")


def plot_worst(runs: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in POLICIES:
        if name not in runs:
            continue
        pts = [(log.epoch + 1, log.global_metrics["worst_balanced_acc"]) for log in runs[name] if log.per_site_metrics]
        ax.plot(*zip(*pts, strict=True), marker="o", ms=3, color=COLORS[name], label=LABELS[name])
    ax.set_xlabel("round")
    ax.set_ylabel("worst-site test balanced accuracy")
    ax.legend(frameon=False)
    finish(fig, ax, "Worst-site accuracy over rounds", out / "b_worst_site_acc.png")


def plot_max_age(runs: dict, runs_dir: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    max_age = None
    for i, name in enumerate(POLICIES):  # different widths/styles so identical lines stay visible
        if name not in runs:
            continue
        ages = [max(log.participation_age.values()) for log in runs[name]]
        ax.plot(
            range(1, len(ages) + 1),
            ages,
            color=COLORS[name],
            label=LABELS[name],
            lw=5 - i,
            ls=("-", "--", "-", "--")[i],
        )
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        cfg = json.loads((runs_dir / f"{name}-s0" / "config.json").read_text(encoding="utf-8"))
        max_age = cfg["policy"].get("max_age", max_age)
    if max_age is not None:
        ax.axhline(max_age, color="black", ls="--", lw=1, label=f"max_age limit ({max_age})")
    ax.set_xlabel("round")
    ax.set_ylabel("max participation age (rounds)")
    ax.legend(frameon=False)
    finish(fig, ax, "Maximum participation age over rounds", out / "c_max_age.png")


def summarize(runs: dict) -> pd.DataFrame:
    rows = []
    for name, logs in runs.items():
        last = final_eval(logs)
        shifted = set(last.sim_only["shifted_sites"])
        accs = {s: m[METRIC] for s, m in last.per_site_metrics.items()}
        eps = list(logs[-1].epsilon_spent.values())
        rows.append(
            {
                "run": name,
                "rounds": len(logs),
                "mean_acc": sum(accs.values()) / len(accs),
                "worst_site_acc": min(accs.values()),
                "shifted_mean_acc": sum(accs[s] for s in shifted) / len(shifted) if shifted else float("nan"),
                "eps_per_site_max": max(eps),
                "eps_per_site_mean": sum(eps) / len(eps),
                "sim_seconds_total": sum(log.round_seconds for log in logs),
                "bytes_total": sum(log.bytes_up + log.bytes_down for log in logs),
            }
        )
    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame, out: Path) -> None:
    df.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    show = df.copy()
    show["run"] = show["run"].map(LABELS)
    show["bytes_total"] = (show["bytes_total"] / 1e6).round(1)
    show = show.rename(columns={"bytes_total": "MB_total"})
    header = "| " + " | ".join(show.columns) + " |\n|" + "---|" * len(show.columns) + "\n"
    body = "".join(
        "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |\n"
        for row in show.itertuples(index=False)
    )
    note = (
        "\nAccuracies are final-round per-site test **balanced accuracy**. ε is the composed budget per site "
        "(sum over all released reports; 0 when nothing is privatized). Time is simulated seconds.\n"
    )
    (out / "summary.md").write_text(f"# Summary\n\n_{MARK}_\n\n{header}{body}{note}", encoding="utf-8")
    print(f"wrote {out / 'summary.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs/prelim"))
    p.add_argument("--out", type=Path, default=Path("docs/figures/prelim"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no runs found in {args.runs}")
    plot_per_site(runs, args.out)
    plot_worst(runs, args.out)
    plot_max_age(runs, args.runs, args.out)
    write_summary(summarize(runs), args.out)


if __name__ == "__main__":
    main()

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
    fig.tight_layout(rect=(0, 0.1, 1, 1) if fig.texts else None)  # leave room for a caption, if any
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
        rounds = range(1, len(runs[name]) + 1)
        raw = [max(log.participation_age.values()) for log in runs[name]]
        # Ages of sites that were online that round (the coverage rule only forces available sites).
        avail = [
            max((a for s, a in log.participation_age.items() if s in log.sim_only.get("available", [s])), default=0)
            for log in runs[name]
        ]
        ax.plot(rounds, raw, color=COLORS[name], lw=1, alpha=0.3)  # faint: raw max over all sites
        ax.plot(rounds, avail, color=COLORS[name], label=LABELS[name], lw=5 - i, ls=("-", "--", "-", "--")[i])
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        cfg = json.loads((runs_dir / f"{name}-s0" / "config.json").read_text(encoding="utf-8"))
        max_age = cfg["policy"].get("max_age", max_age)
    if max_age is not None:
        ax.axhline(max_age, color="black", ls="--", lw=1, label=f"max_age limit ({max_age})")
    ax.set_xlabel("round")
    ax.set_ylabel("max age, available sites (rounds)")
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.005,
        "Faint lines: max over all sites. It exceeds the limit only while the overdue site is offline\n"
        "or, once forced in, misses its deadline. The coverage rule can only force available sites.",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    finish(fig, ax, "Maximum participation age over rounds", out / "c_max_age.png")


def summarize(runs: dict) -> pd.DataFrame:
    rows = []
    for name, logs in runs.items():
        last = final_eval(logs)
        shifted = set(last.sim_only["shifted_sites"])
        accs = {s: m[METRIC] for s, m in last.per_site_metrics.items()}
        eps = list(logs[-1].epsilon_spent.values())
        evals = [log for log in logs if log.per_site_metrics][-3:]  # last 3 evaluation rounds
        mean_last3 = sum(
            sum(m[METRIC] for m in log.per_site_metrics.values()) / len(log.per_site_metrics) for log in evals
        ) / len(evals)
        worst_last3 = sum(min(m[METRIC] for m in log.per_site_metrics.values()) for log in evals) / len(evals)
        n_reports = sum(len(log.telemetry) for log in logs)  # one report = one site's privatized triple in a round
        rows.append(
            {
                "run": name,
                "rounds": len(logs),
                "mean_acc": sum(accs.values()) / len(accs),
                "worst_site_acc": min(accs.values()),
                "mean_acc_last3": mean_last3,
                "worst_site_acc_last3": worst_last3,
                "shifted_mean_acc": sum(accs[s] for s in shifted) / len(shifted) if shifted else float("nan"),
                "reports_per_site": n_reports / len(accs),
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
        "\nAccuracies are per-site test **balanced accuracy**: `mean_acc`, `worst_site_acc` and `shifted_mean_acc` "
        "are at the final round; `*_last3` average the mean / worst-site accuracy over the last 3 evaluations "
        "(rounds 40, 45, 50). `reports_per_site` = telemetry reports (one per signal triple) a site released over "
        "the run, on average. ε is the composed budget per site (sum over all released reports and signals; 0 when "
        "nothing is privatized). Time is simulated seconds.\n"
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

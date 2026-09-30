"""Plain FedAvg training loop: every available site trains every round (tasks A3, A4; Gate G1).

Determinism (same seed -> byte-identical rounds.jsonl):
  * each site gets its own numpy Generator from SeedSequence([seed, site_id]), so a site's
    batches do not depend on how many other sites exist or their order;
  * torch.manual_seed(seed) fixes the new classifier head's initialization;
  * no wall-clock values are logged. Without a `systems` config every site is always available
    and round_seconds is 0.0; with one (A4), availability, deadline misses and round_seconds
    come from the seeded systems simulation. controller_ms is 0.0: there is no controller here.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from privatefair.coordinator.participation import ParticipationTracker
from privatefair.data.pathmnist import SiteData, load_sites
from privatefair.fl.fedavg import FedAvg
from privatefair.interfaces import ClientUpdate, RoundLog, ScheduleMode
from privatefair.runlog import RunLogger
from privatefair.sim.local import TrainConfig, evaluate, train_local
from privatefair.sim.model import Weights, build_model, get_weights, set_weights
from privatefair.sim.systems import (
    JITTER_STREAM,
    PROFILE_STREAM,
    TRACE_STREAM,
    RoundTiming,
    SystemsConfig,
    deadline_seconds,
    generate_trace,
    make_profiles,
    simulate_round,
    systems_rng,
)

POLICY_NAME = "fedavg_all"


@dataclass(frozen=True)
class FLConfig:
    rounds: int = 50
    eval_every: int = 5  # evaluate every site at rounds eval_every, 2*eval_every, ... and the last round


def model_bytes(weights: Weights) -> int:
    return sum(v.numel() * v.element_size() for v in weights.values())


def site_rng(seed: int, site_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, site_id]))


def run_fedavg(
    sites: Sequence[SiteData],
    *,
    num_classes: int,
    train_cfg: TrainConfig,
    fl_cfg: FLConfig,
    seed: int,
    logger: RunLogger,
    pretrained: bool = True,
    systems: SystemsConfig | None = None,
    verbose: bool = False,
) -> Weights:
    """Run FedAvg over all (available) sites, logging one RoundLog per round. Returns the final global weights.

    `verbose` prints one progress line per round. Wall-clock time appears only in that
    printout, never in the log, so Gate G1 is unaffected.
    """
    start = time.perf_counter()
    torch.manual_seed(seed)
    model = build_model(num_classes, pretrained=pretrained)
    global_w = get_weights(model)
    nbytes = model_bytes(global_w)
    aggregator = FedAvg()
    ids = [s.site_id for s in sites]
    rngs = {s.site_id: site_rng(seed, s.site_id) for s in sites}
    shifted = sorted(s.site_id for s in sites if s.shifted)
    tracker = ParticipationTracker()

    if systems is not None:
        profiles = make_profiles(ids, systems, systems_rng(seed, PROFILE_STREAM))
        trace = generate_trace(ids, fl_cfg.rounds, systems, systems_rng(seed, TRACE_STREAM))
        trace.save(logger.dir / "systems_trace.json")  # recorded trace: replayable, citable
        jitter_rng = systems_rng(seed, JITTER_STREAM)
        deadline = deadline_seconds(profiles, train_cfg.local_steps, nbytes, nbytes, systems.deadline_factor)

    for epoch in range(fl_cfg.rounds):
        if systems is None:
            participants = ids
            timing = RoundTiming({}, frozenset(ids), 0.0)
        else:
            participants = sorted(trace.at(epoch))
            steps = train_cfg.local_steps
            timing = simulate_round(
                profiles, participants, steps, nbytes, nbytes, deadline, systems.jitter_sigma, jitter_rng
            )

        updates, train_losses = [], []
        for s in sites:
            if s.site_id not in participants:
                continue
            done = s.site_id in timing.completed
            weights = None
            if done:  # a site that will miss the deadline contributes nothing, so skip its (costly) training
                set_weights(model, global_w)
                res = train_local(model, s.x_train, s.y_train, train_cfg, rngs[s.site_id])
                weights = res.weights
                train_losses.append(res.mean_loss)
            secs = timing.site_seconds.get(s.site_id, 0.0)
            updates.append(ClientUpdate(s.site_id, epoch, ScheduleMode.FULL, weights, secs, nbytes, completed=done))
        global_w = aggregator.aggregate(global_w, updates)
        tracker.update(epoch, timing.completed)

        global_metrics: dict[str, float] = {}
        if train_losses:
            global_metrics["mean_train_loss"] = float(np.mean(train_losses))
        per_site: dict[int, dict[str, float]] = {}
        if (epoch + 1) % fl_cfg.eval_every == 0 or epoch == fl_cfg.rounds - 1:
            set_weights(model, global_w)
            per_site = {s.site_id: evaluate(model, s.x_test, s.y_test, train_cfg, num_classes) for s in sites}
            bal = [m["balanced_acc"] for m in per_site.values()]
            global_metrics |= {"mean_balanced_acc": float(np.mean(bal)), "worst_balanced_acc": float(min(bal))}

        sim_only: dict[str, Any] = {"shifted_sites": shifted}
        if systems is not None:
            sim_only |= {
                "available": participants,
                "deadline_missed": sorted(set(participants) - timing.completed),
                "site_seconds": timing.site_seconds,
                "deadline_seconds": deadline,
            }
        logger.log(
            RoundLog(
                run_id=logger.run_id,
                seed=seed,
                policy=POLICY_NAME,
                epoch=epoch,
                selected=list(participants),
                modes={i: ScheduleMode.FULL.value for i in participants},
                mandatory_sites=[],
                deferral_reasons={},
                telemetry=[],
                posteriors=[],
                participation_age=tracker.ages(ids, epoch),  # server-derived, after this round
                epsilon_spent={i: 0.0 for i in ids},  # no telemetry released
                round_seconds=timing.round_seconds,
                bytes_up=nbytes * len(timing.completed),  # only finished sites upload
                bytes_down=nbytes * len(participants),
                controller_ms=0.0,
                per_site_metrics=per_site,
                global_metrics=global_metrics,
                sim_only=sim_only,
            )
        )
        if verbose:
            _print_progress(epoch, fl_cfg.rounds, global_metrics, time.perf_counter() - start)
    return global_w


def _print_progress(epoch: int, rounds: int, metrics: Mapping[str, float], elapsed: float) -> None:
    done = epoch + 1
    eta_min = elapsed / done * (rounds - done) / 60
    line = f"round {done:>3}/{rounds}"
    if "mean_train_loss" in metrics:  # absent if every participant missed the deadline this round (A4)
        line += f"  train_loss {metrics['mean_train_loss']:.3f}"
    if "worst_balanced_acc" in metrics:
        line += f"  mean_acc {metrics['mean_balanced_acc']:.3f}  worst_acc {metrics['worst_balanced_acc']:.3f}"
    print(f"{line}  [{elapsed / 60:.1f} min, ~{eta_min:.0f} min left]", flush=True)


def run_from_config(
    cfg: Mapping[str, Any],
    seed: int,
    out_dir: str | Path,
    run_id: str,
    sites: Sequence[SiteData] | None = None,
    verbose: bool = False,
) -> Path:
    """Build everything from a parsed YAML config, run, and return the rounds.jsonl path."""
    model_cfg = cfg.get("model", {})
    if model_cfg.get("arch", "resnet18") != "resnet18":
        raise ValueError(f"unsupported arch {model_cfg['arch']!r}")
    if sites is None:
        sites = load_sites(cfg["data"], seed)
    logger = RunLogger(out_dir, run_id, config={**cfg, "seed": seed})
    run_fedavg(
        sites,
        num_classes=model_cfg.get("num_classes", 9),
        train_cfg=TrainConfig(**cfg.get("train", {})),
        fl_cfg=FLConfig(**cfg.get("fl", {})),
        seed=seed,
        logger=logger,
        pretrained=model_cfg.get("pretrained", True),
        systems=SystemsConfig(**cfg["systems"]) if cfg.get("systems") else None,
        verbose=verbose,
    )
    return logger.path

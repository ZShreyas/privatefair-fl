"""Federated training loop with FedAvg aggregation (tasks A3, A4, A6; Gates G1, G3).

Per round:
  availability (A4 trace, or everyone)
  -> [if the policy needs it] true bins per site (A5) -> privatize (B1) + ledger (B3)
     -> wire: TelemetryReport.to_payload() / from_payload()  (the privacy boundary, Gate G3)
  -> coordinator.select(epoch, reports, available)  (B4/B5; plain FedAvg if no policy)
  -> full / compressed local training (compressed = fewer steps) -> simulated timing (A4)
  -> equal-weight FedAvg over completed updates -> coordinator.observe(...) -> RoundLog

The coordinator only ever receives `epoch`, schema-validated `TelemetryReport`s and the
`available` set in select(), and the decision plus `completed` in observe().

Determinism (same seed -> identical rounds.jsonl, except `controller_ms`, which is real
wall-clock time of select(); compare runs with runlog.read_comparable):
  * per-site training streams SeedSequence([seed, site_id]); systems, policy, telemetry and
    true-bin randomness use separate SeedSequence spawn keys, so they never collide;
  * torch.manual_seed(seed) fixes the new classifier head's initialization.
Without a `policy` the log is byte-identical to the A3/A4 loop.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from privatefair.coordinator.participation import ParticipationTracker
from privatefair.coordinator.posterior import decode_report
from privatefair.data.pathmnist import SiteData, load_pathmnist, load_sites
from privatefair.fl.fedavg import FedAvg
from privatefair.interfaces import ClientUpdate, CohortDecision, RoundLog, ScheduleMode, TelemetryReport, TrueBins
from privatefair.privacy.ledger import PrivacyLedger
from privatefair.privacy.rr import RandomizedResponsePrivatizer
from privatefair.runlog import RunLogger
from privatefair.sim.local import TrainConfig, evaluate, train_local
from privatefair.sim.model import Weights, build_model, get_weights, set_weights
from privatefair.sim.policies import PolicySpec, PrivacyConfig, build_policy
from privatefair.sim.shift_detector import ShiftDetector
from privatefair.sim.systems import (
    JITTER_STREAM,
    PROFILE_STREAM,
    TRACE_STREAM,
    RoundTiming,
    SystemsConfig,
    deadline_seconds,
    expected_seconds,
    generate_trace,
    make_profiles,
    simulate_round,
    systems_rng,
)
from privatefair.sim.true_bins import ReadinessInput, TelemetryConfig, TrueBinsSimulator

POLICY_NAME = "fedavg_all"
# Spawn-key streams beyond A4's systems streams (0-2).
POLICY_STREAM, TRUTH_STREAM, REFERENCE_STREAM, TELEMETRY_STREAM = 10, 11, 12, 13


@dataclass(frozen=True)
class FLConfig:
    rounds: int = 50
    eval_every: int = 5  # evaluate every site at rounds eval_every, 2*eval_every, ... and the last round
    compressed_step_fraction: float = 0.5  # COMPRESSED sites run this fraction of local_steps (at least 1)


def model_bytes(weights: Weights) -> int:
    return sum(v.numel() * v.element_size() for v in weights.values())


def site_rng(seed: int, site_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, site_id]))


def stream_rng(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def _to_wire(report: TelemetryReport) -> TelemetryReport:
    """What crosses the network: exactly the five payload keys, schema-checked on arrival."""
    return TelemetryReport.from_payload(report.to_payload())


def _steps_for(mode: ScheduleMode, train_cfg: TrainConfig, fl_cfg: FLConfig) -> int:
    if mode is ScheduleMode.COMPRESSED:
        return max(1, round(fl_cfg.compressed_step_fraction * train_cfg.local_steps))
    return train_cfg.local_steps


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
    policy: PolicySpec | None = None,
    privacy: PrivacyConfig | None = None,
    telemetry: TelemetryConfig | None = None,
    reference: tuple[np.ndarray, np.ndarray] | None = None,
) -> Weights:
    """Run the loop, logging one RoundLog per round. Returns the final global weights.

    `policy=None` (or the `fedavg_all` spec) trains every available site every round. A policy
    that needs truth requires `reference` = (public unshifted images, labels) for the shift detector.
    `verbose` prints one progress line per round; wall-clock time appears only in that printout.
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

    coord = policy.coordinator if policy is not None else None
    policy_name = policy.name if policy is not None else POLICY_NAME
    privacy = privacy or PrivacyConfig()
    ledger = PrivacyLedger(cap=privacy.cap)
    privatizer = RandomizedResponsePrivatizer(privacy.epsilon) if privacy.mode == "rr" else None
    tel_rngs = {sid: stream_rng(seed, TELEMETRY_STREAM, sid) for sid in ids}
    truth_sim = None
    if policy is not None and policy.needs_truth:
        if reference is None:
            raise ValueError(f"policy {policy.name!r} needs true bins: pass reference=(images, labels)")
        detector = ShiftDetector(input_size=train_cfg.input_size, pretrained=pretrained).fit(*reference, num_classes)
        truth_rng = stream_rng(seed, TRUTH_STREAM)
        tel_cfg = telemetry or TelemetryConfig()
        truth_sim = TrueBinsSimulator(sites, detector, tel_cfg, train_cfg, num_classes, truth_rng)

    deadline: float | None = None
    if systems is not None:
        profiles = make_profiles(ids, systems, systems_rng(seed, PROFILE_STREAM))
        trace = generate_trace(ids, fl_cfg.rounds, systems, systems_rng(seed, TRACE_STREAM))
        trace.save(logger.dir / "systems_trace.json")  # recorded trace: replayable, citable
        jitter_rng = systems_rng(seed, JITTER_STREAM)
        deadline = deadline_seconds(profiles, train_cfg.local_steps, nbytes, nbytes, systems.deadline_factor)

    for epoch in range(fl_cfg.rounds):
        available = ids if systems is None else sorted(trace.at(epoch))

        # --- site side: measure, privatize, send -------------------------------------------------
        true_bins: dict[int, TrueBins] = {}
        raw: dict[str, Any] = {}
        reports: dict[int, TelemetryReport] = {}
        if truth_sim is not None:
            readiness = {
                sid: ReadinessInput(
                    available=sid in available,
                    expected_seconds=expected_seconds(profiles[sid], train_cfg.local_steps, nbytes, nbytes),
                    deadline_seconds=deadline,
                )
                if systems is not None
                else ReadinessInput()
                for sid in ids
            }
            set_weights(model, global_w)  # every site measures on the current global model
            true_bins, raw = truth_sim.compute(epoch, model, readiness)
            if systems is not None:
                raw |= {"expected_seconds": {sid: r.expected_seconds for sid, r in readiness.items()}}
                raw |= {"deadline_seconds": deadline}
        if policy is not None and policy.uses_telemetry:
            for sid in available:  # offline sites send nothing
                if privatizer is not None:
                    report, entries = privatizer.privatize(true_bins[sid], tel_rngs[sid])
                    ledger.record(entries)  # raises PrivacyBudgetExceeded past the cap: the run stops
                else:  # privacy.mode "none": non-private baselines see the true bins
                    b = true_bins[sid]
                    report = TelemetryReport(sid, epoch, b.utility, b.readiness, b.shift)
                reports[sid] = _to_wire(report)

        # --- coordinator side ------------------------------------------------------------------
        controller_ms = 0.0
        if coord is None:
            decision = None
            modes = {sid: ScheduleMode.FULL for sid in available}
        else:
            if hasattr(coord, "observe_truth"):  # B6 truth-seeing baselines only (never in coordinator/)
                coord.observe_truth(dict(true_bins), raw)
            t0 = time.perf_counter()
            decision = coord.select(epoch, reports, frozenset(available))
            controller_ms = (time.perf_counter() - t0) * 1000
            modes = {sid: m for sid, m in decision.assignments.items() if m is not ScheduleMode.DEFERRED}
        selected = sorted(modes)
        steps = {sid: _steps_for(modes[sid], train_cfg, fl_cfg) for sid in selected}

        # --- training, timing, aggregation -----------------------------------------------------
        if systems is None:
            timing = RoundTiming({}, frozenset(selected), 0.0)
        else:
            jitter = systems.jitter_sigma
            timing = simulate_round(profiles, selected, steps, nbytes, nbytes, deadline, jitter, jitter_rng)
        updates, train_losses = [], []
        for s in sites:
            if s.site_id not in modes:
                continue
            done = s.site_id in timing.completed
            weights = None
            if done:  # a site that will miss the deadline contributes nothing, so skip its (costly) training
                set_weights(model, global_w)
                cfg = (
                    train_cfg
                    if steps[s.site_id] == train_cfg.local_steps
                    else dataclasses.replace(train_cfg, local_steps=steps[s.site_id])
                )
                res = train_local(model, s.x_train, s.y_train, cfg, rngs[s.site_id])
                weights = res.weights
                train_losses.append(res.mean_loss)
            secs = timing.site_seconds.get(s.site_id, 0.0)
            mode = modes[s.site_id]
            updates.append(ClientUpdate(s.site_id, epoch, mode, weights, secs, nbytes, completed=done))
        global_w = aggregator.aggregate(global_w, updates)
        tracker.update(epoch, timing.completed)
        if coord is not None:
            coord.observe(decision, timing.completed)

        # --- evaluation and logging ------------------------------------------------------------
        global_metrics: dict[str, float] = {}
        if train_losses:
            global_metrics["mean_train_loss"] = float(np.mean(train_losses))
        per_site: dict[int, dict[str, float]] = {}
        if (epoch + 1) % fl_cfg.eval_every == 0 or epoch == fl_cfg.rounds - 1:
            set_weights(model, global_w)
            per_site = {s.site_id: evaluate(model, s.x_test, s.y_test, train_cfg, num_classes) for s in sites}
            bal = [m["balanced_acc"] for m in per_site.values()]
            global_metrics |= {"mean_balanced_acc": float(np.mean(bal)), "worst_balanced_acc": float(min(bal))}

        sim_only = _sim_only(shifted, systems is not None, available, timing, deadline, coord is not None, steps,
                             true_bins, raw)  # fmt: skip
        logger.log(
            _round_log(
                run_id=logger.run_id,
                seed=seed,
                policy_name=policy_name,
                epoch=epoch,
                available=available,
                selected=selected,
                modes=modes,
                decision=decision,
                reports=reports,
                decode_with=privacy if privatizer is not None else None,
                ages=tracker.ages(ids, epoch),
                eps_spent={sid: ledger.spent(sid) for sid in ids},
                timing=timing,
                nbytes=nbytes,
                controller_ms=controller_ms,
                per_site=per_site,
                global_metrics=global_metrics,
                sim_only=sim_only,
            )
        )
        if verbose:
            _print_progress(epoch, fl_cfg.rounds, global_metrics, time.perf_counter() - start)
    return global_w


def _round_log(
    *,
    run_id: str,
    seed: int,
    policy_name: str,
    epoch: int,
    available: Sequence[int],
    selected: list[int],
    modes: Mapping[int, ScheduleMode],
    decision: CohortDecision | None,
    reports: Mapping[int, TelemetryReport],
    decode_with: PrivacyConfig | None,
    ages: dict[int, int],
    eps_spent: dict[int, float],
    timing: RoundTiming,
    nbytes: int,
    controller_ms: float,
    per_site: dict[int, dict[str, float]],
    global_metrics: dict[str, float],
    sim_only: dict[str, Any],
) -> RoundLog:
    if decision is None:  # plain FedAvg: all available sites, all FULL
        all_modes = {sid: m.value for sid, m in modes.items()}
        mandatory, deferral = [], {}
    else:
        all_modes = {sid: m.value for sid, m in decision.assignments.items()}
        mandatory, deferral = sorted(decision.mandatory_sites), dict(decision.deferral_reasons)
    # The coordinator's posteriors, recomputed loop-side with the same channel and priors, for analysis.
    posteriors = []
    if decode_with is not None:
        priors = decode_with.decoding_priors()
        posteriors = [asdict(decode_report(r, decode_with.epsilon, priors)) for _, r in sorted(reports.items())]
    return RoundLog(
        run_id=run_id,
        seed=seed,
        policy=policy_name,
        epoch=epoch,
        selected=selected if decision is not None else list(available),
        modes=all_modes,
        mandatory_sites=mandatory,
        deferral_reasons=deferral,
        telemetry=[r.to_payload() for _, r in sorted(reports.items())],
        posteriors=posteriors,
        participation_age=ages,  # server-derived, after this round
        epsilon_spent=eps_spent,  # cumulative per site
        round_seconds=timing.round_seconds,
        bytes_up=nbytes * len(timing.completed),  # only finished sites upload
        bytes_down=nbytes * len(selected),
        controller_ms=controller_ms,
        per_site_metrics=per_site,
        global_metrics=global_metrics,
        sim_only=sim_only,
    )


def _sim_only(
    shifted: list[int],
    with_systems: bool,
    available: Sequence[int],
    timing: RoundTiming,
    deadline: float | None,
    with_policy: bool,
    steps: Mapping[int, int],
    true_bins: Mapping[int, TrueBins],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Simulator ground truth for offline evaluation only. Never passed to the coordinator."""
    out: dict[str, Any] = {"shifted_sites": shifted}
    if with_systems:
        out |= {
            "available": list(available),
            "deadline_missed": sorted(set(steps) - timing.completed),  # among sites that were asked to train
            "site_seconds": timing.site_seconds,
            "deadline_seconds": deadline,
        }
    if with_policy:
        out["local_steps"] = dict(steps)
        if true_bins:
            out["true_bins"] = [asdict(b) for _, b in sorted(true_bins.items())]
            out["raw_telemetry"] = dict(raw)
    return out


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
    reference: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    """Build everything from a parsed YAML config, run, and return the rounds.jsonl path."""
    model_cfg = cfg.get("model", {})
    if model_cfg.get("arch", "resnet18") != "resnet18":
        raise ValueError(f"unsupported arch {model_cfg['arch']!r}")
    if sites is None:
        sites = load_sites(cfg["data"], seed)
    privacy = PrivacyConfig.from_dict(cfg.get("privacy"))
    policy = build_policy(cfg["policy"], privacy, stream_rng(seed, POLICY_STREAM)) if cfg.get("policy") else None
    tel = dict(cfg.get("telemetry") or {})
    if "shift_thresholds" in tel:
        tel["shift_thresholds"] = tuple(tel["shift_thresholds"])
    telemetry = TelemetryConfig(**tel)
    if policy is not None and policy.needs_truth and reference is None:
        # Public typical-domain reference for the shift detector: unshifted pooled images (simulation shortcut).
        x_all, y_all = load_pathmnist(cfg["data"].get("root", "data/raw"), cfg["data"].get("image_size", 28))
        idx = np.sort(stream_rng(seed, REFERENCE_STREAM).choice(len(y_all), telemetry.reference_size, replace=False))
        reference = (x_all[idx], y_all[idx])
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
        policy=policy,
        privacy=privacy,
        telemetry=telemetry,
        reference=reference,
    )
    return logger.path

"""The PrivateFair coordinator (task B5): posterior score, coverage, shifted-site
slot, and full/compressed/deferred scheduling (report section 4 "Privacy mechanism
and decision rule"; section 7.2 step 4, "Constrained selection and scheduling").

Each epoch's `select()`:
  1. Decode every available site's TelemetryReport into a Posterior (B2).
  2. Reserve `min(required_now(...), capacity)` sites -- the oldest, by
     participation age -- using the earliest-deadline-first feasibility check
     shared with B4's CoverageRandomPolicy
     (`coordinator.participation.required_now`). Forcing in only sites already
     at `age >= max_age` is not enough to bound age on its own: several sites
     can cross that threshold together in a later round with no room left for
     all of them, so the reservation has to look ahead across every site's
     deadline, not just this round's.
  3. If capacity remains, reserve one slot for the best "likely shifted" site
     available (its posterior's single most probable shift bin is the top one --
     a MAP call, not a raw-probability threshold, so the decision is easy to
     audit and log).
  4. Fill any remaining capacity by descending posterior score:
         score = alpha * risk_hat + beta * p_shift_top + gamma * age + delta * readiness_hat
     risk_hat and readiness_hat are the posterior's normalized expected bin index
     (E[bin] / (K-1), in [0, 1]); p_shift_top is Pr(shift = the top bin).
  5. Every selected site is assigned FULL if its posterior's most likely
     readiness bin is >= 2 ("finishes a full round within the deadline" and
     above, per A5's TrueBins definition), else COMPRESSED -- using only the
     already-computed MAP bin, never a raw telemetry value.

This module only ever sees privatized TelemetryReport payloads and its own
participation-age state -- never TrueBins -- so it respects the privacy boundary
tested in tests/test_boundaries.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from privatefair.coordinator.participation import ParticipationTracker, required_now
from privatefair.coordinator.posterior import decode_report
from privatefair.interfaces import (
    ALPHABET_SIZE,
    SIGNALS,
    CohortDecision,
    Posterior,
    ScheduleMode,
    Signal,
    TelemetryReport,
)


def _expected_bin(p: tuple[float, ...]) -> float:
    """E[bin index] under a posterior distribution over an ordinal alphabet."""
    return sum(k * pk for k, pk in enumerate(p))


def risk_hat(posterior: Posterior) -> float:
    """Normalized expected utility/tail-risk bin, in [0, 1]. Higher = more deteriorated."""
    k = ALPHABET_SIZE["utility"]
    return _expected_bin(posterior.p_utility) / (k - 1)


def readiness_hat(posterior: Posterior) -> float:
    """Normalized expected readiness bin, in [0, 1]. Higher = faster / more ready."""
    k = ALPHABET_SIZE["readiness"]
    return _expected_bin(posterior.p_readiness) / (k - 1)


def p_shift_top(posterior: Posterior) -> float:
    """Pr(shift = the most-shifted bin): likelihood the site is in the shifted domain."""
    return posterior.p_shift[-1]


def is_likely_shifted(posterior: Posterior) -> bool:
    """True if the shifted-domain bin is the posterior's single most probable bin (MAP)."""
    return int(np.argmax(posterior.p_shift)) == len(posterior.p_shift) - 1


# Per A5's TrueBins definition (issue #37): readiness bin 2 means "finishes a full
# round within the deadline", bin 3 is faster still. Both can take a FULL load.
FULL_CAPABLE_READINESS_BIN = 2


def is_full_capable(posterior: Posterior) -> bool:
    """True if the posterior's MAP readiness bin can complete a full round within the deadline."""
    return int(np.argmax(posterior.p_readiness)) >= FULL_CAPABLE_READINESS_BIN


@dataclass
class PrivateFairCoordinator:
    """The proposed coordinator: posterior score + coverage + shifted-site slot.

    `epsilon`/`priors` are passed straight through to
    `coordinator.posterior.decode_report` for every available site's
    TelemetryReport, so they must match what the site's Privatizer actually used
    (same shape: one shared value, or a per-signal mapping).

    A site in `available` with no entry in `reports` (e.g. it went silent this
    round) is still eligible for mandatory coverage -- age is server-derived, not
    telemetry -- but contributes score 0.0, is never picked for the shifted-site
    slot (no posterior to judge shift from), and if selected gets COMPRESSED
    (readiness unknown, so never assume it can take a FULL load).

    Two ablation knobs, both from report section 10.E (task B6):
    `coverage=False` disables the reservation step entirely (step 1 below) --
    everything else (shifted slot, score fill) is unaffected, isolating the
    fairness constraint's cost/benefit. `drop_signal` removes one telemetry
    signal from the decision everywhere it would otherwise matter: its score
    weight is forced to 0, "shift" additionally disables the shifted-site slot,
    and "readiness" additionally assigns FULL to every selected site (an
    optimistic default given no readiness info, not COMPRESSED -- forcing
    COMPRESSED would itself halve training compute regardless of need, which
    would confound "cost of losing the signal" with "cost of doing less
    training"; a genuinely slow site's real cost then shows up correctly as a
    deadline miss in the systems simulation instead).
    """

    capacity: int
    max_age: int
    rng: np.random.Generator
    epsilon: float | Mapping[Signal, float]
    priors: Mapping[Signal, tuple[float, ...]] | None = None
    alpha: float = 1.0  # risk weight
    beta: float = 1.0  # shift weight
    gamma: float = 0.1  # age weight -- raw epochs, kept small relative to the [0, 1]-scaled terms
    delta: float = 1.0  # readiness weight
    coverage: bool = True  # False: no-coverage ablation (B6)
    drop_signal: Signal | None = None  # B6: drop-one-signal ablation
    name: str = "privatefair"
    tracker: ParticipationTracker = field(default_factory=ParticipationTracker)

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {self.capacity}")
        if self.max_age < 0:
            raise ValueError(f"max_age must be >= 0, got {self.max_age}")
        if self.drop_signal is not None and self.drop_signal not in SIGNALS:
            raise ValueError(f"drop_signal must be one of {SIGNALS} or None, got {self.drop_signal!r}")

    def _posterior(self, site_id: int, report: TelemetryReport | None, epoch: int) -> Posterior | None:
        """Decode one site's posterior for this epoch, or None if there's nothing to decode from.

        The default Bayes-decodes `report` (B2). Override this single seam to change
        how a report is interpreted without touching the rest of `select()` -- see
        `baselines.naive.NaiveDecodingCoordinator` (trust the report as exact) and
        `baselines.oracle.RawOracleCoordinator` (ignore the report, use ground truth).
        """
        if report is None:
            return None
        return decode_report(report, self.epsilon, self.priors)

    def score(self, posterior: Posterior, age: int) -> float:
        """The constrained-selection score from report section 4."""
        alpha = 0.0 if self.drop_signal == "utility" else self.alpha
        beta = 0.0 if self.drop_signal == "shift" else self.beta
        delta = 0.0 if self.drop_signal == "readiness" else self.delta
        return (
            alpha * risk_hat(posterior)
            + beta * p_shift_top(posterior)
            + self.gamma * age
            + delta * readiness_hat(posterior)
        )

    def _full_capable(self, site_id: int, posterior: Posterior) -> bool:
        """Whether a selected site can complete a FULL round within the deadline.

        Default: the posterior's MAP readiness bin is >= FULL_CAPABLE_READINESS_BIN
        (`is_full_capable`). Override this -- separately from `_posterior` -- for a
        coordinator whose posterior isn't a real categorical belief and so its
        argmax doesn't track the literal deadline boundary: e.g. the raw-telemetry
        oracle engineers a posterior to hit a target *expectation* (for `score()`'s
        continuous readiness_hat term), which is a different property from "which
        bin is most probable" and does not reliably put the argmax on the correct
        side of the FULL/COMPRESSED threshold. See `baselines.oracle.RawOracleCoordinator`.
        """
        return is_full_capable(posterior)

    def _is_likely_shifted(self, site_id: int, posterior: Posterior) -> bool:
        """Whether a site is eligible for the shifted-site representation slot.

        Default: the posterior's MAP shift bin is the top (shifted) one
        (`is_likely_shifted`). Same override rationale as `_full_capable`.
        """
        return is_likely_shifted(posterior)

    def select(self, epoch: int, reports: Mapping[int, TelemetryReport], available: frozenset[int]) -> CohortDecision:
        candidates = sorted(available)
        ages = {s: self.tracker.age(s, epoch) for s in candidates}
        posteriors: dict[int, Posterior] = {}
        for s in candidates:
            p = self._posterior(s, reports.get(s), epoch)
            if p is not None:
                posteriors[s] = p
        scores = {s: (self.score(posteriors[s], ages[s]) if s in posteriors else 0.0) for s in candidates}

        overdue_set = frozenset(s for s in candidates if ages[s] >= self.max_age)
        tiebreak = {s: self.rng.random() for s in candidates}

        # 1. Coverage reservation: the minimum set of oldest sites that must be
        # admitted now to keep every site's age bounded (see required_now's
        # docstring) -- not just the sites already overdue. Skipped entirely
        # under the no-coverage ablation.
        need = min(required_now(ages.values(), self.capacity, self.max_age), self.capacity) if self.coverage else 0
        order_by_age = sorted(candidates, key=lambda s: (-ages[s], -tiebreak[s]))
        reserved = frozenset(order_by_age[:need])
        selected: set[int] = set(reserved)

        # 2. Shifted-site representation slot, if capacity remains and the shift
        # signal isn't the one being dropped.
        remaining = self.capacity - len(selected)
        if remaining > 0 and self.drop_signal != "shift":
            shifted_candidates = [
                s
                for s in candidates
                if s not in selected and s in posteriors and self._is_likely_shifted(s, posteriors[s])
            ]
            if shifted_candidates:
                best = max(shifted_candidates, key=lambda s: (p_shift_top(posteriors[s]), tiebreak[s]))
                selected.add(best)

        # 3. Fill remaining capacity by descending score.
        remaining = self.capacity - len(selected)
        if remaining > 0:
            rest = sorted((s for s in candidates if s not in selected), key=lambda s: (-scores[s], -tiebreak[s]))
            selected.update(rest[:remaining])

        selected_frozen = frozenset(selected)
        # Report only sites that were *already* overdue and got in under an active
        # coverage rule as "mandatory" -- a reservation can also admit a site
        # slightly ahead of its deadline as part of keeping the schedule feasible,
        # and with coverage disabled nothing is ever truly mandatory.
        mandatory = (overdue_set & selected_frozen) if self.coverage else frozenset()

        assignments: dict[int, ScheduleMode] = {}
        deferral_reasons: dict[int, str] = {}
        for s in candidates:
            if s in selected_frozen:
                if self.drop_signal == "readiness":
                    # No readiness info to act on: assume FULL rather than forcing
                    # COMPRESSED. Forcing COMPRESSED would itself halve training
                    # compute regardless of whether a site actually needed it,
                    # confounding "cost of losing the signal" with "cost of less
                    # training". A genuinely slow site's real cost shows up
                    # correctly as a deadline miss in the systems simulation.
                    full_capable = True
                else:
                    full_capable = s in posteriors and self._full_capable(s, posteriors[s])
                assignments[s] = ScheduleMode.FULL if full_capable else ScheduleMode.COMPRESSED
            else:
                assignments[s] = ScheduleMode.DEFERRED
                deferral_reasons[s] = (
                    "coverage: overdue but capacity exceeded"
                    if s in overdue_set and self.coverage
                    else "not selected: score"
                )

        return CohortDecision(
            epoch=epoch,
            policy=self.name,
            assignments=assignments,
            mandatory_sites=mandatory,
            scores=scores,
            deferral_reasons=deferral_reasons,
        )

    def observe(self, decision: CohortDecision, completed: frozenset[int]) -> None:
        if not completed <= decision.selected:
            raise ValueError(
                f"observe() got completed sites not in the decision's selection: "
                f"{sorted(completed - decision.selected)}"
            )
        self.tracker.update(decision.epoch, completed)

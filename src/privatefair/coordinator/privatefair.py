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
from privatefair.interfaces import ALPHABET_SIZE, CohortDecision, Posterior, ScheduleMode, Signal, TelemetryReport


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
    name: str = "privatefair"
    tracker: ParticipationTracker = field(default_factory=ParticipationTracker)

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {self.capacity}")
        if self.max_age < 0:
            raise ValueError(f"max_age must be >= 0, got {self.max_age}")

    def score(self, posterior: Posterior, age: int) -> float:
        """The constrained-selection score from report section 4."""
        return (
            self.alpha * risk_hat(posterior)
            + self.beta * p_shift_top(posterior)
            + self.gamma * age
            + self.delta * readiness_hat(posterior)
        )

    def select(self, epoch: int, reports: Mapping[int, TelemetryReport], available: frozenset[int]) -> CohortDecision:
        candidates = sorted(available)
        ages = {s: self.tracker.age(s, epoch) for s in candidates}
        posteriors: dict[int, Posterior] = {
            s: decode_report(reports[s], self.epsilon, self.priors) for s in candidates if s in reports
        }
        scores = {s: (self.score(posteriors[s], ages[s]) if s in posteriors else 0.0) for s in candidates}

        overdue_set = frozenset(s for s in candidates if ages[s] >= self.max_age)
        tiebreak = {s: self.rng.random() for s in candidates}

        # 1. Coverage reservation: the minimum set of oldest sites that must be
        # admitted now to keep every site's age bounded (see required_now's
        # docstring) -- not just the sites already overdue.
        need = min(required_now(ages.values(), self.capacity, self.max_age), self.capacity)
        order_by_age = sorted(candidates, key=lambda s: (-ages[s], -tiebreak[s]))
        reserved = frozenset(order_by_age[:need])
        selected: set[int] = set(reserved)

        # 2. Shifted-site representation slot, if capacity remains.
        remaining = self.capacity - len(selected)
        if remaining > 0:
            shifted_candidates = [
                s for s in candidates if s not in selected and s in posteriors and is_likely_shifted(posteriors[s])
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
        # Report only sites that were *already* overdue and got in as "mandatory" --
        # a reservation can also admit a site slightly ahead of its deadline as part
        # of keeping the schedule feasible; that's not the same as being overdue.
        mandatory = overdue_set & selected_frozen

        assignments: dict[int, ScheduleMode] = {}
        deferral_reasons: dict[int, str] = {}
        for s in candidates:
            if s in selected_frozen:
                full_capable = s in posteriors and is_full_capable(posteriors[s])
                assignments[s] = ScheduleMode.FULL if full_capable else ScheduleMode.COMPRESSED
            else:
                assignments[s] = ScheduleMode.DEFERRED
                deferral_reasons[s] = (
                    "coverage: overdue but capacity exceeded" if s in overdue_set else "not selected: score"
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

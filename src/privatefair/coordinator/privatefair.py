"""The PrivateFair coordinator (task B5): posterior score, coverage, shifted-site
slot, and full/compressed/deferred scheduling (report section 4 "Privacy mechanism
and decision rule"; section 7.2 step 4, "Constrained selection and scheduling").

Each epoch's `select()`:
  1. Decode every available site's TelemetryReport into a Posterior (B2).
  2. Reserve capacity for eligible overdue sites (participation age >= max_age),
     oldest-first and capped at capacity -- the same coverage rule B4's
     CoverageRandomPolicy uses, so the two baselines stay comparable.
  3. If capacity remains, reserve one slot for the best "likely shifted" site
     available (its posterior's single most probable shift bin is the top one --
     a MAP call, not a raw-probability threshold, so the decision is easy to
     audit and log).
  4. Fill any remaining capacity by descending posterior score:
         score = alpha * risk_hat + beta * p_shift_top + gamma * age + delta * readiness_hat
     risk_hat and readiness_hat are the posterior's normalized expected bin index
     (E[bin] / (K-1), in [0, 1]); p_shift_top is Pr(shift = the top bin).
  5. Every selected site is assigned FULL if its posterior's most likely readiness
     bin is the fastest one, else COMPRESSED -- using only the already-computed
     MAP bin, never a raw telemetry value.

This module only ever sees privatized TelemetryReport payloads and its own
participation-age state -- never TrueBins -- so it respects the privacy boundary
tested in tests/test_boundaries.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from privatefair.coordinator.participation import ParticipationTracker
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


def is_likely_full_speed(posterior: Posterior) -> bool:
    """True if the fastest readiness bin is the posterior's single most probable bin (MAP)."""
    return int(np.argmax(posterior.p_readiness)) == len(posterior.p_readiness) - 1


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

        # 1. Mandatory coverage: oldest overdue sites first, capped at capacity.
        overdue_order = sorted(overdue_set, key=lambda s: (-ages[s], -tiebreak[s]))
        mandatory = frozenset(overdue_order[: self.capacity])
        selected: set[int] = set(mandatory)

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
        mandatory = mandatory & selected_frozen  # report only what actually got in as "mandatory"

        assignments: dict[int, ScheduleMode] = {}
        deferral_reasons: dict[int, str] = {}
        for s in candidates:
            if s in selected_frozen:
                full_speed = s in posteriors and is_likely_full_speed(posteriors[s])
                assignments[s] = ScheduleMode.FULL if full_speed else ScheduleMode.COMPRESSED
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

"""RandomPolicy and CoverageRandomPolicy (task B4): baseline Coordinator implementations.

Both are comparators from report section 7.3 / 10.E: "random FedAvg (no adaptive
telemetry)" and "coverage-constrained random selection (isolates the cost/benefit of
fairness)". Neither uses the telemetry reports at all -- posterior/risk-aware
selection is B5's job -- so `reports` is accepted (to satisfy the `Coordinator`
protocol) but unused here.

Both only ever see `available` site ids and their own participation-age state --
never TrueBins -- so they respect the privacy boundary tested in
tests/test_boundaries.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from privatefair.coordinator.participation import ParticipationTracker, required_now
from privatefair.interfaces import CohortDecision, ScheduleMode, TelemetryReport


def _choose(rng: np.random.Generator, candidates: list[int], n: int) -> frozenset[int]:
    """Uniformly choose up to n site ids from candidates, without replacement."""
    if n <= 0 or not candidates:
        return frozenset()
    n = min(n, len(candidates))
    return frozenset(int(s) for s in rng.choice(candidates, size=n, replace=False))


@dataclass
class RandomPolicy:
    """No telemetry, no coverage: pick up to `capacity` available sites uniformly at random."""

    capacity: int
    rng: np.random.Generator
    name: str = "random"
    tracker: ParticipationTracker = field(default_factory=ParticipationTracker)

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {self.capacity}")

    def select(self, epoch: int, reports: Mapping[int, TelemetryReport], available: frozenset[int]) -> CohortDecision:
        candidates = sorted(available)
        selected = _choose(self.rng, candidates, self.capacity)
        assignments = {s: (ScheduleMode.FULL if s in selected else ScheduleMode.DEFERRED) for s in candidates}
        deferral_reasons = {s: "not selected: capacity" for s in candidates if s not in selected}
        return CohortDecision(epoch=epoch, policy=self.name, assignments=assignments, deferral_reasons=deferral_reasons)

    def observe(self, decision: CohortDecision, completed: frozenset[int]) -> None:
        if not completed <= decision.selected:
            raise ValueError(
                f"observe() got completed sites not in the decision's selection: "
                f"{sorted(completed - decision.selected)}"
            )
        self.tracker.update(decision.epoch, completed)


@dataclass
class CoverageRandomPolicy:
    """Coverage-constrained random selection.

    Each round reserves `min(required_now(...), capacity)` sites -- the oldest
    ones, by participation age -- using the shared earliest-deadline-first
    feasibility check in `coordinator.participation.required_now`. That
    reservation is what keeps the coverage bound tight: forcing in only the
    sites already at `age >= max_age` is not enough, because several sites can
    then cross that threshold together in a later round with no room left for
    all of them. `required_now` looks ahead across every site's deadline, not
    just this round's, and returns the minimum that must be admitted now to
    avoid that.

    Whatever capacity is left after the reservation is filled uniformly at
    random -- that's the "random" this policy is named for, and the reservation
    is deliberately the *minimum* needed so as much capacity as possible stays
    random rather than being swallowed by a rigid oldest-first rotation.

    `mandatory_sites` on the returned decision reports only the sites that were
    *already* at `age >= max_age` and got selected -- not every reserved site,
    since a reservation can also admit a site slightly before its deadline as
    part of keeping the schedule feasible. If more already-overdue sites exist
    than capacity allows, the remainder stays deferred with a reason that says
    so explicitly -- capacity, not the policy, is what's limiting in that case.
    """

    capacity: int
    max_age: int
    rng: np.random.Generator
    name: str = "coverage_random"
    tracker: ParticipationTracker = field(default_factory=ParticipationTracker)

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError(f"capacity must be >= 0, got {self.capacity}")
        if self.max_age < 0:
            raise ValueError(f"max_age must be >= 0, got {self.max_age}")

    def select(self, epoch: int, reports: Mapping[int, TelemetryReport], available: frozenset[int]) -> CohortDecision:
        candidates = sorted(available)
        ages = self.tracker.ages(candidates, epoch)
        overdue_set = frozenset(s for s in candidates if ages[s] >= self.max_age)
        tiebreak = {s: self.rng.random() for s in candidates}

        need = min(required_now(ages.values(), self.capacity, self.max_age), self.capacity)
        order = sorted(candidates, key=lambda s: (-ages[s], -tiebreak[s]))
        reserved = frozenset(order[:need])

        remaining = self.capacity - len(reserved)
        rest = [s for s in candidates if s not in reserved]
        extra = _choose(self.rng, rest, remaining)
        selected = reserved | extra
        mandatory = overdue_set & selected

        assignments: dict[int, ScheduleMode] = {}
        deferral_reasons: dict[int, str] = {}
        for s in candidates:
            if s in selected:
                assignments[s] = ScheduleMode.FULL
            else:
                assignments[s] = ScheduleMode.DEFERRED
                deferral_reasons[s] = (
                    "coverage: overdue but capacity exceeded" if s in overdue_set else "not selected: capacity"
                )

        return CohortDecision(
            epoch=epoch,
            policy=self.name,
            assignments=assignments,
            mandatory_sites=mandatory,
            deferral_reasons=deferral_reasons,
        )

    def observe(self, decision: CohortDecision, completed: frozenset[int]) -> None:
        if not completed <= decision.selected:
            raise ValueError(
                f"observe() got completed sites not in the decision's selection: "
                f"{sorted(completed - decision.selected)}"
            )
        self.tracker.update(decision.epoch, completed)

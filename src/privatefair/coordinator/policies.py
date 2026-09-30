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

from privatefair.coordinator.participation import ParticipationTracker
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
        self.tracker.update(decision.epoch, completed)


@dataclass
class CoverageRandomPolicy:
    """Coverage-constrained random selection.

    Sites whose participation age has reached `max_age` are "eligible overdue" and
    are forced into the cohort before anything else; remaining capacity is filled
    uniformly at random from the rest. If more sites are overdue than there is
    capacity for, the most-overdue sites (ties broken by site_id) are prioritized
    and the remainder stays deferred with a reason that says so explicitly --
    capacity, not the policy, is what's limiting in that case.
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
        ages = {s: self.tracker.age(s, epoch) for s in candidates}
        overdue = sorted((s for s in candidates if ages[s] >= self.max_age), key=lambda s: (-ages[s], s))
        overdue_set = frozenset(overdue)
        mandatory = frozenset(overdue[: self.capacity])

        remaining_capacity = self.capacity - len(mandatory)
        rest = [s for s in candidates if s not in mandatory]
        extra = _choose(self.rng, rest, remaining_capacity)
        selected = mandatory | extra

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
        self.tracker.update(decision.epoch, completed)

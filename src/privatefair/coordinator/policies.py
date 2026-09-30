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
        if not completed <= decision.selected:
            raise ValueError(
                f"observe() got completed sites not in the decision's selection: "
                f"{sorted(completed - decision.selected)}"
            )
        self.tracker.update(decision.epoch, completed)


@dataclass
class CoverageRandomPolicy:
    """Coverage-constrained random selection.

    Spare capacity is filled oldest-first (by participation age), not uniformly at
    random. That's the difference that keeps the coverage bound tight: filling
    spare slots at random lets several sites drift toward `max_age` together
    without ever being prioritized over each other, so they can all become
    overdue in the same round -- more than capacity can then admit at once, and
    the bound is blown. Always giving spare capacity to the currently-oldest
    sites (age ties broken by the rng, not site_id) turns this into an
    oldest-first / round-robin schedule, which is what actually keeps every
    site's age bounded as long as `capacity * max_age >= number of sites` (the
    slack a coverage-constrained schedule needs to visit everyone in time).

    Sites whose age has *already* reached `max_age` are still tracked separately
    as `mandatory_sites` on the returned decision (informational: which sites the
    coverage rule was actively forcing in, versus which were merely next in
    line). If more sites are overdue than there is capacity for, the remainder
    stays deferred with a reason that says so explicitly -- capacity, not the
    policy, is what's limiting in that case.
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
        overdue_set = frozenset(s for s in candidates if ages[s] >= self.max_age)

        # Oldest-first fill: sort every candidate by age descending, ties broken
        # by an rng draw (not site_id), and give capacity to the top of that
        # order. This always prioritizes whoever is closest to going overdue,
        # not just those already there.
        tiebreak = {s: self.rng.random() for s in candidates}
        order = sorted(candidates, key=lambda s: (-ages[s], -tiebreak[s]))
        selected = frozenset(order[: self.capacity])
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

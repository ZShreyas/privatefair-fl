"""Tests for RandomPolicy and CoverageRandomPolicy (task B4).

Gate G2 (TEAM_PLAN.md): max participation age is provably bounded under a
controlled trace -- test_gate_g2_max_age_never_exceeds_max_age below, and
test_gate_g2_bound_holds_across_many_seeds_and_settings for the broader claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from privatefair.coordinator.policies import CoverageRandomPolicy, RandomPolicy
from privatefair.interfaces import ScheduleMode

# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------


def test_random_policy_selects_at_most_capacity():
    policy = RandomPolicy(capacity=2, rng=np.random.default_rng(0))
    decision = policy.select(epoch=0, reports={}, available=frozenset(range(5)))
    assert len(decision.selected) == 2
    assert set(decision.assignments) == set(range(5))
    assert all(mode in (ScheduleMode.FULL, ScheduleMode.DEFERRED) for mode in decision.assignments.values())


def test_random_policy_selects_everyone_when_capacity_exceeds_available():
    policy = RandomPolicy(capacity=10, rng=np.random.default_rng(0))
    decision = policy.select(epoch=0, reports={}, available=frozenset({1, 2, 3}))
    assert decision.selected == {1, 2, 3}
    assert decision.deferral_reasons == {}


def test_random_policy_handles_empty_available():
    policy = RandomPolicy(capacity=3, rng=np.random.default_rng(0))
    decision = policy.select(epoch=0, reports={}, available=frozenset())
    assert decision.assignments == {}
    assert decision.selected == frozenset()


def test_random_policy_deferred_sites_have_reasons():
    policy = RandomPolicy(capacity=1, rng=np.random.default_rng(0))
    decision = policy.select(epoch=0, reports={}, available=frozenset({1, 2, 3}))
    deferred = {s for s, m in decision.assignments.items() if m is ScheduleMode.DEFERRED}
    assert deferred == set(decision.deferral_reasons)
    assert len(deferred) == 2


def test_random_policy_rejects_negative_capacity():
    with pytest.raises(ValueError):
        RandomPolicy(capacity=-1, rng=np.random.default_rng(0))


def test_random_policy_observe_updates_tracker():
    policy = RandomPolicy(capacity=2, rng=np.random.default_rng(0))
    decision = policy.select(epoch=0, reports={}, available=frozenset({1, 2, 3}))
    completed_site = next(iter(decision.selected))
    not_completed_site = next(iter(frozenset({1, 2, 3}) - decision.selected))
    policy.observe(decision, completed=frozenset({completed_site}))
    assert policy.tracker.age(completed_site, epoch=1) == 1
    assert policy.tracker.age(not_completed_site, epoch=1) == 2  # never completed


def test_observe_rejects_completed_sites_not_selected():
    policy = RandomPolicy(capacity=1, rng=np.random.default_rng(0))
    decision = policy.select(epoch=0, reports={}, available=frozenset({1, 2, 3}))
    not_selected = next(iter(frozenset({1, 2, 3}) - decision.selected))
    with pytest.raises(ValueError):
        policy.observe(decision, completed=frozenset({not_selected}))


# ---------------------------------------------------------------------------
# CoverageRandomPolicy
# ---------------------------------------------------------------------------


def test_coverage_policy_forces_in_overdue_sites():
    policy = CoverageRandomPolicy(capacity=1, max_age=2, rng=np.random.default_rng(0))
    # site 9 has never participated; by epoch 1 its age is 2 == max_age -> mandatory
    decision = policy.select(epoch=1, reports={}, available=frozenset({9}))
    assert decision.mandatory_sites == {9}
    assert decision.selected == {9}


def test_coverage_policy_does_not_force_sites_below_max_age():
    policy = CoverageRandomPolicy(capacity=0, max_age=5, rng=np.random.default_rng(0))
    decision = policy.select(epoch=0, reports={}, available=frozenset({1}))
    assert decision.mandatory_sites == frozenset()
    assert decision.selected == frozenset()
    assert decision.deferral_reasons[1] == "not selected: capacity"


def test_coverage_policy_capacity_exceeded_by_overdue_sites():
    # 3 sites, all overdue at epoch 0 (age 1 >= max_age 1), but capacity only 2.
    policy = CoverageRandomPolicy(capacity=2, max_age=1, rng=np.random.default_rng(0))
    decision = policy.select(epoch=0, reports={}, available=frozenset({1, 2, 3}))
    assert len(decision.mandatory_sites) == 2
    assert len(decision.selected) == 2
    leftover = (set(decision.assignments) - decision.selected).pop()
    assert decision.deferral_reasons[leftover] == "coverage: overdue but capacity exceeded"


def test_coverage_policy_rejects_bad_params():
    with pytest.raises(ValueError):
        CoverageRandomPolicy(capacity=-1, max_age=1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        CoverageRandomPolicy(capacity=1, max_age=-1, rng=np.random.default_rng(0))


def test_coverage_policy_mandatory_is_subset_of_available():
    # site 1 is not part of this round's `available` set at all, so it cannot
    # appear anywhere in the decision -- mandatory selection can only ever draw
    # from the sites the coordinator was actually given.
    policy = CoverageRandomPolicy(capacity=1, max_age=1, rng=np.random.default_rng(0))
    decision = policy.select(epoch=5, reports={}, available=frozenset({2}))
    assert 1 not in decision.assignments
    assert decision.mandatory_sites <= frozenset({2})
    assert set(decision.assignments) == {2}


def test_gate_g2_max_age_never_exceeds_max_age():
    """Controlled trace: N=6 always-available sites, capacity=3, max_age=2.

    At every epoch, the age computed for every available site (before that
    epoch's selection) must be <= max_age. This is the coverage rule's job: any
    site whose age reaches max_age is mandatory, and here capacity (3) always
    covers the number of simultaneously-overdue sites, so nothing is ever
    deferred past its due date.
    """
    n_sites, capacity, max_age, n_epochs = 6, 3, 2, 25
    policy = CoverageRandomPolicy(capacity=capacity, max_age=max_age, rng=np.random.default_rng(42))
    available = frozenset(range(n_sites))

    max_age_seen = 0
    for epoch in range(n_epochs):
        ages_this_round = {s: policy.tracker.age(s, epoch) for s in available}
        max_age_seen = max(max_age_seen, max(ages_this_round.values()))

        decision = policy.select(epoch=epoch, reports={}, available=available)
        policy.observe(decision, completed=decision.selected)

    assert max_age_seen <= max_age


def test_gate_g2_holds_with_intermittent_availability():
    """Sites that are sometimes unavailable must still not exceed max_age once
    they return, and their unavailable epochs don't count as a policy failure
    (the coordinator cannot force in a site that isn't there)."""
    max_age = 3
    policy = CoverageRandomPolicy(capacity=2, max_age=max_age, rng=np.random.default_rng(1))
    all_sites = frozenset(range(5))

    for epoch in range(20):
        # site 4 is only available every 5th epoch
        available = all_sites if epoch % 5 != 4 else all_sites - {4}
        ages = {s: policy.tracker.age(s, epoch) for s in available}
        assert max(ages.values()) <= max_age
        decision = policy.select(epoch=epoch, reports={}, available=available)
        policy.observe(decision, completed=decision.selected)


def _simulate_worst_age(n_sites: int, capacity: int, max_age: int, seed: int, n_epochs: int = 60) -> int:
    """Run a fixed-N, always-available trace and return the largest age ever
    observed at the start of any epoch (i.e. before that epoch's selection)."""
    policy = CoverageRandomPolicy(capacity=capacity, max_age=max_age, rng=np.random.default_rng(seed))
    available = frozenset(range(n_sites))
    worst = 0
    for epoch in range(n_epochs):
        worst = max(worst, max(policy.tracker.age(s, epoch) for s in available))
        decision = policy.select(epoch=epoch, reports={}, available=available)
        policy.observe(decision, completed=decision.selected)
    return worst


# capacity * max_age >= n_sites in every case here: the slack a coverage
# schedule needs to be able to visit every site within max_age epochs.
# (9, 3, 3) is the tight case originally reported as failing 144/200 seeds
# under uniform-random spare-capacity fill.
FEASIBLE_SETTINGS = [
    (6, 3, 2),  # tight
    (9, 3, 3),  # tight -- the reported failing case
    (8, 2, 4),  # tight
    (5, 2, 3),  # slack
    (11, 4, 3),  # slack
]


@pytest.mark.parametrize("n_sites,capacity,max_age", FEASIBLE_SETTINGS)
def test_gate_g2_bound_holds_across_many_seeds_and_settings(n_sites, capacity, max_age):
    assert capacity * max_age >= n_sites, "test setting must be feasible for any policy to honor max_age"
    violations = [
        (seed, worst)
        for seed in range(200)
        if (worst := _simulate_worst_age(n_sites, capacity, max_age, seed)) > max_age
    ]
    assert not violations, (
        f"n={n_sites} cap={capacity} max_age={max_age}: {len(violations)}/200 seeds exceeded the bound "
        f"(worst cases: {violations[:5]})"
    )


def test_gate_g2_degrades_gracefully_when_infeasible():
    """When capacity*max_age < n_sites, no policy can honor max_age for every
    site -- there just isn't enough capacity to visit everyone that often. The
    oldest-first fill should still cap age at the round-robin bound
    ceil(n_sites/capacity) rather than letting some site's age grow without
    bound while others are repeatedly reselected.
    """
    n_sites, capacity, max_age = 10, 3, 3
    assert capacity * max_age < n_sites  # this setting is infeasible for max_age itself
    round_robin_bound = math.ceil(n_sites / capacity)

    for seed in range(200):
        worst = _simulate_worst_age(n_sites, capacity, max_age, seed)
        assert worst <= round_robin_bound, (
            f"seed={seed}: worst age {worst} exceeded the round-robin bound {round_robin_bound} "
            "even though max_age itself can't be honored here"
        )

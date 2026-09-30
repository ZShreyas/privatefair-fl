"""Tests for the PrivateFair coordinator (task B5).

Done-when (TEAM_PLAN.md): "Selection responds correctly to controlled
risk/readiness/shift perturbations." The scenario tests below construct sites
that differ in exactly one signal and check the coordinator reacts the way
report section 4's score/constraints say it should.

Also covers the coverage bound: test_coverage_bound_holds_across_many_seeds_and_settings
mirrors test_policies.py's Gate G2 sweep, using real (random) telemetry reports
each round, since select()'s reservation logic must hold regardless of what the
score/shifted-slot stages do with the rest of the capacity.
"""

from __future__ import annotations

import numpy as np
import pytest

from privatefair.coordinator.privatefair import (
    PrivateFairCoordinator,
    is_full_capable,
    is_likely_shifted,
    p_shift_top,
    readiness_hat,
    risk_hat,
)
from privatefair.interfaces import K_READINESS, K_SHIFT, K_UTILITY, Posterior, ScheduleMode, TelemetryReport

EPS_CERTAIN = 50.0  # large epsilon -> posterior is (numerically) one-hot at the reported bin


def _report(site_id, epoch, utility, readiness, shift):
    return TelemetryReport(site_id=site_id, epoch=epoch, utility=utility, readiness=readiness, shift=shift)


def _coordinator(**kwargs):
    kwargs.setdefault("rng", np.random.default_rng(0))
    kwargs.setdefault("epsilon", EPS_CERTAIN)
    return PrivateFairCoordinator(**kwargs)


# ---------------------------------------------------------------------------
# Scoring helpers -- hand-computed
# ---------------------------------------------------------------------------


def test_risk_readiness_shift_helpers_on_one_hot_posteriors():
    # one-hot at the worst utility bin, the slowest readiness bin, the shifted-domain bin
    posterior = Posterior(
        site_id=0,
        epoch=0,
        p_utility=tuple(1.0 if i == K_UTILITY - 1 else 0.0 for i in range(K_UTILITY)),
        p_readiness=tuple(1.0 if i == 0 else 0.0 for i in range(K_READINESS)),
        p_shift=tuple(1.0 if i == K_SHIFT - 1 else 0.0 for i in range(K_SHIFT)),
    )
    assert risk_hat(posterior) == pytest.approx(1.0)  # worst bin -> normalized 1.0
    assert readiness_hat(posterior) == pytest.approx(0.0)  # slowest bin -> normalized 0.0
    assert p_shift_top(posterior) == pytest.approx(1.0)
    assert is_likely_shifted(posterior) is True
    assert is_full_capable(posterior) is False


def test_score_hand_computed():
    # utility=4 (worst, K=5) -> risk_hat = 4/4 = 1.0
    # readiness=0 (slowest, K=4) -> readiness_hat = 0/3 = 0.0
    # shift=2 (shifted, K=3) -> p_shift_top = 1.0 (one-hot)
    posterior = Posterior(
        site_id=0,
        epoch=0,
        p_utility=(0.0, 0.0, 0.0, 0.0, 1.0),
        p_readiness=(1.0, 0.0, 0.0, 0.0),
        p_shift=(0.0, 0.0, 1.0),
    )
    coord = _coordinator(capacity=1, max_age=100, alpha=1.0, beta=1.0, gamma=0.1, delta=1.0)
    # score = 1*1.0 + 1*1.0 + 0.1*5 + 1*0.0 = 2.5
    assert coord.score(posterior, age=5) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# select(): risk perturbation
# ---------------------------------------------------------------------------


def test_higher_risk_site_preferred_under_tight_capacity():
    coord = _coordinator(capacity=1, max_age=1000)  # coverage never triggers here
    reports = {
        1: _report(1, 0, utility=0, readiness=1, shift=1),  # strongly improving -> low risk
        2: _report(2, 0, utility=4, readiness=1, shift=1),  # strongly deteriorating -> high risk
    }
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1, 2}))
    assert decision.selected == {2}
    assert decision.scores[2] > decision.scores[1]


def test_score_increases_monotonically_with_reported_risk():
    coord = _coordinator(capacity=5, max_age=1000)
    scores = []
    for utility in range(K_UTILITY):
        reports = {1: _report(1, 0, utility=utility, readiness=0, shift=0)}
        decision = coord.select(epoch=0, reports=reports, available=frozenset({1}))
        scores.append(decision.scores[1])
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


# ---------------------------------------------------------------------------
# select(): shift perturbation / representation slot
# ---------------------------------------------------------------------------


def test_shifted_site_gets_reserved_slot_even_with_lower_score():
    # capacity=2. Site A (1): high risk, best raw score, not shifted. Site B (2):
    # shifted, but low risk/readiness -- worst raw score. Site C (3): moderate
    # score, not shifted -- would normally take the 2nd slot by score alone, but
    # the shifted-site slot should give it to B instead.
    coord = _coordinator(capacity=2, max_age=1000, alpha=1.0, beta=1.0, gamma=0.0, delta=1.0)
    reports = {
        1: _report(1, 0, utility=4, readiness=3, shift=0),  # A: best score, not shifted
        2: _report(2, 0, utility=0, readiness=0, shift=2),  # B: shifted, worst score
        3: _report(3, 0, utility=2, readiness=2, shift=0),  # C: middling score, not shifted
    }
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1, 2, 3}))

    assert decision.scores[3] > decision.scores[2]  # confirms B (2) would lose to C (3) on raw score
    assert decision.selected == {1, 2}  # shifted slot gives B the spot C would otherwise have taken


def test_no_shifted_slot_reserved_when_no_site_is_likely_shifted():
    coord = _coordinator(capacity=1, max_age=1000)
    reports = {
        1: _report(1, 0, utility=1, readiness=1, shift=0),  # typical domain
        2: _report(2, 0, utility=3, readiness=1, shift=0),  # typical domain, higher risk
    }
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1, 2}))
    assert decision.selected == {2}  # pure score fill, no shifted candidate to reserve for


# ---------------------------------------------------------------------------
# select(): readiness perturbation / mode assignment
# ---------------------------------------------------------------------------


def test_readiness_bin_2_and_above_gets_full_below_gets_compressed():
    # Per A5's TrueBins definition (issue #37), readiness bin 2 means "finishes a
    # full round within the deadline" -- so bins 2 and 3 both mean FULL, not only
    # the single fastest bin (3).
    coord = _coordinator(capacity=3, max_age=1000)
    reports = {
        1: _report(1, 0, utility=2, readiness=K_READINESS - 1, shift=1),  # bin 3: fastest
        2: _report(2, 0, utility=2, readiness=2, shift=1),  # bin 2: full-capable, not fastest
        3: _report(3, 0, utility=2, readiness=1, shift=1),  # bin 1: not full-capable
    }
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1, 2, 3}))
    assert decision.selected == {1, 2, 3}
    assert decision.assignments[1] == ScheduleMode.FULL
    assert decision.assignments[2] == ScheduleMode.FULL
    assert decision.assignments[3] == ScheduleMode.COMPRESSED


# ---------------------------------------------------------------------------
# select(): coverage / mandatory sites
# ---------------------------------------------------------------------------


def test_overdue_site_is_forced_in_despite_low_score():
    # max_age=3: site 2 (last participated epoch 0, age 2 at epoch 2) is NOT yet
    # overdue; site 1 (never participated, age 3 at epoch 2) is. Only site 1
    # should be forced in, even though its score is the worst possible.
    coord = _coordinator(capacity=1, max_age=3, alpha=1.0, beta=1.0, gamma=0.0, delta=1.0)
    coord.tracker.update(epoch=0, completed=frozenset({2}))
    reports = {
        1: _report(1, 2, utility=0, readiness=0, shift=0),  # worst possible score, but overdue
        2: _report(2, 2, utility=4, readiness=3, shift=2),  # best possible score, not overdue
    }
    decision = coord.select(epoch=2, reports=reports, available=frozenset({1, 2}))
    assert decision.mandatory_sites == {1}
    assert decision.selected == {1}
    assert decision.deferral_reasons[2] == "not selected: score"


def test_overdue_but_capacity_exceeded_gets_explicit_reason():
    coord = _coordinator(capacity=1, max_age=0)  # everyone is immediately overdue
    reports = {1: _report(1, 0, 0, 0, 0), 2: _report(2, 0, 0, 0, 0)}
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1, 2}))
    assert len(decision.selected) == 1
    leftover = next(iter(frozenset({1, 2}) - decision.selected))
    assert decision.deferral_reasons[leftover] == "coverage: overdue but capacity exceeded"


def _random_report(rng: np.random.Generator, site_id: int, epoch: int) -> TelemetryReport:
    return TelemetryReport(
        site_id=site_id,
        epoch=epoch,
        utility=int(rng.integers(0, K_UTILITY)),
        readiness=int(rng.integers(0, K_READINESS)),
        shift=int(rng.integers(0, K_SHIFT)),
    )


def _simulate_worst_age(n_sites: int, capacity: int, max_age: int, seed: int, n_epochs: int = 60) -> int:
    """Run an always-available trace with random telemetry each round and return
    the largest age ever observed at the start of any epoch (before that
    epoch's selection). Telemetry is randomized -- unlike CoverageRandomPolicy,
    PrivateFairCoordinator's score and shifted-slot stages depend on report
    content, so the coverage bound has to hold no matter what those stages pick
    for the non-reserved capacity.
    """
    coord = PrivateFairCoordinator(capacity=capacity, max_age=max_age, rng=np.random.default_rng(seed), epsilon=1.0)
    report_rng = np.random.default_rng(seed + 1_000_003)  # independent stream for telemetry content
    available = frozenset(range(n_sites))
    worst = 0
    for epoch in range(n_epochs):
        worst = max(worst, max(coord.tracker.age(s, epoch) for s in available))
        reports = {s: _random_report(report_rng, s, epoch) for s in available}
        decision = coord.select(epoch=epoch, reports=reports, available=available)
        coord.observe(decision, completed=decision.selected)
    return worst


# capacity * max_age >= n_sites in every case: includes the exact settings
# reported as failing before the fix (9,3,3), (8,2,4), (12,4,3).
COVERAGE_FEASIBLE_SETTINGS = [
    (9, 3, 3),
    (8, 2, 4),
    (12, 4, 3),
    (6, 3, 2),
    (5, 2, 3),
]


@pytest.mark.parametrize("n_sites,capacity,max_age", COVERAGE_FEASIBLE_SETTINGS)
def test_coverage_bound_holds_across_many_seeds_and_settings(n_sites, capacity, max_age):
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


# ---------------------------------------------------------------------------
# B6 ablations: coverage=False, drop_signal
# ---------------------------------------------------------------------------


def test_no_coverage_ablation_does_not_force_in_overdue_sites():
    # Same setup as test_overdue_site_is_forced_in_despite_low_score, but with
    # coverage disabled: site 1 is overdue (age 3 >= max_age 3) yet has the
    # worst possible score, so without the coverage rule it should lose to
    # site 2 on score instead of being forced in.
    coord = _coordinator(capacity=1, max_age=3, coverage=False, alpha=1.0, beta=1.0, gamma=0.0, delta=1.0)
    coord.tracker.update(epoch=0, completed=frozenset({2}))
    reports = {
        1: _report(1, 2, utility=0, readiness=0, shift=0),  # overdue, worst score
        2: _report(2, 2, utility=4, readiness=3, shift=2),  # not overdue, best score
    }
    decision = coord.select(epoch=2, reports=reports, available=frozenset({1, 2}))
    assert decision.selected == {2}
    assert decision.mandatory_sites == frozenset()  # coverage disabled -> nothing is ever "mandatory"
    assert decision.deferral_reasons[1] == "not selected: score"  # not the "coverage exceeded" wording


def _one_hot_posterior(utility, readiness, shift):
    return Posterior(
        site_id=0,
        epoch=0,
        p_utility=tuple(1.0 if i == utility else 0.0 for i in range(K_UTILITY)),
        p_readiness=tuple(1.0 if i == readiness else 0.0 for i in range(K_READINESS)),
        p_shift=tuple(1.0 if i == shift else 0.0 for i in range(K_SHIFT)),
    )


def test_drop_signal_utility_zeroes_risk_contribution():
    coord = _coordinator(capacity=1, max_age=1000, drop_signal="utility")
    low_risk = coord.score(_one_hot_posterior(utility=0, readiness=0, shift=0), age=0)
    high_risk = coord.score(_one_hot_posterior(utility=4, readiness=0, shift=0), age=0)
    assert low_risk == pytest.approx(high_risk)  # utility no longer moves the score


def test_drop_signal_shift_disables_shifted_slot():
    # Same setup as test_shifted_site_gets_reserved_slot_even_with_lower_score,
    # but with the shift signal dropped: the shifted site (2) should lose its
    # reserved slot and pure score ordering should decide instead.
    coord = _coordinator(capacity=2, max_age=1000, drop_signal="shift", alpha=1.0, beta=1.0, gamma=0.0, delta=1.0)
    reports = {
        1: _report(1, 0, utility=4, readiness=3, shift=0),  # best score, not shifted
        2: _report(2, 0, utility=0, readiness=0, shift=2),  # shifted, worst score
        3: _report(3, 0, utility=2, readiness=2, shift=0),  # middling score, not shifted
    }
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1, 2, 3}))
    assert decision.selected == {1, 3}  # C (3) wins the 2nd slot on score now that shift can't reserve it


def test_drop_signal_readiness_defaults_to_full_regardless_of_reported_bin():
    # No readiness info to act on -> assume FULL optimistically, even for a
    # site that reported the *slowest* bin. Forcing COMPRESSED here would
    # itself cut training compute regardless of whether it was warranted,
    # confounding "lost information" with "lost compute". A genuinely slow
    # site's real cost belongs in the systems simulation (deadline misses),
    # not here.
    coord = _coordinator(capacity=1, max_age=1000, drop_signal="readiness")
    reports = {1: _report(1, 0, utility=2, readiness=0, shift=1)}  # slowest readiness bin
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1}))
    assert decision.selected == {1}
    assert decision.assignments[1] == ScheduleMode.FULL


def test_drop_signal_rejects_invalid_value():
    with pytest.raises(ValueError):
        _coordinator(capacity=1, max_age=1, drop_signal="not_a_signal")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_missing_report_site_is_score_zero_and_compressed_if_selected():
    coord = _coordinator(capacity=2, max_age=1000)
    reports = {1: _report(1, 0, utility=0, readiness=0, shift=0)}  # site 2 sent nothing
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1, 2}))
    assert decision.scores[2] == 0.0
    assert decision.selected == {1, 2}  # capacity covers both
    assert decision.assignments[2] == ScheduleMode.COMPRESSED  # unknown readiness -> never FULL


def test_missing_report_site_can_still_be_mandatory():
    # site 1 recently participated (age 5 at epoch 5, below max_age); site 2 has
    # never participated (age 6, overdue) and sends no report this round -- it
    # should still be forced in on age alone.
    coord = _coordinator(capacity=1, max_age=6)
    coord.tracker.update(epoch=0, completed=frozenset({1}))
    reports = {1: _report(1, 5, utility=2, readiness=2, shift=1)}  # site 2 sends nothing
    decision = coord.select(epoch=5, reports=reports, available=frozenset({1, 2}))
    assert decision.mandatory_sites == {2}
    assert decision.selected == {2}


def test_zero_capacity_defers_everyone():
    coord = _coordinator(capacity=0, max_age=1000)
    reports = {1: _report(1, 0, 0, 0, 0)}
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1}))
    assert decision.selected == frozenset()
    assert decision.deferral_reasons[1] == "not selected: score"


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        _coordinator(capacity=-1, max_age=1)
    with pytest.raises(ValueError):
        _coordinator(capacity=1, max_age=-1)


def test_observe_updates_tracker_and_rejects_non_selected_completed():
    coord = _coordinator(capacity=1, max_age=1000)
    reports = {1: _report(1, 0, 0, 0, 0), 2: _report(2, 0, 0, 0, 0)}
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1, 2}))
    coord.observe(decision, completed=decision.selected)
    (selected_site,) = decision.selected
    assert coord.tracker.age(selected_site, epoch=1) == 1

    not_selected = next(iter(frozenset({1, 2}) - decision.selected))
    with pytest.raises(ValueError):
        coord.observe(decision, completed=frozenset({not_selected}))

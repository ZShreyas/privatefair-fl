"""Tests for the PrivateFair coordinator (task B5).

Done-when (TEAM_PLAN.md): "Selection responds correctly to controlled
risk/readiness/shift perturbations." The scenario tests below construct sites
that differ in exactly one signal and check the coordinator reacts the way
report section 4's score/constraints say it should.
"""

from __future__ import annotations

import numpy as np
import pytest

from privatefair.coordinator.privatefair import (
    PrivateFairCoordinator,
    is_likely_full_speed,
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
    assert is_likely_full_speed(posterior) is False


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


def test_fastest_readiness_gets_full_others_get_compressed():
    coord = _coordinator(capacity=2, max_age=1000)
    reports = {
        1: _report(1, 0, utility=2, readiness=K_READINESS - 1, shift=1),  # fastest
        2: _report(2, 0, utility=2, readiness=1, shift=1),  # not fastest
    }
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1, 2}))
    assert decision.selected == {1, 2}
    assert decision.assignments[1] == ScheduleMode.FULL
    assert decision.assignments[2] == ScheduleMode.COMPRESSED


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

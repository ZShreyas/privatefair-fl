"""Tests for RawOracleCoordinator (task B6, report section 10.E #3)."""

from __future__ import annotations

import numpy as np
import pytest

from privatefair.baselines.oracle import RawOracleCoordinator
from privatefair.coordinator.privatefair import is_full_capable, is_likely_shifted, risk_hat
from privatefair.interfaces import ScheduleMode, TrueBins


def _true(site_id, epoch, utility, readiness, shift):
    return TrueBins(site_id=site_id, epoch=epoch, utility=utility, readiness=readiness, shift=shift)


def _coordinator(**kwargs):
    kwargs.setdefault("rng", np.random.default_rng(0))
    kwargs.setdefault("epsilon", 0.1)  # unused by this coordinator; present for API symmetry
    return RawOracleCoordinator(**kwargs)


def test_name_defaults_to_raw_oracle():
    coord = _coordinator(capacity=1, max_age=100)
    assert coord.name == "raw_oracle"


def test_select_before_observe_truth_treats_every_site_as_missing():
    # No observe_truth call yet -> no ground truth cached -> same "missing report"
    # fallback as PrivateFairCoordinator: score 0, COMPRESSED if selected.
    coord = _coordinator(capacity=1, max_age=1000)
    decision = coord.select(epoch=0, reports={}, available=frozenset({1}))
    assert decision.scores[1] == 0.0
    assert decision.selected == {1}  # capacity=1 still fills from the one candidate
    assert decision.assignments[1] == ScheduleMode.COMPRESSED


def test_observe_truth_feeds_the_next_select_call():
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, utility=4, readiness=3, shift=2)}, raw={})
    # select() never even needs `reports` -- the oracle ignores it entirely.
    decision = coord.select(epoch=0, reports={}, available=frozenset({1}))
    assert decision.scores[1] > 0.0
    assert decision.assignments[1] == ScheduleMode.FULL


def test_oracle_has_zero_uncertainty_unlike_bayes_decoding():
    # Even at a very noisy epsilon, the oracle's risk_hat must be exactly 1.0 --
    # ground truth has no decoding uncertainty to blur it, unlike PrivateFairCoordinator
    # at the same epsilon would have (see test_baselines_naive's equivalent check).
    coord = _coordinator(capacity=1, max_age=1000, epsilon=0.01)
    coord.observe_truth({1: _true(1, 0, utility=4, readiness=0, shift=0)}, raw={})
    posterior = coord._posterior(1, None, 0)
    assert risk_hat(posterior) == pytest.approx(1.0)


def test_oracle_reacts_to_true_shift_and_readiness():
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, utility=2, readiness=2, shift=2)}, raw={})
    posterior = coord._posterior(1, None, 0)
    assert is_likely_shifted(posterior) is True
    assert is_full_capable(posterior) is True  # readiness bin 2 is full-capable (A5/#37)


def test_observe_truth_only_covers_sites_it_was_given():
    coord = _coordinator(capacity=2, max_age=1000)
    coord.observe_truth({1: _true(1, 0, utility=4, readiness=3, shift=2)}, raw={})  # nothing for site 2
    decision = coord.select(epoch=0, reports={}, available=frozenset({1, 2}))
    assert decision.scores[2] == 0.0
    assert decision.selected == {1, 2}  # capacity still covers both
    assert decision.assignments[2] == ScheduleMode.COMPRESSED


def test_observe_truth_state_is_replaced_each_epoch():
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, utility=4, readiness=3, shift=2)}, raw={"epoch": 0})
    coord.observe_truth({1: _true(1, 1, utility=0, readiness=0, shift=0)}, raw={"epoch": 1})  # replaces, not merges
    decision = coord.select(epoch=1, reports={}, available=frozenset({1}))
    assert decision.scores[1] == pytest.approx(0.0 + coord.gamma * coord.tracker.age(1, 1))  # low risk/readiness now


def test_oracle_coverage_and_ablation_params_still_work():
    # Inherited from PrivateFairCoordinator: coverage reservation still applies
    # even though decoding is ground-truth.
    coord = _coordinator(capacity=1, max_age=2)
    coord.tracker.update(epoch=0, completed=frozenset({2}))
    coord.observe_truth(
        {
            1: _true(1, 2, utility=0, readiness=0, shift=0),  # overdue, worst truth
            2: _true(2, 2, utility=4, readiness=3, shift=2),  # not overdue, best truth
        },
        raw={},
    )
    decision = coord.select(epoch=2, reports={}, available=frozenset({1, 2}))
    assert decision.mandatory_sites == {1}
    assert decision.selected == {1}

"""Tests for RawOracleCoordinator (task B6, report section 10.E #3).

Two rounds of review found two distinct bugs, both fixed here:

1. (#41) The oracle scored from the *quantized* true bins (one-hot), making it
   decision-for-decision identical to the "quantized non-private" baseline
   (NaiveDecodingCoordinator + NoPrivacyPrivatizer) -- the oracle-vs-quantized
   gap report 10.E #3/#4 is supposed to measure what coarsening costs was zero
   by construction. Fixed to score from the continuous pre-quantization values
   in `raw` (A6/#40's shape) instead.
   Regression guard: test_diverges_from_quantized_baseline_within_the_same_bin.

2. (follow-up review) `_posterior`'s p_readiness/p_shift are engineered to hit a
   target *expectation* (needed for score()'s continuous readiness_hat/p_shift_top
   terms), which is a different property from "which bin is most probable" --
   so using their argmax for the FULL/COMPRESSED and shifted-slot-eligibility
   *decisions* (`is_full_capable`/`is_likely_shifted`, inherited from
   PrivateFairCoordinator) silently gave the wrong answer: sites comfortably
   within their deadline (expected/deadline ratios of 0.5, 0.67, 1.0) came out
   COMPRESSED. Fixed with `_full_capable`/`_is_likely_shifted` overrides that
   check the exact raw numbers instead of any posterior's argmax.
   Regression guard: test_full_capable_uses_exact_deadline_not_argmax and
   test_full_capable_reviewer_repro_cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from privatefair.baselines.naive import NaiveDecodingCoordinator
from privatefair.baselines.oracle import RawOracleCoordinator, _shift_distribution, _spread_expectation
from privatefair.coordinator.privatefair import is_full_capable, is_likely_shifted, readiness_hat, risk_hat
from privatefair.interfaces import ScheduleMode, TelemetryReport, TrueBins


def _true(site_id, epoch, utility, readiness, shift):
    return TrueBins(site_id=site_id, epoch=epoch, utility=utility, readiness=readiness, shift=shift)


def _coordinator(**kwargs):
    kwargs.setdefault("rng", np.random.default_rng(0))
    kwargs.setdefault("epsilon", 0.1)  # unused by this coordinator; present for API symmetry
    return RawOracleCoordinator(**kwargs)


# ---------------------------------------------------------------------------
# _spread_expectation / _shift_distribution -- hand-checked math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.0, 0.25, 0.5, 0.73, 1.0])
def test_spread_expectation_hits_target_expectation_exactly(target):
    dist = _spread_expectation(target, 5)
    assert sum(dist) == pytest.approx(1.0)
    e = sum(i * p for i, p in enumerate(dist)) / 4  # E[bin] / (k - 1)
    assert e == pytest.approx(target)


def test_spread_expectation_clips_out_of_range():
    assert _spread_expectation(-1.0, 4) == _spread_expectation(0.0, 4)
    assert _spread_expectation(2.0, 4) == _spread_expectation(1.0, 4)


@pytest.mark.parametrize("top", [0.0, 0.3, 0.7, 1.0])
def test_shift_distribution_puts_exact_mass_on_last_bin(top):
    dist = _shift_distribution(top, 3)
    assert sum(dist) == pytest.approx(1.0)
    assert dist[-1] == pytest.approx(top)
    assert dist[0] == pytest.approx(1 - top)


# ---------------------------------------------------------------------------
# The bug from #41: oracle must differ from the quantized baseline
# ---------------------------------------------------------------------------


def test_diverges_from_quantized_baseline_within_the_same_bin():
    # d1 and d2 both land in utility bin 1 ("improving", -0.04 <= d < -0.02 at
    # tau=0.02) -- identical under any quantized/naive-decoding view -- but are
    # meaningfully different continuously. The oracle must tell them apart;
    # NaiveDecodingCoordinator (which only ever sees the quantized bin, same as
    # the "quantized non-private" baseline) cannot.
    d1, d2 = -0.039, -0.021
    oracle = _coordinator(capacity=1, max_age=1000)
    true_bins = {1: _true(1, 0, utility=1, readiness=3, shift=0)}

    oracle.observe_truth(true_bins, {"utility_delta": {1: d1}})
    risk1 = risk_hat(oracle._posterior(1, None, 0))
    oracle.observe_truth(true_bins, {"utility_delta": {1: d2}})
    risk2 = risk_hat(oracle._posterior(1, None, 0))
    assert risk1 != pytest.approx(risk2)

    # Naive decoding (and by extension the quantized-non-private baseline) only
    # ever sees the quantized bin, so it gives the exact same answer for both --
    # the midpoint of bin 1's normalized range (1 / (5-1) = 0.25) -- regardless
    # of which continuous value produced that bin. That's the coarsening-cost
    # gap the oracle vs. quantized comparison needs, and it's now non-zero.
    naive = NaiveDecodingCoordinator(capacity=1, max_age=1000, rng=np.random.default_rng(0), epsilon=0.1)
    report = TelemetryReport(site_id=1, epoch=0, utility=1, readiness=3, shift=0)  # same bin for both d1, d2
    naive_risk = risk_hat(naive._posterior(1, report, 0))
    assert naive_risk == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Missing/default raw values
# ---------------------------------------------------------------------------


def test_first_epoch_missing_utility_delta_defaults_to_neutral_risk():
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 2, 3, 0)}, raw={})  # no utility_delta yet (epoch 0)
    assert risk_hat(coord._posterior(1, None, 0)) == pytest.approx(0.5)


def test_missing_shift_z_defaults_to_not_shifted():
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 2, 3, 0)}, raw={})
    posterior = coord._posterior(1, None, 0)
    assert posterior.p_shift[-1] == pytest.approx(0.0)


def test_missing_systems_timing_defaults_to_fast():
    # No expected_seconds/deadline_seconds in raw (no systems config) -> same
    # "always available and fast" default A5/A6 use elsewhere.
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 2, 3, 0)}, raw={})
    posterior = coord._posterior(1, None, 0)
    assert readiness_hat(posterior) == pytest.approx(1.0)
    assert coord._full_capable(1, posterior) is True


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


def test_name_defaults_to_raw_oracle():
    coord = _coordinator(capacity=1, max_age=100)
    assert coord.name == "raw_oracle"


def test_select_before_observe_truth_treats_every_site_as_missing():
    coord = _coordinator(capacity=1, max_age=1000)
    decision = coord.select(epoch=0, reports={}, available=frozenset({1}))
    assert decision.scores[1] == 0.0
    assert decision.selected == {1}  # capacity=1 still fills from the one candidate
    assert decision.assignments[1] == ScheduleMode.COMPRESSED


def test_observe_truth_feeds_the_next_select_call():
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 4, 3, 2)}, raw={"utility_delta": {1: 0.1}, "shift_z": {1: 10.0}})
    # select() never even needs `reports` -- the oracle ignores it entirely.
    decision = coord.select(epoch=0, reports={}, available=frozenset({1}))
    assert decision.scores[1] > 0.0
    assert decision.assignments[1] == ScheduleMode.FULL


def test_oracle_score_is_exact_not_blurred_by_decoding_uncertainty():
    # Unlike Bayes decoding at low epsilon (which smears the posterior toward
    # uniform -- see test_baselines_naive's equivalent check), the oracle's
    # risk_hat exactly reflects the continuous raw score: no privacy-driven blur.
    coord = _coordinator(capacity=1, max_age=1000, epsilon=0.01)
    coord.observe_truth({1: _true(1, 0, 4, 0, 0)}, raw={"utility_delta": {1: 1.0}})  # far past saturation
    posterior = coord._posterior(1, None, 0)
    assert risk_hat(posterior) == pytest.approx(1.0)


def test_oracle_reacts_to_raw_shift_and_readiness():
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth(
        {1: _true(1, 0, 2, 2, 2)},
        raw={"shift_z": {1: 4.0}, "expected_seconds": {1: 1.0}, "deadline_seconds": 10.0},
    )
    posterior = coord._posterior(1, None, 0)
    assert coord._is_likely_shifted(1, posterior) is True
    assert coord._full_capable(1, posterior) is True


# ---------------------------------------------------------------------------
# _full_capable / _is_likely_shifted: exact rule, not an argmax proxy
# ---------------------------------------------------------------------------


def test_full_capable_uses_exact_deadline_not_argmax():
    # This is the bug: p_readiness is engineered to hit a target *expectation*,
    # so its argmax (what the base class's free is_full_capable checks) does
    # not reliably track the FULL/COMPRESSED deadline boundary. Demonstrate the
    # mismatch directly, then confirm the override gets it right.
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 2, 3, 0)}, raw={"expected_seconds": {1: 5.0}, "deadline_seconds": 10.0})
    posterior = coord._posterior(1, None, 0)  # ratio=0.5, comfortably within deadline
    assert is_full_capable(posterior) is False  # the argmax-based free function gets this wrong
    assert coord._full_capable(1, posterior) is True  # the exact-rule override gets it right


@pytest.mark.parametrize("ratio", [0.0, 0.5, 0.67, 1.0])
def test_full_capable_reviewer_repro_cases(ratio):
    # Exact ratios from review: all comfortably within (or exactly at) the
    # deadline, all must be FULL-capable.
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth(
        {1: _true(1, 0, 2, 3, 0)}, raw={"expected_seconds": {1: ratio * 10.0}, "deadline_seconds": 10.0}
    )
    decision = coord.select(epoch=0, reports={}, available=frozenset({1}))
    assert decision.assignments[1] == ScheduleMode.FULL


def test_full_capable_false_past_the_deadline():
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 2, 3, 0)}, raw={"expected_seconds": {1: 15.0}, "deadline_seconds": 10.0})
    decision = coord.select(epoch=0, reports={}, available=frozenset({1}))
    assert decision.assignments[1] == ScheduleMode.COMPRESSED


def test_is_likely_shifted_uses_exact_threshold_not_argmax():
    # z=3.0: past the argmax flip point (z>2, since dist=[1-z/4, 0, z/4] flips
    # argmax at z/4>0.5) but below shift_bin's actual "shifted" threshold (4.0).
    coord = _coordinator(capacity=2, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 2, 3, 1)}, raw={"shift_z": {1: 3.0}})
    posterior = coord._posterior(1, None, 0)
    assert is_likely_shifted(posterior) is True  # argmax-based free function: past its own flip point
    assert coord._is_likely_shifted(1, posterior) is False  # exact rule: below the real shifted threshold


def test_is_likely_shifted_true_at_the_exact_threshold():
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 2, 3, 2)}, raw={"shift_z": {1: 4.0}})
    posterior = coord._posterior(1, None, 0)
    assert coord._is_likely_shifted(1, posterior) is True


def test_observe_truth_only_covers_sites_it_was_given():
    coord = _coordinator(capacity=2, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 4, 3, 2)}, raw={"utility_delta": {1: 0.1}})  # nothing for site 2
    decision = coord.select(epoch=0, reports={}, available=frozenset({1, 2}))
    assert decision.scores[2] == 0.0
    assert decision.selected == {1, 2}  # capacity still covers both
    assert decision.assignments[2] == ScheduleMode.COMPRESSED


def test_observe_truth_state_is_replaced_each_epoch():
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 4, 3, 2)}, raw={"utility_delta": {1: 1.0}})  # high risk
    coord.observe_truth({1: _true(1, 1, 0, 0, 0)}, raw={"utility_delta": {1: -1.0}})  # replaces, not merges
    posterior = coord._posterior(1, None, epoch=1)
    assert risk_hat(posterior) == pytest.approx(0.0)  # only the second (low-risk) call applies


def test_stale_truth_from_a_different_epoch_is_treated_as_missing():
    # observe_truth was called for epoch 0; select() is then asked about epoch 1
    # (the loop "forgot" to call observe_truth again) -- must not silently reuse
    # epoch 0's truth as if it were current.
    coord = _coordinator(capacity=1, max_age=1000)
    coord.observe_truth({1: _true(1, 0, 4, 3, 2)}, raw={"utility_delta": {1: 1.0}})
    assert coord._posterior(1, None, epoch=1) is None
    decision = coord.select(epoch=1, reports={}, available=frozenset({1}))
    assert decision.scores[1] == 0.0


def test_oracle_coverage_and_ablation_params_still_work():
    # Inherited from PrivateFairCoordinator: coverage reservation is purely
    # age-driven (server state), unaffected by how decoding/scoring works.
    coord = _coordinator(capacity=1, max_age=2)
    coord.tracker.update(epoch=0, completed=frozenset({2}))
    coord.observe_truth(
        {1: _true(1, 2, 0, 0, 0), 2: _true(2, 2, 4, 3, 2)},  # site 1 overdue, site 2 not
        raw={},
    )
    decision = coord.select(epoch=2, reports={}, available=frozenset({1, 2}))
    assert decision.mandatory_sites == {1}
    assert decision.selected == {1}

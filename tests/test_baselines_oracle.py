"""Tests for RawOracleCoordinator (task B6, report section 10.E #3).

Review on #41 found the oracle scored from the *quantized* true bins (one-hot),
making it decision-for-decision identical to the "quantized non-private" baseline
(NaiveDecodingCoordinator + NoPrivacyPrivatizer) -- the oracle-vs-quantized gap
report 10.E #3/#4 is supposed to measure what coarsening costs was zero by
construction. Fixed to score from the continuous pre-quantization values in
`raw` (A6/#40's shape) instead; test_diverges_from_quantized_baseline_within_the_same_bin
below is the regression guard for that specific bug.
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
    assert is_full_capable(posterior) is True


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
    assert is_likely_shifted(posterior) is True
    assert is_full_capable(posterior) is True


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

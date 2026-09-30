"""Tests for NaiveDecodingCoordinator (task B6, report section 7.3)."""

from __future__ import annotations

import numpy as np
import pytest

from privatefair.baselines.naive import NaiveDecodingCoordinator
from privatefair.coordinator.posterior import decode_report
from privatefair.coordinator.privatefair import risk_hat
from privatefair.interfaces import ScheduleMode, TelemetryReport


def _report(site_id, epoch, utility, readiness, shift):
    return TelemetryReport(site_id=site_id, epoch=epoch, utility=utility, readiness=readiness, shift=shift)


def _coordinator(**kwargs):
    kwargs.setdefault("rng", np.random.default_rng(0))
    kwargs.setdefault("epsilon", 0.1)  # deliberately noisy: makes naive vs Bayes diverge sharply
    return NaiveDecodingCoordinator(**kwargs)


def test_name_defaults_to_naive_decoding():
    coord = _coordinator(capacity=1, max_age=100)
    assert coord.name == "naive_decoding"


def test_naive_decoding_trusts_the_report_exactly():
    # epsilon=0.1 is a very noisy RR channel; Bayes-decoding should pull the
    # posterior heavily toward uniform, but naive decoding must not move at all
    # from a one-hot at the reported bin.
    report = _report(1, 0, utility=4, readiness=0, shift=0)
    coord = _coordinator(capacity=1, max_age=1000)

    naive_posterior = coord._posterior(1, report, 0)
    bayes_posterior = decode_report(report, epsilon=0.1)

    assert risk_hat(naive_posterior) == pytest.approx(1.0)  # fully trusts utility=4 (worst bin)
    assert risk_hat(bayes_posterior) < 0.9  # Bayes correction pulls it down noticeably


def test_naive_decoding_still_runs_the_full_coordinator_pipeline():
    # Same coverage, shifted-slot and mode-assignment logic as PrivateFairCoordinator --
    # only decoding differs. Sanity-check a full select() call end to end.
    coord = _coordinator(capacity=2, max_age=1000, alpha=1.0, beta=1.0, gamma=0.0, delta=1.0)
    reports = {
        1: _report(1, 0, utility=4, readiness=3, shift=0),  # high risk, fastest
        2: _report(2, 0, utility=0, readiness=0, shift=0),  # low risk, slowest
    }
    decision = coord.select(epoch=0, reports=reports, available=frozenset({1, 2}))
    assert decision.selected == {1, 2}  # capacity covers both
    assert decision.assignments[1] == ScheduleMode.FULL
    assert decision.assignments[2] == ScheduleMode.COMPRESSED
    assert decision.scores[1] > decision.scores[2]


def test_naive_decoding_missing_report_behaves_like_base_class():
    coord = _coordinator(capacity=0, max_age=1000)
    decision = coord.select(epoch=0, reports={}, available=frozenset({1}))
    assert decision.scores[1] == 0.0
    assert decision.deferral_reasons[1] == "not selected: score"


def test_naive_decoding_epsilon_is_accepted_but_irrelevant_to_the_outcome():
    # Different epsilon values must not change anything, since naive decoding
    # never consults it.
    report = _report(1, 0, utility=2, readiness=1, shift=1)
    low_eps = _coordinator(capacity=1, max_age=1000, epsilon=0.01).select(0, {1: report}, frozenset({1}))
    high_eps = _coordinator(capacity=1, max_age=1000, epsilon=50.0).select(0, {1: report}, frozenset({1}))
    assert low_eps.scores == high_eps.scores

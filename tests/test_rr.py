"""Tests for the K-ary randomized response mechanism (task B1)."""

import math

import numpy as np
import pytest

from privatefair.interfaces import ALPHABET_SIZE, K_READINESS, K_SHIFT, K_UTILITY, SIGNALS, TrueBins
from privatefair.privacy.rr import (
    RandomizedResponsePrivatizer,
    channel_matrix,
    other_prob,
    sample,
    truthful_prob,
)


@pytest.mark.parametrize("epsilon,k", [(0.1, 5), (1.0, 4), (3.0, 3), (10.0, 5)])
def test_truthful_prob_formula(epsilon, k):
    expected = math.exp(epsilon) / (math.exp(epsilon) + k - 1)
    assert truthful_prob(epsilon, k) == pytest.approx(expected)


@pytest.mark.parametrize("epsilon,k", [(0.1, 5), (1.0, 4), (3.0, 3)])
def test_other_prob_formula(epsilon, k):
    expected = 1.0 / (math.exp(epsilon) + k - 1)
    assert other_prob(epsilon, k) == pytest.approx(expected)


@pytest.mark.parametrize("epsilon,k", [(0.5, 5), (2.0, 4), (5.0, 3)])
def test_channel_matrix_rows_are_distributions(epsilon, k):
    m = channel_matrix(epsilon, k)
    assert m.shape == (k, k)
    np.testing.assert_allclose(m.sum(axis=1), np.ones(k))
    for b in range(k):
        assert m[b, b] == pytest.approx(truthful_prob(epsilon, k))
        for y in range(k):
            if y != b:
                assert m[b, y] == pytest.approx(other_prob(epsilon, k))


def test_channel_matrix_rejects_bad_inputs():
    with pytest.raises(ValueError):
        channel_matrix(-1.0, 5)
    with pytest.raises(ValueError):
        channel_matrix(1.0, 1)


def test_sample_empirical_frequency_matches_formula():
    rng = np.random.default_rng(0)
    epsilon, k, true_bin = 1.0, 5, 2
    n = 40_000
    outputs = [sample(true_bin, k, epsilon, rng) for _ in range(n)]
    counts = np.bincount(outputs, minlength=k)
    empirical = counts / n
    expected = channel_matrix(epsilon, k)[true_bin]
    # generous tolerance: 3 * binomial std-dev at n=40000
    tol = 3 * math.sqrt(expected.max() * (1 - expected.max()) / n) + 0.01
    np.testing.assert_allclose(empirical, expected, atol=tol)


def test_sample_rejects_out_of_range_true_bin():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample(5, 5, 1.0, rng)
    with pytest.raises(ValueError):
        sample(-1, 5, 1.0, rng)


def test_large_epsilon_almost_always_reports_true_bin():
    rng = np.random.default_rng(1)
    epsilon, k, true_bin = 50.0, 5, 3
    outputs = [sample(true_bin, k, epsilon, rng) for _ in range(200)]
    assert outputs.count(true_bin) == 200


# ---------------------------------------------------------------------------
# RandomizedResponsePrivatizer
# ---------------------------------------------------------------------------


def test_privatizer_single_epsilon_structure():
    priv = RandomizedResponsePrivatizer(epsilon=2.0)
    true_bins = TrueBins(site_id=7, epoch=3, utility=4, readiness=1, shift=2)
    rng = np.random.default_rng(42)

    report, ledger = priv.privatize(true_bins, rng)

    assert report.site_id == 7
    assert report.epoch == 3
    for signal in SIGNALS:
        assert 0 <= getattr(report, signal) < ALPHABET_SIZE[signal]

    assert len(ledger) == 3
    assert {entry.signal for entry in ledger} == set(SIGNALS)
    for entry in ledger:
        assert entry.site_id == 7
        assert entry.epoch == 3
        assert entry.epsilon == 2.0


def test_privatizer_per_signal_epsilon():
    eps = {"utility": 0.5, "readiness": 1.5, "shift": 3.0}
    priv = RandomizedResponsePrivatizer(epsilon=eps)
    true_bins = TrueBins(site_id=1, epoch=0, utility=0, readiness=0, shift=0)
    rng = np.random.default_rng(0)

    _, ledger = priv.privatize(true_bins, rng)

    spent = {entry.signal: entry.epsilon for entry in ledger}
    assert spent == eps


def test_privatizer_rejects_missing_signal_in_mapping():
    with pytest.raises(ValueError):
        RandomizedResponsePrivatizer(epsilon={"utility": 1.0, "readiness": 1.0})


def test_privatizer_rejects_bad_epsilon():
    with pytest.raises(ValueError):
        RandomizedResponsePrivatizer(epsilon=0.0)
    with pytest.raises(ValueError):
        RandomizedResponsePrivatizer(epsilon={"utility": -1.0, "readiness": 1.0, "shift": 1.0})


def test_privatizer_high_epsilon_reports_true_bin_most_of_the_time():
    priv = RandomizedResponsePrivatizer(epsilon=50.0)
    true_bins = TrueBins(site_id=0, epoch=0, utility=3, readiness=2, shift=1)
    rng = np.random.default_rng(0)

    matches = 0
    trials = 200
    for _ in range(trials):
        report, _ = priv.privatize(true_bins, rng)
        matches += (
            report.utility == true_bins.utility
            and report.readiness == true_bins.readiness
            and report.shift == true_bins.shift
        )
    assert matches == trials


@pytest.mark.parametrize("k", [K_UTILITY, K_READINESS, K_SHIFT])
def test_alphabet_sizes_match_interfaces(k):
    # sanity: interfaces.ALPHABET_SIZE values are exactly the signals' K
    assert k in ALPHABET_SIZE.values()

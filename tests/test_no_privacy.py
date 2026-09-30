"""Tests for NoPrivacyPrivatizer (task B6, report 10.E #4: "quantized non-private telemetry")."""

import numpy as np
import pytest

from privatefair.interfaces import TrueBins
from privatefair.privacy.no_privacy import NoPrivacyPrivatizer


def test_report_matches_true_bins_exactly():
    priv = NoPrivacyPrivatizer()
    true_bins = TrueBins(site_id=3, epoch=1, utility=4, readiness=0, shift=2)
    report, ledger = priv.privatize(true_bins, np.random.default_rng(0))

    assert (report.site_id, report.epoch) == (3, 1)
    assert (report.utility, report.readiness, report.shift) == (4, 0, 2)


def test_no_ledger_spend():
    priv = NoPrivacyPrivatizer()
    _, ledger = priv.privatize(TrueBins(0, 0, 1, 1, 1), np.random.default_rng(0))
    assert ledger == []


def test_deterministic_regardless_of_rng():
    priv = NoPrivacyPrivatizer()
    true_bins = TrueBins(site_id=1, epoch=0, utility=2, readiness=3, shift=1)
    a, _ = priv.privatize(true_bins, np.random.default_rng(0))
    b, _ = priv.privatize(true_bins, np.random.default_rng(999))  # different seed, same result
    assert a == b


@pytest.mark.parametrize("seed", range(20))
def test_many_random_true_bins_round_trip_exactly(seed):
    rng = np.random.default_rng(seed)
    true_bins = TrueBins(
        site_id=int(rng.integers(0, 100)),
        epoch=int(rng.integers(0, 50)),
        utility=int(rng.integers(0, 5)),
        readiness=int(rng.integers(0, 4)),
        shift=int(rng.integers(0, 3)),
    )
    report, ledger = NoPrivacyPrivatizer().privatize(true_bins, rng)
    assert (report.utility, report.readiness, report.shift) == (true_bins.utility, true_bins.readiness, true_bins.shift)
    assert ledger == []

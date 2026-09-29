"""Tests for the privacy ledger: composition accounting and the hard budget cap (task B3)."""

import numpy as np
import pytest

from privatefair.interfaces import LedgerEntry, TrueBins
from privatefair.privacy.ledger import PrivacyBudgetExceeded, PrivacyLedger
from privatefair.privacy.rr import RandomizedResponsePrivatizer


def test_empty_ledger():
    ledger = PrivacyLedger()
    assert ledger.spent(0) == 0.0
    assert ledger.release_count(0) == 0
    assert ledger.entries_for(0) == []
    assert ledger.entries == []


def test_record_single_entry():
    ledger = PrivacyLedger()
    ledger.record(LedgerEntry(site_id=1, epoch=0, signal="utility", epsilon=1.0))
    assert ledger.spent(1) == pytest.approx(1.0)
    assert ledger.release_count(1) == 1


def test_composition_sums_across_signals_and_epochs():
    ledger = PrivacyLedger()
    ledger.record(
        [
            LedgerEntry(site_id=1, epoch=0, signal="utility", epsilon=1.0),
            LedgerEntry(site_id=1, epoch=0, signal="readiness", epsilon=1.0),
            LedgerEntry(site_id=1, epoch=0, signal="shift", epsilon=1.0),
        ]
    )
    ledger.record(
        [
            LedgerEntry(site_id=1, epoch=1, signal="utility", epsilon=0.5),
            LedgerEntry(site_id=1, epoch=1, signal="readiness", epsilon=0.5),
            LedgerEntry(site_id=1, epoch=1, signal="shift", epsilon=0.5),
        ]
    )
    # basic composition: total = sum of every released entry's epsilon
    assert ledger.spent(1) == pytest.approx(4.5)
    assert ledger.release_count(1) == 6
    assert ledger.spent(1, signal="utility") == pytest.approx(1.5)
    assert ledger.release_count(1, signal="utility") == 2


def test_composition_is_per_site():
    ledger = PrivacyLedger()
    ledger.record(LedgerEntry(site_id=1, epoch=0, signal="utility", epsilon=2.0))
    ledger.record(LedgerEntry(site_id=2, epoch=0, signal="utility", epsilon=3.0))
    assert ledger.spent(1) == pytest.approx(2.0)
    assert ledger.spent(2) == pytest.approx(3.0)
    assert ledger.entries_for(1) == [LedgerEntry(1, 0, "utility", 2.0)]


def test_no_cap_allows_unlimited_spend():
    ledger = PrivacyLedger(cap=None)
    for epoch in range(50):
        ledger.record(LedgerEntry(site_id=1, epoch=epoch, signal="utility", epsilon=10.0))
    assert ledger.spent(1) == pytest.approx(500.0)
    assert ledger.remaining(1) is None


def test_cap_allows_spend_exactly_at_the_cap():
    ledger = PrivacyLedger(cap=3.0)
    ledger.record(LedgerEntry(site_id=1, epoch=0, signal="utility", epsilon=1.5))
    ledger.record(LedgerEntry(site_id=1, epoch=1, signal="utility", epsilon=1.5))
    assert ledger.spent(1) == pytest.approx(3.0)
    assert ledger.remaining(1) == pytest.approx(0.0)


def test_cap_rejects_entry_that_would_exceed_it():
    ledger = PrivacyLedger(cap=3.0)
    ledger.record(LedgerEntry(site_id=1, epoch=0, signal="utility", epsilon=2.5))
    with pytest.raises(PrivacyBudgetExceeded):
        ledger.record(LedgerEntry(site_id=1, epoch=1, signal="utility", epsilon=1.0))
    # the rejected entry must NOT be recorded
    assert ledger.spent(1) == pytest.approx(2.5)
    assert ledger.release_count(1) == 1


def test_cap_check_is_atomic_across_a_batch():
    # a report's 3 per-signal entries are recorded together; if the 3rd would blow
    # the cap, none of the 3 should land in the ledger.
    ledger = PrivacyLedger(cap=2.0)
    batch = [
        LedgerEntry(site_id=1, epoch=0, signal="utility", epsilon=1.0),
        LedgerEntry(site_id=1, epoch=0, signal="readiness", epsilon=1.0),
        LedgerEntry(site_id=1, epoch=0, signal="shift", epsilon=1.0),  # pushes total to 3.0 > 2.0
    ]
    with pytest.raises(PrivacyBudgetExceeded):
        ledger.record(batch)
    assert ledger.spent(1) == 0.0
    assert ledger.entries_for(1) == []


def test_cap_is_per_site_not_global():
    ledger = PrivacyLedger(cap=2.0)
    ledger.record(LedgerEntry(site_id=1, epoch=0, signal="utility", epsilon=2.0))
    # site 2 is unaffected by site 1's spend
    ledger.record(LedgerEntry(site_id=2, epoch=0, signal="utility", epsilon=2.0))
    assert ledger.spent(1) == pytest.approx(2.0)
    assert ledger.spent(2) == pytest.approx(2.0)
    with pytest.raises(PrivacyBudgetExceeded):
        ledger.record(LedgerEntry(site_id=1, epoch=1, signal="utility", epsilon=0.01))


def test_remaining_tracks_cap_consumption():
    ledger = PrivacyLedger(cap=5.0)
    assert ledger.remaining(1) == pytest.approx(5.0)
    ledger.record(LedgerEntry(site_id=1, epoch=0, signal="shift", epsilon=2.0))
    assert ledger.remaining(1) == pytest.approx(3.0)


def test_entries_property_is_a_copy():
    ledger = PrivacyLedger()
    ledger.record(LedgerEntry(site_id=1, epoch=0, signal="utility", epsilon=1.0))
    snapshot = ledger.entries
    snapshot.append(LedgerEntry(site_id=99, epoch=0, signal="utility", epsilon=1.0))
    assert ledger.spent(99) == 0.0  # mutating the returned list must not affect the ledger


def test_integration_with_randomized_response_privatizer():
    # exercises the real B1 -> B3 flow: privatize a report, feed its LedgerEntry
    # batch straight into the ledger.
    priv = RandomizedResponsePrivatizer(epsilon=1.0)
    rng = np.random.default_rng(0)
    ledger = PrivacyLedger(cap=10.0)

    for epoch in range(3):
        true_bins = TrueBins(site_id=7, epoch=epoch, utility=1, readiness=2, shift=0)
        _, entries = priv.privatize(true_bins, rng)
        ledger.record(entries)

    assert ledger.release_count(7) == 9  # 3 signals x 3 epochs
    assert ledger.spent(7) == pytest.approx(9.0)  # 9 releases x epsilon=1.0

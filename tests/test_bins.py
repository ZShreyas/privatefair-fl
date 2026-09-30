"""Tests for the telemetry threshold rules (task A5). No torch."""

import pytest

from privatefair.interfaces import TrueBins
from privatefair.sim.bins import readiness_bin, shift_bin, utility_bin


def test_utility_first_epoch_is_stable():
    assert utility_bin(None, 0.7, tau=0.02) == 2


@pytest.mark.parametrize(
    ("curr", "expected"),
    [
        (0.90, 0),
        (0.96, 0),
        (0.97, 1),
        (0.98, 1),
        (0.99, 2),
        (1.0, 2),
        (1.01, 2),
        (1.02, 3),
        (1.03, 3),
        (1.04, 4),
        (2.0, 4),
    ],  # fmt: skip
)
def test_utility_bin_edges(curr, expected):
    # prev = 1.0, tau = 0.02: edges at -4%, -2%, +2%, +4% relative change
    assert utility_bin(1.0, curr, tau=0.02) == expected


def test_utility_is_relative_not_absolute():
    # Same 5% relative rise at very different loss scales -> same bin.
    assert utility_bin(1.0, 1.05, 0.02) == utility_bin(0.2, 0.21, 0.02) == 4


@pytest.mark.parametrize(
    ("available", "expected_s", "expected"),
    [(False, 0.0, 0), (True, 12.0, 1), (True, 10.0, 2), (True, 6.0, 2), (True, 5.0, 3), (True, 0.0, 3)],
)
def test_readiness_bin(available, expected_s, expected):
    assert readiness_bin(available, expected_s, deadline_seconds=10.0, fast_fraction=0.5) == expected


@pytest.mark.parametrize(("z", "expected"), [(0.5, 0), (1.99, 0), (2.0, 1), (3.9, 1), (4.0, 2), (50.0, 2)])
def test_shift_bin(z, expected):
    assert shift_bin(z, (2.0, 4.0)) == expected


def test_all_outputs_are_valid_truebins():
    for u in (utility_bin(1.0, c, 0.02) for c in (0.5, 0.97, 1.0, 1.03, 3.0)):
        for r in (readiness_bin(a, e, 10.0) for a, e in ((False, 0), (True, 20), (True, 8), (True, 1))):
            for s in (shift_bin(z) for z in (0.0, 3.0, 9.0)):
                TrueBins(site_id=0, epoch=0, utility=u, readiness=r, shift=s)  # raises if out of range

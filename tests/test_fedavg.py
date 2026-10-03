"""Tests for equal-weight FedAvg (task A3). Numpy only, no torch."""

import numpy as np

from privatefair.fl.fedavg import FedAvg
from privatefair.interfaces import ClientUpdate, ScheduleMode


def _update(site_id, w, completed=True):
    return ClientUpdate(site_id, 0, ScheduleMode.FULL, w, simulated_seconds=0.0, bytes_up=0, completed=completed)


GLOBAL = {"w": np.zeros(3), "count": np.array(0, dtype=np.int64)}


def test_equal_weight_mean():
    ups = [_update(0, {"w": np.array([1.0, 2.0, 3.0]), "count": np.array(5)}),
           _update(1, {"w": np.array([3.0, 4.0, 5.0]), "count": np.array(9)})]  # fmt: skip
    out = FedAvg().aggregate(GLOBAL, ups)
    np.testing.assert_allclose(out["w"], [2.0, 3.0, 4.0])


def test_incomplete_updates_are_ignored():
    ups = [_update(0, {"w": np.ones(3), "count": np.array(1)}),
           _update(1, {"w": np.full(3, 100.0), "count": np.array(1)}, completed=False)]  # fmt: skip
    np.testing.assert_allclose(FedAvg().aggregate(GLOBAL, ups)["w"], np.ones(3))


def test_no_completed_updates_returns_global():
    ups = [_update(0, {"w": np.ones(3), "count": np.array(1)}, completed=False)]
    assert FedAvg().aggregate(GLOBAL, ups) is GLOBAL
    assert FedAvg().aggregate(GLOBAL, []) is GLOBAL


def test_integer_buffers_copied_not_averaged():
    ups = [_update(0, {"w": np.ones(3), "count": np.array(4)}), _update(1, {"w": np.ones(3), "count": np.array(7)})]
    out = FedAvg().aggregate(GLOBAL, ups)
    assert out["count"].dtype == np.int64 or np.issubdtype(out["count"].dtype, np.integer)
    assert int(out["count"]) == 4


def test_does_not_mutate_inputs():
    a = {"w": np.array([1.0, 1.0, 1.0]), "count": np.array(1)}
    b = {"w": np.array([3.0, 3.0, 3.0]), "count": np.array(1)}
    FedAvg().aggregate(GLOBAL, [_update(0, a), _update(1, b)])
    np.testing.assert_array_equal(a["w"], [1.0, 1.0, 1.0])


def test_partial_key_updates_average_only_sent_keys():
    g = {"frozen": np.full(2, 7.0), "w": np.zeros(2)}
    ups = [_update(0, {"w": np.array([1.0, 1.0])}), _update(1, {"w": np.array([3.0, 5.0])})]
    out = FedAvg().aggregate(g, ups)
    np.testing.assert_allclose(out["w"], [2.0, 3.0])
    np.testing.assert_array_equal(out["frozen"], [7.0, 7.0])


def test_mismatched_key_sets_raise():
    import pytest

    ups = [_update(0, {"w": np.ones(3), "count": np.array(1)}), _update(1, {"w": np.ones(3)})]
    with pytest.raises(ValueError, match="different key sets"):
        FedAvg().aggregate(GLOBAL, ups)

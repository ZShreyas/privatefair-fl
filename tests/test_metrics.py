"""Tests for evaluation metrics (task A2). No torch."""

import numpy as np
import pytest

from privatefair.sim.metrics import balanced_accuracy


def test_perfect_and_all_wrong():
    y = np.array([0, 1, 2, 2])
    assert balanced_accuracy(y, y, 3) == 1.0
    assert balanced_accuracy(y, (y + 1) % 3, 3) == 0.0


def test_differs_from_accuracy_under_imbalance():
    y_true = np.array([0] * 9 + [1])
    y_pred = np.zeros(10, dtype=int)  # always predicts the majority class
    assert (y_pred == y_true).mean() == pytest.approx(0.9)
    assert balanced_accuracy(y_true, y_pred, 2) == pytest.approx(0.5)


def test_classes_absent_from_y_true_are_ignored():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 2, 1, 1])  # class 2 predicted but never true
    assert balanced_accuracy(y_true, y_pred, 3) == pytest.approx((0.5 + 1.0) / 2)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        balanced_accuracy(np.array([]), np.array([]), 2)
    with pytest.raises(ValueError):
        balanced_accuracy(np.array([0, 1]), np.array([0]), 2)

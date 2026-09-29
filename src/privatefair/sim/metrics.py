"""Evaluation metrics (task A2). Numpy only, so CI tests them without torch."""

from __future__ import annotations

import numpy as np


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Mean per-class recall over the classes present in y_true.

    Sites have skewed label mixes (A1), so plain accuracy can look good while a rare class
    is never predicted; balanced accuracy does not hide that.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    support = np.bincount(y_true, minlength=num_classes)
    correct = np.bincount(y_true[y_true == y_pred], minlength=num_classes)
    present = support > 0
    return float((correct[present] / support[present]).mean())

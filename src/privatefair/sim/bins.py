"""Threshold rules that turn local measurements into telemetry bins (task A5). Numpy-free, torch-free.

These run site-side in the simulator. Their inputs (losses, timings, shift scores) never leave
the site; only the resulting bins go to the privatizer (and to RoundLog.sim_only for evaluation).
"""

from __future__ import annotations

from collections.abc import Sequence

from privatefair.interfaces import K_READINESS, K_SHIFT, K_UTILITY


def utility_bin(prev_loss: float | None, curr_loss: float, tau: float) -> int:
    """Relative change in local validation loss -> 0 (strongly improving) .. 2 (stable) .. 4 (strongly deteriorating).

    Relative, so one tau works early in training (loss ~1) and late (loss ~0.2).
    No previous measurement (first epoch) -> stable.
    """
    if prev_loss is None or prev_loss <= 0:
        return K_UTILITY // 2
    d = (curr_loss - prev_loss) / prev_loss
    if d <= -2 * tau:
        return 0
    if d <= -tau:
        return 1
    if d < tau:
        return 2
    if d < 2 * tau:
        return 3
    return 4


def readiness_bin(available: bool, expected_seconds: float, deadline_seconds: float, fast_fraction: float = 0.5) -> int:
    """0 unavailable, 1 a full round would miss the deadline, 2 within deadline, 3 within fast_fraction of it."""
    if not available:
        return 0
    if expected_seconds > deadline_seconds:
        return 1
    if expected_seconds > fast_fraction * deadline_seconds:
        return 2
    return K_READINESS - 1


def shift_bin(z: float, thresholds: Sequence[float] = (2.0, 4.0)) -> int:
    """Normalized shift score -> 0 typical, 1 uncertain, 2 shifted."""
    lo, hi = thresholds
    if z < lo:
        return 0
    if z < hi:
        return 1
    return K_SHIFT - 1

"""K-ary randomized response (report section 4, "Privacy mechanism and decision rule").

For a true K-class telemetry bin B, a site emits Y through K-ary randomized response
with privacy budget epsilon:

    Pr(Y=y | B=b) = e^epsilon / (e^epsilon + K - 1),   y == b
                  = 1         / (e^epsilon + K - 1),   y != b

This gives pure epsilon-local DP for that one report. Each of the three telemetry
signals (utility, readiness, shift) is privatized independently and may use its own
epsilon.

`channel_matrix` is exposed for reuse by the coordinator's posterior decoder (B2),
which needs the exact same Pr(Y=y|B=b) values to invert the channel with Bayes' rule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from privatefair.interfaces import ALPHABET_SIZE, SIGNALS, LedgerEntry, Signal, TelemetryReport, TrueBins


def _check_epsilon(epsilon: float) -> None:
    if not (epsilon > 0 and math.isfinite(epsilon)):
        raise ValueError(f"epsilon must be a positive finite number, got {epsilon}")


def _check_k(k: int) -> None:
    if not isinstance(k, int) or k < 2:
        raise ValueError(f"k must be an int >= 2, got {k}")


def truthful_prob(epsilon: float, k: int) -> float:
    """Pr(Y=b | B=b): probability the report equals the true bin."""
    _check_epsilon(epsilon)
    _check_k(k)
    e = math.exp(epsilon)
    return e / (e + k - 1)


def other_prob(epsilon: float, k: int) -> float:
    """Pr(Y=y | B=b) for any single y != b (uniform over the other k-1 values)."""
    _check_epsilon(epsilon)
    _check_k(k)
    return 1.0 / (math.exp(epsilon) + k - 1)


def channel_matrix(epsilon: float, k: int) -> np.ndarray:
    """The K x K randomized-response channel: matrix[b, y] = Pr(Y=y | B=b).

    Every row sums to 1; the diagonal holds `truthful_prob`, off-diagonal entries
    hold `other_prob`.
    """
    p_true = truthful_prob(epsilon, k)
    p_other = other_prob(epsilon, k)
    matrix = np.full((k, k), p_other, dtype=float)
    np.fill_diagonal(matrix, p_true)
    return matrix


def sample(true_bin: int, k: int, epsilon: float, rng: np.random.Generator) -> int:
    """Draw one K-ary randomized-response output for a true bin."""
    _check_k(k)
    if not 0 <= true_bin < k:
        raise ValueError(f"true_bin must be in [0, {k - 1}], got {true_bin}")
    row = channel_matrix(epsilon, k)[true_bin]
    return int(rng.choice(k, p=row))


@dataclass(frozen=True)
class RandomizedResponsePrivatizer:
    """Site-side `Privatizer`: independent K-ary RR on each of the three signals.

    `epsilon` is either a single float (the same budget spent on every signal) or a
    mapping {"utility": ..., "readiness": ..., "shift": ...} for per-signal budgets.
    """

    epsilon: float | Mapping[Signal, float]

    def __post_init__(self) -> None:
        if isinstance(self.epsilon, Mapping):
            missing = set(SIGNALS) - set(self.epsilon)
            if missing:
                raise ValueError(f"epsilon mapping missing signals: {sorted(missing)}")
            for signal in SIGNALS:
                _check_epsilon(self.epsilon[signal])
        else:
            _check_epsilon(self.epsilon)

    def epsilon_for(self, signal: Signal) -> float:
        if isinstance(self.epsilon, Mapping):
            return self.epsilon[signal]
        return self.epsilon

    def privatize(self, true_bins: TrueBins, rng: np.random.Generator) -> tuple[TelemetryReport, list[LedgerEntry]]:
        reported: dict[str, int] = {}
        ledger: list[LedgerEntry] = []
        for signal in SIGNALS:
            k = ALPHABET_SIZE[signal]
            eps = self.epsilon_for(signal)
            true_bin = getattr(true_bins, signal)
            reported[signal] = sample(true_bin, k, eps, rng)
            ledger.append(LedgerEntry(site_id=true_bins.site_id, epoch=true_bins.epoch, signal=signal, epsilon=eps))
        report = TelemetryReport(
            site_id=true_bins.site_id,
            epoch=true_bins.epoch,
            utility=reported["utility"],
            readiness=reported["readiness"],
            shift=reported["shift"],
        )
        return report, ledger

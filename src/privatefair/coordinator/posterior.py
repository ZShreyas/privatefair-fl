"""Bayes posterior decoding of privatized telemetry (report section 4, step 3 of 7.2).

The coordinator knows the K-ary randomized-response channel that produced a report
(see privatefair.privacy.rr.channel_matrix) and a predeclared prior over the true
bin. Given a noisy report y, Bayes' rule turns that into a posterior belief:

    Pr(B=b | Y=y) = Pr(Y=y | B=b) * Pr(B=b) / sum_b' Pr(Y=y | B=b') * Pr(B=b')

This module only ever sees a privatized TelemetryReport, a public epsilon, and a
public prior -- never TrueBins -- so it respects the privacy boundary enforced by
tests/test_boundaries.py.
"""

from __future__ import annotations

from collections.abc import Mapping

from privatefair.interfaces import ALPHABET_SIZE, SIGNALS, Posterior, Signal, TelemetryReport
from privatefair.privacy.rr import channel_matrix


def uniform_prior(k: int) -> tuple[float, ...]:
    """The maximally uninformative prior: equal mass on every bin."""
    return tuple(1.0 / k for _ in range(k))


def _check_prior(prior: tuple[float, ...], k: int) -> None:
    if len(prior) != k:
        raise ValueError(f"prior must have length {k}, got {len(prior)}")
    if any(p < -1e-9 for p in prior) or abs(sum(prior) - 1.0) > 1e-6:
        raise ValueError(f"prior must be a probability distribution, got {prior}")


def decode_signal(y: int, k: int, epsilon: float, prior: tuple[float, ...] | None = None) -> tuple[float, ...]:
    """Bayes-decode one noisy category into a posterior over the true bin.

    `prior` defaults to uniform. With a uniform prior, the posterior is exactly the
    channel row for `y` -- the randomized-response channel is symmetric, so a report
    of `y` makes every other bin equally (un)likely regardless of which bin `y` is.
    """
    if not 0 <= y < k:
        raise ValueError(f"y must be in [0, {k - 1}], got {y}")
    prior = uniform_prior(k) if prior is None else prior
    _check_prior(prior, k)

    channel = channel_matrix(epsilon, k)
    unnormalized = [prior[b] * channel[b, y] for b in range(k)]
    total = sum(unnormalized)
    if total <= 0:
        raise ValueError("posterior is degenerate: prior assigns zero mass to every bin consistent with y")
    return tuple(u / total for u in unnormalized)


def decode_report(
    report: TelemetryReport,
    epsilon: float | Mapping[Signal, float],
    priors: Mapping[Signal, tuple[float, ...]] | None = None,
) -> Posterior:
    """Bayes-decode all three signals of one TelemetryReport into a Posterior.

    `epsilon` is either one shared budget or a per-signal mapping, matching the
    epsilon shape a site's Privatizer used to produce `report`. `priors` supplies a
    per-signal prior; any signal left out defaults to uniform.
    """
    priors = priors or {}
    decoded: dict[str, tuple[float, ...]] = {}
    for signal in SIGNALS:
        k = ALPHABET_SIZE[signal]
        eps = epsilon[signal] if isinstance(epsilon, Mapping) else epsilon
        y = getattr(report, signal)
        decoded[signal] = decode_signal(y, k, eps, priors.get(signal))
    return Posterior(
        site_id=report.site_id,
        epoch=report.epoch,
        p_utility=decoded["utility"],
        p_readiness=decoded["readiness"],
        p_shift=decoded["shift"],
    )

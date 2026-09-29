"""Tests for Bayes posterior decoding of telemetry reports (task B2)."""

import math

import pytest

from privatefair.coordinator.posterior import decode_report, decode_signal, uniform_prior
from privatefair.interfaces import ALPHABET_SIZE, SIGNALS, TelemetryReport
from privatefair.privacy.rr import channel_matrix


def test_uniform_prior():
    assert uniform_prior(4) == (0.25, 0.25, 0.25, 0.25)


def test_hand_computed_posterior_nonuniform_prior():
    # K=2, epsilon = ln(3) => p_true = 3/4, p_other = 1/4.
    # prior = (0.9, 0.1), observed y = 0.
    # unnormalized = (0.9*0.75, 0.1*0.25) = (0.675, 0.025); total = 0.7
    epsilon = math.log(3)
    posterior = decode_signal(y=0, k=2, epsilon=epsilon, prior=(0.9, 0.1))
    assert posterior[0] == pytest.approx(0.675 / 0.7)
    assert posterior[1] == pytest.approx(0.025 / 0.7)
    assert sum(posterior) == pytest.approx(1.0)


def test_uniform_prior_posterior_equals_channel_row():
    # With a uniform prior, RR's symmetry means the posterior over b is exactly
    # the channel row Pr(Y=y | B=.) -- no separate hand derivation needed to see
    # this holds for any y, k, epsilon.
    epsilon, k, y = 1.0, 5, 2
    posterior = decode_signal(y=y, k=k, epsilon=epsilon)
    expected = tuple(channel_matrix(epsilon, k)[y])  # row is symmetric in (b, y)
    for p, e in zip(posterior, expected, strict=True):
        assert p == pytest.approx(e)


def test_posterior_is_a_distribution():
    posterior = decode_signal(y=1, k=4, epsilon=0.3)
    assert len(posterior) == 4
    assert all(p >= 0 for p in posterior)
    assert sum(posterior) == pytest.approx(1.0)


def test_large_epsilon_gives_near_one_hot_posterior():
    posterior = decode_signal(y=3, k=5, epsilon=50.0)
    assert posterior[3] == pytest.approx(1.0, abs=1e-6)
    for i in range(5):
        if i != 3:
            assert posterior[i] == pytest.approx(0.0, abs=1e-6)


def test_decode_signal_rejects_bad_inputs():
    with pytest.raises(ValueError):
        decode_signal(y=5, k=5, epsilon=1.0)  # y out of range
    with pytest.raises(ValueError):
        decode_signal(y=0, k=5, epsilon=1.0, prior=(1.0, 0.0, 0.0, 0.0))  # wrong length
    with pytest.raises(ValueError):
        decode_signal(y=0, k=5, epsilon=1.0, prior=(0.5,) * 5)  # doesn't sum to 1


# ---------------------------------------------------------------------------
# decode_report
# ---------------------------------------------------------------------------


def test_decode_report_shared_epsilon():
    report = TelemetryReport(site_id=5, epoch=2, utility=4, readiness=0, shift=1)
    posterior = decode_report(report, epsilon=2.0)

    assert posterior.site_id == 5
    assert posterior.epoch == 2
    for signal, dist in zip(SIGNALS, (posterior.p_utility, posterior.p_readiness, posterior.p_shift), strict=True):
        assert len(dist) == ALPHABET_SIZE[signal]
        assert sum(dist) == pytest.approx(1.0)


def test_decode_report_per_signal_epsilon_and_priors():
    report = TelemetryReport(site_id=1, epoch=0, utility=2, readiness=3, shift=0)
    eps = {"utility": 0.5, "readiness": 5.0, "shift": 1.0}
    priors = {"shift": (0.7, 0.2, 0.1)}

    posterior = decode_report(report, epsilon=eps, priors=priors)

    expected_shift = decode_signal(y=0, k=3, epsilon=1.0, prior=(0.7, 0.2, 0.1))
    assert posterior.p_shift == pytest.approx(expected_shift)
    # readiness used a high epsilon and default uniform prior -> should be sharply
    # peaked at the observed bin (3)
    assert posterior.p_readiness[3] > 0.9


def test_decode_report_agrees_with_decode_signal():
    report = TelemetryReport(site_id=0, epoch=0, utility=0, readiness=0, shift=0)
    posterior = decode_report(report, epsilon=1.5)
    assert posterior.p_utility == pytest.approx(decode_signal(0, ALPHABET_SIZE["utility"], 1.5))
    assert posterior.p_readiness == pytest.approx(decode_signal(0, ALPHABET_SIZE["readiness"], 1.5))
    assert posterior.p_shift == pytest.approx(decode_signal(0, ALPHABET_SIZE["shift"], 1.5))

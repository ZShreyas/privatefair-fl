"""Tests for the policy registry and privacy config (task A6). No torch."""

import numpy as np
import pytest

from privatefair.coordinator.policies import CoverageRandomPolicy, RandomPolicy
from privatefair.coordinator.privatefair import PrivateFairCoordinator
from privatefair.sim.policies import PolicySpec, PrivacyConfig, available_policies, build_policy

RNG = np.random.default_rng(0)


def test_registry_has_core_policies():
    assert {"fedavg_all", "random", "coverage_random", "privatefair"} <= set(available_policies())


def test_build_each_policy():
    p = PrivacyConfig(epsilon=2.0)
    assert build_policy({"name": "fedavg_all"}, p, RNG).coordinator is None
    r = build_policy({"name": "random", "capacity": 2}, p, RNG)
    assert isinstance(r.coordinator, RandomPolicy) and not r.uses_telemetry
    c = build_policy({"name": "coverage_random", "capacity": 2, "max_age": 3}, p, RNG)
    assert isinstance(c.coordinator, CoverageRandomPolicy) and not c.uses_telemetry
    f = build_policy({"name": "privatefair", "capacity": 2, "max_age": 3, "gamma": 0.5}, p, RNG)
    assert isinstance(f.coordinator, PrivateFairCoordinator) and f.uses_telemetry
    assert f.coordinator.epsilon == 2.0 and f.coordinator.gamma == 0.5  # decodes with the sites' epsilon


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="unknown policy"):
        build_policy({"name": "nope"}, PrivacyConfig(), RNG)


def test_privatefair_requires_randomized_response():
    with pytest.raises(ValueError, match="rr"):
        build_policy({"name": "privatefair", "capacity": 2, "max_age": 3}, PrivacyConfig(mode="none"), RNG)


def test_needs_truth():
    class TruthSeeing:
        def observe_truth(self, true_bins, raw): ...

    assert PolicySpec("x", None, uses_telemetry=True).needs_truth
    assert PolicySpec("x", TruthSeeing(), uses_telemetry=False).needs_truth
    assert not PolicySpec("x", RandomPolicy(capacity=1, rng=RNG), uses_telemetry=False).needs_truth


@pytest.mark.parametrize(
    "kwargs", [{"mode": "ldp"}, {"epsilon": 0.0}, {"epsilon": {"utility": 1.0}}, {"priors": {"bogus": (1.0,)}}]
)
def test_privacy_config_validation(kwargs):
    with pytest.raises(ValueError):
        PrivacyConfig(**kwargs)


def test_privacy_config_from_dict_converts_priors():
    eps = {"utility": 1, "readiness": 2, "shift": 3}
    p = PrivacyConfig.from_dict({"epsilon": eps, "priors": {"shift": [0.8, 0.1, 0.1]}})
    assert p.decoding_priors() == {"shift": (0.8, 0.1, 0.1)}
    assert PrivacyConfig.from_dict(None) == PrivacyConfig()

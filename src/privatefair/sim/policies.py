"""Build a coordination policy and the privacy setup from config (task A6).

Each registry entry returns a `PolicySpec`: the `Coordinator` (or None for plain all-sites
FedAvg) plus whether it uses telemetry. Policies that do not use telemetry (Random,
CoverageRandom) make sites release nothing, so they spend no privacy budget.

B6 baselines register here too. A baseline that legitimately needs ground truth (the raw
oracle) may define `observe_truth(true_bins, raw)`; the loop calls it before `select()` only
if it exists. Such classes live outside `coordinator/`, so tests/test_boundaries.py still
guarantees no coordinator policy can see truth.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from privatefair.baselines.naive import NaiveDecodingCoordinator
from privatefair.coordinator.policies import CoverageRandomPolicy, RandomPolicy
from privatefair.coordinator.privatefair import PrivateFairCoordinator
from privatefair.interfaces import SIGNALS, Coordinator, Signal
from privatefair.privacy.rr import RandomizedResponsePrivatizer


@dataclass(frozen=True)
class PrivacyConfig:
    mode: str = "rr"  # "rr": K-ary randomized response; "none": sites send their true bins (non-private baselines)
    epsilon: float | Mapping[str, float] = 1.0  # per signal per report: one value, or a per-signal mapping
    cap: float | None = None  # max composed epsilon per site over the run; exceeding it stops the run
    priors: Mapping[str, tuple[float, ...]] | None = None  # coordinator's decoding priors; default uniform

    def __post_init__(self) -> None:
        if self.mode not in ("rr", "none"):
            raise ValueError(f"privacy.mode must be 'rr' or 'none', got {self.mode!r}")
        if self.mode == "rr":
            RandomizedResponsePrivatizer(self.epsilon)  # validates epsilon shape and values
        if self.priors is not None and not set(self.priors) <= set(SIGNALS):
            raise ValueError(f"unknown signals in privacy.priors: {sorted(set(self.priors) - set(SIGNALS))}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> PrivacyConfig:
        d = dict(d or {})
        if d.get("priors"):
            d["priors"] = {k: tuple(v) for k, v in d["priors"].items()}
        return cls(**d)

    def decoding_priors(self) -> Mapping[Signal, tuple[float, ...]] | None:
        return dict(self.priors) if self.priors else None  # type: ignore[return-value]


@dataclass
class PolicySpec:
    name: str
    coordinator: Coordinator | None  # None = plain FedAvg over every available site
    uses_telemetry: bool

    @property
    def needs_truth(self) -> bool:
        """Sites must measure true bins if the policy reads telemetry or is a truth-seeing baseline."""
        return self.uses_telemetry or hasattr(self.coordinator, "observe_truth")


Builder = Callable[[Mapping[str, Any], PrivacyConfig, np.random.Generator], PolicySpec]
_REGISTRY: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    def deco(builder: Builder) -> Builder:
        if name in _REGISTRY:
            raise ValueError(f"policy {name!r} registered twice")
        _REGISTRY[name] = builder
        return builder

    return deco


def available_policies() -> list[str]:
    return sorted(_REGISTRY)


def build_policy(policy_cfg: Mapping[str, Any], privacy: PrivacyConfig, rng: np.random.Generator) -> PolicySpec:
    cfg = dict(policy_cfg)
    name = cfg.pop("name", "fedavg_all")
    if name not in _REGISTRY:
        raise ValueError(f"unknown policy {name!r}; known: {available_policies()}")
    return _REGISTRY[name](cfg, privacy, rng)


@register("fedavg_all")
def _fedavg_all(cfg: Mapping[str, Any], privacy: PrivacyConfig, rng: np.random.Generator) -> PolicySpec:
    return PolicySpec("fedavg_all", None, uses_telemetry=False)


@register("random")
def _random(cfg: Mapping[str, Any], privacy: PrivacyConfig, rng: np.random.Generator) -> PolicySpec:
    return PolicySpec("random", RandomPolicy(capacity=cfg["capacity"], rng=rng), uses_telemetry=False)


@register("coverage_random")
def _coverage_random(cfg: Mapping[str, Any], privacy: PrivacyConfig, rng: np.random.Generator) -> PolicySpec:
    coord = CoverageRandomPolicy(capacity=cfg["capacity"], max_age=cfg["max_age"], rng=rng)
    return PolicySpec("coverage_random", coord, uses_telemetry=False)


@register("privatefair")
def _privatefair(cfg: Mapping[str, Any], privacy: PrivacyConfig, rng: np.random.Generator) -> PolicySpec:
    if privacy.mode != "rr":
        raise ValueError("privatefair Bayes-decodes RR reports and needs privacy.mode 'rr' (non-private variants: B6)")
    weights = {k: cfg[k] for k in ("alpha", "beta", "gamma", "delta") if k in cfg}
    coord = PrivateFairCoordinator(
        capacity=cfg["capacity"],
        max_age=cfg["max_age"],
        rng=rng,
        epsilon=privacy.epsilon,  # must match what sites use, so decoding inverts the right channel
        priors=privacy.decoding_priors(),
        **weights,
    )
    return PolicySpec("privatefair", coord, uses_telemetry=True)


@register("naive_decoding")
def _naive_decoding(cfg: Mapping[str, Any], privacy: PrivacyConfig, rng: np.random.Generator) -> PolicySpec:
    """B6 quantized non-private baseline with privacy.mode "none"; with "rr" it is the naive-decoding ablation."""
    weights = {k: cfg[k] for k in ("alpha", "beta", "gamma", "delta") if k in cfg}
    coord = NaiveDecodingCoordinator(
        capacity=cfg["capacity"],
        max_age=cfg["max_age"],
        rng=rng,
        epsilon=privacy.epsilon,  # inherited field, unused by naive decoding
        priors=privacy.decoding_priors(),
        **weights,
    )
    return PolicySpec("naive_decoding", coord, uses_telemetry=True)

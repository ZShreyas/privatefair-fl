"""Shared contracts for PrivateFair-FL.

This file is the agreement between every part of the codebase:

    data/sim (Track A)  --TrueBins-->  privacy (Track B)  --TelemetryReport-->  coordinator (Track B)
    coordinator  --CohortDecision-->  sim/training loop (Track A)  --ClientUpdate-->  aggregator
    everything  --RoundLog-->  analysis (Member C)

RULE: changes to this file go in their own small PR, approved by BOTH leads.
Everything else may change freely; this file may not.

The privacy boundary (report sections 7.2 and 8.4) is enforced here:
  * TelemetryReport is the ONLY thing a site sends to the coordinator for decisions.
    It contains an epoch id and three categorical reports, nothing else.
  * TrueBins are simulator ground truth. They may be logged for offline evaluation,
    but coordinator code must never import or receive them (see tests/test_boundaries.py).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Telemetry alphabets (report section 4, "Minimal telemetry and control path")
# ---------------------------------------------------------------------------

K_UTILITY = 5  # 0 = strongly improving ... 2 = stable ... 4 = strongly deteriorating
K_READINESS = 4  # 0 = unavailable ... 3 = fast / ready for full participation
K_SHIFT = 3  # 0 = typical domain, 1 = uncertain, 2 = shifted domain

SIGNALS: tuple[str, ...] = ("utility", "readiness", "shift")
ALPHABET_SIZE: dict[str, int] = {"utility": K_UTILITY, "readiness": K_READINESS, "shift": K_SHIFT}
Signal = Literal["utility", "readiness", "shift"]


def _check_bin(name: str, value: int, k: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if not 0 <= int(value) < k:
        raise ValueError(f"{name} must be in [0, {k - 1}], got {value}")


# ---------------------------------------------------------------------------
# Site -> coordinator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrueBins:
    """Simulator-side ground truth for one site at one epoch. NEVER sent to the coordinator."""

    site_id: int
    epoch: int
    utility: int
    readiness: int
    shift: int

    def __post_init__(self) -> None:
        for sig in SIGNALS:
            _check_bin(sig, getattr(self, sig), ALPHABET_SIZE[sig])


@dataclass(frozen=True)
class TelemetryReport:
    """The privatized message a site releases. Exactly these fields, nothing more."""

    site_id: int
    epoch: int
    utility: int
    readiness: int
    shift: int

    PAYLOAD_KEYS = frozenset({"site_id", "epoch", "utility", "readiness", "shift"})

    def __post_init__(self) -> None:
        for sig in SIGNALS:
            _check_bin(sig, getattr(self, sig), ALPHABET_SIZE[sig])

    def to_payload(self) -> dict[str, int]:
        return {k: int(getattr(self, k)) for k in sorted(self.PAYLOAD_KEYS)}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TelemetryReport:
        """Schema-validated parse. Rejects any extra field (e.g. raw loss, sample count)."""
        keys = set(payload)
        if keys != cls.PAYLOAD_KEYS:
            extra, missing = keys - cls.PAYLOAD_KEYS, cls.PAYLOAD_KEYS - keys
            raise ValueError(f"Invalid telemetry payload. extra={sorted(extra)} missing={sorted(missing)}")
        return cls(**{k: payload[k] for k in cls.PAYLOAD_KEYS})


@dataclass(frozen=True)
class LedgerEntry:
    """One privacy-budget expenditure: site released one report of one signal at one epoch."""

    site_id: int
    epoch: int
    signal: Signal
    epsilon: float

    def __post_init__(self) -> None:
        if self.signal not in SIGNALS:
            raise ValueError(f"unknown signal {self.signal!r}")
        if not (self.epsilon > 0 and math.isfinite(self.epsilon)):
            raise ValueError("epsilon must be a positive finite number")


# ---------------------------------------------------------------------------
# Coordinator internals and outputs
# ---------------------------------------------------------------------------


def _check_dist(name: str, p: tuple[float, ...], k: int) -> None:
    if len(p) != k:
        raise ValueError(f"{name} must have length {k}, got {len(p)}")
    if any(x < -1e-9 for x in p) or abs(sum(p) - 1.0) > 1e-6:
        raise ValueError(f"{name} must be a probability distribution, got {p}")


@dataclass(frozen=True)
class Posterior:
    """Coordinator's belief about one site's true bins after decoding a noisy report."""

    site_id: int
    epoch: int
    p_utility: tuple[float, ...]
    p_readiness: tuple[float, ...]
    p_shift: tuple[float, ...]

    def __post_init__(self) -> None:
        _check_dist("p_utility", self.p_utility, K_UTILITY)
        _check_dist("p_readiness", self.p_readiness, K_READINESS)
        _check_dist("p_shift", self.p_shift, K_SHIFT)


class ScheduleMode(str, Enum):
    FULL = "full"
    COMPRESSED = "compressed"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class CohortDecision:
    """What the coordinator decided this round, plus the explanation it must log."""

    epoch: int
    policy: str
    assignments: dict[int, ScheduleMode]  # site_id -> mode, for every site considered
    mandatory_sites: frozenset[int] = frozenset()  # overdue sites forced in by coverage rule
    scores: dict[int, float] = field(default_factory=dict)
    deferral_reasons: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for sid, mode in self.assignments.items():
            if not isinstance(mode, ScheduleMode):
                raise TypeError(f"site {sid}: mode must be ScheduleMode, got {mode!r}")
            if mode is ScheduleMode.DEFERRED and sid not in self.deferral_reasons:
                raise ValueError(f"site {sid} deferred without a deferral reason")
        if not self.mandatory_sites <= set(self.assignments):
            raise ValueError("mandatory_sites must be a subset of assignments")

    @property
    def selected(self) -> frozenset[int]:
        return frozenset(s for s, m in self.assignments.items() if m is not ScheduleMode.DEFERRED)


# ---------------------------------------------------------------------------
# Client -> aggregator
# ---------------------------------------------------------------------------


@dataclass
class ClientUpdate:
    """A trained model returned by a selected site. Goes to the aggregator, not the coordinator."""

    site_id: int
    epoch: int
    mode: ScheduleMode
    weights: Any  # e.g. a PyTorch state_dict
    simulated_seconds: float
    bytes_up: int
    completed: bool = True  # False if the site missed the deadline


# ---------------------------------------------------------------------------
# Logging (one JSON line per round) -- Member C builds analysis against this
# ---------------------------------------------------------------------------


@dataclass
class RoundLog:
    """Machine-readable record of one round. Written as one line of JSONL.

    per_site_metrics is filled at evaluation checkpoints for EVERY site (not only
    participants), e.g. {3: {"balanced_acc": 0.71, "loss": 0.88}}; empty otherwise.
    Fields under `sim_only` are simulator ground truth for offline evaluation.
    """

    run_id: str
    seed: int
    policy: str
    epoch: int
    selected: list[int]
    modes: dict[int, str]
    mandatory_sites: list[int]
    deferral_reasons: dict[int, str]
    telemetry: list[dict[str, int]]  # TelemetryReport payloads
    posteriors: list[dict[str, Any]]
    participation_age: dict[int, int]  # server-derived, after this round
    epsilon_spent: dict[int, float]  # cumulative per site
    round_seconds: float
    bytes_up: int
    bytes_down: int
    controller_ms: float
    per_site_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    global_metrics: dict[str, float] = field(default_factory=dict)
    sim_only: dict[str, Any] = field(default_factory=dict)  # e.g. {"true_bins": [...]}

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=_json_default)

    @classmethod
    def from_json(cls, line: str) -> RoundLog:
        d = json.loads(line)
        for key in ("modes", "deferral_reasons", "participation_age", "epsilon_spent", "per_site_metrics"):
            d[key] = {int(k): v for k, v in d[key].items()}  # JSON turns int keys into strings
        return cls(**d)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Component protocols -- implement these, and the pieces plug together
# ---------------------------------------------------------------------------


class Privatizer(Protocol):
    """Site-side: turn true bins into a privatized report and record the budget spent."""

    def privatize(self, true_bins: TrueBins, rng: np.random.Generator) -> tuple[TelemetryReport, list[LedgerEntry]]: ...


class Coordinator(Protocol):
    """Server-side policy. Keeps participation age internally; never sees TrueBins."""

    name: str

    def select(
        self, epoch: int, reports: Mapping[int, TelemetryReport], available: frozenset[int]
    ) -> CohortDecision: ...

    def observe(self, decision: CohortDecision, completed: frozenset[int]) -> None:
        """Called after the round so the coordinator can update participation age."""
        ...


class Aggregator(Protocol):
    def aggregate(self, global_weights: Any, updates: list[ClientUpdate]) -> Any: ...

"""Systems simulation: site speed, availability traces, deadlines, round time and bytes (task A4).

Numpy-only, so CI tests it without torch. All times are *simulated* seconds, never wall-clock,
so runs stay reproducible (Gate G1).

  * Each site gets a fixed compute speed and bandwidth for the run (lognormal around the medians).
  * Availability is a two-state on/off Markov chain per site. `stickiness` makes outages last
    several rounds (maintenance windows) while the long-run rate stays `p_available`.
  * The deadline is `deadline_factor` x the median expected full-round time, so the share of
    stragglers does not depend on the (arbitrary) time units.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Tags for independent random streams. Using SeedSequence spawn keys keeps these distinct from
# the per-site training streams SeedSequence([seed, site_id]), so enabling systems simulation
# never changes a site's training batches.
PROFILE_STREAM, TRACE_STREAM, JITTER_STREAM = 0, 1, 2


def systems_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream,)))


@dataclass(frozen=True)
class SystemsConfig:
    step_seconds: float = 1.0  # seconds per local SGD step at a median-speed site
    speed_sigma: float = 0.6  # lognormal spread of per-site compute speed
    bandwidth_mbps: float = 50.0  # median up/down bandwidth
    bandwidth_sigma: float = 0.8  # lognormal spread of per-site bandwidth
    jitter_sigma: float = 0.15  # round-to-round lognormal noise on compute time
    p_available: float = 0.85  # long-run fraction of rounds a site is online
    stickiness: float = 0.7  # 0 = independent each round; closer to 1 = longer outages
    deadline_factor: float = 1.5  # deadline = factor x median expected full-round time

    def __post_init__(self) -> None:
        if not 0 < self.p_available <= 1:
            raise ValueError("p_available must be in (0, 1]")
        if not 0 <= self.stickiness < 1:
            raise ValueError("stickiness must be in [0, 1)")
        if min(self.step_seconds, self.bandwidth_mbps, self.deadline_factor) <= 0:
            raise ValueError("step_seconds, bandwidth_mbps and deadline_factor must be > 0")
        if min(self.speed_sigma, self.bandwidth_sigma, self.jitter_sigma) < 0:
            raise ValueError("sigmas must be >= 0")


@dataclass(frozen=True)
class SiteProfile:
    site_id: int
    step_seconds: float
    mbps: float


def make_profiles(site_ids: Sequence[int], cfg: SystemsConfig, rng: np.random.Generator) -> dict[int, SiteProfile]:
    step = cfg.step_seconds * rng.lognormal(0.0, cfg.speed_sigma, size=len(site_ids))
    mbps = cfg.bandwidth_mbps * rng.lognormal(0.0, cfg.bandwidth_sigma, size=len(site_ids))
    return {sid: SiteProfile(sid, float(s), float(m)) for sid, s, m in zip(site_ids, step, mbps, strict=True)}


def transfer_seconds(nbytes: int, mbps: float) -> float:
    return nbytes * 8 / (mbps * 1e6)


def expected_seconds(profile: SiteProfile, steps: int, bytes_down: int, bytes_up: int) -> float:
    """Noise-free round time: download + local steps + upload. Feeds the readiness bin (A5)."""
    return (
        transfer_seconds(bytes_down, profile.mbps)
        + steps * profile.step_seconds
        + transfer_seconds(bytes_up, profile.mbps)
    )


def deadline_seconds(
    profiles: Mapping[int, SiteProfile], steps: int, bytes_down: int, bytes_up: int, factor: float
) -> float:
    return factor * float(np.median([expected_seconds(p, steps, bytes_down, bytes_up) for p in profiles.values()]))


@dataclass(frozen=True)
class AvailabilityTrace:
    site_ids: tuple[int, ...]
    available: np.ndarray  # bool, shape (rounds, n_sites)

    def at(self, epoch: int) -> frozenset[int]:
        if not 0 <= epoch < len(self.available):
            raise IndexError(f"trace covers rounds 0..{len(self.available) - 1}, asked for {epoch}")
        return frozenset(s for s, a in zip(self.site_ids, self.available[epoch], strict=True) if a)

    def save(self, path: str | Path) -> None:
        data = {"site_ids": list(self.site_ids), "available": self.available.astype(int).tolist()}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AvailabilityTrace:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(data["site_ids"]), np.array(data["available"], dtype=bool))


def generate_trace(
    site_ids: Sequence[int], rounds: int, cfg: SystemsConfig, rng: np.random.Generator
) -> AvailabilityTrace:
    """Two-state Markov chain per site with stationary on-rate p_available.

    P(on -> off) = (1 - s)(1 - p), P(off -> on) = (1 - s) p. With s = 0 this is an independent
    coin flip each round; larger s keeps a site in its current state longer.
    """
    p, s = cfg.p_available, cfg.stickiness
    n = len(site_ids)
    avail = np.empty((rounds, n), dtype=bool)
    state = rng.random(n) < p
    for t in range(rounds):
        avail[t] = state
        u = rng.random(n)
        state = np.where(state, u >= (1 - s) * (1 - p), u < (1 - s) * p)
    return AvailabilityTrace(tuple(site_ids), avail)


@dataclass(frozen=True)
class RoundTiming:
    site_seconds: dict[int, float]  # simulated seconds each participant needed
    completed: frozenset[int]  # participants that finished within the deadline
    round_seconds: float  # synchronous round: slowest participant, capped at the deadline


def simulate_round(
    profiles: Mapping[int, SiteProfile],
    participants: Iterable[int],
    steps: int | Mapping[int, int],
    bytes_down: int,
    bytes_up: int,
    deadline: float,
    jitter_sigma: float,
    rng: np.random.Generator,
) -> RoundTiming:
    """Per-site times with compute jitter. `steps` may differ per site (A6: compressed = fewer steps)."""
    ids = sorted(participants)  # fixed order -> reproducible jitter draws
    jitter = rng.lognormal(0.0, jitter_sigma, size=len(ids))
    secs = {}
    for sid, j in zip(ids, jitter, strict=True):
        p = profiles[sid]
        k = steps[sid] if isinstance(steps, Mapping) else steps
        compute = k * p.step_seconds * float(j)
        secs[sid] = transfer_seconds(bytes_down, p.mbps) + compute + transfer_seconds(bytes_up, p.mbps)
    completed = frozenset(sid for sid, t in secs.items() if t <= deadline)
    round_s = min(deadline, max(secs.values())) if secs else 0.0
    return RoundTiming(secs, completed, round_s)

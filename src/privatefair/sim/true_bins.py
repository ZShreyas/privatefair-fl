"""Simulator-side ground-truth telemetry bins (task A5).

Each epoch, every site measures locally:
  * utility   -- relative change in its validation loss on the current global model;
  * readiness -- from availability and expected completion time vs the deadline (A4 supplies these;
                 until A6 wires it in, every site defaults to available and fast);
  * shift     -- frozen-encoder style distance to a public reference, computed once per site.

Output is `TrueBins`, which only the privatizer and RoundLog.sim_only may see. The raw numbers
are returned separately for offline evaluation logging.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import nn

from privatefair.data.pathmnist import SiteData
from privatefair.interfaces import TrueBins
from privatefair.sim.bins import readiness_bin, shift_bin, utility_bin
from privatefair.sim.local import TrainConfig, evaluate
from privatefair.sim.shift_detector import ShiftDetector


@dataclass(frozen=True)
class TelemetryConfig:
    utility_tau: float = 0.02
    val_max_samples: int = 512  # fixed per-site val subset, so loss changes reflect the model, not sampling
    fast_fraction: float = 0.5
    shift_thresholds: tuple[float, float] = (2.0, 4.0)
    reference_size: int = 1000


@dataclass(frozen=True)
class ReadinessInput:
    available: bool = True
    expected_seconds: float = 0.0
    deadline_seconds: float = 1.0


class TrueBinsSimulator:
    def __init__(
        self,
        sites: Sequence[SiteData],
        detector: ShiftDetector,
        cfg: TelemetryConfig,
        train_cfg: TrainConfig,
        num_classes: int,
        rng: np.random.Generator,
    ) -> None:
        self.cfg, self.train_cfg, self.num_classes = cfg, train_cfg, num_classes
        self.sites = list(sites)
        self._val_idx = {
            s.site_id: np.sort(rng.choice(len(s.y_val), size=min(len(s.y_val), cfg.val_max_samples), replace=False))
            for s in self.sites
        }
        # A site's data does not change during a run, so shift is measured once (on its training images).
        self.shift_z = {s.site_id: detector.score(s.x_train, s.y_train, rng) for s in self.sites}
        self.shift_bins = {sid: shift_bin(z, cfg.shift_thresholds) for sid, z in self.shift_z.items()}
        self._prev_loss: dict[int, float] = {}

    def compute(
        self, epoch: int, model: nn.Module, readiness: Mapping[int, ReadinessInput] | None = None
    ) -> tuple[dict[int, TrueBins], dict[str, Any]]:
        """Bins for every site at the start of `epoch`, given the current global `model`."""
        readiness = readiness or {}
        bins, raw_loss, raw_delta = {}, {}, {}
        for s in self.sites:
            idx = self._val_idx[s.site_id]
            loss = evaluate(model, s.x_val[idx], s.y_val[idx], self.train_cfg, self.num_classes)["loss"]
            prev = self._prev_loss.get(s.site_id)
            u = utility_bin(prev, loss, self.cfg.utility_tau)
            self._prev_loss[s.site_id] = loss
            r_in = readiness.get(s.site_id, ReadinessInput())
            r = readiness_bin(r_in.available, r_in.expected_seconds, r_in.deadline_seconds, self.cfg.fast_fraction)
            bins[s.site_id] = TrueBins(s.site_id, epoch, u, r, self.shift_bins[s.site_id])
            raw_loss[s.site_id] = loss
            raw_delta[s.site_id] = None if prev is None else (loss - prev) / prev
        raw = {"val_loss": raw_loss, "utility_delta": raw_delta, "shift_z": dict(self.shift_z)}
        return bins, raw


def sample_reference(images: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Public typical-domain reference: a random sample of *unshifted* pooled images.

    Simulation shortcut: it can overlap site data slightly; a real deployment would ship
    a separate public dataset with the coordinator's broadcast.
    """
    return images[np.sort(rng.choice(len(images), size=min(size, len(images)), replace=False))]

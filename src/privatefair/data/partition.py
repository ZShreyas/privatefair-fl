"""Split one pooled dataset into N simulated hospital sites (task A1).

Sites differ in two ways at once:
  * label skew: each class is spread over sites by a Dirichlet(alpha) draw
    (small alpha -> each site sees only a few classes);
  * unequal sizes: that draw is multiplied by per-site lognormal size weights,
    so some sites are large and some small.

Doing both in one step (instead of skewing, then resizing) keeps the skew intact.
Everything here is numpy-only so it is tested in CI without torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PartitionConfig:
    n_sites: int = 8
    dirichlet_alpha: float = 0.5  # lower = more label skew
    size_sigma: float = 0.8  # lognormal spread of site sizes; 0 = equal expected sizes
    min_site_size: int = 200
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)  # train / val / test
    n_shifted: int = 2

    def __post_init__(self) -> None:
        if self.n_sites < 1:
            raise ValueError("n_sites must be >= 1")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")
        if self.size_sigma < 0:
            raise ValueError("size_sigma must be >= 0")
        if len(self.split) != 3 or any(f < 0 for f in self.split) or abs(sum(self.split) - 1.0) > 1e-6:
            raise ValueError(f"split must be three non-negative fractions summing to 1, got {self.split}")
        if not 0 <= self.n_shifted <= self.n_sites:
            raise ValueError("n_shifted must be in [0, n_sites]")


def site_size_weights(n_sites: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Relative site sizes, lognormal and normalized to sum to 1."""
    w = rng.lognormal(mean=0.0, sigma=sigma, size=n_sites)
    return w / w.sum()


def dirichlet_partition(
    labels: np.ndarray,
    n_sites: int,
    alpha: float,
    size_weights: np.ndarray,
    min_site_size: int,
    rng: np.random.Generator,
    max_tries: int = 100,
) -> list[np.ndarray]:
    """Assign every index of `labels` to exactly one site. Returns sorted index arrays, one per site."""
    labels = np.asarray(labels).reshape(-1)
    classes = np.unique(labels)
    for _ in range(max_tries):
        sites: list[list[np.ndarray]] = [[] for _ in range(n_sites)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            q = rng.dirichlet(np.full(n_sites, alpha)) * size_weights
            q /= q.sum()
            cuts = np.floor(np.cumsum(q)[:-1] * len(idx)).astype(int)
            for s, chunk in enumerate(np.split(idx, cuts)):
                sites[s].append(chunk)
        parts = [np.sort(np.concatenate(chunks)) for chunks in sites]
        if min(len(p) for p in parts) >= min_site_size:
            return parts
    raise ValueError(
        f"could not give every site >= {min_site_size} samples in {max_tries} tries; "
        "lower min_site_size, lower size_sigma or raise dirichlet_alpha"
    )


def split_indices(
    idx: np.ndarray, labels: np.ndarray, split: Sequence[float], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one site's indices into train/val/test with exact sizes, approximately stratified by label.

    Trick: give each sample the key (rank within its class + jitter) / class size, sort by it,
    then cut. Every class is spread evenly along the order, so each cut gets its fair share.
    """
    idx = np.asarray(idx)
    y = np.asarray(labels).reshape(-1)[idx]
    key = np.empty(len(idx))
    for c in np.unique(y):
        pos = np.flatnonzero(y == c)
        key[pos] = (rng.permutation(len(pos)) + rng.random(len(pos))) / len(pos)
    order = idx[np.argsort(key, kind="stable")]
    n_val = round(len(idx) * split[1])
    n_test = round(len(idx) * split[2])
    n_train = len(idx) - n_val - n_test
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def choose_shifted_sites(n_sites: int, n_shifted: int, rng: np.random.Generator) -> frozenset[int]:
    """Which sites get the synthetic acquisition shift. Ground truth for evaluation only."""
    return frozenset(int(s) for s in rng.choice(n_sites, size=n_shifted, replace=False))

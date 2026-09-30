"""PathMNIST loading and site construction (task A1).

The official MedMNIST train/val/test splits are pooled, then partitioned into sites, and
each site gets its own train/val/test. We need per-site test sets to measure worst-site
accuracy, which the single official test set cannot give.

`SiteData.shifted` is simulator ground truth for evaluation (log it under RoundLog.sim_only).
It must never reach the coordinator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from privatefair.data.partition import (
    PartitionConfig,
    choose_shifted_sites,
    dirichlet_partition,
    site_size_weights,
    split_indices,
)
from privatefair.data.shift import ShiftSpec, apply_shift


@dataclass
class SiteData:
    site_id: int
    x_train: np.ndarray  # uint8 (N, H, W, C)
    y_train: np.ndarray  # int64 (N,)
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    shifted: bool  # sim-only ground truth

    @property
    def n_train(self) -> int:
        return len(self.y_train)


def load_pathmnist(root: str | Path, size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """Download (first call only) and pool all PathMNIST splits. Returns uint8 NHWC images, int64 labels."""
    from medmnist import PathMNIST  # lazy: medmnist is an [ml] extra

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    xs, ys = [], []
    for split in ("train", "val", "test"):
        ds = PathMNIST(split=split, download=True, root=str(root), size=size)
        xs.append(ds.imgs)
        ys.append(ds.labels.reshape(-1))
    return np.concatenate(xs), np.concatenate(ys).astype(np.int64)


def build_sites(
    images: np.ndarray,
    labels: np.ndarray,
    cfg: PartitionConfig,
    shift_spec: ShiftSpec,
    rng: np.random.Generator,
) -> list[SiteData]:
    """Partition pooled data into sites, split each site, and shift the chosen sites' images."""
    labels = np.asarray(labels).reshape(-1)
    weights = site_size_weights(cfg.n_sites, cfg.size_sigma, rng)
    parts = dirichlet_partition(labels, cfg.n_sites, cfg.dirichlet_alpha, weights, cfg.min_site_size, rng)
    splits = [split_indices(p, labels, cfg.split, rng) for p in parts]
    # Chosen after the partition, so changing n_shifted never changes which samples a site holds.
    shifted = choose_shifted_sites(cfg.n_sites, cfg.n_shifted, rng)

    sites = []
    for sid, (tr, va, te) in enumerate(splits):
        is_shifted = sid in shifted
        # A site's scanner affects all its data, so train/val/test are all shifted.
        xs = [apply_shift(images[i], shift_spec, rng) if is_shifted else images[i] for i in (tr, va, te)]
        sites.append(SiteData(sid, xs[0], labels[tr], xs[1], labels[va], xs[2], labels[te], is_shifted))
    return sites


def load_sites(data_cfg: Mapping[str, Any], seed: int) -> list[SiteData]:
    """Build sites from the `data:` section of an experiment YAML config."""
    if data_cfg.get("dataset", "pathmnist") != "pathmnist":
        raise ValueError(f"unsupported dataset {data_cfg['dataset']!r}")
    rng = np.random.default_rng(seed)
    images, labels = load_pathmnist(data_cfg.get("root", "data/raw"), data_cfg.get("image_size", 28))
    subsample = data_cfg.get("subsample")
    if subsample:
        keep = np.sort(rng.choice(len(labels), size=int(len(labels) * subsample), replace=False))
        images, labels = images[keep], labels[keep]
    part_keys = ("n_sites", "dirichlet_alpha", "size_sigma", "min_site_size", "n_shifted")
    cfg = PartitionConfig(
        **{k: data_cfg[k] for k in part_keys if k in data_cfg},
        **({"split": tuple(data_cfg["split"])} if "split" in data_cfg else {}),
    )
    shift_spec = ShiftSpec(**data_cfg.get("shift", {}))
    return build_sites(images, labels, cfg, shift_spec, rng)

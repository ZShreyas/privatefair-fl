"""A5 done-when on real PathMNIST: A1's shifted sites land in shift bin 2, normal sites in bin 0."""

import numpy as np
import pytest

pytest.importorskip("medmnist")

from privatefair.data.partition import PartitionConfig  # noqa: E402
from privatefair.data.pathmnist import build_sites, load_pathmnist  # noqa: E402
from privatefair.data.shift import ShiftSpec  # noqa: E402
from privatefair.sim.bins import shift_bin  # noqa: E402
from privatefair.sim.shift_detector import ShiftDetector  # noqa: E402


@pytest.mark.ml
@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 4])  # seed 4 was the hardest case during tuning
def test_shifted_sites_land_in_bin_2(seed):
    x_all, y_all = load_pathmnist("data/raw")  # downloads ~200MB on first run
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(y_all), size=int(0.05 * len(y_all)), replace=False))
    x, y = x_all[keep], y_all[keep]
    spec = ShiftSpec(contrast=0.6, blur_sigma=1.0, noise_std=8.0)
    sites = build_sites(x, y, PartitionConfig(n_sites=8, min_site_size=50, n_shifted=2), spec, rng)

    ref = np.sort(rng.choice(len(y), size=1000, replace=False))  # unshifted pooled images
    det = ShiftDetector().fit(x[ref], y[ref], num_classes=9)
    bins = {s.site_id: shift_bin(det.score(s.x_train, s.y_train, rng)) for s in sites}

    assert all(bins[s.site_id] == 2 for s in sites if s.shifted)
    assert all(bins[s.site_id] == 0 for s in sites if not s.shifted)

"""Real-data smoke test for PathMNIST site construction (task A1). Downloads ~200MB on first run."""

import pytest

pytest.importorskip("medmnist")

from privatefair.data import load_sites  # noqa: E402

DATA_CFG = {
    "dataset": "pathmnist",
    "root": "data/raw",
    "subsample": 0.05,
    "n_sites": 8,
    "dirichlet_alpha": 0.5,
    "size_sigma": 0.8,
    "min_site_size": 50,
    "split": [0.7, 0.15, 0.15],
    "n_shifted": 2,
    "shift": {"contrast": 0.6, "blur_sigma": 1.0, "noise_std": 8.0},
}


@pytest.mark.ml
@pytest.mark.slow
def test_load_sites_pathmnist():
    sites = load_sites(DATA_CFG, seed=0)
    assert len(sites) == 8
    assert sum(s.shifted for s in sites) == 2
    for s in sites:
        assert s.x_train.shape[1:] == (28, 28, 3)
        assert len(s.x_train) == len(s.y_train) and len(s.y_test) > 0
        assert s.y_train.min() >= 0 and s.y_train.max() <= 8
    again = load_sites(DATA_CFG, seed=0)
    assert [len(s.y_train) for s in sites] == [len(s.y_train) for s in again]

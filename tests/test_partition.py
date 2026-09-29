"""Tests for site partitioning and site construction (task A1). Numpy only, no download."""

import numpy as np
import pytest

from privatefair.data import (
    PartitionConfig,
    ShiftSpec,
    build_sites,
    choose_shifted_sites,
    dirichlet_partition,
    site_size_weights,
    split_indices,
)

N, N_CLASSES = 9000, 9


@pytest.fixture
def labels():
    return np.random.default_rng(123).integers(0, N_CLASSES, size=N)


@pytest.fixture
def images():
    return np.random.default_rng(456).integers(0, 256, size=(N, 8, 8, 3), dtype=np.uint8)


def _partition(labels, seed, alpha=0.5, sigma=0.8, n_sites=8, min_size=100):
    rng = np.random.default_rng(seed)
    w = site_size_weights(n_sites, sigma, rng)
    return dirichlet_partition(labels, n_sites, alpha, w, min_size, rng)


def _mean_label_entropy(labels, parts):
    ents = []
    for p in parts:
        freq = np.bincount(labels[p], minlength=N_CLASSES) / len(p)
        freq = freq[freq > 0]
        ents.append(-(freq * np.log(freq)).sum())
    return float(np.mean(ents))


def test_same_seed_same_partition(labels):
    a, b = _partition(labels, seed=0), _partition(labels, seed=0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_different_seed_different_partition(labels):
    a, b = _partition(labels, seed=0), _partition(labels, seed=1)
    assert not all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_sites_disjoint_and_cover_everything(labels):
    parts = _partition(labels, seed=0)
    allidx = np.concatenate(parts)
    assert len(allidx) == N
    assert np.array_equal(np.sort(allidx), np.arange(N))


def test_min_size_and_unequal_sizes(labels):
    parts = _partition(labels, seed=0, min_size=100)
    sizes = [len(p) for p in parts]
    assert min(sizes) >= 100
    assert max(sizes) > 1.5 * min(sizes)


def test_impossible_min_size_raises(labels):
    with pytest.raises(ValueError, match="min_site_size"):
        _partition(labels, seed=0, min_size=N)


def test_lower_alpha_gives_more_label_skew(labels):
    skewed = _partition(labels, seed=0, alpha=0.1, sigma=0.0, min_size=1)
    iid = _partition(labels, seed=0, alpha=100.0, sigma=0.0, min_size=1)
    assert _mean_label_entropy(labels, skewed) < _mean_label_entropy(labels, iid) - 0.3


def test_split_sizes_exact_and_disjoint(labels):
    idx = np.arange(1001)
    tr, va, te = split_indices(idx, labels, (0.7, 0.15, 0.15), np.random.default_rng(0))
    assert (len(va), len(te)) == (round(1001 * 0.15), round(1001 * 0.15))
    assert len(tr) + len(va) + len(te) == 1001
    assert np.array_equal(np.sort(np.concatenate([tr, va, te])), idx)


def test_split_is_roughly_stratified(labels):
    idx = np.arange(N)
    _, _, te = split_indices(idx, labels, (0.7, 0.15, 0.15), np.random.default_rng(0))
    overall = np.bincount(labels, minlength=N_CLASSES) / N
    in_test = np.bincount(labels[te], minlength=N_CLASSES) / len(te)
    assert np.abs(overall - in_test).max() < 0.01


def test_choose_shifted_sites():
    s = choose_shifted_sites(8, 3, np.random.default_rng(0))
    assert len(s) == 3 and s <= set(range(8))


@pytest.mark.parametrize(
    "kwargs", [{"n_sites": 0}, {"dirichlet_alpha": 0}, {"split": (0.5, 0.5, 0.5)}, {"n_shifted": 9}]
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        PartitionConfig(**kwargs)


def test_build_sites_deterministic_and_shifts_only_chosen_sites(images, labels):
    cfg = PartitionConfig(n_sites=6, min_site_size=100, n_shifted=2)
    spec = ShiftSpec(contrast=0.5, blur_sigma=1.0, noise_std=5.0)
    a = build_sites(images, labels, cfg, spec, np.random.default_rng(7))
    b = build_sites(images, labels, cfg, spec, np.random.default_rng(7))
    assert [s.shifted for s in a] == [s.shifted for s in b]
    assert sum(s.shifted for s in a) == 2
    for sa, sb in zip(a, b, strict=True):
        assert np.array_equal(sa.x_train, sb.x_train) and np.array_equal(sa.y_test, sb.y_test)

    # Unshifted sites hold the original pixels; shifted sites do not.
    unshifted = build_sites(images, labels, cfg, ShiftSpec(), np.random.default_rng(7))
    for s, u in zip(a, unshifted, strict=True):
        assert np.array_equal(s.y_train, u.y_train)  # same partition regardless of shift
        assert np.array_equal(s.x_train, u.x_train) != s.shifted

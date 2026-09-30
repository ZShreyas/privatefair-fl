"""Tests for the synthetic acquisition shift (task A1)."""

import numpy as np
import pytest

from privatefair.data import ShiftSpec, apply_shift


@pytest.fixture
def images():
    return np.random.default_rng(0).integers(30, 226, size=(16, 28, 28, 3), dtype=np.uint8)


def _high_freq_energy(x):
    x = x.astype(np.float64)
    return float((np.diff(x, axis=1) ** 2).mean() + (np.diff(x, axis=2) ** 2).mean())


def test_identity_spec_returns_unchanged_copy(images):
    out = apply_shift(images, ShiftSpec(), np.random.default_rng(0))
    assert np.array_equal(out, images) and out is not images


def test_shape_dtype_and_determinism(images):
    spec = ShiftSpec(contrast=0.6, blur_sigma=1.0, noise_std=8.0)
    a = apply_shift(images, spec, np.random.default_rng(1))
    b = apply_shift(images, spec, np.random.default_rng(1))
    assert a.shape == images.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, images)


def test_contrast_lowers_std(images):
    out = apply_shift(images, ShiftSpec(contrast=0.5), np.random.default_rng(0))
    assert out.astype(float).std() < 0.6 * images.astype(float).std()


def test_blur_lowers_high_frequency_energy(images):
    out = apply_shift(images, ShiftSpec(blur_sigma=1.0), np.random.default_rng(0))
    assert _high_freq_energy(out) < 0.5 * _high_freq_energy(images)


def test_rejects_non_uint8():
    with pytest.raises(ValueError):
        apply_shift(np.zeros((1, 4, 4, 3)), ShiftSpec(contrast=0.5), np.random.default_rng(0))


def test_rejects_negative_params():
    with pytest.raises(ValueError):
        ShiftSpec(noise_std=-1.0)

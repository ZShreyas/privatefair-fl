"""Synthetic acquisition shift for "shifted" sites (task A1).

Mimics a hospital with a different scanner/stain protocol: lower contrast, a little blur,
sensor noise. Applied once to uint8 NHWC arrays, so it is deterministic and costs nothing
during training. Numpy-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShiftSpec:
    contrast: float = 1.0  # < 1 pulls pixels toward the per-image mean
    blur_sigma: float = 0.0  # Gaussian blur std in pixels; 0 = off
    noise_std: float = 0.0  # additive Gaussian noise std in 0-255 units; 0 = off

    def __post_init__(self) -> None:
        if self.contrast < 0 or self.blur_sigma < 0 or self.noise_std < 0:
            raise ValueError(f"shift parameters must be >= 0, got {self}")

    @property
    def is_identity(self) -> bool:
        return self.contrast == 1.0 and self.blur_sigma == 0.0 and self.noise_std == 0.0


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, math.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur over H and W of a float NHWC array, reflect padding."""
    k = _gaussian_kernel(sigma)
    r = len(k) // 2
    for axis in (1, 2):
        pad = [(0, 0)] * 4
        pad[axis] = (r, r)
        xp = np.pad(x, pad, mode="reflect")
        n = x.shape[axis]
        x = sum(w * np.take(xp, np.arange(i, i + n), axis=axis) for i, w in enumerate(k))
    return x


def apply_shift(images: np.ndarray, spec: ShiftSpec, rng: np.random.Generator) -> np.ndarray:
    """Return a shifted copy of uint8 images shaped (N, H, W, C)."""
    if images.dtype != np.uint8 or images.ndim != 4:
        raise ValueError(f"expected uint8 array shaped (N, H, W, C), got {images.dtype} {images.shape}")
    if spec.is_identity:
        return images.copy()
    x = images.astype(np.float64)
    if spec.contrast != 1.0:
        mean = x.mean(axis=(1, 2, 3), keepdims=True)
        x = mean + spec.contrast * (x - mean)
    if spec.blur_sigma > 0:
        x = _blur(x, spec.blur_sigma)
    if spec.noise_std > 0:
        x = x + rng.normal(0.0, spec.noise_std, size=x.shape)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)

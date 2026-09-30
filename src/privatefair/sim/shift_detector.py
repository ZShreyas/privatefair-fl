"""Frozen-encoder shift detector (task A5).

A site's "style" is the per-channel mean and std of early ResNet-18 feature maps (up to
`layer1`), pooled over its images. Early layers respond to contrast, blur and noise (how an
image was acquired) more than to which tissue class is shown.

Label skew still leaks into pooled style (different tissues have different textures), so the
public reference is re-weighted to the site's own label mix before comparing. The site knows
its labels locally; only the resulting bin ever leaves it. On PathMNIST (5 seeds, 8 sites)
this took normal sites from z ~3 (max 7) down to z ~1 (max 1.3), while shifted sites stay >= 9.

Score: z = distance(site style, reweighted reference style) / mean distance of random
reweighted reference subsets of the same size. z ~ 1 means "looks like the reference domain".
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from privatefair.sim.model import preprocess


def _style(m: np.ndarray, q: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pooled channel mean and std from per-image first/second moments, optionally weighted."""
    w = np.full(len(m), 1.0 / len(m)) if w is None else w / w.sum()
    mean = w @ m
    std = np.sqrt(np.maximum(w @ q - mean**2, 1e-12))
    return np.concatenate([mean, std])


class ShiftDetector:
    def __init__(self, input_size: int = 64, pretrained: bool = True, batch_size: int = 256, n_null: int = 50) -> None:
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.input_size, self.batch_size, self.n_null = input_size, batch_size, n_null
        self._ref_m: np.ndarray | None = None  # per-image channel means, (R, C)
        self._ref_q: np.ndarray | None = None  # per-image channel second moments, (R, C)

    @torch.no_grad()
    def _moments(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ms, qs = [], []
        for start in range(0, len(images), self.batch_size):
            f = self.encoder(preprocess(images[start : start + self.batch_size], self.input_size))
            ms.append(f.mean(dim=(2, 3)).numpy())
            qs.append((f * f).mean(dim=(2, 3)).numpy())
        return np.concatenate(ms).astype(np.float64), np.concatenate(qs).astype(np.float64)

    def fit(self, reference_images: np.ndarray, reference_labels: np.ndarray, num_classes: int) -> ShiftDetector:
        """Reference = public, labeled images from the typical domain."""
        self._ref_m, self._ref_q = self._moments(reference_images)
        self._ref_y = np.asarray(reference_labels).reshape(-1)
        self._num_classes = num_classes
        self._ref_p = np.bincount(self._ref_y, minlength=num_classes) / len(self._ref_y)
        per_image = np.stack([_style(m[None], q[None]) for m, q in zip(self._ref_m, self._ref_q, strict=True)])
        self._scale = per_image.std(axis=0) + 1e-8  # makes channels comparable
        return self

    @property
    def max_sample_size(self) -> int:
        """Site samples are capped at half the reference so the null subsets stay varied."""
        if self._ref_m is None:
            raise RuntimeError("call fit() first")
        return len(self._ref_m) // 2

    def score(self, images: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> float:
        if self._ref_m is None:
            raise RuntimeError("call fit() before score()")
        n = min(len(images), self.max_sample_size)
        idx = np.sort(rng.choice(len(images), size=n, replace=False))
        site_p = np.bincount(np.asarray(labels).reshape(-1)[idx], minlength=self._num_classes) / n
        # Reference re-weighted to the site's label mix (classes absent from the site get weight 0).
        w = site_p[self._ref_y] / np.maximum(self._ref_p[self._ref_y], 1e-12)
        if w.sum() == 0:  # site classes never seen in the reference: fall back to the plain reference
            w = np.ones(len(self._ref_y))
        ref_style = _style(self._ref_m, self._ref_q, w)

        def dist(style: np.ndarray) -> float:
            return float(np.sqrt(np.mean(((style - ref_style) / self._scale) ** 2)))

        d_site = dist(_style(*self._moments(images[idx])))
        p = w / w.sum()
        null = []
        for _ in range(self.n_null):
            j = rng.choice(len(self._ref_y), size=n, replace=True, p=p)
            null.append(dist(_style(self._ref_m[j], self._ref_q[j])))
        return d_site / float(np.mean(null))

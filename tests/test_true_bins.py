"""Shift detector and TrueBins simulator on synthetic data (task A5). Random-init encoder: no download."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from privatefair.data.pathmnist import SiteData  # noqa: E402
from privatefair.data.shift import ShiftSpec, apply_shift  # noqa: E402
from privatefair.sim.local import TrainConfig  # noqa: E402
from privatefair.sim.model import build_model  # noqa: E402
from privatefair.sim.shift_detector import ShiftDetector  # noqa: E402
from privatefair.sim.true_bins import ReadinessInput, TelemetryConfig, TrueBinsSimulator  # noqa: E402

pytestmark = pytest.mark.ml

SHIFT = ShiftSpec(contrast=0.6, blur_sigma=1.0, noise_std=8.0)
TRAIN = TrainConfig(input_size=32, eval_batch_size=64)


def _images(n, seed):
    """Textured synthetic 'tissue': smooth random blobs plus fine noise."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(40, 216, size=(n, 7, 7, 3)).repeat(4, axis=1).repeat(4, axis=2)
    return np.clip(coarse + rng.normal(0, 20, size=coarse.shape), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def detector():
    torch.manual_seed(0)
    ref_labels = np.random.default_rng(1).integers(0, 3, size=200)
    return ShiftDetector(input_size=32, pretrained=False, n_null=20).fit(_images(200, seed=1), ref_labels, 3)


def test_shifted_images_score_higher(detector):
    rng = np.random.default_rng(0)
    typical = _images(100, seed=2)
    labels = np.random.default_rng(4).integers(0, 3, size=100)
    z_typical = detector.score(typical, labels, rng)
    z_shifted = detector.score(apply_shift(typical, SHIFT, np.random.default_rng(3)), labels, rng)
    assert z_typical < 2.0
    assert z_shifted > 2 * z_typical


def test_score_requires_fit():
    with pytest.raises(RuntimeError):
        ShiftDetector(input_size=32, pretrained=False).score(_images(4, 0), np.zeros(4, int), np.random.default_rng(0))


def _sites():
    sites = []
    for sid in range(3):
        rng = np.random.default_rng(10 + sid)
        x = _images(60, seed=20 + sid)
        if sid == 2:
            x = apply_shift(x, SHIFT, rng)
        y = rng.integers(0, 3, size=60)
        sites.append(SiteData(sid, x[:40], y[:40], x[40:], y[40:], x[40:], y[40:], shifted=(sid == 2)))
    return sites


def _simulator(detector, seed=0):
    return TrueBinsSimulator(_sites(), detector, TelemetryConfig(), TRAIN, 3, np.random.default_rng(seed))


def test_simulator_bins_per_site(detector):
    torch.manual_seed(0)
    model = build_model(3, pretrained=False)
    sim = _simulator(detector)
    bins, raw = sim.compute(0, model)
    assert set(bins) == {0, 1, 2}
    assert all(b.epoch == 0 and b.utility == 2 and b.readiness == 3 for b in bins.values())  # epoch 0 defaults
    assert raw["utility_delta"] == {0: None, 1: None, 2: None}
    assert raw["shift_z"][2] > max(raw["shift_z"][0], raw["shift_z"][1])


def test_readiness_input_is_used(detector):
    torch.manual_seed(0)
    model = build_model(3, pretrained=False)
    bins, _ = _simulator(detector).compute(0, model, {1: ReadinessInput(available=False)})
    assert bins[1].readiness == 0 and bins[0].readiness == 3


def test_second_epoch_uses_loss_change(detector):
    torch.manual_seed(0)
    model = build_model(3, pretrained=False)
    sim = _simulator(detector)
    sim.compute(0, model)
    bins, raw = sim.compute(1, model)  # same model -> zero change -> stable
    assert all(d == pytest.approx(0.0) for d in raw["utility_delta"].values())
    assert all(b.utility == 2 for b in bins.values())


def test_deterministic(detector):
    torch.manual_seed(0)
    model = build_model(3, pretrained=False)
    a = _simulator(detector, seed=5).compute(0, model)
    b = _simulator(detector, seed=5).compute(0, model)
    assert a == b

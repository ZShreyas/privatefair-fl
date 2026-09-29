"""CPU smoke tests for the model, train_local and evaluate (task A2). Random init: no download."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from privatefair.sim.local import TrainConfig, evaluate, train_local  # noqa: E402
from privatefair.sim.model import build_model, get_weights, preprocess, set_weights  # noqa: E402

pytestmark = pytest.mark.ml

CFG = TrainConfig(local_steps=3, batch_size=8, input_size=32, eval_batch_size=16)


@pytest.fixture
def data():
    """Two easily separable classes: dark images (0) vs bright images (1)."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 16)
    x = np.where(y[:, None, None, None] == 0, 40, 200) + rng.integers(-20, 21, size=(32, 28, 28, 3))
    return x.astype(np.uint8), y


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_model(num_classes=2, pretrained=False)


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_preprocess_shape_and_dtype(data):
    t = preprocess(data[0][:4], input_size=32)
    assert t.shape == (4, 3, 32, 32) and t.dtype == torch.float32


def test_weights_round_trip(model):
    w = get_weights(model)
    other = build_model(num_classes=2, pretrained=False)
    set_weights(other, w)
    assert _same(get_weights(other), w)


def test_train_local_changes_weights(model, data):
    before = get_weights(model)
    res = train_local(model, *data, CFG, np.random.default_rng(0))
    assert res.steps == CFG.local_steps and np.isfinite(res.mean_loss)
    assert not _same(before, res.weights)


def test_train_local_is_deterministic(model, data):
    start = get_weights(model)
    a = train_local(model, *data, CFG, np.random.default_rng(1)).weights
    set_weights(model, start)
    b = train_local(model, *data, CFG, np.random.default_rng(1)).weights
    assert _same(a, b)


def test_loss_decreases_on_learnable_data(model, data):
    # Small lr: at 0.01 a random-init ResNet on 32 images is erratic (BatchNorm running stats swing).
    cfg = TrainConfig(local_steps=25, batch_size=16, input_size=32, lr=0.001, eval_batch_size=32)
    before = evaluate(model, *data, cfg, num_classes=2)["loss"]
    train_local(model, *data, cfg, np.random.default_rng(0))
    after = evaluate(model, *data, cfg, num_classes=2)
    assert after["loss"] < 0.5 * before
    assert after["balanced_acc"] >= 0.9


def test_evaluate_keys_and_ranges(model, data):
    m = evaluate(model, *data, CFG, num_classes=2)
    assert set(m) == {"balanced_acc", "acc", "loss"}
    assert 0 <= m["balanced_acc"] <= 1 and 0 <= m["acc"] <= 1 and m["loss"] >= 0


@pytest.mark.slow
def test_pretrained_weights_load():
    m = build_model(num_classes=9, pretrained=True)  # downloads ~45MB on first run
    assert m.fc.out_features == 9

"""Device, AMP and freeze_backbone options (task perf-device). Random init, no download.

CPU tests always run; the CUDA tests run only on a machine where torch.cuda.is_available().
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from privatefair.data.pathmnist import SiteData  # noqa: E402
from privatefair.runlog import RunLogger, read_round_logs  # noqa: E402
from privatefair.sim.fedavg_loop import FLConfig, run_fedavg  # noqa: E402
from privatefair.sim.local import TrainConfig, evaluate, train_local  # noqa: E402
from privatefair.sim.model import (  # noqa: E402
    build_model,
    count_params,
    get_weights,
    resolve_device,
    set_weights,
    trainable_keys,
)

pytestmark = pytest.mark.ml

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
CFG = TrainConfig(local_steps=3, batch_size=8, input_size=32, eval_batch_size=16)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 16)
    x = np.where(y[:, None, None, None] == 0, 40, 200) + rng.integers(-20, 21, size=(32, 28, 28, 3))
    return x.astype(np.uint8), y


def _model(freeze=False):
    torch.manual_seed(0)
    return build_model(num_classes=2, pretrained=False, freeze_backbone=freeze)


def _same(a, b, keys=None):
    return all(torch.equal(a[k], b[k]) for k in (keys or a))


# --- device resolution -------------------------------------------------------------------------
def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("tpu")
    with pytest.raises(ValueError, match="device must be one of"):
        TrainConfig(device="gpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without a GPU")
def test_cuda_without_gpu_fails_loudly():
    with pytest.raises(ValueError, match="cuda"):
        resolve_device("cuda")


# --- AMP ---------------------------------------------------------------------------------------
def test_amp_is_noop_on_cpu(data):
    m = _model()
    start = get_weights(m)
    a = train_local(m, *data, CFG, np.random.default_rng(1)).weights
    set_weights(m, start)
    cfg = TrainConfig(local_steps=3, batch_size=8, input_size=32, eval_batch_size=16, amp=True)
    assert not cfg.amp_active
    assert _same(a, train_local(m, *data, cfg, np.random.default_rng(1)).weights)


# --- freeze_backbone ---------------------------------------------------------------------------
def test_freeze_backbone_trainable_set():
    m = _model(freeze=True)
    trainable, total = count_params(m)
    assert 0 < trainable < total
    assert all(p.requires_grad for p in list(m.layer4[-1].parameters()) + list(m.fc.parameters()))
    assert not any(p.requires_grad for p in m.layer1.parameters())
    keys = trainable_keys(m)
    assert all(k.startswith(("layer4.1.", "fc.")) for k in keys)
    assert "layer4.1.bn2.running_mean" in keys and "layer4.0.conv1.weight" not in keys
    assert trainable_keys(_model(freeze=False)) is None


def test_freeze_backbone_train_local(data):
    m = _model(freeze=True)
    before = get_weights(m)
    res = train_local(m, *data, CFG, np.random.default_rng(0))
    keys = trainable_keys(m)
    assert set(res.weights) == set(keys)
    after = get_weights(m)
    frozen = [k for k in before if k not in keys]
    assert frozen and _same(before, after, frozen)  # params and BN running stats untouched
    assert not _same(before, after, keys)  # the unfrozen part did learn
    assert np.isfinite(evaluate(m, *data, CFG, num_classes=2)["loss"])


def test_frozen_run_matches_nothing_lost(data):
    """Merging the trainable subset back into the global dict is a full, loadable state."""
    m = _model(freeze=True)
    g = get_weights(m)
    res = train_local(m, *data, CFG, np.random.default_rng(0))
    set_weights(m, {**g, **res.weights})  # strict load: no missing / extra keys


# --- full loop ---------------------------------------------------------------------------------
def _sites():
    rng = np.random.default_rng(0)

    def arr(n):
        return rng.integers(0, 256, size=(n, 28, 28, 3), dtype=np.uint8), rng.integers(0, 3, size=n)

    out = []
    for sid in range(3):
        (xtr, ytr), (xva, yva), (xte, yte) = arr(16), arr(4), arr(8)
        out.append(SiteData(sid, xtr, ytr, xva, yva, xte, yte, shifted=(sid == 2)))
    return out


def _run(tmp_path, train_cfg, freeze):
    logger = RunLogger(tmp_path, "r")
    run_fedavg(
        _sites(),
        num_classes=3,
        train_cfg=train_cfg,
        fl_cfg=FLConfig(rounds=2, eval_every=1),
        seed=0,
        logger=logger,
        pretrained=False,
        freeze_backbone=freeze,
    )
    return read_round_logs(logger.path)


def test_loop_freeze_backbone_reduces_bytes(tmp_path):
    full = _run(tmp_path / "full", CFG, freeze=False)
    frozen = _run(tmp_path / "frozen", CFG, freeze=True)
    assert 0 < frozen[0].bytes_up < full[0].bytes_up
    assert "worst_balanced_acc" in frozen[-1].global_metrics


def test_run_info_written(tmp_path):
    _run(tmp_path, CFG, freeze=True)
    import json

    info = json.loads((tmp_path / "r" / "run_info.json").read_text()) if (tmp_path / "r").exists() else None
    if info is None:  # RunLogger layout: fall back to searching
        info = json.loads(next(tmp_path.rglob("run_info.json")).read_text())
    assert info["device"] == "cpu" and info["amp"] is False and info["freeze_backbone"] is True
    assert info["trainable_params"] < info["total_params"]


# --- CUDA (skipped without a GPU) --------------------------------------------------------------
@CUDA
def test_cuda_train_and_evaluate(data):
    cfg = TrainConfig(local_steps=3, batch_size=8, input_size=32, eval_batch_size=16, device="cuda")
    m = _model()
    res = train_local(m, *data, cfg, np.random.default_rng(0))
    assert next(m.parameters()).is_cuda
    assert all(not v.is_cuda for v in res.weights.values())  # weights returned on CPU
    assert np.isfinite(res.mean_loss)
    assert set(evaluate(m, *data, cfg, 2)) == {"balanced_acc", "acc", "loss"}


@CUDA
def test_cuda_amp_trains(data):
    cfg = TrainConfig(local_steps=3, batch_size=8, input_size=32, device="cuda", amp=True)
    assert cfg.amp_active
    assert np.isfinite(train_local(_model(), *data, cfg, np.random.default_rng(0)).mean_loss)


@CUDA
def test_cuda_freeze_loop(tmp_path):
    cfg = TrainConfig(local_steps=2, batch_size=8, input_size=32, eval_batch_size=16, device="cuda")
    logs = _run(tmp_path, cfg, freeze=True)
    assert np.isfinite(logs[-1].global_metrics["mean_balanced_acc"])

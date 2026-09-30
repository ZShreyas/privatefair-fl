"""FedAvg loop tests on tiny fake sites (task A3). Random init, no download.

Gate G1: the same seed twice gives byte-identical rounds.jsonl.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from privatefair.data.pathmnist import SiteData  # noqa: E402
from privatefair.runlog import RunLogger, read_round_logs  # noqa: E402
from privatefair.sim.fedavg_loop import FLConfig, run_fedavg  # noqa: E402
from privatefair.sim.local import TrainConfig  # noqa: E402

pytestmark = pytest.mark.ml

TRAIN = TrainConfig(local_steps=2, batch_size=8, input_size=32, eval_batch_size=32)
FL = FLConfig(rounds=3, eval_every=2)


def _sites():
    rng = np.random.default_rng(0)

    def arr(n):
        return rng.integers(0, 256, size=(n, 28, 28, 3), dtype=np.uint8), rng.integers(0, 3, size=n)

    sites = []
    for sid in range(3):
        (xtr, ytr), (xva, yva), (xte, yte) = arr(16), arr(4), arr(8)
        sites.append(SiteData(sid, xtr, ytr, xva, yva, xte, yte, shifted=(sid == 2)))
    return sites


def _run(tmp_path, seed, run_id):
    logger = RunLogger(tmp_path, run_id)
    run_fedavg(_sites(), num_classes=3, train_cfg=TRAIN, fl_cfg=FL, seed=seed, logger=logger, pretrained=False)
    return logger.path


def test_gate_g1_same_seed_identical_logs(tmp_path):
    a = _run(tmp_path / "first", seed=0, run_id="r").read_bytes()
    b = _run(tmp_path / "second", seed=0, run_id="r").read_bytes()
    assert a == b


def test_different_seed_different_logs(tmp_path):
    a = read_round_logs(_run(tmp_path, seed=0, run_id="a"))
    b = read_round_logs(_run(tmp_path, seed=1, run_id="b"))
    assert [r.global_metrics for r in a] != [r.global_metrics for r in b]


def test_log_contents(tmp_path):
    logs = read_round_logs(_run(tmp_path, seed=0, run_id="a"))
    assert [r.epoch for r in logs] == [0, 1, 2]
    # Checkpoints at round 2 (eval_every) and the last round (3); none at round 1.
    assert [bool(r.per_site_metrics) for r in logs] == [False, True, True]
    for r in logs:
        assert r.selected == [0, 1, 2] and r.policy == "fedavg_all"
        assert r.telemetry == [] and set(r.epsilon_spent.values()) == {0.0}
        assert r.sim_only == {"shifted_sites": [2]}
    last = logs[-1]
    assert set(last.per_site_metrics) == {0, 1, 2}
    assert last.global_metrics["worst_balanced_acc"] <= last.global_metrics["mean_balanced_acc"]


def test_verbose_prints_one_line_per_round_without_changing_logs(tmp_path, capsys):
    quiet = _run(tmp_path / "quiet", seed=0, run_id="r").read_bytes()
    logger = RunLogger(tmp_path / "loud", "r")
    kwargs = {"num_classes": 3, "train_cfg": TRAIN, "fl_cfg": FL, "seed": 0, "logger": logger, "pretrained": False}
    run_fedavg(_sites(), **kwargs, verbose=True)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("round")]
    assert len(lines) == FL.rounds and "worst_acc" in lines[-1]
    assert logger.path.read_bytes() == quiet  # progress printing never touches the log

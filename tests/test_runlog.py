"""Tests for the RoundLog JSONL writer/reader (task A3). No torch."""

import json

from privatefair.interfaces import RoundLog
from privatefair.runlog import RunLogger, read_round_logs


def _log(epoch):
    return RoundLog(
        run_id="r",
        seed=0,
        policy="fedavg_all",
        epoch=epoch,
        selected=[0, 1],
        modes={0: "full", 1: "full"},
        mandatory_sites=[],
        deferral_reasons={},
        telemetry=[],
        posteriors=[],
        participation_age={0: 0, 1: 0},
        epsilon_spent={0: 0.0, 1: 0.0},
        round_seconds=0.0,
        bytes_up=10,
        bytes_down=10,
        controller_ms=0.0,
        per_site_metrics={0: {"balanced_acc": 0.5, "loss": 1.2}} if epoch else {},
        global_metrics={"mean_train_loss": 1.0},
        sim_only={"shifted_sites": [1]},
    )


def test_round_trip(tmp_path):
    logger = RunLogger(tmp_path, "r", config={"seed": 0, "fl": {"rounds": 2}})
    logs = [_log(0), _log(1)]
    for r in logs:
        logger.log(r)
    assert read_round_logs(logger.path) == logs
    assert json.loads((tmp_path / "r" / "config.json").read_text())["fl"]["rounds"] == 2


def test_rerun_same_run_id_starts_clean(tmp_path):
    RunLogger(tmp_path, "r").log(_log(0))
    logger = RunLogger(tmp_path, "r")
    logger.log(_log(1))
    assert [r.epoch for r in read_round_logs(logger.path)] == [1]

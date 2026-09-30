"""Integration tests for the policy-driven loop (task A6). Fake sites, random-init models, no download.

Gate G3: the coordinator only ever receives TelemetryReport payloads (plus epoch and the available set).
"""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from privatefair.coordinator.privatefair import PrivateFairCoordinator  # noqa: E402
from privatefair.data.pathmnist import SiteData  # noqa: E402
from privatefair.interfaces import CohortDecision, ScheduleMode, TelemetryReport, TrueBins  # noqa: E402
from privatefair.privacy.ledger import PrivacyBudgetExceeded  # noqa: E402
from privatefair.runlog import RunLogger, read_comparable, read_round_logs  # noqa: E402
from privatefair.sim.fedavg_loop import FLConfig, run_fedavg  # noqa: E402
from privatefair.sim.local import TrainConfig  # noqa: E402
from privatefair.sim.policies import PolicySpec, PrivacyConfig, build_policy  # noqa: E402
from privatefair.sim.systems import SystemsConfig  # noqa: E402

pytestmark = pytest.mark.ml

TRAIN = TrainConfig(local_steps=4, batch_size=8, input_size=32, eval_batch_size=32)


def _images(n, rng):
    coarse = rng.integers(40, 216, size=(n, 7, 7, 3)).repeat(4, axis=1).repeat(4, axis=2)
    return np.clip(coarse + rng.normal(0, 20, size=coarse.shape), 0, 255).astype(np.uint8)


def _sites():
    rng = np.random.default_rng(0)
    sites = []
    for sid in range(3):
        x, y = _images(28, rng), rng.integers(0, 3, size=28)
        sites.append(SiteData(sid, x[:16], y[:16], x[16:22], y[16:22], x[22:], y[22:], shifted=(sid == 2)))
    return sites


REFERENCE = (_images(60, np.random.default_rng(1)), np.random.default_rng(2).integers(0, 3, size=60))


def _run(tmp_path, spec, privacy=None, systems=None, rounds=3, seed=0):
    logger = RunLogger(tmp_path, "r")
    run_fedavg(
        _sites(), num_classes=3, train_cfg=TRAIN, fl_cfg=FLConfig(rounds=rounds, eval_every=rounds), seed=seed,
        logger=logger, pretrained=False, systems=systems, policy=spec, privacy=privacy or PrivacyConfig(),
        reference=REFERENCE,
    )  # fmt: skip
    return logger


def _privatefair(capacity=2, max_age=3, eps=1.0):
    coord = PrivateFairCoordinator(capacity=capacity, max_age=max_age, rng=np.random.default_rng(0), epsilon=eps)
    return coord


class Spy:
    """Wraps a coordinator and records everything it is given."""

    def __init__(self, inner):
        self.inner, self.name = inner, "spy"
        self.select_args, self.observe_args = [], []

    def select(self, epoch, reports, available):
        self.select_args.append((epoch, reports, available))
        return self.inner.select(epoch, reports, available)

    def observe(self, decision, completed):
        self.observe_args.append((decision, completed))
        self.inner.observe(decision, completed)


def _walk(obj):
    """Yield every object reachable through containers and dataclass fields."""
    yield obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(k)
            yield from _walk(v)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            yield from _walk(v)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _walk(getattr(obj, f.name))


def test_gate_g3_coordinator_only_receives_telemetry_reports(tmp_path):
    spy = Spy(_privatefair())
    systems = SystemsConfig(p_available=0.7, stickiness=0.5)
    _run(tmp_path, PolicySpec("spy", spy, uses_telemetry=True), systems=systems, rounds=4)

    assert len(spy.select_args) == 4 and len(spy.observe_args) == 4
    for epoch, reports, available in spy.select_args:
        assert isinstance(epoch, int) and isinstance(available, frozenset)
        assert set(reports) <= available  # offline sites send nothing
        for sid, rep in reports.items():
            assert type(rep) is TelemetryReport and rep.site_id == sid and rep.epoch == epoch
            assert set(rep.to_payload()) == TelemetryReport.PAYLOAD_KEYS
        for obj in _walk((epoch, reports, available)):
            # Only ints (ids, epoch, bins) and the report objects: no truth, no raw floats, no arrays or tensors.
            assert type(obj) in (int, dict, frozenset, tuple, TelemetryReport, str), type(obj)
            assert not isinstance(obj, (TrueBins, SiteData, np.ndarray, torch.Tensor, float))
    for decision, completed in spy.observe_args:
        assert isinstance(decision, CohortDecision) and completed <= decision.selected


def test_log_contents_privatefair(tmp_path):
    logger = _run(tmp_path, PolicySpec("privatefair", _privatefair(), uses_telemetry=True))
    for r in read_round_logs(logger.path):
        assert len(r.selected) == 2 and len(r.telemetry) == 3 and len(r.posteriors) == 3
        assert {p["site_id"] for p in r.posteriors} == {0, 1, 2}
        assert len(r.sim_only["true_bins"]) == 3 and "shift_z" in r.sim_only["raw_telemetry"]
        assert set(r.deferral_reasons) == set(r.modes) - set(r.selected)
        assert r.controller_ms > 0


def test_ledger_accumulates_and_cap_stops_run(tmp_path):
    eps = {"utility": 0.5, "readiness": 0.25, "shift": 0.25}  # 1.0 per report
    logger = _run(tmp_path, PolicySpec("pf", _privatefair(eps=eps), True), privacy=PrivacyConfig(epsilon=eps))
    spent = [r.epsilon_spent for r in read_round_logs(logger.path)]
    assert spent[-1] == pytest.approx({0: 3.0, 1: 3.0, 2: 3.0})  # 3 rounds x 1.0, every site online
    with pytest.raises(PrivacyBudgetExceeded):
        _run(tmp_path / "capped", PolicySpec("pf", _privatefair(eps=eps), True), PrivacyConfig(epsilon=eps, cap=1.5))


def test_policies_without_telemetry_spend_nothing(tmp_path):
    cfg = {"name": "coverage_random", "capacity": 1, "max_age": 3}
    spec = build_policy(cfg, PrivacyConfig(), np.random.default_rng(0))
    logs = read_round_logs(_run(tmp_path, spec).path)
    for r in logs:
        assert r.telemetry == [] and r.posteriors == [] and set(r.epsilon_spent.values()) == {0.0}
        assert len(r.selected) == 1 and "true_bins" not in r.sim_only  # no truth measured: nobody needs it


class Fixed:
    """Stub coordinator: site 0 FULL, site 1 COMPRESSED, site 2 DEFERRED."""

    name = "fixed"

    def select(self, epoch, reports, available):
        a = {0: ScheduleMode.FULL, 1: ScheduleMode.COMPRESSED, 2: ScheduleMode.DEFERRED}
        assignments = {s: m for s, m in a.items() if s in available}
        return CohortDecision(epoch, "fixed", assignments, deferral_reasons={2: "stub"})

    def observe(self, decision, completed):
        self.last_completed = completed


def test_compressed_sites_train_fewer_steps_and_deferred_sites_do_not_train(tmp_path):
    logs = read_round_logs(_run(tmp_path, PolicySpec("fixed", Fixed(), uses_telemetry=False)).path)
    for r in logs:
        assert r.sim_only["local_steps"] == {"0": 4, "1": 2}  # JSON keys; compressed = half of local_steps
        assert r.selected == [0, 1] and r.modes == {0: "full", 1: "compressed", 2: "deferred"}
        assert r.participation_age[2] > 0 and r.participation_age[0] == 0


class TruthSeeing(Fixed):
    def __init__(self):
        self.seen = []

    def observe_truth(self, true_bins, raw):
        self.seen.append((true_bins, raw))


def test_observe_truth_hook_for_baselines(tmp_path):
    coord = TruthSeeing()
    _run(tmp_path, PolicySpec("oracle", coord, uses_telemetry=False))
    assert len(coord.seen) == 3
    bins, raw = coord.seen[0]
    assert all(type(b) is TrueBins for b in bins.values()) and "val_loss" in raw


def test_privacy_mode_none_sends_true_bins(tmp_path):
    spy = Spy(Fixed())
    logger = _run(tmp_path, PolicySpec("nonprivate", spy, uses_telemetry=True), privacy=PrivacyConfig(mode="none"))
    logs = read_round_logs(logger.path)
    for (_, reports, _), r in zip(spy.select_args, logs, strict=True):
        truth = {b["site_id"]: (b["utility"], b["readiness"], b["shift"]) for b in r.sim_only["true_bins"]}
        assert {s: (x.utility, x.readiness, x.shift) for s, x in reports.items()} == truth
        assert set(r.epsilon_spent.values()) == {0.0} and r.posteriors == []


def test_gate_g1_with_privatefair_and_systems(tmp_path):
    systems = SystemsConfig(p_available=0.7, stickiness=0.5)

    def run(sub):
        spec = PolicySpec("privatefair", _privatefair(), uses_telemetry=True)
        return read_comparable(_run(tmp_path / sub, spec, systems=systems, rounds=4).path)

    assert run("first") == run("second")


def test_fedavg_all_spec_matches_no_policy(tmp_path):
    a = _run(tmp_path / "none", None).path.read_bytes()
    b = _run(tmp_path / "spec", PolicySpec("fedavg_all", None, uses_telemetry=False)).path.read_bytes()
    assert a == b

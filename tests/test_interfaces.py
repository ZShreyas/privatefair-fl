"""Tests for the shared contract. If these fail, something broke the agreement between tracks."""

import pytest

from privatefair.interfaces import (
    K_READINESS,
    K_SHIFT,
    K_UTILITY,
    CohortDecision,
    LedgerEntry,
    Posterior,
    RoundLog,
    ScheduleMode,
    TelemetryReport,
    TrueBins,
)


def test_telemetry_roundtrip():
    r = TelemetryReport(site_id=2, epoch=7, utility=4, readiness=0, shift=2)
    assert TelemetryReport.from_payload(r.to_payload()) == r


def test_telemetry_rejects_extra_fields():
    payload = TelemetryReport(1, 1, 0, 0, 0).to_payload()
    payload["val_loss"] = 0.42  # raw metric must never leave a site
    with pytest.raises(ValueError, match="extra"):
        TelemetryReport.from_payload(payload)


@pytest.mark.parametrize(
    "field,bad", [("utility", K_UTILITY), ("readiness", K_READINESS), ("shift", K_SHIFT), ("shift", -1)]
)
def test_bins_out_of_range(field, bad):
    kwargs = dict(site_id=0, epoch=0, utility=0, readiness=0, shift=0)
    kwargs[field] = bad
    with pytest.raises(ValueError):
        TelemetryReport(**kwargs)
    with pytest.raises(ValueError):
        TrueBins(**kwargs)


def test_bins_reject_non_int():
    with pytest.raises(TypeError):
        TelemetryReport(0, 0, 1.0, 0, 0)
    with pytest.raises(TypeError):
        TelemetryReport(0, 0, True, 0, 0)


def test_ledger_validation():
    LedgerEntry(0, 0, "shift", 1.0)
    with pytest.raises(ValueError):
        LedgerEntry(0, 0, "shift", 0.0)
    with pytest.raises(ValueError):
        LedgerEntry(0, 0, "accuracy", 1.0)


def test_posterior_must_be_distributions():
    Posterior(0, 0, (0.2,) * 5, (0.25,) * 4, (0.5, 0.25, 0.25))
    with pytest.raises(ValueError):
        Posterior(0, 0, (0.2,) * 4, (0.25,) * 4, (0.5, 0.25, 0.25))  # wrong length
    with pytest.raises(ValueError):
        Posterior(0, 0, (0.3,) * 5, (0.25,) * 4, (0.5, 0.25, 0.25))  # sums to 1.5


def test_cohort_decision_rules():
    d = CohortDecision(
        epoch=3,
        policy="test",
        assignments={0: ScheduleMode.FULL, 1: ScheduleMode.COMPRESSED, 2: ScheduleMode.DEFERRED},
        mandatory_sites=frozenset({0}),
        deferral_reasons={2: "unavailable"},
    )
    assert d.selected == {0, 1}
    with pytest.raises(ValueError, match="deferral reason"):
        CohortDecision(epoch=0, policy="t", assignments={2: ScheduleMode.DEFERRED})
    with pytest.raises(ValueError, match="subset"):
        CohortDecision(epoch=0, policy="t", assignments={0: ScheduleMode.FULL}, mandatory_sites=frozenset({5}))


def test_roundlog_json_roundtrip():
    log = RoundLog(
        run_id="r1",
        seed=0,
        policy="random",
        epoch=1,
        selected=[0, 2],
        modes={0: "full", 2: "compressed"},
        mandatory_sites=[2],
        deferral_reasons={1: "slow"},
        telemetry=[TelemetryReport(0, 1, 2, 3, 0).to_payload()],
        posteriors=[],
        participation_age={0: 0, 1: 1, 2: 0},
        epsilon_spent={0: 1.5, 1: 1.5, 2: 1.5},
        round_seconds=12.5,
        bytes_up=1000,
        bytes_down=2000,
        controller_ms=0.4,
        per_site_metrics={0: {"balanced_acc": 0.7}},
    )
    assert RoundLog.from_json(log.to_json()) == log

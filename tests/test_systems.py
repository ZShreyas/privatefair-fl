"""Tests for the systems simulation (task A4). Numpy only, no torch."""

import numpy as np
import pytest

from privatefair.sim.systems import (
    JITTER_STREAM,
    TRACE_STREAM,
    AvailabilityTrace,
    SiteProfile,
    SystemsConfig,
    deadline_seconds,
    expected_seconds,
    generate_trace,
    make_profiles,
    simulate_round,
    systems_rng,
)

IDS = list(range(8))
CFG = SystemsConfig()


def _mean_outage_length(avail):
    lengths = []
    for col in avail.T:
        run = 0
        for a in col:
            if not a:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
    return np.mean(lengths)


def test_profiles_deterministic_and_heterogeneous():
    a = make_profiles(IDS, CFG, systems_rng(0, 0))
    assert a == make_profiles(IDS, CFG, systems_rng(0, 0))
    steps = [p.step_seconds for p in a.values()]
    assert max(steps) > 1.5 * min(steps)


def test_trace_replays_identically_and_round_trips(tmp_path):
    t1 = generate_trace(IDS, 50, CFG, systems_rng(3, TRACE_STREAM))
    t2 = generate_trace(IDS, 50, CFG, systems_rng(3, TRACE_STREAM))
    assert np.array_equal(t1.available, t2.available)
    t1.save(tmp_path / "trace.json")
    loaded = AvailabilityTrace.load(tmp_path / "trace.json")
    assert loaded.site_ids == t1.site_ids and np.array_equal(loaded.available, t1.available)
    assert loaded.at(7) == t1.at(7)


def test_trace_long_run_rate_matches_p_available():
    t = generate_trace(list(range(50)), 400, SystemsConfig(p_available=0.8, stickiness=0.7), systems_rng(0, 1))
    assert t.available.mean() == pytest.approx(0.8, abs=0.03)


def test_stickiness_makes_outages_longer():
    ids = list(range(50))
    iid = generate_trace(ids, 400, SystemsConfig(p_available=0.8, stickiness=0.0), systems_rng(0, 1))
    sticky = generate_trace(ids, 400, SystemsConfig(p_available=0.8, stickiness=0.8), systems_rng(0, 1))
    assert _mean_outage_length(sticky.available) > 2 * _mean_outage_length(iid.available)


def test_trace_index_out_of_range():
    with pytest.raises(IndexError):
        generate_trace(IDS, 3, CFG, systems_rng(0, 1)).at(3)


def test_systems_streams_differ_from_training_streams():
    # Training uses SeedSequence([seed, site_id]); systems streams must not collide with site 0, 1, 2.
    for stream in range(3):
        sys_draw = systems_rng(5, stream).random(4)
        for site in range(3):
            train_draw = np.random.default_rng(np.random.SeedSequence([5, site])).random(4)
            assert not np.array_equal(sys_draw, train_draw)


def test_expected_seconds_grows_with_steps_and_bytes():
    p = SiteProfile(0, step_seconds=2.0, mbps=8.0)  # 1 MB/s
    assert expected_seconds(p, 10, 0, 0) == pytest.approx(20.0)
    assert expected_seconds(p, 10, 1_000_000, 1_000_000) == pytest.approx(22.0)
    assert expected_seconds(p, 5, 0, 0) < expected_seconds(p, 10, 0, 0)


def test_slow_site_misses_deadline_fast_site_does_not():
    profiles = {0: SiteProfile(0, 1.0, 1000.0), 1: SiteProfile(1, 10.0, 1000.0)}
    t = simulate_round(profiles, [0, 1], 10, 0, 0, deadline=20.0, jitter_sigma=0.0, rng=systems_rng(0, JITTER_STREAM))
    assert t.completed == {0}
    assert t.site_seconds[1] == pytest.approx(100.0)
    assert t.round_seconds == pytest.approx(20.0)  # capped at the deadline


def test_per_site_steps_and_empty_round():
    profiles = {0: SiteProfile(0, 1.0, 1000.0), 1: SiteProfile(1, 1.0, 1000.0)}
    t = simulate_round(profiles, [0, 1], {0: 10, 1: 5}, 0, 0, 100.0, 0.0, systems_rng(0, 2))
    assert t.site_seconds[0] == pytest.approx(2 * t.site_seconds[1])
    empty = simulate_round(profiles, [], 10, 0, 0, 100.0, 0.1, systems_rng(0, 2))
    assert empty.completed == frozenset() and empty.round_seconds == 0.0


def test_deadline_factor_scales_median():
    profiles = make_profiles(IDS, CFG, systems_rng(0, 0))
    d1 = deadline_seconds(profiles, 20, 10**6, 10**6, factor=1.0)
    d2 = deadline_seconds(profiles, 20, 10**6, 10**6, factor=2.0)
    assert d2 == pytest.approx(2 * d1)
    missing = [sid for sid, p in profiles.items() if expected_seconds(p, 20, 10**6, 10**6) > d1]
    assert 0 < len(missing) < len(profiles)  # median deadline: some but not all sites too slow


BAD = [{"p_available": 0.0}, {"stickiness": 1.0}, {"deadline_factor": 0}, {"speed_sigma": -1}]


@pytest.mark.parametrize("kwargs", BAD)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        SystemsConfig(**kwargs)

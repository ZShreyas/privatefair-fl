"""Tests for server-side participation-age tracking (task B4)."""

from privatefair.coordinator.participation import ParticipationTracker


def test_never_participated_site_ages_from_epoch_zero():
    t = ParticipationTracker()
    assert t.age(1, epoch=0) == 1
    assert t.age(1, epoch=5) == 6
    assert t.last_participated_epoch(1) is None


def test_age_resets_after_participation():
    t = ParticipationTracker()
    t.update(epoch=3, completed=frozenset({1}))
    assert t.last_participated_epoch(1) == 3
    assert t.age(1, epoch=3) == 0
    assert t.age(1, epoch=4) == 1
    assert t.age(1, epoch=7) == 4


def test_sites_are_tracked_independently():
    t = ParticipationTracker()
    t.update(epoch=1, completed=frozenset({1}))
    t.update(epoch=2, completed=frozenset({2}))
    assert t.age(1, epoch=5) == 4
    assert t.age(2, epoch=5) == 3
    assert t.age(3, epoch=5) == 6  # never participated


def test_update_only_touches_completed_sites():
    t = ParticipationTracker()
    t.update(epoch=0, completed=frozenset({1, 2}))
    t.update(epoch=1, completed=frozenset({1}))  # site 2 not completed this time
    assert t.age(1, epoch=2) == 1
    assert t.age(2, epoch=2) == 2  # still dated to epoch 0


def test_ages_batches_age_for_roundlog():
    t = ParticipationTracker()
    t.update(epoch=0, completed=frozenset({1}))
    assert t.ages([1, 2, 3], epoch=2) == {1: 2, 2: 3, 3: 3}

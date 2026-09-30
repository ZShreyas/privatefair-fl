"""Tests for server-side participation-age tracking (task B4).

required_now() is the shared earliest-deadline-first reservation rule used by
both CoverageRandomPolicy and PrivateFairCoordinator; its own multi-seed bound
tests live alongside those coordinators in test_policies.py / test_privatefair.py.
The tests here check the formula itself in isolation, by hand.
"""

from privatefair.coordinator.participation import ParticipationTracker, required_now


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


# ---------------------------------------------------------------------------
# required_now
# ---------------------------------------------------------------------------


def test_required_now_no_sites():
    assert required_now([], capacity=2, max_age=3) == 0


def test_required_now_nothing_due_soon():
    # capacity=10 comfortably covers max_age=5 for a handful of low-age sites.
    assert required_now([0, 1, 2], capacity=10, max_age=5) == 0


def test_required_now_counts_currently_overdue():
    # k=0 term: 3 sites already at age >= max_age, capacity irrelevant at k=0.
    assert required_now([3, 3, 3], capacity=1, max_age=3) == 3


def test_required_now_looks_ahead_past_the_current_deadline():
    # None of these 3 sites is overdue yet (age=2 < max_age=3), but all 3 will
    # hit age 3 next round if none is served now, and capacity=1 can only clear
    # one of them per round -- so 2 must be admitted now, not 0.
    assert required_now([2, 2, 2], capacity=1, max_age=3) == 2


def test_required_now_never_negative():
    assert required_now([0], capacity=10, max_age=5) == 0

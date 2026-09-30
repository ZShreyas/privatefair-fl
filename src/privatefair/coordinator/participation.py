"""Server-side participation-age tracking (report section 4, row "Participation age").

The coordinator derives this itself from its own schedule log; it is never
requested from or reported by a client -- there is no wire message for it, only
the `Coordinator.observe()` call after each round (see interfaces.py).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class ParticipationTracker:
    """Tracks, per site, the last epoch at which it completed participation."""

    _last_participated: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def age(self, site_id: int, epoch: int) -> int:
        """Epochs since `site_id` last completed participation, as of `epoch`.

        A site that has never participated is treated as having last participated
        at epoch -1 (before the run started), so its age already grows from epoch 0
        onward -- an unseen site is maximally overdue, not exempt from coverage.
        """
        last = self._last_participated.get(site_id, -1)
        return epoch - last

    def ages(self, site_ids: Iterable[int], epoch: int) -> dict[int, int]:
        """`age()` for several sites at once -- e.g. for RoundLog.participation_age."""
        return {site_id: self.age(site_id, epoch) for site_id in site_ids}

    def update(self, epoch: int, completed: frozenset[int]) -> None:
        """Record that these sites completed participation in `epoch`."""
        for site_id in completed:
            self._last_participated[site_id] = epoch

    def last_participated_epoch(self, site_id: int) -> int | None:
        """The last epoch `site_id` completed participation, or None if never."""
        return self._last_participated.get(site_id)


def required_now(ages: Iterable[int], capacity: int, max_age: int) -> int:
    """Minimum number of sites that must be admitted *this* round to keep every
    site's age from exceeding `max_age`, given only `capacity` slots per round
    from now on.

    This is the earliest-deadline-first demand-bound feasibility check: a site
    with current age `a` must be served again within `max_age - a` further
    rounds (0 if it's already due). For each lookahead window of `k+1` rounds
    (this round plus `k` more, for `k` in `0..max_age`), every site due within
    that window must fit inside the `capacity * k` slots the window has left
    *after* this round -- so `count(age >= max_age - k) - capacity * k` sites
    from that window's demand cannot be deferred even one more round. The
    binding (largest) such deficit across every window is what must be
    admitted now; forcing only the sites already at `age >= max_age` (`k=0`)
    is not enough on its own, because several sites can then cross the
    threshold together in a later round with no room left for all of them.

    Reserving `min(required_now(...), capacity)` oldest sites like this, before
    any other selection logic runs, is what actually keeps a coordinator's
    coverage bound tight -- filling spare capacity by any other rule (random,
    score, ...) for the *rest* of the slots does not affect the guarantee.
    """
    ages = list(ages)
    if not ages:
        return 0
    needs = (sum(1 for a in ages if a >= max_age - k) - capacity * k for k in range(max_age + 1))
    return max(0, max(needs))

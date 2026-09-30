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

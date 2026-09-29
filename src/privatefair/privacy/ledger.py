"""Privacy ledger: per-site/signal/epoch epsilon accounting and a hard budget cap.

Report section 8.4: "Privacy accounting is auditable. Store per-client,
per-signal, per-epoch epsilon values and the reported release count; fail a run
if a configured budget is exceeded." Objective 2 additionally asks to "track the
composed per-site local-DP budget across decision epochs."

Composition here is *basic* (sequential) composition: a site's total privacy
loss is the sum of the epsilons of every report it has released, across every
signal and epoch. That is the standard, safe (if not tight) local-DP
composition bound; a tighter advanced-composition accountant is future work,
not required for this task.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from privatefair.interfaces import LedgerEntry, Signal


class PrivacyBudgetExceeded(RuntimeError):
    """Raised when recording an entry would push a site's composed epsilon over its cap."""


@dataclass
class PrivacyLedger:
    """Accumulates `LedgerEntry` releases and enforces an optional per-site epsilon cap.

    `cap`, if given, is the maximum composed epsilon (summed over every signal and
    epoch released so far) any single site may accumulate. `record` checks the
    whole batch it is given before committing any of it: a report is released as
    one atomic group of per-signal entries (that's how a Privatizer emits them), so
    either the whole report's entries are accepted or none are -- the ledger never
    ends up holding a partial report.
    """

    cap: float | None = None
    _entries: list[LedgerEntry] = field(default_factory=list, init=False, repr=False)
    _spent: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def record(self, entries: LedgerEntry | Iterable[LedgerEntry]) -> None:
        """Record one or more LedgerEntry releases, enforcing the cap per site."""
        batch = [entries] if isinstance(entries, LedgerEntry) else list(entries)
        projected = dict(self._spent)
        for entry in batch:
            new_total = projected.get(entry.site_id, 0.0) + entry.epsilon
            if self.cap is not None and new_total > self.cap + 1e-9:
                raise PrivacyBudgetExceeded(
                    f"site {entry.site_id}: recording epsilon={entry.epsilon} for signal "
                    f"{entry.signal!r} at epoch {entry.epoch} would bring composed spend to "
                    f"{new_total:.6g}, exceeding cap {self.cap:.6g}"
                )
            projected[entry.site_id] = new_total
        self._entries.extend(batch)
        self._spent = projected

    def spent(self, site_id: int, signal: Signal | None = None) -> float:
        """Composed epsilon spent by a site so far (optionally restricted to one signal)."""
        if signal is None:
            return self._spent.get(site_id, 0.0)
        return sum(e.epsilon for e in self._entries if e.site_id == site_id and e.signal == signal)

    def release_count(self, site_id: int, signal: Signal | None = None) -> int:
        """Number of reports a site has released (optionally restricted to one signal)."""
        return sum(1 for e in self._entries if e.site_id == site_id and (signal is None or e.signal == signal))

    def entries_for(self, site_id: int) -> list[LedgerEntry]:
        return [e for e in self._entries if e.site_id == site_id]

    @property
    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def remaining(self, site_id: int) -> float | None:
        """Epsilon left before this site hits the cap, or None if there is no cap."""
        if self.cap is None:
            return None
        return max(0.0, self.cap - self.spent(site_id))

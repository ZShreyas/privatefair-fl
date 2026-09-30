"""No-privacy Privatizer (task B6, report section 10.E #4: "quantized but non-private telemetry").

Implements the `Privatizer` protocol from interfaces.py exactly like
`privacy.rr.RandomizedResponsePrivatizer` (B1), but releases a site's true bins
verbatim -- no K-ary randomized response, no noise. This isolates the effect of
*coarsening* (the alphabet is still only K categories) from the effect of
randomized-response *noise*, per report section 7.3's ablation plan.

No privacy is spent releasing these reports, so `privatize()` returns an empty
ledger -- nothing for `privacy.ledger.PrivacyLedger` to record.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from privatefair.interfaces import LedgerEntry, TelemetryReport, TrueBins


@dataclass(frozen=True)
class NoPrivacyPrivatizer:
    """Site-side `Privatizer`: releases `true_bins` as the report, unchanged."""

    def privatize(self, true_bins: TrueBins, rng: np.random.Generator) -> tuple[TelemetryReport, list[LedgerEntry]]:
        report = TelemetryReport(
            site_id=true_bins.site_id,
            epoch=true_bins.epoch,
            utility=true_bins.utility,
            readiness=true_bins.readiness,
            shift=true_bins.shift,
        )
        return report, []

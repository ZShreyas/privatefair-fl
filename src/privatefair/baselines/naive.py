"""Naive-decoding coordinator (task B6, report section 7.3).

Report section 7.3 asks to "compare posterior inference against naively treating a
noisy category as true." `NaiveDecodingCoordinator` is that comparator: it runs
the exact same coverage / shifted-slot / score / mode-assignment pipeline as
`PrivateFairCoordinator` (B5), but skips the Bayes correction in
`coordinator.posterior.decode_report` and instead trusts a report's bin outright
(`one_hot_posterior` -- full mass on the reported bin, zero uncertainty).

This same class also serves report 10.E's "quantized but non-private telemetry"
baseline (#4): pair it with `privacy.no_privacy.NoPrivacyPrivatizer` on the site
side (true bins sent with no RR noise at all) and naive decoding is then not an
approximation but the *correct* interpretation, since there is no noise left to
correct for. The two scenarios differ only in which Privatizer the sites use --
not in how the coordinator reads the report -- which is why one class covers both
of TEAM_PLAN's "naive decoding" and "quantized non-private" baselines.

Lives outside `coordinator/` alongside B6's other baselines (see baselines/__init__.py),
though nothing here actually needs ground truth -- it only reads TelemetryReport,
same as PrivateFairCoordinator.
"""

from __future__ import annotations

from dataclasses import dataclass

from privatefair.coordinator.posterior import one_hot_posterior
from privatefair.coordinator.privatefair import PrivateFairCoordinator
from privatefair.interfaces import Posterior, TelemetryReport


@dataclass
class NaiveDecodingCoordinator(PrivateFairCoordinator):
    """PrivateFairCoordinator, but a report's bin is trusted exactly instead of Bayes-decoded.

    `epsilon`/`priors` are still required fields (inherited from PrivateFairCoordinator)
    but are unused here -- there is no decoding step left for them to parameterize.
    Kept for API symmetry with PrivateFairCoordinator (same constructor shape, so
    callers/config can swap between the two without special-casing arguments).
    """

    name: str = "naive_decoding"

    def _posterior(self, site_id: int, report: TelemetryReport | None, epoch: int) -> Posterior | None:
        if report is None:
            return None
        return one_hot_posterior(report.site_id, report.epoch, report.utility, report.readiness, report.shift)

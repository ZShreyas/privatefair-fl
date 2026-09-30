"""Raw-telemetry oracle coordinator (task B6, report section 10.E #3: "raw-telemetry
adaptive oracle (upper bound; not privacy-preserving)").

Decodes from `TrueBins` -- the exact bins a site's Privatizer would otherwise
privatize -- instead of a coordinator-received `TelemetryReport`. It's fed via
`observe_truth`, an optional hook the training loop calls before `select()` on
any policy that defines one (agreed with A6 on issue #13). Ordinary `Coordinator`
policies never define `observe_truth`, so they never see `TrueBins`; the
`interfaces.py` `Coordinator` protocol is a structural `Protocol`, so adding this
method doesn't touch that frozen contract at all, and `tests/test_boundaries.py`'s
guard (scoped to `coordinator/`) is unaffected since this class lives outside
that package specifically because it needs ground truth.

Scope note: this uses the *quantized* true bins (one-hot posteriors -- zero
uncertainty, but still only K categories), not the raw pre-quantization
continuous values (val_loss, shift_z, ...) that `raw` also carries. `raw` is
accepted and stored for a future continuous-valued score, but the current score
formula matches `PrivateFairCoordinator`'s categorical one exactly, just fed
certain (ground-truth) inputs instead of decoded ones. That already isolates
what this project's ablation plan needs from an oracle -- the effect of privacy
noise, holding categorical coarsening fixed -- and stays robust to A6's raw-value
dict shape still stabilizing. A richer continuous-valued oracle is a natural
follow-up once that shape settles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from privatefair.coordinator.posterior import one_hot_posterior
from privatefair.coordinator.privatefair import PrivateFairCoordinator
from privatefair.interfaces import Posterior, TelemetryReport, TrueBins


@dataclass
class RawOracleCoordinator(PrivateFairCoordinator):
    """PrivateFairCoordinator, but decoding uses TrueBins (ground truth) instead of a
    privatized TelemetryReport.

    Call `observe_truth(true_bins, raw)` once per epoch, before `select()`, with
    that epoch's true bins for every site (e.g. from `sim.true_bins.TrueBinsSimulator.compute`).
    `select()` itself keeps the normal `Coordinator` signature and can be called
    with `reports={}` -- this oracle never reads it.

    `epsilon`/`priors` are still required fields (inherited from PrivateFairCoordinator)
    but unused -- there is no privatized report to decode.
    """

    name: str = "raw_oracle"
    _true_bins: dict[int, TrueBins] = field(default_factory=dict, init=False, repr=False)
    _raw: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def observe_truth(self, true_bins: dict[int, TrueBins], raw: dict[str, Any]) -> None:
        """Called by the training loop before select(), with this epoch's ground truth."""
        self._true_bins = dict(true_bins)
        self._raw = dict(raw)

    def _posterior(self, site_id: int, report: TelemetryReport | None, epoch: int) -> Posterior | None:
        true = self._true_bins.get(site_id)
        if true is None:
            return None
        return one_hot_posterior(true.site_id, true.epoch, true.utility, true.readiness, true.shift)

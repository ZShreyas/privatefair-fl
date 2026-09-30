"""Raw-telemetry oracle coordinator (task B6, report section 10.E #3: "raw-telemetry
adaptive oracle (upper bound; not privacy-preserving)").

Decodes from `TrueBins`/`raw` -- the exact, pre-quantization measurements a site's
Privatizer would otherwise coarsen and privatize -- instead of a coordinator-received
`TelemetryReport`. It's fed via `observe_truth`, an optional hook the training loop
calls before `select()` on any policy that defines one (agreed with A6 on issue #13).
Ordinary `Coordinator` policies never define `observe_truth`, so they never see
`TrueBins`; the `interfaces.py` `Coordinator` protocol is a structural `Protocol`,
so adding this method doesn't touch that frozen contract at all, and
`tests/test_boundaries.py`'s guard (scoped to `coordinator/`) is unaffected since
this class lives outside that package specifically because it needs ground truth.

Scores from the *continuous* pre-quantization values in `raw` (A6/#40's shape:
val_loss, utility_delta, shift_z, expected_seconds, deadline_seconds), not the
quantized `TrueBins` alone. That distinction is the whole point of having this
class separate from B6's "quantized non-private" baseline
(`baselines.naive.NaiveDecodingCoordinator` + `privacy.no_privacy.NoPrivacyPrivatizer`):
report 10.E's oracle (#3) vs. quantized-non-private (#4) gap is supposed to measure
what coarsening into K categories costs. Scoring the oracle from the one-hot of the
same quantized bin the non-private baseline already sees makes that gap zero by
construction -- caught in review (both made identical decisions in 1500/1500
simulated rounds before that fix). Using the raw values here gives the oracle real
information the coarsened baselines don't have, so the comparison is meaningful.

The FULL/COMPRESSED and shifted-slot-eligibility *decisions* (as opposed to the
continuous score) use the exact ground-truth boundary (`expected_seconds <=
deadline_seconds`; `shift_z >= shift_scale`) via `_full_capable`/
`_is_likely_shifted` overrides, not the argmax of the engineered posterior --
also caught in review: that argmax tracks a target *expectation*, a different
property, and does not reliably land on the correct side of either threshold
(e.g. expected/deadline ratios of 0.5, 0.67 and 1.0 -- all comfortably within
the deadline -- came out COMPRESSED under the argmax approach).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from privatefair.coordinator.privatefair import PrivateFairCoordinator
from privatefair.interfaces import ALPHABET_SIZE, Posterior, TelemetryReport, TrueBins


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _spread_expectation(target: float, k: int) -> tuple[float, ...]:
    """A K-point distribution whose normalized expected bin (E[bin]/(K-1)) equals
    `target` (clipped to [0, 1]) exactly, with mass split between the two integer
    bins bracketing `target * (K-1)`.

    This is how a continuous raw score gets fed into `risk_hat`/`readiness_hat`,
    which are defined as exactly that normalized expectation -- so this
    construction reuses those functions (and everything built on them:
    `score()`, coverage, shifted-slot fill) unchanged, just with continuous
    rather than quantized input.
    """
    target = _clip01(target)
    pos = target * (k - 1)
    lo = int(pos)
    hi = min(lo + 1, k - 1)
    frac = pos - lo
    dist = [0.0] * k
    dist[lo] += 1 - frac
    dist[hi] += frac
    return tuple(dist)


def _shift_distribution(top_prob: float, k: int) -> tuple[float, ...]:
    """A K-point distribution with exactly `top_prob` (clipped to [0, 1]) mass on
    the last bin, the rest on the first.

    Unlike utility/readiness, `p_shift_top` (`= p_shift[-1]`) is used directly as
    a probability-like score in `PrivateFairCoordinator.score()`, and
    `is_likely_shifted` checks whether the last bin is the argmax -- so this
    puts `top_prob` there directly rather than going through the expected-bin
    construction above, which would not preserve that value.
    """
    top_prob = _clip01(top_prob)
    dist = [0.0] * k
    dist[0] = 1 - top_prob
    dist[-1] = top_prob
    return tuple(dist)


@dataclass
class RawOracleCoordinator(PrivateFairCoordinator):
    """PrivateFairCoordinator, but decoding uses raw ground-truth measurements
    (`TrueBins` + `raw`) instead of a privatized/decoded `TelemetryReport`.

    Call `observe_truth(true_bins, raw)` once per epoch, before `select()`, with
    that epoch's ground truth for every site (e.g. from
    `sim.true_bins.TrueBinsSimulator.compute`, combined with A6's `raw` additions
    for `expected_seconds`/`deadline_seconds`). `select()` itself keeps the
    normal `Coordinator` signature and can be called with `reports={}` -- this
    oracle never reads it.

    If `select()` is called for an epoch that doesn't match the epoch
    `observe_truth` was last called with (e.g. the loop forgot to call it that
    round), every site is treated as having no ground truth available -- the
    same "missing report" fallback as `PrivateFairCoordinator` -- rather than
    silently reusing a previous round's stale truth.

    `epsilon`/`priors` are still required fields (inherited from PrivateFairCoordinator)
    but unused -- there is no privatized report to decode.

    `utility_tau`/`shift_scale` set the continuous-to-[0,1] mapping for the risk
    and shift *scores*, and `shift_scale` doubles as the exact shifted-slot
    threshold in `_is_likely_shifted` (`shift_z >= shift_scale`). Both default to
    `sim.true_bins.TelemetryConfig`'s defaults (`utility_tau=0.02`, the outer
    utility-bin edge is `2*utility_tau`; the default `shift_thresholds` upper
    edge is 4.0). Pass the same values your `TelemetryConfig` uses if you've
    changed them. The FULL/COMPRESSED threshold (`_full_capable`) needs no such
    constant -- it compares `expected_seconds` to `deadline_seconds` directly.
    """

    name: str = "raw_oracle"
    utility_tau: float = 0.02
    shift_scale: float = 4.0
    _true_bins: dict[int, TrueBins] = field(default_factory=dict, init=False, repr=False)
    _raw: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _truth_epoch: int | None = field(default=None, init=False, repr=False)

    def observe_truth(self, true_bins: dict[int, TrueBins], raw: dict[str, Any]) -> None:
        """Called by the training loop before select(), with this epoch's ground truth."""
        self._true_bins = dict(true_bins)
        self._raw = dict(raw)
        self._truth_epoch = next((b.epoch for b in self._true_bins.values()), None)

    def _risk_raw(self, site_id: int) -> float:
        delta = self._raw.get("utility_delta", {}).get(site_id)
        if delta is None:  # first epoch: no previous loss to compare against -- same "no signal" as A5's utility_bin
            return 0.5
        return _clip01((delta + 2 * self.utility_tau) / (4 * self.utility_tau))

    def _shift_raw(self, site_id: int) -> float:
        z = self._raw.get("shift_z", {}).get(site_id)
        return 0.0 if z is None else _clip01(z / self.shift_scale)

    def _readiness_raw(self, site_id: int) -> float:
        expected = self._raw.get("expected_seconds", {}).get(site_id)
        deadline = self._raw.get("deadline_seconds")
        if expected is None or not deadline:  # no systems config: same "always fast" default as A5/A6
            return 1.0
        return _clip01(1 - expected / deadline)

    def _posterior(self, site_id: int, report: TelemetryReport | None, epoch: int) -> Posterior | None:
        true = self._true_bins.get(site_id)
        if true is None or self._truth_epoch != epoch:
            return None
        return Posterior(
            site_id=true.site_id,
            epoch=true.epoch,
            p_utility=_spread_expectation(self._risk_raw(site_id), ALPHABET_SIZE["utility"]),
            p_readiness=_spread_expectation(self._readiness_raw(site_id), ALPHABET_SIZE["readiness"]),
            p_shift=_shift_distribution(self._shift_raw(site_id), ALPHABET_SIZE["shift"]),
        )

    def _full_capable(self, site_id: int, posterior: Posterior) -> bool:
        """Exact rule, not an argmax proxy: can this site finish a full round by
        the deadline? `_posterior`'s `p_readiness` is engineered to hit a target
        *expectation* (for `score()`'s continuous readiness_hat term) -- a
        different property from "which bin is most probable" -- so its argmax
        (what the base class's `is_full_capable` checks) does not reliably land
        on the correct side of the FULL/COMPRESSED threshold. This checks the
        real numbers directly instead. True with no systems config (no
        expected/deadline in `raw`), matching the "always fast" default used
        elsewhere.
        """
        expected = self._raw.get("expected_seconds", {}).get(site_id)
        deadline = self._raw.get("deadline_seconds")
        if expected is None or not deadline:
            return True
        return expected <= deadline

    def _is_likely_shifted(self, site_id: int, posterior: Posterior) -> bool:
        """Exact rule: true shift score at or past the "shifted" threshold
        (matches `sim.bins.shift_bin`'s boundary, `shift_scale`), not an argmax
        proxy over `_posterior`'s engineered distribution. Same rationale as
        `_full_capable`.
        """
        z = self._raw.get("shift_z", {}).get(site_id)
        return z is not None and z >= self.shift_scale

"""Equal-weight FedAvg (task A3).

Every completed update counts the same, regardless of site size or telemetry: the project
rule is that noisy telemetry is never an aggregation weight, and equal weights keep small
hospitals from being drowned out.

No torch import: averaging only needs `+` and `/`, so this works on torch state_dicts and
on numpy arrays (which the CI tests use).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from privatefair.interfaces import ClientUpdate


def _is_float(v: Any) -> bool:
    if hasattr(v, "is_floating_point"):  # torch.Tensor
        return bool(v.is_floating_point())
    return bool(np.issubdtype(np.asarray(v).dtype, np.floating))


def _copy(v: Any) -> Any:
    return v.clone() if hasattr(v, "clone") else np.array(v, copy=True)


class FedAvg:
    """Implements the `Aggregator` protocol."""

    def aggregate(self, global_weights: dict[str, Any], updates: list[ClientUpdate]) -> dict[str, Any]:
        done = [u for u in updates if u.completed]  # deadline misses contribute nothing
        if not done:
            return global_weights
        out = {}
        for key in global_weights:
            vals = [u.weights[key] for u in done]
            # Integer buffers (BatchNorm's num_batches_tracked) are counters, not parameters.
            out[key] = sum(vals[1:], vals[0]) / len(vals) if _is_float(vals[0]) else _copy(vals[0])
        return out

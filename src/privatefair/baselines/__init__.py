"""Track B: comparator/ablation Coordinators for the experiment matrix (task B6, report 10.E).

Grouped here, separately from `coordinator/`, because the raw-telemetry oracle
(`baselines.oracle.RawOracleCoordinator`) sees simulator ground truth (`TrueBins`)
by design -- the whole point of an "upper bound, not privacy-preserving" baseline.
`tests/test_boundaries.py` only scans `coordinator/`, so this package is exempt
from that guard, and the other baselines here (which don't need ground truth) are
grouped alongside it for a single place to find every comparator.
"""

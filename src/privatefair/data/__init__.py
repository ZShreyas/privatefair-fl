"""Track A: dataset loading, site partitioning, synthetic shift."""

from privatefair.data.partition import (
    PartitionConfig,
    choose_shifted_sites,
    dirichlet_partition,
    site_size_weights,
    split_indices,
)
from privatefair.data.pathmnist import SiteData, build_sites, load_pathmnist, load_sites
from privatefair.data.shift import ShiftSpec, apply_shift

__all__ = [
    "PartitionConfig",
    "ShiftSpec",
    "SiteData",
    "apply_shift",
    "build_sites",
    "choose_shifted_sites",
    "dirichlet_partition",
    "load_pathmnist",
    "load_sites",
    "site_size_weights",
    "split_indices",
]

"""Context features of a partition for the bandit."""

from __future__ import annotations

import numpy as np

from src.config import DEFAULT_NPROBE, MAX_PARTITION_SIZE
from src.partition import Partition
from src.types import IndexStats

CONTEXT_DIM: int = 6

# upper bound of each feature
_CLAMP: np.ndarray = np.array([2.0, 2.0, 1.0, 1.0, 4.0, 1.0], dtype=np.float64)


def extract_context(
    partition: Partition,
    stats: IndexStats,
    size_cap: int = MAX_PARTITION_SIZE,
    *,
    total_access: float,
    nprobe: int = DEFAULT_NPROBE,
    drift_scale: float | None = None,
    queries: float | None = None,
) -> np.ndarray:
    """The six context features of a partition."""
    size = partition.size()
    drift = partition.centroid_drift()
    # drift divided by the drift limit
    if drift_scale is not None:
        drift = drift / max(drift_scale, 1e-9)
    total = partition.total_size()
    tombstone_frac = len(partition.tombstones) / max(total, 1)
    norm_size = size / max(size_cap, 1)
    if queries is None:
        queries = total_access / max(1, nprobe)
    norm_access = min(1.0, partition.access_count / queries) if queries > 0 else 0.0
    rel_size = size / max(stats.mean_partition_size, 1.0)

    x = np.array(
        [norm_size, drift, norm_access, tombstone_frac, rel_size, 1.0],
        dtype=np.float64,
    )
    return np.minimum(np.maximum(x, 0.0), _CLAMP)

"""The six numbers that describe one partition to the bandit, see `extract_context`."""

from __future__ import annotations

import numpy as np

from src.config import DEFAULT_NPROBE, MAX_PARTITION_SIZE
from src.partition import Partition
from src.types import IndexStats

CONTEXT_DIM: int = 6

# the upper bound of each feature. the LinUCB analysis assumes the context has a bounded norm, and
# without these bounds the size features could grow without limit for a partition far over the
# threshold or far over the mean. access and the tombstone share are fractions already
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
    """Return the context vector of a partition, six features each kept within a fixed range.

    The features are the size over the split threshold `size_cap`, the centroid drift over
    `drift_scale`, the access fraction A, the share of tombstoned vectors, the size over the mean
    partition size, and a constant 1. A is the share of routed queries that scanned the
    partition, as in Quake section 4.1, the quantity the cost model ranks on. `queries` is the
    routed query count, and when it is not given it is estimated as `total_access` over `nprobe`.
    `total_access` has no default, because A cannot be computed from one partition alone.
    """
    size = partition.size()
    drift = partition.centroid_drift()
    # drift is a distance, about 13 on SIFT and about 1 on GIST, so it is divided by the drift
    # limit to put both datasets on one scale, as the size is divided by its threshold
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

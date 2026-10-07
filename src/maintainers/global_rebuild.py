"""The periodic rebuild baseline of Ada-IVF, section 5.1.

Each time inserts plus deletes since the last rebuild reach 2.5 percent of the live count, the
index is rebuilt from its live vectors with exact k-means at the same number of partitions, unless
`n_partitions` fixes another number, as the control run at the bandit's partition count does.
"""

from __future__ import annotations

import numpy as np

from src.config import MAINTENANCE_CHECK_INTERVAL, REBUILD_FRACTION_THRESHOLD
from src.index import IVFIndex
from src.maintainers.base import Maintainer
from src.types import MaintenanceReport


class GlobalRebuildMaintainer(Maintainer):
    """Rebuild the whole index with k-means after a fixed share of the vectors has changed."""

    name = "global_rebuild"

    def __init__(
        self,
        rebuild_fraction: float = REBUILD_FRACTION_THRESHOLD,
        n_partitions: int | None = None,
        check_interval: int = MAINTENANCE_CHECK_INTERVAL,
        seed: int = 42,
    ) -> None:
        """Create the policy. `n_partitions` fixes the partition count of every rebuild if given."""
        super().__init__(check_interval=check_interval)
        self.rebuild_fraction = rebuild_fraction
        self.n_partitions = n_partitions
        self.seed = seed
        self._last_rebuild_total: int = 0

    def start(self, index: IVFIndex, step: int) -> None:
        """Start counting updates from the start of the stream.

        Ada-IVF counts the 2.5 percent over the workload that follows the build, so the updates
        that grew the starting index do not count.
        """
        super().start(index, step)
        self._last_rebuild_total = index._total_inserts + index._total_deletes

    def should_maintain(self, index: IVFIndex, step: int) -> bool:
        """Return True once the updates since the last rebuild reach the set share of live vectors."""
        total_modifications = index._total_inserts + index._total_deletes
        mods_since_rebuild = total_modifications - self._last_rebuild_total
        live_count = max(1, len(index.vector_to_partition))
        threshold = self.rebuild_fraction * live_count
        return mods_since_rebuild >= threshold

    def maintain(self, index: IVFIndex) -> MaintenanceReport:
        """Rebuild the index from its live vectors with exact k-means.

        The work is one read of every live vector plus every distance the k-means computed, its
        iterations times vectors times centroids, which is what makes a rebuild far more expensive
        than refreshing every centroid.
        """
        all_ids: list[int] = []
        all_vecs: list[np.ndarray] = []
        for p in index.partitions.values():
            live_ids, live_vecs = p.get_live_vectors()
            all_ids.extend(live_ids)
            for v in live_vecs:
                all_vecs.append(v)

        if not all_vecs:
            return MaintenanceReport(triggered=True)

        n_part = self.n_partitions or len(index.partitions)
        n_part = min(n_part, len(all_vecs))

        all_matrix = np.array(all_vecs, dtype=np.float32)

        index.partitions.clear()
        index.vector_to_partition.clear()
        index.next_partition_id = 0
        index._invalidate_cache()

        index.build(all_matrix, all_ids, n_partitions=n_part, seed=self.seed)

        self._last_rebuild_total = index._total_inserts + index._total_deletes

        return MaintenanceReport(
            bulk_kmeans_distance_evaluations=index.last_build_distance_evaluations,
            triggered=True,
            num_splits=0,
            num_merges=0,
            num_repartitioned=len(all_ids),
            num_centroids_recomputed=n_part,
            vectors_processed=len(all_ids),
        )

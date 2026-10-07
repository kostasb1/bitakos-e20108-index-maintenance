"""The three DeDrift strategies, Baranchuk et al., ICCV 2023, section 5.1.

The paper applies them on a schedule, every one to six months of collected content. Here they use
the trigger of the global rebuild, an update once 2.5 percent of the live vectors have changed, so
DeDrift and a full rebuild differ only in what they do and not in how often.

Lazy moves every centroid to the mean of the vectors currently assigned to it, with no
reassignment. Split collects the vectors of the k largest clusters into B1, sets k2 to the ceiling
of the size of B1 over mu, the median cluster size of the whole index, collects the k2 minus k
smallest clusters into B2, runs k-means with k2 centroids on B1 and B2 together and replaces those
k2 clusters, so the total cluster count stays the same. Hybrid runs Lazy and then Split.
"""

from __future__ import annotations

import numpy as np

from src.config import MAINTENANCE_CHECK_INTERVAL, REBUILD_FRACTION_THRESHOLD
from src.index import IVFIndex
from src.maintainers.base import Maintainer, kmeans_distance_evaluations, merge_partitions
from src.partition import Partition
from src.types import MaintenanceReport

VARIANTS = ("lazy", "split", "hybrid")

# the number of largest clusters collected into B1, as a fraction of the partition count. DeDrift
# section 5.1 uses k = 8 at K = 4096 and k = 64 at K = 16384, so k over K is 0.20 and 0.39
# percent, a fraction of the index rather than a count. 0.20 percent is the lower end of that
# range. at the partition counts used here it rounds to one cluster per pass
DEDRIFT_N_LARGEST_FRACTION: float = 0.0020


class DeDriftMaintainer(Maintainer):
    """DeDrift Lazy, Split or Hybrid, run whenever a set share of the index has changed."""

    name = "dedrift"

    def __init__(
        self,
        variant: str = "hybrid",
        n_largest: int | None = None,
        n_largest_fraction: float = DEDRIFT_N_LARGEST_FRACTION,
        update_fraction: float = REBUILD_FRACTION_THRESHOLD,
        check_interval: int = MAINTENANCE_CHECK_INTERVAL,
        seed: int = 42,
    ) -> None:
        """Create the policy for one variant, "lazy", "split" or "hybrid".

        `n_largest` fixes the number of largest clusters in B1 and overrides `n_largest_fraction`.
        """
        super().__init__(check_interval=check_interval)
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.n_largest = n_largest
        self.n_largest_fraction = n_largest_fraction
        self.update_fraction = update_fraction
        self.seed = seed
        self.name = f"dedrift_{variant}"
        self._last_update_total = 0

    def start(self, index: IVFIndex, step: int) -> None:
        """Start counting updates from the start of the stream, as the global rebuild does."""
        super().start(index, step)
        self._last_update_total = index._total_inserts + index._total_deletes

    def should_maintain(self, index: IVFIndex, step: int) -> bool:
        """Return True once the updates since the last step reach the set share of live vectors."""
        if not index.partitions:
            return False
        total = index._total_inserts + index._total_deletes
        live = max(1, len(index.vector_to_partition))
        return (total - self._last_update_total) >= self.update_fraction * live

    def maybe_maintain(self, index: IVFIndex, step: int):
        """Run the base check, and restart the update count whenever a step ran."""
        report = super().maybe_maintain(index, step)
        if report is not None:
            self._last_update_total = index._total_inserts + index._total_deletes
        return report

    def maintain(self, index: IVFIndex) -> MaintenanceReport:
        """Run the variant's update step and return what it did."""
        report = MaintenanceReport(triggered=True)
        if self.variant in ("lazy", "hybrid"):
            self._lazy(index, report)
        if self.variant in ("split", "hybrid"):
            self._split(index, report)
        return report

    def n_largest_for(self, n_partitions: int) -> int:
        """Return the number of largest clusters to collect into B1, at least one."""
        if self.n_largest is not None:
            return self.n_largest
        return max(1, round(self.n_largest_fraction * n_partitions))

    def _lazy(self, index: IVFIndex, report: MaintenanceReport) -> None:
        """Move each centroid to the mean of its live vectors, without reassigning any vector.

        The work is one read of every live vector.
        """
        for pid in list(index.partition_ids()):
            p = index.partitions[pid]
            if p.size() == 0:
                continue
            report.vectors_processed += p.size()
            index.recompute_centroid(pid)
            report.num_centroids_recomputed += 1

    def _split(self, index: IVFIndex, report: MaintenanceReport) -> None:
        """Recluster the largest clusters together with the smallest ones, keeping K constant."""
        from sklearn.cluster import KMeans

        # every partition counts, empty ones included. mu is "the median cluster size of the
        # whole IVF", so an empty cluster counts as size zero in the median, and it belongs in B2
        # too, since B2 supplies free cluster slots and an empty cluster is exactly that
        sizes = {pid: p.size() for pid, p in index.partitions.items()}
        if len(sizes) < 2:
            return
        median_size = float(np.median(list(sizes.values())))
        if median_size <= 0:
            # more than half the index is empty, so k2 is undefined. DeDrift assumes a populated
            # index and does not cover this case, so the step is skipped
            return

        by_size = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
        k = min(self.n_largest_for(len(by_size)), len(by_size))
        largest = [pid for pid, _ in by_size[:k]]
        n_b1 = sum(sizes[pid] for pid in largest)

        k2 = int(np.ceil(n_b1 / median_size))
        # k2 must exceed k for this to be a repartition, and cannot use more clusters than exist
        k2 = max(k + 1, k2)
        k2 = min(k2, len(by_size))
        n_smallest = k2 - k
        if n_smallest <= 0:
            return
        smallest = [pid for pid, _ in by_size[-n_smallest:] if pid not in set(largest)]
        if len(largest) + len(smallest) != k2:
            # the largest and smallest sets overlap, the index is too small to repartition
            return

        involved = largest + smallest
        parts = [index.partitions[pid] for pid in involved]
        all_ids: list[int] = []
        blocks: list[np.ndarray] = []
        for p in parts:
            live_ids, live_vecs = p.get_live_vectors()
            if live_ids:
                all_ids.extend(live_ids)
                blocks.append(live_vecs)
        if len(all_ids) < k2:
            return

        matrix = np.vstack(blocks).astype(np.float32)
        # the vectors are charged one read, and the k-means distances are counted separately below
        report.vectors_processed += len(all_ids)

        km = KMeans(n_clusters=k2, random_state=self.seed, n_init=3)
        labels = km.fit_predict(matrix)
        report.kmeans_distance_evaluations += kmeans_distance_evaluations(km, len(matrix))
        centroids = km.cluster_centers_.astype(np.float32)

        children = [Partition(-1, centroids[i]) for i in range(k2)]
        for vid, vec, lbl in zip(all_ids, matrix, labels, strict=True):
            children[int(lbl)].add(vid, vec)
        children = [c for c in children if c.size() > 0]
        # the result must have exactly k2 clusters to keep K constant. scikit-learn moves empty
        # clusters, so this should never fail, and the check makes sure it is noticed if it does
        assert len(children) == k2, f"DeDrift split kept {len(children)} clusters, expected {k2}"
        if not children:
            return

        # join the involved clusters into one and split that into the k-means result, which keeps
        # the map from vectors to partitions consistent through the swap
        merged = merge_partitions(parts)
        index.apply_merge(involved, merged)
        merged_pid = sorted(index.partition_ids())[-1]
        index.apply_split(merged_pid, children)

        report.num_merges += 1
        report.num_splits += len(children)
        report.num_repartitioned += len(all_ids)

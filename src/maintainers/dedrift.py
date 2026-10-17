"""the three DeDrift strategies, baranchuk et al., ICCV 2023, section 5.1"""

from __future__ import annotations

import numpy as np

from src.config import MAINTENANCE_CHECK_INTERVAL, REBUILD_FRACTION_THRESHOLD
from src.index import IVFIndex
from src.maintainers.base import Maintainer, kmeans_distance_evaluations, merge_partitions
from src.partition import Partition
from src.types import MaintenanceReport

VARIANTS = ("lazy", "split", "hybrid")

# share of K taken as the largest clusters, DeDrift section 5.1
DEDRIFT_N_LARGEST_FRACTION: float = 0.0020


class DeDriftMaintainer(Maintainer):
    """DeDrift Lazy, split or Hybrid"""

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
        """create the policy for one variant, 'lazy', 'split' or 'hybrid'"""
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
        """start counting updates at the start of the stream"""
        super().start(index, step)
        self._last_update_total = index._total_inserts + index._total_deletes

    def should_maintain(self, index: IVFIndex, step: int) -> bool:
        """act once enough of the vectors have changed"""
        if not index.partitions:
            return False
        total = index._total_inserts + index._total_deletes
        live = max(1, len(index.vector_to_partition))
        return (total - self._last_update_total) >= self.update_fraction * live

    def maybe_maintain(self, index: IVFIndex, step: int):
        """run the base check, and restart the update count whenever a step ran"""
        report = super().maybe_maintain(index, step)
        if report is not None:
            self._last_update_total = index._total_inserts + index._total_deletes
        return report

    def maintain(self, index: IVFIndex) -> MaintenanceReport:
        """run the variant's update step and return what it did"""
        report = MaintenanceReport(triggered=True)
        if self.variant in ("lazy", "hybrid"):
            self._lazy(index, report)
        if self.variant in ("split", "hybrid"):
            self._split(index, report)
        return report

    def n_largest_for(self, n_partitions: int) -> int:
        """return the number of largest clusters to collect into B1, at least one"""
        if self.n_largest is not None:
            return self.n_largest
        return max(1, round(self.n_largest_fraction * n_partitions))

    def _lazy(self, index: IVFIndex, report: MaintenanceReport) -> None:
        """move each centroid to the mean of its vectors"""
        for pid in list(index.partition_ids()):
            p = index.partitions[pid]
            if p.size() == 0:
                continue
            report.vectors_processed += p.size()
            index.recompute_centroid(pid)
            report.num_centroids_recomputed += 1

    def _split(self, index: IVFIndex, report: MaintenanceReport) -> None:
        """recluster the largest and smallest clusters, keeping K"""
        from sklearn.cluster import KMeans

        # empty partitions count in the median
        sizes = {pid: p.size() for pid, p in index.partitions.items()}
        if len(sizes) < 2:
            return
        median_size = float(np.median(list(sizes.values())))
        if median_size <= 0:
            # mostly empty index, skip
            return

        by_size = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
        k = min(self.n_largest_for(len(by_size)), len(by_size))
        largest = [pid for pid, _ in by_size[:k]]
        n_b1 = sum(sizes[pid] for pid in largest)

        k2 = int(np.ceil(n_b1 / median_size))
        k2 = max(k + 1, k2)
        k2 = min(k2, len(by_size))
        n_smallest = k2 - k
        if n_smallest <= 0:
            return
        smallest = [pid for pid, _ in by_size[-n_smallest:] if pid not in set(largest)]
        if len(largest) + len(smallest) != k2:
            # sets overlap, index too small
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
        # one read per vector, k-means counted below
        report.vectors_processed += len(all_ids)

        km = KMeans(n_clusters=k2, random_state=self.seed, n_init=3)
        labels = km.fit_predict(matrix)
        report.kmeans_distance_evaluations += kmeans_distance_evaluations(km, len(matrix))
        centroids = km.cluster_centers_.astype(np.float32)

        children = [Partition(-1, centroids[i]) for i in range(k2)]
        for vid, vec, lbl in zip(all_ids, matrix, labels, strict=True):
            children[int(lbl)].add(vid, vec)
        children = [c for c in children if c.size() > 0]
        # k-means must return exactly k2 clusters
        assert len(children) == k2, f"DeDrift split kept {len(children)} clusters, expected {k2}"
        if not children:
            return

        # swap the clusters in one step
        merged = merge_partitions(parts)
        index.apply_merge(involved, merged)
        merged_pid = sorted(index.partition_ids())[-1]
        index.apply_split(merged_pid, children)

        report.num_merges += 1
        report.num_splits += len(children)
        report.num_repartitioned += len(all_ids)

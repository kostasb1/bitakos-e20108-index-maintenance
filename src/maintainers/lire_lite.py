"""LIRE, the update protocol of SPFresh (SOSP 2023), in memory.

One pass does three things in this order. It removes empty partitions. It splits every partition
whose stored length, tombstones included, exceeds the split limit, compacting it first and
skipping the split if it then holds fewer than the limit. Then it merges every partition of at most
the merge limit live vectors into the nearest of 64 neighbours that still fits. Each split and
merge ends with the reassignment checks of SPFresh section 3.3. The same class grows the starting
index of every policy, see `src/staple.py`.
"""

from __future__ import annotations

from src.config import (
    BOUNDARY_REASSIGN_TOP_K,
    LIRE_MERGE_CANDIDATES,
    MAINTENANCE_CHECK_INTERVAL,
    MAX_PARTITION_SIZE,
    MIN_PARTITION_SIZE,
)
from src.index import IVFIndex
from src.maintainers.base import (
    Maintainer,
    collect_empty_partitions,
    merge_into_survivor,
    nearest_partitions_in_order,
    reassign_lire_after_merge,
    reassign_lire_after_split,
    split_partition_2means,
    top_k_nearest_partitions,
)
from src.types import MaintenanceReport


class LireLiteMaintainer(Maintainer):
    """Keep every partition between a merge limit and a split limit, as SPFresh LIRE does."""

    name = "lire_lite"

    def __init__(
        self,
        max_partition_size: int = MAX_PARTITION_SIZE,
        min_partition_size: int = MIN_PARTITION_SIZE,
        boundary_top_k: int = BOUNDARY_REASSIGN_TOP_K,
        merge_candidates: int = LIRE_MERGE_CANDIDATES,
        check_interval: int = MAINTENANCE_CHECK_INTERVAL,
        enable_merge: bool = True,
        enable_compact: bool = True,
        seed: int = 42,
    ) -> None:
        """Create the policy with its split and merge limits.

        `boundary_top_k` is the number of neighbouring partitions the reassignment considers, and
        `merge_candidates` the number of neighbours a merge looks through for a partner.
        """
        super().__init__(check_interval=check_interval)
        self.max_partition_size = max_partition_size
        self.min_partition_size = min_partition_size
        self.boundary_top_k = boundary_top_k
        self.merge_candidates = merge_candidates
        self.enable_merge = enable_merge
        self.enable_compact = enable_compact
        self.seed = seed

    def should_maintain(self, index: IVFIndex, step: int) -> bool:
        """Return True if any partition is too long, small enough to merge, or empty.

        The split check reads the stored length, which still includes the vectors deleted since
        the last compaction, because SPFresh checks the length of the posting, SPFresh section 4.1.
        """
        for p in index.partitions.values():
            if p.total_size() > self.max_partition_size:
                return True
            # at or below the threshold, as in the SPFresh implementation
            if self.enable_merge and 0 < p.size() <= self.min_partition_size:
                return True
            # an empty partition is removed rather than merged, so it triggers a pass even when
            # merging is switched off
            if p.is_empty():
                return True
        return False

    def _collect(self, index: IVFIndex, pid: int, report: MaintenanceReport) -> bool:
        """Compact a partition if it has tombstones, and return whether it did.

        The work is the stored length, counted in compaction_reads, which the reported work leaves
        out.
        """
        partition = index.partitions[pid]
        if not (self.enable_compact and partition.tombstones):
            return False
        report.vectors_processed += partition.total_size()
        report.compaction_reads += partition.total_size()
        index.compact_partition(pid)
        return True

    def maintain(self, index: IVFIndex) -> MaintenanceReport:
        """Run one LIRE pass, collect, split, then merge, and return what it did."""
        report = MaintenanceReport(triggered=True)
        touched: set[int] = set()

        # first, so an empty partition is never a split neighbour or a merge target
        report.num_collected += len(collect_empty_partitions(index))

        def _length(p):
            """The stored length, tombstones included, which the split limit is checked against."""
            return p.total_size()

        oversized = [
            pid
            for pid in list(index.partition_ids())
            if _length(index.partitions[pid]) > self.max_partition_size
        ]
        for pid in oversized:
            if pid not in index.partitions:
                continue

            partition = index.partitions[pid]

            if self._collect(index, pid, report):
                # SPFresh section 4.2.1, the dead vectors of a long partition are removed first and
                # the limit checked again, and a partition that then fits is not split. it fits
                # only strictly below the limit, as in the SPFresh implementation
                touched.add(pid)
                if partition.size() < self.max_partition_size:
                    continue

            report.vectors_processed += partition.size()
            children = split_partition_2means(partition, random_state=self.seed, meter=report)
            if len(children) < 2:
                continue

            neighbors = top_k_nearest_partitions(
                index, partition.centroid, self.boundary_top_k, exclude={pid}, meter=report
            )

            index.apply_split(pid, children)
            report.num_splits += 1

            new_pids = sorted(index.partition_ids())[-len(children):]
            touched.update(new_pids)

            # SPFresh section 3.3, the two necessary conditions decide which vectors are checked,
            # with the deleted centroid as the reference. a checked vector moves to the nearest
            # centroid in the local set, where SPFresh searches the whole index, and a partition
            # a reassignment pushes over the limit is split on the next pass rather than at once.
            # a whole index search was measured to halve the misplaced vectors at six times the
            # work with no consistent effect on cost
            moves, examined = reassign_lire_after_split(
                index, partition.centroid, new_pids, neighbors, meter=report)
            report.num_reassigned += moves
            report.num_examined += examined
            report.vectors_processed += moves
            touched.update(neighbors)

        if self.enable_merge:
            undersized = [
                pid
                for pid in list(index.partition_ids())
                if 0 < index.partitions[pid].size() <= self.min_partition_size
            ]
            for pid in undersized:
                if pid not in index.partitions:
                    continue

                partition = index.partitions[pid]
                # the list above is built once, and a partition can grow before its turn, by an
                # earlier merge or a reassignment, so the size is checked again here, as SPFresh
                # recounts it when the merge job runs
                if not 0 < partition.size() <= self.min_partition_size:
                    continue
                # SPFresh section 3.2, the merge walks the nearest partitions in order and takes
                # the first whose combined length still fits under the limit. the fit is the live
                # length of this partition plus the stored length of the neighbour, since SPFresh
                # records a length on each append and resets it only when the posting is rewritten
                candidates = nearest_partitions_in_order(
                    index, partition.centroid, self.merge_candidates, exclude={pid},
                    meter=report)
                nbr_pid = next(
                    (c for c in candidates
                     if partition.size() + index.partitions[c].total_size()
                     < self.max_partition_size),
                    None,
                )
                if nbr_pid is None:
                    continue
                nbr = index.partitions[nbr_pid]

                # SPFresh section 3.2, delete the shorter partition with its centroid and append
                # its vectors to the other one, which keeps its own centroid
                short_pid, long_pid = (
                    (pid, nbr_pid) if partition.size() <= nbr.size() else (nbr_pid, pid))
                short_centroid = index.partitions[short_pid].centroid.copy()
                # SPFresh writes the merged partition with the live vectors of both, so the dead
                # vectors of both are removed here
                self._collect(index, short_pid, report)
                self._collect(index, long_pid, report)
                # read before the merge, which empties the shorter partition into the other
                short_size = index.partitions[short_pid].size()
                long_size = index.partitions[long_pid].size()
                moved_ids = merge_into_survivor(index, short_pid, long_pid)
                # both partitions are charged, since SPFresh rewrites both, and every policy that
                # merges charges the same way, so merge work is comparable between them
                report.vectors_processed += short_size + long_size
                report.num_merges += 1
                touched.add(long_pid)
                nbrs = top_k_nearest_partitions(
                    index, index.partitions[long_pid].centroid,
                    self.boundary_top_k, exclude={long_pid}, meter=report)
                moves = reassign_lire_after_merge(
                    index, long_pid, moved_ids, nbrs, short_centroid, meter=report)
                report.num_reassigned += moves
                report.vectors_processed += moves

        report.partitions_touched = sorted(touched)
        return report

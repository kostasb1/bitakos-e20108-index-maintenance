"""Quake's maintenance loop (OSDI 2025)"""

from __future__ import annotations

import numpy as np

from src.config import (
    MAINTENANCE_CHECK_INTERVAL,
    SPLIT_ACCESS_ALPHA,
)
from src.index import IVFIndex
from src.maintainers.base import (
    Maintainer,
    apply_partition_deletion,
    collect_empty_partitions,
    plan_partition_deletion,
    reassign_boundary,
    top_k_nearest_partitions,
)
from src.partition import Partition
from src.types import CostModel, MaintenanceReport

# tau, Quake paper p. 15
QUAKE_TAU_NS: float = 250.0
# alpha, measured on this index (Quake uses 0.9)
QUAKE_ALPHA: float = SPLIT_ACCESS_ALPHA
# defaults from the Quake common.h
QUAKE_MIN_PARTITION_SIZE: int = 32
QUAKE_PARTITION_REDUCTION_THRESHOLD: float = 0.3
# refinement radius and one k-means iteration, Quake paper p. 10
QUAKE_REFINEMENT_RADIUS: int = 50
# split with 5 Faiss k-means iterations
QUAKE_SPLIT_ITERATIONS: int = 5
# Faiss clustering defaults
FAISS_MAX_POINTS_PER_CENTROID: int = 256
FAISS_CLUSTERING_SEED: int = 1234
FAISS_EMPTY_CLUSTER_EPS: float = 1.0 / 1024.0


def _live_sum(p: Partition) -> np.ndarray:
    """sum of a partition's live vectors, in float64"""
    # view, not a copy
    live_vecs, _ = p._live_block()
    return live_vecs.sum(axis=0, dtype=np.float64)


def faiss_split(partition: Partition, niter: int = QUAKE_SPLIT_ITERATIONS, k: int = 2,
                seed: int = FAISS_CLUSTERING_SEED, meter=None) -> list[Partition]:
    """split a partition with Faiss k-means, as Quake does"""
    live_ids, live_vecs = partition.get_live_vectors()
    n = len(live_vecs)
    if n < k:
        single = Partition(-1, partition.centroid)
        for vid, v in zip(live_ids, live_vecs, strict=True):
            single.add(vid, v)
        return [single]

    x = np.asarray(live_vecs, dtype=np.float32)
    rng = np.random.default_rng(seed)
    cap = FAISS_MAX_POINTS_PER_CENTROID * k
    train = x[rng.permutation(n)[:cap]] if n > cap else x
    centroids = train[rng.permutation(len(train))[:k]].copy()

    def assign(v: np.ndarray, c: np.ndarray) -> np.ndarray:
        """return the index of the nearest centre for every row of `v`"""
        d = ((v[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)

    for _ in range(niter):
        labels = assign(train, centroids)
        counts = np.bincount(labels, minlength=k)
        for j in range(k):
            if counts[j]:
                centroids[j] = train[labels == j].mean(axis=0)
        for j in np.flatnonzero(counts == 0):
            p = counts / counts.sum()
            donor = int(rng.choice(k, p=p))
            sign = np.where(np.arange(x.shape[1]) % 2 == 0, 1.0, -1.0).astype(np.float32)
            base = centroids[donor].copy()
            centroids[j] = base * (1 + sign * FAISS_EMPTY_CLUSTER_EPS)
            centroids[donor] = base * (1 - sign * FAISS_EMPTY_CLUSTER_EPS)
            counts[j] = counts[donor] // 2
            counts[donor] -= counts[j]

    labels = assign(x, centroids)
    if meter is not None:
        meter.kmeans_distance_evaluations += niter * len(train) * k + n * k

    children = [Partition(-(j + 1), centroids[j]) for j in range(k)]
    for vid, v, lbl in zip(live_ids, live_vecs, labels, strict=True):
        children[int(lbl)].add(vid, v)
    return [c for c in children if c.size() > 0]


class CostDrivenQuakeMaintainer(Maintainer):
    """Quake, act only when the predicted cost change clears tau"""

    name = "cost_driven_quake"

    def __init__(
        self,
        cost_model: CostModel,
        tau: float = QUAKE_TAU_NS,
        alpha: float = QUAKE_ALPHA,
        max_actions_per_maintain: int = 20,
        boundary_top_k: int = QUAKE_REFINEMENT_RADIUS,
        check_interval: int = MAINTENANCE_CHECK_INTERVAL,
        access_decay: float | None = None,
        nprobe: int = 10,
        seed: int = 42,
        min_partition_size: int = QUAKE_MIN_PARTITION_SIZE,
        partition_reduction_threshold: float = QUAKE_PARTITION_REDUCTION_THRESHOLD,
        recompute_centroids: bool = True,
        delete_tau: float | None = None,
    ) -> None:
        """create the policy with its cost model and thresholds"""
        super().__init__(check_interval=check_interval)
        self.cost_model = cost_model
        self.tau = tau
        self.delete_tau = tau if delete_tau is None else delete_tau
        self.alpha = alpha
        self.max_actions = max_actions_per_maintain
        self.boundary_top_k = boundary_top_k
        # access window equals the maintenance interval, Quake paper p. 15
        self.access_decay = access_decay
        self.nprobe = nprobe
        self.seed = seed
        self.min_partition_size = min_partition_size
        self.partition_reduction_threshold = partition_reduction_threshold
        self.recompute_centroids = recompute_centroids
        # size of each partition at the previous pass
        self._snapshot_sizes: dict[int, int] | None = None
        # sum of live vectors at the previous pass
        self._snapshot_sums: dict[int, np.ndarray] = {}
        # live count at handover
        self._total_vectors: int | None = None

    def start(self, index: IVFIndex, step: int) -> None:
        """snapshot the index at handover"""
        super().start(index, step)
        self._snapshot_sizes = {}
        self._snapshot(index)
        self._total_vectors = len(index.vector_to_partition)

    def _snapshot(self, index: IVFIndex, pids=None) -> None:
        """record the size and the vector sum of the given partitions, or of all of them"""
        for pid in index.partition_ids() if pids is None else pids:
            if pid in index.partitions:
                self._snapshot_sizes[pid] = index.partitions[pid].size()
                self._snapshot_sums[pid] = _live_sum(index.partitions[pid])

    def _access_fraction(self, index: IVFIndex) -> dict[int, float]:
        """access fraction A of every partition"""
        total = sum(p.access_count for p in index.partitions.values())
        if total <= 0:
            return {pid: 0.0 for pid in index.partitions}
        queries = index.query_count(self.nprobe)
        return {pid: min(1.0, p.access_count / queries)
                for pid, p in index.partitions.items()}

    def _centroid_delta(self, n_partitions: int, added: int) -> float:
        """change in centroid scan cost from adding partitions"""
        lam = self.cost_model.centroid_latency
        return lam(n_partitions + added) - lam(n_partitions)

    def delta_split_estimate(self, access: float, size: int, n_partitions: int) -> float:
        """return the predicted cost change of splitting a partition, Quake equation 6"""
        if size < 2:
            return 0.0
        lam = self.cost_model.scan_latency
        return (self._centroid_delta(n_partitions, +1)
                - access * lam(size)
                + 2.0 * self.alpha * access * lam(size // 2))

    def delta_split_exact(self, access: float, children: list, n_partitions: int) -> float:
        """return the cost change of a split over the real children, Quake equation 4"""
        lam = self.cost_model.scan_latency
        parent_size = sum(c.size() for c in children)
        after = sum(self.alpha * access * lam(c.size()) for c in children)
        return (self._centroid_delta(n_partitions, +1)
                - access * lam(parent_size)
                + after)

    def delta_merge_estimate(
        self, access: float, size: int,
        receivers: list[tuple[float, int]], n_partitions: int,
    ) -> float:
        """return the predicted cost change of deleting a partition, Quake equation 5"""
        if n_partitions < 2 or not receivers:
            return 0.0
        lam = self.cost_model.scan_latency
        r = len(receivers)
        d_size = size / r
        d_access = access / r
        swell = sum((a + d_access) * lam(int(s + d_size)) - a * lam(s) for a, s in receivers)
        return (self._centroid_delta(n_partitions, -1)
                - access * lam(size)
                + swell)

    def delta_merge_exact(
        self, access: float, size: int,
        receivers: list[tuple[float, int, int]], n_partitions: int,
    ) -> float:
        """return the cost change of a delete over the receivers its vectors really go to"""
        if n_partitions < 2 or not receivers:
            return 0.0
        lam = self.cost_model.scan_latency
        d_access = access / len(receivers)
        swell = sum((a + d_access) * lam(s + moved) - a * lam(s) for a, s, moved in receivers)
        return (self._centroid_delta(n_partitions, -1)
                - access * lam(size)
                + swell)

    def delta_merge_estimate_uniform(
        self, access: float, size: int, n_partitions: int,
        avg_access: float, avg_size: float,
    ) -> float:
        """return the stage 1 delete estimate exactly as the released code computes it"""
        t = n_partitions
        if t <= 1:
            return 0.0
        lam = self.cost_model.scan_latency
        overhead = self.cost_model.centroid_latency(t - 1) - self.cost_model.centroid_latency(t)
        cost_old = (t - 1) * avg_access * lam(avg_size) + access * lam(size)
        merged_size = avg_size + size / (t - 1)
        merged_access = avg_access + access / (t - 1)
        if size < t:
            cost_new = (size * merged_access * lam(avg_size + 1)
                        + (t - size - 1) * merged_access * lam(avg_size))
        else:
            cost_new = (t - 1) * merged_access * lam(int(np.ceil(merged_size)))
        return overhead + cost_new - cost_old

    def _decide(self, index: IVFIndex, frac: dict[int, float],
                delete_factors: dict[int, float], report: MaintenanceReport):
        """decide every delete and split of the pass from one snapshot of the index"""
        n_parts = len(index.partitions)
        avg_access = (index.vectors_scanned_routed / index.queries_routed / self._total_vectors
                      if index.queries_routed > 0 and self._total_vectors else 0.0)
        avg_size = sum(p.size() for p in index.partitions.values()) // max(1, n_parts)
        cost_deletes: list[tuple[float, int]] = []
        splits: list[tuple[float, int]] = []
        forced: list[int] = []
        for pid, p in list(index.partitions.items()):
            size = p.size()
            access = frac.get(pid, 0.0)
            chosen = False
            dc = self.delta_merge_estimate_uniform(access, size, n_parts, avg_access, avg_size)
            if dc < -self.delete_tau:
                # delete check follows Quake equation 5
                if size > self.min_partition_size:
                    plan = plan_partition_deletion(index, pid, meter=report)
                    receivers = [(frac.get(r, 0.0), index.partitions[r].size(), len(vids))
                                 for r, vids in plan.items()]
                    if self.delta_merge_exact(access, size, receivers, n_parts) < -self.delete_tau:
                        cost_deletes.append((dc, pid))
                        chosen = True
                    else:
                        report.num_merges_declined += 1
                else:
                    cost_deletes.append((dc, pid))
                    chosen = True
            elif size > self.min_partition_size:
                # split only above the minimum size
                ds = self.delta_split_estimate(access, size, n_parts)
                if ds < -self.tau:
                    splits.append((ds, pid))
                    chosen = True
            # forced delete for partitions that shrank too much
            if not chosen and delete_factors.get(pid, -1.0) > self.partition_reduction_threshold:
                forced.append(pid)
        cost_deletes.sort()
        splits.sort()
        return ([pid for _, pid in cost_deletes[: self.max_actions]], forced,
                [pid for _, pid in splits[: self.max_actions]])

    def _delete_together(self, index: IVFIndex, pids: list[int],
                         report: MaintenanceReport) -> set[int]:
        """delete the given partitions in one step, and return the ids deleted"""
        doomed = [pid for pid in dict.fromkeys(pids) if pid in index.partitions]
        doomed = doomed[: max(0, len(index.partitions) - 1)]
        gone = set(doomed)
        for pid in doomed:
            size_before = index.partitions[pid].size()
            plan = plan_partition_deletion(index, pid, meter=report, exclude=gone)
            moved = apply_partition_deletion(index, pid, plan)
            # charge the read and the moved vectors
            report.vectors_processed += size_before + moved
            report.num_reassigned += moved
            report.num_merges += 1
        return gone

    def _split_and_refine(self, index: IVFIndex, frac: dict[int, float], splits: list[int],
                          report: MaintenanceReport) -> None:
        """split the chosen partitions, then refine around them"""
        children: list[int] = []
        for pid in splits:
            p = index.partitions[pid]
            report.vectors_processed += p.size()
            parts = faiss_split(p, meter=report)
            if len(parts) < 2:
                continue
            # keep the split only if the exact check still passes
            exact = self.delta_split_exact(frac.get(pid, 0.0), parts, len(index.partitions))
            if exact >= -self.tau:
                report.num_splits_declined += 1
                continue
            index.apply_split(pid, parts)
            report.num_splits += 1
            children.extend(sorted(index.partition_ids())[-len(parts):])
        if not children:
            return
        # refine the children with their neighbours
        refine: set[int] = set()
        for c in children:
            refine.update(top_k_nearest_partitions(
                index, index.partitions[c].centroid, self.boundary_top_k, meter=report))
        moves = reassign_boundary(index, sorted(refine), [], meter=report)
        report.num_reassigned += moves
        report.vectors_processed += moves
        # snapshot after refinement
        self._snapshot(index, children + sorted(refine))

    def should_maintain(self, index: IVFIndex, step: int) -> bool:
        """return True once any partition has been accessed"""
        return any(p.access_count > 0 for p in index.partitions.values())

    def maintain(self, index: IVFIndex) -> MaintenanceReport:
        """run one Quake pass"""
        report = MaintenanceReport(triggered=True)

        if self.access_decay is not None:
            for p in index.partitions.values():
                p.decay_access(self.access_decay)
            index.queries_routed *= self.access_decay
            index.vectors_scanned_routed *= self.access_decay

        # no handover, treat as just built
        if self._snapshot_sizes is None:
            self._snapshot_sizes = {}
            self._snapshot(index)
        if self._total_vectors is None:
            self._total_vectors = len(index.vector_to_partition)

        # shrinkage since the previous pass
        delete_factors = self._delete_factors(index)

        # step 1, refresh changed centroids
        if self.recompute_centroids:
            for pid in list(index.partition_ids()):
                p = index.partitions[pid]
                if p.size() == 0:
                    continue
                # unseen partition, all of it counts as change
                previous = self._snapshot_sizes.get(pid)
                delta = p.size() if previous is None else abs(p.size() - previous)
                if delta == 0:
                    continue
                report.vectors_processed += delta
                if previous is None:
                    index.recompute_centroid(pid)
                else:
                    shift = _live_sum(p) - self._snapshot_sums[pid]
                    p.update_centroid(
                        ((previous * p.centroid + shift) / p.size()).astype(np.float32))
                    index._invalidate_cache()
                report.num_centroids_recomputed += 1

        # snapshot at the start of the pass
        self._snapshot(index)

        frac = self._access_fraction(index)
        cost_deletes, forced_deletes, splits = self._decide(index, frac, delete_factors, report)
        deleted = self._delete_together(index, cost_deletes + forced_deletes, report)
        report.num_forced_merges += len(deleted & set(forced_deletes))
        self._split_and_refine(index, frac, splits, report)
        # step 7, delete empty partitions
        report.num_collected += len(collect_empty_partitions(index))
        self._snapshot_sizes = {pid: s for pid, s in self._snapshot_sizes.items()
                                if pid in index.partitions}
        self._snapshot_sums = {pid: s for pid, s in self._snapshot_sums.items()
                               if pid in index.partitions}
        if self.access_decay is None:
            # close the access window
            for p in index.partitions.values():
                p.access_count = 0.0
            index.queries_routed = 0.0
            index.vectors_scanned_routed = 0.0
        return report

    def _delete_factors(self, index: IVFIndex) -> dict[int, float]:
        """how much each partition shrank since the last pass"""
        factors: dict[int, float] = {}
        for pid, p in index.partitions.items():
            previous = self._snapshot_sizes.get(pid)
            if previous is None or previous <= 0:
                continue
            lost = previous - p.size()
            if lost > 0:
                factors[pid] = lost / previous
        return factors

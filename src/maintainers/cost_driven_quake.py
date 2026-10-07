"""The maintenance loop of Quake (OSDI 2025), as the authors' released implementation performs it.

`maintain` is one whole pass, in the order of Algorithm 4.2 of the thesis. It refreshes changed
centroids by the change since the last pass, decides every delete and split from one snapshot,
applies the deletes together, splits and checks each split on the real children, refines the 50
nearest partitions of every child, removes empty partitions and resets the access window. Run with
tau at 50 ns it is the declared sensitivity configuration `cost_driven_quake_tau50`, see
`scripts/final_sweep.build`.

References to Quake's code name the file and function in the released implementation, for example
maintenance_policies.cpp, so each step can be checked against it.
"""

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

# the threshold tau, from the maintenance parameters on page 15 of the final Quake paper, "We set
# tau = 250ns"
QUAKE_TAU_NS: float = 250.0
# the share of the parent access each child of a split is assumed to keep. Quake publishes 0.9,
# "We fix alpha = 0.9, which worked well across all benchmarks". this uses 0.86, the value measured
# on this index, because IVFIndex.apply_split credits the children with that value, and the check
# of a split must predict what the index will actually assign
QUAKE_ALPHA: float = SPLIT_ACCESS_ALPHA
# from the released policy loop and its defaults in common.h. the minimum partition size gates the
# split, and it also gates the check of a delete, so a partition at or below it is deleted on the
# estimate alone. the reduction threshold drives a delete that bypasses the cost model, fired when
# a partition has lost more than this share of the size it held at the previous pass
QUAKE_MIN_PARTITION_SIZE: int = 32
QUAKE_PARTITION_REDUCTION_THRESHOLD: float = 0.3
# Quake page 10, "For LIRE and Quake, we set the partition refinement radius r = 50" and "use one
# iteration of k-means for refinement". one iteration is one assignment pass against the existing
# centroids, which is what reassign_boundary does
QUAKE_REFINEMENT_RADIUS: int = 50
# the split is Faiss k-means with 5 iterations, the default of split_knn_iterations in common.h,
# and every other parameter at its Faiss default
QUAKE_SPLIT_ITERATIONS: int = 5
# Faiss ClusteringParameters defaults, max_points_per_centroid and seed, and the relative nudge
# Faiss applies when it revives an empty centre
FAISS_MAX_POINTS_PER_CENTROID: int = 256
FAISS_CLUSTERING_SEED: int = 1234
FAISS_EMPTY_CLUSTER_EPS: float = 1.0 / 1024.0


def _live_sum(p: Partition) -> np.ndarray:
    """Return the sum of a partition's live vectors, in float64 so differences keep their digits."""
    # the view rather than a copy, since this runs for every partition on every pass
    live_vecs, _ = p._live_block()
    return live_vecs.sum(axis=0, dtype=np.float64)


def faiss_split(partition: Partition, niter: int = QUAKE_SPLIT_ITERATIONS, k: int = 2,
                seed: int = FAISS_CLUSTERING_SEED, meter=None) -> list[Partition]:
    """Split a partition as Quake's CPU path does, Faiss k-means followed by one assignment.

    Faiss trains on at most max_points_per_centroid times k points, a random sample when the
    partition is larger, starts the centres at k distinct training points, and runs exactly
    `niter` rounds of assign then average, with no early stop. A centre left empty is revived as a
    copy of a non empty one, chosen in proportion to size, and the two are nudged apart. Every
    vector of the partition is then assigned once to the final centres.

    The work charged is what that computes, `niter` rounds over the training points against k
    centres plus the final assignment of all n vectors, all counted as k-means distances. The
    random draws use numpy rather than the Faiss generator, so they differ from Faiss's for the
    same seed while the procedure is the same.
    """
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
        """Return the index of the nearest centre for every row of `v`."""
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
    """Quake section 4.2, an action is taken only when its predicted cost change clears tau.

    The predicted change delta C includes the cost of the centroid a split adds or a delete
    removes, so it can decline a split and can price a delete at all.
    """

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
        """Create the policy with its cost model and thresholds.

        `tau` is the split threshold and `delete_tau` the delete threshold, equal to `tau` unless
        given. The released code keeps the two separate, while the paper states a single tau.
        `max_actions_per_maintain` caps the deletes and the splits per pass, and the reported
        runs set it high enough that it never binds.
        """
        super().__init__(check_interval=check_interval)
        self.cost_model = cost_model
        self.tau = tau
        self.delete_tau = tau if delete_tau is None else delete_tau
        self.alpha = alpha
        self.max_actions = max_actions_per_maintain
        self.boundary_top_k = boundary_top_k
        # Quake page 15, "The window size for access frequency statistics is set equal to the
        # maintenance interval", so A is the probe count since the previous pass. None gives that
        # window, reset at the end of every pass. a number instead decays the counts by that factor
        # each pass, which is not used by the reported runs
        self.access_decay = access_decay
        self.nprobe = nprobe
        self.seed = seed
        self.min_partition_size = min_partition_size
        self.partition_reduction_threshold = partition_reduction_threshold
        self.recompute_centroids = recompute_centroids
        # the size of each partition at the previous pass, what Quake divides the shrinkage by.
        # None until the stream hands over the index, see start
        self._snapshot_sizes: dict[int, int] | None = None
        # the sum of the live vectors of each partition at the same moment. the current sum minus this
        # is the change Quake adds to the stored centroid instead of averaging the partition again
        self._snapshot_sums: dict[int, np.ndarray] = {}
        # the live count when the policy takes over, fixed afterwards, as the hit count
        # tracker keeps it
        self._total_vectors: int | None = None

    def start(self, index: IVFIndex, step: int) -> None:
        """Take a snapshot of the index at handover, so the first pass sees only later changes.

        Quake resets every partition's change record after building, and the handover is the
        build as far as this policy can see.
        """
        super().start(index, step)
        self._snapshot_sizes = {}
        self._snapshot(index)
        self._total_vectors = len(index.vector_to_partition)

    def _snapshot(self, index: IVFIndex, pids=None) -> None:
        """Record the size and the vector sum of the given partitions, or of all of them."""
        for pid in index.partition_ids() if pids is None else pids:
            if pid in index.partitions:
                self._snapshot_sizes[pid] = index.partitions[pid].size()
                self._snapshot_sums[pid] = _live_sum(index.partitions[pid])

    def _access_fraction(self, index: IVFIndex) -> dict[int, float]:
        """Return A for every partition, the share of routed queries in the window that scanned it."""
        total = sum(p.access_count for p in index.partitions.values())
        if total <= 0:
            return {pid: 0.0 for pid in index.partitions}
        queries = index.query_count(self.nprobe)
        return {pid: min(1.0, p.access_count / queries)
                for pid, p in index.partitions.items()}

    def _centroid_delta(self, n_partitions: int, added: int) -> float:
        """Return the change in centroid scan cost when `added` partitions are added, delta O.

        The index is flat, so the level above the partitions is the list of centroids, which every
        query scans in full. A split adds a centroid and pays lambda_c(K plus 1) minus
        lambda_c(K), and a delete gets it back. Without this term delta C would be negative for
        every accessed partition at any size, and the policy would split without limit.
        """
        lam = self.cost_model.centroid_latency
        return lam(n_partitions + added) - lam(n_partitions)

    def delta_split_estimate(self, access: float, size: int, n_partitions: int) -> float:
        """Return the predicted cost change of splitting a partition, Quake equation 6.

        It assumes two equal halves, and each child keeps alpha times the parent access, so the
        two children together hold 2 alpha A, not A.
        """
        if size < 2:
            return 0.0
        lam = self.cost_model.scan_latency
        return (self._centroid_delta(n_partitions, +1)
                - access * lam(size)
                + 2.0 * self.alpha * access * lam(size // 2))

    def delta_split_exact(self, access: float, children: list, n_partitions: int) -> float:
        """Return the cost change of a split over the real children, Quake equation 4.

        Quake applies a split tentatively, measures the real sizes and keeps it only if the
        recomputed change still clears tau. This evaluates before applying instead, which reaches
        the same decision without undoing anything. Each child is still credited alpha times the
        parent access, since the real access after a split is not known until queries arrive.
        """
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
        """Return the predicted cost change of deleting a partition, Quake equation 5.

        The deleted partition's vectors and access are assumed to spread equally over the
        receivers, given as (access, size) pairs. Not used by the reported runs, which estimate
        with `delta_merge_estimate_uniform` and check with `delta_merge_exact`.
        """
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
        """Return the cost change of a delete over the receivers its vectors really go to.

        This is Quake equation 5 evaluated as the check of section 4.2.3, "we measure the actual
        resulting partition sizes (and the exact receiver partitions for merges)". `receivers`
        holds (access, size, vectors received) for each receiver. The access each receiver gains
        stays the equal share of the estimate, since the paper keeps the stage 1 frequency
        assumptions, so only the sizes and the receiver set become exact.
        """
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
        """Return the stage 1 delete estimate exactly as the released code computes it.

        This follows compute_delete_delta in maintenance_cost_estimator.cpp. The other T minus 1
        partitions are all taken as an average partition, of `avg_size` vectors at `avg_access`,
        and the deleted partition is spread equally over them. When it holds fewer vectors than
        there are partitions, at most `size` of them gain one vector each. This estimate decides
        which deletes are considered at all.
        """
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
        """Decide every delete and split of the pass from one snapshot of the index.

        This follows the decision loop of maintenance_policies.cpp. The partition count, the
        access fractions and the average size are fixed before the loop, and nothing is applied
        until every decision is taken. Returns the cost driven deletes, the forced deletes and the
        splits, the most beneficial first.

        The average access is the vectors each routed query scanned over the vectors at handover,
        averaged over the window, and the average size is the live total over the partition
        count in integer division, as in the released code.
        """
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
                # the check follows equation 5 of the paper over the receivers the vectors really go
                # to. the released check, compute_delete_delta_w_reassign, adds the deleted
                # partition cost where equation 5 subtracts it and charges every receiver the
                # whole partition, which would reject nearly every checked delete, so the paper is
                # followed here. as in the released loop, a partition at or below the minimum size
                # is deleted on the estimate alone. the receivers are its nearest other partitions
                # at the snapshot, including ones deleted later in the same pass
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
                # a split is offered only above the minimum size, and only to a partition the
                # delete branch did not consider
                ds = self.delta_split_estimate(access, size, n_parts)
                if ds < -self.tau:
                    splits.append((ds, pid))
                    chosen = True
            # a partition no other action chose that lost more than the threshold share of its
            # size since the previous pass is deleted whatever the cost model says. this is the
            # only action in the released policy that does not consult delta C
            if not chosen and delete_factors.get(pid, -1.0) > self.partition_reduction_threshold:
                forced.append(pid)
        cost_deletes.sort()
        splits.sort()
        return ([pid for _, pid in cost_deletes[: self.max_actions]], forced,
                [pid for _, pid in splits[: self.max_actions]])

    def _delete_together(self, index: IVFIndex, pids: list[int],
                         report: MaintenanceReport) -> set[int]:
        """Delete the given partitions in one step, and return the ids deleted.

        As in Quake's delete_partitions, every listed centroid is removed first and only then are
        the vectors of each reinserted, each into its nearest remaining partition, so no vector is
        moved twice. Each receiver keeps its own centroid, and at least one partition is always
        kept.
        """
        doomed = [pid for pid in dict.fromkeys(pids) if pid in index.partitions]
        doomed = doomed[: max(0, len(index.partitions) - 1)]
        gone = set(doomed)
        for pid in doomed:
            size_before = index.partitions[pid].size()
            plan = plan_partition_deletion(index, pid, meter=report, exclude=gone)
            moved = apply_partition_deletion(index, pid, plan)
            # the deleted partition is read and every moved vector is written, and the receivers
            # are not rewritten as a whole the way a pairwise merge rewrites its survivor
            report.vectors_processed += size_before + moved
            report.num_reassigned += moved
            report.num_merges += 1
        return gone

    def _split_and_refine(self, index: IVFIndex, frac: dict[int, float], splits: list[int],
                          report: MaintenanceReport) -> None:
        """Split the chosen partitions, checking each on its real children, then refine around them.

        Splits come after the deletes, so a partition splits what it holds once the deleted
        vectors have landed. One refinement then covers all the new children.
        """
        children: list[int] = []
        for pid in splits:
            p = index.partitions[pid]
            report.vectors_processed += p.size()
            parts = faiss_split(p, meter=report)
            if len(parts) < 2:
                continue
            # the split is applied only if the exact change over the real children still clears
            # tau
            exact = self.delta_split_exact(frac.get(pid, 0.0), parts, len(index.partitions))
            if exact >= -self.tau:
                report.num_splits_declined += 1
                continue
            index.apply_split(pid, parts)
            report.num_splits += 1
            children.extend(sorted(index.partition_ids())[-len(parts):])
        if not children:
            return
        # as in the local refinement of Quake, each child brings its own nearest partitions, itself
        # among them, and the union is refined together with one assignment pass against
        # unchanged centroids
        refine: set[int] = set()
        for c in children:
            refine.update(top_k_nearest_partitions(
                index, index.partitions[c].centroid, self.boundary_top_k, meter=report))
        moves = reassign_boundary(index, sorted(refine), [], meter=report)
        report.num_reassigned += moves
        report.vectors_processed += moves
        # Quake resets the change record of new and refined partitions, so they take their
        # snapshot after the refinement
        self._snapshot(index, children + sorted(refine))

    def should_maintain(self, index: IVFIndex, step: int) -> bool:
        """Return True once any partition has been accessed.

        Quake has no trigger of its own. It runs maintenance after every operation and skips a
        pass only while the access window is still empty, which an index no query has reached
        stands in for here.
        """
        return any(p.access_count > 0 for p in index.partitions.values())

    def maintain(self, index: IVFIndex) -> MaintenanceReport:
        """Run one Quake pass, refresh, decide, delete, split and refine, collect, reset the window."""
        report = MaintenanceReport(triggered=True)

        if self.access_decay is not None:
            for p in index.partitions.values():
                p.decay_access(self.access_decay)
            index.queries_routed *= self.access_decay
            index.vectors_scanned_routed *= self.access_decay

        # an index maintained without a handover, as in a quick check, is taken as just built
        if self._snapshot_sizes is None:
            self._snapshot_sizes = {}
            self._snapshot(index)
        if self._total_vectors is None:
            self._total_vectors = len(index.vector_to_partition)

        # the shrinkage of each partition since the previous pass, read before anything in this
        # pass changes a size, in the order Quake reads it
        delete_factors = self._delete_factors(index)

        # Quake step 1, every changed partition gets its centroid refreshed before any decision.
        # Quake skips a partition that did not change, and for one that did it shifts the stored
        # centroid by the change instead of averaging the partition again, so the work is the
        # number of changed vectors, not the partition size
        if self.recompute_centroids:
            for pid in list(index.partition_ids()):
                p = index.partitions[pid]
                if p.size() == 0:
                    continue
                # a partition missing from the snapshot was never seen by this policy, so its
                # whole size counts as change
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

        # the snapshot is taken here, at the start of the pass, as Quake resets the change record
        # when it refreshes centroids. the vectors a receiver absorbs from the deletes below then
        # count as growth, so giving them back later is not read as shrinkage
        self._snapshot(index)

        frac = self._access_fraction(index)
        cost_deletes, forced_deletes, splits = self._decide(index, frac, delete_factors, report)
        deleted = self._delete_together(index, cost_deletes + forced_deletes, report)
        report.num_forced_merges += len(deleted & set(forced_deletes))
        self._split_and_refine(index, frac, splits, report)
        # Quake step 7, every partition left with no vectors is deleted at the end of the pass,
        # whatever the cost model says. nothing moves, so no work is charged
        report.num_collected += len(collect_empty_partitions(index))
        self._snapshot_sizes = {pid: s for pid, s in self._snapshot_sizes.items()
                                if pid in index.partitions}
        self._snapshot_sums = {pid: s for pid, s in self._snapshot_sums.items()
                               if pid in index.partitions}
        if self.access_decay is None:
            # the access window closes with the pass, so the next pass reads only the queries
            # that arrive after this one
            for p in index.partitions.values():
                p.access_count = 0.0
            index.queries_routed = 0.0
            index.vectors_scanned_routed = 0.0
        return report

    def _delete_factors(self, index: IVFIndex) -> dict[int, float]:
        """Return the share of its previous size each shrunken partition has lost since the last pass.

        Partitions that did not shrink are left out, as Quake's get_delete_factor returns a
        negative value for them.
        """
        factors: dict[int, float] = {}
        for pid, p in index.partitions.items():
            previous = self._snapshot_sizes.get(pid)
            if previous is None or previous <= 0:
                continue
            lost = previous - p.size()
            if lost > 0:
                factors[pid] = lost / previous
        return factors

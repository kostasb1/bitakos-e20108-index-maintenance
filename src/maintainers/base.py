"""The policy interface, and the restructuring operations every policy shares.

A policy subclasses `Maintainer` and implements `should_maintain` and `maintain`. The stream calls
`start` once, when it takes over the starting index, and `maybe_maintain` after every update.
`maybe_maintain` returns at once unless `check_interval` updates have passed since the last check,
and otherwise asks `should_maintain`, runs `maintain`, and adds the returned `MaintenanceReport` to
the running totals that the result row reads.

The functions below do the restructuring. Each takes the report of the current pass as `meter`
and adds the distances it computes, so two policies that take the same action pay the same price.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from src.config import BALANCED_SPLIT, MAINTENANCE_CHECK_INTERVAL
from src.index import IVFIndex
from src.partition import Partition
from src.types import MaintenanceReport
from src.vector import l2_distance_batch


def drift_limit(
    index: IVFIndex,
    drift_fraction: float,
    drift_threshold: float | None = None,
) -> float:
    """Return the drift threshold as a distance, `drift_fraction` times the data scale.

    A drift threshold is a distance, so a fixed number would mean different things on datasets of
    different scale, and a fraction of the scale behaves the same everywhere. An explicit
    `drift_threshold` overrides the fraction.
    """
    if drift_threshold is not None:
        return float(drift_threshold)
    scale = index.data_scale if index.data_scale > 0.0 else 1.0
    return float(drift_fraction) * scale


class Maintainer(ABC):
    """The base class of every maintenance policy, with the running totals of its work."""

    name: str = "base"

    def __init__(self, check_interval: int = MAINTENANCE_CHECK_INTERVAL) -> None:
        """Create a policy that is offered the chance to act every `check_interval` updates."""
        self.check_interval = check_interval
        self._last_check_step: int = 0
        self._cumulative_cost_seconds: float = 0.0
        self._cumulative_work: int = 0
        # the operation counts behind _cumulative_work, kept so that work can be priced again
        # from a finished result file without running anything. they are reported only
        self._cumulative_splits: int = 0
        self._cumulative_merges: int = 0
        self._cumulative_reassigned: int = 0
        # kept apart from _cumulative_reassigned because the two operations have costs of
        # different order, see MaintenanceReport
        self._cumulative_repartitioned: int = 0
        # empty partitions removed, kept apart from merges, see MaintenanceReport.num_collected
        self._cumulative_collected: int = 0
        self._cumulative_centroids: int = 0
        # splits the exact recomputation declined after the estimate admitted them. it tells a
        # check that rejected candidates apart from one that was never offered any. zero for every
        # policy that does not use predicted cost change
        self._cumulative_declined: int = 0
        # merges declined at the same check, see MaintenanceReport.num_merges_declined
        self._cumulative_merges_declined: int = 0
        # vectors the LIRE conditions admitted for a distance check, see MaintenanceReport
        self._cumulative_examined: int = 0
        # the distance part of _cumulative_work
        self._cumulative_distances: int = 0
        # the k-means parts of work, priced separately by the calibration, see MaintenanceReport
        self._cumulative_kmeans: int = 0
        self._cumulative_bulk_kmeans: int = 0
        # the compaction part of _cumulative_work, see MaintenanceReport
        self._cumulative_compaction: int = 0

    def start(self, index: IVFIndex, step: int) -> None:
        """Note that the stream takes over the index at update `step`.

        The starting index arrives with thousands of updates behind it that are not part of the
        stream, so the check interval counts from here, and a policy whose trigger reads the
        index's lifetime counters takes its baseline here too.
        """
        self._last_check_step = step

    @abstractmethod
    def should_maintain(self, index: IVFIndex, step: int) -> bool:
        """Return True if the policy wants to act at this check."""
        ...

    @abstractmethod
    def maintain(self, index: IVFIndex) -> MaintenanceReport:
        """Run one maintenance pass and return what it did."""
        ...

    def maybe_maintain(self, index: IVFIndex, step: int) -> MaintenanceReport | None:
        """Offer the policy a chance to act, at most once every `check_interval` updates.

        If it acts, its report is added to the running totals and returned, otherwise None.
        """
        if step - self._last_check_step < self.check_interval:
            return None
        self._last_check_step = step

        if not self.should_maintain(index, step):
            return None

        t0 = time.perf_counter()
        report = self.maintain(index)
        report.wall_time_seconds = time.perf_counter() - t0
        report.triggered = True
        self._cumulative_cost_seconds += report.wall_time_seconds
        self._cumulative_work += (report.vectors_processed + report.distance_evaluations
                                  + report.kmeans_distance_evaluations
                                  + report.bulk_kmeans_distance_evaluations)
        self._cumulative_distances += report.distance_evaluations
        self._cumulative_kmeans += report.kmeans_distance_evaluations
        self._cumulative_bulk_kmeans += report.bulk_kmeans_distance_evaluations
        self._cumulative_compaction += report.compaction_reads
        self._cumulative_splits += report.num_splits
        self._cumulative_merges += report.num_merges
        self._cumulative_reassigned += report.num_reassigned
        self._cumulative_repartitioned += report.num_repartitioned
        self._cumulative_collected += report.num_collected
        self._cumulative_centroids += report.num_centroids_recomputed
        self._cumulative_declined += report.num_splits_declined
        self._cumulative_merges_declined += report.num_merges_declined
        self._cumulative_examined += report.num_examined
        return report


def balance_constrained_assignment(
    vectors: np.ndarray, centroids: np.ndarray, max_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster `vectors` with a penalty on large clusters, and return the labels and centroids.

    SPANN equation 1 adds lambda times the squared deviation of each cluster size from the mean to
    the k-means objective. SPFresh relaxes it into an assignment penalty, the distance to a centre
    plus lambda times that cluster's current size, so a cluster that grows too fast becomes more
    expensive to join. The penalty reads the counts of the previous iteration, which makes the
    result independent of the order the vectors are visited in, and lambda is recomputed each
    iteration from the data rather than fixed. Switched off in the reported runs, see
    config.BALANCED_SPLIT.
    """
    k = len(centroids)
    n = len(vectors)
    counts = np.zeros(k, dtype=np.float64)
    centroids = centroids.astype(np.float64, copy=True)
    lam = 0.0
    labels = None

    for _ in range(max_iter):
        diff = vectors[:, None, :] - centroids[None, :, :]
        dist = np.einsum("nkd,nkd->nk", diff, diff)
        cost = dist + lam * counts
        new_labels = cost.argmin(axis=1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        counts = np.bincount(labels, minlength=k).astype(np.float64)

        for j in range(k):
            if counts[j] > 0:
                centroids[j] = vectors[labels == j].mean(axis=0)

        # the spread of assignment cost inside the largest cluster divided by the number of
        # vectors, so lambda takes the scale of the data rather than a fixed constant
        biggest = int(counts.argmax())
        chosen = cost[np.arange(n), labels][labels == biggest]
        if chosen.size:
            lam = max(0.0, float(chosen.max() - chosen.mean()) / max(1, n))

    return labels, centroids.astype(np.float32)


def kmeans_distance_evaluations(km, n_points: int) -> int:
    """Return the distance evaluations a fitted scikit-learn KMeans made.

    Every Lloyd iteration compares every point with every centre, and k-means++ seeding is counted
    as one more iteration. scikit-learn reports the iteration count of the best initialisation
    only, so with several initialisations the others are assumed to have run as long, and the
    seeding tries more candidates than one per centre, so this is a lower bound. Random seeding
    compares nothing and adds no iteration.
    """
    seeding = 1 if km.init == "k-means++" else 0
    return int(km.n_init) * (int(km.n_iter_) + seeding) * int(n_points) * int(km.n_clusters)


def split_partition_2means(
    partition: Partition, random_state: int = 42, balanced: bool = BALANCED_SPLIT,
    meter=None, n_init: int = 3, max_iter: int = 300, init: str = "k-means++",
) -> list[Partition]:
    """Split a partition in two with 2-means, and return the new partitions.

    The children get placeholder ids and the caller puts them into the index. A partition that
    cannot be split, or whose vectors all fall in one cluster, comes back as a single partition.
    The defaults are those of scikit-learn, and the Quake policy passes its own.
    """
    from sklearn.cluster import KMeans

    live_ids, live_vecs = partition.get_live_vectors()
    if len(live_vecs) < 2:
        new_p = Partition(-1, partition.centroid)
        for vid, v in zip(live_ids, live_vecs, strict=True):
            new_p.add(vid, v)
        return [new_p]

    km = KMeans(n_clusters=2, random_state=random_state, n_init=n_init, max_iter=max_iter,
                init=init)
    labels = km.fit_predict(live_vecs)
    centroids = km.cluster_centers_.astype(np.float32)
    if meter is not None:
        meter.kmeans_distance_evaluations += kmeans_distance_evaluations(km, len(live_vecs))

    if balanced:
        # plain 2-means places the centres, and the balanced pass then refines them
        labels, centroids = balance_constrained_assignment(live_vecs, centroids)

    children = [Partition(-1, centroids[0]), Partition(-2, centroids[1])]
    for vid, v, lbl in zip(live_ids, live_vecs, labels, strict=True):
        children[int(lbl)].add(vid, v)

    children = [c for c in children if c.size() > 0]  # every vector in one cluster
    return children


def merge_partitions(partitions: list[Partition], placeholder_id: int = -1) -> Partition:
    """Join partitions into one, centred on the mean of all their live vectors.

    The access counts are added. The result has a placeholder id and the caller puts it into the
    index. The bandit merges this way.
    """
    if not partitions:
        raise ValueError("Cannot merge empty list of partitions")
    if len(partitions) == 1:
        return partitions[0]

    merged = Partition(placeholder_id, partitions[0].centroid.copy())
    for p in partitions:
        live_ids, live_vecs = p.get_live_vectors()
        for vid, v in zip(live_ids, live_vecs, strict=True):
            merged.add(vid, v)
    if merged.size() > 0:
        merged.update_centroid(merged.true_mean())
    merged.access_count = sum(p.access_count for p in partitions)
    return merged


def collect_empty_partitions(index: IVFIndex, keep_at_least: int = 1) -> list[int]:
    """Remove partitions with no live vectors, and return the ids removed.

    An empty partition is still selected by the router, where it returns nothing, so it wastes a
    probe and lowers recall at a fixed nprobe while costing nothing in the scan count. Removing it
    is not a merge, since there is nothing to move, so callers count it in `num_collected` and
    charge no work. `keep_at_least` never lets the index reach zero partitions, which would break
    routing.
    """
    empty = [pid for pid, p in index.partitions.items() if p.is_empty()]
    removable = len(index.partitions) - keep_at_least
    if removable <= 0:
        return []
    collected = empty[:removable]
    for pid in collected:
        # the deletes already removed every id of this partition from vector_to_partition, so
        # removing it cannot strand a live vector
        del index.partitions[pid]
    if collected:
        index._invalidate_cache()
    return collected


def plan_partition_deletion(
    index: IVFIndex, pid: int, meter=None, exclude: set[int] | None = None,
) -> dict[int, list[int]]:
    """Return which partition each vector of `pid` would move to if `pid` were deleted.

    This is the merge of Quake section 4.2.1, "After deletion, the vectors are reassigned to their
    respective nearest existing partitions." Each vector goes to its own nearest centroid, so there
    can be several receivers, and every receiver keeps its own centroid. It differs from the
    SPFresh merge, which appends a whole partition to one chosen partner, see
    `merge_into_survivor`.

    The plan is returned rather than applied, so the caller can compute the exact cost change over
    the real receivers before committing. Quake applies tentatively and rolls back on rejection,
    and evaluating first reaches the same decision without an undo. `exclude` names partitions
    deleted in the same batch, which cannot receive vectors, since Quake removes them all before
    reinserting any vector.
    """
    if pid not in index.partitions or len(index.partitions) < 2:
        return {}
    live_ids, live_vecs = index.partitions[pid].get_live_vectors()
    if not live_ids:
        return {}

    skip = {pid} | (exclude or set())
    others = [q for q in index.partition_ids() if q not in skip]
    if not others:
        return {}
    centroids = np.stack([index.partitions[q].centroid for q in others]).astype(np.float32)
    sq = np.einsum("ij,ij->i", centroids, centroids)
    # one matrix product over all the vectors instead of a lookup per vector, using the same
    # expansion of the squared distance as the router
    nearest = (sq[None, :] - 2.0 * (np.asarray(live_vecs, dtype=np.float32) @ centroids.T)).argmin(1)
    if meter is not None:
        meter.distance_evaluations += len(live_ids) * len(others)

    plan: dict[int, list[int]] = {}
    for vid, j in zip(live_ids, nearest, strict=True):
        plan.setdefault(others[int(j)], []).append(int(vid))
    return plan


def apply_partition_deletion(index: IVFIndex, pid: int, plan: dict[int, list[int]]) -> int:
    """Delete `pid` and move its vectors as `plan` says. Returns the number of vectors moved."""
    partition = index.partitions[pid]
    live_ids, live_vecs = partition.get_live_vectors()
    by_id = dict(zip(live_ids, live_vecs, strict=True))
    moved = 0
    for receiver_pid, vids in plan.items():
        if receiver_pid not in index.partitions:
            continue
        receiver = index.partitions[receiver_pid]
        for vid in vids:
            if vid not in by_id:
                continue
            receiver.add(vid, by_id[vid])
            index.vector_to_partition[vid] = receiver_pid
            moved += 1
    # the access count of the removed partition is not passed on. the queries that scanned it will
    # scan the receivers now holding its vectors and be counted there as they arrive
    del index.partitions[pid]
    index._invalidate_cache()
    return moved


def merge_into_survivor(index: IVFIndex, short_pid: int, long_pid: int) -> list[int]:
    """Merge partition `short_pid` into `long_pid`, as SPFresh section 3.2 does.

    The shorter partition and its centroid are deleted and its vectors appended to the other,
    which keeps its own centroid rather than moving to the combined mean. Returns the ids moved.
    """
    short = index.partitions[short_pid]
    moved_ids, moved_vecs = short.get_live_vectors()
    survivor = index.partitions[long_pid]
    for vid, vec in zip(moved_ids, moved_vecs, strict=True):
        survivor.add(vid, vec)
        index.vector_to_partition[vid] = long_pid
    survivor.access_count += short.access_count
    del index.partitions[short_pid]
    index._invalidate_cache()
    return list(moved_ids)


def reassign_lire_after_split(
    index: IVFIndex,
    old_centroid: np.ndarray,
    new_pids: list[int],
    neighbor_pids: list[int],
    meter=None,
) -> tuple[int, int]:
    """Repair vector placement after a split with the two LIRE conditions of SPFresh section 3.3.

    The conditions decide which vectors are worth checking at all, which is what LIRE saves over
    checking every vector in the neighbourhood. A checked vector moves to the nearest local
    centroid if that is strictly closer than its own. Returns the number moved and the number
    checked.
    """
    local_pids = [pid for pid in dict.fromkeys([*new_pids, *neighbor_pids])
                  if pid in index.partitions]
    if len(local_pids) < 2:
        return 0, 0
    centroids = np.vstack([index.partitions[pid].centroid for pid in local_pids])
    pos = {pid: i for i, pid in enumerate(local_pids)}
    new_rows = [pos[pid] for pid in new_pids if pid in pos]
    if not new_rows:
        return 0, 0

    moves = examined = 0
    for src_pid in local_pids:
        if src_pid not in index.partitions:
            continue
        src = index.partitions[src_pid]
        live_ids, live_vecs = src.get_live_vectors()
        if len(live_vecs) == 0:
            continue
        from_split = src_pid in set(new_pids)
        d_old = np.linalg.norm(live_vecs - old_centroid, axis=1)
        d_new = np.linalg.norm(
            live_vecs[:, None, :] - centroids[new_rows][None, :, :], axis=2).min(axis=1)
        # condition 1, for a vector of a new partition, the deleted centroid was at least as close
        # as every new one. condition 2, for a vector of a neighbour, some new centroid is at least
        # as close as the deleted one was. d_new is the minimum over the new centroids, which
        # turns both conditions into the same comparison
        candidates = d_old <= d_new if from_split else d_new <= d_old
        if meter is not None:
            # the conditions cost one distance to the deleted centroid and one to each new one, for
            # every vector, and each admitted vector then pays one per local centroid
            meter.distance_evaluations += len(live_vecs) * (1 + len(new_rows))
            meter.distance_evaluations += int(candidates.sum()) * len(local_pids)
        src_row = pos[src_pid]
        for idx in np.flatnonzero(candidates):
            examined += 1
            v = live_vecs[idx]
            dists = np.linalg.norm(centroids - v, axis=1)
            best = int(np.argmin(dists))
            if local_pids[best] != src_pid and dists[best] < dists[src_row]:
                index.apply_reassign(int(live_ids[idx]), src_pid, local_pids[best])
                moves += 1
    return moves, examined


def reassign_lire_after_merge(
    index: IVFIndex,
    merged_pid: int,
    moved_ids: list[int],
    neighbor_pids: list[int],
    old_centroid: np.ndarray,
    meter=None,
) -> int:
    """Repair vector placement after a LIRE merge. Returns the number of vectors moved.

    Only the vectors of the deleted partition can be misplaced by a merge, SPFresh section 3.3.
    Following the authors' implementation, a moved vector is reconsidered only when the surviving
    centroid is farther from it than its deleted centroid `old_centroid` was, and it then moves to
    the nearest local centroid if that is strictly closer.
    """
    if merged_pid not in index.partitions or not moved_ids:
        return 0
    local_pids = [pid for pid in dict.fromkeys([merged_pid, *neighbor_pids])
                  if pid in index.partitions]
    if len(local_pids) < 2:
        return 0
    centroids = np.vstack([index.partitions[pid].centroid for pid in local_pids])
    src_row = 0
    moves = 0
    merged = index.partitions[merged_pid]
    for vid in moved_ids:
        if index.vector_to_partition.get(vid) != merged_pid:
            continue
        if merged.is_deleted(vid):
            continue
        v = merged.get_vector(vid)
        if meter is not None:
            meter.distance_evaluations += 2
        if not (np.linalg.norm(v - centroids[src_row])
                > np.linalg.norm(v - np.asarray(old_centroid, dtype=np.float32))):
            continue
        dists = np.linalg.norm(centroids - v, axis=1)
        if meter is not None:
            meter.distance_evaluations += len(local_pids)
        best = int(np.argmin(dists))
        if local_pids[best] != merged_pid and dists[best] < dists[src_row]:
            index.apply_reassign(vid, merged_pid, local_pids[best])
            moves += 1
    return moves


def top_k_nearest_partitions(
    index: IVFIndex,
    target_centroid: np.ndarray,
    k: int,
    exclude: set[int] | None = None,
    meter=None,
) -> list[int]:
    """Return the ids of the `k` partitions with centroids nearest to `target_centroid`, in no order."""
    if exclude is None:
        exclude = set()
    candidate_pids = [pid for pid in index.partition_ids() if pid not in exclude]
    if not candidate_pids:
        return []

    centroids = np.vstack([index.partitions[pid].centroid for pid in candidate_pids])
    dists = l2_distance_batch(target_centroid, centroids)
    if meter is not None:
        meter.distance_evaluations += len(candidate_pids)

    k = min(k, len(candidate_pids))
    nearest_idx = (
        np.argpartition(dists, k - 1)[:k] if k < len(candidate_pids) else np.arange(k)
    )
    return [candidate_pids[i] for i in nearest_idx]


def nearest_partitions_in_order(
    index: IVFIndex,
    target_centroid: np.ndarray,
    k: int,
    exclude: set[int] | None = None,
    meter=None,
) -> list[int]:
    """Return the ids of the `k` partitions with centroids nearest to `target_centroid`, nearest first.

    The SPFresh merge scans the nearest partitions in order and takes the first whose combined
    length fits, so the order matters here, unlike in `top_k_nearest_partitions`.
    """
    if exclude is None:
        exclude = set()
    candidate_pids = [pid for pid in index.partition_ids() if pid not in exclude]
    if not candidate_pids:
        return []

    centroids = np.vstack([index.partitions[pid].centroid for pid in candidate_pids])
    dists = l2_distance_batch(target_centroid, centroids)
    if meter is not None:
        meter.distance_evaluations += len(candidate_pids)

    k = min(k, len(candidate_pids))
    nearest_idx = (
        np.argpartition(dists, k - 1)[:k] if k < len(candidate_pids)
        else np.arange(len(candidate_pids))
    )
    nearest_idx = nearest_idx[np.argsort(dists[nearest_idx])]
    return [candidate_pids[int(i)] for i in nearest_idx]


def reassign_boundary(
    index: IVFIndex,
    target_pids: list[int],
    candidate_source_pids: list[int],
    meter=None,
) -> int:
    """Move every vector of the given partitions to its nearest centroid among them.

    A vector moves only when another centroid in the set is strictly closer than its own, in
    either direction between the partitions. Running it twice changes nothing more. Returns the
    number of vectors moved.
    """
    local_pids = [pid for pid in dict.fromkeys([*target_pids, *candidate_source_pids])
                  if pid in index.partitions]
    if len(local_pids) < 2:
        return 0

    centroids = np.vstack([index.partitions[pid].centroid for pid in local_pids])
    pos = {pid: i for i, pid in enumerate(local_pids)}

    moves = 0
    for src_pid in local_pids:
        if src_pid not in index.partitions:
            continue
        src = index.partitions[src_pid]
        live_ids, live_vecs = src.get_live_vectors()
        if len(live_vecs) == 0:
            continue
        if meter is not None:
            meter.distance_evaluations += len(live_vecs) * len(local_pids)
        src_row = pos[src_pid]
        for vid, v in zip(live_ids, live_vecs, strict=True):
            dists = np.linalg.norm(centroids - v, axis=1)
            best = int(np.argmin(dists))
            if local_pids[best] != src_pid and dists[best] < dists[src_row]:
                index.apply_reassign(vid, src_pid, local_pids[best])
                moves += 1
    return moves


def recompute_all_centroids(index: IVFIndex, drift_threshold: float = 0.0) -> int:
    """Move every centroid whose drift exceeds `drift_threshold` to its partition mean.

    Returns the number of centroids moved.
    """
    updated = 0
    for pid in list(index.partition_ids()):
        p = index.partitions[pid]
        if p.size() == 0:
            continue
        if p.centroid_drift() > drift_threshold:
            index.recompute_centroid(pid)
            updated += 1
    return updated

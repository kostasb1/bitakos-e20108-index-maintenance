"""The reward of the bandit policy, built on the total Quake cost C.

`index_cost` is C, the sum over partitions of min(1, A) times lambda(size), plus lambda_c(K). The
bandit reads C before and after an action, and `compute_reward` turns the change into a reward in
[-1, 1], after subtracting a weighted action cost and adding two small terms, one for drift a
centroid refresh closed and one for tombstones an action removed.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.partition import Partition


@dataclass
class PartitionSnapshot:
    """The state of a partition just before an action, kept to compare with the state after it."""

    size: int
    access: float
    drift: float
    tombstone_count: int
    total_size: int
    cost: float = 0.0


def partition_latency_cost(partition: Partition, cost_model=None) -> float:
    """Return what a partition costs the workload, access count times lambda(size).

    This is Quake equation 1. With no cost model, lambda is the live size.
    """
    lam = cost_model.scan_latency(partition.size()) if cost_model else float(partition.size())
    return partition.access_count * lam


def index_cost(index, cost_model=None, nprobe: int = 10) -> float:
    """Return the total Quake cost C, the sum of A times lambda(size) plus lambda_c(K).

    A is the share of routed queries that scanned the partition, so the partition term and the
    centroid term are both per query and can be added. The centroid term is the delta O of Quake
    equations 4 and 5, since every query compares itself with all K centroids. With no cost model,
    lambda is the vector count.

    The bandit scores an action by the change in this total. Scoring only the acted partition
    would miss two effects. The reassignment after a split moves vectors and their access into the
    children from neighbouring partitions, so the children would be charged for what they gained
    while the neighbours were never credited for what they lost. And without the centroid term a
    merge could never lower the cost, and nothing would oppose a split.
    """
    lam = cost_model.scan_latency if cost_model else float
    lam_c = cost_model.centroid_latency if cost_model else float
    queries = index.query_count(nprobe)
    scan = 0.0
    if queries > 0:
        for p in index.partitions.values():
            scan += min(1.0, p.access_count / queries) * lam(p.size())
    return scan + lam_c(len(index.partitions))


def snapshot(partition: Partition, cost_model=None) -> PartitionSnapshot:
    """Record the current state of a partition."""
    return PartitionSnapshot(
        size=partition.size(),
        access=partition.access_count,
        drift=partition.centroid_drift(),
        tombstone_count=len(partition.tombstones),
        total_size=partition.total_size(),
        cost=partition_latency_cost(partition, cost_model),
    )


def combine_snapshots(snaps: list[PartitionSnapshot]) -> PartitionSnapshot:
    """Combine the snapshots of the partitions a merge consumes into one.

    Costs are added, which is the true combined cost whatever the shape of lambda. Drift and access
    are averaged, weighted by size.
    """
    total_live = sum(s.size for s in snaps)
    if total_live > 0:
        drift = sum(s.drift * s.size for s in snaps) / total_live
        access = sum(s.access * s.size for s in snaps) / total_live
    else:
        drift = 0.0
        access = sum(s.access for s in snaps)
    return PartitionSnapshot(
        size=total_live,
        access=access,
        drift=drift,
        tombstone_count=sum(s.tombstone_count for s in snaps),
        total_size=sum(s.total_size for s in snaps),
        cost=sum(s.cost for s in snaps),
    )


def compute_reward(
    before: PartitionSnapshot,
    after_partitions: list[Partition],
    cost_model=None,
    action_cost: float = 0.0,
    drift_scale: float = 0.1,
    w_cost: float = 0.1,
    w_drift_guard: float = 0.1,
    w_compact: float = 0.1,
    *,
    delta_index_cost: float | None = None,
    cost_scale: float | None = None,
    clip: float | None = None,
    drift_credit: bool = True,
) -> float:
    """Return the reward of an action.

    The reward is the cost reduction, minus `w_cost` times the action cost, plus `w_drift_guard`
    times the drift closed over `drift_scale`, plus `w_compact` times the drop in the tombstone
    share. With `delta_index_cost`, which the bandit passes, the cost reduction is the change in
    `index_cost` across the action divided by `cost_scale`. Otherwise it is the relative change in
    the acted partitions' own cost. The drift is divided by a fixed scale rather than by its value
    before the action, so closing a small drift does not score the same as closing a large one.
    `clip` bounds the reward, which the analysis of LinUCB assumes.
    """
    if not after_partitions:
        return 0.0

    n = len(after_partitions)

    if delta_index_cost is not None:
        cost_reduction = -delta_index_cost / max(cost_scale or 0.0, 1e-9)
    else:
        cost_before = before.cost
        cost_after = sum(partition_latency_cost(p, cost_model) for p in after_partitions)
        cost_reduction = (cost_before - cost_after) / max(cost_before, 1e-6)

    # drift is the distance from a centroid to the mean of its own members, so the before and
    # after values are comparable only while the members stay the same. a split or a merge changes
    # them, and crediting the difference there made merges look profitable while they raised the
    # cost, so the caller turns this term off for those actions
    if drift_credit:
        drift_after = sum(p.centroid_drift() for p in after_partitions) / n
        drift_guard = (before.drift - drift_after) / max(drift_scale, 1e-6)
    else:
        drift_guard = 0.0

    frac_before = before.tombstone_count / max(before.total_size, 1)
    frac_after = sum(len(p.tombstones) for p in after_partitions) / max(
        sum(p.total_size() for p in after_partitions), 1
    )
    delta_compact = frac_before - frac_after

    reward = (
        cost_reduction
        - w_cost * action_cost
        + w_drift_guard * drift_guard
        + w_compact * delta_compact
    )
    if clip is not None:
        reward = max(-clip, min(clip, reward))
    return reward

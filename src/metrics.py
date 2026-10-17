"""The scoring, how well maintenance worked, never how a policy decides."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from src.calibration import has_price_list, list_price
from src.config import MAX_PARTITION_SIZE
from src.index import IVFIndex
from src.partition import Partition
from src.types import CostModel, QualityReport


def recall_at_k(
    index: IVFIndex,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    nprobe: int,
) -> float:
    """Return the mean recall at `k` over `queries` at a fixed `nprobe`."""
    per_query = recall_at_k_per_query(index, queries, ground_truth, k, nprobe)
    return float(per_query.mean())


def compute_live_ground_truth(index: IVFIndex, queries: np.ndarray, k: int) -> np.ndarray:
    """Return the ids of the exact `k` nearest live vectors of every query."""
    # exact ties broken by stacking order
    all_ids: list[int] = []
    blocks: list[np.ndarray] = []
    for p in index.partitions.values():
        live_ids, live_vecs = p.get_live_vectors()
        if live_ids:
            all_ids.extend(live_ids)
            blocks.append(live_vecs)

    n_queries = len(queries)
    if not all_ids:
        return np.empty((n_queries, 0), dtype=np.int64)

    matrix = np.vstack(blocks).astype(np.float32)
    id_arr = np.asarray(all_ids, dtype=np.int64)
    k_eff = min(k, len(all_ids))

    from src.dataset import compute_ground_truth

    positions = compute_ground_truth(matrix, queries.astype(np.float32), k=k_eff)
    return id_arr[positions]


def recall_at_k_live(
    index: IVFIndex,
    queries: np.ndarray,
    k: int,
    nprobe: int,
) -> float:
    """Mean recall at a fixed nprobe."""
    live_gt = compute_live_ground_truth(index, queries, k)
    if live_gt.shape[1] == 0:
        return 0.0
    per_query = recall_at_k_per_query(index, queries, live_gt, k, nprobe)
    return float(per_query.mean())


def recall_at_k_per_query(
    index: IVFIndex,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    nprobe: int,
) -> np.ndarray:
    """Recall at `k` of each query."""
    results = np.zeros(len(queries))

    for i, q in enumerate(queries):
        # do not record access
        retrieved_pairs = index.search(q, k, nprobe, record_access=False)
        retrieved = {vid for vid, _ in retrieved_pairs}
        truth = set(ground_truth[i, :k].tolist())

        if not truth:
            results[i] = 1.0
            continue

        results[i] = len(retrieved & truth) / len(truth)

    return results


def query_scan_cost(
    index: IVFIndex, queries: np.ndarray, nprobe: int, include_centroids: bool = False,
    stored: bool = False,
) -> float:
    """Return the mean number of vectors a query scans at `nprobe`."""
    if not index.partitions:
        return 0.0
    sizes = {pid: (p.total_size() if stored else p.size())
             for pid, p in index.partitions.items()}
    total = 0
    for q in queries:
        for pid in index._nearest_partitions(q, nprobe):
            total += sizes[pid]
    scanned = total / len(queries)
    return scanned + len(index.partitions) if include_centroids else scanned


def priced_scan_cost(index: IVFIndex, queries: np.ndarray, nprobe: int,
                     stored: bool = False) -> float:
    """Return the mean Faiss price of the lists a query probes at `nprobe`."""
    if not index.partitions:
        return 0.0
    pids = list(index.partitions)
    lengths = np.array([index.partitions[pid].total_size() if stored
                        else index.partitions[pid].size() for pid in pids])
    price = dict(zip(pids, list_price(lengths, index.dim), strict=True))
    total = 0.0
    for q in queries:
        for pid in index._nearest_partitions(q, nprobe):
            total += price[pid]
    return total / len(queries)


def _read_between(read: Callable[[int], float], nprobe: float, n_parts: int,
                  nprobe_grid: tuple[int, ...] | None = None) -> float:
    """Interpolate a reading between two nprobe values."""
    if float(nprobe).is_integer() or nprobe <= 1 or nprobe >= n_parts:
        return read(int(round(nprobe)))
    if nprobe_grid is None:
        lo = int(np.floor(nprobe))
        hi = lo + 1
    else:
        points = sorted({min(g, n_parts) for g in nprobe_grid})
        if nprobe <= points[0] or nprobe >= points[-1]:
            return read(int(round(nprobe)))
        lo = max(g for g in points if g <= nprobe)
        hi = min(g for g in points if g >= nprobe)
    c_lo, c_hi = read(lo), read(hi)
    return c_lo + (nprobe - lo) / (hi - lo) * (c_hi - c_lo)


def priced_cost_at(index: IVFIndex, queries: np.ndarray, nprobe: float, stored: bool = False,
                   nprobe_grid: tuple[int, ...] | None = None) -> float:
    """Return the priced scan at the nprobe that met the recall target."""
    return _read_between(lambda n: priced_scan_cost(index, queries, n, stored), nprobe,
                         len(index.partitions), nprobe_grid)


def stored_cost_at(index: IVFIndex, queries: np.ndarray, nprobe: float,
                   nprobe_grid: tuple[int, ...] | None = None) -> float:
    """Scan count at the target recall, with tombstones scanned."""
    return _read_between(lambda n: query_scan_cost(index, queries, n, stored=True), nprobe,
                         len(index.partitions), nprobe_grid)


def cost_at_target_recall(
    index: IVFIndex,
    queries: np.ndarray,
    k: int,
    target_recall: float = 0.9,
    nprobe_grid: tuple[int, ...] | None = None,
) -> tuple[float, float, float, float]:
    """Return the scan cost at the nprobe where mean recall reaches `target_recall`."""
    if nprobe_grid is None:
        return costs_at_target_recalls(index, queries, k, (target_recall,))[target_recall]
    live_gt = compute_live_ground_truth(index, queries, k)
    if live_gt.shape[1] == 0:
        return 0.0, 0, 0.0, 0.0
    n_parts = len(index.partitions)
    prev_npr: int | None = None
    prev_rec = 0.0
    rec = 0.0
    std = 0.0
    for npr in nprobe_grid:
        if npr > n_parts:
            npr = n_parts
        per_query = recall_at_k_per_query(index, queries, live_gt, k, npr)
        rec, std = float(per_query.mean()), float(per_query.std())
        if rec >= target_recall:
            if prev_npr is None:
                return query_scan_cost(index, queries, npr, include_centroids=False), npr, rec, std
            frac = (target_recall - prev_rec) / (rec - prev_rec)
            cost_lo = query_scan_cost(index, queries, prev_npr, include_centroids=False)
            cost_hi = query_scan_cost(index, queries, npr, include_centroids=False)
            interp_cost = cost_lo + frac * (cost_hi - cost_lo)
            interp_npr = prev_npr + frac * (npr - prev_npr)
            return interp_cost, interp_npr, target_recall, std
        if npr >= n_parts:
            return query_scan_cost(index, queries, npr, include_centroids=False), npr, rec, std
        prev_npr, prev_rec = npr, rec
    return (query_scan_cost(index, queries, nprobe_grid[-1], include_centroids=False),
            nprobe_grid[-1], rec, std)


def costs_at_target_recalls(
    index: IVFIndex, queries: np.ndarray, k: int, targets: tuple[float, ...],
) -> dict[float, tuple[float, float, float, float]]:
    """Return `cost_at_target_recall` for several targets, one result per target."""
    live_gt = compute_live_ground_truth(index, queries, k)
    if live_gt.shape[1] == 0:
        return {t: (0.0, 0, 0.0, 0.0) for t in targets}
    n_parts = len(index.partitions)
    measured: dict[int, np.ndarray] = {}
    return {t: _cost_at_target_every_integer(index, queries, live_gt, k, t, n_parts, measured)
            for t in targets}


def cost_rebuilt_at_own_k(
    index: IVFIndex, queries: np.ndarray, k: int, target_recall: float = 0.9, seed: int = 0,
) -> dict:
    """Query cost of a fresh k-means at the same K."""
    ids: list[int] = []
    blocks: list[np.ndarray] = []
    for p in index.partitions.values():
        live_ids, live_vecs = p.get_live_vectors()
        if live_ids:
            ids.extend(live_ids)
            blocks.append(live_vecs)
    fresh = IVFIndex(dim=index.dim)
    fresh.build(np.vstack(blocks).astype(np.float32), ids, n_partitions=len(index.partitions),
                seed=seed)
    cost, npr, rec, _ = cost_at_target_recall(fresh, queries, k, target_recall)
    r = dict(cost=cost, nprobe=npr, achieved=rec, parts=len(fresh.partitions),
             live=len(fresh.vector_to_partition))
    if has_price_list(index.dim):
        r["cost_priced"] = priced_cost_at(fresh, queries, npr)
    return r


def _cost_at_target_every_integer(
    index: IVFIndex, queries: np.ndarray, live_gt: np.ndarray, k: int, target_recall: float,
    n_parts: int, measured: dict[int, np.ndarray] | None = None,
) -> tuple[float, float, float, float]:
    """The doubling and bisection search of `cost_at_target_recall`."""
    if measured is None:
        measured = {}

    def rec(n: int) -> float:
        """Return the mean recall at nprobe `n`, measuring it only once."""
        if n not in measured:
            measured[n] = recall_at_k_per_query(index, queries, live_gt, k, n)
        return float(measured[n].mean())

    lo, hi = 0, 1
    while rec(hi) < target_recall:
        if hi >= n_parts:
            # target not reachable, report the last point
            return (query_scan_cost(index, queries, hi, include_centroids=False), hi, rec(hi),
                    float(measured[hi].std()))
        lo, hi = hi, min(2 * hi, n_parts)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rec(mid) >= target_recall:
            hi = mid
        else:
            lo = mid
    std = float(measured[hi].std())
    if lo == 0:
        # met at nprobe 1
        return query_scan_cost(index, queries, 1, include_centroids=False), 1, rec(1), std
    frac = (target_recall - rec(lo)) / (rec(hi) - rec(lo))
    cost_lo = query_scan_cost(index, queries, lo, include_centroids=False)
    cost_hi = query_scan_cost(index, queries, hi, include_centroids=False)
    return cost_lo + frac * (cost_hi - cost_lo), lo + frac, target_recall, std


def adaptive_query_cost(
    index: IVFIndex,
    queries: np.ndarray,
    k: int,
    target_recall: float = 0.9,
    radius_factor: float = 1.0,
) -> tuple[float, float, float]:
    """Return the cost at the target recall using a per query adaptive nprobe."""
    live_gt = compute_live_ground_truth(index, queries, k)
    if live_gt.shape[1] == 0:
        return 0.0, 0.0, 0.0
    total_scanned = 0
    total_probed = 0
    recalls = np.empty(len(queries))
    for i, q in enumerate(queries):
        res, n_probed, scanned = index.search_adaptive(
            q, k, target_recall=target_recall, radius_factor=radius_factor)
        total_scanned += scanned
        total_probed += n_probed
        retrieved = {vid for vid, _ in res}
        truth = set(live_gt[i, :k].tolist())
        recalls[i] = len(retrieved & truth) / len(truth) if truth else 1.0
    n = len(queries)
    return total_scanned / n, float(recalls.mean()), total_probed / n


def amortized_cost(partition: Partition) -> float:
    """Access count times live size. Not used in the reported runs."""
    return partition.access_count * partition.size()


def partition_quality(partition: Partition, cost_model: CostModel | None = None) -> float:
    """Quality score of a partition, lower is better. Not used in the reported runs."""
    if cost_model is not None:
        return cost_model.predict_latency(
            size=partition.size(),
            drift=partition.centroid_drift(),
            access=partition.access_count,
        )

    alpha = 1.0 / MAX_PARTITION_SIZE
    beta = 1.0
    gamma = 0.001

    return (
        alpha * partition.size()
        + beta * (partition.centroid_drift() ** 2)
        + gamma * partition.access_count
    )


def index_quality(index: IVFIndex, cost_model: CostModel | None = None) -> QualityReport:
    """Quality score over all partitions, and the worst one. Not used in the reported runs."""
    stats_list = []
    qualities = []
    worst_id = -1
    worst_q = -float("inf")

    for pid, partition in index.partitions.items():
        q = partition_quality(partition, cost_model)
        qualities.append(q)

        if q > worst_q:
            worst_q = q
            worst_id = pid

        stats_list.append(partition.stats(quality=q))

    qs = np.array(qualities) if qualities else np.array([0.0])

    return QualityReport(
        mean_quality=float(qs.mean()),
        std_quality=float(qs.std()),
        worst_partition_id=worst_id,
        worst_quality=float(worst_q) if worst_id >= 0 else 0.0,
        per_partition=stats_list,
    )


def size_imbalance(index: IVFIndex) -> float:
    """Return the largest partition size over the mean, 1.0 when perfectly balanced."""
    stats = index.stats()
    if stats.mean_partition_size == 0:
        return 1.0
    return stats.max_partition_size / stats.mean_partition_size


def reconstruction_error(index: IVFIndex, sample_size: int = 10_000, seed: int = 0) -> float:
    """Return the mean squared distance from sampled vectors to their centroid."""
    errors: list[float] = []
    partitions = list(index.partitions.values())
    if not partitions:
        return 0.0

    total_live = sum(p.size() for p in partitions)
    if total_live == 0:
        return 0.0

    frac = min(1.0, sample_size / total_live)
    rng = np.random.default_rng(seed)
    for p in partitions:
        if p.size() == 0:
            continue

        _, live_vecs = p.get_live_vectors()

        take = min(len(live_vecs), max(1, round(frac * len(live_vecs))))
        sample_idx = rng.choice(len(live_vecs), size=take, replace=False)

        diffs = live_vecs[sample_idx] - p.centroid
        errors.extend(np.einsum("ij,ij->i", diffs, diffs).tolist())

    return float(np.mean(errors)) if errors else 0.0


def query_latency_percentiles(
    index: IVFIndex,
    queries: np.ndarray,
    nprobe: int,
    k: int,
) -> dict[str, float]:
    """Search time percentiles in ms. Not used in the reported runs."""
    times_ms: list[float] = []

    for q in queries:
        t0 = time.perf_counter_ns()
        _ = index.search(q, k, nprobe, record_access=False)
        elapsed_ns = time.perf_counter_ns() - t0
        times_ms.append(elapsed_ns / 1e6)

    arr = np.array(times_ms)
    return {
        "mean": float(arr.mean()),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "p99":  float(np.percentile(arr, 99)),
    }

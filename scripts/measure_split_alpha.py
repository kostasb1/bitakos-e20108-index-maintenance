"""Measure alpha, the share of a parent's access each child of a split receives, Quake equation 6.

    python scripts/measure_split_alpha.py --dataset sift1m --seed 42

Quake estimates the cost change of splitting a partition as

    delta Split = delta O+ - A * lambda(s) + 2 * alpha * A * lambda(s / 2)

where alpha is the share of the parent's access each of the two children is expected to get.
Quake publishes alpha = 0.9. Here alpha is not left as an assumption, it is measured. A query
probes the nprobe nearest centroids, so a partition can be split and the same queries routed again,
counting how much of the parent's access the children actually receive.

    alpha = (A_left + A_right) / (2 * A_parent)

Routing depends only on centroids, so the reassignment after a split, which moves vectors without
moving centroids, cannot change this measurement. It is applied anyway so the split is the full
operation. The measurement uses an index built with k-means on a random fifth of the dataset,
before the starting index of the main runs existed, and it gave 0.862, 0.862 and 0.824 on SIFT and
0.880 on GIST, pooled as 0.86 in config.SPLIT_ACCESS_ALPHA.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import copy  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DEFAULT_NPROBE, DEFAULT_PARTITIONS, RESULTS_DIR  # noqa: E402
from src.dataset import (  # noqa: E402
    load_gist1m,
    load_sift1m,
    split_base_for_streaming,
)
from src.index import IVFIndex  # noqa: E402
from src.maintainers.base import (  # noqa: E402
    reassign_boundary,
    split_partition_2means,
    top_k_nearest_partitions,
)


def route_all(index: IVFIndex, queries: np.ndarray, nprobe: int) -> list[np.ndarray]:
    """Return the probed partition ids of every query, computed for all queries at once.

    Only the set of probed partitions matters here, so the order within each set may differ from
    `IVFIndex._nearest_partitions`.
    """
    matrix, order = index._get_centroid_matrix()
    order_arr = np.asarray(order)

    # squared distance by expansion. the query norm is the same for every centroid of a row, so it
    # does not affect the ranking
    cent_sq = np.einsum("ij,ij->i", matrix, matrix)
    cross = queries @ matrix.T
    sq_dists = cent_sq[None, :] - 2.0 * cross

    n_parts = len(order)
    nprobe_actual = min(nprobe, n_parts)
    if nprobe_actual < n_parts:
        top = np.argpartition(sq_dists, nprobe_actual - 1, axis=1)[:, :nprobe_actual]
    else:
        top = np.tile(np.arange(n_parts), (len(queries), 1))
    return [order_arr[row] for row in top]


def verify_router(index: IVFIndex, queries: np.ndarray, nprobe: int, n_check: int = 25) -> None:
    """Check that `route_all` gives the same probe sets as the index itself."""
    probed = route_all(index, queries[:n_check], nprobe)
    for q, got in zip(queries[:n_check], probed, strict=True):
        want = set(index._nearest_partitions(q, nprobe))
        assert set(got.tolist()) == want, "batched router disagrees with IVFIndex"


def access_counts(probe_sets: list[np.ndarray], n_queries: int) -> dict[int, float]:
    """Return the share of queries that probe each partition, A in Quake equation 1."""
    counts: dict[int, int] = {}
    for row in probe_sets:
        for pid in row.tolist():
            counts[pid] = counts.get(pid, 0) + 1
    return {pid: c / n_queries for pid, c in counts.items()}


def build_query_pool(
    queries: np.ndarray, mode: str, seed: int, n_queries: int,
    hot_fraction: float, hot_prob: float,
) -> np.ndarray:
    """Return the query pool, concentrated near one query or uniform as a robustness check."""
    qrng = np.random.default_rng(seed + 7)
    if mode == "uniform":
        return queries[qrng.integers(0, len(queries), n_queries)]

    # concentrated, `hot_prob` of the queries from the `hot_fraction` nearest to the first query
    n_hot = max(1, int(len(queries) * hot_fraction))
    d0 = np.linalg.norm(queries - queries[0], axis=1)
    hot_q_idx = np.argsort(d0)[:n_hot]
    fh = qrng.random(n_queries) < hot_prob
    picks = np.where(
        fh,
        hot_q_idx[qrng.integers(0, n_hot, n_queries)],
        qrng.integers(0, len(queries), n_queries),
    )
    return queries[picks]


def pick_candidates(index: IVFIndex, n_candidates: int, min_size: int) -> list[int]:
    """Return partitions spread across the size range, so alpha is not read off one size alone."""
    sized = sorted(
        ((pid, p.size()) for pid, p in index.partitions.items() if p.size() >= min_size),
        key=lambda t: t[1],
    )
    if not sized:
        return []
    if len(sized) <= n_candidates:
        return [pid for pid, _ in sized]
    idx = np.linspace(0, len(sized) - 1, n_candidates).round().astype(int)
    return [sized[int(i)][0] for i in dict.fromkeys(idx.tolist())]


def measure_one(
    index: IVFIndex, pid: int, queries: np.ndarray, nprobe: int,
    before_probes: list[np.ndarray], seed: int, boundary_top_k: int,
) -> dict | None:
    """Split one partition on a copy of the index and measure the access its children get.

    Returns None for a partition no query probed, or one whose split put everything on one side.
    """
    n_queries = len(queries)
    a_before = access_counts(before_probes, n_queries)
    parent_access = a_before.get(pid, 0.0)
    if parent_access <= 0.0:
        return None  # never probed, so alpha is undefined for it

    work = copy.deepcopy(index)
    parent_size = work.partitions[pid].size()

    children = split_partition_2means(work.partitions[pid], random_state=seed)
    if len(children) < 2:
        return None  # 2-means put everything in one cluster

    neighbors = top_k_nearest_partitions(
        work, work.partitions[pid].centroid, boundary_top_k, exclude={pid}
    )
    work.apply_split(pid, children)
    new_pids = sorted(work.partition_ids())[-len(children):]
    reassign_boundary(work, new_pids, neighbors)

    after_probes = route_all(work, queries, nprobe)
    a_after = access_counts(after_probes, n_queries)
    child_access = sum(a_after.get(c, 0.0) for c in new_pids)

    # of the queries that probed the parent, how many now probe both children, one, or neither
    child_set = set(new_pids)
    hits = [len(child_set & set(after.tolist()))
            for before, after in zip(before_probes, after_probes, strict=True)
            if pid in before.tolist()]
    n_par = len(hits)

    return {
        "parent_pid": pid,
        "parent_size": parent_size,
        "child_sizes": "|".join(str(work.partitions[c].size()) for c in new_pids),
        "parent_access": parent_access,
        "child_access_total": child_access,
        "alpha": child_access / (2.0 * parent_access),
        "queries_probing_parent": n_par,
        "frac_probing_both": sum(1 for h in hits if h == 2) / n_par if n_par else 0.0,
        "frac_probing_one": sum(1 for h in hits if h == 1) / n_par if n_par else 0.0,
        "frac_probing_neither": sum(1 for h in hits if h == 0) / n_par if n_par else 0.0,
    }


def main() -> None:
    """Build the index, split a range of partitions one at a time, and report alpha."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="sift1m", choices=["sift1m", "gist1m"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--initial-fraction", type=float, default=0.2)
    p.add_argument("--n-queries", type=int, default=2000)
    p.add_argument("--n-candidates", type=int, default=30)
    p.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE)
    p.add_argument("--boundary-top-k", type=int, default=25)
    p.add_argument("--query-mode", default="hot_cold", choices=["hot_cold", "uniform"])
    p.add_argument("--hot-fraction", type=float, default=0.1)
    p.add_argument("--hot-prob", type=float, default=0.8)
    p.add_argument("--min-candidate-size", type=int, default=2)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "split_alpha.csv")
    args = p.parse_args()

    base, queries, _ = load_sift1m() if args.dataset == "sift1m" else load_gist1m()

    # a k-means index on a random share of the dataset, about 100 vectors per partition
    iv, ii, _, _ = split_base_for_streaming(
        base, initial_fraction=args.initial_fraction, seed=args.seed)
    n_part = min(DEFAULT_PARTITIONS, max(2, len(iv) // 100))
    index = IVFIndex(dim=base.shape[1])
    index.build(iv, ii, n_partitions=n_part, seed=args.seed)
    avg = max(1, len(iv) // n_part)
    print(f"index {n_part} partitions, {len(iv)} vectors, mean size {avg}")

    qpool = build_query_pool(
        queries, args.query_mode, args.seed, args.n_queries,
        args.hot_fraction, args.hot_prob)
    verify_router(index, qpool, args.nprobe)

    before_probes = route_all(index, qpool, args.nprobe)
    candidates = pick_candidates(index, args.n_candidates, args.min_candidate_size)
    print(f"measuring {len(candidates)} partitions, query mode {args.query_mode}")

    rows = []
    for n, pid in enumerate(candidates, 1):
        row = measure_one(
            index, pid, qpool, args.nprobe, before_probes,
            args.seed, args.boundary_top_k)
        if row is None:
            continue
        row["dataset"] = args.dataset
        row["seed"] = args.seed
        row["query_mode"] = args.query_mode
        row["nprobe"] = args.nprobe
        row["mean_partition_size"] = avg
        rows.append(row)
        print(f"  [{n}/{len(candidates)}] pid {pid} size {row['parent_size']} "
              f"alpha {row['alpha']:.3f}")

    if not rows:
        print("no measurable partitions")
        return

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    a = df["alpha"]
    print(f"\nalpha over {len(df)} splits, {args.dataset}, {args.query_mode} queries")
    print(f"  mean   {a.mean():.3f}")
    print(f"  median {a.median():.3f}")
    print(f"  sd     {a.std():.3f}")
    print(f"  range  {a.min():.3f} to {a.max():.3f}")
    print(f"  both children probed   {df['frac_probing_both'].mean():.3f}")
    print(f"  one child probed       {df['frac_probing_one'].mean():.3f}")
    print(f"  neither child probed   {df['frac_probing_neither'].mean():.3f}")
    # 0.5 is what dividing the access of the parent between the children would give, the value this
    # code assumed before alpha was measured
    print("\nours assumes 0.5, Quake reference implementation uses 0.9")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

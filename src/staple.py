"""Grow the starting index under LIRE."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import DATA_DIR
from src.index import IVFIndex
from src.maintainers.lire_lite import LireLiteMaintainer

# number of workload clusters, as in Big ANN
N_WORKLOAD_CLUSTERS: int = 64

# stored cluster labels, keep them for reproducibility
WORKLOAD_LABEL_CACHE_DIR: Path = DATA_DIR / "workload_labels"


def workload_labels(
    vectors: np.ndarray, n_clusters: int = N_WORKLOAD_CLUSTERS, seed: int = 42
) -> np.ndarray:
    """Assign every vector to a workload cluster, cached on disk."""
    WORKLOAD_LABEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = (WORKLOAD_LABEL_CACHE_DIR /
             f"labels_{len(vectors)}_{vectors.shape[1]}_c{n_clusters}_s{seed}.npy")
    if cache.exists():
        return np.load(cache)

    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), size=min(100_000, len(vectors)), replace=False)
    km = KMeans(n_clusters=n_clusters, n_init=1, max_iter=10, random_state=seed)
    km.fit(vectors[idx].astype(np.float32))
    c = km.cluster_centers_.astype(np.float32)
    sq = np.einsum("ij,ij->i", c, c)
    out = np.empty(len(vectors), dtype=np.int32)
    for i in range(0, len(vectors), 50_000):
        blk = vectors[i:i + 50_000].astype(np.float32)
        out[i:i + 50_000] = (sq[None, :] - 2.0 * (blk @ c.T)).argmin(1)
    np.save(cache, out)
    return out


def clustered_runbook(
    spare_per_cluster: np.ndarray,
    n_inserts: int,
    rng: np.random.Generator,
    insert_delete_ratio: float = 1.0,
    n_rounds: int = 5,
    alpha: tuple[float, ...] = (100.0, 15.0, 10.0, 5.0, 3.0),
    delete_fraction: tuple[float, float] = (0.5, 0.9),
) -> list[tuple[str, int, int]]:
    """Return the Big ANN clustered arrival schedule as a list of batches."""
    n_clusters = len(spare_per_cluster)
    if n_clusters == 0 or n_inserts <= 0:
        return []
    weights = rng.dirichlet(alpha, size=n_clusters)
    for c in range(n_clusters):
        rng.shuffle(weights[c])
    total_spare = float(spare_per_cluster.sum())
    share = np.minimum(spare_per_cluster / total_spare * float(n_inserts), spare_per_cluster)
    per_cell = np.floor(share[:, None] * weights).astype(np.int64)
    lo, hi = delete_fraction
    mean_frac = 0.5 * (lo + hi)
    ops: list[tuple[str, int, int]] = []
    for r in range(n_rounds):
        for c in range(n_clusters):
            if per_cell[c, r] > 0:
                ops.append(("insert", c, int(per_cell[c, r])))
        for c in range(n_clusters):
            if per_cell[c, r] <= 0:
                continue
            u = float(rng.uniform(lo, hi)) / mean_frac
            n_del = int(round(u * per_cell[c, r] / max(insert_delete_ratio, 1e-9)))
            if n_del > 0:
                ops.append(("delete", c, n_del))
    return ops


@dataclass
class Staple:
    """A grown starting index and what the stream needs from it."""

    index: IVFIndex
    hi: int
    lo: int
    live: int
    step: int
    rng: np.random.Generator
    members: dict[int, np.ndarray]
    cursor: dict[int, int]
    resident: dict[int, list[int]]
    age_ids: list[int]
    alive: list[bool]
    pos: dict[int, int]
    splits: int = 0
    merges: int = 0
    consumed: set[int] = field(default_factory=set)

    def sizes(self) -> np.ndarray:
        """Return the live size of every partition of the grown index."""
        return np.array([p.size() for p in self.index.partitions.values()])

    def surviving_ids(self) -> list[int]:
        """Return the ids of the live vectors, oldest first."""
        return [v for i, v in enumerate(self.age_ids) if self.alive[i]]


def grow_staple(
    base: np.ndarray,
    labels: np.ndarray,
    cap: int = 50_000,
    target_avg: int = 100,
    bootstrap: int = 5_000,
    build_ratio: float = 4.0,
    window: int = 8,
    seed: int = 42,
    n_clusters: int = N_WORKLOAD_CLUSTERS,
) -> Staple:
    """Grow an index from a small k-means start to `cap` live vectors under LIRE."""
    rng = np.random.default_rng(seed)
    members: dict[int, np.ndarray] = {}
    for c in range(n_clusters):
        rows = np.flatnonzero(labels == c)
        rng.shuffle(rows)
        members[c] = rows
    cursor = dict.fromkeys(members, 0)
    spare_per_cluster = np.array([len(members[c]) for c in range(n_clusters)])

    # inserts needed to reach the cap
    net = max(0.05, 1.0 - 1.0 / build_ratio)
    budget = int(cap / net * 1.25)
    book = clustered_runbook(spare_per_cluster, budget, rng, insert_delete_ratio=build_ratio)

    hi = 4 * target_avg
    lo = max(2, hi // window)
    k0 = max(2, bootstrap // target_avg)
    index = IVFIndex(dim=base.shape[1])

    boot_rows: list[int] = []
    for c in range(n_clusters):  # spread over every cluster
        take = bootstrap // n_clusters
        boot_rows.extend(members[c][cursor[c]:cursor[c] + take].tolist())
        cursor[c] += take
    index.build(base[boot_rows].astype(np.float32), list(boot_rows),
                n_partitions=k0, seed=seed)

    maint = LireLiteMaintainer(max_partition_size=hi, min_partition_size=lo,
                               enable_merge=True, seed=seed)
    resident: dict[int, list[int]] = {c: [] for c in range(n_clusters)}
    for r in boot_rows:
        resident[int(labels[r])].append(r)
    live = len(boot_rows)
    consumed = set(boot_rows)
    age_ids: list[int] = list(boot_rows)
    alive: list[bool] = [True] * len(boot_rows)
    pos = {v: i for i, v in enumerate(boot_rows)}

    step, stop = 0, False
    for kind, c, count in book:
        if stop:
            break
        for _ in range(count):
            if kind == "insert":
                if cursor[c] >= len(members[c]):
                    break
                row = int(members[c][cursor[c]])
                cursor[c] += 1
                index.insert(row, base[row].astype(np.float32))
                resident[c].append(row)
                consumed.add(row)
                pos[row] = len(age_ids)
                age_ids.append(row)
                alive.append(True)
                live += 1
            else:
                if not resident[c]:
                    break
                i = int(rng.integers(0, len(resident[c])))
                resident[c][i], resident[c][-1] = resident[c][-1], resident[c][i]
                vid = resident[c].pop()
                index.delete(vid)
                alive[pos[vid]] = False
                live -= 1
            step += 1
            maint.maybe_maintain(index, step)
            if live >= cap:
                stop = True
                break

    # stop when LIRE has nothing left to do
    for _ in range(50):
        rep = maint.maintain(index)
        if not (rep.num_splits or rep.num_merges):
            break

    return Staple(
        index=index, hi=hi, lo=lo, live=live, step=step, rng=rng,
        members=members, cursor=cursor, resident=resident,
        age_ids=age_ids, alive=alive, pos=pos,
        splits=int(maint._cumulative_splits), merges=int(maint._cumulative_merges),
        consumed=consumed,
    )


def sliding_window_cohorts(
    staple: Staple, max_t: int
) -> dict[int, list[int]]:
    """Expire the starting index over the first half of the stream."""
    surviving = staple.surviving_ids()
    w = max(1, max_t // 2)
    chunk = max(1, len(surviving) // w)
    due: dict[int, list[int]] = {}
    for i in range(w):
        seg = surviving[i * chunk:(i + 1) * chunk] if i < w - 1 else surviving[i * chunk:]
        if seg:
            due.setdefault(i, []).extend(seg)
    return due

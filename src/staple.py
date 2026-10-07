"""Grow the starting index under a maintenance policy instead of building it with k-means.

Global k-means puts no limit on partition size, so it cannot balance partitions when a partition
holds fewer vectors than the space has dimensions. On GIST, with about 100 vectors per partition
in 960 dimensions, it leaves 14.6 percent of partitions below the merge threshold and the
smallest with a single vector, and no choice of the number of vectors, K or the average size
avoids that. An index grown from empty under a maintenance policy is bounded in both directions by
construction, because a split leaves two halves of about half the split threshold and inserts only
ever add. On GIST at 200,000 vectors with the same K and thresholds, the grown index has a smallest
partition of 82 vectors and none outside the thresholds, where exact Lloyd gives 1 and 45.

The arrivals follow the NeurIPS 2023 Big ANN streaming runbook, inserts and deletes local to one
cluster at a time, in shuffled Dirichlet burst rounds. Starting from empty, the first burst into
a cluster creates structure, which is how the published generator works.

The construction differs from Big ANN in two declared ways. It deletes one vector for every four
inserted, instead of Big ANN's delete fraction of 0.5 to 0.9, which would consume about 667,000
vectors to reach 200,000 live ones, while the arrival schedule itself is kept. And the index is
maintained by LIRE, SPFresh section 3.2, which reacts to size alone and never reads query access,
so the starting index carries no query distribution.

The result, a `Staple`, keeps the size thresholds it was grown to satisfy, and callers read
`Staple.hi` and `Staple.lo` instead of deriving thresholds from the final average size. Ada-IVF
also publishes its size limits as fixed constants rather than as multiples of an average.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import DATA_DIR
from src.index import IVFIndex
from src.maintainers.lire_lite import LireLiteMaintainer

# the number of workload clusters the arrival schedule walks over, the number Big ANN uses for
# its streaming runbooks
N_WORKLOAD_CLUSTERS: int = 64

# where the workload cluster labels are stored. labelling 1,000,000 vectors takes about a minute.
# the stored labels also anchor reproducibility, because k-means over many vectors gives a
# different answer at a different thread count, so a refit under other thread settings would
# change every starting index and every number derived from it. do not delete this directory to
# force a refit
WORKLOAD_LABEL_CACHE_DIR: Path = DATA_DIR / "workload_labels"


def workload_labels(
    vectors: np.ndarray, n_clusters: int = N_WORKLOAD_CLUSTERS, seed: int = 42
) -> np.ndarray:
    """Assign every base vector to a workload cluster, stored on disk after the first call.

    Workload clusters are the regions a burst of arrivals falls into, and they are not the index
    partitions. They depend on the data and the seed alone, so every policy in a comparison sees
    the same stream. The clusters are fitted with k-means on a sample of 100,000 vectors, and
    every vector is then assigned to the nearest cluster centre.
    """
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
    """Return the Big ANN clustered arrival schedule as a list of batches.

    Each cluster draws Dirichlet weights over the rounds, one of them much larger than the rest,
    and the weights are shuffled per cluster, so each cluster has one large burst in a round chosen
    at random. Each batch is an
    operation kind, insert or delete, a cluster and a count. Deletes follow each round's inserts,
    a random fraction of them divided by `insert_delete_ratio`.
    """
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
    """A grown starting index, with everything a measured stream needs to continue from it.

    `rng` is the generator the construction used, passed on rather than reseeded. The stream draws
    its cluster walk from the state the construction left, so a fresh generator would give a
    different stream.

    `members` holds the vector rows of each workload cluster in arrival order, and `cursor` the
    next unused position in each. `age_ids` lists every inserted vector in insertion order, and
    `alive` marks which are still live, so the list is always sorted oldest first.
    """

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
    """Grow an index from a small k-means start to `cap` live vectors under LIRE.

    The first `bootstrap` vectors, spread over every workload cluster, are partitioned with
    k-means at `target_avg` vectors per partition. The rest arrive by the clustered schedule, and
    LIRE maintains the index as it grows. The split threshold is 4 times `target_avg`, and the
    merge threshold is the split threshold divided by `window`. The reported runs use a window of
    8, the ratio SPFresh publishes, 80 over 10, which with a target of 100 gives 400 and 50.
    """
    rng = np.random.default_rng(seed)
    members: dict[int, np.ndarray] = {}
    for c in range(n_clusters):
        rows = np.flatnonzero(labels == c)
        rng.shuffle(rows)
        members[c] = rows
    cursor = dict.fromkeys(members, 0)
    spare_per_cluster = np.array([len(members[c]) for c in range(n_clusters)])

    # each insert adds 1 minus 1 over the ratio live vectors net, so budget the inserts the cap needs
    net = max(0.05, 1.0 - 1.0 / build_ratio)
    budget = int(cap / net * 1.25)
    book = clustered_runbook(spare_per_cluster, budget, rng, insert_delete_ratio=build_ratio)

    hi = 4 * target_avg
    lo = max(2, hi // window)
    k0 = max(2, bootstrap // target_avg)
    index = IVFIndex(dim=base.shape[1])

    boot_rows: list[int] = []
    for c in range(n_clusters):          # the bootstrap is spread over every cluster
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

    # construction ends when LIRE has no work left, not when the cap is reached. maybe_maintain
    # acts only every check interval, so stopping at the cap could leave work undone. on three GIST
    # starting indexes this loop changed nothing, so it is a safeguard
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
    """Schedule the vectors of the starting index to expire over the first half of the stream.

    The starting index is the content of the retention window at time 0, not a permanent layer
    beneath it. Left permanent, it would give every partition a floor it could never fall below.
    It is split into `max_t // 2` groups, oldest first, one expiring per time step, the same lag
    every later batch expires at. Returns a map from time step to the ids that expire then.
    """
    surviving = staple.surviving_ids()
    w = max(1, max_t // 2)
    chunk = max(1, len(surviving) // w)
    due: dict[int, list[int]] = {}
    for i in range(w):
        seg = surviving[i * chunk:(i + 1) * chunk] if i < w - 1 else surviving[i * chunk:]
        if seg:
            due.setdefault(i, []).extend(seg)
    return due

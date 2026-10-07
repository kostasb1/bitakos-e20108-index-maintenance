"""The retention window stream that every reported number is measured on.

A run starts from a grown starting index (`src/staple.py`) and continues with a Big ANN style
sliding window. Arrivals walk the 64 workload clusters in a seeded random order, with a Gaussian
spread of `width` clusters around the current position, and every batch expires `max_t // 2` time
steps after it arrived, so one region drains while another fills. The starting index itself
expires first, over the same lag. Every update is followed by `query_ratio` routed queries from a
pool that concentrates on one region of the query space, and those queries are the access signal
the Quake and bandit policies read.

The order of the random draws matters. Every policy starts from a copy of the same starting index
and continues its generator, so the cluster walk, the inserts and the expiry schedule are the same
for every policy. The second draw in `run_stream` is a shuffle whose result is never read, kept
because it advances the generator, and removing it would change every arrival.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable

import numpy as np

from src.calibration import has_price_list
from src.config import DEFAULT_K, DEFAULT_NPROBE, MAINTENANCE_CHECK_INTERVAL
from src.metrics import (
    compute_live_ground_truth,
    costs_at_target_recalls,
    priced_cost_at,
    recall_at_k_live,
    recall_at_k_per_query,
    stored_cost_at,
)
from src.staple import N_WORKLOAD_CLUSTERS, Staple, sliding_window_cohorts

# the points of the run, as fractions, at which the scoring is read besides the start and the
# end. the main result is the reading at the end, chosen before any run was seen, and the mean over
# these points and the end is reported beside it
CHECKPOINT_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75)

# the routed queries are served at the nprobe that meets this recall, the same target as the
# scoring, measured again every ROUTE_CALIBRATION_INTERVAL updates on a fixed sample of the routed
# pool. at a fixed nprobe a finer index always looks cheaper, because the recall it loses is never
# priced, so the access signal would carry exactly the bias the fixed recall scoring removes. Quake
# picks nprobe per query to meet a recall target, and this is the same idea per window
ROUTE_TARGET_RECALL: float = 0.9
ROUTE_CALIBRATION_QUERIES: int = 100

# the recall targets read beside the main 0.9, and the tag each one carries in a reading and in a
# result row, so a ranking can be checked on either side of the target
EXTRA_TARGETS: dict[float, str] = {0.8: "r80", 0.95: "r95"}
ROUTE_CALIBRATION_INTERVAL: int = MAINTENANCE_CHECK_INTERVAL


def recall_matched_nprobe(index, queries: np.ndarray, start: int,
                          target: float = ROUTE_TARGET_RECALL, k: int = DEFAULT_K) -> int:
    """Return the smallest nprobe at which mean recall at `k` on `queries` meets `target`.

    The search starts from `start`, the previous window's value, and moves one step at a time,
    since the index changes little between windows. The ground truth is exact over the live
    vectors. Nothing here records access, so finding the nprobe cannot change the signal it is
    found for.
    """
    gt = compute_live_ground_truth(index, queries, k)
    n_parts = len(index.partitions)
    if gt.shape[1] == 0 or n_parts == 0:
        return max(1, min(start, max(1, n_parts)))
    seen: dict[int, float] = {}

    def rec(n: int) -> float:
        """Return the mean recall at nprobe `n`, measuring it only once."""
        if n not in seen:
            seen[n] = float(recall_at_k_per_query(index, queries, gt, k, n).mean())
        return seen[n]

    n = max(1, min(start, n_parts))
    if rec(n) >= target:
        while n > 1 and rec(n - 1) >= target:
            n -= 1
        return n
    while n < n_parts:
        n += 1
        if rec(n) >= target:
            break
    return n


def query_pool(queries: np.ndarray, seed: int, hot_fraction: float, hot_prob: float,
               eval_queries: int, pool_size: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Return a routed query pool and a scoring sample drawn from that same pool.

    The sample is not held out, so a policy that learns from access would be scored on queries it
    learned from. The reported runs use `split_query_pools` instead. Not used by the reported runs.
    """
    qrng = np.random.default_rng(seed + 7)
    n_hot = max(1, int(len(queries) * hot_fraction))
    fh = qrng.random(pool_size) < hot_prob
    qpool = queries[np.where(fh, qrng.integers(0, n_hot, pool_size),
                             qrng.integers(0, len(queries), pool_size))].astype(np.float32)
    eval_q = qpool[np.random.default_rng(seed + 11).choice(
        len(qpool), size=min(eval_queries, len(qpool)), replace=False)]
    return qpool, eval_q


def query_region(queries: np.ndarray, seed: int, fraction: float) -> tuple[int, np.ndarray]:
    """Return the region the concentrated queries come from, the `fraction` nearest to one query.

    The centre query is drawn from the seed, so where the concentration falls also varies across
    seeds. Quake section 7.1 and Ada-IVF section 5.1.2 both concentrate their skewed workloads
    spatially in this way. Returns the centre and the region, nearest first, ties broken by query
    id.
    """
    n = max(1, int(len(queries) * fraction))
    centre = int(np.random.default_rng(seed + 19).integers(len(queries)))
    d = np.linalg.norm(queries.astype(np.float64) - queries[centre].astype(np.float64), axis=1)
    return centre, np.argsort(d, kind="stable")[:n]


def split_query_pools(queries: np.ndarray, seed: int, hot_fraction: float, hot_prob: float,
                      eval_queries: int, pool_size: int = 4000):
    """Return the routed pool, a sample of it, and a held out sample that is never routed.

    The query ids are split in two halves, the region and the rest alike. The stream routes a
    mixture drawn from the first half, `hot_prob` of it from that half of the region and the rest
    from the whole half. The scoring reads a mixture drawn the same way from the second half,
    which the stream never routes, so no policy is scored on a query it learned from, and both
    halves concentrate on the same region. The sample of the routed pool is kept so the gap
    between the two can be reported for every policy.
    """
    rng = np.random.default_rng(seed + 13)
    _, region = query_region(queries, seed, hot_fraction)
    inside = rng.permutation(region)
    outside = rng.permutation(np.setdiff1d(np.arange(len(queries)), region))
    halves = []
    for h in (0, 1):
        region_h = inside[h::2]
        halves.append((region_h, np.concatenate([region_h, outside[h::2]])))

    def draw(half, n, r):
        """Draw `n` query ids from one half, `hot_prob` of them from its part of the region."""
        region_h, all_h = half
        from_region = r.random(n) < hot_prob
        return np.where(from_region, region_h[r.integers(0, len(region_h), n)],
                        all_h[r.integers(0, len(all_h), n)])

    qrng = np.random.default_rng(seed + 7)
    pool_ids = draw(halves[0], pool_size, qrng)
    erng = np.random.default_rng(seed + 11)
    served_ids = pool_ids[erng.choice(pool_size, size=eval_queries, replace=False)]
    held_ids = draw(halves[1], eval_queries, erng)
    assert not set(held_ids) & set(pool_ids), "the held out sample must never be routed"
    f = np.float32
    return queries[pool_ids].astype(f), queries[served_ids].astype(f), queries[held_ids].astype(f)


def measure(index, eval_q: np.ndarray, t: int, extra_targets: bool = True) -> dict:
    """Read the scoring at time step `t` and return it as a dictionary.

    The main reading is the cost at recall 0.9, at the nprobe that meets it, since a cost at a
    fixed nprobe can be gamed by splitting. With `extra_targets` the cost at 0.8 and 0.95 is read
    too, on the same ground truth. Also returns the raw recall at the default nprobe, the cost
    with tombstones scanned, the partition count and the live and stored sizes, and, when a price
    list exists, the same costs priced in Faiss time.
    """
    targets = (0.9, *EXTRA_TARGETS) if extra_targets else (0.9,)
    readings = costs_at_target_recalls(index, eval_q, k=DEFAULT_K, targets=targets)
    extra = [(t, tag) for t, tag in EXTRA_TARGETS.items() if t in readings]
    c, npr, rec, std = readings[0.9]
    raw = recall_at_k_live(index, eval_q, k=DEFAULT_K, nprobe=DEFAULT_NPROBE)
    sizes = np.array([p.size() for p in index.partitions.values()])
    # the same reading with tombstones scanned, the SPFresh delete model, at the same nprobe
    cs = stored_cost_at(index, eval_q, npr) if len(sizes) else 0.0
    r = dict(t=t, cost=c, cost_stored=cs, nprobe=npr, achieved=rec, recall_std=std,
             raw_recall=raw,
             parts=len(sizes), live=int(sizes.sum()),
             stored=int(sum(p.total_size() for p in index.partitions.values())))
    for target, tag in extra:
        r[f"cost_{tag}"], r[f"nprobe_{tag}"] = readings[target][0], readings[target][1]
    # the same readings priced in Faiss time, each probed list at its length. the reported query
    # cost is built from these, and the counts above stay as the raw readings
    if has_price_list(index.dim):
        r["cost_priced"] = priced_cost_at(index, eval_q, npr)
        r["cost_priced_stored"] = priced_cost_at(index, eval_q, npr, stored=True)
        for target, tag in extra:
            r[f"cost_priced_{tag}"] = priced_cost_at(index, eval_q, readings[target][1])
    return r


def run_stream(
    staple0: Staple,
    base: np.ndarray,
    maint,
    qpool: np.ndarray,
    eval_q: np.ndarray,
    seed: int,
    measure_ops: int = 120_000,
    max_t: int = 100,
    width: float = 3.0,
    query_ratio: int = 8,
    log: Callable[[dict], None] | None = None,
    served_q: np.ndarray | None = None,
) -> dict:
    """Run one policy over the retention window, starting from a copy of `staple0`.

    Each of the `max_t` time steps inserts about `measure_ops // max_t` vectors around the current
    workload cluster and deletes the vectors due to expire. Every update is followed by
    `query_ratio` routed queries and a chance for the policy to act, and the routing nprobe is
    recalibrated every 1,000 updates. Returns the readings at t = 0, at the checkpoints and at the
    end, the policy, the final index and the counts of the stream. `log` is called with each
    reading as it is taken, and with `served_q` every reading is repeated on that sample under a
    served_ prefix.
    """
    st = copy.deepcopy(staple0)
    index, members, cursor = st.index, st.members, st.cursor

    rng = st.rng
    walk = rng.permutation(N_WORKLOAD_CLUSTERS)
    _order = np.concatenate([np.ones(measure_ops // 2, np.int8),
                             np.zeros(measure_ops - measure_ops // 2, np.int8)])
    rng.shuffle(_order)
    # the routed queries come from their own generator, so routing can never change the arrivals
    qrng = np.random.default_rng(seed + 7)

    per_t = max(1, measure_ops // max_t)
    due = sliding_window_cohorts(st, max_t)

    def reading(t: int) -> dict:
        """Read the scoring at `t`, and on the served sample too when one is given."""
        r = measure(index, eval_q, t)
        if served_q is not None:
            r.update({f"served_{k}": v for k, v in
                      measure(index, served_q, t, extra_targets=False).items() if k != "t"})
        return r

    rows = [reading(0)]
    if log:
        log(rows[-1])
    # a fixed sample of the routed pool, from its own generator so no other draw changes
    calib_q = qpool[np.random.default_rng(seed + 17).choice(
        len(qpool), size=min(ROUTE_CALIBRATION_QUERIES, len(qpool)), replace=False)]
    route_nprobe = recall_matched_nprobe(index, calib_q, DEFAULT_NPROBE)
    route_log = [route_nprobe]
    since_calibration = 0

    def calibrate() -> None:
        """Count one update, and recalibrate the routing nprobe every interval."""
        nonlocal route_nprobe, since_calibration
        since_calibration += 1
        if since_calibration >= ROUTE_CALIBRATION_INTERVAL:
            route_nprobe = recall_matched_nprobe(index, calib_q, route_nprobe)
            route_log.append(route_nprobe)
            since_calibration = 0

    maint.start(index, st.step)
    n_ins = n_del = 0
    # every insert is routed to its nearest centroid, K comparisons, which nothing else charges.
    # it grows with K, so a policy that splits pays it on every insert
    insert_routing = 0
    t1 = time.perf_counter()
    check = {int(max_t * f) for f in CHECKPOINT_FRACTIONS}

    for t in range(max_t):
        frac = t / max(1, max_t - 1)
        c = int(walk[min(N_WORKLOAD_CLUSTERS - 1, max(0, int(round(
            frac * (N_WORKLOAD_CLUSTERS - 1) + width * rng.standard_normal()))))])
        batch: list[int] = []
        for _ in range(per_t):
            tries, cc = 0, c
            while cursor[cc] >= len(members[cc]) and tries < N_WORKLOAD_CLUSTERS:
                cc = (cc + 1) % N_WORKLOAD_CLUSTERS
                tries += 1
            if cursor[cc] >= len(members[cc]):
                break
            row = int(members[cc][cursor[cc]])
            cursor[cc] += 1
            insert_routing += len(index.partitions)
            index.insert(row, base[row].astype(np.float32))
            batch.append(row)
            n_ins += 1
            calibrate()
            for _ in range(query_ratio):
                index.route(qpool[int(qrng.integers(0, len(qpool)))], route_nprobe)
            maint.maybe_maintain(index, st.step + n_ins + n_del)
        # every batch expires a fixed lag later, nothing stays forever
        expire = t + max(1, max_t // 2)
        if expire < max_t:
            due.setdefault(expire, []).extend(batch)
        for vid in due.pop(t, []):
            if index.vector_to_partition.get(vid) is None:
                continue
            index.delete(vid)
            n_del += 1
            calibrate()
            for _ in range(query_ratio):
                index.route(qpool[int(qrng.integers(0, len(qpool)))], route_nprobe)
            maint.maybe_maintain(index, st.step + n_ins + n_del)
        if t in check:
            rows.append(reading(t))
            if log:
                log(rows[-1])

    rows.append(reading(max_t))
    if log:
        log(rows[-1])
    return dict(rows=rows, maintainer=maint, index=index, inserts=n_ins, deletes=n_del,
                insert_routing=insert_routing, route_nprobe=route_log,
                seconds=time.perf_counter() - t1)

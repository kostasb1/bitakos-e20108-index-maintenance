"""Measure the prices of the cost unit, on Faiss IndexIVFFlat rather than on the Python index.

    python scripts/calibrate_query_cost.py
    python scripts/calibrate_query_cost.py --lambda-only

A query ranks every centroid, then opens nprobe lists and scans the vectors in them. The Python
index cannot price either step, because the cost of each of its calls is mostly interpreter
overhead, so both are measured on Faiss IndexIVFFlat, one query at a time on one thread, as Quake
measures. The full search and the centroid ranking alone are timed, and their difference is the
list scan.

The list scan is measured on a design that separates list length from list count. The index
always holds N_REF distinct vectors, the size of the indexes the prices are applied to, and every
list holds exactly m of them, so a query at nprobe n scans exactly n times m. For each m the time
to open and scan one more list, t(m), is the slope of scan time against n. The query cost prices
each probed list at t(its length), in units of one scan distance, the time per vector over long
lists, see `src/calibration.py`. The centroid ranking is priced per centroid, from flat indexes of
growing size.

The script also prices maintenance work by kind, small and large k-means and a vector read or
copy, and writes the scan cost curves the Quake and bandit policies read, so they price actions
in the unit they are scored in. Every price is the median of five independent calibrations, with
the range stored beside it, in results/cost_model/query_cost_calibration.yaml. With --lambda-only
the curves are rebuilt from the stored prices and nothing is measured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.calibration import per_list_overhead_ns, scan_unit_ns  # noqa: E402
from src.config import COST_MODEL_DIR  # noqa: E402
from src.cost_model import lambda_from_prices, save_profiled_cost_model  # noqa: E402
from src.dataset import load_gist1m, load_sift1m  # noqa: E402

CALIBRATION_PATH = COST_MODEL_DIR / "query_cost_calibration.yaml"
# the starting index holds 50,000 live vectors and the stream grows it to about 60,000
N_REF, NQ, K_NN = 51_200, 300, 10
# list lengths the policies reach and either side of them. below 25 the list count is capped, so
# those points hold fewer than N_REF vectors in all, 10,240 at 5 and 20,480 at 10
LIST_SIZES = (0, 5, 10, 25, 50, 100, 200, 400, 800, 1600)
MAX_LISTS = 2048
NPROBES = (1, 2, 4, 8, 16, 32, 64)
CENTROID_COUNTS = (128, 256, 512, 1024, 2048)


def calibrate(base: np.ndarray, queries: np.ndarray, seed: int = 0) -> dict:
    """Measure the list price curve and the centroid price once, and return them in scan distances.

    Each measurement times whole loops of single query calls, one loop for the full search and
    one for the centroid ranking alone, so the Python overhead of each call cancels in the
    difference. The minimum over five loops is kept, since an interruption only ever adds time.
    """
    import time

    import faiss

    faiss.omp_set_num_threads(1)
    rng = np.random.default_rng(seed)
    pool = np.ascontiguousarray(base[rng.choice(len(base), N_REF, replace=False)].astype("float32"))
    xq = np.ascontiguousarray(queries[:NQ].astype("float32"))
    d = pool.shape[1]

    def best_ns(call) -> float:
        """Return the fastest of five timed loops over the queries, in nanoseconds per query."""
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            for i in range(NQ):
                call(xq[i:i + 1])
            best = min(best, (time.perf_counter() - t0) * 1e9 / NQ)
        return best

    # every list holds exactly m distinct vectors, so nprobe n scans n times m, and the slope of
    # scan time against n is the time to open and scan one list of m
    list_ns = []
    for m in LIST_SIZES:
        n_lists = MAX_LISTS if m == 0 else min(MAX_LISTS, N_REF // m)
        quant = faiss.IndexFlatL2(d)
        quant.add(np.ascontiguousarray(pool[rng.choice(N_REF, n_lists, replace=False)]))
        index = faiss.IndexIVFFlat(quant, d, n_lists)
        index.is_trained = True
        for lst in range(n_lists if m else 0):
            codes = np.ascontiguousarray(pool[lst * m:(lst + 1) * m]).view("uint8")
            ids = np.arange(lst * m, (lst + 1) * m, dtype="int64")
            index.invlists.add_entries(lst, m, faiss.swig_ptr(ids), faiss.swig_ptr(codes))
        index.ntotal = n_lists * m
        ns = [n for n in NPROBES if n <= n_lists]
        ys = []
        for n in ns:
            index.nprobe = n
            faiss.cvar.indexIVF_stats.reset()
            full = best_ns(lambda q, ix=index: ix.search(q, K_NN))
            ndis = faiss.cvar.indexIVF_stats.ndis / (5 * NQ)
            assert abs(ndis - n * m) < 1e-6, f"m {m} nprobe {n} scanned {ndis}"
            ys.append(full - best_ns(lambda q, n=n, qt=quant: qt.search(q, n)))
        list_ns.append(float(np.polyfit(np.array(ns, dtype=float), np.array(ys), 1)[0]))
    unit = scan_unit_ns(LIST_SIZES, list_ns)

    # the centroid ranking, one pass over K centroids, priced per centroid
    cent = []
    for k in CENTROID_COUNTS:
        flat = faiss.IndexFlatL2(d)
        flat.add(np.ascontiguousarray(pool[:k]))
        cent.append(best_ns(lambda q, fl=flat: fl.search(q, 16)))
    per_centroid = float(np.polyfit(np.array(CENTROID_COUNTS, dtype=float), np.array(cent),
                                    1)[0])
    # list_overhead is a diagnostic that nothing prices with, the single fixed cost per list a
    # straight line through the curve implies. it is kept because its range crosses zero on both
    # datasets, which shows no such constant exists. ns_per_vector_faiss is the unit, the time per
    # vector over lists of at least 200 vectors
    return {"list_sizes": [int(m) for m in LIST_SIZES],
            "list_prices": [float(t / unit) for t in list_ns],
            "list_overhead": float(per_list_overhead_ns(LIST_SIZES, list_ns) / unit),
            "centroid_ratio": per_centroid / unit,
            "ns_per_vector_faiss": unit, "faiss_version": faiss.__version__}


def calibrate_work(base: np.ndarray, ns_per_scan_distance: float) -> dict:
    """Measure the price of each kind of maintenance operation, in scan distances.

    Faiss k-means runs a fixed number of Lloyd iterations, so its time divides exactly by
    iterations times points times centroids. It is measured at two shapes, because batching gains
    very differently. A split or a DeDrift step clusters a few hundred points into a handful of
    centroids, and a rebuild clusters the whole index into hundreds. A vector read or copy is timed
    on contiguous blocks the size of a partition, since a partition is stored as one block.
    """
    import time

    import faiss

    faiss.omp_set_num_threads(1)
    rng = np.random.default_rng(1)
    d = base.shape[1]

    def kmeans_ns(n: int, k: int, niter: int, reps: int) -> float:
        """Return the fastest time per distance of a Faiss k-means of n points into k centres."""
        best = float("inf")
        for r in range(reps):
            x = np.ascontiguousarray(base[rng.choice(len(base), n, replace=False)].astype("float32"))
            km = faiss.Kmeans(d, k, niter=niter, nredo=1, verbose=False, seed=r,
                              max_points_per_centroid=n)
            t0 = time.perf_counter()
            km.train(x)
            best = min(best, (time.perf_counter() - t0) * 1e9 / (niter * n * k))
        return best

    small = float(np.mean([kmeans_ns(400, 2, 20, 30), kmeans_ns(1200, 4, 20, 20)]))
    bulk = kmeans_ns(60_000, 180, 25, 3)

    pool = np.ascontiguousarray(base[:60_000].astype("float32"))
    dst = np.empty((400, d), dtype="float32")
    best = float("inf")
    for _ in range(30):
        start = int(rng.integers(0, len(pool) - 400))
        t0 = time.perf_counter()
        for _ in range(200):
            blk = pool[start:start + 400]
            np.copyto(dst, blk)
            _ = dst.sum(axis=0)
        best = min(best, (time.perf_counter() - t0) * 1e9 / (200 * 400))
    return {"kmeans_small": small / ns_per_scan_distance,
            "kmeans_bulk": bulk / ns_per_scan_distance,
            "read_copy": best / ns_per_scan_distance}


REPEATS = 5


def main() -> None:
    """Calibrate both datasets REPEATS times, store the medians and ranges, and write the curves.

    With --lambda-only the curves are rebuilt from the stored prices and nothing is measured.
    """
    if "--lambda-only" in sys.argv:
        stored = yaml.safe_load(CALIBRATION_PATH.read_text())
        for dim, entry in stored.items():
            write_lambda(int(dim), entry["median"])
        return
    out = {}
    for name, loader in (("sift1m", load_sift1m), ("gist1m", load_gist1m)):
        base, queries, _ = loader()
        dim = base.shape[1]
        runs = []
        for r in range(REPEATS):
            cal = calibrate(base, queries, seed=r)
            cal.update(calibrate_work(base, cal["ns_per_vector_faiss"]))
            runs.append(cal)
            print(f"d={dim} run {r}: unit {cal['ns_per_vector_faiss']:.2f} ns, list overhead "
                  f"{cal['list_overhead']:.2f}, centroid {cal['centroid_ratio']:.3f}, small k "
                  f"means {cal['kmeans_small']:.2f}, bulk {cal['kmeans_bulk']:.3f}, read or copy "
                  f"{cal['read_copy']:.2f}, list prices "
                  + ", ".join(f"{m}:{p:.1f}" for m, p in zip(cal["list_sizes"],
                                                            cal["list_prices"], strict=True)),
                  flush=True)
        keys = ("list_overhead", "centroid_ratio", "kmeans_small", "kmeans_bulk", "read_copy",
                "ns_per_vector_faiss")
        med = {k: float(np.median([r[k] for r in runs])) for k in keys}
        spread = {k: [float(min(r[k] for r in runs)), float(max(r[k] for r in runs))] for k in keys}
        # the curve is taken point by point, each list length with its own median and range
        curves = np.array([r["list_prices"] for r in runs])
        med["list_sizes"] = list(runs[0]["list_sizes"])
        med["list_prices"] = [float(v) for v in np.median(curves, axis=0)]
        spread["list_prices"] = [[float(v) for v in curves.min(axis=0)],
                                 [float(v) for v in curves.max(axis=0)]]
        out[dim] = {"dataset": name, "faiss_version": runs[0]["faiss_version"],
                    "repeats": REPEATS, "n_ref": N_REF, "median": med, "range": spread}
        print(f"d={dim} MEDIAN " + ", ".join(f"{k} {med[k]:.3f} [{spread[k][0]:.3f}, "
                                           f"{spread[k][1]:.3f}]" for k in keys), flush=True)
        write_lambda(dim, med)
    CALIBRATION_PATH.write_text(yaml.safe_dump(out, sort_keys=True))
    print(f"-> {CALIBRATION_PATH}")


def write_lambda(dim: int, med: dict) -> None:
    """Write the scan cost curve for `dim` dimensions, which is the Faiss list price curve itself."""
    path = COST_MODEL_DIR / f"profiled_lambda_dim{dim}.yaml"
    model = lambda_from_prices(med)
    save_profiled_cost_model(model, path)
    print(f"   {path.name}, lambda(25) {model.scan_latency(25):.0f} ns, lambda(400) "
          f"{model.scan_latency(400):.0f} ns, centroid {model.centroid_ns:.2f} ns", flush=True)


if __name__ == "__main__":
    main()

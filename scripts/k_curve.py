"""The query cost of fresh builds at several partition counts K, with the build spread at each.

    python scripts/k_curve.py --dataset gist1m --seed 42

The ranking of the policies depends on the starting partition count. This measures the property of
the data behind that. The live vectors at the end of the no maintenance run of a seed are built
fresh with exact k-means at K0 times each multiple, K0 being the partition count of the starting
index, each in `--builds` row orders, and scored exactly as the main runs score, on their 400 held
out queries and on a larger held out sample drawn the same way. The multiples were fixed before any
run, 2 is the half size check, 3 and 4.5 bracket the bandit and Quake on GIST, and 8.5 reaches
Quake at tau 50.

The gain a fixed K policy would see from a larger starting K, and the gain a policy that grows K
gets from the K it reaches, can both be read off this curve. Every index here is a fresh build, so
it says nothing about maintenance. Results go to results/raw/k_curve_<dataset>_s<seed>.csv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rebuild_noise import cv, score, spread  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.dataset import load_gist1m, load_sift1m  # noqa: E402
from src.index import IVFIndex  # noqa: E402
from src.maintainers.no_op import NoOpMaintainer  # noqa: E402
from src.staple import N_WORKLOAD_CLUSTERS, grow_staple, workload_labels  # noqa: E402
from src.stream import run_stream, split_query_pools  # noqa: E402

MULTIPLES = (1.0, 1.5, 2.0, 3.0, 4.5, 8.5)


def main() -> None:
    """Run the no maintenance stream, then build and score its final vectors at every multiple of K0."""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["sift1m", "gist1m"], default="gist1m")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--builds", type=int, default=3)
    p.add_argument("--large", type=int, default=2000)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    out_path = a.output or RESULTS_DIR / f"k_curve_{a.dataset}_s{a.seed}.csv"

    base, queries, _ = load_sift1m() if a.dataset == "sift1m" else load_gist1m()
    labels = workload_labels(base, N_WORKLOAD_CLUSTERS, a.seed)
    st = grow_staple(base, labels, cap=50_000, target_avg=100, bootstrap=5_000, build_ratio=4.0,
                     window=8, seed=a.seed)
    qpool, _, eval_q = split_query_pools(queries, a.seed, 0.1, 0.8, 400)
    _, _, large_q = split_query_pools(queries, a.seed, 0.1, 0.8, a.large)
    t = time.perf_counter()
    out = run_stream(st, base, NoOpMaintainer(), qpool, eval_q, a.seed, measure_ops=120_000,
                     max_t=100, width=3.0, query_ratio=8)
    end = out["index"]
    k0 = len(end.partitions)
    print(f"no_op cell {time.perf_counter() - t:.0f}s, K0 {k0}", flush=True)

    ids: list[int] = []
    blocks: list[np.ndarray] = []
    for part in end.partitions.values():
        live_ids, live_vecs = part.get_live_vectors()
        if live_ids:
            ids.extend(live_ids)
            blocks.append(live_vecs)
    vecs = np.vstack(blocks).astype(np.float32)
    ids_arr = np.asarray(ids)
    c, n = score(end, eval_q)
    cl, nl = score(end, large_q)
    rows = [dict(kind="no_op_end", mult=1.0, K=k0, i=0, cost_400=c, nprobe_400=n,
                 cost_large=cl, nprobe_large=nl)]
    print(f"no_op end  400 {c:8.1f} at {n:5.2f}   {a.large} {cl:8.1f} at {nl:5.2f}", flush=True)

    perm_rng = np.random.default_rng(1000 + a.seed)
    perms = [np.arange(len(ids))] + [perm_rng.permutation(len(ids)) for _ in range(a.builds - 1)]
    for mult in MULTIPLES:
        k = int(round(mult * k0))
        for i, perm in enumerate(perms):
            t = time.perf_counter()
            ix = IVFIndex(dim=end.dim)
            ix.build(vecs[perm], ids_arr[perm].tolist(), n_partitions=k, seed=a.seed)
            c, n = score(ix, eval_q)
            cl, nl = score(ix, large_q)
            rows.append(dict(kind="build", mult=mult, K=k, i=i, cost_400=c, nprobe_400=n,
                             cost_large=cl, nprobe_large=nl))
            print(f"K {k:5d} build {i}  400 {c:8.1f} at {n:5.2f}   {a.large} {cl:8.1f} at "
                  f"{nl:5.2f}  ({time.perf_counter() - t:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    b = df[df.kind == "build"]
    ref = b[b.mult == 1.0]
    print(f"\n{a.dataset} seed {a.seed}, K0 {k0}, live {len(ids)}, against the mean fresh build "
          f"at K0")
    for mult, g in b.groupby("mult"):
        print(f"  x{mult:<4} K {int(g.K.iloc[0]):5d}  400 {100 * (g.cost_400.mean() / ref.cost_400.mean() - 1):+6.1f}%"
              f" cv {cv(g.cost_400):4.1f} spread {spread(g.cost_400):5.1f}   "
              f"{a.large} {100 * (g.cost_large.mean() / ref.cost_large.mean() - 1):+6.1f}%"
              f" cv {cv(g.cost_large):4.1f}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()

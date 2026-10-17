"""How much of the query cost spread comes from the queries and how much from the build."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# pin threads, as in final_sweep
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.calibration import query_cost  # noqa: E402
from src.config import DEFAULT_K, RESULTS_DIR  # noqa: E402
from src.dataset import load_gist1m, load_sift1m  # noqa: E402
from src.index import IVFIndex  # noqa: E402
from src.maintainers.no_op import NoOpMaintainer  # noqa: E402
from src.metrics import cost_at_target_recall, priced_cost_at  # noqa: E402
from src.staple import N_WORKLOAD_CLUSTERS, grow_staple, workload_labels  # noqa: E402
from src.stream import run_stream, split_query_pools  # noqa: E402


def score(index: IVFIndex, q: np.ndarray) -> tuple[float, float]:
    """Return the query cost at recall 0.9 and its nprobe."""
    _, npr, _, _ = cost_at_target_recall(index, q, DEFAULT_K, 0.9)
    return query_cost(priced_cost_at(index, q, npr), len(index.partitions), index.dim), float(npr)


def spread(x) -> float:
    """Return the largest value over the smallest, minus one, in percent."""
    x = np.asarray(x, dtype=float)
    return 100 * (x.max() / x.min() - 1)


def cv(x) -> float:
    """Coefficient of variation, in percent."""
    x = np.asarray(x, dtype=float)
    return 100 * x.std(ddof=1) / x.mean()


def main() -> None:
    """Run the no maintenance stream, then score repeated builds of its end state."""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["sift1m", "gist1m"], default="sift1m")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--builds", type=int, default=8)
    p.add_argument("--large", type=int, default=4000)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    out_path = a.output or RESULTS_DIR / f"rebuild_noise_{a.dataset}_s{a.seed}.csv"

    # defaults of the main runs
    base, queries, _ = load_sift1m() if a.dataset == "sift1m" else load_gist1m()
    labels = workload_labels(base, N_WORKLOAD_CLUSTERS, a.seed)
    st = grow_staple(base, labels, cap=50_000, target_avg=100, bootstrap=5_000, build_ratio=4.0,
                     window=8, seed=a.seed)
    qpool, _, eval_q = split_query_pools(queries, a.seed, 0.1, 0.8, 400)
    # larger held-out sample, same draw
    _, _, large_q = split_query_pools(queries, a.seed, 0.1, 0.8, a.large)
    t = time.perf_counter()
    out = run_stream(st, base, NoOpMaintainer(), qpool, eval_q, a.seed, measure_ops=120_000,
                     max_t=100, width=3.0, query_ratio=8)
    end = out["index"]
    print(f"no_op cell {time.perf_counter() - t:.0f}s, K {len(end.partitions)}", flush=True)

    ids: list[int] = []
    blocks: list[np.ndarray] = []
    for part in end.partitions.values():
        live_ids, live_vecs = part.get_live_vectors()
        if live_ids:
            ids.extend(live_ids)
            blocks.append(live_vecs)
    vecs = np.vstack(blocks).astype(np.float32)
    ids_arr = np.asarray(ids)
    k = len(end.partitions)
    print(f"live {len(ids)}, eval distinct {len(np.unique(eval_q, axis=0))} of {len(eval_q)}, "
          f"large distinct {len(np.unique(large_q, axis=0))} of {len(large_q)}", flush=True)

    rows = []
    c, n = score(end, eval_q)
    cl, nl = score(end, large_q)
    rows.append(dict(kind="no_op_end", i=0, cost_400=c, nprobe_400=n, cost_large=cl,
                     nprobe_large=nl))
    fixed = {"no_op_end": end}
    perm_rng = np.random.default_rng(1000 + a.seed)
    for i in range(a.builds):
        t = time.perf_counter()
        perm = np.arange(len(ids)) if i == 0 else perm_rng.permutation(len(ids))
        ix = IVFIndex(dim=end.dim)
        ix.build(vecs[perm], ids_arr[perm].tolist(), n_partitions=k, seed=a.seed)
        c, n = score(ix, eval_q)
        cl, nl = score(ix, large_q)
        rows.append(dict(kind="build", i=i, cost_400=c, nprobe_400=n, cost_large=cl,
                         nprobe_large=nl))
        print(f"build {i}  400 {c:8.1f} at {n:5.2f}   {a.large} {cl:8.1f} at {nl:5.2f}  "
              f"({time.perf_counter() - t:.0f}s)", flush=True)
        if i == 0:
            fixed["build_0"] = ix

    n_slices = a.large // 400
    for name, ix in fixed.items():
        for s in range(n_slices):
            c, n = score(ix, large_q[s * 400:(s + 1) * 400])
            rows.append(dict(kind=f"slice_{name}", i=s, cost_400=c, nprobe_400=n))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    b = df[df.kind == "build"]
    print(f"\n{a.dataset} seed {a.seed}, K {k}, live {len(ids)}")
    print(f"  builds on the sweep's 400    spread {spread(b.cost_400):5.1f}%  cv {cv(b.cost_400):4.1f}%")
    print(f"  builds on {a.large} held out     spread {spread(b.cost_large):5.1f}%  "
          f"cv {cv(b.cost_large):4.1f}%")
    for name in fixed:
        s = df[df.kind == f"slice_{name}"]
        print(f"  {name:>9}, {len(s)} slices of 400  spread {spread(s.cost_400):5.1f}%  "
              f"cv {cv(s.cost_400):4.1f}%")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()

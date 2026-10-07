"""What one DeDrift Split pass does to the index, measured pass by pass on one main run.

    python scripts/dedrift_split_trace.py --dataset gist1m --seed 42

DeDrift Split ends 22.0 percent above no maintenance on GIST, on all five seeds. The policy
follows DeDrift section 5.1, "We train k-means with k2 centroids on B1 union B2, and replace the k2
involved clusters", so the question is what that step does here, at K near 178, where the largest
cluster rule of 0.2 percent of K rounds to one cluster a pass. The proposed mechanism is that each
pass pulls the smallest clusters, wherever they are, into one k-means with the largest, so their
vectors end up under centroids that are not the nearest to them. This measures it.

Each pass is read before and after the step, without changing anything.

    mse_all        mean squared distance of every live vector to its own centroid
    mse_b1, mse_b2 the same over the vectors of the largest cluster and of the smallest clusters
    nearest_b1, nearest_b2
                   the share of those vectors whose own centroid is the nearest of all K. A vector
                   whose centroid is not its nearest is found only when a query probes past the
                   nearest partitions
    b2_slot_gap    the distance from each smallest cluster's old centroid to the nearest centroid
                   after the pass, over the median spacing between nearest centroids before it

The measurement wraps DeDriftMaintainer._split from outside and the run goes through
final_sweep.main, so the run is the same as in the main results, and the script checks at the end
that its query cost equals the main result row of the same seed. Results go to
results/raw/diag_2809.
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

import final_sweep as fs  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.maintainers.dedrift import DeDriftMaintainer  # noqa: E402

TRACE: list[dict] = []


def _nearest_fraction(vecs: np.ndarray, own: np.ndarray, cents: np.ndarray) -> float:
    """Return the share of rows whose own centroid is at least as near as every centroid in `cents`."""
    if len(vecs) == 0:
        return float("nan")
    v = vecs.astype(np.float64)
    c = cents.astype(np.float64)
    d_all = (v * v).sum(1)[:, None] - 2 * v @ c.T + (c * c).sum(1)[None, :]
    d_own = ((v - own.astype(np.float64)) ** 2).sum(1)
    return float(np.mean(d_own <= d_all.min(1) + 1e-6 * np.abs(d_own)))


def _traced_split(orig):
    """Return a replacement for DeDriftMaintainer._split that runs `orig` and records the pass."""
    def split(self, index, report):
        """Copy the index state, run the real split, and append the measurements to TRACE."""
        pre = {}
        for pid, p in index.partitions.items():
            ids, vecs = p.get_live_vectors()
            pre[pid] = (p.centroid.astype(np.float32).copy(), list(ids), np.array(vecs, copy=True))
        orig(self, index, report)
        gone = [pid for pid in pre if pid not in index.partitions]
        row = dict(pass_=len(TRACE), K=len(pre), involved=len(gone))
        if not gone:
            TRACE.append(row)
            return
        k = min(self.n_largest_for(len(pre)), len(gone))
        by_size = sorted(gone, key=lambda q: len(pre[q][1]), reverse=True)
        b1, b2 = by_size[:k], by_size[k:]
        pre_c = np.stack([pre[q][0] for q in pre])
        post_ids = list(index.partitions)
        post_c = np.stack([index.partitions[q].centroid for q in post_ids]).astype(np.float32)

        def mse(pids, when):
            """Mean squared distance of the given clusters' vectors to their centroid, before or after."""
            tot, n = 0.0, 0
            for q in pids:
                c, ids, vecs = pre[q]
                if not ids:
                    continue
                if when == "pre":
                    own = np.broadcast_to(c, vecs.shape)
                else:
                    own = np.stack([index.partitions[index.vector_to_partition[v]].centroid
                                    for v in ids])
                tot += float(((vecs.astype(np.float64) - own) ** 2).sum())
                n += len(ids)
            return tot / max(n, 1), n

        def nearest(pids, when):
            """Share of the given clusters' vectors whose own centroid is nearest, before or after."""
            rows = [pre[q] for q in pids if pre[q][1]]
            if not rows:
                return float("nan")
            vecs = np.vstack([r[2] for r in rows])
            if when == "pre":
                own = np.vstack([np.broadcast_to(r[0], r[2].shape) for r in rows])
                return _nearest_fraction(vecs, own, pre_c)
            own = np.stack([index.partitions[index.vector_to_partition[v]].centroid
                            for r in rows for v in r[1]])
            return _nearest_fraction(vecs, own, post_c)

        all_pre = mse(list(pre), "pre")
        # after the pass every live vector is in index.partitions, so it is read there
        tot, n = 0.0, 0
        for p in index.partitions.values():
            ids, vecs = p.get_live_vectors()
            if ids:
                tot += float(((np.asarray(vecs, np.float64) - p.centroid) ** 2).sum())
                n += len(ids)
        cc = pre_c.astype(np.float64)
        dd = (cc * cc).sum(1)[:, None] - 2 * cc @ cc.T + (cc * cc).sum(1)[None, :]
        np.fill_diagonal(dd, np.inf)
        spacing = float(np.median(np.sqrt(np.maximum(dd.min(1), 0))))
        gaps = [float(np.sqrt(((post_c.astype(np.float64) - pre[q][0]) ** 2).sum(1).min()))
                / spacing for q in b2]
        row.update(
            k=k, k2=len(gone), n_b1=sum(len(pre[q][1]) for q in b1),
            n_b2=sum(len(pre[q][1]) for q in b2),
            mse_all_pre=all_pre[0], mse_all_post=tot / max(n, 1),
            mse_b1_pre=mse(b1, "pre")[0], mse_b1_post=mse(b1, "post")[0],
            mse_b2_pre=mse(b2, "pre")[0], mse_b2_post=mse(b2, "post")[0],
            nearest_b1_pre=nearest(b1, "pre"), nearest_b1_post=nearest(b1, "post"),
            nearest_b2_pre=nearest(b2, "pre"), nearest_b2_post=nearest(b2, "post"),
            b2_slot_gap=float(np.mean(gaps)) if gaps else float("nan"),
            centroid_spacing=spacing)
        TRACE.append(row)
    return split


def main() -> None:
    """Run one DeDrift Split run with the pass recorder, write the trace, and check the run matches."""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["sift1m", "gist1m"], default="gist1m")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = RESULTS_DIR / "diag_2809"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{'sift' if a.dataset == 'sift1m' else 'gist'}_s{a.seed}"
    cell_csv = out / f"dedrift_split_cell_{tag}.csv"
    trace_csv = out / f"dedrift_split_trace_{tag}.csv"

    DeDriftMaintainer._split = _traced_split(DeDriftMaintainer._split)
    sys.argv = [sys.argv[0], "--dataset", a.dataset, "--seeds", str(a.seed),
                "--maintainers", "dedrift_split", "--no-figures", "--output", str(cell_csv)]
    fs.main()
    t = pd.DataFrame(TRACE)
    t.to_csv(trace_csv, index=False)

    cell = pd.read_csv(cell_csv).iloc[0]
    sweep = pd.concat([pd.read_csv(q) for q in (RESULTS_DIR / "final_2709").glob(f"{tag}*.csv")])
    ref = sweep[(sweep.maintainer == "dedrift_split") & (sweep.seed == a.seed)].query_cost.iloc[0]
    same = abs(cell.query_cost - ref) <= 1e-9 * abs(ref)
    print(f"\nquery_cost {cell.query_cost:.6f} against the final_2709 row {ref:.6f}, "
          f"{'identical' if same else 'DIFFERENT, the trace is not of the sweep cell'}")
    d = t.dropna(subset=["mse_all_pre"])
    print(f"{len(t)} passes, {len(d)} that repartitioned, k {d.k.median():.0f}, k2 "
          f"{d.k2.median():.0f}, B1 {d.n_b1.median():.0f} vectors, B2 {d.n_b2.median():.0f}")
    for col in ("mse_all", "mse_b1", "mse_b2", "nearest_b1", "nearest_b2"):
        print(f"  {col:11s} before {d[col + '_pre'].mean():10.4g}  after {d[col + '_post'].mean():10.4g}")
    print(f"  b2 slot gap, nearest centroid after over median spacing, mean {d.b2_slot_gap.mean():.2f}")
    print(f"-> {trace_csv}")


if __name__ == "__main__":
    main()

"""tables and figures from the main results"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.calibration import prices  # noqa: E402

DIMS = {"sift1m": 128, "gist1m": 960}
# main result first, then the extra readings
PAIRED_COLS = ("query_cost", "query_cost_mean", "query_cost_r80", "query_cost_r95",
               "query_cost_rebuilt_k")
RANKING_COLS = ["query_cost", "query_cost_mean", "served_query_cost", "work_calibrated",
                "compaction_reads", "ingest", "total_compute", "total_vs_no_op", "efficiency",
                "break_even_reads", "parts", "nprobe"]
LABELS = {"no_op": "no maintenance", "global_rebuild": "global rebuild", "lire_lite": "LIRE",
          "dedrift_lazy": "DeDrift Lazy", "dedrift_split": "DeDrift Split",
          "dedrift_hybrid": "DeDrift Hybrid", "cost_driven_quake": "Quake",
          "cost_driven_quake_tau50": "Quake at tau 50 ns (sensitivity)", "bandit": "bandit"}


def load(sweep: Path) -> dict[str, pd.DataFrame]:
    """read every result file in `sweep`, one table per dataset"""
    out = {}
    for ds in DIMS:
        tag = "sift" if ds == "sift1m" else "gist"
        files = sorted(glob.glob(str(sweep / f"{tag}_*.csv")))
        if not files:
            continue
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        dup = d.duplicated(["maintainer", "seed"], keep=False)
        if dup.any():
            raise SystemExit(f"{ds}: duplicate cells {d[dup][['maintainer', 'seed']].values}")
        mixed = d.groupby("maintainer")["fp"].nunique()
        if (mixed > 1).any():
            raise SystemExit(f"{ds}: a strategy carries two fingerprints, {mixed[mixed > 1]}")
        out[ds] = d
    return out


def enrich(d: pd.DataFrame, dim: int) -> pd.DataFrame:
    """add the columns derived against no maintenance on the same seed"""
    d = d.copy()
    base = d[d.maintainer == "no_op"].set_index("seed")
    updates = d.inserts + d.deletes
    saved = d.seed.map(base.query_cost_mean) - d.query_cost_mean
    d["saved_total"] = saved * d.queries_served
    d["efficiency"] = np.where(d.work_calibrated > 0, d.saved_total / d.work_calibrated, np.nan)
    # inf when the saving is not positive, missing when there is no no_op row
    d["break_even_reads"] = np.where(saved.isna(), np.nan,
                                     np.where(saved > 0, d.work_calibrated / (updates * saved),
                                              np.inf))
    d["total_vs_no_op"] = 100 * (d.total_compute / d.seed.map(base.total_compute) - 1)
    d["stored_vs_no_op"] = 100 * (d.query_cost_stored / d.seed.map(base.query_cost_stored) - 1)
    d["total_tomb_vs_no_op"] = 100 * (d.total_compute_tombstone
                                      / d.seed.map(base.total_compute_tombstone) - 1)
    return d


def paired_vs_no_op(d: pd.DataFrame, col: str = "query_cost") -> pd.DataFrame:
    """compare every policy with no maintenance on the same seed, for one cost column"""
    p = d.pivot(index="seed", columns="maintainer", values=col)
    return summarise(np.log(p.div(p["no_op"], axis=0)), col, f"{col}_vs_no_op")


def _pct(x):
    """turn a log ratio into a percent change"""
    return 100 * (np.exp(x) - 1)


def summarise(logs: pd.DataFrame, prefix: str, head: str) -> pd.DataFrame:
    """mean and paired t interval of per-seed log ratios, in percent"""
    n = logs.notna().sum()
    mean = logs.mean()
    half = stats.t.ppf(0.975, n - 1) * logs.std(ddof=1) / np.sqrt(n)
    out = pd.DataFrame({head: _pct(mean), f"{prefix}_ci_lo": _pct(mean - half),
                        f"{prefix}_ci_hi": _pct(mean + half), f"{prefix}_below": (logs < 0).sum(),
                        f"{prefix}_seeds": n})
    for seed in logs.index:
        out[f"{prefix}_s{seed}"] = _pct(logs.loc[seed])
    return out


def equal_k(d: pd.DataFrame) -> pd.DataFrame:
    """compare each policy's own index with a fresh exact k-means at the K it ended at"""
    p = d.pivot(index="seed", columns="maintainer", values="query_cost")
    f = d.pivot(index="seed", columns="maintainer", values="query_cost_rebuilt_k")
    return summarise(np.log(p / f), "own_vs_fresh", "own_vs_fresh")


def control(d: pd.DataFrame, name: str, ref: str) -> pd.DataFrame:
    """compare a control run with a reference policy on the same seed"""
    rows = []
    for col in ("query_cost", "query_cost_mean", "work_calibrated"):
        p = d.pivot(index="seed", columns="maintainer", values=col)[[name, ref]].dropna()
        s = summarise(np.log(p[[name]].div(p[ref], axis=0)), "r", "vs_ref").loc[name]
        per = " ".join(f"{s[f'r_s{seed}']:+.1f}" for seed in p.index)
        rows.append(dict(control=name, ref=ref, reading=col, vs_ref=s.vs_ref, ci_lo=s.r_ci_lo,
                         ci_hi=s.r_ci_hi, below=int(s.r_below), seeds=int(s.r_seeds),
                         per_seed=per))
    return pd.DataFrame(rows)


def tombstone(d: pd.DataFrame) -> pd.DataFrame:
    """ranking under the SPFresh delete model"""
    cols = ["query_cost_stored", "stored_vs_no_op", "work_calibrated_tombstone",
            "total_compute_tombstone", "total_tomb_vs_no_op", "stored", "live"]
    return d.groupby("maintainer")[cols].mean().sort_values("query_cost_stored")


def ranking(d: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """mean over seeds of every column, with the paired comparisons"""
    cols = [c for c in (cols or RANKING_COLS) if c in d.columns]
    g = d.groupby("maintainer")
    t = g[cols].mean()
    t["seeds"] = g.seed.nunique()
    for c in PAIRED_COLS:
        if c in d.columns:
            t = t.join(paired_vs_no_op(d, c))
    return t.sort_values("query_cost_vs_no_op")


def wins(d: pd.DataFrame, col: str = "query_cost") -> pd.DataFrame:
    """return, for every pair of policies, on how many seeds the row policy is cheaper"""
    p = d.pivot(index="seed", columns="maintainer", values=col)
    names = list(p.columns)
    return pd.DataFrame({a: {b: int((p[a] < p[b]).sum()) for b in names} for a in names}).T


def centroid_sensitivity(d: pd.DataFrame, ds: str, calibrated: float) -> pd.DataFrame:
    """rank the policies with the centroid price and with a price of 1"""
    rows = []
    for label, c in (("calibrated", calibrated), ("1", 1.0)):
        q = d.priced_scan + c * d.parts
        t = d.assign(q=q).groupby("maintainer").q.mean().sort_values()
        for rank, (m, v) in enumerate(t.items(), start=1):
            rows.append(dict(dataset=ds, price=label, centroid_price=c, maintainer=m,
                             query_cost=v, rank=rank))
    return pd.DataFrame(rows)


def frontier(tables: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame], out: Path) -> None:
    """plot query cost against maintenance work"""
    fig, axes = plt.subplots(1, len(tables), figsize=(6.2 * len(tables), 4.8), squeeze=False)
    for ax, (ds, t) in zip(axes[0], tables.items(), strict=True):
        d = data[ds]
        for m, r in t.iterrows():
            g = d[d.maintainer == m]
            x = max(r.work_calibrated, 1.0)
            ax.errorbar(x, r.query_cost, yerr=[[r.query_cost - g.query_cost.min()],
                                               [g.query_cost.max() - r.query_cost]],
                        fmt="o", capsize=3, markersize=7)
            ax.annotate(LABELS.get(m, m), (x, r.query_cost), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("maintenance work, calibrated scan distances (log)")
        ax.set_ylabel("query cost at recall 0.9, held out, calibrated")
        ax.set_title(f"{ds}, mean over {d.seed.nunique()} seeds, range as bars")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"frontier.{ext}", dpi=160)
    plt.close(fig)


def main() -> None:
    """read the results, write every table and the figure, and print the summaries"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=Path("results/raw/final_2709"))
    ap.add_argument("--out", type=Path, default=Path("results/final_2709"))
    # control run folder, a missing folder skips it
    ap.add_argument("--control", type=Path, default=Path("results/raw/control_2809"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    data = {ds: enrich(d, DIMS[ds]) for ds, d in load(a.sweep).items()}
    controls = load(a.control) if a.control.is_dir() else {}
    tables, sens = {}, []
    for ds, d in data.items():
        t = ranking(d)
        tables[ds] = t
        t.to_csv(a.out / f"ranking_{ds}.csv")
        wins(d).to_csv(a.out / f"wins_{ds}.csv")
        tomb = tombstone(d)
        tomb.to_csv(a.out / f"tombstone_{ds}.csv")
        sens.append(centroid_sensitivity(d, ds, prices(DIMS[ds])["centroid_ratio"]))
        print(f"\n=== {ds}, {d.seed.nunique()} seeds, {len(d)} cells")
        print("  headline, end of run query cost at recall 0.9 against no_op, paired t interval")
        print(t[["query_cost", "query_cost_vs_no_op", "query_cost_ci_lo", "query_cost_ci_hi",
                 "query_cost_below", "query_cost_seeds", "work_calibrated", "efficiency",
                 "break_even_reads", "parts"]].round(2).to_string())
        print("  the headline per seed, percent against no_op")
        print(t[[c for c in t.columns if c.startswith("query_cost_s")
                 and c != "query_cost_seeds"]].round(1).to_string())
        beside = [f"{c}_vs_no_op" for c in PAIRED_COLS[1:] if f"{c}_vs_no_op" in t.columns]
        print("  beside it, never in its place, percent against no_op")
        print(t[beside].round(1).to_string())
        print(f"\n  {ds}, tombstones scanned, the SPFresh delete model")
        print(tomb.round(2).to_string())
        if "query_cost_rebuilt_k" in d.columns:
            ek = equal_k(d).join(d.groupby("maintainer").parts.mean().rename("K"))
            ek = ek.sort_values("own_vs_fresh")
            ek.to_csv(a.out / f"equal_k_{ds}.csv")
            print(f"\n  {ds}, equal K, own index against a fresh build at its own K, percent")
            print(ek.round(1).to_string())
        if ds in controls:
            c = controls[ds]
            # the control must come from the same set of runs
            b = d[d.maintainer == "bandit"].set_index("seed").parts
            k = c.set_index("seed").parts
            if not (k == b.reindex(k.index)).all():
                raise SystemExit(f"{ds}: the control's K per seed {k.to_dict()} is not the "
                                 f"bandit's {b.to_dict()} in {a.sweep}, pass --control to "
                                 f"the control of this sweep or to a directory that does not "
                                 f"exist")
            both = pd.concat([d, c], ignore_index=True)
            ct = pd.concat([control(both, name, "bandit") for name in c.maintainer.unique()])
            ct.to_csv(a.out / f"control_{ds}.csv", index=False)
            print(f"\n  {ds}, control arms against the bandit on the same seed, percent")
            print(ct.round(1).to_string(index=False))
    s = pd.concat(sens)
    s.to_csv(a.out / "centroid_sensitivity.csv", index=False)
    print("\nrank by centroid price")
    print(s.pivot_table(index=["dataset", "maintainer"], columns="price",
                        values="rank").to_string())
    frontier(tables, data, a.out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()

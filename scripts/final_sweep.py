"""Run the experiments into one result file."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# pin threads before numpy is imported, k-means depends on the thread count
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.calibration import prices, query_cost, work_cost  # noqa: E402
from src.config import (  # noqa: E402
    COST_MODEL_DIR,
    DEFAULT_K,
    DEFAULT_NPROBE,
    FIGURES_DIR,
    RESULTS_DIR,
)
from src.dataset import (  # noqa: E402
    load_gist1m,
    load_sift1m,
    make_synthetic,
)
from src.maintainers.bandit import BanditMaintainer  # noqa: E402
from src.maintainers.cost_driven_quake import CostDrivenQuakeMaintainer  # noqa: E402
from src.maintainers.dedrift import DeDriftMaintainer  # noqa: E402
from src.maintainers.global_rebuild import GlobalRebuildMaintainer  # noqa: E402
from src.maintainers.lire_lite import LireLiteMaintainer  # noqa: E402
from src.maintainers.no_op import NoOpMaintainer  # noqa: E402
from src.metrics import cost_rebuilt_at_own_k  # noqa: E402
from src.provenance import fingerprint, write_sidecar  # noqa: E402
from src.staple import N_WORKLOAD_CLUSTERS, grow_staple, workload_labels  # noqa: E402
from src.stream import EXTRA_TARGETS, run_stream, split_query_pools  # noqa: E402

# the compared policies
MAINTAINERS = ["no_op", "global_rebuild", "lire_lite",
               "dedrift_lazy", "dedrift_split", "dedrift_hybrid",
               "cost_driven_quake", "bandit"]
DEDRIFT_VARIANTS = ["dedrift_lazy", "dedrift_split", "dedrift_hybrid"]
# variants not in the reported results
ABLATION_VARIANTS = ["bandit_linear", "lire_lite_no_merge"]
# Quake with tau = 50 ns, the value of the Quake SIFT1M experiment
SENSITIVITY_VARIANTS = ["cost_driven_quake_tau50"]
QUAKE_TAU_ARM_NS = 50.0
_PROFILED: dict = {}


def profile_path(dim: int) -> Path:
    """Return the path of the stored scan cost curve for vectors of `dim` dimensions."""
    return COST_MODEL_DIR / f"profiled_lambda_dim{dim}.yaml"


def profiled_cost_model(dim: int, reprofile: bool = False):
    """Load the scan cost curve for `dim` dimensions."""
    rebuild = ("lambda is the Faiss price curve, rebuild it with "
               "python scripts/calibrate_query_cost.py, or with --lambda-only from the stored "
               "price list")
    if reprofile:
        raise SystemExit(rebuild)
    if dim not in _PROFILED:
        from src.cost_model import load_profiled_cost_model

        path = profile_path(dim)
        if not path.exists():
            raise SystemExit(f"{path} is missing. {rebuild}")
        _PROFILED[dim] = load_profiled_cost_model(path)
    return _PROFILED[dim]


DEDRIFT_N_LARGEST: int | None = None  # None uses the fraction


def build(name: str, seed: int, hi: int, lo: int, cost_model):
    """Create the policy called `name`, configured as in the reported runs."""
    if name == "no_op":
        return NoOpMaintainer()
    if name == "global_rebuild":
        return GlobalRebuildMaintainer(seed=seed)
    if name in ("lire_lite", "lire_lite_no_merge"):
        return LireLiteMaintainer(
            max_partition_size=hi, min_partition_size=lo,
            enable_merge=(name == "lire_lite"), seed=seed)
    if name.startswith("dedrift_"):
        return DeDriftMaintainer(variant=name.split("_", 1)[1], n_largest=DEDRIFT_N_LARGEST,
                                 seed=seed)
    uncap = 1_000_000  # no limit on actions per pass
    if name == "cost_driven_quake":
        # Quake, published tau
        return CostDrivenQuakeMaintainer(
            cost_model=cost_model, max_actions_per_maintain=uncap,
            nprobe=DEFAULT_NPROBE, seed=seed)
    if name == "cost_driven_quake_tau50":
        return CostDrivenQuakeMaintainer(
            cost_model=cost_model, max_actions_per_maintain=uncap,
            nprobe=DEFAULT_NPROBE, seed=seed, tau=QUAKE_TAU_ARM_NS, delete_tau=QUAKE_TAU_ARM_NS)
    if name in ("bandit", "bandit_linear"):
        # bandit without the cost model
        return BanditMaintainer(
            cost_model=None if name == "bandit_linear" else cost_model,
            max_partition_size=hi, min_partition_size=lo,
            # every partition is a candidate, discount once per pass
            min_split_size=max(1, hi // 4), n_candidates=None, discount_mode="pass",
            # only records reversals, changes no decision
            coverage_penalty_weight=2.0, seed=seed, track_reversals=True)
    raise ValueError(name)


COST_AWARE = ("cost_driven_quake", "cost_driven_quake_tau50", "bandit")


def run_one_staple(base, name, seed, staple, qpool, eval_q, args, served_q=None) -> dict:
    """Run one policy on one seed and return its result row."""
    model = profiled_cost_model(base.shape[1]) if name in COST_AWARE else None
    maint = build(name, seed, staple.hi, staple.lo, model)
    out = run_stream(staple, base, maint, qpool, eval_q, seed,
                     measure_ops=args.measure_ops, max_t=args.max_t, width=args.width,
                     query_ratio=args.query_ratio, served_q=served_q)
    rows = out["rows"]
    s, e = rows[0], rows[-1]
    m = out["maintainer"]
    dim = base.shape[1]

    def qc(r: dict, prefix: str = "", priced: str = "cost_priced") -> float:
        """Query cost of a reading."""
        return query_cost(r[prefix + priced], r[prefix + "parts"], dim)

    # two delete models, the reported one removes deleted vectors at once
    p = prices(dim)
    reads = (int(m._cumulative_work) - int(m._cumulative_distances)
             - int(m._cumulative_kmeans) - int(m._cumulative_bulk_kmeans))
    compaction = int(m._cumulative_compaction)
    work_cal = work_cost(reads - compaction, m._cumulative_distances, m._cumulative_kmeans,
                         m._cumulative_bulk_kmeans, dim)
    work_tomb = work_cost(reads, m._cumulative_distances, m._cumulative_kmeans,
                          m._cumulative_bulk_kmeans, dim)
    # cost of routing inserts
    routing = p["centroid_ratio"] * int(out["insert_routing"])
    ingest = routing + p["read_copy"] * int(out["deletes"])
    queries_served = (int(out["inserts"]) + int(out["deletes"])) * args.query_ratio
    q_mean = float(np.mean([qc(r) for r in rows[1:]]))
    q_stored_mean = float(np.mean([qc(r, priced="cost_priced_stored") for r in rows[1:]]))
    row = {
        "maintainer": name, "seed": seed,
        # no ratio in the retention window
        "ratio": 1.0, "workload": "staple", "protocol": "heldout",
        # main result, query cost at recall 0.9 at the end of the run
        "query_cost0": qc(s), "query_cost": qc(e), "query_cost_mean": q_mean,
        "served_query_cost": qc(e, "served_") if served_q is not None else float("nan"),
        # raw readings
        "fixed_recall_cost0": s["cost"], "fixed_recall_cost": e["cost"],
        "fixed_recall_cost_mean": float(np.mean([r["cost"] for r in rows[1:]])),
        "fixed_recall_cost_k": e["cost"] + e["parts"],
        # priced scan without the centroid term
        "priced_scan": e["cost_priced"],
        "priced_scan_mean": float(np.mean([r["cost_priced"] for r in rows[1:]])),
        "recall": e["raw_recall"], "fixed_recall_achieved": e["achieved"],
        "fixed_recall_std": e["recall_std"],
        "nprobe": float(e["nprobe"]), "parts": int(e["parts"]), "live": int(e["live"]),
        "stored": int(e["stored"]),
        # maintenance work, raw and priced
        "work": int(m._cumulative_work), "work_calibrated": work_cal,
        "work_reads": reads, "distances": int(m._cumulative_distances),
        "kmeans_distances": int(m._cumulative_kmeans),
        "bulk_kmeans_distances": int(m._cumulative_bulk_kmeans),
        # total cost of the run
        "queries_served": queries_served,
        "total_compute": queries_served * q_mean + work_cal + ingest,
        "compaction_reads": compaction, "insert_routing": int(out["insert_routing"]),
        # nprobe of the routed queries
        "route_nprobe_mean": float(np.mean(out["route_nprobe"])),
        "route_nprobe_end": int(out["route_nprobe"][-1]),
        "ingest": ingest,
        # SPFresh delete model
        "fixed_recall_cost_stored": e["cost_stored"],
        "query_cost_stored": qc(e, priced="cost_priced_stored"),
        "query_cost_stored_mean": q_stored_mean,
        "work_calibrated_tombstone": work_tomb,
        "total_compute_tombstone": queries_served * q_stored_mean + work_tomb + routing,
        "splits": int(m._cumulative_splits), "merges": int(m._cumulative_merges),
        "reassigned": int(m._cumulative_reassigned),
        "repartitioned": int(m._cumulative_repartitioned),
        "collected": int(m._cumulative_collected),
        "centroids": int(m._cumulative_centroids),
        "declined": int(m._cumulative_declined),
        "merges_declined": int(m._cumulative_merges_declined),
        "examined": int(m._cumulative_examined),
        "inserts": int(out["inserts"]), "deletes": int(out["deletes"]),
    }
    # readings at recall 0.8 and 0.95
    for tag in EXTRA_TARGETS.values():
        priced = f"cost_priced_{tag}"
        row[f"query_cost_{tag}"] = qc(e, priced=priced)
        row[f"query_cost_{tag}_mean"] = float(np.mean([qc(r, priced=priced) for r in rows[1:]]))
        row[f"fixed_recall_cost_{tag}"] = e[f"cost_{tag}"]
        row[f"nprobe_{tag}"] = float(e[f"nprobe_{tag}"])
    # fresh k-means at the final K, to separate the effect of K
    rebuilt = cost_rebuilt_at_own_k(out["index"], eval_q, k=DEFAULT_K, target_recall=0.9,
                                    seed=seed)
    row["query_cost_rebuilt_k"] = query_cost(rebuilt["cost_priced"], rebuilt["parts"], dim)
    row["fixed_recall_cost_rebuilt_k"] = rebuilt["cost"]
    row["nprobe_rebuilt_k"] = float(rebuilt["nprobe"])
    if served_q is not None:
        row["served_fixed_recall_cost"] = e["served_cost"]
        row["served_nprobe"] = float(e["served_nprobe"])
    # readings during the run
    for r in rows[1:-1]:
        row[f"query_cost_t{r['t']}"] = qc(r)
        row[f"cost_t{r['t']}"] = r["cost"]
        row[f"nprobe_t{r['t']}"] = float(r["nprobe"])
        row[f"parts_t{r['t']}"] = int(r["parts"])
        row[f"recall_t{r['t']}"] = r["raw_recall"]
    if hasattr(m, "action_counts"):
        for k, v in m.action_counts.items():
            row[f"arm_{k.lower()}"] = int(v)
        row["reversals"] = int(getattr(m, "_reversals", 0))
    return row


def aggregate_and_plot(df: pd.DataFrame, figures_dir: Path, write_figures: bool = True) -> None:
    """Print the mean and spread of cost and work per policy, and optionally plot them."""
    global COST_COL, WORK_COL
    COST_COL = "query_cost" if "query_cost" in df.columns else "fixed_recall_cost"
    WORK_COL = "work_calibrated" if "work_calibrated" in df.columns else "work"
    if write_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
    for ratio, g in df.groupby("ratio"):
        agg = g.groupby("maintainer").agg(
            fr_mean=(COST_COL, "mean"), fr_std=(COST_COL, "std"),
            work_mean=(WORK_COL, "mean"), work_std=(WORK_COL, "std"),
            rec=("recall", "mean")).reset_index().sort_values("fr_mean")
        print(f"\n=== ratio={ratio} (mean +/- std over {g.seed.nunique()} seeds) ===", flush=True)
        for _, r in agg.iterrows():
            print(f"  {r.maintainer:>14}: fixed_recall_cost={r.fr_mean:7.0f}+/-{r.fr_std:5.0f}  "
                  f"work={r.work_mean:9.0f}+/-{r.work_std:8.0f}  recall={r.rec:.3f}", flush=True)

        if not write_figures:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 5))
        for _, r in agg.iterrows():
            ax.errorbar(r.work_mean, r.fr_mean,
                        xerr=(r.work_std if np.isfinite(r.work_std) else 0),
                        yerr=(r.fr_std if np.isfinite(r.fr_std) else 0),
                        fmt="o", markersize=9, capsize=4, label=r.maintainer)
            ax.annotate(r.maintainer, (r.work_mean, r.fr_mean),
                        textcoords="offset points", xytext=(7, 4), fontsize=8)
        ax.set_xlabel("Maintenance work (vectors processed)")
        ax.set_ylabel("Query cost at recall 0.9 (vectors / query)")
        ax.set_title(f"Fixed-recall Pareto, growth ratio={ratio} (mean +/- std)")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out = figures_dir / f"final_fixed_recall_pareto_ratio{ratio}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"  figure -> {out}", flush=True)


def main() -> None:
    """Parse the arguments, run every requested combination, and write the result file."""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sift1m", choices=["sift1m", "gist1m", "synthetic"])
    p.add_argument("--workload", default="staple", choices=["staple"],
                   help="the retention window stream of src/stream.py")
    # starting index and stream defaults
    p.add_argument("--cap", type=int, default=50_000)
    p.add_argument("--target-avg", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=5_000)
    p.add_argument("--build-ratio", type=float, default=4.0)
    p.add_argument("--window", type=int, default=8,
                   help="split threshold over merge threshold, 8 is SPFresh's 80 over 10")
    p.add_argument("--measure-ops", type=int, default=120_000)
    p.add_argument("--max-t", type=int, default=100)
    p.add_argument("--width", type=float, default=3.0)
    p.add_argument("--eval-queries", type=int, default=400)
    p.add_argument("--n-ops", type=int, default=60000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    p.add_argument("--ratios", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--initial-fraction", type=float, default=0.2)
    p.add_argument("--query-ratio", type=int, default=8)
    p.add_argument("--hot-fraction", type=float, default=0.1)
    p.add_argument("--hot-prob", type=float, default=0.8)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "final_sweep.csv")
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    p.add_argument("--no-figures", action="store_true",
                   help="skip the summary figure, for runs over part of the policies")
    p.add_argument("--maintainers", nargs="+", default=MAINTAINERS,
                   choices=sorted(set(MAINTAINERS + DEDRIFT_VARIANTS + ABLATION_VARIANTS
                                      + SENSITIVITY_VARIANTS)),
                   help="the policies to run, the ablation and sensitivity variants included")
    p.add_argument("--dedrift-n-largest", type=int, default=None,
                   help="a fixed number of largest DeDrift clusters. the default derives it from "
                        "DEDRIFT_N_LARGEST_FRACTION times K, the published 0.20 percent")
    p.add_argument("--delete-csf", type=float, default=0.0,
                   help="not used by the retention window stream, kept as a run parameter")
    p.add_argument("--reprofile-lambda", action="store_true",
                   help="not available, lambda is the Faiss price curve written by "
                        "scripts/calibrate_query_cost.py")
    p.add_argument("--resume", action="store_true",
                   help="keep the rows already in --output whose fingerprint matches the current "
                        "code, machine and parameters, and run only what is missing")
    p.add_argument("--drop-stale", action="store_true",
                   help="with --resume, allow discarding rows whose fingerprint does not match. "
                        "without it a mismatch stops the run rather than mixing two code "
                        "versions in one file")
    p.add_argument("--dry-run", action="store_true",
                   help="print which runs would run and which would be reused, then exit")
    args = p.parse_args()

    print(f"Loading {args.dataset}...", flush=True)
    if args.dataset == "sift1m":
        base, queries, gt = load_sift1m()
    elif args.dataset == "gist1m":
        base, queries, gt = load_gist1m()
    else:
        base, queries, gt = make_synthetic(n=50_000, dim=64, n_clusters=20, seed=args.seeds[0])
    # load the scan cost curve once
    if any(n in COST_AWARE for n in args.maintainers):
        profiled_cost_model(base.shape[1], reprofile=args.reprofile_lambda)

    args.ratios = [1.0]
    # run parameters for the fingerprint
    params = {"dataset": args.dataset, "workload": args.workload, "n_ops": args.n_ops,
              "cap": args.cap, "target_avg": args.target_avg, "bootstrap": args.bootstrap,
              "build_ratio": args.build_ratio, "window": args.window,
              "measure_ops": args.measure_ops, "max_t": args.max_t, "width": args.width,
              "eval_queries": args.eval_queries,
              "initial_fraction": args.initial_fraction, "query_ratio": args.query_ratio,
              "hot_fraction": args.hot_fraction, "hot_prob": args.hot_prob,
              "delete_csf": args.delete_csf, "dedrift_n_largest": args.dedrift_n_largest}
    # one fingerprint per policy
    fps: dict[str, tuple[str, dict]] = {}

    def fp_for(name: str) -> tuple[str, dict]:
        """Return the fingerprint and provenance of a policy's rows, computed once."""
        if name not in fps:
            fps[name] = fingerprint(params, name)
        return fps[name]

    for name in args.maintainers:
        fp, prov = fp_for(name)
        dirty = " dirty" if prov["git_dirty"] else ""
        print(f"fingerprint {name:>18} {fp}  code {prov['code_hash'][:12]}  "
              f"commit {prov['git_commit']}{dirty}", flush=True)

    rows: list[dict] = []
    reusable: set[tuple[str, int, float]] = set()
    if args.resume and args.output.exists():
        prior = pd.read_csv(args.output)
        has_fp = "fp" in prior.columns
        current = (prior.apply(lambda r: str(r["fp"]) == fp_for(str(r["maintainer"]))[0], axis=1)
                   if has_fp and len(prior) else pd.Series(False, index=prior.index))
        fresh = prior[current]
        stale = prior[~current]
        if len(stale) and not args.drop_stale:
            print(f"{len(stale)} of {len(prior)} rows in {args.output} do not carry their "
                  f"maintainer's current fingerprint.", flush=True)
            for _, r in stale.iterrows():
                seen = r["fp"] if has_fp else "no fingerprint"
                print(f"  stale  r={r['ratio']} s={r['seed']} "
                      f"{r['maintainer']:>18}  {seen}", flush=True)
            print("refusing to mix two generations of code in one CSV. pass --drop-stale to "
                  "discard them, or point --output somewhere new to keep both.", flush=True)
            return
        if len(stale):
            print(f"discarding {len(stale)} stale rows, --drop-stale was given", flush=True)
        rows = fresh.to_dict("records")
        reusable = {(str(r["maintainer"]), int(r["seed"]), float(r["ratio"])) for r in rows}
        print(f"resuming, {len(rows)} cells reused from {args.output}", flush=True)

    wanted = [(ratio, seed, name) for ratio in args.ratios for seed in args.seeds
              for name in args.maintainers]
    todo = [c for c in wanted if (c[2], int(c[1]), float(c[0])) not in reusable]
    print(f"{len(wanted)} cells requested, {len(wanted) - len(todo)} reused, "
          f"{len(todo)} to run", flush=True)
    if args.dry_run:
        for ratio, seed, name in todo:
            print(f"  would run  r={ratio} s={seed} {name}", flush=True)
        return

    total = len(todo)
    t0 = time.perf_counter()
    global DEDRIFT_N_LARGEST
    DEDRIFT_N_LARGEST = args.dedrift_n_largest

    # one starting index per seed, copied for each policy
    staples: dict[int, tuple] = {}

    def staple_for(seed: int):
        """Return the starting index and the query samples of a seed, building them once."""
        if seed not in staples:
            labels = workload_labels(base, N_WORKLOAD_CLUSTERS, seed)
            st = grow_staple(base, labels, cap=args.cap, target_avg=args.target_avg,
                             bootstrap=args.bootstrap, build_ratio=args.build_ratio,
                             window=args.window, seed=seed)
            qpool, served_q, eval_q = split_query_pools(queries, seed, args.hot_fraction,
                                                        args.hot_prob,
                                                        args.eval_queries)
            staples[seed] = (st, qpool, eval_q, served_q)
            print(f"  staple s={seed}, K {len(st.sizes())}, live {st.live}, "
                  f"thresholds {st.hi} and {st.lo}", flush=True)
        return staples[seed]

    for done, (ratio, seed, name) in enumerate(todo, start=1):
        rt = time.perf_counter()
        try:
            st, qpool, eval_q, served_q = staple_for(seed)
            r = run_one_staple(base, name, seed, st, qpool, eval_q, args, served_q=served_q)
            fp, prov = fp_for(name)
            r["fp"] = fp
            rows.append(r)
            print(f"[{done}/{total}] r={ratio} s={seed} {name:>14}: "
                  f"cost={r.get('query_cost', r['fixed_recall_cost']):.0f} "
                  f"work={r.get('work_calibrated', r['work']):.3g} rec={r['recall']:.3f} "
                  f"({time.perf_counter()-rt:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)  # written after every run
            write_sidecar(args.output, fp, prov)
        except Exception:  # noqa: BLE001
            print(f"[{done}/{total}] r={ratio} s={seed} {name}: ERROR", flush=True)
            traceback.print_exc()

    print(f"\nAll {total} runs done in {(time.perf_counter()-t0)/60:.0f} min. "
          f"Tidy CSV -> {args.output}", flush=True)
    aggregate_and_plot(pd.DataFrame(rows), args.figures_dir,
                       write_figures=not args.no_figures)


if __name__ == "__main__":
    main()

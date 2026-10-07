"""Run the experiments, every combination of dataset, seed and policy, into one result file.

Each run grows or reuses the starting index of its seed, runs one policy over the retention window
stream of `src/stream.py`, and writes one row with every reading, the work and the operation counts.
The file is written after every run, each row carries a fingerprint of the code, machine and
parameters, and `--resume` reruns only what is missing, so a long set of runs can be stopped and
restarted. The exact commands behind the reported results are in the README.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# pin the thread pools to one before numpy or scikit-learn are imported, since both read these
# variables only at import time. k-means over many vectors gives different answers at different
# thread counts, so without pinning the runs would not be reproducible
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

# the compared policies. each is a baseline or an implementation of a published method, except the
# bandit, which is the contribution of the thesis
MAINTAINERS = ["no_op", "global_rebuild", "lire_lite",
               "dedrift_lazy", "dedrift_split", "dedrift_hybrid",
               "cost_driven_quake", "bandit"]
DEDRIFT_VARIANTS = ["dedrift_lazy", "dedrift_split", "dedrift_hybrid"]
# variants that change one thing about a policy. bandit_linear is the bandit without the cost
# model, so its reward uses access times size. lire_lite_no_merge is LIRE with merging switched
# off. neither is in the reported results
ABLATION_VARIANTS = ["bandit_linear", "lire_lite_no_merge"]
# the declared tau sensitivity configuration, the Quake policy at 50 ns for split and delete alike.
# Quake stays at the paper value of 250 ns and is what the thesis reports as Quake. 50 ns is the
# value the Quake authors use for Quake in their released SIFT1M maintenance experiment, so it is
# taken from a source and not chosen by result
SENSITIVITY_VARIANTS = ["cost_driven_quake_tau50"]
QUAKE_TAU_ARM_NS = 50.0
_PROFILED: dict = {}


def profile_path(dim: int) -> Path:
    """Return the path of the stored scan cost curve for vectors of `dim` dimensions."""
    return COST_MODEL_DIR / f"profiled_lambda_dim{dim}.yaml"


def profiled_cost_model(dim: int, reprofile: bool = False):
    """Return the scan cost curve lambda(s) for `dim` dimensions, read once from its stored file.

    The stored curve is the Faiss price curve that scripts/calibrate_query_cost.py writes, the same
    one the scoring prices with. It is never measured here, because a curve measured again would
    differ, and every run must read the same one.
    """
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


DEDRIFT_N_LARGEST: int | None = None  # from --dedrift-n-largest, None uses the fraction


def build(name: str, seed: int, hi: int, lo: int, cost_model):
    """Create the policy called `name`, configured as in the reported runs.

    `hi` and `lo` are the split and merge thresholds the starting index was grown to satisfy. The
    bandit's minimum split size is `hi` over 4, the target average partition size.
    """
    if name == "no_op":
        return NoOpMaintainer()
    if name == "global_rebuild":
        return GlobalRebuildMaintainer(seed=seed)
    if name in ("lire_lite", "lire_lite_no_merge"):
        return LireLiteMaintainer(
            max_partition_size=hi, min_partition_size=lo,
            enable_merge=(name == "lire_lite"), seed=seed)
    if name.startswith("dedrift_"):
        # the number of largest clusters is a fraction of K, see dedrift.py
        return DeDriftMaintainer(variant=name.split("_", 1)[1], n_largest=DEDRIFT_N_LARGEST,
                                 seed=seed)
    uncap = 1_000_000  # no limit on actions per pass, every policy does what it decides
    if name == "cost_driven_quake":
        # Quake section 4.2.2, actions admitted by predicted cost change. tau is the published
        # value
        return CostDrivenQuakeMaintainer(
            cost_model=cost_model, max_actions_per_maintain=uncap,
            nprobe=DEFAULT_NPROBE, seed=seed)
    if name == "cost_driven_quake_tau50":
        # the declared sensitivity configuration, see SENSITIVITY_VARIANTS
        return CostDrivenQuakeMaintainer(
            cost_model=cost_model, max_actions_per_maintain=uncap,
            nprobe=DEFAULT_NPROBE, seed=seed, tau=QUAKE_TAU_ARM_NS, delete_tau=QUAKE_TAU_ARM_NS)
    if name in ("bandit", "bandit_linear"):
        # bandit_linear withholds the cost model, so its reward prices a partition as access times
        # size. no drift_threshold is passed, so the drift limit is DRIFT_FRACTION times the data
        # scale
        return BanditMaintainer(
            cost_model=None if name == "bandit_linear" else cost_model,
            max_partition_size=hi, min_partition_size=lo,
            # every partition is a candidate each pass, as Quake considers every partition, and
            # the learner discounts once per pass
            min_split_size=max(1, hi // 4), n_candidates=None, discount_mode="pass",
            # track_reversals only records how often a split or merge was undone, and changes no
            # decision
            coverage_penalty_weight=2.0, seed=seed, track_reversals=True)
    raise ValueError(name)


COST_AWARE = ("cost_driven_quake", "cost_driven_quake_tau50", "bandit")


def run_one_staple(base, name, seed, staple, qpool, eval_q, args, served_q=None) -> dict:
    """Run one policy on one seed over the retention window stream, and return its result row.

    Every policy of a seed runs from a copy of one starting index and sees the same stream.
    `eval_q` is the held out scoring sample and `served_q` a sample of the routed pool, see
    `src/stream.split_query_pools`. Every cost and work column is priced by `src/calibration.py`.
    """
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
        """The reported query cost of a reading, priced lists plus the ranking of all K centroids."""
        return query_cost(r[prefix + priced], r[prefix + "parts"], dim)

    # two delete models. in the reported one a deleted vector is removed at once, as in Quake and
    # Faiss, so the scoring reads live vectors, compaction gains nothing and its reads are left out
    # of the work, and every delete pays one copy. in the SPFresh model a deleted vector is scanned
    # until a compaction removes it, so compaction counts and deletes are free. both are measured
    # in every run, the second as a sensitivity check
    p = prices(dim)
    reads = (int(m._cumulative_work) - int(m._cumulative_distances)
             - int(m._cumulative_kmeans) - int(m._cumulative_bulk_kmeans))
    compaction = int(m._cumulative_compaction)
    work_cal = work_cost(reads - compaction, m._cumulative_distances, m._cumulative_kmeans,
                         m._cumulative_bulk_kmeans, dim)
    work_tomb = work_cost(reads, m._cumulative_distances, m._cumulative_kmeans,
                          m._cumulative_bulk_kmeans, dim)
    # the cost of taking in updates, charged to the total and not to maintenance work. an insert
    # compares against all K centroids, so it grows with what maintenance did to K
    routing = p["centroid_ratio"] * int(out["insert_routing"])
    ingest = routing + p["read_copy"] * int(out["deletes"])
    queries_served = (int(out["inserts"]) + int(out["deletes"])) * args.query_ratio
    q_mean = float(np.mean([qc(r) for r in rows[1:]]))
    q_stored_mean = float(np.mean([qc(r, priced="cost_priced_stored") for r in rows[1:]]))
    row = {
        "maintainer": name, "seed": seed,
        # the retention window has no insert to delete ratio, so the column is fixed at 1.0
        "ratio": 1.0, "workload": "staple", "protocol": "heldout",
        # the main result, calibrated query cost at recall 0.9 on held out queries at the end of
        # the run, every probed list and centroid comparison priced in Faiss scan distances
        "query_cost0": qc(s), "query_cost": qc(e), "query_cost_mean": q_mean,
        "served_query_cost": qc(e, "served_") if served_q is not None else float("nan"),
        # the raw readings the main result is built from, vectors scanned alone and with the
        # centroid comparisons
        "fixed_recall_cost0": s["cost"], "fixed_recall_cost": e["cost"],
        "fixed_recall_cost_mean": float(np.mean([r["cost"] for r in rows[1:]])),
        "fixed_recall_cost_k": e["cost"] + e["parts"],
        # the priced scan alone, before the centroid term, so the centroid price can be varied
        # afterwards
        "priced_scan": e["cost_priced"],
        "priced_scan_mean": float(np.mean([r["cost_priced"] for r in rows[1:]])),
        "recall": e["raw_recall"], "fixed_recall_achieved": e["achieved"],
        "fixed_recall_std": e["recall_std"],
        "nprobe": float(e["nprobe"]), "parts": int(e["parts"]), "live": int(e["live"]),
        "stored": int(e["stored"]),
        # work counted with every operation at weight one, and priced. reads and copies, scan
        # distances, small k-means and large k-means are counted apart and priced by the
        # calibration
        "work": int(m._cumulative_work), "work_calibrated": work_cal,
        "work_reads": reads, "distances": int(m._cumulative_distances),
        "kmeans_distances": int(m._cumulative_kmeans),
        "bulk_kmeans_distances": int(m._cumulative_bulk_kmeans),
        # what the whole workload cost, every query served at the mean query cost of the run plus
        # the maintenance and the updates, all in scan distances
        "queries_served": queries_served,
        "total_compute": queries_served * q_mean + work_cal + ingest,
        "compaction_reads": compaction, "insert_routing": int(out["insert_routing"]),
        # the nprobe the routed queries were served at, matched to recall 0.9 per window
        "route_nprobe_mean": float(np.mean(out["route_nprobe"])),
        "route_nprobe_end": int(out["route_nprobe"][-1]),
        "ingest": ingest,
        # the SPFresh delete model beside it, tombstones scanned and compactions counted
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
    # the same reading at recall 0.8 and 0.95, at the end of the run and as the mean over it, so a
    # ranking can be checked on either side of the 0.9 target
    for tag in EXTRA_TARGETS.values():
        priced = f"cost_priced_{tag}"
        row[f"query_cost_{tag}"] = qc(e, priced=priced)
        row[f"query_cost_{tag}_mean"] = float(np.mean([qc(r, priced=priced) for r in rows[1:]]))
        row[f"fixed_recall_cost_{tag}"] = e[f"cost_{tag}"]
        row[f"nprobe_{tag}"] = float(e[f"nprobe_{tag}"])
    # a fresh exact k-means over the final live vectors at the K this policy reached, priced like
    # the main result. the difference separates the gain from the K a policy reaches from the gain
    # from its partitions at that K. it can only be read here, since the final index is not kept
    rebuilt = cost_rebuilt_at_own_k(out["index"], eval_q, k=DEFAULT_K, target_recall=0.9,
                                    seed=seed)
    row["query_cost_rebuilt_k"] = query_cost(rebuilt["cost_priced"], rebuilt["parts"], dim)
    row["fixed_recall_cost_rebuilt_k"] = rebuilt["cost"]
    row["nprobe_rebuilt_k"] = float(rebuilt["nprobe"])
    if served_q is not None:
        row["served_fixed_recall_cost"] = e["served_cost"]
        row["served_nprobe"] = float(e["served_nprobe"])
    # the readings during the run, so a policy that is good midway and poor at the end shows
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
    """Print the mean and spread of cost and work per policy, and optionally plot them.

    The figure is written only when asked, because a run over part of the policies would
    otherwise overwrite the figure of the full set with a partial one.
    """
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
    # the starting index and stream parameters, the defaults of the reported runs
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
    # load the scan cost curve once up front, so every run prices actions with the same curve
    if any(n in COST_AWARE for n in args.maintainers):
        profiled_cost_model(base.shape[1], reprofile=args.reprofile_lambda)

    # the retention window has no ratio axis, see run_one_staple
    args.ratios = [1.0]
    # every run parameter that is not already a column, so two rows with the same fingerprint are
    # comparable and two with different ones are not, see src/provenance.py
    params = {"dataset": args.dataset, "workload": args.workload, "n_ops": args.n_ops,
              "cap": args.cap, "target_avg": args.target_avg, "bootstrap": args.bootstrap,
              "build_ratio": args.build_ratio, "window": args.window,
              "measure_ops": args.measure_ops, "max_t": args.max_t, "width": args.width,
              "eval_queries": args.eval_queries,
              "initial_fraction": args.initial_fraction, "query_ratio": args.query_ratio,
              "hot_fraction": args.hot_fraction, "hot_prob": args.hot_prob,
              "delete_csf": args.delete_csf, "dedrift_n_largest": args.dedrift_n_largest}
    # one fingerprint per policy, covering only the code that can affect its rows
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

    # one starting index per seed, grown once and copied for each policy, which is what makes
    # every policy of a seed see the same stream. the workload labels are stored on disk, the
    # starting index is kept in memory for the run
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

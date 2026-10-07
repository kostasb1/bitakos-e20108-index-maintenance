# Reproducing the results

This document gives every command behind the thesis, what it needs, how long it takes, what it writes, and what every column of a result file means.

## Contents

1. [Requirements](#requirements)
2. [Checking the stored results without running anything](#checking-the-stored-results-without-running-anything)
3. [The full pipeline](#the-full-pipeline)
4. [Running experiments with final_sweep.py](#running-experiments-with-final_sweeppy)
5. [Runtimes](#runtimes)
6. [Diagnostic scripts](#diagnostic-scripts)
7. [Output files](#output-files)
8. [Columns of a result row](#columns-of-a-result-row)
9. [Columns of the analysis tables](#columns-of-the-analysis-tables)
10. [Troubleshooting](#troubleshooting)

## Requirements

| Item | Value used for the results |
|---|---|
| Python | 3.12.9 |
| Libraries | as pinned in `requirements.txt` |
| Operating system | Windows 11, 64 bit |
| Threads | one, set by the scripts themselves |
| Memory | up to about 7.5 GB per GIST run |
| Disk | about 6 GB for the datasets |

```bash
pip install -r requirements.txt
python scripts/download_datasets.py --dataset all
```

Runs are deterministic, but only with the same library versions. A different scikit-learn can give a different k-means, which changes every number while leaving every conclusion to be checked again. Results produced on another machine carry a different fingerprint and are not mixed with the stored ones.

## Checking the stored results without running anything

Every number in Chapter 6 and Appendix I can be checked from the files in `results/` alone:

```bash
python scripts/analyse_final.py
python scripts/analyse_final.py --sweep results/raw/sens_ta50_2709 --out results/sens_ta50_2709 --control none
python scripts/make_figures.py
```

These take seconds and need neither the datasets nor Faiss timing. `analyse_final.py` prints the main table, the per seed values, the readings beside the main one, the SPFresh delete model, the equal K reading and the control, and writes the tables listed in [output files](#output-files).

## The full pipeline

To produce everything again, in order:

| Step | Command | Writes | Time |
|---|---|---|---|
| 1. Data | `python scripts/download_datasets.py --dataset all` | `data/sift/`, `data/gist/` | depends on the connection |
| 2. Prices | `python scripts/calibrate_query_cost.py` | `results/cost_model/query_cost_calibration.yaml`, `profiled_lambda_dim128.yaml`, `profiled_lambda_dim960.yaml` | not recorded |
| 3. Main runs | `python results/raw/final_2709/run_jobs.py` | `results/raw/final_2709/*.csv` | see [runtimes](#runtimes) |
| 4. Half size check | `python results/raw/sens_ta50_2709/run_jobs.py` | `results/raw/sens_ta50_2709/*.csv` | similar to the SIFT part of step 3 |
| 5. Control | `python scripts/control_rebuild_at_k.py --seeds 42 1 7 13 23` | `results/raw/control_2809/*.csv` | about five SIFT rebuild runs |
| 6. Tables | `python scripts/analyse_final.py` and the half size variant above | `results/final_2709/`, `results/sens_ta50_2709/` | seconds |
| 7. Diagnostics | see [diagnostic scripts](#diagnostic-scripts) | `results/raw/` | up to a few hours each |
| 8. Figures | `python scripts/make_figures.py` | `scripts/figures/` | seconds |

Step 2 measures time, so its output will differ slightly from the stored prices, and every later number will move with it. To reproduce the stored results exactly, skip step 2 and use the stored price files.

Step 3 runs the jobs listed in `results/raw/final_2709/jobs.txt`, six at a time and at most three GIST jobs at once, since a GIST process needs up to about 7.5 GB. Each job writes its own CSV and log and passes `--resume`, so an interrupted job reruns only its missing runs. Step 4 does the same with `--target-avg 50`.

## Running experiments with final_sweep.py

`scripts/final_sweep.py` runs every requested combination of seed and policy on one dataset into one CSV.

```bash
python scripts/final_sweep.py --dataset sift1m --seeds 42 1 --maintainers no_op lire_lite bandit --no-figures --output results/raw/my_runs.csv
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | `sift1m` | `sift1m`, `gist1m` or `synthetic` |
| `--seeds` | 42 1 7 | seeds to run |
| `--maintainers` | the eight policies | any of the names in [policies.md](policies.md#overview) |
| `--output` | `results/raw/final_sweep.csv` | the result file, written after every run |
| `--resume` | off | keep rows with a matching fingerprint and run only the rest |
| `--drop-stale` | off | with `--resume`, allow discarding rows with another fingerprint |
| `--dry-run` | off | print what would run and stop |
| `--no-figures` | off | skip the summary figure |
| `--cap` | 50,000 | live vectors in the starting index |
| `--target-avg` | 100 | target vectors per partition, sets the split limit to 4 times this |
| `--window` | 8 | split limit over merge limit |
| `--bootstrap` | 5,000 | vectors in the k-means start of the starting index |
| `--build-ratio` | 4.0 | inserts per delete while growing the starting index |
| `--measure-ops` | 120,000 | inserts in the stream |
| `--max-t` | 100 | time steps of the stream |
| `--width` | 3.0 | spread of arrivals around the current cluster |
| `--query-ratio` | 8 | routed queries per update |
| `--eval-queries` | 400 | held out queries per reading |
| `--hot-fraction` | 0.1 | share of queries in the concentrated region |
| `--hot-prob` | 0.8 | share of draws taken from the region |
| `--dedrift-n-largest` | none | a fixed number of largest clusters for DeDrift Split |

`--n-ops`, `--ratios`, `--initial-fraction` and `--delete-csf` are kept as recorded parameters and do not affect the retention window stream. `--synthetic` data has no price list, so it cannot produce the priced columns and is useful only for checking that the code runs.

A quick check that the installation works, a few minutes on SIFT, most of it growing the starting index:

```bash
python scripts/final_sweep.py --dataset sift1m --seeds 42 --maintainers lire_lite --measure-ops 3000 --max-t 6 --no-figures --output results/raw/check.csv
```

## Runtimes

Mean minutes per run, measured in the main runs with six runs going at once, so a run on its own is somewhat faster.

| Policy | SIFT1M | GIST1M |
|---|---|---|
| No maintenance | 5 | 24 |
| Global rebuild | 20 | 109 |
| LIRE | 4 | 21 |
| DeDrift Lazy | 4 | 23 |
| DeDrift Split | 4 | 26 |
| DeDrift Hybrid | 4 | 25 |
| Quake | 4 | 55 |
| Quake, τ = 50 ns | 5 | 129 |
| Bandit | 8 | 27 |

Most of the time goes to the readings, which compute exact ground truth and search for the recall matched nprobe. The longest single GIST run took about three and a half hours. Growing the starting index adds a few minutes per seed, once.

## Diagnostic scripts

| Script | Question | Command | Output |
|---|---|---|---|
| `control_rebuild_at_k.py` | how much of the bandit's lead is its partition count | `--seeds 42 1 7 13 23` | `results/raw/control_2809/` |
| `k_curve.py` | how query cost changes with K on fresh builds | `--dataset sift1m --seed 42`, also GIST seeds 42 and 1 | `results/raw/k_curve_*.csv` (stored copies in `diag_2809/`) |
| `rebuild_noise.py` | how precise one reading is | `--dataset sift1m --seed 42` | `results/raw/rebuild_noise_sift1m_s42.csv` |
| `dedrift_split_trace.py` | what each DeDrift Split pass does | `--dataset gist1m --seed 42` | `results/raw/diag_2809/` |
| `measure_split_alpha.py` | how much access a split child receives | `--dataset sift1m --seed 42` | `results/raw/split_alpha.csv` |

`dedrift_split_trace.py` checks at the end that its run reproduces the main result row of the same seed exactly, which confirms the trace is of the same run.

`measure_split_alpha.py` measures alpha on a k-means index of a random fifth of the dataset. The stored files `split_alpha_*.csv` are the measurements behind the value 0.86.

## Output files

**Raw results**, `results/raw/<set>/`:

| File | Content |
|---|---|
| `<tag>.csv` | one row per run, see the columns below |
| `<tag>.csv.provenance.json` | for each fingerprint in the file, what it was computed from |
| `jobs.txt`, `run_jobs.py` | the job list and the runner of the set |

`final_2709` holds the main runs, `sens_ta50_2709` the half size check, `control_2809` the control, `diag_2809` the K curves and the DeDrift trace.

**Analysis tables**, `results/final_2709/` and `results/sens_ta50_2709/`:

| File | Content |
|---|---|
| `ranking_<dataset>.csv` | per policy, means over seeds and the paired comparisons against no maintenance |
| `wins_<dataset>.csv` | for each pair, the number of seeds on which the row policy is cheaper |
| `tombstone_<dataset>.csv` | the ranking under the SPFresh delete model |
| `equal_k_<dataset>.csv` | each policy against a fresh build at its own K |
| `control_<dataset>.csv` | the control against the bandit |
| `centroid_sensitivity.csv` | rankings at the calibrated centroid price and at 1 |
| `frontier.pdf`, `frontier.png` | query cost against work |

## Columns of a result row

Written by `final_sweep.run_one_staple`. Costs are in scan distances per query, work in scan distances over the run, unless stated otherwise. "End" means the reading at the end of the run.

**Identity**

| Column | Meaning |
|---|---|
| `maintainer` | the policy name |
| `seed` | the seed |
| `ratio` | always 1.0, kept for compatibility |
| `workload`, `protocol` | always `staple` and `heldout` |
| `fp` | the fingerprint, see [measurement.md](measurement.md#fingerprints) |

**Query cost, the main readings**

| Column | Meaning |
|---|---|
| `query_cost` | **the main result**, priced query cost at recall 0.9 on held out queries, at the end |
| `query_cost0` | the same at the start |
| `query_cost_mean` | the mean over the readings at 25, 50, 75 and 100 |
| `query_cost_t25`, `_t50`, `_t75` | the readings during the run |
| `served_query_cost` | at the end, on the served sample of routed queries |
| `query_cost_r80`, `query_cost_r95` | at recall 0.8 and 0.95, at the end |
| `query_cost_r80_mean`, `query_cost_r95_mean` | the same, mean over the run |
| `query_cost_rebuilt_k` | a fresh k-means of the final live vectors at the same K |
| `query_cost_stored`, `query_cost_stored_mean` | the SPFresh delete model |

**Raw readings**

| Column | Meaning |
|---|---|
| `fixed_recall_cost`, `fixed_recall_cost0`, `fixed_recall_cost_mean` | vectors scanned per query at recall 0.9 |
| `fixed_recall_cost_k` | the same plus K, each centroid counted as one distance |
| `fixed_recall_cost_r80`, `_r95`, `_stored`, `_rebuilt_k` | the raw count for the other readings |
| `cost_t25`, `_t50`, `_t75` | the raw count during the run |
| `priced_scan`, `priced_scan_mean` | the priced list scan before the centroid term |
| `served_fixed_recall_cost` | the raw count on the served sample |
| `nprobe`, `nprobe_r80`, `nprobe_r95`, `nprobe_rebuilt_k`, `served_nprobe`, `nprobe_t*` | the interpolated nprobe of each reading |
| `recall`, `recall_t*` | recall at a fixed nprobe of 10 |
| `fixed_recall_achieved`, `fixed_recall_std` | the recall reached at the target and its spread over queries |
| `parts`, `parts_t*` | partition count, K |
| `live`, `stored` | live and stored vectors at the end |

**Work**

| Column | Meaning |
|---|---|
| `work_calibrated` | **reported maintenance work**, priced, compaction left out |
| `work_calibrated_tombstone` | the same with compaction included, the SPFresh model |
| `work` | all counts added at weight one |
| `work_reads` | vectors read or copied |
| `distances`, `kmeans_distances`, `bulk_kmeans_distances` | the distance counts by kind |
| `compaction_reads` | vectors read by compaction |
| `insert_routing` | centroid comparisons made by inserts |
| `ingest` | the priced cost of taking in updates |
| `queries_served` | routed queries over the run, 8 per update |
| `total_compute`, `total_compute_tombstone` | queries served times mean query cost, plus work, plus ingestion |
| `route_nprobe_mean`, `route_nprobe_end` | the nprobe the routed queries were served at |

**Actions**

| Column | Meaning |
|---|---|
| `inserts`, `deletes` | updates in the stream |
| `splits`, `merges` | completed splits and merges, Quake deletes counted as merges |
| `declined`, `merges_declined` | splits and deletes rejected by an exact check |
| `reassigned` | single vectors moved between partitions |
| `repartitioned` | vectors moved by a rebuild or DeDrift Split |
| `collected` | empty partitions removed |
| `centroids` | centroid recomputations |
| `examined` | vectors checked by the LIRE conditions |
| `arm_split`, `arm_centroid_update`, `arm_merge`, `arm_compact`, `arm_no_op` | bandit only, how often each action was taken |
| `reversals` | bandit only, splits or merges undone soon after |

## Columns of the analysis tables

`ranking_<dataset>.csv` has one row per policy with the mean over seeds of the main columns, and for each of `query_cost`, `query_cost_mean`, `query_cost_r80`, `query_cost_r95` and `query_cost_rebuilt_k`:

| Column | Meaning |
|---|---|
| `<col>_vs_no_op` | mean change against no maintenance, percent, from the mean log ratio |
| `<col>_ci_lo`, `<col>_ci_hi` | the 95% paired t interval, percent |
| `<col>_below` | seeds on which the policy is cheaper than no maintenance |
| `<col>_seeds` | seeds compared |
| `<col>_s<seed>` | the change on each seed, percent |

Also `efficiency`, `break_even_reads`, `total_vs_no_op`, see [measurement.md](measurement.md#totals-efficiency-and-break-even).

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `profiled_lambda_dim128.yaml is missing` | the cost curves are not in `results/cost_model/` | restore them, or run `python scripts/calibrate_query_cost.py --lambda-only` to rebuild them from the stored prices |
| `rows ... do not carry their maintainer's current fingerprint` | the output file holds rows from other code, another machine or other parameters | write to a new file, or pass `--drop-stale` to discard them |
| `a strategy carries two fingerprints` | a result folder mixes code versions | rerun the affected runs into a clean folder |
| `the control's K per seed ... is not the bandit's` | the control was run against another set of runs | point `--control` at the matching control, or at `none` |
| `MemoryError` on GIST | too many GIST runs at once | run at most three in parallel |

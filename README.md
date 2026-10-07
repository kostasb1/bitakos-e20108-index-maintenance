# Index maintenance for vector data

The code of the bachelor thesis *Τεχνικές Συντήρησης Ευρετηρίων για Διανυσματικά Δεδομένα*, Konstantinos Bitakos, Department of Digital Systems, University of Piraeus, 2026.

The thesis compares ways of maintaining an IVF (inverted file) vector index while vectors are inserted and deleted. The code includes an IVF index written in Python and NumPy, the maintenance policies of SPFresh, DeDrift and Quake, a periodic full rebuild, a contextual bandit policy, and a scoring procedure. The scoring compares every policy at a fixed recall of 0.9, on queries no policy has seen, with query cost and maintenance work priced in one unit measured on Faiss.

The repository contains the code and the result files behind every number in the thesis. It is a snapshot for examination and will not be developed further.

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | How the code is organised, one run end to end, the data structures, routing, work accounting, determinism |
| [docs/policies.md](docs/policies.md) | Every maintenance policy in detail, its parameters, their sources and the differences from the papers |
| [docs/measurement.md](docs/measurement.md) | The scoring, the Faiss prices, query cost, work, the delete models, the statistics, the fingerprints |
| [docs/reproducing.md](docs/reproducing.md) | Every command, runtimes, output files and every column of a result file |
| [docs/glossary.md](docs/glossary.md) | The terms used in the code, with their sources |
| [docs/reference.md](docs/reference.md) | Every file, class and function, with its signature and explanation, generated from the code |

## Installation

The results were produced with Python 3.12.9 on Windows, on one thread.

```bash
python -m venv .venv
.venv\Scripts\activate          # on Linux or macOS, source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the library versions that were used. A different version of scikit-learn can produce a different k-means, and then the numbers will differ from the stored results, even though every run is deterministic given its seed.

## Data

```bash
python scripts/download_datasets.py --dataset all
```

This downloads SIFT1M (about 0.5 GB) and GIST1M (about 5.5 GB) into `data/`. The first run of each seed also computes the workload cluster labels and stores them in `data/workload_labels/`.

## Reproducing the results

Every reported number comes from the files under `results/`, so the tables can be checked without running anything. To run the experiments again:

| Result in the thesis | Command | Output |
|---|---|---|
| Prices of the cost unit (Section 5.3, Table 5.1) and the scan cost curves | `python scripts/calibrate_query_cost.py` | `results/cost_model/` |
| Main results, nine policies, SIFT1M and GIST1M, five seeds | `python results/raw/final_2709/run_jobs.py` | `results/raw/final_2709/` |
| Half size check, target of 50 vectors per partition, SIFT1M | `python results/raw/sens_ta50_2709/run_jobs.py` | `results/raw/sens_ta50_2709/` |
| Control run, global rebuild at the bandit's partition count | `python scripts/control_rebuild_at_k.py --seeds 42 1 7 13 23` | `results/raw/control_2809/` |
| Tables of Chapter 6 and Appendix I | `python scripts/analyse_final.py` | `results/final_2709/` |
| The same tables for the half size check | `python scripts/analyse_final.py --sweep results/raw/sens_ta50_2709 --out results/sens_ta50_2709 --control none` | `results/sens_ta50_2709/` |
| K curve (Section 6.5) | `python scripts/k_curve.py --dataset sift1m --seed 42`, and GIST1M seeds 42 and 1 | `results/raw/k_curve_*.csv` |
| Spread of one reading (Section 6.2) | `python scripts/rebuild_noise.py --dataset sift1m --seed 42` | `results/raw/rebuild_noise_*.csv` |
| DeDrift Split per pass (Section 6.8) | `python scripts/dedrift_split_trace.py --dataset gist1m --seed 42` | `results/raw/diag_2809/` |
| Measured alpha of 0.86 (Section 4.6) | `python scripts/measure_split_alpha.py --dataset sift1m` | `results/raw/split_alpha_*.csv` |
| Figures of the thesis | `python scripts/make_figures.py` | `scripts/figures/` |

The stored K curves are in `results/raw/diag_2809/`, and the spread and alpha files directly in `results/raw/`. A single run can also be started directly, for example:

```bash
python scripts/final_sweep.py --dataset sift1m --seeds 42 --maintainers no_op lire_lite bandit --no-figures --output results/raw/example.csv
```

A GIST1M run of one policy on one seed takes from minutes to a few hours, depending on the policy. Each result row carries a fingerprint of the code, the machine and the parameters, and `--resume` reruns only the runs that are missing or were produced by different code.

## Layout

```
src/
  vector.py           the partition scan, distances from one query to a block of vectors
  partition.py        one partition, its vectors, tombstones, centroid and access count
  index.py            the IVF index, routing, inserts, deletes, splits, merges, reassignment
  maintainers/
    base.py           the policy interface and the shared restructuring operations
    no_op.py          no maintenance, the reference
    global_rebuild.py periodic full k-means rebuild, the Ada-IVF baseline
    lire_lite.py      LIRE, the SPFresh protocol
    dedrift.py        DeDrift Lazy, Split and Hybrid
    cost_driven_quake.py   the Quake maintenance loop
    bandit.py         the contextual bandit policy
  bandit/             context features, the LinUCB learner and the reward of the bandit
  staple.py           growing the starting index under LIRE
  stream.py           the retention window stream, query routing and readings
  metrics.py          ground truth, the fixed recall search and the scored cost
  calibration.py      the measured Faiss prices
  cost_model.py       the scan cost curve lambda(s) the Quake and bandit policies decide with
  provenance.py       the fingerprint of every result row
  config.py, types.py, dataset.py
scripts/              the experiments, the analysis and the figures, see the table above
results/              the result files and the measured prices
```

Every module and function has a docstring that explains what it does and why.

## Policies not included

Three policies from earlier stages of the work are not part of the thesis results and their code is left out. `cost_driven_old` combined Quake's cost model with a trigger of our own and was replaced by the faithful Quake port. `drift_aware` and `hybrid` were policies of our own whose parameters had no published source, so they were withdrawn rather than reported.

## License

The code is released under the MIT License, see [LICENSE](LICENSE). MIT is the most common license for academic and research code. It lets anyone read, run and build on the code, provided the copyright notice is kept, which is what research code shared for examination and reuse needs.

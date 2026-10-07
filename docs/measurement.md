# Measurement

This document describes how a policy is scored: what one reading measures, how query cost and maintenance work are priced, how the results are compared across seeds, and how every row is tied to the code that produced it. The modules are `src/metrics.py`, `src/calibration.py`, `src/stream.py` (function `measure`), `scripts/analyse_final.py` and `src/provenance.py`.

## Contents

1. [Three meanings of cost](#three-meanings-of-cost)
2. [Queries: routed, served and held out](#queries-routed-served-and-held-out)
3. [One reading](#one-reading)
4. [Ground truth over live vectors](#ground-truth-over-live-vectors)
5. [The fixed recall search](#the-fixed-recall-search)
6. [The cost unit and the Faiss prices](#the-cost-unit-and-the-faiss-prices)
7. [Query cost](#query-cost)
8. [Maintenance work](#maintenance-work)
9. [Two delete models](#two-delete-models)
10. [Readings beside the main one](#readings-beside-the-main-one)
11. [Totals, efficiency and break even](#totals-efficiency-and-break-even)
12. [Statistics](#statistics)
13. [Lambda, the policy side curve](#lambda-the-policy-side-curve)
14. [Fingerprints](#fingerprints)

## Three meanings of cost

The word cost names three different things in this code, and keeping them apart is a design rule.

| Name | Computed in | Read by | Purpose |
|---|---|---|---|
| Scored cost | `metrics.py`, `calibration.py` | the scoring only | the outcome every policy is judged by |
| Policy cost | `cost_model.py`, inside Quake and the bandit | Quake and the bandit | how those policies choose actions |
| Work | `MaintenanceReport` counters | the result row | what running maintenance cost |

No policy imports `metrics.py` or `calibration.py`. Policy cost and scored cost share one price curve, so a policy decides in the unit it is scored in, but a policy can still be wrong about what its actions will do, and the scored cost is the independent check.

## Queries: routed, served and held out

`stream.split_query_pools` divides the query set of the dataset (10,000 queries on SIFT, 1,000 on GIST).

1. **Region.** One query, drawn from the seed, is the centre. The 10% of queries nearest to it form the region (`query_region`). Quake section 7.1 and Ada-IVF section 5.1.2 both concentrate their skewed query workloads in space in this way.
2. **Halves.** The query ids are split in two at random, the region and the rest alike.
3. **Routed pool.** 4,000 draws from the first half, 80% from its part of the region and 20% from the whole half. The stream routes only these.
4. **Served sample.** 400 queries taken from the routed pool.
5. **Held out sample.** 400 draws from the second half, mixed the same way. The stream never routes these, so no policy is scored on a query it learned from.

The main result uses the held out sample. The served sample is read beside it, so the gap between the two shows how much a policy adapted to the queries it saw.

## One reading

`stream.measure` is called at time steps 0, 25, 50, 75 and 100. Each reading returns:

| Key | Meaning |
|---|---|
| `cost` | vectors scanned per query at the nprobe where recall reaches 0.9 |
| `nprobe` | that nprobe, interpolated, so fractional |
| `achieved`, `recall_std` | the recall reached and its spread over queries |
| `cost_r80`, `cost_r95` | the same at recall 0.8 and 0.95 |
| `cost_stored` | vectors scanned with tombstones counted, the SPFresh delete model |
| `cost_priced` | the probed lists priced in Faiss scan distances, at the same nprobe |
| `cost_priced_stored`, `cost_priced_r80`, `cost_priced_r95` | the same pricing for the other readings |
| `raw_recall` | recall at a fixed nprobe of 10 |
| `parts`, `live`, `stored` | partition count, live and stored vectors |

## Ground truth over live vectors

`metrics.compute_live_ground_truth`. The published ground truth of SIFT1M and GIST1M refers to the full dataset. After inserts and deletes it would count deleted vectors as correct answers and miss vectors that were never inserted, mixing the quality of the index with the content of the stream. Every reading therefore recomputes the exact 10 nearest live vectors of each query, by exhaustive comparison with a Faiss flat index (`IndexFlatL2`), or with scikit-learn when Faiss is not installed.

Recall at 10 of one query is the share of its 10 true neighbours that the search returned.

## The fixed recall search

A cost read at a fixed nprobe can be gamed. Splitting partitions makes each probe cheaper while recall quietly falls, and nothing would record the loss. So every reading holds recall fixed and asks what reaching it costs.

`metrics._cost_at_target_every_integer`, used by `costs_at_target_recalls`:

1. **Double** nprobe from 1 (1, 2, 4, 8, …) until mean recall reaches the target.
2. **Bisect** inside the last doubling until two neighbouring values n and n + 1 bracket the target, recall below it at n and at or above it at n + 1.
3. **Interpolate** linearly between n and n + 1 to the point where recall equals the target exactly, and report the cost and the nprobe there.

Bisection is valid because recall never falls as nprobe rises: the partitions probed at n are a subset of those probed at n + 1. Over 44,400 per query steps on both datasets, none was observed to fall. Each recall pass is measured once and reused, so the readings at 0.8 and 0.95 cost only a few extra passes on the same ground truth.

If the target cannot be reached even with every partition probed, that point is reported.

## The cost unit and the Faiss prices

The Python index is three to four times slower per vector than Faiss, and much of each call is interpreter overhead, so its own timings would price every operation wrongly. All prices are instead measured once on Faiss `IndexIVFFlat`, single threaded, one query at a time, by `scripts/calibrate_query_cost.py`, on an index of 51,200 vectors, the size of the experiments. Each price is the median of five calibrations, stored with its range in `results/cost_model/query_cost_calibration.yaml`.

**The unit.** One scan distance is the time one more vector adds to the scan of a long list: 11.94 ns on SIFT1M and 119.72 ns on GIST1M. Every other price is a multiple of it.

**List prices.** The time to open and scan one list of each length, in scan distances.

| List length | 0 | 5 | 10 | 25 | 50 | 100 | 200 | 400 | 800 | 1,600 |
|---|---|---|---|---|---|---|---|---|---|---|
| SIFT1M | 0.6 | 10.5 | 15.3 | 32.9 | 49.3 | 89.3 | 165.9 | 379.1 | 768.8 | 1,569.9 |
| GIST1M | 0.0 | 4.1 | 9.1 | 25.8 | 49.9 | 101.9 | 205.7 | 410.4 | 806.8 | 1,608.0 |

On SIFT a short list costs more per vector than a long one, a list of 5 about 2 scan distances per vector against about 1 for long lists. A single price per probed list, a constant added to the vector count, does not fit this. Fitted as a straight line, the per list constant crosses zero across calibrations on both datasets. So each list is priced at its own length on the curve, by linear interpolation between the measured lengths, extended along the last segment above 1,600 (`calibration.list_price`).

**Other prices**, in scan distances:

| Operation | SIFT1M | GIST1M | Used for |
|---|---|---|---|
| one centroid comparison | 0.757 | 0.395 | ranking the K centroids, every query |
| one small k-means distance | 3.372 | 1.446 | splits and DeDrift Split, a few centres over a few hundred points |
| one bulk k-means distance | 0.230 | 0.147 | the global rebuild, hundreds of centres over the whole index |
| one vector read or copy | 2.587 | 2.470 | merges, moves, refreshes, compaction |

A k-means distance inside a large batched k-means is far cheaper than a scan distance, while one in a tiny k-means is dearer, which is why the two are priced apart.

## Query cost

`calibration.query_cost`. The reported cost of one query at recall 0.9:

```
query cost = Σ over probed lists of list_price(list length)  +  centroid price × K
```

Every query compares itself with all K centroids and then opens and scans nprobe lists. Counting scanned vectors alone would charge nothing for the centroid ranking or for opening a list, and would favour a policy that fragments the index. The priced scan is interpolated on the same nprobe bracket as the count of scanned vectors, so both describe the same operating point (`metrics.priced_cost_at`).

The headline reported in the thesis is this query cost at the end of the run, on the held out queries.

## Maintenance work

`calibration.work_cost`, applied in `final_sweep.run_one_staple`:

```
work = distances
     + kmeans_small price × small k-means distances
     + kmeans_bulk price × bulk k-means distances
     + read_copy price × (vectors read or copied − compaction reads)
```

The counts come from the policy's running totals, see [architecture.md](architecture.md#work-accounting). Compaction reads are subtracted because under the reported delete model compaction gains nothing, see below.

**Ingestion** is charged to the total but not to maintenance work. Every insert compares the new vector with all K centroids, priced as centroid comparisons, so a policy that raises K also raises the cost of every later insert. Every delete is charged one copy.

## Two delete models

| | Reported model | SPFresh model |
|---|---|---|
| A deleted vector | is gone at once, as in Quake and Faiss | stays in its list until compaction |
| A query scans | live vectors | live and dead vectors |
| Compaction | gains nothing, its reads are left out of work | shortens later scans, its reads count as work |
| Columns | `query_cost`, `work_calibrated`, `total_compute` | `query_cost_stored`, `work_calibrated_tombstone`, `total_compute_tombstone` |

Recall is the same under both, since a dead vector is filtered after it is read, so both use the same nprobe and differ by the dead vectors alone.

## Readings beside the main one

| Reading | Column | Question it answers |
|---|---|---|
| mean over the run | `query_cost_mean` | is a policy good throughout, or only at the end |
| recall 0.8 and 0.95 | `query_cost_r80`, `query_cost_r95` | does the ranking hold either side of the target |
| served queries | `served_query_cost` | how much did a policy adapt to the queries it saw |
| fresh build at own K | `query_cost_rebuilt_k` | how much of the gain is the K reached rather than the maintenance |
| centroid price of 1 | computed in `analyse_final.py` | does the ranking depend on the centroid price |
| checkpoints | `query_cost_t25`, `_t50`, `_t75` | the trajectory |

**Fresh build at own K** (`metrics.cost_rebuilt_at_own_k`). At the end of a run the live vectors are partitioned again with exact k-means at the same partition count and scored the same way. Different policies reach very different K, from about 79 to more than 2,000, and K alone changes the query cost a great deal. The policy's own cost minus this one is what a rebuild at that K would still have saved.

## Totals, efficiency and break even

```
queries served = (inserts + deletes) × 8
total compute  = queries served × mean query cost  +  work  +  ingestion
```

Computed in `scripts/analyse_final.py`, against no maintenance on the same seed:

```
saving per query = mean query cost of no maintenance − mean query cost of the policy
efficiency       = saving per query × queries served / work
break even reads = work / ((inserts + deletes) × saving per query)
```

Break even reads is how many queries per update it takes before a policy's saving repays its work. It is infinite when there is no saving, and missing when no row for no maintenance exists for that seed.

## Statistics

`scripts/analyse_final.py`. The rules were fixed before the main runs.

1. **Pair by seed.** Each policy is compared with no maintenance on the same dataset and seed, so the level each seed sets cancels.
2. **Log ratio.** For each seed, log(policy cost / no maintenance cost). A saving and a loss by the same factor are then symmetric.
3. **Interval.** The mean log ratio over the five seeds with a paired t interval, t at 97.5% with 4 degrees of freedom, both turned back into percent change.
4. **Count.** The number of seeds below no maintenance is reported beside it, along with every per seed value. Five of five corresponds to an exact two sided sign test p of 0.0625, so it is never presented as significant on its own.
5. **Order.** Rankings are ordered by the paired ratio, never by mean absolute cost, which a seed where every method is expensive would dominate.
6. **No pooling.** Datasets are never combined.

The resolution of a reading is limited by the k-means build itself. Building the same live vectors in different row orders gives query costs with a coefficient of variation of about 6% on the 400 held out queries, about 4.7 points from the build and about 3 from the query sample (`scripts/rebuild_noise.py`). Differences below about 10% on SIFT and 20% on GIST cannot be resolved with five seeds.

## Lambda, the policy side curve

`src/cost_model.py`. Quake and the bandit price candidate actions with λ(s), the cost of scanning a partition of s vectors, and λ_c(K), the cost of ranking K centroids. The stored curves, `results/cost_model/profiled_lambda_dim128.yaml` and `..._dim960.yaml`, are the Faiss list prices above converted to nanoseconds by `lambda_from_prices`, with the centroid price per centroid. Between measured sizes λ is interpolated linearly, above the largest it is extended along the last segment, and below the smallest it is scaled down linearly.

The curves are inputs, not outputs. `final_sweep.py` refuses to measure them again, since a new timing would differ and would change every decision.

## Fingerprints

`src/provenance.py`. Every result row has an `fp` column, a 16 character hash, and every result file has a `.provenance.json` file beside it with the details.

The fingerprint of a row covers:

| Part | Content |
|---|---|
| code | the syntax trees of every file under `src/` and of `final_sweep.py`, docstrings removed, so comments never change it |
| own code only | another policy's own module is left out, so editing the bandit changes only bandit rows |
| prices | `query_cost_calibration.yaml` |
| cost curves | the lambda files, for Quake and the bandit only |
| machine | computer name, processor architecture, Python, numpy, scikit-learn and Faiss versions |
| threads | the three thread variables |
| parameters | every run parameter that is not a column |

Git fields are recorded in the provenance file for people to read but are not part of the fingerprint, so an identical tree has the same fingerprint before and after a commit.

With `--resume`, `final_sweep.py` keeps rows whose fingerprint matches the current code and runs only the rest. A row with a different fingerprint stops the run unless `--drop-stale` is given, so two code versions are never mixed in one file. `analyse_final.py` refuses a file in which one policy carries two fingerprints.

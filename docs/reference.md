# API reference

Every module, class and function of the code, generated from the docstrings, in the order the code is layered. For how the parts work together, see [architecture.md](architecture.md). Names starting with an underscore are internal to their module.

## Contents

- **Core**: [`src/vector.py`](#srcvectorpy), [`src/partition.py`](#srcpartitionpy), [`src/index.py`](#srcindexpy), [`src/types.py`](#srctypespy), [`src/config.py`](#srcconfigpy)
- **Shared**: [`src/maintainers/base.py`](#srcmaintainersbasepy), [`src/cost_model.py`](#srccost_modelpy), [`src/calibration.py`](#srccalibrationpy), [`src/dataset.py`](#srcdatasetpy)
- **Policies**: [`src/maintainers/no_op.py`](#srcmaintainersno_oppy), [`src/maintainers/global_rebuild.py`](#srcmaintainersglobal_rebuildpy), [`src/maintainers/lire_lite.py`](#srcmaintainerslire_litepy), [`src/maintainers/dedrift.py`](#srcmaintainersdedriftpy), [`src/maintainers/cost_driven_quake.py`](#srcmaintainerscost_driven_quakepy), [`src/maintainers/bandit.py`](#srcmaintainersbanditpy), [`src/bandit/context.py`](#srcbanditcontextpy), [`src/bandit/linucb.py`](#srcbanditlinucbpy), [`src/bandit/reward.py`](#srcbanditrewardpy)
- **Measurement**: [`src/metrics.py`](#srcmetricspy)
- **Workload**: [`src/staple.py`](#srcstaplepy), [`src/stream.py`](#srcstreampy)
- **Support**: [`src/provenance.py`](#srcprovenancepy), [`src/__init__.py`](#src__init__py), [`src/bandit/__init__.py`](#srcbandit__init__py), [`src/maintainers/__init__.py`](#srcmaintainers__init__py)
- **Scripts**: [`scripts/final_sweep.py`](#scriptsfinal_sweeppy), [`scripts/analyse_final.py`](#scriptsanalyse_finalpy), [`scripts/calibrate_query_cost.py`](#scriptscalibrate_query_costpy), [`scripts/control_rebuild_at_k.py`](#scriptscontrol_rebuild_at_kpy), [`scripts/k_curve.py`](#scriptsk_curvepy), [`scripts/rebuild_noise.py`](#scriptsrebuild_noisepy), [`scripts/dedrift_split_trace.py`](#scriptsdedrift_split_tracepy), [`scripts/measure_split_alpha.py`](#scriptsmeasure_split_alphapy), [`scripts/make_figures.py`](#scriptsmake_figurespy), [`scripts/download_datasets.py`](#scriptsdownload_datasetspy)

318 classes and functions in 35 files.

## Core

### src/vector.py

113 lines.

Distance arithmetic, and the partition scan that every query runs.

`l2_distance_batch` is the most frequently called function in the project. It computes the
distances from one query to every vector of one partition. It reuses scratch buffers, so a call
allocates only the array it returns, and its output is bit identical to
`np.linalg.norm(matrix - query, axis=1)`.

| Constant | Value |
|---|---|
| `_SCAN_BUFFERS` | `{}` |
| `_SCAN_MIN_ROWS` | `1024` |

#### `l2_distance`

*function, lines 14 to 16*

```python
def l2_distance(a: np.ndarray, b: np.ndarray) -> float
```

Return the Euclidean distance between two vectors.

#### `l2_distance_squared`

*function, lines 19 to 26*

```python
def l2_distance_squared(a: np.ndarray, b: np.ndarray) -> float
```

Return the squared Euclidean distance between two vectors.

Ranking by squared distance gives the same order as ranking by distance, so callers that only
compare distances use this and skip the square root.

#### `_scan_buffers`

*function, lines 36 to 50*

```python
def _scan_buffers(rows: int, dim: int, dtype: np.dtype)
```

Return three scratch arrays sized for a partition of `rows` vectors.

The first two hold the differences and their squares, the third the per vector sums. A
buffer is replaced only when a larger partition arrives.

#### `l2_distance_batch`

*function, lines 53 to 79*

```python
def l2_distance_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray
```

Return the Euclidean distance from `query` to every row of `matrix`.

This is the partition scan. It performs the same operations in the same order as
`np.linalg.norm(matrix - query, axis=1)`, subtract, square, sum and square root, so the
result is bit identical to it. The difference is that the two intermediate arrays come from
reused buffers instead of being allocated on every call.

#### `l2_distance_matrix`

*function, lines 82 to 86*

```python
def l2_distance_matrix(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray
```

Return the matrix of Euclidean distances between every query and every row of `matrix`.

#### `nearest_k_indices`

*function, lines 89 to 98*

```python
def nearest_k_indices(query: np.ndarray, matrix: np.ndarray, k: int) -> np.ndarray
```

Return the row indices of the `k` rows nearest to `query`, nearest first.

#### `compute_centroid`

*function, lines 101 to 105*

```python
def compute_centroid(vectors: np.ndarray) -> np.ndarray
```

Return the mean of a non empty set of vectors.

#### `normalize`

*function, lines 108 to 113*

```python
def normalize(v: np.ndarray) -> np.ndarray
```

Return `v` scaled to unit length, or a copy of `v` if its length is zero.

### src/partition.py

251 lines.

One IVF partition, its vectors, its centroid and its access count.

A delete marks the vector with a tombstone and returns at once. The vector stays in storage until
`compact` rewrites the partition, so `size()` counts live vectors and `total_size()` counts stored
ones. Every size rule reads `size()`, except two in LIRE, the split trigger and the capacity check
for a merge partner, which read the stored length as SPFresh does.

`get_live_vectors()` returns `(ids, vectors)` as an independent copy that a caller may keep.
`_live_block()` returns `(vectors, ids)`, in the opposite order, possibly as a view into the
storage, and is meant for the scan only.

#### `Partition`

*class, lines 21 to 251*

```python
class Partition
```

A partition of the index, its stored vectors, tombstones, centroid and access count.

#### `Partition.__init__`

*method, lines 24 to 48*

```python
def __init__(self, partition_id: int, centroid: np.ndarray) -> None
```

Create an empty partition with the given id and centroid.

#### `Partition.size`

*method, lines 50 to 52*

```python
def size(self) -> int
```

Return the number of live vectors, stored vectors minus tombstones.

#### `Partition.total_size`

*method, lines 54 to 56*

```python
def total_size(self) -> int
```

Return the number of stored vectors, tombstoned ones included.

#### `Partition.is_empty`

*method, lines 58 to 60*

```python
def is_empty(self) -> bool
```

Return True if the partition holds no live vector.

#### `Partition._ensure_capacity`

*method, lines 62 to 73*

```python
def _ensure_capacity(self, required: int) -> None
```

Grow the storage block so it holds at least `required` rows.

The capacity doubles each time, as in Quake, so a series of appends costs constant time
per append on average.

#### `Partition._invalidate_live`

*method, lines 75 to 78*

```python
def _invalidate_live(self) -> None
```

Forget the cached live rows and ids, after any change to the partition.

#### `Partition.add`

*method, lines 80 to 97*

```python
def add(self, vec_id: int, vec: np.ndarray) -> None
```

Store a vector in the partition.

If the id is present as a tombstone, the vector is written over the old row and revived,
otherwise it is appended.

#### `Partition.mark_deleted`

*method, lines 99 to 117*

```python
def mark_deleted(self, vec_id: int) -> None
```

Delete a vector by marking it with a tombstone.

The delete takes constant time and moves nothing, so it is charged no maintenance work. The
vector stays stored until `compact` rewrites the partition, and that compaction pays the
cost, proportional to the stored length. A policy that never compacts never pays, but it
keeps the dead vectors in storage until the run ends, and stored minus live size is
reported for every run so the debt stays visible.

This follows SPFresh, which leaves a deleted vector in its posting until a split or merge
rewrites it. Quake removes a vector at once instead. The reported query cost reads live
vectors only, which is the Quake and Faiss model, and the SPFresh model, in which
tombstones are scanned, is measured beside it, see `metrics.stored_cost_at`.

#### `Partition.is_deleted`

*method, lines 119 to 121*

```python
def is_deleted(self, vec_id: int) -> bool
```

Return True if the vector is marked with a tombstone.

#### `Partition.compact`

*method, lines 123 to 141*

```python
def compact(self) -> int
```

Rewrite the storage with the live vectors only, and return how many were removed.

#### `Partition.true_mean`

*method, lines 143 to 158*

```python
def true_mean(self) -> np.ndarray
```

Return the mean of the live vectors, or the centroid if there are none.

The mean is cached until the partition changes.

#### `Partition.centroid_drift`

*method, lines 160 to 167*

```python
def centroid_drift(self) -> float
```

Return the distance between the centroid and the mean of the live vectors.

An empty partition has zero drift by definition.

#### `Partition.update_centroid`

*method, lines 169 to 174*

```python
def update_centroid(self, new_centroid: np.ndarray) -> None
```

Replace the centroid and recompute the radius around it.

#### `Partition._recompute_radius`

*method, lines 176 to 182*

```python
def _recompute_radius(self) -> None
```

Set the radius to the largest distance from the centroid to a live vector.

#### `Partition.radius`

*method, lines 184 to 191*

```python
def radius(self) -> float
```

Return an upper bound on the distance from the centroid to any live vector.

A delete can only shrink the true radius, so the stored value stays a valid upper bound
without being recomputed on every delete, which would cost a full pass over the partition.
`compact` and `update_centroid` tighten it again, since they read the live vectors anyway.

#### `Partition.record_access`

*method, lines 193 to 195*

```python
def record_access(self) -> None
```

Count one routed query that probed this partition.

#### `Partition.decay_access`

*method, lines 197 to 199*

```python
def decay_access(self, factor: float) -> None
```

Multiply the access count by `factor`, so older queries weigh less.

#### `Partition.get_vector`

*method, lines 201 to 203*

```python
def get_vector(self, vec_id: int) -> np.ndarray
```

Return the stored vector with the given id.

#### `Partition._live_block`

*method, lines 205 to 221*

```python
def _live_block(self) -> tuple[np.ndarray, list[int]]
```

Return the live vectors and their ids, for the scan only.

When nothing is tombstoned the vectors are a view into the storage, so a caller must
neither write to them nor keep them across a change to the partition. Use
`get_live_vectors` for an independent copy.

#### `Partition.get_live_vectors`

*method, lines 223 to 233*

```python
def get_live_vectors(self) -> tuple[list[int], np.ndarray]
```

Return the ids and a copy of the vectors of the live vectors, safe for the caller to keep.

The vectors are returned in the centroid number type.

#### `Partition.stats`

*method, lines 235 to 244*

```python
def stats(self, quality: float = 0.0) -> PartitionStats
```

Return a summary of the partition, its sizes, drift and access count.

#### `Partition.__repr__`

*method, lines 246 to 251*

```python
def __repr__(self) -> str
```

Return a short description with the id, size, drift and access count.

### src/index.py

533 lines.

The IVF index, a set of partitions with one centroid each, and the operations policies share.

Routing has two stages. `_prefilter` ranks every centroid in float64 by the squared norm minus twice
the dot product and keeps nprobe plus 16 candidates, then `_exact_scores` ranks those candidates by
the plain float32 squared difference. The selected partitions are therefore the ones the plain
computation would select, found faster.

A query can be sent in two ways. `route` is a query from the stream, it counts access on every
probed partition and scans nothing. `search` answers a query, and the scoring always calls it with
`record_access=False`, so measuring never changes what a policy sees.

Policies change which partition holds a vector only through `apply_split`, `apply_merge` and
`apply_reassign`. A child of a split is credited `SPLIT_ACCESS_ALPHA` (0.86) of the access count
of its parent.

| Constant | Value |
|---|---|
| `ROUTING_CANDIDATE_MARGIN` | `16` |

#### `IVFIndex`

*class, lines 37 to 471*

```python
class IVFIndex
```

An inverted file index, partitions with centroids, plus routing and update operations.

#### `IVFIndex.__init__`

*method, lines 40 to 69*

```python
def __init__(self, dim: int) -> None
```

Create an empty index for vectors of `dim` dimensions.

#### `IVFIndex.build`

*method, lines 71 to 115*

```python
def build(self, vectors: np.ndarray, vector_ids: list[int], n_partitions: int, seed: int = 42, cache: bool = False) -> None
```

Partition `vectors` with exact k-means into `n_partitions` partitions.

Any existing partitions are replaced and every access count starts again from zero. With
`cache` the k-means result is stored on disk and reused by a later build of the same
vectors. Only an initial build should use it, a rebuild during a run partitions a
different set of vectors every time, so its entries would never be read again.

#### `IVFIndex.insert`

*method, lines 117 to 124*

```python
def insert(self, vec_id: int, vec: np.ndarray) -> None
```

Add a vector to the partition with the nearest centroid.

#### `IVFIndex.delete`

*method, lines 126 to 137*

```python
def delete(self, vec_id: int) -> bool
```

Delete a vector by tombstoning it in its partition.

Returns False if the vector is not in the index.

#### `IVFIndex.search`

*method, lines 139 to 183*

```python
def search(self, query: np.ndarray, k: int, nprobe: int, record_access: bool = True ) -> list[tuple[int, float]]
```

Return the `k` nearest vectors found in the `nprobe` nearest partitions.

The result is a list of (vector id, distance) pairs, nearest first. With `record_access`
False, which the scoring always uses, the query leaves the access counts untouched, so
measuring cannot change what the access based policies see.

#### `IVFIndex.search_batch`

*method, lines 185 to 189*

```python
def search_batch(self, queries: np.ndarray, k: int, nprobe: int, record_access: bool = True ) -> list[list[tuple[int, float]]]
```

Run `search` for every query and return the list of results.

#### `IVFIndex.search_adaptive`

*method, lines 191 to 257*

```python
def search_adaptive(self, query: np.ndarray, k: int, target_recall: float = 0.9, max_nprobe: int | None = None, record_access: bool = False, radius_factor: float = 1.0) -> tuple[list[tuple[int, float]], int, int]
```

Search with a growing nprobe until a geometric bound guarantees the target recall.

Partitions are probed nearest centroid first. After each one, a found neighbour counts as
guaranteed if no unprobed partition can hold anything closer, judged by the centroid
distance minus the partition radius. The search stops once the guaranteed share of the
`k` neighbours reaches `target_recall`. A `radius_factor` of 1.0 gives the exact bound,
which in high dimensions probes nearly everything, and smaller values stop sooner. Returns
the results, the number of partitions probed and the number of vectors scanned.

#### `IVFIndex.stats`

*method, lines 259 to 275*

```python
def stats(self) -> IndexStats
```

Return a summary of the index, sizes, drift and size imbalance.

#### `IVFIndex.partition_ids`

*method, lines 277 to 279*

```python
def partition_ids(self) -> list[int]
```

Return the ids of all partitions in ascending order.

#### `IVFIndex.get_partition`

*method, lines 281 to 283*

```python
def get_partition(self, pid: int) -> Partition
```

Return the partition with the given id.

#### `IVFIndex.apply_split`

*method, lines 285 to 308*

```python
def apply_split(self, pid: int, children: list[Partition]) -> None
```

Replace partition `pid` with `children`, which get new ids.

Each child is credited `SPLIT_ACCESS_ALPHA` times the access count of the parent, as in
Quake section 4.2.2, so the children together hold more than the parent did. That matches
what was measured, about 48 percent of the queries that probed the parent probe both
children afterwards.

#### `IVFIndex.apply_merge`

*method, lines 310 to 325*

```python
def apply_merge(self, pids: list[int], merged: Partition) -> None
```

Replace the partitions in `pids` with `merged`, which gets a new id.

#### `IVFIndex.apply_reassign`

*method, lines 327 to 342*

```python
def apply_reassign(self, vec_id: int, from_pid: int, to_pid: int) -> None
```

Move one vector from partition `from_pid` to partition `to_pid`.

The vector takes an equal share of the source access count with it, the source count
divided by its live size.

#### `IVFIndex.recompute_centroid`

*method, lines 344 to 351*

```python
def recompute_centroid(self, pid: int) -> float
```

Move the centroid of `pid` to the mean of its live vectors, and return the drift closed.

#### `IVFIndex.compact_partition`

*method, lines 353 to 355*

```python
def compact_partition(self, pid: int) -> int
```

Compact partition `pid` and return how many tombstoned vectors were removed.

#### `IVFIndex._prefilter`

*method, lines 357 to 377*

```python
def _prefilter(self, query: np.ndarray, k: int) -> np.ndarray
```

Return the positions of about `k` + 16 centroids nearest to `query`, in no order.

The squared distance equals the squared centroid norm, minus twice the dot product, plus
the squared query norm. The last term is the same for every centroid, so ranking by the
first two gives the same order, and that costs one cached vector and one matrix vector
product instead of a full difference matrix, about ten times faster at GIST size.

#### `IVFIndex._exact_scores`

*method, lines 379 to 387*

```python
def _exact_scores(self, rows: np.ndarray, query: np.ndarray) -> np.ndarray
```

Return the float32 squared distances from `query` to the centroids at `rows`.

This is the reference arithmetic, subtract then sum the squares, and the final selection
is decided on it.

#### `IVFIndex._nearest_partitions`

*method, lines 389 to 409*

```python
def _nearest_partitions(self, query: np.ndarray, nprobe: int) -> list[int]
```

Return the ids of the `nprobe` partitions with the nearest centroids, nearest first.

Ties in distance are broken by partition id. The order matters because `search` takes its
top `k` over the probed partitions concatenated, so two vectors at exactly equal distance
are separated by which partition came first, and a fixed order keeps that reproducible.
Records no access.

#### `IVFIndex.query_count`

*method, lines 411 to 419*

```python
def query_count(self, nprobe: int) -> float
```

Return the number of routed queries, the denominator of the access fraction.

When no query has been routed, which happens for an index whose access counts were set by
hand, the count is estimated as the total of the access counts divided by `nprobe`.

#### `IVFIndex.route`

*method, lines 421 to 432*

```python
def route(self, query: np.ndarray, nprobe: int) -> list[int]
```

Route a query from the stream, counting access on the probed partitions.

Nothing is scanned, the query only feeds the access signal the policies read. Returns the
ids of the probed partitions.

#### `IVFIndex._find_nearest_partition`

*method, lines 434 to 439*

```python
def _find_nearest_partition(self, vec: np.ndarray) -> int
```

Return the id of the partition with the nearest centroid, used for inserts.

#### `IVFIndex._get_centroid_matrix`

*method, lines 441 to 451*

```python
def _get_centroid_matrix(self) -> tuple[np.ndarray, list[int]]
```

Return the float32 matrix of centroids and the partition id of each row.

The matrix is built on first use and kept until a centroid or partition changes.

#### `IVFIndex._get_centroid_matrix64`

*method, lines 453 to 464*

```python
def _get_centroid_matrix64(self) -> tuple[np.ndarray, np.ndarray, list[int]]
```

Return the float64 centroid matrix, the squared norm of each centroid, and the ids.

Built from the float32 matrix and cleared together with it, so the norms always belong to
the current centroids.

#### `IVFIndex._invalidate_cache`

*method, lines 466 to 471*

```python
def _invalidate_cache(self) -> None
```

Forget the cached centroid matrices, after any change to a centroid or partition.

#### `_kmeans_cache_key`

*function, lines 474 to 495*

```python
def _kmeans_cache_key(vectors: np.ndarray, n_partitions: int, seed: int) -> str
```

Return the file name key for a stored k-means result.

The key covers the content of the vectors rather than a dataset name, so a different split or
a changed live set can never match the initial build. It also covers the scikit-learn version
and the thread setting, because Lloyd k-means gives the same answer for a fixed seed only at a
fixed thread count.

#### `_lloyd_partition`

*function, lines 498 to 533*

```python
def _lloyd_partition(vectors: np.ndarray, n_partitions: int, seed: int, cache: bool ) -> tuple[np.ndarray, np.ndarray, int]
```

Run exact k-means and return the labels, the centroids and the iteration count.

The iteration count is 0 when the result comes from the disk cache.

### src/types.py

187 lines.

Plain data records shared across the code.

`MaintenanceReport` is the important one. Every maintenance pass returns one, and
`Maintainer.maybe_maintain` adds its counters to the running totals that become the work columns
of a result row. `CostModel` is the linear cost model the policies fall back to when no profiled
scan cost curve is given, and `ExperimentConfig` is not used by the reported runs.

#### `PartitionStats`

*class, lines 19 to 27*

```python
class PartitionStats
```

A summary of one partition, its live size, tombstones, drift and access count.

#### `IndexStats`

*class, lines 31 to 43*

```python
class IndexStats
```

A summary of the whole index, its sizes, drift and size imbalance.

#### `MaintenanceReport`

*class, lines 47 to 102*

```python
class MaintenanceReport
```

What one maintenance pass did, as counts of actions and of the work they took.

Work has two parts. `vectors_processed` counts every vector read, copied or averaged, and the
distance fields count every distance evaluation between two vectors. Both are one pass over
the dimensions of a vector, which is why they can be added. Deciding whether to act is not
charged, only carrying out an action is.

#### `QualityReport`

*class, lines 106 to 113*

```python
class QualityReport
```

Partition quality over the whole index, its mean, spread and worst partition.

#### `CostModel`

*class, lines 117 to 158*

```python
class CostModel
```

A linear latency model, alpha times size plus beta times drift plus gamma times access.

#### `CostModel.predict_latency`

*method, lines 126 to 129*

```python
def predict_latency(self, size: int, drift: float = 0.0, access: int = 0) -> float
```

Return the predicted latency, never below zero.

#### `CostModel.scan_latency`

*method, lines 131 to 136*

```python
def scan_latency(self, size: int) -> float
```

Return lambda(s) of Quake section 4.1, the cost of scanning a partition of `size` vectors.

This linear model has no measured curve, so it uses its linear form.

#### `CostModel.centroid_latency`

*method, lines 138 to 143*

```python
def centroid_latency(self, n_centroids: int) -> float
```

Return the cost of comparing a query with `n_centroids` centroids.

Priced as a scan of that many vectors.

#### `CostModel.to_dict`

*method, lines 145 to 153*

```python
def to_dict(self) -> dict[str, float]
```

Return the coefficients as a dictionary.

#### `CostModel.from_dict`

*method, lines 156 to 158*

```python
def from_dict(cls, data: dict[str, float]) -> CostModel
```

Build a model from a dictionary of coefficients.

#### `ExperimentConfig`

*class, lines 162 to 187*

```python
class ExperimentConfig
```

Parameters of one experiment, kept for compatibility and not used by the reported runs.

#### `ExperimentConfig.to_yaml`

*method, lines 179 to 181*

```python
def to_yaml(self, path: Path) -> None
```

Write the configuration to a YAML file.

#### `ExperimentConfig.from_yaml`

*method, lines 184 to 187*

```python
def from_yaml(cls, path: Path) -> ExperimentConfig
```

Read a configuration from a YAML file.

### src/config.py

80 lines.

Constants shared by the code, each with its source.

The constants that govern the reported runs are MAINTENANCE_CHECK_INTERVAL,
REBUILD_FRACTION_THRESHOLD, SPLIT_ACCESS_ALPHA, BOUNDARY_REASSIGN_TOP_K, LIRE_MERGE_CANDIDATES,
DRIFT_FRACTION, the two KMEANS constants and DEFAULT_K. The starting index and its size
thresholds come from `src/staple.py`, not from here. There is no global random seed, every seed is
passed explicitly to the function that uses it.

| Constant | Value |
|---|---|
| `DEFAULT_PARTITIONS` | `1000` |
| `DEFAULT_NPROBE` | `10` |
| `DEFAULT_K` | `10` |
| `MIN_PARTITION_SIZE` | `500` |
| `MAX_PARTITION_SIZE` | `2000` |
| `DRIFT_FRACTION` | `0.0129` |
| `DRIFT_TOTAL_FRACTION` | `7.8` |
| `SPLIT_ACCESS_ALPHA` | `0.86` |
| `BALANCED_SPLIT` | `False` |
| `BOUNDARY_REASSIGN_TOP_K` | `25` |
| `LIRE_MERGE_CANDIDATES` | `64` |
| `MAINTENANCE_CHECK_INTERVAL` | `1000` |
| `REBUILD_FRACTION_THRESHOLD` | `0.025` |
| `MEASUREMENT_INTERVAL` | `5000` |
| `KMEANS_MAX_ITER` | `25` |
| `KMEANS_N_INIT` | `1` |
| `PROJECT_ROOT` | `Path(__file__).resolve().parent.parent` |
| `DATA_DIR` | `PROJECT_ROOT / 'data'` |
| `RESULTS_DIR` | `PROJECT_ROOT / 'results' / 'raw'` |
| `FIGURES_DIR` | `PROJECT_ROOT / 'results' / 'figures'` |
| `COST_MODEL_DIR` | `PROJECT_ROOT / 'results' / 'cost_model'` |
| `KMEANS_CACHE_DIR` | `DATA_DIR / 'kmeans_cache'` |

## Shared

### src/maintainers/base.py

574 lines.

The policy interface, and the restructuring operations every policy shares.

A policy subclasses `Maintainer` and implements `should_maintain` and `maintain`. The stream calls
`start` once, when it takes over the starting index, and `maybe_maintain` after every update.
`maybe_maintain` returns at once unless `check_interval` updates have passed since the last check,
and otherwise asks `should_maintain`, runs `maintain`, and adds the returned `MaintenanceReport` to
the running totals that the result row reads.

The functions below do the restructuring. Each takes the report of the current pass as `meter`
and adds the distances it computes, so two policies that take the same action pay the same price.

#### `drift_limit`

*function, lines 27 to 41*

```python
def drift_limit(index: IVFIndex, drift_fraction: float, drift_threshold: float | None = None) -> float
```

Return the drift threshold as a distance, `drift_fraction` times the data scale.

A drift threshold is a distance, so a fixed number would mean different things on datasets of
different scale, and a fraction of the scale behaves the same everywhere. An explicit
`drift_threshold` overrides the fraction.

#### `Maintainer`

*class, lines 44 to 134*

```python
class Maintainer(ABC)
```

The base class of every maintenance policy, with the running totals of its work.

#### `Maintainer.__init__`

*method, lines 49 to 80*

```python
def __init__(self, check_interval: int = MAINTENANCE_CHECK_INTERVAL) -> None
```

Create a policy that is offered the chance to act every `check_interval` updates.

#### `Maintainer.start`

*method, lines 82 to 89*

```python
def start(self, index: IVFIndex, step: int) -> None
```

Note that the stream takes over the index at update `step`.

The starting index arrives with thousands of updates behind it that are not part of the
stream, so the check interval counts from here, and a policy whose trigger reads the
index's lifetime counters takes its baseline here too.

#### `Maintainer.should_maintain`

*method, lines 92 to 94*

```python
def should_maintain(self, index: IVFIndex, step: int) -> bool
```

Return True if the policy wants to act at this check.

#### `Maintainer.maintain`

*method, lines 97 to 99*

```python
def maintain(self, index: IVFIndex) -> MaintenanceReport
```

Run one maintenance pass and return what it did.

#### `Maintainer.maybe_maintain`

*method, lines 101 to 134*

```python
def maybe_maintain(self, index: IVFIndex, step: int) -> MaintenanceReport | None
```

Offer the policy a chance to act, at most once every `check_interval` updates.

If it acts, its report is added to the running totals and returned, otherwise None.

#### `balance_constrained_assignment`

*function, lines 137 to 178*

```python
def balance_constrained_assignment(vectors: np.ndarray, centroids: np.ndarray, max_iter: int = 20) -> tuple[np.ndarray, np.ndarray]
```

Cluster `vectors` with a penalty on large clusters, and return the labels and centroids.

SPANN equation 1 adds lambda times the squared deviation of each cluster size from the mean to
the k-means objective. SPFresh relaxes it into an assignment penalty, the distance to a centre
plus lambda times that cluster's current size, so a cluster that grows too fast becomes more
expensive to join. The penalty reads the counts of the previous iteration, which makes the
result independent of the order the vectors are visited in, and lambda is recomputed each
iteration from the data rather than fixed. Switched off in the reported runs, see
config.BALANCED_SPLIT.

#### `kmeans_distance_evaluations`

*function, lines 181 to 191*

```python
def kmeans_distance_evaluations(km, n_points: int) -> int
```

Return the distance evaluations a fitted scikit-learn KMeans made.

Every Lloyd iteration compares every point with every centre, and k-means++ seeding is counted
as one more iteration. scikit-learn reports the iteration count of the best initialisation
only, so with several initialisations the others are assumed to have run as long, and the
seeding tries more candidates than one per centre, so this is a lower bound. Random seeding
compares nothing and adds no iteration.

#### `split_partition_2means`

*function, lines 194 to 229*

```python
def split_partition_2means(partition: Partition, random_state: int = 42, balanced: bool = BALANCED_SPLIT, meter=None, n_init: int = 3, max_iter: int = 300, init: str = "k-means++") -> list[Partition]
```

Split a partition in two with 2-means, and return the new partitions.

The children get placeholder ids and the caller puts them into the index. A partition that
cannot be split, or whose vectors all fall in one cluster, comes back as a single partition.
The defaults are those of scikit-learn, and the Quake policy passes its own.

#### `merge_partitions`

*function, lines 232 to 251*

```python
def merge_partitions(partitions: list[Partition], placeholder_id: int = -1) -> Partition
```

Join partitions into one, centred on the mean of all their live vectors.

The access counts are added. The result has a placeholder id and the caller puts it into the
index. The bandit merges this way.

#### `collect_empty_partitions`

*function, lines 254 to 274*

```python
def collect_empty_partitions(index: IVFIndex, keep_at_least: int = 1) -> list[int]
```

Remove partitions with no live vectors, and return the ids removed.

An empty partition is still selected by the router, where it returns nothing, so it wastes a
probe and lowers recall at a fixed nprobe while costing nothing in the scan count. Removing it
is not a merge, since there is nothing to move, so callers count it in `num_collected` and
charge no work. `keep_at_least` never lets the index reach zero partitions, which would break
routing.

#### `plan_partition_deletion`

*function, lines 277 to 315*

```python
def plan_partition_deletion(index: IVFIndex, pid: int, meter=None, exclude: set[int] | None = None) -> dict[int, list[int]]
```

Return which partition each vector of `pid` would move to if `pid` were deleted.

This is the merge of Quake section 4.2.1, "After deletion, the vectors are reassigned to their
respective nearest existing partitions." Each vector goes to its own nearest centroid, so there
can be several receivers, and every receiver keeps its own centroid. It differs from the
SPFresh merge, which appends a whole partition to one chosen partner, see
`merge_into_survivor`.

The plan is returned rather than applied, so the caller can compute the exact cost change over
the real receivers before committing. Quake applies tentatively and rolls back on rejection,
and evaluating first reaches the same decision without an undo. `exclude` names partitions
deleted in the same batch, which cannot receive vectors, since Quake removes them all before
reinserting any vector.

#### `apply_partition_deletion`

*function, lines 318 to 338*

```python
def apply_partition_deletion(index: IVFIndex, pid: int, plan: dict[int, list[int]]) -> int
```

Delete `pid` and move its vectors as `plan` says. Returns the number of vectors moved.

#### `merge_into_survivor`

*function, lines 341 to 356*

```python
def merge_into_survivor(index: IVFIndex, short_pid: int, long_pid: int) -> list[int]
```

Merge partition `short_pid` into `long_pid`, as SPFresh section 3.2 does.

The shorter partition and its centroid are deleted and its vectors appended to the other,
which keeps its own centroid rather than moving to the combined mean. Returns the ids moved.

#### `reassign_lire_after_split`

*function, lines 359 to 414*

```python
def reassign_lire_after_split(index: IVFIndex, old_centroid: np.ndarray, new_pids: list[int], neighbor_pids: list[int], meter=None) -> tuple[int, int]
```

Repair vector placement after a split with the two LIRE conditions of SPFresh section 3.3.

The conditions decide which vectors are worth checking at all, which is what LIRE saves over
checking every vector in the neighbourhood. A checked vector moves to the nearest local
centroid if that is strictly closer than its own. Returns the number moved and the number
checked.

#### `reassign_lire_after_merge`

*function, lines 417 to 460*

```python
def reassign_lire_after_merge(index: IVFIndex, merged_pid: int, moved_ids: list[int], neighbor_pids: list[int], old_centroid: np.ndarray, meter=None) -> int
```

Repair vector placement after a LIRE merge. Returns the number of vectors moved.

Only the vectors of the deleted partition can be misplaced by a merge, SPFresh section 3.3.
Following the authors' implementation, a moved vector is reconsidered only when the surviving
centroid is farther from it than its deleted centroid `old_centroid` was, and it then moves to
the nearest local centroid if that is strictly closer.

#### `top_k_nearest_partitions`

*function, lines 463 to 486*

```python
def top_k_nearest_partitions(index: IVFIndex, target_centroid: np.ndarray, k: int, exclude: set[int] | None = None, meter=None) -> list[int]
```

Return the ids of the `k` partitions with centroids nearest to `target_centroid`, in no order.

#### `nearest_partitions_in_order`

*function, lines 489 to 518*

```python
def nearest_partitions_in_order(index: IVFIndex, target_centroid: np.ndarray, k: int, exclude: set[int] | None = None, meter=None) -> list[int]
```

Return the ids of the `k` partitions with centroids nearest to `target_centroid`, nearest first.

The SPFresh merge scans the nearest partitions in order and takes the first whose combined
length fits, so the order matters here, unlike in `top_k_nearest_partitions`.

#### `reassign_boundary`

*function, lines 521 to 558*

```python
def reassign_boundary(index: IVFIndex, target_pids: list[int], candidate_source_pids: list[int], meter=None) -> int
```

Move every vector of the given partitions to its nearest centroid among them.

A vector moves only when another centroid in the set is strictly closer than its own, in
either direction between the partitions. Running it twice changes nothing more. Returns the
number of vectors moved.

#### `recompute_all_centroids`

*function, lines 561 to 574*

```python
def recompute_all_centroids(index: IVFIndex, drift_threshold: float = 0.0) -> int
```

Move every centroid whose drift exceeds `drift_threshold` to its partition mean.

Returns the number of centroids moved.

### src/cost_model.py

463 lines.

The scan cost curve lambda(s) that the Quake and bandit policies decide with.

lambda(s) is the cost of scanning one partition of s vectors, in nanoseconds, and lambda_c(K) is
the cost of ranking K centroids. A policy uses them to price an action before taking it. The
Quake policy prices every candidate action with them, and the reward of the bandit is the change
in the total Quake cost built on them.

The stored curves, `results/cost_model/profiled_lambda_dim128.yaml` and `..._dim960.yaml`, are
built by `lambda_from_prices` from the same Faiss price list the scoring uses, and
`scripts/calibrate_query_cost.py` writes them. A policy therefore decides in the unit it is scored
in. The profiling and fitting functions below measure the scan of this Python code instead, and
no reported run uses them.

The scoring never calls anything here. A policy that carries a cost model is making a claim about
what a query costs, and the scoring is the independent check of whether that claim paid off.

#### `benchmark_partition_scan`

*function, lines 32 to 63*

```python
def benchmark_partition_scan(dim: int = 128, sizes: list[int] | None = None, n_trials: int = 20, seed: int = 42) -> pd.DataFrame
```

Time the partition scan on random partitions of each size.

Returns one row per trial with the size and the wall time in nanoseconds. Not used by the
reported runs.

#### `fit_cost_model`

*function, lines 66 to 89*

```python
def fit_cost_model(timings_df: pd.DataFrame) -> CostModel
```

Fit latency as alpha times size through the origin, and return it as a linear model.

The line is forced through the origin because a free intercept comes out negative over these
sizes, which would predict zero cost for small partitions. Drift and access get zero weight.

#### `ProfiledCostModel`

*class, lines 93 to 190*

```python
class ProfiledCostModel
```

lambda(s) as a table of measured points, interpolated between them, as in Quake section 4.1.

Quake does not assume a formula for the scan cost, it measures it. `sizes` and `latencies`
are the measured points. On the stored curves the latencies are the Faiss time to open and
scan a list, so `dispatch_ns` and `probe_ns` are zero and `centroid_ns` prices one centroid
comparison, see `lambda_from_prices`.

The two other terms exist for curves measured on the Python scan. `dispatch_ns` is the part of
a call that does not depend on the number of vectors, which in Python is the interpreter
starting the numpy operations, about 4.3 microseconds at both dimensions. It is subtracted
because a compiled engine such as Quake does not pay it, and leaving it in would make every
split look expensive. `probe_ns` is a fixed cost added to every scan of a non empty partition.

#### `ProfiledCostModel._raw`

*method, lines 123 to 129*

```python
def _raw(self, s: float) -> float
```

Return the measured curve at size `s`, extended past the last point along its last segment.

#### `ProfiledCostModel.scan_latency`

*method, lines 131 to 145*

```python
def scan_latency(self, size: int) -> float
```

Return lambda(size), the cost of scanning one partition of `size` vectors.

Below the smallest measured size the first point is scaled down linearly. A curve whose
first point is at size 0, as the Faiss curve is, charges that point for an empty partition,
since the router does probe empty partitions.

#### `ProfiledCostModel.centroid_latency`

*method, lines 147 to 161*

```python
def centroid_latency(self, n_centroids: int) -> float
```

Return lambda_c(K), the cost of comparing a query with `n_centroids` centroids.

This is the top level scan every query performs, the delta O term of Quake equations 4 and
5. It is one pass over the centroids, so it pays no per partition cost.

#### `ProfiledCostModel.predict_latency`

*method, lines 163 to 165*

```python
def predict_latency(self, size: int, drift: float = 0.0, access: int = 0) -> float
```

Return the scan cost of a partition of `size` vectors. Drift and access are ignored.

#### `ProfiledCostModel.to_dict`

*method, lines 167 to 177*

```python
def to_dict(self) -> dict
```

Return the model as a dictionary for storing in YAML.

#### `ProfiledCostModel.from_dict`

*method, lines 180 to 190*

```python
def from_dict(cls, payload: dict) -> ProfiledCostModel
```

Build a model from a dictionary, with defaults for any missing term.

#### `lambda_from_prices`

*function, lines 193 to 211*

```python
def lambda_from_prices(entry: dict) -> ProfiledCostModel
```

Build lambda(s) as the Faiss time to open and scan a list of s vectors.

`entry` is the price list of one dimension from
`results/cost_model/query_cost_calibration.yaml`, the same curve the scoring prices every
probed list with, so a policy prices an action in the unit it is scored in. Quake measures
lambda on the engine that serves its queries, and the closest equivalent here is Faiss
IndexIVFFlat rather than the Python scan. Opening a list is already inside the curve, so no
dispatch or per partition term is added, and a centroid comparison is priced as the scoring
prices it.

#### `profile_scan_latency`

*function, lines 214 to 241*

```python
def profile_scan_latency(dim: int = 128, sizes: list[int] | None = None, n_trials: int = 40, inner: int | None = None, seed: int = 42, window_ns: float = 2e6) -> ProfiledCostModel
```

Measure lambda(s) of the Python scan. Not used by the reported runs.

Each sample times a run of `inner` scans and divides, so the time per scan is large compared
with the clock resolution and the loop overhead. Each size then takes the minimum over trials,
because interruptions only ever add time, so the minimum is the least disturbed sample. With
`inner` unset the repeat count is chosen per size to fill a window of `window_ns`, so small
partitions are repeated often and large ones are not repeated needlessly.

#### `_min_batched_scan_ns`

*function, lines 244 to 269*

```python
def _min_batched_scan_ns(rng: np.random.Generator, size: int, dim: int, n_trials: int, inner: int | None, window_ns: float) -> float
```

Return the minimum over trials of the mean time of a batch of scans, for one size.

#### `profile_dispatch_ns`

*function, lines 272 to 289*

```python
def profile_dispatch_ns(dim: int = 128, n_trials: int = 40, seed: int = 42, window_ns: float = 2e6, reference_size: int = 25) -> float
```

Measure the part of one scan call that does not depend on the number of vectors.

The scan is timed at one vector and at `reference_size` vectors, and the line through the two
is extended back to zero vectors. The random generator is seeded apart from
`profile_scan_latency`, so adding this measurement leaves the curve unchanged.

#### `fit_profiled_cost_model`

*function, lines 292 to 302*

```python
def fit_profiled_cost_model(timings_df: pd.DataFrame) -> ProfiledCostModel
```

Build lambda(s) from benchmark timings, the median time at each size.

The median is used because it is less affected by an occasional interrupted trial than the
mean.

#### `benchmark_cost_model_full`

*function, lines 305 to 354*

```python
def benchmark_cost_model_full(dim: int = 128, sizes: list[int] | None = None, drift_levels: list[float] | None = None, access_levels: list[float] | None = None, n_trials: int = 20, seed: int = 42) -> pd.DataFrame
```

Time the scan while also varying drift and access, to test whether they affect latency.

Not used by the reported runs.

#### `fit_cost_model_full`

*function, lines 357 to 415*

```python
def fit_cost_model_full(timings_df: pd.DataFrame) -> tuple[CostModel, dict]
```

Fit latency on size, drift and access by least squares, with the significance of each.

Returns the fitted linear model and a dictionary with the estimate, standard error, t
statistic and p value of every coefficient, and the R squared of the full and the size only
fit.

#### `save_cost_model`

*function, lines 418 to 425*

```python
def save_cost_model(model: CostModel, path: Path) -> None
```

Write a linear cost model and a description of the machine to a YAML file.

#### `load_cost_model`

*function, lines 428 to 431*

```python
def load_cost_model(path: Path) -> CostModel
```

Read a linear cost model from a YAML file.

#### `save_profiled_cost_model`

*function, lines 434 to 445*

```python
def save_profiled_cost_model(model: ProfiledCostModel, path: Path) -> None
```

Write a measured scan cost curve and a description of the machine to a YAML file.

A timed curve differs from one measurement to the next, so it is measured once and stored, as
Quake does, and every run reads the stored curve. That keeps a run determined by its seed.

#### `load_profiled_cost_model`

*function, lines 448 to 451*

```python
def load_profiled_cost_model(path: Path) -> ProfiledCostModel
```

Read a scan cost curve from a YAML file.

#### `_machine_info`

*function, lines 454 to 463*

```python
def _machine_info() -> dict[str, str]
```

Return the platform, processor, Python version and numpy version of this machine.

### src/calibration.py

131 lines.

Calibrated query cost and calibrated work, the common unit every method is scored in.

Both are counts of operations, each priced against one scan distance of Faiss IndexIVFFlat on the
machine that ran the experiments, so no method is scored in its own units. The prices come from
`scripts/calibrate_query_cost.py`, as the median of five independent calibrations, and are stored
in `results/cost_model/query_cost_calibration.yaml`.

The query cost of one query at the target recall is

    the sum over probed partitions of list_price(size), plus centroid_ratio times K

since every query ranks all K centroids and then opens and scans nprobe partitions. list_price is
a measured curve, what Faiss takes to open and scan one list of a given length, in scan
distances. Counting scanned vectors alone would charge nothing for ranking the centroids or for
opening a list, which would let a method that fragments the index look cheap. A single price per
probed list does not work either, because the cost per vector itself depends on list length,
about 15 to 17 ns for lists of 25 vectors against about 10 ns for 100 to 800 on SIFT, so each list
is priced at its own length. Every price is measured on an index of the same size as the ones it
prices, 51,200 vectors.

The work of a run is

    distances, plus kmeans_small times small k-means distances, plus kmeans_bulk times bulk
    k-means distances, plus read_copy times vectors read or copied

since the methods spend their work on different kinds of operation, a rebuild on large k-means
runs, a split on a small one, LIRE and Quake on reassignment distances, a merge on copies. Pricing
each kind makes the totals comparable without changing what any method does.

No policy reads this module. It is the price list of the scoring, and the scan cost curves the
policies read are written from it by `scripts/calibrate_query_cost.py`.

| Constant | Value |
|---|---|
| `CALIBRATION_PATH` | `COST_MODEL_DIR / 'query_cost_calibration.yaml'` |

#### `_load`

*function, lines 48 to 50*

```python
def _load(path: str) -> dict
```

Read the calibration file, cached by path.

#### `prices`

*function, lines 53 to 63*

```python
def prices(dim: int, path: Path | None = None) -> dict
```

Return the calibrated prices for vectors of `dim` dimensions.

Raises KeyError if the file has no entry for that dimension.

#### `has_price_list`

*function, lines 66 to 76*

```python
def has_price_list(dim: int, path: Path | None = None) -> bool
```

Return True if a list price curve exists for `dim` dimensions.

An index in a dimension nothing was calibrated for is read without the priced cost. A full
run needs the priced reading for every row, so there the missing entry raises an error later
instead of being skipped here.

#### `query_cost`

*function, lines 79 to 81*

```python
def query_cost(priced: float, n_partitions: int, dim: int) -> float
```

Return the query cost, the priced partition scan plus the ranking of all K centroids.

#### `list_price`

*function, lines 84 to 100*

```python
def list_price(sizes, dim: int, path: Path | None = None) -> np.ndarray
```

Return what scanning a list of each given size costs in Faiss, in scan distances.

The price includes opening the list. It is interpolated between the measured list sizes, and
above the largest one it is extended along the last measured segment, where the cost per
vector has settled.

#### `scan_unit_ns`

*function, lines 103 to 111*

```python
def scan_unit_ns(lengths, list_ns, from_m: int = 200) -> float
```

Return one scan distance in nanoseconds, the time one more vector adds to a long list.

It is the least squares slope of list time against list length, over lists of at least
`from_m` vectors.

#### `per_list_overhead_ns`

*function, lines 114 to 123*

```python
def per_list_overhead_ns(lengths, list_ns, lo: int = 25, hi: int = 400) -> float
```

Return the fixed cost per list that a straight line through the curve implies.

This is the intercept of the fit over the list lengths the methods reach. It is stored as a
diagnostic and used in no price. Its range crosses zero on both datasets, which is why lists
are priced on the whole measured curve instead of as vectors plus a constant.

#### `work_cost`

*function, lines 126 to 131*

```python
def work_cost(reads: float, distances: float, kmeans_small: float, kmeans_bulk: float, dim: int) -> float
```

Return calibrated maintenance work, each kind of operation priced in scan distances.

### src/dataset.py

144 lines.

Readers for SIFT1M and GIST1M, and exact ground truth.

The reported runs use `load_sift1m`, `load_gist1m` and `compute_ground_truth`. The files live under
data/ and `scripts/download_datasets.py` fetches them. `make_synthetic` builds a small dataset for
quick checks, and `split_base_for_streaming` is not used by the reported runs.

#### `read_fvecs`

*function, lines 17 to 27*

```python
def read_fvecs(path: Path) -> np.ndarray
```

Read a .fvecs file into a float32 matrix.

Each row is stored as its dimension, a 32 bit integer, followed by that many floats.

#### `read_ivecs`

*function, lines 30 to 37*

```python
def read_ivecs(path: Path) -> np.ndarray
```

Read a .ivecs file, the same layout as .fvecs with integers, into an int32 matrix.

#### `load_sift1m`

*function, lines 40 to 54*

```python
def load_sift1m(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Load SIFT1M and return the base vectors, the queries and the ground truth.

The shapes are one million by 128, ten thousand by 128 and ten thousand by 100.

#### `load_gist1m`

*function, lines 57 to 71*

```python
def load_gist1m(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Load GIST1M and return the base vectors, the queries and the ground truth.

The shapes are one million by 960, one thousand by 960 and one thousand by 100.

#### `make_synthetic`

*function, lines 74 to 97*

```python
def make_synthetic(n: int, dim: int, n_clusters: int = 50, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Return a Gaussian mixture dataset, base vectors, queries and ground truth, for quick checks.

#### `compute_ground_truth`

*function, lines 100 to 119*

```python
def compute_ground_truth(base: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray
```

Return the row indices of the exact `k` nearest base vectors of every query.

Uses an exhaustive Faiss index when Faiss is installed, and scikit-learn otherwise. Both
compare every query with every vector, so the answer is exact either way.

#### `split_base_for_streaming`

*function, lines 122 to 144*

```python
def split_base_for_streaming(base: np.ndarray, initial_fraction: float = 0.5, seed: int = 42) -> tuple[np.ndarray, list[int], np.ndarray, list[int]]
```

Split the base vectors at random into an initial set and a pool of later inserts.

The ids are the original row numbers, so they still match the published ground truth after
the pool vectors are inserted.

## Policies

### src/maintainers/no_op.py

21 lines.

No maintenance, the reference every other policy is compared with.

#### `NoOpMaintainer`

*class, lines 10 to 21*

```python
class NoOpMaintainer(Maintainer)
```

A policy that never acts, so the index changes only through inserts and deletes.

#### `NoOpMaintainer.should_maintain`

*method, lines 15 to 17*

```python
def should_maintain(self, index: IVFIndex, step: int) -> bool
```

Never ask for maintenance.

#### `NoOpMaintainer.maintain`

*method, lines 19 to 21*

```python
def maintain(self, index: IVFIndex) -> MaintenanceReport
```

Do nothing and report that no maintenance ran.

### src/maintainers/global_rebuild.py

94 lines.

The periodic rebuild baseline of Ada-IVF, section 5.1.

Each time inserts plus deletes since the last rebuild reach 2.5 percent of the live count, the
index is rebuilt from its live vectors with exact k-means at the same number of partitions, unless
`n_partitions` fixes another number, as the control run at the bandit's partition count does.

#### `GlobalRebuildMaintainer`

*class, lines 18 to 94*

```python
class GlobalRebuildMaintainer(Maintainer)
```

Rebuild the whole index with k-means after a fixed share of the vectors has changed.

#### `GlobalRebuildMaintainer.__init__`

*method, lines 23 to 35*

```python
def __init__(self, rebuild_fraction: float = REBUILD_FRACTION_THRESHOLD, n_partitions: int | None = None, check_interval: int = MAINTENANCE_CHECK_INTERVAL, seed: int = 42) -> None
```

Create the policy. `n_partitions` fixes the partition count of every rebuild if given.

#### `GlobalRebuildMaintainer.start`

*method, lines 37 to 44*

```python
def start(self, index: IVFIndex, step: int) -> None
```

Start counting updates from the start of the stream.

Ada-IVF counts the 2.5 percent over the workload that follows the build, so the updates
that grew the starting index do not count.

#### `GlobalRebuildMaintainer.should_maintain`

*method, lines 46 to 52*

```python
def should_maintain(self, index: IVFIndex, step: int) -> bool
```

Return True once the updates since the last rebuild reach the set share of live vectors.

#### `GlobalRebuildMaintainer.maintain`

*method, lines 54 to 94*

```python
def maintain(self, index: IVFIndex) -> MaintenanceReport
```

Rebuild the index from its live vectors with exact k-means.

The work is one read of every live vector plus every distance the k-means computed, its
iterations times vectors times centroids, which is what makes a rebuild far more expensive
than refreshing every centroid.

### src/maintainers/lire_lite.py

215 lines.

LIRE, the update protocol of SPFresh (SOSP 2023), in memory.

One pass does three things in this order. It removes empty partitions. It splits every partition
whose stored length, tombstones included, exceeds the split limit, compacting it first and
skipping the split if it then holds fewer than the limit. Then it merges every partition of at most
the merge limit live vectors into the nearest of 64 neighbours that still fits. Each split and
merge ends with the reassignment checks of SPFresh section 3.3. The same class grows the starting
index of every policy, see `src/staple.py`.

#### `LireLiteMaintainer`

*class, lines 34 to 215*

```python
class LireLiteMaintainer(Maintainer)
```

Keep every partition between a merge limit and a split limit, as SPFresh LIRE does.

#### `LireLiteMaintainer.__init__`

*method, lines 39 to 62*

```python
def __init__(self, max_partition_size: int = MAX_PARTITION_SIZE, min_partition_size: int = MIN_PARTITION_SIZE, boundary_top_k: int = BOUNDARY_REASSIGN_TOP_K, merge_candidates: int = LIRE_MERGE_CANDIDATES, check_interval: int = MAINTENANCE_CHECK_INTERVAL, enable_merge: bool = True, enable_compact: bool = True, seed: int = 42) -> None
```

Create the policy with its split and merge limits.

`boundary_top_k` is the number of neighbouring partitions the reassignment considers, and
`merge_candidates` the number of neighbours a merge looks through for a partner.

#### `LireLiteMaintainer.should_maintain`

*method, lines 64 to 80*

```python
def should_maintain(self, index: IVFIndex, step: int) -> bool
```

Return True if any partition is too long, small enough to merge, or empty.

The split check reads the stored length, which still includes the vectors deleted since
the last compaction, because SPFresh checks the length of the posting, SPFresh section 4.1.

#### `LireLiteMaintainer._collect`

*method, lines 82 to 94*

```python
def _collect(self, index: IVFIndex, pid: int, report: MaintenanceReport) -> bool
```

Compact a partition if it has tombstones, and return whether it did.

The work is the stored length, counted in compaction_reads, which the reported work leaves
out.

#### `LireLiteMaintainer.maintain`

*method, lines 96 to 215*

```python
def maintain(self, index: IVFIndex) -> MaintenanceReport
```

Run one LIRE pass, collect, split, then merge, and return what it did.

### src/maintainers/dedrift.py

183 lines.

The three DeDrift strategies, Baranchuk et al., ICCV 2023, section 5.1.

The paper applies them on a schedule, every one to six months of collected content. Here they use
the trigger of the global rebuild, an update once 2.5 percent of the live vectors have changed, so
DeDrift and a full rebuild differ only in what they do and not in how often.

Lazy moves every centroid to the mean of the vectors currently assigned to it, with no
reassignment. Split collects the vectors of the k largest clusters into B1, sets k2 to the ceiling
of the size of B1 over mu, the median cluster size of the whole index, collects the k2 minus k
smallest clusters into B2, runs k-means with k2 centroids on B1 and B2 together and replaces those
k2 clusters, so the total cluster count stays the same. Hybrid runs Lazy and then Split.

| Constant | Value |
|---|---|
| `VARIANTS` | `('lazy', 'split', 'hybrid')` |
| `DEDRIFT_N_LARGEST_FRACTION` | `0.002` |

#### `DeDriftMaintainer`

*class, lines 33 to 183*

```python
class DeDriftMaintainer(Maintainer)
```

DeDrift Lazy, Split or Hybrid, run whenever a set share of the index has changed.

#### `DeDriftMaintainer.__init__`

*method, lines 38 to 60*

```python
def __init__(self, variant: str = "hybrid", n_largest: int | None = None, n_largest_fraction: float = DEDRIFT_N_LARGEST_FRACTION, update_fraction: float = REBUILD_FRACTION_THRESHOLD, check_interval: int = MAINTENANCE_CHECK_INTERVAL, seed: int = 42) -> None
```

Create the policy for one variant, "lazy", "split" or "hybrid".

`n_largest` fixes the number of largest clusters in B1 and overrides `n_largest_fraction`.

#### `DeDriftMaintainer.start`

*method, lines 62 to 65*

```python
def start(self, index: IVFIndex, step: int) -> None
```

Start counting updates from the start of the stream, as the global rebuild does.

#### `DeDriftMaintainer.should_maintain`

*method, lines 67 to 73*

```python
def should_maintain(self, index: IVFIndex, step: int) -> bool
```

Return True once the updates since the last step reach the set share of live vectors.

#### `DeDriftMaintainer.maybe_maintain`

*method, lines 75 to 80*

```python
def maybe_maintain(self, index: IVFIndex, step: int)
```

Run the base check, and restart the update count whenever a step ran.

#### `DeDriftMaintainer.maintain`

*method, lines 82 to 89*

```python
def maintain(self, index: IVFIndex) -> MaintenanceReport
```

Run the variant's update step and return what it did.

#### `DeDriftMaintainer.n_largest_for`

*method, lines 91 to 95*

```python
def n_largest_for(self, n_partitions: int) -> int
```

Return the number of largest clusters to collect into B1, at least one.

#### `DeDriftMaintainer._lazy`

*method, lines 97 to 108*

```python
def _lazy(self, index: IVFIndex, report: MaintenanceReport) -> None
```

Move each centroid to the mean of its live vectors, without reassigning any vector.

The work is one read of every live vector.

#### `DeDriftMaintainer._split`

*method, lines 110 to 183*

```python
def _split(self, index: IVFIndex, report: MaintenanceReport) -> None
```

Recluster the largest clusters together with the smallest ones, keeping K constant.

### src/maintainers/cost_driven_quake.py

542 lines.

The maintenance loop of Quake (OSDI 2025), as the authors' released implementation performs it.

`maintain` is one whole pass, in the order of Algorithm 4.2 of the thesis. It refreshes changed
centroids by the change since the last pass, decides every delete and split from one snapshot,
applies the deletes together, splits and checks each split on the real children, refines the 50
nearest partitions of every child, removes empty partitions and resets the access window. Run with
tau at 50 ns it is the declared sensitivity configuration `cost_driven_quake_tau50`, see
`scripts/final_sweep.build`.

References to Quake's code name the file and function in the released implementation, for example
maintenance_policies.cpp, so each step can be checked against it.

| Constant | Value |
|---|---|
| `QUAKE_TAU_NS` | `250.0` |
| `QUAKE_ALPHA` | `SPLIT_ACCESS_ALPHA` |
| `QUAKE_MIN_PARTITION_SIZE` | `32` |
| `QUAKE_PARTITION_REDUCTION_THRESHOLD` | `0.3` |
| `QUAKE_REFINEMENT_RADIUS` | `50` |
| `QUAKE_SPLIT_ITERATIONS` | `5` |
| `FAISS_MAX_POINTS_PER_CENTROID` | `256` |
| `FAISS_CLUSTERING_SEED` | `1234` |
| `FAISS_EMPTY_CLUSTER_EPS` | `1.0 / 1024.0` |

#### `_live_sum`

*function, lines 62 to 66*

```python
def _live_sum(p: Partition) -> np.ndarray
```

Return the sum of a partition's live vectors, in float64 so differences keep their digits.

#### `faiss_split`

*function, lines 69 to 126*

```python
def faiss_split(partition: Partition, niter: int = QUAKE_SPLIT_ITERATIONS, k: int = 2, seed: int = FAISS_CLUSTERING_SEED, meter=None) -> list[Partition]
```

Split a partition as Quake's CPU path does, Faiss k-means followed by one assignment.

Faiss trains on at most max_points_per_centroid times k points, a random sample when the
partition is larger, starts the centres at k distinct training points, and runs exactly
`niter` rounds of assign then average, with no early stop. A centre left empty is revived as a
copy of a non empty one, chosen in proportion to size, and the two are nudged apart. Every
vector of the partition is then assigned once to the final centres.

The work charged is what that computes, `niter` rounds over the training points against k
centres plus the final assignment of all n vectors, all counted as k-means distances. The
random draws use numpy rather than the Faiss generator, so they differ from Faiss's for the
same seed while the procedure is the same.

#### `CostDrivenQuakeMaintainer`

*class, lines 129 to 542*

```python
class CostDrivenQuakeMaintainer(Maintainer)
```

Quake section 4.2, an action is taken only when its predicted cost change clears tau.

The predicted change delta C includes the cost of the centroid a split adds or a delete
removes, so it can decline a split and can price a delete at all.

#### `CostDrivenQuakeMaintainer.__init__`

*method, lines 138 to 186*

```python
def __init__(self, cost_model: CostModel, tau: float = QUAKE_TAU_NS, alpha: float = QUAKE_ALPHA, max_actions_per_maintain: int = 20, boundary_top_k: int = QUAKE_REFINEMENT_RADIUS, check_interval: int = MAINTENANCE_CHECK_INTERVAL, access_decay: float | None = None, nprobe: int = 10, seed: int = 42, min_partition_size: int = QUAKE_MIN_PARTITION_SIZE, partition_reduction_threshold: float = QUAKE_PARTITION_REDUCTION_THRESHOLD, recompute_centroids: bool = True, delete_tau: float | None = None) -> None
```

Create the policy with its cost model and thresholds.

`tau` is the split threshold and `delete_tau` the delete threshold, equal to `tau` unless
given. The released code keeps the two separate, while the paper states a single tau.
`max_actions_per_maintain` caps the deletes and the splits per pass, and the reported
runs set it high enough that it never binds.

#### `CostDrivenQuakeMaintainer.start`

*method, lines 188 to 197*

```python
def start(self, index: IVFIndex, step: int) -> None
```

Take a snapshot of the index at handover, so the first pass sees only later changes.

Quake resets every partition's change record after building, and the handover is the
build as far as this policy can see.

#### `CostDrivenQuakeMaintainer._snapshot`

*method, lines 199 to 204*

```python
def _snapshot(self, index: IVFIndex, pids=None) -> None
```

Record the size and the vector sum of the given partitions, or of all of them.

#### `CostDrivenQuakeMaintainer._access_fraction`

*method, lines 206 to 213*

```python
def _access_fraction(self, index: IVFIndex) -> dict[int, float]
```

Return A for every partition, the share of routed queries in the window that scanned it.

#### `CostDrivenQuakeMaintainer._centroid_delta`

*method, lines 215 to 224*

```python
def _centroid_delta(self, n_partitions: int, added: int) -> float
```

Return the change in centroid scan cost when `added` partitions are added, delta O.

The index is flat, so the level above the partitions is the list of centroids, which every
query scans in full. A split adds a centroid and pays lambda_c(K plus 1) minus
lambda_c(K), and a delete gets it back. Without this term delta C would be negative for
every accessed partition at any size, and the policy would split without limit.

#### `CostDrivenQuakeMaintainer.delta_split_estimate`

*method, lines 226 to 237*

```python
def delta_split_estimate(self, access: float, size: int, n_partitions: int) -> float
```

Return the predicted cost change of splitting a partition, Quake equation 6.

It assumes two equal halves, and each child keeps alpha times the parent access, so the
two children together hold 2 alpha A, not A.

#### `CostDrivenQuakeMaintainer.delta_split_exact`

*method, lines 239 to 252*

```python
def delta_split_exact(self, access: float, children: list, n_partitions: int) -> float
```

Return the cost change of a split over the real children, Quake equation 4.

Quake applies a split tentatively, measures the real sizes and keeps it only if the
recomputed change still clears tau. This evaluates before applying instead, which reaches
the same decision without undoing anything. Each child is still credited alpha times the
parent access, since the real access after a split is not known until queries arrive.

#### `CostDrivenQuakeMaintainer.delta_merge_estimate`

*method, lines 254 to 273*

```python
def delta_merge_estimate(self, access: float, size: int, receivers: list[tuple[float, int]], n_partitions: int) -> float
```

Return the predicted cost change of deleting a partition, Quake equation 5.

The deleted partition's vectors and access are assumed to spread equally over the
receivers, given as (access, size) pairs. Not used by the reported runs, which estimate
with `delta_merge_estimate_uniform` and check with `delta_merge_exact`.

#### `CostDrivenQuakeMaintainer.delta_merge_exact`

*method, lines 275 to 294*

```python
def delta_merge_exact(self, access: float, size: int, receivers: list[tuple[float, int, int]], n_partitions: int) -> float
```

Return the cost change of a delete over the receivers its vectors really go to.

This is Quake equation 5 evaluated as the check of section 4.2.3, "we measure the actual
resulting partition sizes (and the exact receiver partitions for merges)". `receivers`
holds (access, size, vectors received) for each receiver. The access each receiver gains
stays the equal share of the estimate, since the paper keeps the stage 1 frequency
assumptions, so only the sizes and the receiver set become exact.

#### `CostDrivenQuakeMaintainer.delta_merge_estimate_uniform`

*method, lines 296 to 321*

```python
def delta_merge_estimate_uniform(self, access: float, size: int, n_partitions: int, avg_access: float, avg_size: float) -> float
```

Return the stage 1 delete estimate exactly as the released code computes it.

This follows compute_delete_delta in maintenance_cost_estimator.cpp. The other T minus 1
partitions are all taken as an average partition, of `avg_size` vectors at `avg_access`,
and the deleted partition is spread equally over them. When it holds fewer vectors than
there are partitions, at most `size` of them gain one vector each. This estimate decides
which deletes are considered at all.

#### `CostDrivenQuakeMaintainer._decide`

*method, lines 323 to 383*

```python
def _decide(self, index: IVFIndex, frac: dict[int, float], delete_factors: dict[int, float], report: MaintenanceReport)
```

Decide every delete and split of the pass from one snapshot of the index.

This follows the decision loop of maintenance_policies.cpp. The partition count, the
access fractions and the average size are fixed before the loop, and nothing is applied
until every decision is taken. Returns the cost driven deletes, the forced deletes and the
splits, the most beneficial first.

The average access is the vectors each routed query scanned over the vectors at handover,
averaged over the window, and the average size is the live total over the partition
count in integer division, as in the released code.

#### `CostDrivenQuakeMaintainer._delete_together`

*method, lines 385 to 406*

```python
def _delete_together(self, index: IVFIndex, pids: list[int], report: MaintenanceReport) -> set[int]
```

Delete the given partitions in one step, and return the ids deleted.

As in Quake's delete_partitions, every listed centroid is removed first and only then are
the vectors of each reinserted, each into its nearest remaining partition, so no vector is
moved twice. Each receiver keeps its own centroid, and at least one partition is always
kept.

#### `CostDrivenQuakeMaintainer._split_and_refine`

*method, lines 408 to 445*

```python
def _split_and_refine(self, index: IVFIndex, frac: dict[int, float], splits: list[int], report: MaintenanceReport) -> None
```

Split the chosen partitions, checking each on its real children, then refine around them.

Splits come after the deletes, so a partition splits what it holds once the deleted
vectors have landed. One refinement then covers all the new children.

#### `CostDrivenQuakeMaintainer.should_maintain`

*method, lines 447 to 454*

```python
def should_maintain(self, index: IVFIndex, step: int) -> bool
```

Return True once any partition has been accessed.

Quake has no trigger of its own. It runs maintenance after every operation and skips a
pass only while the access window is still empty, which an index no query has reached
stands in for here.

#### `CostDrivenQuakeMaintainer.maintain`

*method, lines 456 to 526*

```python
def maintain(self, index: IVFIndex) -> MaintenanceReport
```

Run one Quake pass, refresh, decide, delete, split and refine, collect, reset the window.

#### `CostDrivenQuakeMaintainer._delete_factors`

*method, lines 528 to 542*

```python
def _delete_factors(self, index: IVFIndex) -> dict[int, float]
```

Return the share of its previous size each shrunken partition has lost since the last pass.

Partitions that did not shrink are left out, as Quake's get_delete_factor returns a
negative value for them.

### src/maintainers/bandit.py

780 lines.

The bandit policy, a contextual bandit that picks one of five actions for each partition.

`maintain` is one pass. Every partition is a trial, with a six number context, the set of actions
its guards allow, a LinUCB choice, the action, and a reward from the change in the total Quake cost.

The constructor carries options for experiments. The reported configuration is set in
`scripts/final_sweep.build`, which passes the starting index thresholds, a minimum split size of
100, every partition as a candidate, discounting once per pass, and a coverage penalty weight of
2.0. The parameter named `alpha` here is the exploration weight of LinUCB, written beta in the
thesis, and is unrelated to SPLIT_ACCESS_ALPHA. These options are off in the reported runs, the
query grounded reward, the do no harm margin, cooldowns, the recency feature, the potential based
reward and the warm start.

| Constant | Value |
|---|---|
| `_N_ARMS` | `len(Action)` |
| `_ACTION_COST_MULTIPLIER` | `{Action.SPLIT: 2.0, Action.MERGE: 1.5, Action.CENTROID_UPDATE: 1.0,...` |

#### `Action`

*class, lines 57 to 64*

```python
class Action(IntEnum)
```

The five actions the bandit can take on a partition.

#### `BanditMaintainer`

*class, lines 80 to 780*

```python
class BanditMaintainer(Maintainer)
```

A contextual bandit that learns which action to take on each partition.

#### `BanditMaintainer.__init__`

*method, lines 85 to 213*

```python
def __init__(self, alpha: float = 1.0, discount: float = 0.99, access_decay: float = 0.9, n_candidates: int = 20, n_explore: int = 200, drift_fraction: float = DRIFT_FRACTION, cost_model=None, max_partition_size: int = MAX_PARTITION_SIZE, min_partition_size: int = MIN_PARTITION_SIZE, min_split_size: int = 0, frag_penalty_weight: float = 2.0, check_interval: int = MAINTENANCE_CHECK_INTERVAL, seed: int = 42, eval_queries: np.ndarray | None = None, eval_k: int = 10, eval_nprobe: int = 10, recall_penalty_weight: float = 5.0, act_margin: float | None = None, min_partitions: int | None = None, max_partitions: int | None = None, coverage_penalty_weight: float = 0.0, cooldown_base: int = 0, cooldown_min: int = 1, cooldown_change_frac: float = 0.5, reversal_penalty_weight: float = 0.0, use_recency_feature: bool = False, drift_threshold: float | None = None, recency_window: int = 4, track_reversals: bool = False, potential_reward: bool = False, potential_cost_weight: float = 0.1, nprobe: int = DEFAULT_NPROBE, merge_candidates: int = LIRE_MERGE_CANDIDATES, reward_clip: float = 1.0, discount_mode: str = "arm", warm_start: bool = False) -> None
```

Create the policy.

The main parameters are `alpha`, the exploration weight, `discount`, the forgetting
factor of the learner, `access_decay`, the factor the access counts are multiplied by each
pass, `n_explore`, the number of random decisions at the start, `cost_model`, the scan
cost curve the reward is priced with, the split and merge thresholds, `min_split_size`,
and `coverage_penalty_weight`. `n_candidates` None makes every partition a candidate.

#### `BanditMaintainer.should_maintain`

*method, lines 215 to 229*

```python
def should_maintain(self, index: IVFIndex, step: int) -> bool
```

Return True if a partition is over the split threshold, under the merge threshold or
empty, or if the mean centroid drift is over the drift limit.

Reading the mean drift recomputes the mean of every changed partition, which is charged as
work, directly to the running total since no report exists yet.

#### `BanditMaintainer.maintain`

*method, lines 231 to 371*

```python
def maintain(self, index: IVFIndex) -> MaintenanceReport
```

Run one pass, choosing, applying and learning from one action per candidate partition.

#### `BanditMaintainer._select_candidates`

*method, lines 373 to 407*

```python
def _select_candidates(self, index: IVFIndex) -> list[int]
```

Return the partitions to decide on in this pass.

With `n_candidates` None, which the reported runs use, that is every partition, as Quake
considers every partition each pass. Otherwise it is the top ones by (1 + access) times
size, plus a few of the smallest partitions under the merge threshold, so the merge action
sees the partitions it is meant for.

#### `BanditMaintainer._eval_cost_quality`

*method, lines 409 to 419*

```python
def _eval_cost_quality(self, index: IVFIndex) -> tuple[float, float]
```

Return the mean vectors scanned and the mean neighbour distance on the evaluation queries.

Used only by the optional query grounded reward. Lower is better for both.

#### `BanditMaintainer._warm_start`

*method, lines 421 to 457*

```python
def _warm_start(self, index: IVFIndex, stats, report) -> None
```

Give the learner estimated rewards for every partition and action before it acts.

This is the warm start of ARROW-CB, built from the running index and the cost model. For
every partition and every action it could take, the reward is estimated without acting,
on the same scale as the real reward, see `_whatif`. Not used by the reported runs.

#### `BanditMaintainer._whatif`

*method, lines 459 to 505*

```python
def _whatif(self, action: Action, pid: int, p: Partition, index: IVFIndex, access: float, scale: float, avg_access: float, avg_size: int, drift_scale: float, meter=None) -> float
```

Estimate the reward an action would earn, without taking it.

Refresh, compaction and no action are estimated exactly. A split is priced with Quake
equation 6 and a merge with Quake's uniform delete estimate, which are approximations.

#### `BanditMaintainer._cost_scale`

*method, lines 507 to 521*

```python
def _cost_scale(self, index: IVFIndex, c_before: float) -> float
```

Return the unit the cost change is measured in, the mean scan cost per partition.

That is the scan part of C divided by K. Dividing by the acted partition's own cost
instead would divide by nearly zero for a partition queries rarely reach. Before any query
has been counted the scan part is zero, so nprobe partitions of the mean size are used.

#### `BanditMaintainer._merge_target`

*method, lines 523 to 539*

```python
def _merge_target(self, pid: int, partition: Partition, index: IVFIndex, meter=None) -> int | None
```

Return the merge partner, the nearest partition whose combined size fits, or None.

As in SPFresh section 3.2, the nearest partitions are tried in order. With cooldowns on, a
partition that was just split off is skipped, since merging with it would undo that split.

#### `BanditMaintainer._available_arms`

*method, lines 541 to 564*

```python
def _available_arms(self, pid: int, partition: Partition, index: IVFIndex, meter=None) -> list[int]
```

Return the actions whose guards allow them on this partition.

No action is always allowed, a refresh needs live vectors, compaction needs tombstones,
a split needs at least `min_split_size` vectors, and a merge needs a partner that fits.
Finding a merge partner compares the partition with every centroid, which is charged as
work, since a bandit that considers every partition would otherwise make about K squared
distance computations per pass for free.

#### `BanditMaintainer._action_cost`

*method, lines 566 to 575*

```python
def _action_cost(self, action: Action, before) -> float
```

Return the cost of an action, its weight times the vectors it touches over the split threshold.

A split also pays the fragmentation overhead.

#### `BanditMaintainer._badness`

*method, lines 577 to 583*

```python
def _badness(self, index: IVFIndex) -> float
```

Return the sum of (1 + access) times size squared, for the optional potential reward.

It grows faster than linearly with size, so a split lowers it and undoing one raises it.

#### `BanditMaintainer._op_horizon`

*method, lines 585 to 593*

```python
def _op_horizon(self, base_intervals: float, partition: Partition) -> float
```

Return how long, in updates, a cooldown lasts for this partition.

A partition queried more often than average gets a shorter cooldown, so the policy can
revisit it sooner.

#### `BanditMaintainer._in_cooldown`

*method, lines 595 to 610*

```python
def _in_cooldown(self, pid: int, partition: Partition, reverse_of: str) -> bool
```

Return True if acting now would undo a recent `reverse_of` action on this partition.

Always False when cooldowns are off, as in the reported runs. A partition whose size has
changed a lot since that action is no longer held back.

#### `BanditMaintainer._signed_recency`

*method, lines 612 to 623*

```python
def _signed_recency(self, pid: int, partition: Partition) -> float
```

Return the optional recency feature, positive after a recent split, negative after a merge.

It fades from 1 to 0 over the recency window.

#### `BanditMaintainer._reversal_penalty`

*method, lines 625 to 639*

```python
def _reversal_penalty(self, pid: int, partition: Partition, action: Action) -> float
```

Return the optional penalty for undoing a recent split or merge, zero when it is off.

#### `BanditMaintainer._execute`

*method, lines 641 to 670*

```python
def _execute(self, action: Action, pid: int, index: IVFIndex, report: MaintenanceReport ) -> list[Partition]
```

Apply an action to a partition and return the partitions that result.

Sets `_last_refused` when the action changed nothing, judged by whether the report moved,
so a refusal inside any of the helpers is detected the same way.

#### `BanditMaintainer._do_split`

*method, lines 672 to 713*

```python
def _do_split(self, pid: int, partition: Partition, index: IVFIndex, report: MaintenanceReport ) -> list[Partition]
```

Split a partition with 2-means and reassign the vectors near the new boundary.

The reassignment covers the two children and the 25 nearest partitions of the parent.
Returns the children, or the partition itself if the split was refused.

#### `BanditMaintainer._do_centroid_update`

*method, lines 715 to 726*

```python
def _do_centroid_update(self, pid: int, index: IVFIndex, report: MaintenanceReport ) -> list[Partition]
```

Move a partition's centroid to the mean of its live vectors.

#### `BanditMaintainer._do_merge`

*method, lines 728 to 768*

```python
def _do_merge(self, pid: int, partition: Partition, index: IVFIndex, report: MaintenanceReport ) -> list[Partition]
```

Join a partition with its merge partner into one centred on the mean of both.

Unlike LIRE, where the surviving partition keeps its centroid. Both partitions are
charged, as in every policy that merges. Returns the merged partition.

#### `BanditMaintainer._do_compact`

*method, lines 770 to 780*

```python
def _do_compact(self, pid: int, partition: Partition, index: IVFIndex, report: MaintenanceReport ) -> list[Partition]
```

Compact a partition, charged its stored length. Does nothing if it has no tombstones.

### src/bandit/context.py

56 lines.

The six numbers that describe one partition to the bandit, see `extract_context`.

| Constant | Value |
|---|---|
| `CONTEXT_DIM` | `6` |
| `_CLAMP` | `np.array([2.0, 2.0, 1.0, 1.0, 4.0, 1.0], dtype=np.float64)` |

#### `extract_context`

*function, lines 19 to 56*

```python
def extract_context(partition: Partition, stats: IndexStats, size_cap: int = MAX_PARTITION_SIZE, *, total_access: float, nprobe: int = DEFAULT_NPROBE, drift_scale: float | None = None, queries: float | None = None) -> np.ndarray
```

Return the context vector of a partition, six features each kept within a fixed range.

The features are the size over the split threshold `size_cap`, the centroid drift over
`drift_scale`, the access fraction A, the share of tombstoned vectors, the size over the mean
partition size, and a constant 1. A is the share of routed queries that scanned the
partition, as in Quake section 4.1, the quantity the cost model ranks on. `queries` is the
routed query count, and when it is not given it is estimated as `total_access` over `nprobe`.
`total_access` has no default, because A cannot be computed from one partition alone.

### src/bandit/linucb.py

246 lines.

The LinUCB learners of the bandit policy.

The reported runs use `WarmLinUCB` with `discount_mode="pass"` and no warm start. Each action keeps
discounted sums of the outer products of its contexts and of reward times context, and once per
maintenance pass `end_of_pass` discounts every action by 0.99, which is D-LinUCB with the pass as
the round. The ridge term stays at the identity. `LinUCBAgent` is an earlier learner that discounts
an action only when that action is updated, and the reported runs do not use it.

| Constant | Value |
|---|---|
| `ARROW_LAMBDAS` | `(0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 1.0)` |

#### `LinUCBAgent`

*class, lines 15 to 96*

```python
class LinUCBAgent
```

Disjoint LinUCB with discounted updates, D-LinUCB of Russac, Vernade and Cappe, 2019.

Not used by the reported runs. For each action a, theta_a is V_a inverse times b_a, and the
confidence width is alpha times the square root of x V_a^-1 Vt_a V_a^-1 x, where Vt_a is
discounted at gamma squared. With a discount below 1 this is the width D-LinUCB bounds, while
the plain width of stationary LinUCB would overstate the variance. alpha = 1.0 is the
practical choice of Li et al., 2010, not a theoretical value. Unlike the paper, an action is
discounted only when it is updated, so an action that is rarely chosen forgets more slowly.

#### `LinUCBAgent.__init__`

*method, lines 26 to 42*

```python
def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0, discount: float = 0.99) -> None
```

Create a learner with `n_arms` actions, each starting at the identity ridge term.

#### `LinUCBAgent.scores`

*method, lines 44 to 56*

```python
def scores(self, context: np.ndarray) -> np.ndarray
```

Return the upper confidence score of every action for `context`.

#### `LinUCBAgent.select_arm`

*method, lines 58 to 69*

```python
def select_arm(self, context: np.ndarray, available: list[int] | None = None) -> int
```

Return the action with the highest score, chosen among `available` if it is given.

As in Li et al., Algorithm 1, the choice is over the actions allowed in the current round,
so the learner is never charged for, or taught by, an action that could not be taken.

#### `LinUCBAgent.update`

*method, lines 71 to 86*

```python
def update(self, arm: int, context: np.ndarray, reward: float) -> None
```

Learn the reward of one action, discounting that action's past evidence first.

The ridge term must stay at the identity, as in Li et al., equation 5. Discounting the
whole matrix would shrink it over a run, so (1 - g) times the identity is added back after
each discount, which keeps it exactly at the identity while the data still fades at the
chosen rate. Vt gets the same treatment at g squared.

#### `LinUCBAgent.total_updates`

*method, lines 89 to 91*

```python
def total_updates(self) -> int
```

The number of updates made so far.

#### `LinUCBAgent.theta`

*method, lines 93 to 96*

```python
def theta(self, arm: int) -> np.ndarray
```

Return the current estimate of an action's weight vector.

#### `WarmLinUCB`

*class, lines 106 to 246*

```python
class WarmLinUCB
```

Disjoint LinUCB with an optional warm start, weighted as in ARROW-CB, and a choice of clock.

Each action keeps real feedback in the sums G, G2 and h and warm start estimates in S, S2 and
hs, so the model for a weight lambda is V = I + (1 - lambda) S + lambda G, Vt = I +
(1 - lambda)^2 S2 + lambda^2 G2 and b = (1 - lambda) hs + lambda h, with lambda weighting the
real feedback as in ARROW-CB equation 3. Before each real outcome is learned, the model of
every lambda predicts it, the squared errors add up, and the lambda with the least error
drives the next decision. The reported runs use no warm start and a single lambda of 1.

With `discount_mode` "arm" an action's evidence is discounted when that action is updated.
With "pass", which the reported runs use, every action is discounted once per maintenance
pass whether it was chosen or not, as D-LinUCB discounts every round, so an action left
unchosen regains uncertainty and is tried again. The memory is then measured in stream time,
not in the number of candidates a pass looks at.

#### `WarmLinUCB.__init__`

*method, lines 123 to 151*

```python
def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0, discount: float = 0.99, discount_mode: str = "pass", lambdas: tuple[float, ...] = (1.0,)) -> None
```

Create a learner with `n_arms` actions and no evidence yet.

#### `WarmLinUCB.lam`

*method, lines 154 to 156*

```python
def lam(self) -> float
```

The weight on real feedback currently in use.

#### `WarmLinUCB.total_updates`

*method, lines 159 to 161*

```python
def total_updates(self) -> int
```

The number of real outcomes learned so far.

#### `WarmLinUCB.add_warm`

*method, lines 163 to 173*

```python
def add_warm(self, arm: int, contexts: np.ndarray, rewards: np.ndarray) -> None
```

Add warm start examples for an action, estimated rewards rather than observed ones.

#### `WarmLinUCB._model`

*method, lines 175 to 181*

```python
def _model(self, arm: int, lam: float)
```

Return V, Vt and b of an action for the weight `lam`.

#### `WarmLinUCB.theta`

*method, lines 183 to 186*

```python
def theta(self, arm: int, lam: float | None = None) -> np.ndarray
```

Return the weight vector estimate of an action, for `lam` or the current weight.

#### `WarmLinUCB.scores`

*method, lines 188 to 197*

```python
def scores(self, context: np.ndarray) -> np.ndarray
```

Return the upper confidence score of every action for `context`.

#### `WarmLinUCB.select_arm`

*method, lines 199 to 206*

```python
def select_arm(self, context: np.ndarray, available: list[int] | None = None) -> int
```

Return the action with the highest score, chosen among `available` if it is given.

#### `WarmLinUCB.update`

*method, lines 208 to 228*

```python
def update(self, arm: int, context: np.ndarray, reward: float) -> None
```

Learn the observed reward of one action in one context.

#### `WarmLinUCB.end_of_pass`

*method, lines 230 to 246*

```python
def end_of_pass(self) -> None
```

Discount the evidence of every action once, at the end of a maintenance pass.

Only in "pass" mode. The record of which weight predicts best is discounted too, since the
best weight can change as the index changes.

### src/bandit/reward.py

157 lines.

The reward of the bandit policy, built on the total Quake cost C.

`index_cost` is C, the sum over partitions of min(1, A) times lambda(size), plus lambda_c(K). The
bandit reads C before and after an action, and `compute_reward` turns the change into a reward in
[-1, 1], after subtracting a weighted action cost and adding two small terms, one for drift a
centroid refresh closed and one for tombstones an action removed.

#### `PartitionSnapshot`

*class, lines 17 to 25*

```python
class PartitionSnapshot
```

The state of a partition just before an action, kept to compare with the state after it.

#### `partition_latency_cost`

*function, lines 28 to 34*

```python
def partition_latency_cost(partition: Partition, cost_model=None) -> float
```

Return what a partition costs the workload, access count times lambda(size).

This is Quake equation 1. With no cost model, lambda is the live size.

#### `index_cost`

*function, lines 37 to 58*

```python
def index_cost(index, cost_model=None, nprobe: int = 10) -> float
```

Return the total Quake cost C, the sum of A times lambda(size) plus lambda_c(K).

A is the share of routed queries that scanned the partition, so the partition term and the
centroid term are both per query and can be added. The centroid term is the delta O of Quake
equations 4 and 5, since every query compares itself with all K centroids. With no cost model,
lambda is the vector count.

The bandit scores an action by the change in this total. Scoring only the acted partition
would miss two effects. The reassignment after a split moves vectors and their access into the
children from neighbouring partitions, so the children would be charged for what they gained
while the neighbours were never credited for what they lost. And without the centroid term a
merge could never lower the cost, and nothing would oppose a split.

#### `snapshot`

*function, lines 61 to 70*

```python
def snapshot(partition: Partition, cost_model=None) -> PartitionSnapshot
```

Record the current state of a partition.

#### `combine_snapshots`

*function, lines 73 to 93*

```python
def combine_snapshots(snaps: list[PartitionSnapshot]) -> PartitionSnapshot
```

Combine the snapshots of the partitions a merge consumes into one.

Costs are added, which is the true combined cost whatever the shape of lambda. Drift and access
are averaged, weighted by size.

#### `compute_reward`

*function, lines 96 to 157*

```python
def compute_reward(before: PartitionSnapshot, after_partitions: list[Partition], cost_model=None, action_cost: float = 0.0, drift_scale: float = 0.1, w_cost: float = 0.1, w_drift_guard: float = 0.1, w_compact: float = 0.1, *, delta_index_cost: float | None = None, cost_scale: float | None = None, clip: float | None = None, drift_credit: bool = True) -> float
```

Return the reward of an action.

The reward is the cost reduction, minus `w_cost` times the action cost, plus `w_drift_guard`
times the drift closed over `drift_scale`, plus `w_compact` times the drop in the tombstone
share. With `delta_index_cost`, which the bandit passes, the cost reduction is the change in
`index_cost` across the action divided by `cost_scale`. Otherwise it is the relative change in
the acted partitions' own cost. The drift is divided by a fixed scale rather than by its value
before the action, so closing a small drift does not score the same as closing a large one.
`clip` bounds the reward, which the analysis of LinUCB assumes.

## Measurement

### src/metrics.py

511 lines.

The scoring, how well maintenance worked, never how a policy decides.

Nothing in `src/maintainers/` or `src/bandit/` reads this module. Keeping the two apart matters
because the word cost names three different things in this code.

1. Scored cost, computed here. The reported query cost has three parts, the nprobe at which recall
   on the held out queries reaches the target (`costs_at_target_recalls`), the Faiss price of every
   list probed at that nprobe (`priced_cost_at`), and a price per centroid that
   `src/calibration.query_cost` adds. The number of vectors scanned (`query_scan_cost`) is kept
   beside it as the raw count. Every search here runs with `record_access=False`, so measuring
   never changes the access counts the policies read, and no policy can see the scored cost.
2. Policy cost, in `src/cost_model.py` and inside the policies. The scan cost curve lambda(s) and
   the quantities built on it, the delta C of the Quake policy and the reward of the bandit. These
   are part of a policy, how it chooses what to do, and several policies use no cost model at all
   and decide on size or a schedule alone.
3. Work, the counters of `MaintenanceReport`, what running the maintenance itself cost, collected
   during the stream rather than here.

A policy cost model may disagree with the scored cost, and keeping them apart is what lets a
disagreement show. They share one price curve, built from the scoring price list by
`cost_model.lambda_from_prices`, so a policy decides in the unit it is scored in.

#### `recall_at_k`

*function, lines 38 to 47*

```python
def recall_at_k(index: IVFIndex, queries: np.ndarray, ground_truth: np.ndarray, k: int, nprobe: int) -> float
```

Return the mean recall at `k` over `queries` at a fixed `nprobe`.

#### `compute_live_ground_truth`

*function, lines 50 to 79*

```python
def compute_live_ground_truth(index: IVFIndex, queries: np.ndarray, k: int) -> np.ndarray
```

Return the ids of the exact `k` nearest live vectors of every query.

The published ground truth refers to the original dataset, so after inserts and deletes it
would mix index quality with neighbours that were deleted or never inserted. This computes the
exact neighbours over what the index holds right now.

#### `recall_at_k_live`

*function, lines 82 to 93*

```python
def recall_at_k_live(index: IVFIndex, queries: np.ndarray, k: int, nprobe: int) -> float
```

Return the mean recall at `k` at a fixed `nprobe`, against the live ground truth.

#### `recall_at_k_per_query`

*function, lines 96 to 118*

```python
def recall_at_k_per_query(index: IVFIndex, queries: np.ndarray, ground_truth: np.ndarray, k: int, nprobe: int) -> np.ndarray
```

Return the recall at `k` of each query, the share of its true neighbours that were found.

#### `query_scan_cost`

*function, lines 121 to 141*

```python
def query_scan_cost(index: IVFIndex, queries: np.ndarray, nprobe: int, include_centroids: bool = False, stored: bool = False) -> float
```

Return the mean number of vectors a query scans at `nprobe`.

With `include_centroids` the K centroid comparisons every query makes are added, each counted
as one distance like a vector comparison. With `stored` the tombstoned vectors are counted as
scanned too, the SPFresh delete model. The default counts live vectors only, the delete model
of Quake and Faiss, which is the reported one.

#### `priced_scan_cost`

*function, lines 144 to 162*

```python
def priced_scan_cost(index: IVFIndex, queries: np.ndarray, nprobe: int, stored: bool = False) -> float
```

Return the mean Faiss price of the lists a query probes at `nprobe`.

Each probed list is priced at its length by the measured curve of `src/calibration.py`,
including the cost of opening it. The probed lists are the same as in `query_scan_cost`, which
counts their vectors instead. With `stored` each list is priced at its length with tombstones.

#### `_read_between`

*function, lines 165 to 184*

```python
def _read_between(read: Callable[[int], float], nprobe: float, n_parts: int, nprobe_grid: tuple[int, ...] | None = None) -> float
```

Return a reading at a fractional nprobe, interpolated between the two nprobe values around it.

The two values are the integers either side by default, or the surrounding points of an
explicit grid, the same bracket the counted cost was interpolated on.

#### `priced_cost_at`

*function, lines 187 to 195*

```python
def priced_cost_at(index: IVFIndex, queries: np.ndarray, nprobe: float, stored: bool = False, nprobe_grid: tuple[int, ...] | None = None) -> float
```

Return the priced scan at the nprobe that met the recall target.

It is interpolated on the same bracket as the count of scanned vectors, so the priced and the
counted reading describe the same operating point.

#### `stored_cost_at`

*function, lines 198 to 207*

```python
def stored_cost_at(index: IVFIndex, queries: np.ndarray, nprobe: float, nprobe_grid: tuple[int, ...] | None = None) -> float
```

Return the scan count at the nprobe that met the recall target, with tombstones scanned.

This is the SPFresh delete model, reported beside the live cost. A deleted vector is filtered
out after it is read, so recall is the same under both models and so is the nprobe that meets
the target, and the two readings differ by the dead vectors alone.

#### `cost_at_target_recall`

*function, lines 210 to 265*

```python
def cost_at_target_recall(index: IVFIndex, queries: np.ndarray, k: int, target_recall: float = 0.9, nprobe_grid: tuple[int, ...] | None = None) -> tuple[float, float, float, float]
```

Return the scan cost at the nprobe where mean recall reaches `target_recall`.

A cost at a fixed nprobe can be gamed, since splitting partitions lowers it while recall
silently drops. So the nprobe is raised until recall meets the target, and the cost is read
there, interpolated between the two neighbouring nprobe values. Returns the cost, the
interpolated nprobe, the recall reached and the spread of the per query recall.

By default the two values are found by doubling nprobe from 1 until recall meets the target
and then bisecting inside the last doubling. Bisection is valid because recall never falls as
nprobe rises, since the partitions probed at n are among those probed at n plus 1. An explicit
`nprobe_grid` walks that grid instead.

The K centroid comparisons are left out of this raw count, as in the DeDrift efficiency
metric. The reported query cost adds them back at their calibrated price.

The recall spread is measured at the first nprobe that met the target, since no per query
values exist between two measured nprobe values.

#### `costs_at_target_recalls`

*function, lines 268 to 283*

```python
def costs_at_target_recalls(index: IVFIndex, queries: np.ndarray, k: int, targets: tuple[float, ...]) -> dict[float, tuple[float, float, float, float]]
```

Return `cost_at_target_recall` for several targets, one result per target.

The ground truth is computed once and every recall pass is reused across targets, so the
readings at 0.8 and 0.95 beside the main 0.9 cost only a few extra passes. Each result equals
what `cost_at_target_recall` returns for that target alone.

#### `cost_rebuilt_at_own_k`

*function, lines 286 to 311*

```python
def cost_rebuilt_at_own_k(index: IVFIndex, queries: np.ndarray, k: int, target_recall: float = 0.9, seed: int = 0) -> dict
```

Return the cost that a fresh exact k-means over the live vectors would have at the same K.

Part of any method's gain comes from the number of partitions it reaches. This reading keeps
that number and replaces the partitions with a fresh build, so the maintained cost minus this
one is what a rebuild at that K would still have saved. The rebuild goes into a new index, so
`index` is not changed. It is read once at the end of a run, since the final index is not kept.

#### `_cost_at_target_every_integer`

*function, lines 314 to 352*

```python
def _cost_at_target_every_integer(index: IVFIndex, queries: np.ndarray, live_gt: np.ndarray, k: int, target_recall: float, n_parts: int, measured: dict[int, np.ndarray] | None = None) -> tuple[float, float, float, float]
```

The doubling and bisection search of `cost_at_target_recall`.

Every recall pass is measured once and kept in `measured`, so neither the bisection nor a
second target repeats one.

#### `adaptive_query_cost`

*function, lines 355 to 382*

```python
def adaptive_query_cost(index: IVFIndex, queries: np.ndarray, k: int, target_recall: float = 0.9, radius_factor: float = 1.0) -> tuple[float, float, float]
```

Return the cost at the target recall using a per query adaptive nprobe.

Uses `IVFIndex.search_adaptive` for every query. Returns the mean vectors scanned, the mean
recall and the mean number of partitions probed. Not used by the reported runs.

#### `amortized_cost`

*function, lines 385 to 390*

```python
def amortized_cost(partition: Partition) -> float
```

Return access count times live size, a simple measure of what a partition costs queries.

Not used by the reported runs.

#### `partition_quality`

*function, lines 393 to 414*

```python
def partition_quality(partition: Partition, cost_model: CostModel | None = None) -> float
```

Return a quality score of a partition, lower is healthier. Not used by the reported runs.

With a cost model it is the predicted latency, otherwise a weighted sum of size, squared drift
and access count.

#### `index_quality`

*function, lines 417 to 442*

```python
def index_quality(index: IVFIndex, cost_model: CostModel | None = None) -> QualityReport
```

Return the quality score over all partitions and the worst one. Not used by the reported runs.

#### `size_imbalance`

*function, lines 445 to 450*

```python
def size_imbalance(index: IVFIndex) -> float
```

Return the largest partition size over the mean, 1.0 when perfectly balanced.

#### `reconstruction_error`

*function, lines 453 to 483*

```python
def reconstruction_error(index: IVFIndex, sample_size: int = 10_000, seed: int = 0) -> float
```

Return the mean squared distance from sampled vectors to their centroid.

Not used by the reported runs. The sample takes at least one vector from every non empty
partition, so small partitions are over represented, more so the more an index is
fragmented.

#### `query_latency_percentiles`

*function, lines 486 to 511*

```python
def query_latency_percentiles(index: IVFIndex, queries: np.ndarray, nprobe: int, k: int) -> dict[str, float]
```

Return the mean and the 50th, 95th and 99th percentile search time in milliseconds.

This is a wall clock measurement, so it enters no reported number, and the reported runs do
not call it. It records no access.

## Workload

### src/staple.py

287 lines.

Grow the starting index under a maintenance policy instead of building it with k-means.

Global k-means puts no limit on partition size, so it cannot balance partitions when a partition
holds fewer vectors than the space has dimensions. On GIST, with about 100 vectors per partition
in 960 dimensions, it leaves 14.6 percent of partitions below the merge threshold and the
smallest with a single vector, and no choice of the number of vectors, K or the average size
avoids that. An index grown from empty under a maintenance policy is bounded in both directions by
construction, because a split leaves two halves of about half the split threshold and inserts only
ever add. On GIST at 200,000 vectors with the same K and thresholds, the grown index has a smallest
partition of 82 vectors and none outside the thresholds, where exact Lloyd gives 1 and 45.

The arrivals follow the NeurIPS 2023 Big ANN streaming runbook, inserts and deletes local to one
cluster at a time, in shuffled Dirichlet burst rounds. Starting from empty, the first burst into
a cluster creates structure, which is how the published generator works.

The construction differs from Big ANN in two declared ways. It deletes one vector for every four
inserted, instead of Big ANN's delete fraction of 0.5 to 0.9, which would consume about 667,000
vectors to reach 200,000 live ones, while the arrival schedule itself is kept. And the index is
maintained by LIRE, SPFresh section 3.2, which reacts to size alone and never reads query access,
so the starting index carries no query distribution.

The result, a `Staple`, keeps the size thresholds it was grown to satisfy, and callers read
`Staple.hi` and `Staple.lo` instead of deriving thresholds from the final average size. Ada-IVF
also publishes its size limits as fixed constants rather than as multiples of an average.

| Constant | Value |
|---|---|
| `N_WORKLOAD_CLUSTERS` | `64` |
| `WORKLOAD_LABEL_CACHE_DIR` | `DATA_DIR / 'workload_labels'` |

#### `workload_labels`

*function, lines 50 to 79*

```python
def workload_labels(vectors: np.ndarray, n_clusters: int = N_WORKLOAD_CLUSTERS, seed: int = 42 ) -> np.ndarray
```

Assign every base vector to a workload cluster, stored on disk after the first call.

Workload clusters are the regions a burst of arrivals falls into, and they are not the index
partitions. They depend on the data and the seed alone, so every policy in a comparison sees
the same stream. The clusters are fitted with k-means on a sample of 100,000 vectors, and
every vector is then assigned to the nearest cluster centre.

#### `clustered_runbook`

*function, lines 82 to 122*

```python
def clustered_runbook(spare_per_cluster: np.ndarray, n_inserts: int, rng: np.random.Generator, insert_delete_ratio: float = 1.0, n_rounds: int = 5, alpha: tuple[float, ...] = (100.0, 15.0, 10.0, 5.0, 3.0), delete_fraction: tuple[float, float] = (0.5, 0.9)) -> list[tuple[str, int, int]]
```

Return the Big ANN clustered arrival schedule as a list of batches.

Each cluster draws Dirichlet weights over the rounds, one of them much larger than the rest,
and the weights are shuffled per cluster, so each cluster has one large burst in a round chosen
at random. Each batch is an
operation kind, insert or delete, a cluster and a count. Deletes follow each round's inserts,
a random fraction of them divided by `insert_delete_ratio`.

#### `Staple`

*class, lines 126 to 160*

```python
class Staple
```

A grown starting index, with everything a measured stream needs to continue from it.

`rng` is the generator the construction used, passed on rather than reseeded. The stream draws
its cluster walk from the state the construction left, so a fresh generator would give a
different stream.

`members` holds the vector rows of each workload cluster in arrival order, and `cursor` the
next unused position in each. `age_ids` lists every inserted vector in insertion order, and
`alive` marks which are still live, so the list is always sorted oldest first.

#### `Staple.sizes`

*method, lines 154 to 156*

```python
def sizes(self) -> np.ndarray
```

Return the live size of every partition of the grown index.

#### `Staple.surviving_ids`

*method, lines 158 to 160*

```python
def surviving_ids(self) -> list[int]
```

Return the ids of the live vectors, oldest first.

#### `grow_staple`

*function, lines 163 to 266*

```python
def grow_staple(base: np.ndarray, labels: np.ndarray, cap: int = 50_000, target_avg: int = 100, bootstrap: int = 5_000, build_ratio: float = 4.0, window: int = 8, seed: int = 42, n_clusters: int = N_WORKLOAD_CLUSTERS) -> Staple
```

Grow an index from a small k-means start to `cap` live vectors under LIRE.

The first `bootstrap` vectors, spread over every workload cluster, are partitioned with
k-means at `target_avg` vectors per partition. The rest arrive by the clustered schedule, and
LIRE maintains the index as it grows. The split threshold is 4 times `target_avg`, and the
merge threshold is the split threshold divided by `window`. The reported runs use a window of
8, the ratio SPFresh publishes, 80 over 10, which with a target of 100 gives 400 and 50.

#### `sliding_window_cohorts`

*function, lines 269 to 287*

```python
def sliding_window_cohorts(staple: Staple, max_t: int ) -> dict[int, list[int]]
```

Schedule the vectors of the starting index to expire over the first half of the stream.

The starting index is the content of the retention window at time 0, not a permanent layer
beneath it. Left permanent, it would give every partition a floor it could never fall below.
It is split into `max_t // 2` groups, oldest first, one expiring per time step, the same lag
every later batch expires at. Returns a map from time step to the ids that expire then.

### src/stream.py

308 lines.

The retention window stream that every reported number is measured on.

A run starts from a grown starting index (`src/staple.py`) and continues with a Big ANN style
sliding window. Arrivals walk the 64 workload clusters in a seeded random order, with a Gaussian
spread of `width` clusters around the current position, and every batch expires `max_t // 2` time
steps after it arrived, so one region drains while another fills. The starting index itself
expires first, over the same lag. Every update is followed by `query_ratio` routed queries from a
pool that concentrates on one region of the query space, and those queries are the access signal
the Quake and bandit policies read.

The order of the random draws matters. Every policy starts from a copy of the same starting index
and continues its generator, so the cluster walk, the inserts and the expiry schedule are the same
for every policy. The second draw in `run_stream` is a shuffle whose result is never read, kept
because it advances the generator, and removing it would change every arrival.

| Constant | Value |
|---|---|
| `CHECKPOINT_FRACTIONS` | `(0.25, 0.5, 0.75)` |
| `ROUTE_TARGET_RECALL` | `0.9` |
| `ROUTE_CALIBRATION_QUERIES` | `100` |
| `EXTRA_TARGETS` | `{0.8: 'r80', 0.95: 'r95'}` |
| `ROUTE_CALIBRATION_INTERVAL` | `MAINTENANCE_CHECK_INTERVAL` |

#### `recall_matched_nprobe`

*function, lines 56 to 86*

```python
def recall_matched_nprobe(index, queries: np.ndarray, start: int, target: float = ROUTE_TARGET_RECALL, k: int = DEFAULT_K) -> int
```

Return the smallest nprobe at which mean recall at `k` on `queries` meets `target`.

The search starts from `start`, the previous window's value, and moves one step at a time,
since the index changes little between windows. The ground truth is exact over the live
vectors. Nothing here records access, so finding the nprobe cannot change the signal it is
found for.

#### `query_pool`

*function, lines 89 to 103*

```python
def query_pool(queries: np.ndarray, seed: int, hot_fraction: float, hot_prob: float, eval_queries: int, pool_size: int = 4000) -> tuple[np.ndarray, np.ndarray]
```

Return a routed query pool and a scoring sample drawn from that same pool.

The sample is not held out, so a policy that learns from access would be scored on queries it
learned from. The reported runs use `split_query_pools` instead. Not used by the reported runs.

#### `query_region`

*function, lines 106 to 117*

```python
def query_region(queries: np.ndarray, seed: int, fraction: float) -> tuple[int, np.ndarray]
```

Return the region the concentrated queries come from, the `fraction` nearest to one query.

The centre query is drawn from the seed, so where the concentration falls also varies across
seeds. Quake section 7.1 and Ada-IVF section 5.1.2 both concentrate their skewed workloads
spatially in this way. Returns the centre and the region, nearest first, ties broken by query
id.

#### `split_query_pools`

*function, lines 120 to 154*

```python
def split_query_pools(queries: np.ndarray, seed: int, hot_fraction: float, hot_prob: float, eval_queries: int, pool_size: int = 4000)
```

Return the routed pool, a sample of it, and a held out sample that is never routed.

The query ids are split in two halves, the region and the rest alike. The stream routes a
mixture drawn from the first half, `hot_prob` of it from that half of the region and the rest
from the whole half. The scoring reads a mixture drawn the same way from the second half,
which the stream never routes, so no policy is scored on a query it learned from, and both
halves concentrate on the same region. The sample of the routed pool is kept so the gap
between the two can be reported for every policy.

#### `measure`

*function, lines 157 to 187*

```python
def measure(index, eval_q: np.ndarray, t: int, extra_targets: bool = True) -> dict
```

Read the scoring at time step `t` and return it as a dictionary.

The main reading is the cost at recall 0.9, at the nprobe that meets it, since a cost at a
fixed nprobe can be gamed by splitting. With `extra_targets` the cost at 0.8 and 0.95 is read
too, on the same ground truth. Also returns the raw recall at the default nprobe, the cost
with tombstones scanned, the partition count and the live and stored sizes, and, when a price
list exists, the same costs priced in Faiss time.

#### `run_stream`

*function, lines 190 to 308*

```python
def run_stream(staple0: Staple, base: np.ndarray, maint, qpool: np.ndarray, eval_q: np.ndarray, seed: int, measure_ops: int = 120_000, max_t: int = 100, width: float = 3.0, query_ratio: int = 8, log: Callable[[dict], None] | None = None, served_q: np.ndarray | None = None) -> dict
```

Run one policy over the retention window, starting from a copy of `staple0`.

Each of the `max_t` time steps inserts about `measure_ops // max_t` vectors around the current
workload cluster and deletes the vectors due to expire. Every update is followed by
`query_ratio` routed queries and a chance for the policy to act, and the routing nprobe is
recalibrated every 1,000 updates. Returns the readings at t = 0, at the checkpoints and at the
end, the policy, the final index and the counts of the stream. `log` is called with each
reading as it is taken, and with `served_q` every reading is repeated on that sample under a
served_ prefix.

## Support

### src/provenance.py

242 lines.

A fingerprint for every result row, so a run can be resumed and two code versions never mixed.

A run is fully determined by its seed and reproduces exactly, but only while the code, the machine
and the run parameters stay the same. A row that records none of these cannot be checked, so each
row carries a fingerprint of them. That is what lets a set of runs stop halfway and resume, and
what lets a merge of two result files refuse rows produced by different code.

The fingerprint rests on a hash of the file contents, not on version control state, so an
uncommitted edit changes it exactly as a commit does. The git fields are recorded for people to
read and are not part of the fingerprint.

| Constant | Value |
|---|---|
| `_CODE_GLOBS` | `('src/**/*.py', 'scripts/final_sweep.py')` |
| `_CODE_GLOB_ROOTS` | `('src', 'scripts/final_sweep.py')` |
| `_MODEL_GLOBS` | `('cost_model.yaml', 'profiled_lambda_dim*.yaml')` |
| `_THREAD_VARS` | `('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')` |
| `PARAM_KEYS` | `('dataset', 'n_ops', 'initial_fraction', 'query_ratio', 'hot_fracti...` |
| `_PRIVATE_MODULES` | `{'no_op': ('src/maintainers/no_op.py',), 'global_rebuild': ('src/ma...` |
| `_LAMBDA_READERS` | `('cost_driven_quake', 'cost_driven_quake_tau50', 'bandit')` |

#### `_family`

*function, lines 59 to 71*

```python
def _family(maintainer: str) -> str | None
```

Return the module group a policy name belongs to, or None for a policy with no own module.

LIRE has no own module here because its file is shared, see _PRIVATE_MODULES.

#### `_code_bytes`

*function, lines 74 to 95*

```python
def _code_bytes(p: Path) -> bytes
```

Return the bytes of a file as they enter the hash.

A Python file is hashed as its syntax tree with docstrings removed, so editing a comment or a
docstring never changes a fingerprint. Any other file, or a Python file that does not parse,
is hashed as raw bytes with Windows line endings normalised, so the same file hashes the same
on Windows and Linux.

#### `_hash_files`

*function, lines 98 to 104*

```python
def _hash_files(paths: list[Path]) -> str
```

Return one hash over the relative paths and contents of `paths`, in sorted order.

#### `code_hash`

*function, lines 107 to 125*

```python
def code_hash(maintainer: str | None = None) -> str
```

Return the hash of the code that can affect a policy's rows.

With no policy it covers every source file. With one, it covers the shared files plus that
policy's own modules, and never another policy's modules.

#### `scoreboard_hash`

*function, lines 128 to 131*

```python
def scoreboard_hash() -> str
```

Return the hash of the calibrated price list, which prices every row of every policy.

#### `cost_model_hash`

*function, lines 134 to 139*

```python
def cost_model_hash() -> str
```

Return the hash of the stored cost model files, or "absent" if there are none.

#### `_git`

*function, lines 142 to 149*

```python
def _git(*args: str) -> str
```

Run a git command in the project folder and return its output, or "unknown" on failure.

#### `machine_id`

*function, lines 152 to 171*

```python
def machine_id() -> str
```

Return a description of the machine and of the library versions that affect results.

The machine name alone is not enough, since the library versions are what change the
partitioning between machines. Faiss is included because the ground truth uses it when it is
installed and falls back to scikit-learn otherwise, and the two can order exact ties
differently.

#### `provenance`

*function, lines 174 to 194*

```python
def provenance(params: dict, maintainer: str | None = None) -> dict
```

Return everything that decides whether two rows of a policy are comparable.

Raises KeyError if a run parameter is missing from `params`.

#### `fingerprint`

*function, lines 197 to 208*

```python
def fingerprint(params: dict, maintainer: str | None = None) -> tuple[str, dict]
```

Return the fingerprint of a policy's rows and the provenance it was computed from.

The git fields are left out of the fingerprint, so an identical tree has the same fingerprint
before and after a commit. The policy name is part of it, so two policies never share one.

#### `sidecar_path`

*function, lines 211 to 213*

```python
def sidecar_path(output: Path) -> Path
```

Return the path of the provenance file that goes with a result file.

#### `write_sidecar`

*function, lines 216 to 231*

```python
def write_sidecar(output: Path, digest: str, prov: dict) -> None
```

Add the provenance of a fingerprint to the provenance file of a result file.

Entries are added rather than the file overwritten, so a resumed result file keeps the
details of every fingerprint it contains.

#### `read_sidecar`

*function, lines 234 to 242*

```python
def read_sidecar(output: Path) -> dict
```

Return the provenance file of a result file, or an empty dictionary if there is none.

### src/__init__.py

3 lines.

The IVF index, the maintenance policies and the measurement code of the thesis.

### src/bandit/__init__.py

14 lines.

The parts of the bandit policy, its context features, its LinUCB learner and its reward.

### src/maintainers/__init__.py

19 lines.

The maintenance policies compared in the thesis, all built on the `Maintainer` interface.

## Scripts

### scripts/final_sweep.py

498 lines.

Run the experiments, every combination of dataset, seed and policy, into one result file.

Each run grows or reuses the starting index of its seed, runs one policy over the retention window
stream of `src/stream.py`, and writes one row with every reading, the work and the operation counts.
The file is written after every run, each row carries a fingerprint of the code, machine and
parameters, and `--resume` reruns only what is missing, so a long set of runs can be stopped and
restarted. The exact commands behind the reported results are in the README.

| Constant | Value |
|---|---|
| `MAINTAINERS` | `['no_op', 'global_rebuild', 'lire_lite', 'dedrift_lazy', 'dedrift_s...` |
| `DEDRIFT_VARIANTS` | `['dedrift_lazy', 'dedrift_split', 'dedrift_hybrid']` |
| `ABLATION_VARIANTS` | `['bandit_linear', 'lire_lite_no_merge']` |
| `SENSITIVITY_VARIANTS` | `['cost_driven_quake_tau50']` |
| `QUAKE_TAU_ARM_NS` | `50.0` |
| `_PROFILED` | `{}` |
| `DEDRIFT_N_LARGEST` | `None` |
| `COST_AWARE` | `('cost_driven_quake', 'cost_driven_quake_tau50', 'bandit')` |

#### `profile_path`

*function, lines 75 to 77*

```python
def profile_path(dim: int) -> Path
```

Return the path of the stored scan cost curve for vectors of `dim` dimensions.

#### `profiled_cost_model`

*function, lines 80 to 99*

```python
def profiled_cost_model(dim: int, reprofile: bool = False)
```

Return the scan cost curve lambda(s) for `dim` dimensions, read once from its stored file.

The stored curve is the Faiss price curve that scripts/calibrate_query_cost.py writes, the same
one the scoring prices with. It is never measured here, because a curve measured again would
differ, and every run must read the same one.

#### `build`

*function, lines 105 to 148*

```python
def build(name: str, seed: int, hi: int, lo: int, cost_model)
```

Create the policy called `name`, configured as in the reported runs.

`hi` and `lo` are the split and merge thresholds the starting index was grown to satisfy. The
bandit's minimum split size is `hi` over 4, the target average partition size.

#### `run_one_staple`

*function, lines 154 to 278*

```python
def run_one_staple(base, name, seed, staple, qpool, eval_q, args, served_q=None) -> dict
```

Run one policy on one seed over the retention window stream, and return its result row.

Every policy of a seed runs from a copy of one starting index and sees the same stream.
`eval_q` is the held out scoring sample and `served_q` a sample of the routed pool, see
`src/stream.split_query_pools`. Every cost and work column is priced by `src/calibration.py`.

#### `aggregate_and_plot`

*function, lines 281 to 321*

```python
def aggregate_and_plot(df: pd.DataFrame, figures_dir: Path, write_figures: bool = True) -> None
```

Print the mean and spread of cost and work per policy, and optionally plot them.

The figure is written only when asked, because a run over part of the policies would
otherwise overwrite the figure of the full set with a partial one.

#### `main`

*function, lines 324 to 494*

```python
def main() -> None
```

Parse the arguments, run every requested combination, and write the result file.

### scripts/analyse_final.py

314 lines.

Tables and figures from the main results, every number priced by src/calibration.py.

    python scripts/analyse_final.py --sweep results/raw/final_2709 --out results/final_2709

Reads the result files of one set of runs, checks that no run is duplicated and that each policy
carries one fingerprint, and writes

    ranking_<dataset>.csv       per policy, the mean over seeds of every reported column and the
                                paired comparison against no maintenance, see below
    wins_<dataset>.csv          per pair, on how many seeds the row policy beats the column one
    centroid_sensitivity.csv    the query cost at the calibrated centroid price and at a price of 1
    tombstone_<dataset>.csv     the same ranking under the SPFresh delete model, see below
    equal_k_<dataset>.csv       each policy's own index against a fresh build at its own K, so how
                                it maintains is read apart from the K it reaches
    control_<dataset>.csv       a control run from --control, paired with the policy it controls on
                                the same seed, the rebuild at the bandit's K
    frontier.pdf, frontier.png  query cost against maintenance work, one panel per dataset

Efficiency is the query cost a policy saved against no maintenance over the whole workload,
divided by the work it spent, so a unit of maintenance is judged by what it bought. The break even
read rate is how many queries per update it takes before that saving repays the work, infinite
when there is no saving.

The reported delete model removes a deleted vector at once, as Quake and Faiss do, so the query
cost reads live vectors and the work leaves out compaction, which gains nothing under that model.
The SPFresh model, where a deleted vector is scanned until compaction removes it, is the
sensitivity check beside it.

What is reported, and how it is compared, was fixed before the main runs. The main result is the
end of run query cost at recall 0.9. Each query cost is compared with no maintenance within one
dataset and seed as a log ratio, then over seeds with a paired t interval, every per seed value
and the count of seeds below no maintenance. The ranking is ordered by that paired ratio, and
datasets are never pooled.

| Constant | Value |
|---|---|
| `DIMS` | `{'sift1m': 128, 'gist1m': 960}` |
| `PAIRED_COLS` | `('query_cost', 'query_cost_mean', 'query_cost_r80', 'query_cost_r95...` |
| `RANKING_COLS` | `['query_cost', 'query_cost_mean', 'served_query_cost', 'work_calibr...` |
| `LABELS` | `{'no_op': 'no maintenance', 'global_rebuild': 'global rebuild', 'li...` |

#### `load`

*function, lines 68 to 88*

```python
def load(sweep: Path) -> dict[str, pd.DataFrame]
```

Read every result file in `sweep`, one table per dataset.

Stops if a run appears twice or a policy carries two fingerprints, which would mean rows from
different code versions.

#### `enrich`

*function, lines 91 to 113*

```python
def enrich(d: pd.DataFrame, dim: int) -> pd.DataFrame
```

Add the columns derived against no maintenance on the same seed.

These are the total saving, the efficiency, the break even read rate and the totals relative
to no maintenance. Efficiency and the break even rate cover the whole workload, so they use
the mean query cost over the run, not the end of run value.

#### `paired_vs_no_op`

*function, lines 116 to 125*

```python
def paired_vs_no_op(d: pd.DataFrame, col: str = "query_cost") -> pd.DataFrame
```

Compare every policy with no maintenance on the same seed, for one cost column.

Per seed, the log of a policy's cost over that of no maintenance, then the mean over seeds
with a paired t interval and the count of seeds below no maintenance. The log makes a saving
and a loss by the same factor symmetric and removes the level each seed sets. With five seeds,
even five of five below is an exact two sided sign test p of 0.0625.

#### `_pct`

*function, lines 128 to 130*

```python
def _pct(x)
```

Turn a log ratio into a percent change.

#### `summarise`

*function, lines 133 to 146*

```python
def summarise(logs: pd.DataFrame, prefix: str, head: str) -> pd.DataFrame
```

Summarise log ratios per seed as a mean with a paired t interval, all in percent.

Also returns the count of seeds below zero and every per seed value.

#### `equal_k`

*function, lines 149 to 157*

```python
def equal_k(d: pd.DataFrame) -> pd.DataFrame
```

Compare each policy's own index with a fresh exact k-means at the K it ended at.

How well a policy maintains is then read apart from the K it reaches. Below zero means its own
index is cheaper than the fresh build.

#### `control`

*function, lines 160 to 174*

```python
def control(d: pd.DataFrame, name: str, ref: str) -> pd.DataFrame
```

Compare a control run with a reference policy on the same seed.

The comparison covers the end of run query cost, the mean over the run and the work, each
summarised as log ratios.

#### `tombstone`

*function, lines 177 to 181*

```python
def tombstone(d: pd.DataFrame) -> pd.DataFrame
```

Return the ranking under the SPFresh delete model, tombstones scanned, means over seeds.

#### `ranking`

*function, lines 184 to 198*

```python
def ranking(d: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame
```

Return the mean over seeds of every reported column, with the paired comparisons.

The order is the paired ratio of the main result against no maintenance. The mean absolute
cost is not used as the order, because a seed where every method is expensive would dominate
it.

#### `wins`

*function, lines 201 to 205*

```python
def wins(d: pd.DataFrame, col: str = "query_cost") -> pd.DataFrame
```

Return, for every pair of policies, on how many seeds the row policy is cheaper.

#### `centroid_sensitivity`

*function, lines 208 to 221*

```python
def centroid_sensitivity(d: pd.DataFrame, ds: str, calibrated: float) -> pd.DataFrame
```

Recompute the query cost at the calibrated centroid price and at a price of 1, and rank both.

The centroid term grows with K, so it is the price a ranking between methods that reach
different K depends on.

#### `frontier`

*function, lines 224 to 245*

```python
def frontier(tables: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame], out: Path) -> None
```

Plot query cost against maintenance work, one panel per dataset, the range over seeds as bars.

#### `main`

*function, lines 248 to 310*

```python
def main() -> None
```

Read the results, write every table and the figure, and print the summaries.

### scripts/calibrate_query_cost.py

234 lines.

Measure the prices of the cost unit, on Faiss IndexIVFFlat rather than on the Python index.

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

| Constant | Value |
|---|---|
| `CALIBRATION_PATH` | `COST_MODEL_DIR / 'query_cost_calibration.yaml'` |
| `LIST_SIZES` | `(0, 5, 10, 25, 50, 100, 200, 400, 800, 1600)` |
| `MAX_LISTS` | `2048` |
| `NPROBES` | `(1, 2, 4, 8, 16, 32, 64)` |
| `CENTROID_COUNTS` | `(128, 256, 512, 1024, 2048)` |
| `REPEATS` | `5` |

#### `calibrate`

*function, lines 56 to 125*

```python
def calibrate(base: np.ndarray, queries: np.ndarray, seed: int = 0) -> dict
```

Measure the list price curve and the centroid price once, and return them in scan distances.

Each measurement times whole loops of single query calls, one loop for the full search and
one for the centroid ranking alone, so the Python overhead of each call cancels in the
difference. The minimum over five loops is kept, since an interruption only ever adds time.

#### `calibrate_work`

*function, lines 128 to 173*

```python
def calibrate_work(base: np.ndarray, ns_per_scan_distance: float) -> dict
```

Measure the price of each kind of maintenance operation, in scan distances.

Faiss k-means runs a fixed number of Lloyd iterations, so its time divides exactly by
iterations times points times centroids. It is measured at two shapes, because batching gains
very differently. A split or a DeDrift step clusters a few hundred points into a handful of
centroids, and a rebuild clusters the whole index into hundreds. A vector read or copy is timed
on contiguous blocks the size of a partition, since a partition is stored as one block.

#### `main`

*function, lines 179 to 221*

```python
def main() -> None
```

Calibrate both datasets REPEATS times, store the medians and ranges, and write the curves.

With --lambda-only the curves are rebuilt from the stored prices and nothing is measured.

#### `write_lambda`

*function, lines 224 to 230*

```python
def write_lambda(dim: int, med: dict) -> None
```

Write the scan cost curve for `dim` dimensions, which is the Faiss list price curve itself.

### scripts/control_rebuild_at_k.py

79 lines.

The control run, the global rebuild at the partition count the bandit ended at on the same seed.

    python scripts/control_rebuild_at_k.py --seeds 42 1 7 13 23

The policies that keep K fixed hold about 176 partitions on SIFT, while the bandit grows to 408 to
546, so part of the bandit's lead may be the K it reaches rather than how it maintains partitions.
This runs the unmodified global rebuild on its usual trigger, with the partition count set to the
bandit's final K for each seed, so its query cost at every reading and the work a rebuild at that
K spends are both measured, through the same `final_sweep.run_one_staple` as every other run. The
bandit's final K per seed is read from the main results in results/raw/final_2709.

It is a control, not one of the compared policies. Its purpose was fixed before it ran, and its
result is reported whichever way it came out. The configuration is added from outside, by
extending the list of allowed names and wrapping `final_sweep.build`, so no other file changes.
Results go to results/raw/control_2809.

| Constant | Value |
|---|---|
| `NAME` | `'global_rebuild_at_bandit_k'` |
| `SWEEP` | `RESULTS_DIR / 'final_2709'` |
| `OUT` | `RESULTS_DIR / 'control_2809'` |

#### `bandit_k`

*function, lines 41 to 46*

```python
def bandit_k(dataset: str) -> dict[int, int]
```

Return the bandit's final partition count for each seed, from the main results.

#### `main`

*function, lines 49 to 75*

```python
def main() -> None
```

Run the control for the given seeds through final_sweep.

### scripts/k_curve.py

115 lines.

The query cost of fresh builds at several partition counts K, with the build spread at each.

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

| Constant | Value |
|---|---|
| `MULTIPLES` | `(1.0, 1.5, 2.0, 3.0, 4.5, 8.5)` |

#### `main`

*function, lines 46 to 111*

```python
def main() -> None
```

Run the no maintenance stream, then build and score its final vectors at every multiple of K0.

### scripts/rebuild_noise.py

146 lines.

How much of the spread in the query cost comes from the query sample, and how much from the build.

    python scripts/rebuild_noise.py --dataset sift1m --seed 42

At the end of a run every policy of a seed holds the same live vectors, since the stream picks its
deletes from the data and never from the index, and the policies that keep K fixed also hold the
same K. A fresh exact k-means of those vectors at that K, with the same seed, still varied by 5.5
to 17.7 percent on SIFT and 8.6 to 22.9 on GIST, and the only input that differs is the row order.
This separates the two possible sources of that spread.

Builds are fresh exact k-means of the end of run live vectors at the same K and seed, each given
the rows in a different order, scored on the 400 held out queries of the main runs and on 4,000
held out queries drawn the same way. Slices score one fixed index on disjoint slices of 400 of the
4,000, for the end of run index and for the first fresh build.

If the spread over builds shrinks on the 4,000 and the slices vary as much as the builds do on 400,
the query sample is the source. If the builds still vary on 4,000, the clusterings really differ
in quality. Results go to results/raw/rebuild_noise_<dataset>_s<seed>.csv.

#### `score`

*function, lines 48 to 51*

```python
def score(index: IVFIndex, q: np.ndarray) -> tuple[float, float]
```

Return the reported query cost on `q`, priced at the nprobe that meets recall 0.9, and that nprobe.

#### `spread`

*function, lines 54 to 57*

```python
def spread(x) -> float
```

Return the largest value over the smallest, minus one, in percent.

#### `cv`

*function, lines 60 to 63*

```python
def cv(x) -> float
```

Return the coefficient of variation, the sample standard deviation over the mean, in percent.

#### `main`

*function, lines 66 to 142*

```python
def main() -> None
```

Run the no maintenance stream, then score repeated builds and query slices of its end state.

### scripts/dedrift_split_trace.py

176 lines.

What one DeDrift Split pass does to the index, measured pass by pass on one main run.

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

| Constant | Value |
|---|---|
| `TRACE` | `[]` |

#### `_nearest_fraction`

*function, lines 52 to 60*

```python
def _nearest_fraction(vecs: np.ndarray, own: np.ndarray, cents: np.ndarray) -> float
```

Return the share of rows whose own centroid is at least as near as every centroid in `cents`.

#### `_traced_split`

*function, lines 63 to 138*

```python
def _traced_split(orig)
```

Return a replacement for DeDriftMaintainer._split that runs `orig` and records the pass.

#### `main`

*function, lines 141 to 172*

```python
def main() -> None
```

Run one DeDrift Split run with the pass recorder, write the trace, and check the run matches.

### scripts/measure_split_alpha.py

257 lines.

Measure alpha, the share of a parent's access each child of a split receives, Quake equation 6.

    python scripts/measure_split_alpha.py --dataset sift1m --seed 42

Quake estimates the cost change of splitting a partition as

    delta Split = delta O+ - A * lambda(s) + 2 * alpha * A * lambda(s / 2)

where alpha is the share of the parent's access each of the two children is expected to get.
Quake publishes alpha = 0.9. Here alpha is not left as an assumption, it is measured. A query
probes the nprobe nearest centroids, so a partition can be split and the same queries routed again,
counting how much of the parent's access the children actually receive.

    alpha = (A_left + A_right) / (2 * A_parent)

Routing depends only on centroids, so the reassignment after a split, which moves vectors without
moving centroids, cannot change this measurement. It is applied anyway so the split is the full
operation. The measurement uses an index built with k-means on a random fifth of the dataset,
before the starting index of the main runs existed, and it gave 0.862, 0.862 and 0.824 on SIFT and
0.880 on GIST, pooled as 0.86 in config.SPLIT_ACCESS_ALPHA.

#### `route_all`

*function, lines 50 to 71*

```python
def route_all(index: IVFIndex, queries: np.ndarray, nprobe: int) -> list[np.ndarray]
```

Return the probed partition ids of every query, computed for all queries at once.

Only the set of probed partitions matters here, so the order within each set may differ from
`IVFIndex._nearest_partitions`.

#### `verify_router`

*function, lines 74 to 79*

```python
def verify_router(index: IVFIndex, queries: np.ndarray, nprobe: int, n_check: int = 25) -> None
```

Check that `route_all` gives the same probe sets as the index itself.

#### `access_counts`

*function, lines 82 to 88*

```python
def access_counts(probe_sets: list[np.ndarray], n_queries: int) -> dict[int, float]
```

Return the share of queries that probe each partition, A in Quake equation 1.

#### `build_query_pool`

*function, lines 91 to 110*

```python
def build_query_pool(queries: np.ndarray, mode: str, seed: int, n_queries: int, hot_fraction: float, hot_prob: float) -> np.ndarray
```

Return the query pool, concentrated near one query or uniform as a robustness check.

#### `pick_candidates`

*function, lines 113 to 124*

```python
def pick_candidates(index: IVFIndex, n_candidates: int, min_size: int) -> list[int]
```

Return partitions spread across the size range, so alpha is not read off one size alone.

#### `measure_one`

*function, lines 127 to 177*

```python
def measure_one(index: IVFIndex, pid: int, queries: np.ndarray, nprobe: int, before_probes: list[np.ndarray], seed: int, boundary_top_k: int) -> dict | None
```

Split one partition on a copy of the index and measure the access its children get.

Returns None for a partition no query probed, or one whose split put everything on one side.

#### `main`

*function, lines 180 to 253*

```python
def main() -> None
```

Build the index, split a range of partitions one at a time, and report alpha.

### scripts/make_figures.py

365 lines.

The figures of the thesis, in English and Greek, written to scripts/figures.

    python scripts/make_figures.py

The first five are illustrations on synthetic data or plots of the stored Faiss price list in
results/cost_model/query_cost_calibration.yaml. The last three read the result files of the main
runs, the half size runs and the K curve, so those must exist under results/. The script also
prints the worked numbers the thesis quotes for Quake's split threshold, so the text can be
checked against the code.

| Constant | Value |
|---|---|
| `HERE` | `Path(__file__).resolve().parent` |
| `ROOT` | `HERE.parent` |
| `OUT` | `HERE / 'figures'` |
| `CAL` | `yaml.safe_load((ROOT / 'results/cost_model/query_cost_calibration.y...` |
| `TEXT` | `{'en': {'ivf_q': 'query', 'ivf_note': 'squares: centroids, dots: st...` |
| `RAW` | `ROOT / 'results' / 'raw'` |
| `ORDER` | `['global_rebuild', 'lire_lite', 'dedrift_lazy', 'dedrift_split', 'd...` |
| `NAMES` | `{'en': {'global_rebuild': 'Global rebuild', 'lire_lite': 'LIRE', 'd...` |
| `RTEXT` | `{'en': {'change': 'change in query cost against no maintenance, %',...` |

#### `fig_ivf`

*function, lines 68 to 98*

```python
def fig_ivf(lang)
```

Draw a two dimensional IVF example, partitions, centroids and the two a query opens.

#### `fig_pipeline`

*function, lines 101 to 133*

```python
def fig_pipeline(lang)
```

Draw the pipeline of one experiment, from the dataset to the scoring.

#### `fig_retention`

*function, lines 136 to 159*

```python
def fig_retention(lang)
```

Draw the retention window, each batch expiring 50 time steps after it arrives.

#### `fig_list_price`

*function, lines 162 to 182*

```python
def fig_list_price(lang)
```

Plot the Faiss price per vector against list length, for both datasets.

#### `fig_quake_threshold`

*function, lines 185 to 208*

```python
def fig_quake_threshold(lang)
```

Plot the smallest access fraction at which Quake splits a partition of each size.

#### `runs`

*function, lines 238 to 244*

```python
def runs(sweep, tag)
```

Read the result rows of one set of runs and add the change against no maintenance, in percent.

#### `summary`

*function, lines 247 to 250*

```python
def summary(sweep_dir, dataset)
```

Read the ranking table that scripts/analyse_final.py wrote for one dataset.

#### `_dotpanel`

*function, lines 253 to 271*

```python
def _dotpanel(ax, series, lang, show_labels)
```

Draw per seed changes as dots and the mean with its interval, one row per policy.

#### `fig_results_change`

*function, lines 274 to 291*

```python
def fig_results_change(lang)
```

Plot the change in query cost against no maintenance, SIFT at both sizes and GIST.

#### `fig_results_frontier`

*function, lines 294 to 315*

```python
def fig_results_frontier(lang)
```

Plot the change in query cost against maintenance work, per seed and on average.

#### `fig_k_curve`

*function, lines 318 to 343*

```python
def fig_k_curve(lang)
```

Plot the query cost of fresh builds against the partition count, as a multiple of the start.

### scripts/download_datasets.py

89 lines.

Download SIFT1M and GIST1M into data/.

    python scripts/download_datasets.py --dataset all

SIFT1M is tried from a Hugging Face mirror first and falls back to the original TEXMEX FTP
archive. GIST1M comes from the TEXMEX FTP archive. A dataset already present is skipped. SIFT1M
takes about 0.5 GB on disk and GIST1M about 5.5 GB.

| Constant | Value |
|---|---|
| `SIFT1M_FTP` | `'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'` |
| `SIFT1M_HF` | `'https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main'` |
| `GIST1M_FTP` | `'ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz'` |

#### `_download`

*function, lines 28 to 34*

```python
def _download(url: str, dest: Path) -> None
```

Download `url` to the file `dest`, creating its folder if needed.

#### `download_sift1m`

*function, lines 37 to 57*

```python
def download_sift1m(data_dir: Path) -> None
```

Download SIFT1M into `data_dir`/sift, from the mirror or else from the FTP archive.

#### `download_gist1m`

*function, lines 60 to 70*

```python
def download_gist1m(data_dir: Path) -> None
```

Download GIST1M into `data_dir`/gist from the FTP archive.

#### `main`

*function, lines 73 to 85*

```python
def main() -> None
```

Parse the arguments and download the requested datasets.

"""Constants shared by the code, each with its source.

The constants that govern the reported runs are MAINTENANCE_CHECK_INTERVAL,
REBUILD_FRACTION_THRESHOLD, SPLIT_ACCESS_ALPHA, BOUNDARY_REASSIGN_TOP_K, LIRE_MERGE_CANDIDATES,
DRIFT_FRACTION, the two KMEANS constants and DEFAULT_K. The starting index and its size
thresholds come from `src/staple.py`, not from here. There is no global random seed, every seed is
passed explicitly to the function that uses it.
"""

from pathlib import Path

# a default number of partitions and nprobe. the reported runs grow their starting index to about
# 176 partitions with src/staple.py and read query cost at the nprobe that meets the recall target,
# so neither value reaches a reported cost. DEFAULT_NPROBE is still the nprobe of the raw recall
# column and the first guess of the nprobe search in src/stream.py
DEFAULT_PARTITIONS: int = 1000
DEFAULT_NPROBE: int = 10
# neighbours returned per query, the k of recall at 10. the number of partitions is called K
DEFAULT_K: int = 10

# the partition size limits Ada-IVF publishes in section 5.1, "the maximum/minimum partition size
# to 2000/500". they are the defaults of the LIRE and bandit policies when built directly. the
# reported runs pass the thresholds the starting index was grown to instead, split above 400 and
# merge at or below 50, see src/staple.py
MIN_PARTITION_SIZE: int = 500
MAX_PARTITION_SIZE: int = 2000

# the bandit drift threshold, as a fraction of the data scale, the norm of the per dimension
# standard deviation of the vectors. 0.0129 is 5.0 divided by the SIFT scale of 387, so one value
# behaves the same on datasets of different scale
DRIFT_FRACTION: float = 0.0129
# not used by the reported runs
DRIFT_TOTAL_FRACTION: float = 7.8
# the fraction of the parent access count each child inherits after a split, alpha in Quake
# section 4.2.2. each child inherits it, so the two children together hold 2 alpha. measured on
# this index by scripts/measure_split_alpha.py as 0.862, 0.862 and 0.824 on SIFT and 0.880 on
# GIST, and the pooled value is used. Quake publishes 0.9, measured on its own benchmarks
SPLIT_ACCESS_ALPHA: float = 0.86
# whether the split balances the two halves, a relaxed form of SPANN equation 1. off, because the
# balanced version was measured to do worse here, and matching SPFresh exactly would first need
# its fixed balance factor ported. DeDrift is unaffected, it calls k-means directly as its paper
# specifies
BALANCED_SPLIT: bool = False
# how many neighbouring partitions the LIRE boundary repair reconsiders after a split or merge.
# this is the reindexing radius Ada-IVF publishes in section 5.1, "using a reindexing radius of
# 25". Quake uses 50 for its LIRE, the two papers differ, and this follows Ada-IVF
BOUNDARY_REASSIGN_TOP_K: int = 25
# how many neighbouring partitions the LIRE merge considers as a partner, the InternalResultNum
# of the SPFresh SIFT1M configuration
LIRE_MERGE_CANDIDATES: int = 64
# how often, in updates, every policy is offered the chance to act. Quake runs maintenance after
# each operation, so this is a deliberate coarsening that makes a full run affordable, applied to
# every policy equally
MAINTENANCE_CHECK_INTERVAL: int = 1000
# when the global rebuild runs, as a fraction of the vectors modified since the last one. this is
# the Rebuild baseline Ada-IVF publishes in section 5.1, "we trigger rebuilding after 2.5 percent
# of the total number of vectors in the workload have been modified"
REBUILD_FRACTION_THRESHOLD: float = 0.025

# not used by the reported runs
MEASUREMENT_INTERVAL: int = 5000

# exact Lloyd k-means for every full build. the minibatch approximation was measured to leave
# about a quarter of the GIST partitions below the merge threshold at birth, so a merging policy
# would have spent its first passes repairing the build. both values are the Faiss defaults,
# niter 25 and nredo 1
KMEANS_MAX_ITER: int = 25
KMEANS_N_INIT: int = 1

# paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "raw"
FIGURES_DIR: Path = PROJECT_ROOT / "results" / "figures"
COST_MODEL_DIR: Path = PROJECT_ROOT / "results" / "cost_model"
# where stored k-means results live, next to the datasets they are computed from
KMEANS_CACHE_DIR: Path = DATA_DIR / "kmeans_cache"

for _d in (RESULTS_DIR, FIGURES_DIR, COST_MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

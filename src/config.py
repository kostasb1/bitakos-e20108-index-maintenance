"""Constants shared by the code, each with its source."""

from pathlib import Path

# defaults, not used for reported costs
DEFAULT_PARTITIONS: int = 1000
DEFAULT_NPROBE: int = 10
# neighbours per query (k)
DEFAULT_K: int = 10

# Ada-IVF partition size limits, the reported runs use 400 and 50
MIN_PARTITION_SIZE: int = 500
MAX_PARTITION_SIZE: int = 2000

# bandit drift threshold, as a fraction of the data scale
DRIFT_FRACTION: float = 0.0129
# not used by the reported runs
DRIFT_TOTAL_FRACTION: float = 7.8
# split access share alpha, measured on this index (Quake uses 0.9)
SPLIT_ACCESS_ALPHA: float = 0.86
# balanced split, off
BALANCED_SPLIT: bool = False
# LIRE reassignment radius, Ada-IVF value
BOUNDARY_REASSIGN_TOP_K: int = 25
# merge partner candidates, SPFresh value
LIRE_MERGE_CANDIDATES: int = 64
# updates between maintenance passes
MAINTENANCE_CHECK_INTERVAL: int = 1000
# rebuild trigger, Ada-IVF section 5.1
REBUILD_FRACTION_THRESHOLD: float = 0.025

# not used by the reported runs
MEASUREMENT_INTERVAL: int = 5000

# exact k-means, Faiss defaults
KMEANS_MAX_ITER: int = 25
KMEANS_N_INIT: int = 1

# paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results" / "raw"
FIGURES_DIR: Path = PROJECT_ROOT / "results" / "figures"
COST_MODEL_DIR: Path = PROJECT_ROOT / "results" / "cost_model"
# k-means cache
KMEANS_CACHE_DIR: Path = DATA_DIR / "kmeans_cache"

for _d in (RESULTS_DIR, FIGURES_DIR, COST_MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

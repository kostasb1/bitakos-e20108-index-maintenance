"""Plain data records shared across the code.

`MaintenanceReport` is the important one. Every maintenance pass returns one, and
`Maintainer.maybe_maintain` adds its counters to the running totals that become the work columns
of a result row. `CostModel` is the linear cost model the policies fall back to when no profiled
scan cost curve is given, and `ExperimentConfig` is not used by the reported runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PartitionStats:
    """A summary of one partition, its live size, tombstones, drift and access count."""

    partition_id: int
    size: int
    tombstone_count: int
    centroid_drift: float
    access_count: float
    quality: float


@dataclass
class IndexStats:
    """A summary of the whole index, its sizes, drift and size imbalance."""

    num_partitions: int
    total_vectors: int
    live_vectors: int
    mean_partition_size: float
    max_partition_size: int
    min_partition_size: int
    mean_drift: float
    max_drift: float
    size_imbalance_ratio: float
    data_scale: float = 0.0


@dataclass
class MaintenanceReport:
    """What one maintenance pass did, as counts of actions and of the work they took.

    Work has two parts. `vectors_processed` counts every vector read, copied or averaged, and the
    distance fields count every distance evaluation between two vectors. Both are one pass over
    the dimensions of a vector, which is why they can be added. Deciding whether to act is not
    charged, only carrying out an action is.
    """

    triggered: bool = False
    num_splits: int = 0
    # splits that the cost estimate admitted and the exact recomputation on the real children then
    # declined, the verify step of Quake. zero for every policy that does not use predicted cost
    num_splits_declined: int = 0
    num_merges: int = 0
    # merges that the estimate admitted and the exact recomputation declined, Quake section 4.2.3
    num_merges_declined: int = 0
    # partitions removed by the forced delete of Quake, which fires when a partition has lost more
    # than a set fraction of its size since the previous pass, without consulting the cost model.
    # a Quake delete moves the vectors to their nearest partitions and is counted in num_merges,
    # so these are counted there too, and separately here because no cost estimate chose them
    num_forced_merges: int = 0
    # distance evaluations between a vector and a centroid made while carrying out actions. k-means
    # pays iterations times points times centroids, a reassignment pays points times candidate
    # centroids, and a neighbour lookup inside an action pays one per partition
    distance_evaluations: int = 0
    # the same, made inside k-means, counted apart because a library runs k-means as batched
    # matrix products, so one of its distances is cheaper than a scan distance. small is the
    # 2-means of a split or the k2-means of DeDrift, a few centroids over a few hundred points.
    # bulk is a full rebuild, hundreds of centroids over the whole index. the calibration prices
    # each kind against a scan distance, so work is comparable across methods
    kmeans_distance_evaluations: int = 0
    bulk_kmeans_distance_evaluations: int = 0
    # the part of vectors_processed spent on compaction, the stored size each compaction reads.
    # kept apart because the reported query cost reads live vectors only, the delete model of Quake
    # and Faiss, under which compaction gains nothing, so the reported work leaves it out. the
    # tombstone model, under which compaction does gain, is reported beside it
    compaction_reads: int = 0
    # vectors moved one at a time between existing partitions, the reassignment of SPFresh LIRE
    # and the reindexing radius of Ada-IVF. its cost grows with the boundary, not the index
    num_reassigned: int = 0
    # empty partitions removed. this is not a merge, since an empty partition has nothing to move,
    # and counting it as one would inflate the merge rate
    num_collected: int = 0
    # vectors moved by a wholesale repartition, a global rebuild or a DeDrift split. its cost grows
    # with the index, so it is counted apart from num_reassigned
    num_repartitioned: int = 0
    # vectors that passed the two necessary conditions of LIRE and were therefore distance checked,
    # SPFresh section 3.3. num_reassigned divided by this is how often the conditions led to a
    # move. zero for every policy except LIRE, and counted on the split path only, since the merge
    # path applies no condition
    num_examined: int = 0
    num_centroids_recomputed: int = 0
    wall_time_seconds: float = 0.0
    vectors_processed: int = 0  # the deterministic work count
    partitions_touched: list[int] = field(default_factory=list)


@dataclass
class QualityReport:
    """Partition quality over the whole index, its mean, spread and worst partition."""

    mean_quality: float
    std_quality: float
    worst_partition_id: int
    worst_quality: float
    per_partition: list[PartitionStats] = field(default_factory=list)


@dataclass
class CostModel:
    """A linear latency model, alpha times size plus beta times drift plus gamma times access."""

    alpha: float
    beta: float
    gamma: float
    intercept: float
    r_squared: float

    def predict_latency(self, size: int, drift: float = 0.0, access: int = 0) -> float:
        """Return the predicted latency, never below zero."""
        raw = self.alpha * size + self.beta * drift + self.gamma * access + self.intercept
        return max(0.0, raw)

    def scan_latency(self, size: int) -> float:
        """Return lambda(s) of Quake section 4.1, the cost of scanning a partition of `size` vectors.

        This linear model has no measured curve, so it uses its linear form.
        """
        return self.predict_latency(size=size)

    def centroid_latency(self, n_centroids: int) -> float:
        """Return the cost of comparing a query with `n_centroids` centroids.

        Priced as a scan of that many vectors.
        """
        return self.predict_latency(size=n_centroids)

    def to_dict(self) -> dict[str, float]:
        """Return the coefficients as a dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> CostModel:
        """Build a model from a dictionary of coefficients."""
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Parameters of one experiment, kept for compatibility and not used by the reported runs."""

    dataset_name: str
    n_vectors: int
    n_partitions: int
    maintainer_name: str
    workload_type: str
    n_operations: int
    insert_delete_ratio: float
    drift_rate: float
    nprobe: int
    k: int
    seed: int
    measurement_interval: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_yaml(self, path: Path) -> None:
        """Write the configuration to a YAML file."""
        path.write_text(yaml.safe_dump(self.__dict__, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Read a configuration from a YAML file."""
        data = yaml.safe_load(path.read_text())
        return cls(**data)

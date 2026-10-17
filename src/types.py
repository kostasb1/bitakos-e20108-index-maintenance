"""plain data records shared across the code"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PartitionStats:
    """a summary of one partition, its live size, tombstones, drift and access count"""

    partition_id: int
    size: int
    tombstone_count: int
    centroid_drift: float
    access_count: float
    quality: float


@dataclass
class IndexStats:
    """a summary of the whole index, its sizes, drift and size imbalance"""

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
    """what one maintenance pass did, as counts of actions and of the work they took"""

    triggered: bool = False
    num_splits: int = 0
    # splits declined by the exact check
    num_splits_declined: int = 0
    num_merges: int = 0
    # merges declined by the exact check
    num_merges_declined: int = 0
    # forced deletes of Quake
    num_forced_merges: int = 0
    # distance evaluations while acting
    distance_evaluations: int = 0
    # k-means distances, small and bulk
    kmeans_distance_evaluations: int = 0
    bulk_kmeans_distance_evaluations: int = 0
    # compaction reads
    compaction_reads: int = 0
    # vectors reassigned one at a time
    num_reassigned: int = 0
    # empty partitions removed
    num_collected: int = 0
    # vectors moved by a rebuild or DeDrift split
    num_repartitioned: int = 0
    # vectors checked by the LIRE conditions
    num_examined: int = 0
    num_centroids_recomputed: int = 0
    wall_time_seconds: float = 0.0
    vectors_processed: int = 0  # the deterministic work count
    partitions_touched: list[int] = field(default_factory=list)


@dataclass
class QualityReport:
    """partition quality over the whole index, its mean, spread and worst partition"""

    mean_quality: float
    std_quality: float
    worst_partition_id: int
    worst_quality: float
    per_partition: list[PartitionStats] = field(default_factory=list)


@dataclass
class CostModel:
    """linear latency model"""

    alpha: float
    beta: float
    gamma: float
    intercept: float
    r_squared: float

    def predict_latency(self, size: int, drift: float = 0.0, access: int = 0) -> float:
        """return the predicted latency, never below zero"""
        raw = self.alpha * size + self.beta * drift + self.gamma * access + self.intercept
        return max(0.0, raw)

    def scan_latency(self, size: int) -> float:
        """lambda(s), the cost of scanning a partition of `size` vectors"""
        return self.predict_latency(size=size)

    def centroid_latency(self, n_centroids: int) -> float:
        """return the cost of comparing a query with `n_centroids` centroids"""
        return self.predict_latency(size=n_centroids)

    def to_dict(self) -> dict[str, float]:
        """return the coefficients as a dictionary"""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> CostModel:
        """build a model from a dictionary of coefficients"""
        return cls(**data)


@dataclass
class ExperimentConfig:
    """old experiment settings, not used"""

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
        """write the configuration to a YAML file"""
        path.write_text(yaml.safe_dump(self.__dict__, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """read a configuration from a YAML file"""
        data = yaml.safe_load(path.read_text())
        return cls(**data)

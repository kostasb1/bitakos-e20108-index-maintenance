"""One IVF partition, its vectors, its centroid and its access count."""

from __future__ import annotations

import numpy as np

from src.types import PartitionStats
from src.vector import compute_centroid, l2_distance


class Partition:
    """One partition, its vectors, tombstones, centroid and access count."""

    def __init__(self, partition_id: int, centroid: np.ndarray) -> None:
        """Create an empty partition with the given id and centroid."""
        self.partition_id = partition_id
        self.centroid: np.ndarray = centroid.copy()
        self.vector_ids: list[int] = []
        # vectors in one contiguous block, _n rows used
        self._buf: np.ndarray = np.empty((0, centroid.shape[0]), dtype=centroid.dtype)
        self._n: int = 0
        self._id_to_pos: dict[int, int] = {}  # id to row, tombstones included
        self.tombstones: set[int] = set()
        # live rows and ids, rebuilt after a change
        self._live_rows: np.ndarray | None = None
        self._live_ids: list[int] | None = None
        # access count, a float because the bandit decays it
        self.access_count: float = 0.0
        self._cached_true_mean: np.ndarray | None = None
        self._dirty: bool = True
        self._radius: float = 0.0

    def size(self) -> int:
        """Return the number of live vectors, stored vectors minus tombstones."""
        return len(self.vector_ids) - len(self.tombstones)

    def total_size(self) -> int:
        """Return the number of stored vectors, tombstoned ones included."""
        return len(self.vector_ids)

    def is_empty(self) -> bool:
        """Return True if the partition holds no live vector."""
        return self.size() == 0

    def _ensure_capacity(self, required: int) -> None:
        """Grow the storage block so it holds at least `required` rows."""
        if required <= self._buf.shape[0]:
            return
        new_cap = max(8, required, self._buf.shape[0] * 2)
        grown = np.empty((new_cap, self._buf.shape[1]), dtype=self._buf.dtype)
        grown[:self._n] = self._buf[:self._n]
        self._buf = grown

    def _invalidate_live(self) -> None:
        """Forget the cached live rows and ids, after any change to the partition."""
        self._live_rows = None
        self._live_ids = None

    def add(self, vec_id: int, vec: np.ndarray) -> None:
        """Store a vector in the partition."""
        self._invalidate_live()
        if vec_id in self._id_to_pos:
            self._buf[self._id_to_pos[vec_id]] = vec
            self.tombstones.discard(vec_id)
        else:
            self._ensure_capacity(self._n + 1)
            self._buf[self._n] = vec
            self.vector_ids.append(vec_id)
            self._id_to_pos[vec_id] = self._n
            self._n += 1
        self._radius = max(self._radius, float(np.linalg.norm(vec - self.centroid)))
        self._dirty = True

    def mark_deleted(self, vec_id: int) -> None:
        """Delete a vector by marking it with a tombstone."""
        if vec_id not in self._id_to_pos:
            raise KeyError(f"vec_id {vec_id} not in partition {self.partition_id}")
        self.tombstones.add(vec_id)
        self._invalidate_live()
        self._dirty = True

    def is_deleted(self, vec_id: int) -> bool:
        """Return True if the vector is marked with a tombstone."""
        return vec_id in self.tombstones

    def compact(self) -> int:
        """Remove tombstoned vectors, return how many."""
        if not self.tombstones:
            return 0
        removed = len(self.tombstones)

        # copy live rows into a new block
        rows, new_ids = self._live_block()
        new_ids = list(new_ids)
        self._buf = np.ascontiguousarray(rows, dtype=self._buf.dtype)
        self._n = len(new_ids)
        self.vector_ids = new_ids
        self.tombstones.clear()
        self._id_to_pos = {vid: i for i, vid in enumerate(self.vector_ids)}
        self._invalidate_live()
        self._dirty = True
        self._recompute_radius()
        return removed

    def true_mean(self) -> np.ndarray:
        """Return the mean of the live vectors, or the centroid if there are none."""
        if self._cached_true_mean is not None and not self._dirty:
            return self._cached_true_mean

        if self.size() == 0:
            self._cached_true_mean = self.centroid.copy()
        else:
            live_vecs = self.get_live_vectors()[1]
            self._cached_true_mean = compute_centroid(live_vecs)

        self._dirty = False
        return self._cached_true_mean

    def centroid_drift(self) -> float:
        """Return the distance between the centroid and the mean of the live vectors."""
        if self.size() == 0:
            return 0.0
        return l2_distance(self.centroid, self.true_mean())

    def update_centroid(self, new_centroid: np.ndarray) -> None:
        """Replace the centroid and recompute the radius around it."""
        self.centroid = new_centroid.copy()
        if self.size() == 0:
            self._dirty = True
        self._recompute_radius()

    def _recompute_radius(self) -> None:
        """Set the radius to the largest distance from the centroid to a live vector."""
        if self.size() == 0:
            self._radius = 0.0
            return
        live_vecs = self.get_live_vectors()[1]
        self._radius = float(np.linalg.norm(live_vecs - self.centroid, axis=1).max())

    def radius(self) -> float:
        """Return an upper bound on the distance from the centroid to any live vector."""
        return self._radius

    def record_access(self) -> None:
        """Count one routed query that probed this partition."""
        self.access_count += 1

    def decay_access(self, factor: float) -> None:
        """Multiply the access count by `factor`, so older queries weigh less."""
        self.access_count *= factor

    def get_vector(self, vec_id: int) -> np.ndarray:
        """Return the stored vector with the given id."""
        return self._buf[self._id_to_pos[vec_id]]

    def _live_block(self) -> tuple[np.ndarray, list[int]]:
        """Return the live vectors and their ids, for the scan only."""
        if not self.tombstones:
            return self._buf[:self._n], self.vector_ids
        if self._live_rows is None:
            tomb = self.tombstones
            self._live_ids = [vid for vid in self.vector_ids if vid not in tomb]
            self._live_rows = np.fromiter(
                (self._id_to_pos[vid] for vid in self._live_ids),
                dtype=np.intp, count=len(self._live_ids),
            )
        return self._buf[self._live_rows], self._live_ids

    def get_live_vectors(self) -> tuple[list[int], np.ndarray]:
        """Ids and a copy of the live vectors."""
        if self.size() == 0:
            d = self.centroid.shape[0]
            return [], np.empty((0, d), dtype=self.centroid.dtype)
        rows, live_ids = self._live_block()
        # astype copies
        return list(live_ids), rows.astype(self.centroid.dtype, copy=True)

    def stats(self, quality: float = 0.0) -> PartitionStats:
        """Return a summary of the partition, its sizes, drift and access count."""
        return PartitionStats(
            partition_id=self.partition_id,
            size=self.size(),
            tombstone_count=len(self.tombstones),
            centroid_drift=self.centroid_drift(),
            access_count=self.access_count,
            quality=quality,
        )

    def __repr__(self) -> str:
        """Return a short description with the id, size, drift and access count."""
        return (
            f"Partition(id={self.partition_id}, size={self.size()}, "
            f"drift={self.centroid_drift():.3f}, access={self.access_count})"
        )

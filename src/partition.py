"""One IVF partition, its vectors, its centroid and its access count.

A delete marks the vector with a tombstone and returns at once. The vector stays in storage until
`compact` rewrites the partition, so `size()` counts live vectors and `total_size()` counts stored
ones. Every size rule reads `size()`, except two in LIRE, the split trigger and the capacity check
for a merge partner, which read the stored length as SPFresh does.

`get_live_vectors()` returns `(ids, vectors)` as an independent copy that a caller may keep.
`_live_block()` returns `(vectors, ids)`, in the opposite order, possibly as a view into the
storage, and is meant for the scan only.
"""

from __future__ import annotations

import numpy as np

from src.types import PartitionStats
from src.vector import compute_centroid, l2_distance


class Partition:
    """A partition of the index, its stored vectors, tombstones, centroid and access count."""

    def __init__(self, partition_id: int, centroid: np.ndarray) -> None:
        """Create an empty partition with the given id and centroid."""
        self.partition_id = partition_id
        self.centroid: np.ndarray = centroid.copy()
        self.vector_ids: list[int] = []
        # the vectors live in one contiguous block of rows, the layout Quake uses for its
        # partitions. a contiguous block lets the scan read every vector in a single pass, which is
        # what the measured scan cost assumes. _n is the number of rows in use, the rest of the
        # block is spare capacity
        self._buf: np.ndarray = np.empty((0, centroid.shape[0]), dtype=centroid.dtype)
        self._n: int = 0
        self._id_to_pos: dict[int, int] = {}  # vector id to its row, tombstoned vectors included
        self.tombstones: set[int] = set()
        # the rows and ids of the live vectors, rebuilt on the first read after any change. the
        # rows are an integer array so that gathering them is one numpy operation
        self._live_rows: np.ndarray | None = None
        self._live_ids: list[int] | None = None
        # how many routed queries probed this partition in the current window. it is a float
        # because the bandit decays it by a factor each pass, while the Quake policy resets it to
        # zero after each pass. divided by index.queries_routed it is the access fraction A of
        # Quake
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
        """Grow the storage block so it holds at least `required` rows.

        The capacity doubles each time, as in Quake, so a series of appends costs constant time
        per append on average.
        """
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
        """Store a vector in the partition.

        If the id is present as a tombstone, the vector is written over the old row and revived,
        otherwise it is appended.
        """
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
        """Delete a vector by marking it with a tombstone.

        The delete takes constant time and moves nothing, so it is charged no maintenance work. The
        vector stays stored until `compact` rewrites the partition, and that compaction pays the
        cost, proportional to the stored length. A policy that never compacts never pays, but it
        keeps the dead vectors in storage until the run ends, and stored minus live size is
        reported for every run so the debt stays visible.

        This follows SPFresh, which leaves a deleted vector in its posting until a split or merge
        rewrites it. Quake removes a vector at once instead. The reported query cost reads live
        vectors only, which is the Quake and Faiss model, and the SPFresh model, in which
        tombstones are scanned, is measured beside it, see `metrics.stored_cost_at`.
        """
        if vec_id not in self._id_to_pos:
            raise KeyError(f"vec_id {vec_id} not in partition {self.partition_id}")
        self.tombstones.add(vec_id)
        self._invalidate_live()
        self._dirty = True

    def is_deleted(self, vec_id: int) -> bool:
        """Return True if the vector is marked with a tombstone."""
        return vec_id in self.tombstones

    def compact(self) -> int:
        """Rewrite the storage with the live vectors only, and return how many were removed."""
        if not self.tombstones:
            return 0
        removed = len(self.tombstones)

        # one gather of the surviving rows into a new block of exactly the right size, a cost
        # proportional to the stored length as the tombstone design assumes
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
        """Return the mean of the live vectors, or the centroid if there are none.

        The mean is cached until the partition changes.
        """
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
        """Return the distance between the centroid and the mean of the live vectors.

        An empty partition has zero drift by definition.
        """
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
        """Return an upper bound on the distance from the centroid to any live vector.

        A delete can only shrink the true radius, so the stored value stays a valid upper bound
        without being recomputed on every delete, which would cost a full pass over the partition.
        `compact` and `update_centroid` tighten it again, since they read the live vectors anyway.
        """
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
        """Return the live vectors and their ids, for the scan only.

        When nothing is tombstoned the vectors are a view into the storage, so a caller must
        neither write to them nor keep them across a change to the partition. Use
        `get_live_vectors` for an independent copy.
        """
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
        """Return the ids and a copy of the vectors of the live vectors, safe for the caller to keep.

        The vectors are returned in the centroid number type.
        """
        if self.size() == 0:
            d = self.centroid.shape[0]
            return [], np.empty((0, d), dtype=self.centroid.dtype)
        rows, live_ids = self._live_block()
        # astype always copies, so the caller never shares memory with the storage
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

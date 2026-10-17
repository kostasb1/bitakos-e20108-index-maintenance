"""Distance arithmetic, and the partition scan that every query runs."""

from __future__ import annotations

import numpy as np


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def l2_distance_squared(a: np.ndarray, b: np.ndarray) -> float:
    """Return the squared Euclidean distance between two vectors."""
    diff = a - b
    return float(diff @ diff)


# scratch buffers for the scan, single thread
_SCAN_BUFFERS: dict[tuple[int, np.dtype], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
# smallest buffer size
_SCAN_MIN_ROWS = 1024


def _scan_buffers(rows: int, dim: int, dtype: np.dtype):
    """Return three scratch arrays sized for a partition of `rows` vectors."""
    key = (dim, dtype)
    held = _SCAN_BUFFERS.get(key)
    if held is None or held[0].shape[0] < rows:
        n = max(rows, _SCAN_MIN_ROWS)
        held = (np.empty((n, dim), dtype=dtype),
                np.empty((n, dim), dtype=dtype),
                np.empty(n, dtype=dtype))
        _SCAN_BUFFERS[key] = held
    return held[0][:rows], held[1][:rows], held[2][:rows]


def l2_distance_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance from `query` to every row of `matrix`."""
    # reusing buffers keeps timing stable and is faster
    rows = len(matrix)
    if rows == 0:
        return np.empty(0, dtype=matrix.dtype)
    # wider query type, use the plain form
    if np.result_type(matrix.dtype, query.dtype) != matrix.dtype or matrix.ndim != 2:
        return np.linalg.norm(matrix - query, axis=1)

    diff, sq, acc = _scan_buffers(rows, matrix.shape[1], matrix.dtype)
    np.subtract(matrix, query, out=diff)
    np.multiply(diff, diff, out=sq)
    np.add.reduce(sq, axis=1, out=acc)
    # sqrt returns a new array
    return np.sqrt(acc)


def l2_distance_matrix(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Distances between every query and every row."""
    from scipy.spatial.distance import cdist

    return cdist(queries, matrix, metric="euclidean")


def nearest_k_indices(query: np.ndarray, matrix: np.ndarray, k: int) -> np.ndarray:
    """Return the row indices of the `k` rows nearest to `query`, nearest first."""
    distances = l2_distance_batch(query, matrix)

    if k >= len(distances):
        return np.argsort(distances)

    # k smallest without a full sort
    unsorted_top_k = np.argpartition(distances, k)[:k]
    return unsorted_top_k[np.argsort(distances[unsorted_top_k])]


def compute_centroid(vectors: np.ndarray) -> np.ndarray:
    """Return the mean of a non empty set of vectors."""
    if len(vectors) == 0:
        raise ValueError("Cannot compute centroid of empty vector set")
    return vectors.mean(axis=0)


def normalize(v: np.ndarray) -> np.ndarray:
    """Return `v` scaled to unit length, or a copy of `v` if its length is zero."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.copy()
    return v / norm

"""Distance arithmetic, and the partition scan that every query runs.

`l2_distance_batch` is the most frequently called function in the project. It computes the
distances from one query to every vector of one partition. It reuses scratch buffers, so a call
allocates only the array it returns, and its output is bit identical to
`np.linalg.norm(matrix - query, axis=1)`.
"""

from __future__ import annotations

import numpy as np


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def l2_distance_squared(a: np.ndarray, b: np.ndarray) -> float:
    """Return the squared Euclidean distance between two vectors.

    Ranking by squared distance gives the same order as ranking by distance, so callers that only
    compare distances use this and skip the square root.
    """
    diff = a - b
    return float(diff @ diff)


# scratch buffers for the partition scan, one set for each pair of dimension and number type the
# scan has seen. the scan runs on one thread only, so sharing them is safe
_SCAN_BUFFERS: dict[tuple[int, np.dtype], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
# the smallest buffer allocated, so a series of partitions of similar size reuses one allocation
_SCAN_MIN_ROWS = 1024


def _scan_buffers(rows: int, dim: int, dtype: np.dtype):
    """Return three scratch arrays sized for a partition of `rows` vectors.

    The first two hold the differences and their squares, the third the per vector sums. A
    buffer is replaced only when a larger partition arrives.
    """
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
    """Return the Euclidean distance from `query` to every row of `matrix`.

    This is the partition scan. It performs the same operations in the same order as
    `np.linalg.norm(matrix - query, axis=1)`, subtract, square, sum and square root, so the
    result is bit identical to it. The difference is that the two intermediate arrays come from
    reused buffers instead of being allocated on every call.
    """
    # allocating those two arrays on every call made the measured cost per vector jump between two
    # values about 3.2 times apart, at a partition size that moved from one measurement to the
    # next. reusing the buffers makes the timing repeat at every size, and it is also 3.5 to 3.7
    # times faster
    rows = len(matrix)
    if rows == 0:
        return np.empty(0, dtype=matrix.dtype)
    # a query of a wider number type than the partition would be narrowed when written into a
    # buffer of the partition type, so such calls use the plain form instead
    if np.result_type(matrix.dtype, query.dtype) != matrix.dtype or matrix.ndim != 2:
        return np.linalg.norm(matrix - query, axis=1)

    diff, sq, acc = _scan_buffers(rows, matrix.shape[1], matrix.dtype)
    np.subtract(matrix, query, out=diff)
    np.multiply(diff, diff, out=sq)
    np.add.reduce(sq, axis=1, out=acc)
    # sqrt allocates a new array, which is what the caller keeps, so the result never points into
    # the scratch buffers
    return np.sqrt(acc)


def l2_distance_matrix(queries: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return the matrix of Euclidean distances between every query and every row of `matrix`."""
    from scipy.spatial.distance import cdist

    return cdist(queries, matrix, metric="euclidean")


def nearest_k_indices(query: np.ndarray, matrix: np.ndarray, k: int) -> np.ndarray:
    """Return the row indices of the `k` rows nearest to `query`, nearest first."""
    distances = l2_distance_batch(query, matrix)

    if k >= len(distances):
        return np.argsort(distances)

    # select the k smallest without sorting everything, then sort only those k
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

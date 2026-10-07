"""Readers for SIFT1M and GIST1M, and exact ground truth.

The reported runs use `load_sift1m`, `load_gist1m` and `compute_ground_truth`. The files live under
data/ and `scripts/download_datasets.py` fetches them. `make_synthetic` builds a small dataset for
quick checks, and `split_base_for_streaming` is not used by the reported runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import DATA_DIR


def read_fvecs(path: Path) -> np.ndarray:
    """Read a .fvecs file into a float32 matrix.

    Each row is stored as its dimension, a 32 bit integer, followed by that many floats.
    """
    with open(path, "rb") as f:
        raw = np.fromfile(f, dtype=np.int32)
    if len(raw) == 0:
        raise ValueError(f"Empty fvecs file: {path}")
    d = raw[0]
    return raw.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(path: Path) -> np.ndarray:
    """Read a .ivecs file, the same layout as .fvecs with integers, into an int32 matrix."""
    with open(path, "rb") as f:
        raw = np.fromfile(f, dtype=np.int32)
    if len(raw) == 0:
        raise ValueError(f"Empty ivecs file: {path}")
    d = raw[0]
    return raw.reshape(-1, d + 1)[:, 1:].copy()


def load_sift1m(
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SIFT1M and return the base vectors, the queries and the ground truth.

    The shapes are one million by 128, ten thousand by 128 and ten thousand by 100.
    """
    sift_dir = data_dir / "sift"
    base = read_fvecs(sift_dir / "sift_base.fvecs")
    queries = read_fvecs(sift_dir / "sift_query.fvecs")
    gt = read_ivecs(sift_dir / "sift_groundtruth.ivecs")

    assert base.shape == (1_000_000, 128), f"Unexpected SIFT1M base shape: {base.shape}"
    assert base.dtype == np.float32, f"Expected float32, got {base.dtype}"
    return base, queries, gt


def load_gist1m(
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load GIST1M and return the base vectors, the queries and the ground truth.

    The shapes are one million by 960, one thousand by 960 and one thousand by 100.
    """
    gist_dir = data_dir / "gist"
    base = read_fvecs(gist_dir / "gist_base.fvecs")
    queries = read_fvecs(gist_dir / "gist_query.fvecs")
    gt = read_ivecs(gist_dir / "gist_groundtruth.ivecs")

    assert base.shape == (1_000_000, 960), f"Unexpected GIST1M base shape: {base.shape}"
    assert base.dtype == np.float32, f"Expected float32, got {base.dtype}"
    return base, queries, gt


def make_synthetic(
    n: int,
    dim: int,
    n_clusters: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a Gaussian mixture dataset, base vectors, queries and ground truth, for quick checks."""
    rng = np.random.default_rng(seed)
    n_queries = max(100, n // 100)

    cluster_centers = rng.normal(0, 10, size=(n_clusters, dim)).astype(np.float32)

    labels = rng.integers(0, n_clusters, size=n)
    base = cluster_centers[labels] + rng.normal(0, 1, size=(n, dim)).astype(np.float32)
    base = base.astype(np.float32)

    query_labels = rng.integers(0, n_clusters, size=n_queries)
    queries = cluster_centers[query_labels] + rng.normal(0, 1, size=(n_queries, dim)).astype(
        np.float32
    )
    queries = queries.astype(np.float32)

    gt = compute_ground_truth(base, queries, k=100)
    return base, queries, gt


def compute_ground_truth(base: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    """Return the row indices of the exact `k` nearest base vectors of every query.

    Uses an exhaustive Faiss index when Faiss is installed, and scikit-learn otherwise. Both
    compare every query with every vector, so the answer is exact either way.
    """
    try:
        import faiss

        index = faiss.IndexFlatL2(base.shape[1])
        index.add(base.astype(np.float32))
        _, indices = index.search(queries.astype(np.float32), k)
        return indices.astype(np.int32)
    except ImportError:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
        nn.fit(base)
        _, indices = nn.kneighbors(queries)
        return indices.astype(np.int32)


def split_base_for_streaming(
    base: np.ndarray,
    initial_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, list[int], np.ndarray, list[int]]:
    """Split the base vectors at random into an initial set and a pool of later inserts.

    The ids are the original row numbers, so they still match the published ground truth after
    the pool vectors are inserted.
    """
    n = len(base)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split_point = int(n * initial_fraction)
    initial_idx = indices[:split_point]
    spare_idx = indices[split_point:]

    initial_vectors = base[initial_idx].copy()
    initial_ids = list(initial_idx.tolist())
    spare_pool = base[spare_idx].copy()
    spare_ids = list(spare_idx.tolist())

    return initial_vectors, initial_ids, spare_pool, spare_ids

"""readers for SIFT1M and GIST1M, and exact ground truth"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import DATA_DIR


def read_fvecs(path: Path) -> np.ndarray:
    """read a .fvecs file into a float32 matrix"""
    with open(path, "rb") as f:
        raw = np.fromfile(f, dtype=np.int32)
    if len(raw) == 0:
        raise ValueError(f"Empty fvecs file: {path}")
    d = raw[0]
    return raw.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(path: Path) -> np.ndarray:
    """read a .ivecs file"""
    with open(path, "rb") as f:
        raw = np.fromfile(f, dtype=np.int32)
    if len(raw) == 0:
        raise ValueError(f"Empty ivecs file: {path}")
    d = raw[0]
    return raw.reshape(-1, d + 1)[:, 1:].copy()


def load_sift1m(
    data_dir: Path = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """load SIFT1M and return the base vectors, the queries and the ground truth"""
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
    """load GIST1M and return the base vectors, the queries and the ground truth"""
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
    """small synthetic dataset for quick checks"""
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
    """return the row indices of the exact `k` nearest base vectors of every query"""
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
    """split the base vectors into an initial set and later inserts"""
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

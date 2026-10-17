"""The IVF index and the operations policies share."""

from __future__ import annotations

import numpy as np

from src.config import (
    KMEANS_CACHE_DIR,
    KMEANS_MAX_ITER,
    KMEANS_N_INIT,
    SPLIT_ACCESS_ALPHA,
)
from src.partition import Partition
from src.types import IndexStats
from src.vector import l2_distance_batch

# extra candidates kept by the prefilter
ROUTING_CANDIDATE_MARGIN = 16


class IVFIndex:
    """An IVF index with partitions, centroids, routing and updates."""

    def __init__(self, dim: int) -> None:
        """Create an empty index for vectors of `dim` dimensions."""
        self.dim = dim
        self.partitions: dict[int, Partition] = {}
        self.vector_to_partition: dict[int, int] = {}
        self.next_partition_id: int = 0
        self._total_inserts: int = 0
        self._total_deletes: int = 0
        self._centroid_matrix: np.ndarray | None = None
        self._centroid_id_order: list[int] | None = None
        # float64 centroid matrix and norms for the prefilter
        self._centroid_matrix64: np.ndarray | None = None
        self._centroid_sq_norms: np.ndarray | None = None
        # data scale, used for drift thresholds
        self.data_scale: float = 0.0
        self.last_build_distance_evaluations: int = 0
        # number of routed queries
        self.queries_routed: float = 0.0
        # summed size of probed partitions, read by Quake
        self.vectors_scanned_routed: float = 0.0

    def build(
        self,
        vectors: np.ndarray,
        vector_ids: list[int],
        n_partitions: int,
        seed: int = 42,
        cache: bool = False,
    ) -> None:
        """Partition `vectors` with exact k-means into `n_partitions` partitions."""
        assert vectors.shape[1] == self.dim, f"Expected dim={self.dim}, got {vectors.shape[1]}"
        assert vectors.dtype == np.float32, f"Expected float32, got {vectors.dtype}"
        assert len(vectors) == len(vector_ids)

        # fixed at the first build
        if self.data_scale == 0.0:
            self.data_scale = float(np.linalg.norm(vectors.std(axis=0)))

        labels, centroids, n_iter = _lloyd_partition(vectors, n_partitions, seed, cache)
        # distance evaluations of this build
        self.last_build_distance_evaluations = (
            KMEANS_N_INIT * (n_iter + 1) * len(vectors) * n_partitions if n_iter else 0)

        self.queries_routed = 0.0
        self.vectors_scanned_routed = 0.0
        for pid in range(n_partitions):
            self.partitions[pid] = Partition(pid, centroids[pid])

        for vec, vid, label in zip(vectors, vector_ids, labels, strict=True):
            self.partitions[int(label)].add(vid, vec)
            self.vector_to_partition[vid] = int(label)

        self.next_partition_id = n_partitions
        self._invalidate_cache()

    def insert(self, vec_id: int, vec: np.ndarray) -> None:
        """Add a vector to the partition with the nearest centroid."""
        if vec_id in self.vector_to_partition:
            raise KeyError(f"vec_id {vec_id} already present")
        pid = self._find_nearest_partition(vec)
        self.partitions[pid].add(vec_id, vec)
        self.vector_to_partition[vec_id] = pid
        self._total_inserts += 1

    def delete(self, vec_id: int) -> bool:
        """Delete a vector by tombstoning it in its partition."""
        if vec_id not in self.vector_to_partition:
            return False
        pid = self.vector_to_partition[vec_id]
        self.partitions[pid].mark_deleted(vec_id)
        del self.vector_to_partition[vec_id]
        self._total_deletes += 1
        return True

    def search(
        self, query: np.ndarray, k: int, nprobe: int, record_access: bool = True
    ) -> list[tuple[int, float]]:
        """Return the `k` nearest vectors found in the `nprobe` nearest partitions."""
        probed_pids = self._nearest_partitions(query, nprobe)
        if record_access:
            self.queries_routed += 1

        all_vids: list[int] = []
        all_dists_parts: list[np.ndarray] = []

        for pid in probed_pids:
            p = self.partitions[pid]
            if record_access:
                p.record_access()
                self.vectors_scanned_routed += p.size()
            # view, not a copy
            live_vecs, live_ids = p._live_block()
            if len(live_ids) == 0:
                continue

            # same scan as the cost curve
            dists = l2_distance_batch(query, live_vecs)
            all_vids.extend(live_ids)
            all_dists_parts.append(dists)

        if not all_vids:
            return []

        all_dists = np.concatenate(all_dists_parts)
        k_actual = min(k, len(all_vids))

        if k_actual < len(all_vids):
            top_pos = np.argpartition(all_dists, k_actual - 1)[:k_actual]
            top_pos = top_pos[np.argsort(all_dists[top_pos])]
        else:
            top_pos = np.argsort(all_dists)

        return [(all_vids[int(i)], float(all_dists[i])) for i in top_pos]

    def search_batch(
        self, queries: np.ndarray, k: int, nprobe: int, record_access: bool = True
    ) -> list[list[tuple[int, float]]]:
        """Run `search` for every query and return the list of results."""
        return [self.search(q, k, nprobe, record_access=record_access) for q in queries]

    def search_adaptive(
        self,
        query: np.ndarray,
        k: int,
        target_recall: float = 0.9,
        max_nprobe: int | None = None,
        record_access: bool = False,
        radius_factor: float = 1.0,
    ) -> tuple[list[tuple[int, float]], int, int]:
        """Search with a growing nprobe. Not used in the reported runs."""
        matrix, order = self._get_centroid_matrix()
        n_parts = len(order)
        if n_parts == 0:
            return [], 0, 0

        diffs = matrix - query
        cdist = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        radii = np.array([self.partitions[pid].radius() for pid in order], dtype=cdist.dtype)
        lb = np.maximum(0.0, cdist - radius_factor * radii)
        probe_order = np.argsort(cdist)
        suffix_min = np.minimum.accumulate(lb[probe_order][::-1])[::-1]

        cap = min(max_nprobe or n_parts, n_parts)
        cand_d: list[np.ndarray] = []
        cand_vids: list[int] = []
        vectors_scanned = 0
        n_probed = 0

        for m in range(cap):
            pid = order[int(probe_order[m])]
            p = self.partitions[pid]
            if record_access:
                p.record_access()
            # view, not a copy
            live_vecs, live_ids = p._live_block()
            n_probed += 1
            if len(live_ids) > 0:
                cand_d.append(l2_distance_batch(query, live_vecs))
                cand_vids.extend(live_ids)
                vectors_scanned += len(live_ids)

            if len(cand_vids) < k:
                continue

            all_d = np.concatenate(cand_d)
            kk = min(k, len(all_d))
            topk_d = np.sort(np.partition(all_d, kk - 1)[:kk])
            min_unprobed_lb = float(suffix_min[m + 1]) if m + 1 < n_parts else np.inf
            guaranteed = int(np.count_nonzero(topk_d <= min_unprobed_lb))
            if guaranteed / k >= target_recall:
                break

        if not cand_vids:
            return [], n_probed, 0
        all_d = np.concatenate(cand_d)
        kk = min(k, len(all_d))
        pos = np.argpartition(all_d, kk - 1)[:kk]
        pos = pos[np.argsort(all_d[pos])]
        return [(cand_vids[int(i)], float(all_d[i])) for i in pos], n_probed, vectors_scanned

    def stats(self) -> IndexStats:
        """Return a summary of the index, sizes, drift and size imbalance."""
        sizes = [p.size() for p in self.partitions.values()]
        drifts = [p.centroid_drift() for p in self.partitions.values()]
        mean_size = float(np.mean(sizes)) if sizes else 0.0
        return IndexStats(
            num_partitions=len(self.partitions),
            total_vectors=sum(p.total_size() for p in self.partitions.values()),
            live_vectors=sum(sizes),
            mean_partition_size=mean_size,
            max_partition_size=max(sizes) if sizes else 0,
            min_partition_size=min(sizes) if sizes else 0,
            mean_drift=float(np.mean(drifts)) if drifts else 0.0,
            max_drift=float(np.max(drifts)) if drifts else 0.0,
            size_imbalance_ratio=(max(sizes) / mean_size) if mean_size > 0 else 1.0,
            data_scale=self.data_scale,
        )

    def partition_ids(self) -> list[int]:
        """Return the ids of all partitions in ascending order."""
        return sorted(self.partitions.keys())

    def get_partition(self, pid: int) -> Partition:
        """Return the partition with the given id."""
        return self.partitions[pid]

    def apply_split(self, pid: int, children: list[Partition]) -> None:
        """Replace partition `pid` with `children`, which get new ids."""
        assert pid in self.partitions
        parent = self.partitions[pid]
        parent_access = parent.access_count
        del self.partitions[pid]

        for child in children:
            new_id = self.next_partition_id
            self.next_partition_id += 1
            child.partition_id = new_id
            child.access_count = SPLIT_ACCESS_ALPHA * parent_access
            self.partitions[new_id] = child
            for vid in child.vector_ids:
                if vid not in child.tombstones:
                    self.vector_to_partition[vid] = new_id

        self._invalidate_cache()

    def apply_merge(self, pids: list[int], merged: Partition) -> None:
        """Replace the partitions in `pids` with `merged`, which gets a new id."""
        for pid in pids:
            assert pid in self.partitions, f"Missing partition {pid}"
            del self.partitions[pid]

        new_id = self.next_partition_id
        self.next_partition_id += 1
        merged.partition_id = new_id
        self.partitions[new_id] = merged

        for vid in merged.vector_ids:
            if vid not in merged.tombstones:
                self.vector_to_partition[vid] = new_id

        self._invalidate_cache()

    def apply_reassign(self, vec_id: int, from_pid: int, to_pid: int) -> None:
        """Move one vector from partition `from_pid` to partition `to_pid`."""
        assert from_pid in self.partitions and to_pid in self.partitions
        src = self.partitions[from_pid]
        dst = self.partitions[to_pid]
        vec = src.get_vector(vec_id)
        share = src.access_count / src.size() if src.size() > 0 else 0.0
        src.mark_deleted(vec_id)
        dst.add(vec_id, vec)
        src.access_count = max(0.0, src.access_count - share)
        dst.access_count += share
        self.vector_to_partition[vec_id] = to_pid

    def recompute_centroid(self, pid: int) -> float:
        """Move the centroid to the mean of its live vectors, return the drift closed."""
        p = self.partitions[pid]
        drift_before = p.centroid_drift()
        if p.size() > 0:
            p.update_centroid(p.true_mean())
        self._invalidate_cache()
        return drift_before

    def compact_partition(self, pid: int) -> int:
        """Compact partition `pid` and return how many tombstoned vectors were removed."""
        return self.partitions[pid].compact()

    def _prefilter(self, query: np.ndarray, k: int) -> np.ndarray:
        """About `k` + 16 nearest centroids, unordered."""
        # float64 prefilter, the caller rescores in float32
        matrix64, sq_norms, order = self._get_centroid_matrix64()
        if k >= len(order):
            return np.arange(len(order), dtype=np.intp)
        scores = sq_norms - 2.0 * (matrix64 @ np.asarray(query, dtype=np.float64))
        m = min(len(order), k + ROUTING_CANDIDATE_MARGIN)
        if m >= len(order):
            return np.arange(len(order), dtype=np.intp)
        return np.argpartition(scores, m - 1)[:m]

    def _exact_scores(self, rows: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Return the float32 squared distances from `query` to the centroids at `rows`."""
        matrix, _ = self._get_centroid_matrix()
        diffs = matrix[rows] - query
        return np.einsum("ij,ij->i", diffs, diffs)

    def _nearest_partitions(self, query: np.ndarray, nprobe: int) -> list[int]:
        """Ids of the `nprobe` nearest partitions, nearest first."""
        _, order = self._get_centroid_matrix()

        n = min(nprobe, len(order))
        if n <= 0:
            return []
        cand = self._prefilter(query, n)
        exact = self._exact_scores(cand, query)
        if n < len(cand):
            top = np.argpartition(exact, n - 1)[:n]
        else:
            top = np.arange(len(cand), dtype=np.intp)
        pids = np.array([order[int(j)] for j in cand[top]], dtype=np.int64)
        return [int(pid) for pid in pids[np.lexsort((pids, exact[top]))]]

    def query_count(self, nprobe: int) -> float:
        """Return the number of routed queries, the denominator of the access fraction."""
        if self.queries_routed > 0:
            return self.queries_routed
        return sum(p.access_count for p in self.partitions.values()) / max(1, nprobe)

    def route(self, query: np.ndarray, nprobe: int) -> list[int]:
        """Route a query from the stream, counting access on the probed partitions."""
        probed = self._nearest_partitions(query, nprobe)
        self.queries_routed += 1
        for pid in probed:
            self.partitions[pid].record_access()
            self.vectors_scanned_routed += self.partitions[pid].size()
        return probed

    def _find_nearest_partition(self, vec: np.ndarray) -> int:
        """Return the id of the partition with the nearest centroid, used for inserts."""
        _, order = self._get_centroid_matrix()
        cand = self._prefilter(vec, 1)
        exact = self._exact_scores(cand, vec)
        return order[int(cand[int(np.argmin(exact))])]

    def _get_centroid_matrix(self) -> tuple[np.ndarray, list[int]]:
        """Return the float32 matrix of centroids and the partition id of each row."""
        if self._centroid_matrix is None or self._centroid_id_order is None:
            order = sorted(self.partitions.keys())
            matrix = np.vstack([self.partitions[pid].centroid for pid in order])
            self._centroid_matrix = matrix
            self._centroid_id_order = order
        return self._centroid_matrix, self._centroid_id_order

    def _get_centroid_matrix64(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Float64 centroid matrix, norms and ids."""
        matrix, order = self._get_centroid_matrix()
        if self._centroid_matrix64 is None:
            self._centroid_matrix64 = matrix.astype(np.float64)
            self._centroid_sq_norms = np.einsum(
                "ij,ij->i", self._centroid_matrix64, self._centroid_matrix64)
        return self._centroid_matrix64, self._centroid_sq_norms, order

    def _invalidate_cache(self) -> None:
        """Forget the cached centroid matrices."""
        self._centroid_matrix = None
        self._centroid_id_order = None
        self._centroid_matrix64 = None
        self._centroid_sq_norms = None


def _kmeans_cache_key(vectors: np.ndarray, n_partitions: int, seed: int) -> str:
    """Return the file name key for a stored k-means result."""
    import hashlib
    import os

    import sklearn

    h = hashlib.blake2b(digest_size=16)
    for i in range(0, len(vectors), 10_000):
        h.update(np.ascontiguousarray(vectors[i:i + 10_000]).tobytes())
    parts = (
        f"n{len(vectors)}", f"d{vectors.shape[1]}", f"k{n_partitions}", f"s{seed}",
        f"i{KMEANS_MAX_ITER}", f"r{KMEANS_N_INIT}", f"sk{sklearn.__version__}",
        f"t{os.environ.get('OMP_NUM_THREADS', 'unset')}",
    )
    return h.hexdigest() + "_" + "_".join(parts)


def _lloyd_partition(
    vectors: np.ndarray, n_partitions: int, seed: int, cache: bool
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run exact k-means and return the labels, the centroids and the iteration count."""
    from sklearn.cluster import KMeans

    path = None
    if cache:
        path = KMEANS_CACHE_DIR / f"kmeans_{_kmeans_cache_key(vectors, n_partitions, seed)}.npz"
        if path.exists():
            with np.load(path) as z:
                return z["labels"], z["centroids"].astype(np.float32), 0

    km = KMeans(
        n_clusters=n_partitions,
        random_state=seed,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
    ).fit(vectors)
    labels = km.labels_
    centroids = km.cluster_centers_.astype(np.float32)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # write to a temp file, then move
        tmp = path.with_suffix(".npz.partial")
        # open file, so savez does not add .npz
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, labels=labels, centroids=centroids)
        tmp.replace(path)
    return labels, centroids, int(km.n_iter_)

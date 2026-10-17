"""The scan cost curve lambda(s) that the Quake and bandit policies decide with."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

from src.types import CostModel
from src.vector import l2_distance_batch


def benchmark_partition_scan(
    dim: int = 128,
    sizes: list[int] | None = None,
    n_trials: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Time the partition scan on random partitions of each size."""
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]

    import time

    rng = np.random.default_rng(seed)
    rows = []

    for s in sizes:
        matrix = rng.random((s, dim), dtype=np.float32)
        _ = l2_distance_batch(rng.random(dim, dtype=np.float32), matrix)  # warm the cache

        for trial in range(n_trials):
            q = rng.random(dim, dtype=np.float32)
            t0 = time.perf_counter_ns()
            _ = l2_distance_batch(q, matrix)
            elapsed = time.perf_counter_ns() - t0

            rows.append({"size": s, "trial": trial, "wall_time_ns": elapsed})

    return pd.DataFrame(rows)


def fit_cost_model(timings_df: pd.DataFrame) -> CostModel:
    """Fit latency as alpha times size. Not used in the reported runs."""
    x_feat = timings_df[["size"]].values
    y = timings_df["wall_time_ns"].values

    reg = LinearRegression(fit_intercept=False)
    reg.fit(x_feat, y)

    pred = reg.predict(x_feat)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum(y ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return CostModel(
        alpha=float(reg.coef_[0]),
        beta=0.0,
        gamma=0.0,
        intercept=0.0,
        r_squared=r2,
    )


@dataclass
class ProfiledCostModel:
    """lambda(s) as a table of measured points, interpolated."""

    sizes: np.ndarray
    latencies: np.ndarray
    r_squared: float = 1.0
    # zero without a dispatch measurement
    dispatch_ns: float = 0.0
    # cost of opening a partition
    probe_ns: float = 0.0
    # centroid comparison as a fraction of a vector comparison
    centroid_ratio: float = 1.0
    # centroid comparison in ns
    centroid_ns: float = 0.0

    def _raw(self, s: float) -> float:
        """Measured curve at size `s`, extended past the last point."""
        if s >= self.sizes[-1]:
            slope = ((self.latencies[-1] - self.latencies[-2])
                     / (self.sizes[-1] - self.sizes[-2]))
            return float(self.latencies[-1] + slope * (s - self.sizes[-1]))
        return float(np.interp(s, self.sizes, self.latencies))

    def scan_latency(self, size: int) -> float:
        """Return lambda(size), the cost of scanning one partition of `size` vectors."""
        s = float(max(0, size))
        if s <= self.sizes[0]:
            first = float(self.latencies[0]) - self.dispatch_ns
            base = first * (s / self.sizes[0]) if self.sizes[0] > 0 else first
        else:
            base = self._raw(s) - self.dispatch_ns
        # fixed cost of opening the partition
        return base + (self.probe_ns if s > 0 else 0.0)

    def centroid_latency(self, n_centroids: int) -> float:
        """Return lambda_c(K), the cost of comparing a query with `n_centroids` centroids."""
        s = float(max(0, n_centroids))
        if self.centroid_ns > 0.0:
            return self.centroid_ns * s
        if s <= self.sizes[0]:
            first = float(self.latencies[0]) - self.dispatch_ns
            base = first * (s / self.sizes[0]) if self.sizes[0] > 0 else 0.0
        else:
            base = self._raw(s) - self.dispatch_ns
        return self.centroid_ratio * base

    def predict_latency(self, size: int, drift: float = 0.0, access: int = 0) -> float:
        """Scan cost of a partition of `size` vectors."""
        return self.scan_latency(size)

    def to_dict(self) -> dict:
        """Return the model as a dictionary for storing in YAML."""
        return {
            "sizes":       [float(s) for s in self.sizes],
            "latencies":   [float(v) for v in self.latencies],
            "r_squared":   float(self.r_squared),
            "dispatch_ns": float(self.dispatch_ns),
            "probe_ns": float(self.probe_ns),
            "centroid_ratio": float(self.centroid_ratio),
            "centroid_ns": float(self.centroid_ns),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ProfiledCostModel:
        """Build a model from a dictionary, with defaults for any missing term."""
        return cls(
            sizes=np.asarray(payload["sizes"], dtype=float),
            latencies=np.asarray(payload["latencies"], dtype=float),
            r_squared=float(payload.get("r_squared", 1.0)),
            dispatch_ns=float(payload.get("dispatch_ns", 0.0)),
            probe_ns=float(payload.get("probe_ns", 0.0)),
            centroid_ratio=float(payload.get("centroid_ratio", 1.0)),
            centroid_ns=float(payload.get("centroid_ns", 0.0)),
        )


def lambda_from_prices(entry: dict) -> ProfiledCostModel:
    """Build lambda(s) as the Faiss time to open and scan a list of s vectors."""
    unit = float(entry["ns_per_vector_faiss"])
    return ProfiledCostModel(
        sizes=np.asarray(entry["list_sizes"], dtype=float),
        latencies=np.asarray(entry["list_prices"], dtype=float) * unit,
        r_squared=1.0, dispatch_ns=0.0, probe_ns=0.0,
        centroid_ratio=float(entry["centroid_ratio"]),
        centroid_ns=float(entry["centroid_ratio"]) * unit,
    )


def profile_scan_latency(
    dim: int = 128,
    sizes: list[int] | None = None,
    n_trials: int = 40,
    inner: int | None = None,
    seed: int = 42,
    window_ns: float = 2e6,
) -> ProfiledCostModel:
    """Measure lambda(s) of the Python scan. Not used by the reported runs."""
    if sizes is None:
        sizes = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]

    rng = np.random.default_rng(seed)
    latencies = [_min_batched_scan_ns(rng, s, dim, n_trials, inner, window_ns) for s in sizes]

    return ProfiledCostModel(
        sizes=np.asarray(sizes, dtype=float),
        latencies=np.asarray(latencies, dtype=float),
        dispatch_ns=profile_dispatch_ns(dim=dim, n_trials=n_trials, seed=seed,
                                        window_ns=window_ns),
    )


def _min_batched_scan_ns(
    rng: np.random.Generator, size: int, dim: int,
    n_trials: int, inner: int | None, window_ns: float,
) -> float:
    """Fastest mean scan time for one size."""
    import time

    matrix = rng.random((size, dim), dtype=np.float32)
    q = rng.random(dim, dtype=np.float32)
    l2_distance_batch(q, matrix)  # warm the cache

    if inner is None:
        t0 = time.perf_counter_ns()
        l2_distance_batch(q, matrix)
        single = max(1.0, float(time.perf_counter_ns() - t0))
        reps = int(min(2000, max(1, round(window_ns / single))))
    else:
        reps = inner

    best = float("inf")
    for _ in range(n_trials):
        t0 = time.perf_counter_ns()
        for _ in range(reps):
            l2_distance_batch(q, matrix)
        best = min(best, (time.perf_counter_ns() - t0) / reps)
    return best


def profile_dispatch_ns(
    dim: int = 128,
    n_trials: int = 40,
    seed: int = 42,
    window_ns: float = 2e6,
    reference_size: int = 25,
) -> float:
    """Measure the part of one scan call that does not depend on the number of vectors."""
    rng = np.random.default_rng(seed + 1)
    one = _min_batched_scan_ns(rng, 1, dim, n_trials, None, window_ns)
    ref = _min_batched_scan_ns(rng, reference_size, dim, n_trials, None, window_ns)
    per_vector = (ref - one) / (reference_size - 1)
    return max(0.0, one - per_vector)


def fit_profiled_cost_model(timings_df: pd.DataFrame) -> ProfiledCostModel:
    """Build lambda(s) from benchmark timings, the median time at each size."""
    g = timings_df.groupby("size")["wall_time_ns"].median().sort_index()
    return ProfiledCostModel(
        sizes=np.asarray(g.index, dtype=float),
        latencies=np.asarray(g.to_numpy(), dtype=float),
    )


def benchmark_cost_model_full(
    dim: int = 128,
    sizes: list[int] | None = None,
    drift_levels: list[float] | None = None,
    access_levels: list[float] | None = None,
    n_trials: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Time the scan while varying drift and access. Not used in the reported runs."""
    import time

    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    if drift_levels is None:
        drift_levels = [0.0, 0.5, 1.0, 2.0]
    if access_levels is None:
        access_levels = [0.0, 10.0, 100.0, 1000.0]

    rng = np.random.default_rng(seed)
    rows = []

    for s in sizes:
        for drift in drift_levels:
            for access in access_levels:
                matrix = rng.random((s, dim), dtype=np.float32)
                offset = rng.standard_normal(dim).astype(np.float32)
                norm = float(np.linalg.norm(offset))
                if norm > 0:
                    offset = offset / norm * np.float32(drift)
                _centroid = matrix.mean(axis=0) + offset  # computed but never scanned

                _ = l2_distance_batch(rng.random(dim, dtype=np.float32), matrix)

                for trial in range(n_trials):
                    q = rng.random(dim, dtype=np.float32)
                    t0 = time.perf_counter_ns()
                    _ = l2_distance_batch(q, matrix)
                    elapsed = time.perf_counter_ns() - t0
                    rows.append({
                        "size": s,
                        "drift": drift,
                        "access": access,
                        "trial": trial,
                        "wall_time_ns": elapsed,
                    })

    return pd.DataFrame(rows)


def fit_cost_model_full(timings_df: pd.DataFrame) -> tuple[CostModel, dict]:
    """Fit latency on size, drift and access. Not used in the reported runs."""
    from scipy import stats

    feature_names = ["size", "drift", "access"]
    x_feat = timings_df[feature_names].to_numpy(dtype=float)
    y = timings_df["wall_time_ns"].to_numpy(dtype=float)
    n = len(y)

    x_design = np.column_stack([np.ones(n), x_feat])
    p = x_design.shape[1]

    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    coef = xtx_inv @ x_design.T @ y

    pred = x_design @ coef
    resid = y - pred
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_full = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    dof = max(n - p, 1)
    sigma2 = ss_res / dof
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, coef / se, 0.0)
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=dof)

    size_only = fit_cost_model(timings_df[["size", "wall_time_ns"]])

    labels = ["intercept", *feature_names]
    diagnostics = {
        "coefficients": {
            labels[i]: {
                "estimate": float(coef[i]),
                "std_error": float(se[i]),
                "t_stat": float(t_stats[i]),
                "p_value": float(p_values[i]),
            }
            for i in range(p)
        },
        "r_squared_full": r2_full,
        "r_squared_size_only": size_only.r_squared,
        "n_observations": n,
    }

    model = CostModel(
        alpha=float(coef[1]),
        beta=float(coef[2]),
        gamma=float(coef[3]),
        intercept=float(coef[0]),
        r_squared=r2_full,
    )
    return model, diagnostics


def save_cost_model(model: CostModel, path: Path) -> None:
    """Write a linear cost model and a description of the machine to a YAML file."""
    payload = {
        "model": model.to_dict(),
        "machine_info": _machine_info(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def load_cost_model(path: Path) -> CostModel:
    """Read a linear cost model from a YAML file."""
    payload = yaml.safe_load(path.read_text())
    return CostModel.from_dict(payload["model"])


def save_profiled_cost_model(model: ProfiledCostModel, path: Path) -> None:
    """Write a scan cost curve to YAML."""
    payload = {
        "model": model.to_dict(),
        "machine_info": _machine_info(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def load_profiled_cost_model(path: Path) -> ProfiledCostModel:
    """Read a scan cost curve from a YAML file."""
    payload = yaml.safe_load(path.read_text())
    return ProfiledCostModel.from_dict(payload["model"])


def _machine_info() -> dict[str, str]:
    """Platform and library versions."""
    import platform

    return {
        "platform":       platform.platform(),
        "processor":      platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "numpy_version":  np.__version__,
    }

"""The scan cost curve lambda(s) that the Quake and bandit policies decide with.

lambda(s) is the cost of scanning one partition of s vectors, in nanoseconds, and lambda_c(K) is
the cost of ranking K centroids. A policy uses them to price an action before taking it. The
Quake policy prices every candidate action with them, and the reward of the bandit is the change
in the total Quake cost built on them.

The stored curves, `results/cost_model/profiled_lambda_dim128.yaml` and `..._dim960.yaml`, are
built by `lambda_from_prices` from the same Faiss price list the scoring uses, and
`scripts/calibrate_query_cost.py` writes them. A policy therefore decides in the unit it is scored
in. The profiling and fitting functions below measure the scan of this Python code instead, and
no reported run uses them.

The scoring never calls anything here. A policy that carries a cost model is making a claim about
what a query costs, and the scoring is the independent check of whether that claim paid off.
"""

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
    """Time the partition scan on random partitions of each size.

    Returns one row per trial with the size and the wall time in nanoseconds. Not used by the
    reported runs.
    """
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
    """Fit latency as alpha times size through the origin, and return it as a linear model.

    The line is forced through the origin because a free intercept comes out negative over these
    sizes, which would predict zero cost for small partitions. Drift and access get zero weight.
    """
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
    """lambda(s) as a table of measured points, interpolated between them, as in Quake section 4.1.

    Quake does not assume a formula for the scan cost, it measures it. `sizes` and `latencies`
    are the measured points. On the stored curves the latencies are the Faiss time to open and
    scan a list, so `dispatch_ns` and `probe_ns` are zero and `centroid_ns` prices one centroid
    comparison, see `lambda_from_prices`.

    The two other terms exist for curves measured on the Python scan. `dispatch_ns` is the part of
    a call that does not depend on the number of vectors, which in Python is the interpreter
    starting the numpy operations, about 4.3 microseconds at both dimensions. It is subtracted
    because a compiled engine such as Quake does not pay it, and leaving it in would make every
    split look expensive. `probe_ns` is a fixed cost added to every scan of a non empty partition.
    """

    sizes: np.ndarray
    latencies: np.ndarray
    r_squared: float = 1.0
    # zero for a curve built without a dispatch measurement, which then prices the raw curve
    dispatch_ns: float = 0.0
    # the cost of opening one partition beyond its vectors. zero on the stored curves, where the
    # opening is already part of every measured point
    probe_ns: float = 0.0
    # one centroid comparison as a fraction of one partition vector comparison, used when
    # centroid_ns is not set
    centroid_ratio: float = 1.0
    # the cost of one centroid comparison in nanoseconds. when set, the centroid scan is priced
    # linearly in K, as the scoring prices it
    centroid_ns: float = 0.0

    def _raw(self, s: float) -> float:
        """Return the measured curve at size `s`, extended past the last point along its last segment."""
        if s >= self.sizes[-1]:
            slope = ((self.latencies[-1] - self.latencies[-2])
                     / (self.sizes[-1] - self.sizes[-2]))
            return float(self.latencies[-1] + slope * (s - self.sizes[-1]))
        return float(np.interp(s, self.sizes, self.latencies))

    def scan_latency(self, size: int) -> float:
        """Return lambda(size), the cost of scanning one partition of `size` vectors.

        Below the smallest measured size the first point is scaled down linearly. A curve whose
        first point is at size 0, as the Faiss curve is, charges that point for an empty partition,
        since the router does probe empty partitions.
        """
        s = float(max(0, size))
        if s <= self.sizes[0]:
            first = float(self.latencies[0]) - self.dispatch_ns
            base = first * (s / self.sizes[0]) if self.sizes[0] > 0 else first
        else:
            base = self._raw(s) - self.dispatch_ns
        # the fixed cost of opening the partition, paid whenever it is probed at all
        return base + (self.probe_ns if s > 0 else 0.0)

    def centroid_latency(self, n_centroids: int) -> float:
        """Return lambda_c(K), the cost of comparing a query with `n_centroids` centroids.

        This is the top level scan every query performs, the delta O term of Quake equations 4 and
        5. It is one pass over the centroids, so it pays no per partition cost.
        """
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
        """Return the scan cost of a partition of `size` vectors. Drift and access are ignored."""
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
    """Build lambda(s) as the Faiss time to open and scan a list of s vectors.

    `entry` is the price list of one dimension from
    `results/cost_model/query_cost_calibration.yaml`, the same curve the scoring prices every
    probed list with, so a policy prices an action in the unit it is scored in. Quake measures
    lambda on the engine that serves its queries, and the closest equivalent here is Faiss
    IndexIVFFlat rather than the Python scan. Opening a list is already inside the curve, so no
    dispatch or per partition term is added, and a centroid comparison is priced as the scoring
    prices it.
    """
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
    """Measure lambda(s) of the Python scan. Not used by the reported runs.

    Each sample times a run of `inner` scans and divides, so the time per scan is large compared
    with the clock resolution and the loop overhead. Each size then takes the minimum over trials,
    because interruptions only ever add time, so the minimum is the least disturbed sample. With
    `inner` unset the repeat count is chosen per size to fill a window of `window_ns`, so small
    partitions are repeated often and large ones are not repeated needlessly.
    """
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
    """Return the minimum over trials of the mean time of a batch of scans, for one size."""
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
    """Measure the part of one scan call that does not depend on the number of vectors.

    The scan is timed at one vector and at `reference_size` vectors, and the line through the two
    is extended back to zero vectors. The random generator is seeded apart from
    `profile_scan_latency`, so adding this measurement leaves the curve unchanged.
    """
    rng = np.random.default_rng(seed + 1)
    one = _min_batched_scan_ns(rng, 1, dim, n_trials, None, window_ns)
    ref = _min_batched_scan_ns(rng, reference_size, dim, n_trials, None, window_ns)
    per_vector = (ref - one) / (reference_size - 1)
    return max(0.0, one - per_vector)


def fit_profiled_cost_model(timings_df: pd.DataFrame) -> ProfiledCostModel:
    """Build lambda(s) from benchmark timings, the median time at each size.

    The median is used because it is less affected by an occasional interrupted trial than the
    mean.
    """
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
    """Time the scan while also varying drift and access, to test whether they affect latency.

    Not used by the reported runs.
    """
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
    """Fit latency on size, drift and access by least squares, with the significance of each.

    Returns the fitted linear model and a dictionary with the estimate, standard error, t
    statistic and p value of every coefficient, and the R squared of the full and the size only
    fit.
    """
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
    """Write a measured scan cost curve and a description of the machine to a YAML file.

    A timed curve differs from one measurement to the next, so it is measured once and stored, as
    Quake does, and every run reads the stored curve. That keeps a run determined by its seed.
    """
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
    """Return the platform, processor, Python version and numpy version of this machine."""
    import platform

    return {
        "platform":       platform.platform(),
        "processor":      platform.processor() or "unknown",
        "python_version": platform.python_version(),
        "numpy_version":  np.__version__,
    }

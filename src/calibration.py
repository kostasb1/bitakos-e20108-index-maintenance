"""prices that turn operation counts into one cost unit"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from src.config import COST_MODEL_DIR

CALIBRATION_PATH: Path = COST_MODEL_DIR / "query_cost_calibration.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    """read the calibration file, cached by path"""
    return yaml.safe_load(Path(path).read_text())


def prices(dim: int, path: Path | None = None) -> dict:
    """return the calibrated prices for vectors of `dim` dimensions"""
    path = CALIBRATION_PATH if path is None else path
    data = _load(str(path))
    entry = data.get(dim) or data.get(str(dim))
    if entry is None:
        raise KeyError(f"no calibration for dimension {dim} in {path}")
    return entry.get("median", entry)


def has_price_list(dim: int, path: Path | None = None) -> bool:
    """return True if a list price curve exists for `dim` dimensions"""
    try:
        return "list_prices" in prices(dim, path)
    except KeyError:
        return False


def query_cost(priced: float, n_partitions: int, dim: int) -> float:
    """query cost, priced partition scans plus centroid ranking"""
    return float(priced + prices(dim)["centroid_ratio"] * n_partitions)


def list_price(sizes, dim: int, path: Path | None = None) -> np.ndarray:
    """Faiss price of scanning a list of each size"""
    p = prices(dim, path)
    xs = np.asarray(p["list_sizes"], dtype=float)
    ys = np.asarray(p["list_prices"], dtype=float)
    s = np.asarray(sizes, dtype=float)
    out = np.interp(s, xs, ys)
    above = s > xs[-1]
    if above.any():
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        out[above] = ys[-1] + slope * (s[above] - xs[-1])
    return out


def scan_unit_ns(lengths, list_ns, from_m: int = 200) -> float:
    """one scan distance in nanoseconds"""
    lengths, list_ns = np.asarray(lengths, dtype=float), np.asarray(list_ns, dtype=float)
    keep = lengths >= from_m
    return float(np.polyfit(lengths[keep], list_ns[keep], 1)[0])


def per_list_overhead_ns(lengths, list_ns, lo: int = 25, hi: int = 400) -> float:
    """return the fixed cost per list that a straight line through the curve implies"""
    lengths, list_ns = np.asarray(lengths, dtype=float), np.asarray(list_ns, dtype=float)
    keep = (lengths >= lo) & (lengths <= hi)
    return float(np.polyfit(lengths[keep], list_ns[keep], 1)[1])


def work_cost(reads: float, distances: float, kmeans_small: float, kmeans_bulk: float,
              dim: int) -> float:
    """maintenance work priced in scan distances"""
    p = prices(dim)
    return float(distances + p["kmeans_small"] * kmeans_small + p["kmeans_bulk"] * kmeans_bulk
                 + p["read_copy"] * reads)

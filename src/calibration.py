"""Calibrated query cost and calibrated work, the common unit every method is scored in.

Both are counts of operations, each priced against one scan distance of Faiss IndexIVFFlat on the
machine that ran the experiments, so no method is scored in its own units. The prices come from
`scripts/calibrate_query_cost.py`, as the median of five independent calibrations, and are stored
in `results/cost_model/query_cost_calibration.yaml`.

The query cost of one query at the target recall is

    the sum over probed partitions of list_price(size), plus centroid_ratio times K

since every query ranks all K centroids and then opens and scans nprobe partitions. list_price is
a measured curve, what Faiss takes to open and scan one list of a given length, in scan
distances. Counting scanned vectors alone would charge nothing for ranking the centroids or for
opening a list, which would let a method that fragments the index look cheap. A single price per
probed list does not work either, because the cost per vector itself depends on list length,
about 15 to 17 ns for lists of 25 vectors against about 10 ns for 100 to 800 on SIFT, so each list
is priced at its own length. Every price is measured on an index of the same size as the ones it
prices, 51,200 vectors.

The work of a run is

    distances, plus kmeans_small times small k-means distances, plus kmeans_bulk times bulk
    k-means distances, plus read_copy times vectors read or copied

since the methods spend their work on different kinds of operation, a rebuild on large k-means
runs, a split on a small one, LIRE and Quake on reassignment distances, a merge on copies. Pricing
each kind makes the totals comparable without changing what any method does.

No policy reads this module. It is the price list of the scoring, and the scan cost curves the
policies read are written from it by `scripts/calibrate_query_cost.py`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from src.config import COST_MODEL_DIR

CALIBRATION_PATH: Path = COST_MODEL_DIR / "query_cost_calibration.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    """Read the calibration file, cached by path."""
    return yaml.safe_load(Path(path).read_text())


def prices(dim: int, path: Path | None = None) -> dict:
    """Return the calibrated prices for vectors of `dim` dimensions.

    Raises KeyError if the file has no entry for that dimension.
    """
    path = CALIBRATION_PATH if path is None else path
    data = _load(str(path))
    entry = data.get(dim) or data.get(str(dim))
    if entry is None:
        raise KeyError(f"no calibration for dimension {dim} in {path}")
    return entry.get("median", entry)


def has_price_list(dim: int, path: Path | None = None) -> bool:
    """Return True if a list price curve exists for `dim` dimensions.

    An index in a dimension nothing was calibrated for is read without the priced cost. A full
    run needs the priced reading for every row, so there the missing entry raises an error later
    instead of being skipped here.
    """
    try:
        return "list_prices" in prices(dim, path)
    except KeyError:
        return False


def query_cost(priced: float, n_partitions: int, dim: int) -> float:
    """Return the query cost, the priced partition scan plus the ranking of all K centroids."""
    return float(priced + prices(dim)["centroid_ratio"] * n_partitions)


def list_price(sizes, dim: int, path: Path | None = None) -> np.ndarray:
    """Return what scanning a list of each given size costs in Faiss, in scan distances.

    The price includes opening the list. It is interpolated between the measured list sizes, and
    above the largest one it is extended along the last measured segment, where the cost per
    vector has settled.
    """
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
    """Return one scan distance in nanoseconds, the time one more vector adds to a long list.

    It is the least squares slope of list time against list length, over lists of at least
    `from_m` vectors.
    """
    lengths, list_ns = np.asarray(lengths, dtype=float), np.asarray(list_ns, dtype=float)
    keep = lengths >= from_m
    return float(np.polyfit(lengths[keep], list_ns[keep], 1)[0])


def per_list_overhead_ns(lengths, list_ns, lo: int = 25, hi: int = 400) -> float:
    """Return the fixed cost per list that a straight line through the curve implies.

    This is the intercept of the fit over the list lengths the methods reach. It is stored as a
    diagnostic and used in no price. Its range crosses zero on both datasets, which is why lists
    are priced on the whole measured curve instead of as vectors plus a constant.
    """
    lengths, list_ns = np.asarray(lengths, dtype=float), np.asarray(list_ns, dtype=float)
    keep = (lengths >= lo) & (lengths <= hi)
    return float(np.polyfit(lengths[keep], list_ns[keep], 1)[1])


def work_cost(reads: float, distances: float, kmeans_small: float, kmeans_bulk: float,
              dim: int) -> float:
    """Return calibrated maintenance work, each kind of operation priced in scan distances."""
    p = prices(dim)
    return float(distances + p["kmeans_small"] * kmeans_small + p["kmeans_bulk"] * kmeans_bulk
                 + p["read_copy"] * reads)

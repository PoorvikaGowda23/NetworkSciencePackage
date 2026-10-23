"""
timevarying.py
==============
Time-resolved memory estimation across clock hours of the day.

Implements binning of transition records into clock-time intervals (e.g. hourly
or 15-minute intervals) and fitting of the memory parameter p per bin.

For the MEM+SBM model, the community affinity matrix C is held fixed from the
global fit across all records (or the entire day), and only p is fitted per bin
to maintain numerical stability and consistency.
"""

from __future__ import annotations

import datetime
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from memory_trp.aggregation import TemporalGraph
from memory_trp.models import (
    _compute_Z_vectorized,
    _precompute_sbm_arrays,
    fit_mem_fixed_point,
    fit_mem_sbm,
)
from memory_trp.records import Records


# ---------------------------------------------------------------------------
# Clock time conversion
# ---------------------------------------------------------------------------

def record_to_clock_time(
    t_seconds: int | float | np.ndarray,
    timezone_str: str = "Europe/Paris",
) -> np.ndarray | float:
    """Convert absolute timestamps (seconds) to clock time in hours [0.0, 24.0).

    Handles both Unix epoch timestamps (> 1e8) and relative seconds-from-midnight
    timestamps (<= 1e8).

    Parameters
    ----------
    t_seconds : int, float, or np.ndarray
        Timestamp(s) in seconds.
    timezone_str : str, default="Europe/Paris"
        Timezone used when interpreting Unix epoch timestamps.
        Most SocioPatterns datasets were collected in France.

    Returns
    -------
    np.ndarray or float
        Time of day as a float in [0.0, 24.0). For example, 14:30 is 14.5.
    """
    is_array = isinstance(t_seconds, np.ndarray)
    arr = np.asarray(t_seconds, dtype=float)

    # If Unix epoch (> 1e8)
    if np.any(arr > 1e8):
        try:
            tz = ZoneInfo(timezone_str)
        except Exception:
            tz = datetime.timezone.utc

        def _epoch_to_hour(val: float) -> float:
            dt = datetime.datetime.fromtimestamp(val, tz=tz)
            return dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        if is_array:
            vfunc = np.vectorize(_epoch_to_hour, otypes=[float])
            return vfunc(arr)
        return _epoch_to_hour(float(arr))

    # Seconds within day or days offset
    hours = (arr % 86400.0) / 3600.0
    return hours if is_array else float(hours)


# ---------------------------------------------------------------------------
# Fixed-C SBM fitting for p
# ---------------------------------------------------------------------------

def fit_p_fixed_C(
    records: Records,
    C: np.ndarray,
    labels: np.ndarray,
    p_init: float = 0.5,
) -> float:
    """Fit memory parameter p for MEM+SBM holding affinity matrix C fixed.

    Minimizes the negative SBM log-likelihood with respect to scalar p in [0, 1].

    Parameters
    ----------
    records : Records
        Transition records for the bin.
    C : np.ndarray
        Fixed K x K community affinity matrix from the global fit.
    labels : np.ndarray
        Node community labels of length n.
    p_init : float, default=0.5
        Initial guess for scalar optimization.

    Returns
    -------
    float
        Fitted memory parameter p in [0.0, 1.0].
    """
    n_records = len(records)
    if n_records == 0:
        return 0.0

    k = C.shape[0]
    lbl_cur, lbl_nxt, lbl_prev, comm_counts, in_mask = _precompute_sbm_arrays(
        records, labels, k
    )
    Z = _compute_Z_vectorized(lbl_cur, lbl_prev, C, comm_counts, k)
    B = C[lbl_cur, lbl_nxt] / Z

    n_in = int(np.sum(in_mask))
    n_out = n_records - n_in

    # Boundary cases
    if n_in == 0:
        return 0.0
    if n_out == 0:
        return 1.0

    A_in = 1.0 / records.mem_size[in_mask].astype(float)
    B_in = B[in_mask]
    diff_in = A_in - B_in

    def neg_ll(p: float) -> float:
        p_val = np.clip(p, 1e-12, 1.0 - 1e-12)
        term_in = np.log(np.maximum(B_in + p_val * diff_in, 1e-15))
        ll = float(np.sum(term_in)) + n_out * float(np.log(1.0 - p_val))
        return -ll

    res = minimize_scalar(neg_ll, bounds=(0.0, 1.0), method="bounded")
    return float(np.clip(res.x, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API: fit_p_by_time_bin
# ---------------------------------------------------------------------------

def fit_p_by_time_bin(
    records: Records,
    graph_info: TemporalGraph | dict[str, Any],
    bin_minutes: int = 60,
    model: str = "MEM",
    C_global: np.ndarray | None = None,
    min_records: int = 200,
) -> dict[float, float]:
    """Fit memory parameter p separately in each clock-time bin.

    Converts transition records to clock times using records.time_idx,
    graph.start_time, and graph.t_res. Groups records into time bins of width
    `bin_minutes`. Bins with fewer than `min_records` transitions are skipped.

    For MEM+SBM, the community matrix C is held fixed from a global fit
    (either passed in via `C_global` or computed globally once from `records`).

    Parameters
    ----------
    records : Records
        Transition records extracted from time-respecting paths.
    graph_info : TemporalGraph or dict
        Graph object or metadata dictionary providing:
          - start_time: int (epoch or start seconds)
          - t_res: int (seconds per snapshot)
          - n: int (number of nodes)
          - labels: np.ndarray or None (community labels)
    bin_minutes : int, default=60
        Width of each clock-time bin in minutes (e.g. 60 for hourly, 15 for 15-min).
    model : str, default="MEM"
        Model to fit: "MEM" or "MEM+SBM".
    C_global : np.ndarray or None, default=None
        Optional precomputed community affinity matrix for MEM+SBM.
        If None and model is "MEM+SBM", fitted once globally on `records`.
    min_records : int, default=200
        Minimum number of records required in a bin to perform fitting.

    Returns
    -------
    dict[float, float]
        Mapping from bin center clock hour (e.g. 8.5 for 8:00-9:00) to fitted p.
    """
    if len(records) == 0:
        return {}

    # Extract metadata from TemporalGraph or dict
    if isinstance(graph_info, TemporalGraph):
        start_time = graph_info.start_time
        t_res = graph_info.t_res
        n = graph_info.n
        labels = graph_info.labels
    else:
        start_time = int(graph_info["start_time"])
        t_res = int(graph_info["t_res"])
        n = int(graph_info["n"])
        labels = graph_info.get("labels", None)

    is_sbm = model.upper() in ["MEM+SBM", "MEM_SBM", "SBM"]
    if is_sbm and labels is None:
        raise ValueError("Cannot fit MEM+SBM without node community labels.")

    # Absolute timestamp of each record: start_time + time_idx * t_res
    t_abs = start_time + records.time_idx * t_res
    clock_hours = record_to_clock_time(t_abs)

    bin_width_hours = bin_minutes / 60.0

    # For MEM+SBM, compute global C once if not supplied
    C_mat = C_global
    if is_sbm and C_mat is None:
        global_fit = fit_mem_sbm(records, labels=labels)
        C_mat = global_fit.C

    # Discretize into bin indices
    bin_indices = np.floor(clock_hours / bin_width_hours).astype(int)
    unique_bins = sorted(np.unique(bin_indices))

    results: dict[float, float] = {}

    for b in unique_bins:
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count < min_records:
            continue

        bin_rec = records[mask]
        bin_center = (b + 0.5) * bin_width_hours

        if is_sbm:
            assert C_mat is not None and labels is not None
            p_val = fit_p_fixed_C(bin_rec, C=C_mat, labels=labels)
        else:
            p_val = fit_mem_fixed_point(bin_rec, n=n).p

        results[float(round(bin_center, 4))] = float(p_val)

    return results


def fit_p_by_time_bin_df(
    records: Records,
    graph_info: TemporalGraph | dict[str, Any],
    bin_minutes: int = 60,
    model: str = "MEM",
    C_global: np.ndarray | None = None,
    min_records: int = 200,
) -> pd.DataFrame:
    """Tabular variant of fit_p_by_time_bin returning a pandas DataFrame."""
    if len(records) == 0:
        return pd.DataFrame(columns=["bin_center", "bin_start", "bin_end", "p", "n_records"])

    if isinstance(graph_info, TemporalGraph):
        start_time = graph_info.start_time
        t_res = graph_info.t_res
        n = graph_info.n
        labels = graph_info.labels
    else:
        start_time = int(graph_info["start_time"])
        t_res = int(graph_info["t_res"])
        n = int(graph_info["n"])
        labels = graph_info.get("labels", None)

    is_sbm = model.upper() in ["MEM+SBM", "MEM_SBM", "SBM"]
    t_abs = start_time + records.time_idx * t_res
    clock_hours = record_to_clock_time(t_abs)
    bin_width_hours = bin_minutes / 60.0

    C_mat = C_global
    if is_sbm and C_mat is None:
        if labels is None:
            raise ValueError("Cannot fit MEM+SBM without node community labels.")
        global_fit = fit_mem_sbm(records, labels=labels)
        C_mat = global_fit.C

    bin_indices = np.floor(clock_hours / bin_width_hours).astype(int)
    unique_bins = sorted(np.unique(bin_indices))

    rows = []
    for b in unique_bins:
        mask = bin_indices == b
        count = int(np.sum(mask))
        if count < min_records:
            continue

        bin_rec = records[mask]
        bin_start = b * bin_width_hours
        bin_end = (b + 1) * bin_width_hours
        bin_center = (b + 0.5) * bin_width_hours

        if is_sbm:
            assert C_mat is not None and labels is not None
            p_val = fit_p_fixed_C(bin_rec, C=C_mat, labels=labels)
        else:
            p_val = fit_mem_fixed_point(bin_rec, n=n).p

        rows.append({
            "bin_center": round(bin_center, 4),
            "bin_start": round(bin_start, 4),
            "bin_end": round(bin_end, 4),
            "p": round(p_val, 5),
            "n_records": count,
        })

    return pd.DataFrame(rows)

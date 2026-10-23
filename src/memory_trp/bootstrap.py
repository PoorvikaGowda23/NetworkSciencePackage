"""
bootstrap.py
============
Non-parametric bootstrap estimation for memory parameter uncertainty.

Provides resampled confidence intervals for the memory parameter p under MEM
and MEM+SBM models by resampling transition records with replacement.
"""

from __future__ import annotations

from typing import NamedTuple
import numpy as np

from config import N_BOOTSTRAP, RANDOM_SEED
from memory_trp.models import fit_mem_fixed_point, fit_mem_sbm
from memory_trp.records import Records
from memory_trp.timevarying import fit_p_fixed_C


class BootstrapResult(NamedTuple):
    """Container for bootstrap estimate and confidence intervals.

    Attributes
    ----------
    mean : float
        Empirical mean of bootstrap estimates p_1*, ..., p_B*.
    p2_5 : float
        2.5th percentile (lower bound of 95% two-sided confidence interval).
    p97_5 : float
        97.5th percentile (upper bound of 95% two-sided confidence interval).
    """

    mean: float
    p2_5: float
    p97_5: float


def bootstrap_p(
    records: Records,
    n: int,
    model: str = "MEM",
    B: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
    labels: np.ndarray | None = None,
    C_global: np.ndarray | None = None,
) -> BootstrapResult:
    """Resample transition records with replacement B times, refit, and return confidence intervals.

    Parameters
    ----------
    records : Records
        Transition records container extracted from time-respecting paths.
    n : int
        Total number of nodes in the network.
    model : str, default="MEM"
        Model to fit: "MEM" (fixed-point iteration) or "MEM+SBM".
    B : int, default=N_BOOTSTRAP
        Number of bootstrap replications.
    rng : np.random.Generator or None, default=None
        Numpy random generator for reproducible bootstrap draws.
    labels : np.ndarray or None, default=None
        Node community labels of length n, required when model is "MEM+SBM".
    C_global : np.ndarray or None, default=None
        Optional precomputed community interaction matrix for MEM+SBM.
        If None and model is "MEM+SBM", fitted once globally on `records`.

    Returns
    -------
    BootstrapResult
        NamedTuple with fields `mean`, `p2_5`, and `p97_5`. Also unpackable as a 3-tuple.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    n_records = len(records)
    if n_records == 0:
        return BootstrapResult(0.0, 0.0, 0.0)

    is_sbm = model.upper() in ["MEM+SBM", "MEM_SBM", "SBM"]

    if is_sbm:
        if labels is None:
            raise ValueError("labels must be provided for MEM+SBM bootstrap.")
        if C_global is None:
            global_fit = fit_mem_sbm(records, labels=labels)
            C_global = global_fit.C

    p_samples = np.empty(B, dtype=float)

    for b in range(B):
        # Resample records with replacement
        boot_idx = rng.choice(n_records, size=n_records, replace=True)
        boot_rec = records[boot_idx]

        if is_sbm:
            assert C_global is not None and labels is not None
            p_val = fit_p_fixed_C(boot_rec, C=C_global, labels=labels)
        else:
            p_val = fit_mem_fixed_point(boot_rec, n=n).p

        p_samples[b] = p_val

    mean_val = float(np.mean(p_samples))
    p2_5 = float(np.percentile(p_samples, 2.5))
    p97_5 = float(np.percentile(p_samples, 97.5))

    return BootstrapResult(mean=mean_val, p2_5=p2_5, p97_5=p97_5)

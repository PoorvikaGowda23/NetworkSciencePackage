"""
Tests for models.py (MEM model)
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import pytest

from memory_trp.models import (
    FitResult,
    bic,
    fit_mem_fixed_point,
    fit_mem_numeric,
    log_likelihood_mem,
    simulate_mem_records,
)


def test_mem_recovery_p_03():
    """Simulate records with p=0.3, fit with both methods, and confirm they agree and recover 0.3."""
    rng = np.random.default_rng(42)
    true_p = 0.3
    n = 60
    num_records = 35_000
    mem_size = 5

    records = simulate_mem_records(
        n=n,
        num_records=num_records,
        p=true_p,
        rng=rng,
        mem_size=mem_size,
    )

    # 1. Fixed-point estimation
    res_fp = fit_mem_fixed_point(records, n=n)
    assert isinstance(res_fp, FitResult)
    assert res_fp.converged is True
    assert res_fp.n_records == num_records
    assert res_fp.n_params == 1

    # 2. Numerical scalar optimization
    res_num = fit_mem_numeric(records, n=n)
    assert isinstance(res_num, FitResult)
    assert res_num.converged is True

    # Confirm both methods agree tightly (< 1e-4)
    assert abs(res_fp.p - res_num.p) < 1e-4, f"Mismatch: fp={res_fp.p}, num={res_num.p}"

    # Confirm parameter recovery (within 0.02 of true p=0.3)
    assert abs(res_fp.p - true_p) < 0.02, f"Failed recovery: got {res_fp.p}, expected ~{true_p}"

    # Log-likelihoods and BICs should match
    assert abs(res_fp.loglik - res_num.loglik) < 1e-2
    assert abs(res_fp.bic - res_num.bic) < 1e-2


def test_mem_log_likelihood_maximum():
    """Verify that log-likelihood is higher at true p than distant values."""
    rng = np.random.default_rng(123)
    records = simulate_mem_records(n=40, num_records=10_000, p=0.4, rng=rng, mem_size=4)

    ll_true = log_likelihood_mem(0.4, records, n=40)
    ll_low = log_likelihood_mem(0.1, records, n=40)
    ll_high = log_likelihood_mem(0.8, records, n=40)

    assert ll_true > ll_low
    assert ll_true > ll_high


def test_mem_corner_cases():
    """Verify behavior on empty records."""
    rng = np.random.default_rng(999)
    # Empty records
    empty_records = simulate_mem_records(n=10, num_records=0, p=0.5, rng=rng, mem_size=3)
    res = fit_mem_fixed_point(empty_records, n=10)
    assert res.p == 0.0
    assert res.n_records == 0
    assert res.converged is True


def test_bic_function():
    """Verify bic formula and parameter counts for MEM and MEM+SBM."""
    # Test formula: n_params * ln(n_records) - 2 * loglik
    expected_mem = 1 * np.log(1000) - 2 * (-100.0)
    assert np.isclose(bic(-100.0, 1, 1000), expected_mem)

    # For MEM+SBM with k=4 communities:
    # Parameter count: 1 + k*(k+1)/2 - 1 = 1 + 10 - 1 = 10
    k = 4
    n_params_sbm = 1 + k * (k + 1) // 2 - 1
    assert n_params_sbm == 10
    expected_sbm = 10 * np.log(1000) - 2 * (-80.0)
    assert np.isclose(bic(-80.0, n_params_sbm, 1000), expected_sbm)

    # Empty records gives 0.0
    assert bic(-100.0, 1, 0) == 0.0


def test_fit_p_per_group():
    """Verify that fit_p_per_group fits p per community and filters small groups."""
    from memory_trp.models import fit_p_per_group, simulate_mem_sbm_records
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 60
    # 3 communities: groups 0 and 1 have 25 nodes, group 2 has only 10 nodes
    labels = np.array([0] * 25 + [1] * 25 + [2] * 10)
    C = np.array([
        [2.0, 0.4, 0.2],
        [0.4, 2.0, 0.2],
        [0.2, 0.2, 2.0],
    ])

    records = simulate_mem_sbm_records(
        n=n, labels=labels, num_records=800, p=0.35, C=C, rng=rng, mem_size=4
    )

    # Fit per group with min_records=100 and small B=10 for speed
    df = fit_p_per_group(records, labels=labels, C=C, n=n, min_records=100, B=10, rng=rng)

    assert isinstance(df, pd.DataFrame)
    assert set(["group", "n_records", "p_hat", "ci_low", "ci_high"]).issubset(df.columns)
    for _, row in df.iterrows():
        assert row["n_records"] >= 100
        assert 0.0 <= row["p_hat"] <= 1.0
        assert row["ci_low"] <= row["ci_high"]


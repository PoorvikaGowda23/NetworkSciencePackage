"""
tests/test_sbm.py
=================
Tests for the MEM+SBM model added in Step 7.

Test plan
---------
1. test_sbm_p_recovery
   Simulate records from a known p and C, fit with fit_mem_sbm, and check that
   the recovered p is close to the true value.

2. test_sbm_C_structure
   After fitting, C must be symmetric, have positive entries, and mean == 1.

3. test_sbm_single_community_matches_mem
   When all labels are 0 (one community), MEM+SBM should reduce to plain MEM
   (same p, same log-likelihood to within numerical tolerance).

4. test_sbm_loglik_higher_than_flat_C
   After fitting, the log-likelihood under the fitted C must be >= the
   log-likelihood under the flat C=ones (the optimiser can only improve).
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import pytest

from memory_trp.models import (
    FitResult,
    SBMFitResult,
    community_interaction_matrix,
    fit_mem_fixed_point,
    fit_mem_sbm,
    log_likelihood_mem,
    log_likelihood_sbm,
    simulate_mem_sbm_records,
    simulate_mem_records,
    _normalise_C,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_labels(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Assign n nodes to k communities as evenly as possible (then shuffle)."""
    labels = np.repeat(np.arange(k), n // k + 1)[:n]
    rng.shuffle(labels)
    return labels.astype(int)


def _make_C(k: int, assortative: float = 3.0) -> np.ndarray:
    """Return a normalised K x K affinity matrix with strong diagonal (assortative)."""
    C = np.ones((k, k), dtype=float)
    np.fill_diagonal(C, assortative)
    return _normalise_C(C)


# ---------------------------------------------------------------------------
# Test 1: p recovery
# ---------------------------------------------------------------------------

def test_sbm_p_recovery():
    """Simulate records from known p and C; fit_mem_sbm should recover p within 0.05."""
    rng = np.random.default_rng(7)

    n, k = 80, 4
    true_p = 0.35
    mem_size = 5

    labels = _make_labels(n, k, rng)
    C_true = _make_C(k, assortative=4.0)

    records = simulate_mem_sbm_records(
        n=n,
        labels=labels,
        num_records=30_000,
        p=true_p,
        C=C_true,
        rng=rng,
        mem_size=mem_size,
    )

    result = fit_mem_sbm(records, labels=labels, p0=0.5)

    assert isinstance(result, SBMFitResult)
    assert result.n_records == 30_000
    assert 0.0 <= result.p <= 1.0, f"p out of range: {result.p}"
    assert abs(result.p - true_p) < 0.05, (
        f"p recovery failed: got {result.p:.4f}, expected ~{true_p}"
    )


# ---------------------------------------------------------------------------
# Test 2: C structure
# ---------------------------------------------------------------------------

def test_sbm_C_structure():
    """Fitted C must be symmetric, positive, and have mean == 1."""
    rng = np.random.default_rng(13)

    n, k = 60, 3
    labels = _make_labels(n, k, rng)
    C_true = _make_C(k, assortative=3.0)

    records = simulate_mem_sbm_records(
        n=n, labels=labels, num_records=10_000,
        p=0.25, C=C_true, rng=rng, mem_size=4,
    )

    result = fit_mem_sbm(records, labels=labels)

    C = result.C
    assert C.shape == (k, k), f"Wrong C shape: {C.shape}"
    assert np.allclose(C, C.T, atol=1e-10), "C is not symmetric"
    assert np.all(C > 0), f"C has non-positive entries: {C}"
    assert abs(C.mean() - 1.0) < 1e-6, f"mean(C) = {C.mean()}, expected 1.0"


# ---------------------------------------------------------------------------
# Test 3: single community matches plain MEM
# ---------------------------------------------------------------------------

def test_sbm_single_community_matches_mem():
    """With all nodes in one community, MEM+SBM must give the same p as plain MEM."""
    rng = np.random.default_rng(99)
    n = 50
    mem_size = 5

    labels = np.zeros(n, dtype=int)

    true_p = 0.4
    records = simulate_mem_records(
        n=n, num_records=20_000, p=true_p, rng=rng, mem_size=mem_size
    )

    res_mem = fit_mem_fixed_point(records, n=n)
    res_sbm = fit_mem_sbm(records, labels=labels)

    assert abs(res_mem.p - res_sbm.p) < 0.02, (
        f"p mismatch: MEM={res_mem.p:.4f}, SBM={res_sbm.p:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: fitted C beats flat C in log-likelihood
# ---------------------------------------------------------------------------

def test_sbm_loglik_higher_than_flat_C():
    """After fitting, the optimised C must give >= log-likelihood than flat C."""
    rng = np.random.default_rng(42)

    n, k = 70, 3
    labels = _make_labels(n, k, rng)
    C_true = _make_C(k, assortative=5.0)

    records = simulate_mem_sbm_records(
        n=n, labels=labels, num_records=15_000,
        p=0.3, C=C_true, rng=rng, mem_size=5,
    )

    C_flat = np.ones((k, k), dtype=float)
    ll_flat = log_likelihood_sbm(0.5, C_flat, records, labels, n)

    result = fit_mem_sbm(records, labels=labels, init_C=None, p0=0.5)
    ll_fitted = result.loglik

    assert ll_fitted >= ll_flat - 1e-6, (
        f"Fitted log-likelihood ({ll_fitted:.2f}) < flat-C ({ll_flat:.2f})"
    )


# ---------------------------------------------------------------------------
# Test 5: community interaction matrix
# ---------------------------------------------------------------------------

def test_community_interaction_matrix():
    """Test community interaction matrix computation from a TemporalGraph."""
    import scipy.sparse as sp
    from memory_trp.aggregation import TemporalGraph

    n, k = 10, 2
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    # snapshot 1: within community 0 (0-1), within community 1 (5-6)
    row = [0, 1, 5, 6]
    col = [1, 0, 6, 5]
    data = [2.0, 2.0, 3.0, 3.0]
    snap = sp.csr_matrix((data, (row, col)), shape=(n, n))
    graph = TemporalGraph(name="test", n=n, t_res=20, snapshots=[snap], labels=labels, start_time=0)

    C = community_interaction_matrix(graph)
    assert C.shape == (2, 2)
    assert np.allclose(C, C.T)
    assert np.isclose(C.mean(), 1.0)
    # within-community weights should exceed cross-community weights
    assert C[0, 0] > C[0, 1]
    assert C[1, 1] > C[0, 1]


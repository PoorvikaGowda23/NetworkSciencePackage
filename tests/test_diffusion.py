"""
test_diffusion.py
=================
Unit tests for heat diffusion and synthetic network generator.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import warnings
import numpy as np
import pytest
import scipy.sparse as sp

from memory_trp.aggregation import TemporalGraph
from memory_trp.diffusion import run_diffusion, _normalized_entropy
from memory_trp.generator import generate_memory_graph


def test_normalized_entropy_bounds():
    """Entropy must be 0 for unit vector and 1 for uniform vector."""
    n = 100
    # Point mass -> H = 0
    u_point = np.zeros(n)
    u_point[0] = 1.0
    assert np.isclose(_normalized_entropy(u_point, n), 0.0)

    # Uniform mass -> H = 1
    u_unif = np.ones(n) / n
    assert np.isclose(_normalized_entropy(u_unif, n), 1.0)


def test_run_diffusion_disconnected():
    """Diffusion on an empty graph (no edges) leaves entropy at 0."""
    n = 20
    empty_snap = sp.csr_matrix((n, n), dtype=float)
    g = TemporalGraph(name="empty", n=n, t_res=20, snapshots=[empty_snap] * 5, labels=None, start_time=0)
    rng = np.random.default_rng(42)

    h = run_diffusion(g, beta=0.03, rng=rng)
    assert np.isclose(h, 0.0)


def test_run_diffusion_spreading():
    """Diffusion on a connected graph spreads probability and increases entropy."""
    n = 10
    # Complete graph in every snapshot
    A = np.ones((n, n)) - np.eye(n)
    snap = sp.csr_matrix(A)
    g = TemporalGraph(name="clique", n=n, t_res=20, snapshots=[snap] * 10, labels=None, start_time=0)
    rng = np.random.default_rng(42)

    # With beta = 0.05, 1/d = 1/9 ~ 0.11 > 0.05 (stable)
    h = run_diffusion(g, beta=0.05, rng=rng)
    assert 0.0 < h <= 1.0


def test_run_diffusion_clipping_warning():
    """Large beta causing severe instability emits a warning."""
    n = 10
    A = np.ones((n, n)) - np.eye(n)
    snap = sp.csr_matrix(A)
    g = TemporalGraph(name="clique", n=n, t_res=20, snapshots=[snap] * 20, labels=None, start_time=0)
    rng = np.random.default_rng(42)

    # Huge beta = 2.0 (way above 1/d_max) will cause large negative values and clipping
    with pytest.warns(UserWarning, match="total clipped probability mass"):
        h = run_diffusion(g, beta=2.0, rng=rng, clip_warn_threshold=0.01)
    assert 0.0 <= h <= 1.0


def test_generate_memory_graph_structure():
    """Synthetic generator produces expected number of snapshots and nodes."""
    rng = np.random.default_rng(123)
    g = generate_memory_graph(n=30, T=20, d=4.0, alpha=0.3, m_hat=3, rng=rng, verbose=False)
    assert g.n == 30
    assert len(g.snapshots) == 20
    for snap in g.snapshots:
        assert snap.shape == (30, 30)
        # Check symmetry
        diff = snap - snap.T
        assert diff.nnz == 0

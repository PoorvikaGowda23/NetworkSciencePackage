"""
tests/test_sir.py
=================
Unit tests for Step 17 SIR epidemic spreading simulation and rate calibration.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import scipy.sparse as sp

from memory_trp.aggregation import TemporalGraph
from memory_trp.sir import SIRResult, calibrate_rate, simulate_sir


def _build_toy_graph(n: int = 10, n_snaps: int = 15, seed: int = 42) -> TemporalGraph:
    """Helper to construct a small synthetic temporal graph for unit testing."""
    rng = np.random.default_rng(seed)
    snapshots = []
    for _ in range(n_snaps):
        u = rng.integers(0, n, size=4)
        v = rng.integers(0, n, size=4)
        mask = u != v
        u, v = u[mask], v[mask]
        data = np.ones(len(u), dtype=float)
        adj = sp.csr_matrix((data, (u, v)), shape=(n, n), dtype=float)
        snapshots.append(adj + adj.T)

    return TemporalGraph(
        name="toy_test",
        n=n,
        t_res=20,
        snapshots=snapshots,
        labels=None,
        start_time=0,
    )


def test_sir_result_unpacking_and_types():
    """Verify that simulate_sir returns an SIRResult with correct types and 3-tuple unpacking."""
    graph = _build_toy_graph()
    rng = np.random.default_rng(123)

    res = simulate_sir(graph, transmission_rate=0.5, recovery_snapshots=3, rng=rng)

    assert isinstance(res, SIRResult)
    assert 0.0 <= res.final_fraction <= 1.0
    assert 0 <= res.peak_snapshot < len(graph.snapshots)
    assert isinstance(res.infected_counts, np.ndarray)
    assert res.time_to_peak >= 0

    # 3-tuple unpacking
    frac, peak, counts = res
    assert frac == res.final_fraction
    assert peak == res.peak_snapshot
    assert np.array_equal(counts, res.infected_counts)


def test_sir_zero_transmission_rate():
    """With zero transmission rate, only the single seed node should ever be infected."""
    graph = _build_toy_graph(n=8, n_snaps=10)
    rng = np.random.default_rng(42)

    res = simulate_sir(graph, transmission_rate=0.0, recovery_snapshots=2, rng=rng)

    # Exactly 1 node out of 8 was infected (the seed)
    expected_frac = 1.0 / graph.n
    assert abs(res.final_fraction - expected_frac) < 1e-9
    assert res.infected_counts[0] == 1
    # After recovery, infected count drops to 0
    assert res.infected_counts[-1] == 0


def test_sir_high_transmission_dense_graph():
    """With high transmission rate and long recovery on a dense graph, outbreak should spread."""
    # Build complete graph in every snapshot
    n = 6
    adj = sp.csr_matrix(np.ones((n, n)) - np.eye(n))
    snapshots = [adj.copy() for _ in range(8)]
    dense_g = TemporalGraph(name="dense", n=n, t_res=20, snapshots=snapshots, labels=None, start_time=0)

    rng = np.random.default_rng(42)
    res = simulate_sir(
        dense_g,
        transmission_rate=10.0,
        recovery_snapshots=5,
        rng=rng,
        start_snapshot=0,
        seed_node=0,
    )

    # All nodes should be reached
    assert res.final_fraction == 1.0
    assert np.max(res.infected_counts) >= 2


def test_calibrate_rate_returns_positive_float():
    """Verify that calibrate_rate returns a valid positive transmission rate."""
    graph = _build_toy_graph(n=12, n_snaps=20)
    rng = np.random.default_rng(42)

    rate = calibrate_rate(
        graph=graph,
        target_final_fraction=0.40,
        recovery_snapshots=5,
        n_calib_runs=10,
        rng=rng,
    )

    assert isinstance(rate, float)
    assert rate > 0.0

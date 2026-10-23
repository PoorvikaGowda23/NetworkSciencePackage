"""
test_bootstrap.py
=================
Unit tests for bootstrap confidence intervals and time-shuffled null model.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import scipy.sparse as sp
import pytest

from memory_trp.aggregation import TemporalGraph
from memory_trp.bootstrap import BootstrapResult, bootstrap_p
from memory_trp.nulls import time_shuffle_graph
from memory_trp.records import Records


def test_bootstrap_p_basic():
    """Bootstrap should return mean and 95% CI with p2_5 <= mean <= p97_5."""
    n = 20
    # Create records with mixed in_memory
    n_rec = 100
    in_mem = np.zeros(n_rec, dtype=bool)
    in_mem[:30] = True  # 30% in memory
    rec = Records(
        cur=np.zeros(n_rec, dtype=int),
        prev=np.ones(n_rec, dtype=int),
        nxt=np.full(n_rec, 2, dtype=int),
        in_memory=in_mem,
        mem_size=np.full(n_rec, 3, dtype=int),
        mem_sets=np.zeros((n_rec, 3), dtype=int),
        time_idx=np.zeros(n_rec, dtype=int),
    )
    rng = np.random.default_rng(42)
    res = bootstrap_p(rec, n=n, model="MEM", B=50, rng=rng)

    assert isinstance(res, BootstrapResult)
    assert 0.0 <= res.p2_5 <= res.mean <= res.p97_5 <= 1.0


def test_time_shuffle_graph_preservation():
    """Time shuffle should preserve individual snapshot structure but permute ordering."""
    n = 10
    rng = np.random.default_rng(123)

    # 5 snapshots with different edge counts: 1, 2, 3, 4, 5
    snaps = []
    for num_edges in [1, 2, 3, 4, 5]:
        A = np.zeros((n, n), dtype=float)
        for e in range(num_edges):
            A[e, e + 1] = 1.0
            A[e + 1, e] = 1.0
        snaps.append(sp.csr_matrix(A))

    g = TemporalGraph(name="test", n=n, t_res=20, snapshots=snaps, labels=None, start_time=0)
    g_shuff = time_shuffle_graph(g, rng=rng)

    assert g_shuff.n == g.n
    assert len(g_shuff.snapshots) == len(g.snapshots)

    # Total active edges across all snapshots must be exactly identical
    orig_total_edges = sum(s.nnz for s in g.snapshots)
    shuff_total_edges = sum(s.nnz for s in g_shuff.snapshots)
    assert orig_total_edges == shuff_total_edges

    # Set of edge counts must be identical
    orig_counts = sorted([s.nnz for s in g.snapshots])
    shuff_counts = sorted([s.nnz for s in g_shuff.snapshots])
    assert orig_counts == shuff_counts

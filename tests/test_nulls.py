"""
Tests for nulls.py (Erdos-Renyi null model)
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import pytest

from memory_trp.aggregation import build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.nulls import er_null_graph


def test_er_null_properties():
    """Verify edge count preservation, symmetry, no self-loops, and weight 1."""
    contacts, labels = load_contacts("hospital")
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    days = split_by_day(contacts, "hospital")
    graph = build_temporal_graph(days[0], n=n, labels=labels, t_res=20, name="hospital")

    rng = np.random.default_rng(42)
    null_graph = er_null_graph(graph, rng=rng)

    assert null_graph.n == graph.n
    assert len(null_graph.snapshots) == len(graph.snapshots)

    for orig_s, null_s in zip(graph.snapshots, null_graph.snapshots):
        # 1. Preserves number of edges exactly
        assert null_s.nnz == orig_s.nnz

        if null_s.nnz > 0:
            # 2. Symmetric
            assert (null_s - null_s.T).nnz == 0
            # 3. No self loops
            assert np.all(null_s.diagonal() == 0)
            # 4. Weight 1
            assert np.all(null_s.data == 1.0)


def test_er_null_reproducibility():
    """Verify deterministic randomization with identical random seeds."""
    contacts, labels = load_contacts("hospital")
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    days = split_by_day(contacts, "hospital")
    graph = build_temporal_graph(days[0], n=n, labels=labels, t_res=20, name="hospital")

    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    null1 = er_null_graph(graph, rng=rng1)
    null2 = er_null_graph(graph, rng=rng2)

    for s1, s2 in zip(null1.snapshots, null2.snapshots):
        assert (s1 - s2).nnz == 0

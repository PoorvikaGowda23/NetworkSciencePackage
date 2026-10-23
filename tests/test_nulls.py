"""
Tests for nulls.py (Erdos-Renyi null model)
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import pytest

from memory_trp.aggregation import TemporalGraph, build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.nulls import er_null_graph, sbm_null_graph


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


def test_sbm_null_properties():
    """Verify SBM null preserves edge count, symmetry, no self-loops, and weights."""
    contacts, labels = load_contacts("hospital")
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    days = split_by_day(contacts, "hospital")
    graph = build_temporal_graph(days[0], n=n, labels=labels, t_res=20, name="hospital")

    rng = np.random.default_rng(42)
    sbm_null = sbm_null_graph(graph, labels=labels, rng=rng)

    assert sbm_null.n == graph.n
    assert len(sbm_null.snapshots) == len(graph.snapshots)
    assert np.array_equal(sbm_null.labels, labels)

    for orig_s, null_s in zip(graph.snapshots, sbm_null.snapshots):
        assert null_s.nnz == orig_s.nnz
        if null_s.nnz > 0:
            assert (null_s - null_s.T).nnz == 0
            assert np.all(null_s.diagonal() == 0)
            assert np.all(null_s.data == 1.0)


def test_sbm_null_homophily_preservation():
    """Verify SBM null preserves homophily compared to Erdos-Renyi."""
    contacts, labels = load_contacts("primaryschool")
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    days = split_by_day(contacts, "primaryschool")
    graph = build_temporal_graph(days[0], n=n, labels=labels, t_res=20, name="primary_d0")

    rng = np.random.default_rng(123)
    sbm_null = sbm_null_graph(graph, labels=labels, rng=rng)
    er_null = er_null_graph(graph, rng=rng)

    def within_community_fraction(g: TemporalGraph, lbl: np.ndarray) -> float:
        within = 0
        total = 0
        for s in g.snapshots:
            if s.nnz == 0:
                continue
            coo = s.tocoo()
            mask = coo.row < coo.col
            r = lbl[coo.row[mask]]
            c = lbl[coo.col[mask]]
            within += int(np.sum(r == c))
            total += int(np.sum(mask))
        return within / max(total, 1)

    emp_frac = within_community_fraction(graph, labels)
    sbm_frac = within_community_fraction(sbm_null, labels)
    er_frac = within_community_fraction(er_null, labels)

    # In primary school (highly assortative by class), SBM null within-community
    # fraction must closely track empirical and be much larger than ER null.
    assert sbm_frac > er_frac * 1.5, f"SBM frac ({sbm_frac}) should greatly exceed ER frac ({er_frac})"
    assert abs(sbm_frac - emp_frac) < 0.15, f"SBM frac ({sbm_frac}) should be close to emp ({emp_frac})"


def test_sbm_null_reproducibility():
    """Verify deterministic SBM null generation with identical seeds."""
    contacts, labels = load_contacts("hospital")
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    days = split_by_day(contacts, "hospital")
    graph = build_temporal_graph(days[0], n=n, labels=labels, t_res=20, name="hospital")

    rng1 = np.random.default_rng(777)
    rng2 = np.random.default_rng(777)

    null1 = sbm_null_graph(graph, labels=labels, rng=rng1)
    null2 = sbm_null_graph(graph, labels=labels, rng=rng2)

    for s1, s2 in zip(null1.snapshots, null2.snapshots):
        assert (s1 - s2).nnz == 0

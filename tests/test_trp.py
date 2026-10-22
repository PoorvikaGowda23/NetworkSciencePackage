"""
Tests for trp.py
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from memory_trp.aggregation import TemporalGraph, build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.trp import (
    build_neighbor_index,
    end_of_interaction,
    generate_many_trps,
    generate_trp,
)


def test_end_of_interaction_synthetic():
    """Verify end_of_interaction on consecutive and interrupted contacts."""
    n = 4
    # Snapshots:
    # 0: (0, 1) active
    # 1: (0, 1) active
    # 2: (0, 1) active
    # 3: (0, 2) active ((0, 1) inactive)
    # 4: (0, 1) active again
    snaps = []
    for edges in [[(0, 1)], [(0, 1)], [(0, 1)], [(0, 2)], [(0, 1)]]:
        row, col = [], []
        for u, v in edges:
            row.extend([u, v])
            col.extend([v, u])
        data = np.ones(len(row), dtype=float)
        snaps.append(sp.csr_matrix((data, (row, col)), shape=(n, n)))

    graph = TemporalGraph(name="test", n=n, t_res=20, snapshots=snaps, labels=None, start_time=0)

    # Starting at t=0, the consecutive run of (0, 1) ends at t=2
    assert end_of_interaction(graph, 0, 1, 0) == 2
    assert end_of_interaction(graph, 1, 0, 0) == 2
    # Starting at t=1, ends at t=2
    assert end_of_interaction(graph, 0, 1, 1) == 2
    # Starting at t=2, ends at t=2
    assert end_of_interaction(graph, 0, 1, 2) == 2
    # Starting at t=4, ends at t=4
    assert end_of_interaction(graph, 0, 1, 4) == 4
    # (0, 2) at t=3 ends at t=3
    assert end_of_interaction(graph, 0, 2, 3) == 3


def test_generate_trp_non_backtracking_and_validity():
    """Verify non-backtracking properties and reproducibility on empirical data."""
    contacts, labels = load_contacts("hospital")
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    days = split_by_day(contacts, "hospital")
    graph = build_temporal_graph(days[0], n=n, labels=labels, t_res=20, name="hospital")

    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)

    path1 = generate_trp(graph, length=15, rng=rng1)
    path2 = generate_trp(graph, length=15, rng=rng2)

    assert path1 is not None
    assert path2 is not None

    nodes1, times1 = path1
    nodes2, times2 = path2

    # Reproducibility check
    assert nodes1 == nodes2
    assert times1 == times2
    assert len(nodes1) == 15
    assert len(times1) == 15

    # Check non-backtracking conditions
    for i in range(len(nodes1) - 1):
        assert nodes1[i] != nodes1[i + 1], f"Self repeat v_x == v_{{x+1}} at step {i}"
    for i in range(len(nodes1) - 2):
        assert nodes1[i] != nodes1[i + 2], f"Backtracking v_{{x-1}} == v_{{x+1}} at step {i}"

    # Check chronological ordering
    for i in range(len(times1) - 1):
        assert times1[i] <= times1[i + 1], f"Non-chronological times at step {i}"


def test_generate_many_trps():
    """Verify generate_many_trps collects required number of valid paths."""
    contacts, labels = load_contacts("hs2011")
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    days = split_by_day(contacts, "hs2011")
    graph = build_temporal_graph(days[0], n=n, labels=labels, t_res=20, name="hs2011_d0")

    rng = np.random.default_rng(42)
    num_paths = 50
    path_len = 20

    paths = generate_many_trps(graph, num_paths=num_paths, length=path_len, rng=rng)
    assert len(paths) == num_paths

    for nodes, times in paths:
        assert len(nodes) == path_len
        assert len(times) == path_len
        # Strict non-backtracking
        for i in range(len(nodes) - 1):
            assert nodes[i] != nodes[i + 1]
        for i in range(len(nodes) - 2):
            assert nodes[i] != nodes[i + 2]

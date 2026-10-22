"""
Tests for aggregation.py
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from memory_trp.aggregation import (
    TemporalGraph,
    build_temporal_graph,
    snapshot_densities,
    split_by_day,
)
from memory_trp.loading import load_contacts


def test_split_by_day_hospital():
    """Hospital dataset must NOT be split: return as a single table."""
    contacts, labels = load_contacts("hospital")
    days = split_by_day(contacts, "hospital")
    assert len(days) == 1
    assert len(days[0]) == len(contacts)


def test_split_by_day_all_datasets():
    """Verify daily partitioning across datasets matches observation days."""
    expected_days = {
        "primaryschool": 2,
        "hs2011": 4,
        "hs2012": 7,
        "hs2013": 5,
        "conference": 2,
        "workplace": 10,
        "hospital": 1,
        "malawi": 14,
    }
    for name, exp_n in expected_days.items():
        contacts, _ = load_contacts(name)
        days = split_by_day(contacts, name)
        assert len(days) == exp_n, f"{name}: expected {exp_n} days, got {len(days)}"
        # Verify all contacts accounted for
        total_len = sum(len(d) for d in days)
        assert total_len == len(contacts), f"{name}: contact count mismatch"


def test_build_temporal_graph_synthetic():
    """Test graph construction on a controlled synthetic contact sequence."""
    n = 5
    # Window duration = 60s
    # t=0: (0, 1), (1, 2) -> window 0
    # t=20: (0, 1) -> window 0
    # t=40: (0, 1) -> window 0
    # t=120: (2, 3) -> window 2 (window 1 at [60, 120) is empty)
    # self-loop at t=20: (4, 4) should be ignored
    data = [
        {"t": 0, "i": 0, "j": 1},
        {"t": 0, "i": 1, "j": 2},
        {"t": 20, "i": 0, "j": 1},
        {"t": 20, "i": 4, "j": 4},  # self loop
        {"t": 40, "i": 1, "j": 0},  # reverse direction
        {"t": 120, "i": 2, "j": 3},
    ]
    df = pd.DataFrame(data)
    labels = np.array([0, 0, 1, 1, 2])

    t_res = 60
    graph = build_temporal_graph(df, n=n, labels=labels, t_res=t_res, name="toy")

    assert isinstance(graph, TemporalGraph)
    assert graph.name == "toy"
    assert graph.n == n
    assert graph.t_res == t_res
    assert graph.start_time == 0

    # Windows should span t=0 to t=120: (120 - 0) // 60 + 1 = 3 windows
    assert len(graph.snapshots) == 3

    # Window 0: edges (0, 1) with weight 3 (at t=0, 20, 40), and (1, 2) with weight 1 (at t=0)
    s0 = graph.snapshots[0]
    assert isinstance(s0, sp.csr_matrix)
    assert s0[0, 1] == 3.0
    assert s0[1, 0] == 3.0
    assert s0[1, 2] == 1.0
    assert s0[2, 1] == 1.0
    assert s0[4, 4] == 0.0  # no self loops
    # Check symmetry
    assert (s0 - s0.T).nnz == 0

    # Window 1 is empty: must exist and have 0 edges
    s1 = graph.snapshots[1]
    assert s1.nnz == 0
    assert s1.shape == (n, n)

    # Window 2: edge (2, 3) with weight 1
    s2 = graph.snapshots[2]
    assert s2[2, 3] == 1.0
    assert s2[3, 2] == 1.0
    assert (s2 - s2.T).nnz == 0

    # Test snapshot densities
    densities = snapshot_densities(graph)
    assert len(densities) == 3
    assert densities[0] == 2  # (0,1) and (1,2)
    assert densities[1] == 0  # empty
    assert densities[2] == 1  # (2,3)


def test_build_temporal_graph_real_data():
    """Test build_temporal_graph and snapshot_densities on real hospital data."""
    contacts, labels = load_contacts("hospital")
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    days = split_by_day(contacts, "hospital")
    day0 = days[0]

    graph = build_temporal_graph(day0, n=n, labels=labels, t_res=20, name="hospital")
    assert graph.n == n
    assert len(graph.snapshots) > 0
    assert graph.start_time == day0["t"].min()

    # Verify every snapshot is symmetric and has no self-loops
    for s in graph.snapshots:
        if s.nnz > 0:
            assert (s - s.T).nnz == 0
            assert np.all(s.diagonal() == 0)

    # Densities
    densities = snapshot_densities(graph)
    assert len(densities) == len(graph.snapshots)
    assert densities.sum() > 0
    assert np.all(densities >= 0)

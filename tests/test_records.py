"""
Tests for records.py
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import pytest

from memory_trp.records import Records, build_records, build_records_time


def test_handmade_example():
    """Hand-made test requested by specification:

    Path nodes: [0, 1, 2, 3, 4, 5] (1-indexed positions 1..6), m = 4.
    At position a = 5 (node 4), the memory set must be {1, 2}.
    """
    nodes = [0, 1, 2, 3, 4, 5]
    times = [10, 10, 20, 30, 40, 50]
    paths = [(nodes, times)]
    m = 4

    records = build_records(paths, m=m)

    assert isinstance(records, Records)
    # Positions:
    # a=2: memory set empty -> skipped
    # a=3 (node 2): memory set is {0}
    # a=4 (node 3): memory set is {0, 1}
    # a=5 (node 4): memory set is {1, 2}
    assert len(records) == 3

    # Check position a=5 (which corresponds to record index 2, where cur=4)
    # Find record where cur == 4
    idx_a5 = np.where(records.cur == 4)[0]
    assert len(idx_a5) == 1
    idx = idx_a5[0]

    assert records.prev[idx] == 3
    assert records.nxt[idx] == 5
    assert records.mem_size[idx] == 2

    # Memory set must be {1, 2}
    mem_set_a5 = set(records.mem_sets[idx][: records.mem_size[idx]])
    assert mem_set_a5 == {1, 2}
    assert records.in_memory[idx] is np.False_ or records.in_memory[idx] == False
    assert records.time_idx[idx] == 50  # time when step to node 5 happened


def test_in_memory_true():
    """Verify in_memory is True when nxt returns to a node in the memory set."""
    # At a=5 (node 4), memory set is {1, 2}. nxt is 2 -> in_memory is True
    nodes = [0, 1, 2, 3, 4, 2]
    times = [0, 1, 2, 3, 4, 5]
    paths = [(nodes, times)]
    m = 4

    records = build_records(paths, m=m)
    idx_a5 = np.where(records.cur == 4)[0][0]

    assert records.nxt[idx_a5] == 2
    mem_set_a5 = set(records.mem_sets[idx_a5][: records.mem_size[idx_a5]])
    assert mem_set_a5 == {1, 2}
    assert bool(records.in_memory[idx_a5]) is True


def test_records_padding_and_types():
    """Verify array types and -1 padding in mem_sets."""
    nodes = [10, 20, 30, 40, 50, 60, 70]
    paths = [nodes]  # test passing list of nodes directly
    m = 5

    records = build_records(paths, m=m)
    assert len(records) > 0
    assert isinstance(records.cur, np.ndarray)
    assert isinstance(records.prev, np.ndarray)
    assert isinstance(records.nxt, np.ndarray)
    assert isinstance(records.in_memory, np.ndarray)
    assert isinstance(records.mem_size, np.ndarray)
    assert isinstance(records.mem_sets, np.ndarray)
    assert isinstance(records.time_idx, np.ndarray)

    # Check padding: wherever position >= mem_size, value must be -1
    for r in range(len(records)):
        size = records.mem_size[r]
        assert np.all(records.mem_sets[r, size:] == -1)
        assert np.all(records.mem_sets[r, :size] != -1)


def test_build_records_time():
    """Verify build_records_time correctly enforces the tau look-back window in seconds."""
    # Nodes: 0 -> 1 -> 2 -> 3 -> 4 -> 2
    # Snapshot indices: 0, 3, 6, 9, 12, 15. With t_res=20s, physical times are:
    # 0s, 60s, 120s, 180s, 240s, 300s
    nodes = [0, 1, 2, 3, 4, 2]
    times = [0, 3, 6, 9, 12, 15]
    paths = [(nodes, times)]

    # With tau = 150 seconds (2.5 minutes):
    # At i=1 (cur=1, t=60s): k in {0..-1} -> empty -> skipped.
    # At i=2 (cur=2, t=120s): k=0 (t=0s, dt=120 <= 150) -> mem = {0}.
    # At i=3 (cur=3, t=180s): k=0 (dt=180 > 150, expired), k=1 (t=60s, dt=120 <= 150) -> mem = {1}.
    # At i=4 (cur=4, t=240s): k=0,1 expired. k=2 (t=120s, dt=120 <= 150) -> mem = {2}. nxt=2 -> in_memory=True.
    records = build_records_time(paths, tau_seconds=150.0, t_res=20)

    assert len(records) == 3
    # First record corresponds to cur=2
    assert records.cur[0] == 2
    assert records.prev[0] == 1
    assert records.nxt[0] == 3
    assert records.mem_size[0] == 1
    assert set(records.mem_sets[0][:1]) == {0}
    assert bool(records.in_memory[0]) is False

    # Second record corresponds to cur=3
    assert records.cur[1] == 3
    assert records.prev[1] == 2
    assert records.nxt[1] == 4
    assert records.mem_size[1] == 1
    assert set(records.mem_sets[1][:1]) == {1}
    assert bool(records.in_memory[1]) is False

    # Third record corresponds to cur=4, returning to 2
    assert records.cur[2] == 4
    assert records.prev[2] == 3
    assert records.nxt[2] == 2
    assert records.mem_size[2] == 1
    assert set(records.mem_sets[2][:1]) == {2}
    assert bool(records.in_memory[2]) is True


"""
Tests for records.py
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import pytest

from memory_trp.records import Records, build_records


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

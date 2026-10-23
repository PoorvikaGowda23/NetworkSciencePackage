"""
test_timevarying.py
===================
Unit tests for time-varying memory estimation and binning routines.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import pytest

from memory_trp.records import Records
from memory_trp.timevarying import (
    fit_p_by_time_bin,
    fit_p_fixed_C,
    record_to_clock_time,
)


def test_record_to_clock_time_relative():
    """Relative seconds should correctly map to hours [0, 24)."""
    # 36000 seconds = 10:00:00 AM (10.0 hours)
    assert np.isclose(record_to_clock_time(36000), 10.0)

    # Day 2: 86400 + 43200 = 129600 seconds = 12:00:00 PM (12.0 hours)
    assert np.isclose(record_to_clock_time(129600), 12.0)

    # Array input
    arr = np.array([0, 3600, 7200, 86400 + 3600])
    res = record_to_clock_time(arr)
    assert np.allclose(res, [0.0, 1.0, 2.0, 1.0])


def test_record_to_clock_time_epoch():
    """Epoch timestamps should correctly resolve hour with timezone."""
    # 2012-11-19 05:36:20 UTC -> Paris time (UTC+1) is 06:36:20 (6.6055... hours)
    ts = 1353303380
    h = record_to_clock_time(ts, timezone_str="Europe/Paris")
    assert 6.5 < h < 6.7


def test_records_getitem():
    """Records indexing and boolean slicing must preserve all arrays."""
    rec = Records(
        cur=np.array([1, 2, 3]),
        prev=np.array([0, 1, 2]),
        nxt=np.array([2, 3, 4]),
        in_memory=np.array([True, False, True]),
        mem_size=np.array([2, 3, 2]),
        mem_sets=np.array([[0, 1], [1, 2], [2, 3]]),
        time_idx=np.array([10, 20, 30]),
    )
    sub = rec[np.array([True, False, True])]
    assert len(sub) == 2
    assert np.array_equal(sub.cur, [1, 3])
    assert np.array_equal(sub.time_idx, [10, 30])


def test_fit_p_by_time_bin_skips_small():
    """Bins with fewer than min_records must be skipped."""
    n = 20
    # Create 150 records in bin hour 8 and 250 in bin hour 10
    time_idx = np.concatenate([np.full(150, 100), np.full(250, 400)])
    rec = Records(
        cur=np.zeros(400, dtype=int),
        prev=np.ones(400, dtype=int),
        nxt=np.full(400, 2, dtype=int),
        in_memory=np.zeros(400, dtype=bool),
        mem_size=np.full(400, 3, dtype=int),
        mem_sets=np.zeros((400, 3), dtype=int),
        time_idx=time_idx,
    )
    # start_time = 0, t_res = 60s
    # time_idx 100 -> 6000s -> 1.66h (bin 1)
    # time_idx 400 -> 24000s -> 6.66h (bin 6)
    graph_info = {"start_time": 0, "t_res": 60, "n": n}
    res = fit_p_by_time_bin(rec, graph_info, bin_minutes=60, model="MEM", min_records=200)

    # Bin 1 has 150 records (< 200), bin 6 has 250 records (>= 200)
    assert len(res) == 1
    assert 6.0 <= list(res.keys())[0] <= 7.0

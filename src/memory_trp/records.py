"""
records.py
==========
Build transition records from time-respecting paths for memory estimation.

Given sampled paths and a memory horizon m, this module constructs records
containing (current, previous, next) node transitions and the corresponding
memory set of recently visited nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Records:
    """Transition records extracted from time-respecting paths.

    Attributes
    ----------
    cur : np.ndarray
        1-D array of current node IDs (v_a).
    prev : np.ndarray
        1-D array of previous node IDs (v_{a-1}).
    nxt : np.ndarray
        1-D array of next node IDs (v_{a+1}).
    in_memory : np.ndarray
        1-D boolean array: True if nxt was present in the memory set at step a.
    mem_size : np.ndarray
        1-D array indicating the number of unique nodes in the memory set.
    mem_sets : np.ndarray
        2-D array of shape (N, max_mem_size) containing node IDs in memory set,
        padded with -1.
    time_idx : np.ndarray
        1-D array of snapshot indices at which the transition occurred.
    """

    cur: np.ndarray
    prev: np.ndarray
    nxt: np.ndarray
    in_memory: np.ndarray
    mem_size: np.ndarray
    mem_sets: np.ndarray
    time_idx: np.ndarray

    def __len__(self) -> int:
        """Return the number of transition records."""
        return len(self.cur)


def build_records(
    paths: list[tuple[list[int], list[int]] | list[int]],
    m: int,
) -> Records:
    """Build transition records from time-respecting paths for horizon m.

    For a path v_1..v_L, at 1-indexed position a, the memory set consists of
    nodes at positions a - m + 1 through a - 2 (excluding the current node v_a
    and previous node v_{a-1}).

    Assumption:
    -----------
    Early positions whose look-back memory set is empty (e.g. a <= 2, or
    where look-back window contains no nodes) are skipped. Only transitions
    with at least one node in memory are included.

    Parameters
    ----------
    paths : list[tuple[list[int], list[int]] | list[int]]
        List of paths, either as (nodes, times) tuples or node lists.
    m : int
        Memory horizon length (number of steps to look back).

    Returns
    -------
    Records
        Container with numpy arrays of transition records.
    """
    if m < 1:
        raise ValueError(f"Memory horizon m must be >= 1, got {m}.")

    cur_list: list[int] = []
    prev_list: list[int] = []
    nxt_list: list[int] = []
    in_mem_list: list[bool] = []
    mem_size_list: list[int] = []
    mem_sets_list: list[list[int]] = []
    time_idx_list: list[int] = []

    for path_item in paths:
        if isinstance(path_item, tuple):
            nodes, times = path_item
        else:
            nodes = path_item
            times = list(range(len(nodes)))

        path_len = len(nodes)
        # We need a previous node (a >= 2) and a next node (a <= path_len - 1)
        # Using 0-indexed position i where a = i + 1:
        # i goes from 1 to path_len - 2 (inclusive)
        for i in range(1, path_len - 1):
            a = i + 1  # 1-indexed position

            # 1-indexed memory window: [a - m + 1, a - 2]
            # Convert to 0-indexed slice [start_idx : end_idx_exclusive]
            # 0-indexed start = max(0, (a - m + 1) - 1) = max(0, a - m)
            # 0-indexed end = (a - 2) - 1 = a - 3 = i - 2
            # Python slice end (exclusive) = (i - 2) + 1 = i - 1
            start_idx = max(0, a - m)
            end_idx = i - 1  # exclusive

            # Assumption: Skip any record whose memory set is empty
            if end_idx <= start_idx:
                continue

            window_nodes = nodes[start_idx:end_idx]
            # Unique nodes in memory set (ordered by first occurrence)
            mem_set = list(dict.fromkeys(window_nodes))
            if len(mem_set) == 0:
                continue

            cur_node = nodes[i]
            prev_node = nodes[i - 1]
            nxt_node = nodes[i + 1]
            t_idx = times[i + 1]

            cur_list.append(cur_node)
            prev_list.append(prev_node)
            nxt_list.append(nxt_node)
            in_mem_list.append(nxt_node in mem_set)
            mem_size_list.append(len(mem_set))
            mem_sets_list.append(mem_set)
            time_idx_list.append(t_idx)

    # Number of valid records
    n_records = len(cur_list)
    if n_records == 0:
        return Records(
            cur=np.empty(0, dtype=int),
            prev=np.empty(0, dtype=int),
            nxt=np.empty(0, dtype=int),
            in_memory=np.empty(0, dtype=bool),
            mem_size=np.empty(0, dtype=int),
            mem_sets=np.empty((0, 0), dtype=int),
            time_idx=np.empty(0, dtype=int),
        )

    # Pad memory sets into a 2D array with -1
    max_mem_len = max(len(s) for s in mem_sets_list)
    mem_sets_arr = np.full((n_records, max_mem_len), fill_value=-1, dtype=int)
    for r_idx, s in enumerate(mem_sets_list):
        mem_sets_arr[r_idx, : len(s)] = s

    return Records(
        cur=np.array(cur_list, dtype=int),
        prev=np.array(prev_list, dtype=int),
        nxt=np.array(nxt_list, dtype=int),
        in_memory=np.array(in_mem_list, dtype=bool),
        mem_size=np.array(mem_size_list, dtype=int),
        mem_sets=mem_sets_arr,
        time_idx=np.array(time_idx_list, dtype=int),
    )


# Alias for flexible importing
extract_records = build_records

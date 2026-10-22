"""
trp.py
======
Non-backtracking time-respecting path (TRP) generation on temporal networks.
Implements Algorithm 1 from Guerrini et al. (EPJ Data Science 2026).

Functions
---------
end_of_interaction
    Find the last snapshot of an uninterrupted interaction between two nodes.
build_neighbor_index
    Precompute per-snapshot neighbor lists and active nodes from CSR matrices.
generate_trp
    Sample a single non-backtracking time-respecting path of specified length.
generate_many_trps
    Sample multiple valid non-backtracking time-respecting paths with retries.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config import MAX_TRP_RETRIES_PER_PATH
from memory_trp.aggregation import TemporalGraph


@dataclass
class NeighborIndex:
    """Precomputed neighbor structures for fast TRP generation.

    Attributes
    ----------
    active_snaps : np.ndarray
        Array of snapshot indices containing at least one edge.
    active_nodes_per_snap : dict[int, np.ndarray]
        Mapping from snapshot index to array of active node IDs in that snapshot.
    node_active_snaps : list[np.ndarray]
        For each node v, sorted array of snapshot indices where v has degree > 0.
    adj : list[list[tuple[np.ndarray, np.ndarray] | None] | None]
        adj[t][v] gives (neighbors_array, weights_array) for node v at snapshot t.
    """

    active_snaps: np.ndarray
    active_nodes_per_snap: dict[int, np.ndarray]
    node_active_snaps: list[np.ndarray]
    adj: list[list[tuple[np.ndarray, np.ndarray] | None] | None]


def build_neighbor_index(graph: TemporalGraph) -> NeighborIndex:
    """Precompute neighbor lists and active node indices from CSR snapshots.

    Parameters
    ----------
    graph : TemporalGraph
        Temporal graph instance with CSR snapshots.

    Returns
    -------
    NeighborIndex
        Cached neighbor and activity lookup structures.
    """
    num_snaps = len(graph.snapshots)
    n = graph.n

    active_snaps_list: list[int] = []
    active_nodes_per_snap: dict[int, np.ndarray] = {}
    node_active_snaps_acc: list[list[int]] = [[] for _ in range(n)]
    adj: list[list[tuple[np.ndarray, np.ndarray] | None] | None] = [None] * num_snaps

    for t, s in enumerate(graph.snapshots):
        if s.nnz == 0:
            continue

        active_snaps_list.append(t)
        deg = s.indptr[1:] - s.indptr[:-1]
        act_nodes = np.where(deg > 0)[0]
        active_nodes_per_snap[t] = act_nodes

        snap_adj: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n
        for v in act_nodes:
            start = s.indptr[v]
            end = s.indptr[v + 1]
            snap_adj[v] = (s.indices[start:end], s.data[start:end])
            node_active_snaps_acc[v].append(t)

        adj[t] = snap_adj

    node_active_snaps = [
        np.array(node_active_snaps_acc[v], dtype=int) for v in range(n)
    ]
    active_snaps = np.array(active_snaps_list, dtype=int)

    return NeighborIndex(
        active_snaps=active_snaps,
        active_nodes_per_snap=active_nodes_per_snap,
        node_active_snaps=node_active_snaps,
        adj=adj,
    )


def end_of_interaction(
    graph: TemporalGraph,
    u: int,
    v: int,
    t: int,
    index: NeighborIndex | None = None,
) -> int:
    """Find the last snapshot of the consecutive run where edge (u, v) is active.

    Starting from snapshot t (inclusive), checks consecutive subsequent snapshots
    t + 1, t + 2, ... and returns the index of the last snapshot in which edge
    (u, v) remains uninterruptedly active.

    Parameters
    ----------
    graph : TemporalGraph
        Temporal graph instance.
    u : int
        First endpoint of the edge.
    v : int
        Second endpoint of the edge.
    t : int
        Starting snapshot index where edge (u, v) is active.
    index : NeighborIndex | None, default=None
        Optional precomputed neighbor index for acceleration.

    Returns
    -------
    int
        Index of the last snapshot in the uninterrupted run.
    """
    curr_t = t
    num_snaps = len(graph.snapshots)

    while curr_t + 1 < num_snaps:
        next_t = curr_t + 1
        if index is not None:
            next_adj = index.adj[next_t]
            if next_adj is None or next_adj[u] is None:
                break
            nbrs_u, _ = next_adj[u]
            idx = np.searchsorted(nbrs_u, v)
            if idx < len(nbrs_u) and nbrs_u[idx] == v:
                curr_t = next_t
            else:
                break
        else:
            next_s = graph.snapshots[next_t]
            if next_s.nnz == 0:
                break
            start = next_s.indptr[u]
            end = next_s.indptr[u + 1]
            cols = next_s.indices[start:end]
            idx = np.searchsorted(cols, v)
            if idx < len(cols) and cols[idx] == v:
                curr_t = next_t
            else:
                break

    return curr_t


def generate_trp(
    graph: TemporalGraph,
    length: int,
    rng: np.random.Generator,
    index: NeighborIndex | None = None,
) -> tuple[list[int], list[int]] | None:
    """Generate a single non-backtracking time-respecting path of specified length.

    Implements Algorithm 1 of Guerrini et al. (2026):
    1. Sample t1 uniformly from snapshots with at least one active edge.
    2. Sample v1 uniformly from active nodes at t1.
    3. Sample v2 from neighbors of v1 at t1 proportional to edge weight.
    4. At step x with last two nodes (u, v), advance time to end_of_interaction(u, v).
       Next candidates are neighbors of v at that time excluding u.
       If none, advance to first later snapshot where v has a neighbor other than u.
       If no such snapshot exists, return None.
    5. Paths are strictly non-backtracking: v_{x+1} != v_x and v_{x+1} != v_{x-1}.

    Parameters
    ----------
    graph : TemporalGraph
        Temporal graph instance.
    length : int
        Target number of nodes in the path (must be >= 2).
    rng : np.random.Generator
        Explicit numpy random generator for reproducible sampling.
    index : NeighborIndex | None, default=None
        Optional precomputed neighbor index.

    Returns
    -------
    tuple[list[int], list[int]] | None
        Pair of (nodes, times) of length `length`, or None if path cannot be extended.
    """
    if length < 2:
        raise ValueError(f"Path length must be at least 2, got {length}.")

    idx = index if index is not None else build_neighbor_index(graph)
    if len(idx.active_snaps) == 0:
        return None

    # Step 1: Pick t1 uniformly at random among snapshots with >= 1 edge
    t1_pos = int(rng.integers(0, len(idx.active_snaps)))
    t1 = int(idx.active_snaps[t1_pos])

    # Step 2: Pick v1 uniformly at random among active nodes at t1
    act_nodes = idx.active_nodes_per_snap[t1]
    v1_pos = int(rng.integers(0, len(act_nodes)))
    v1 = int(act_nodes[v1_pos])

    # Step 3: Pick v2 from neighbors of v1 at t1 proportional to edge weight
    nbrs1, w1 = idx.adj[t1][v1]  # type: ignore[index]
    probs1 = w1 / w1.sum()
    v2 = int(rng.choice(nbrs1, p=probs1))

    nodes: list[int] = [v1, v2]
    times: list[int] = [t1, t1]
    curr_t = t1

    # Step 4: Extend path until target length is reached
    while len(nodes) < length:
        u = nodes[-2]
        v = nodes[-1]

        # End of interaction between u and v
        t_end = end_of_interaction(graph, u, v, curr_t, index=idx)

        # Look for candidates at t_end (neighbors of v excluding u)
        chosen_cand: np.ndarray | None = None
        chosen_weights: np.ndarray | None = None
        chosen_t: int = t_end

        snap_adj_end = idx.adj[t_end]
        if snap_adj_end is not None and snap_adj_end[v] is not None:
            nbrs_v, w_v = snap_adj_end[v]
            mask = nbrs_v != u
            if np.any(mask):
                chosen_cand = nbrs_v[mask]
                chosen_weights = w_v[mask]

        # If no candidates at t_end, search for first later snapshot
        if chosen_cand is None:
            act_times = idx.node_active_snaps[v]
            start_pos = int(np.searchsorted(act_times, t_end + 1))
            for i in range(start_pos, len(act_times)):
                t_cand = int(act_times[i])
                snap_adj_cand = idx.adj[t_cand]
                nbrs_c, w_c = snap_adj_cand[v]  # type: ignore[index]
                mask_c = nbrs_c != u
                if np.any(mask_c):
                    chosen_t = t_cand
                    chosen_cand = nbrs_c[mask_c]
                    chosen_weights = w_c[mask_c]
                    break

        # If v never interacts with a node other than u again, discard
        if chosen_cand is None or chosen_weights is None:
            return None

        # Sample next node proportional to edge weight
        probs = chosen_weights / chosen_weights.sum()
        next_v = int(rng.choice(chosen_cand, p=probs))

        nodes.append(next_v)
        times.append(chosen_t)
        curr_t = chosen_t

    return nodes, times


def generate_many_trps(
    graph: TemporalGraph,
    num_paths: int,
    length: int,
    rng: np.random.Generator,
) -> list[tuple[list[int], list[int]]]:
    """Generate multiple valid non-backtracking time-respecting paths with retries.

    Precomputes the neighbor index once and repeatedly samples paths until
    `num_paths` valid paths of length `length` are collected.

    Parameters
    ----------
    graph : TemporalGraph
        Temporal graph instance.
    num_paths : int
        Number of valid paths required.
    length : int
        Length (number of nodes) for each path.
    rng : np.random.Generator
        Explicit numpy random generator for reproducible sampling.

    Returns
    -------
    list[tuple[list[int], list[int]]]
        List of (nodes, times) tuples of length `num_paths`.

    Raises
    ------
    RuntimeError
        If maximum retry attempts exceeded without collecting enough valid paths.
    """
    if num_paths <= 0:
        return []

    index = build_neighbor_index(graph)
    if len(index.active_snaps) == 0:
        raise ValueError("Cannot sample paths from a graph with no active edges.")

    paths: list[tuple[list[int], list[int]]] = []
    max_total_attempts = num_paths * MAX_TRP_RETRIES_PER_PATH
    total_attempts = 0

    while len(paths) < num_paths:
        total_attempts += 1
        if total_attempts > max_total_attempts:
            raise RuntimeError(
                f"Failed to generate {num_paths} paths of length {length} "
                f"after {total_attempts} attempts. Only {len(paths)} valid paths formed."
            )

        result = generate_trp(graph, length, rng, index=index)
        if result is not None:
            paths.append(result)

    return paths

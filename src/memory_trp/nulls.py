"""
nulls.py
========
Null models for temporal networks.
Implements randomized temporal graphs to assess statistical significance of memory.

Functions
---------
er_null_graph
    Erdos-Renyi null model preserving the number of edges per snapshot.
sbm_null_graph
    Stochastic Block Model null preserving snapshot edge counts and community homophily.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from memory_trp.aggregation import TemporalGraph
from memory_trp.models import community_interaction_matrix


def er_null_graph(graph: TemporalGraph, rng: np.random.Generator) -> TemporalGraph:
    """Construct an Erdos-Renyi null temporal graph from an empirical graph.

    For every snapshot, preserves the exact number of active edges (nnz // 2)
    but places them uniformly at random between distinct node pairs with weight 1.
    Each snapshot is symmetric with no self-loops.

    Parameters
    ----------
    graph : TemporalGraph
        Empirical temporal graph instance.
    rng : np.random.Generator
        Explicit numpy random generator for reproducible randomization.

    Returns
    -------
    TemporalGraph
        New temporal graph with Erdos-Renyi randomized snapshots.
    """
    n = graph.n
    num_snaps = len(graph.snapshots)

    # Precompute all possible undirected node pairs (u < v)
    triu_u, triu_v = np.triu_indices(n, k=1)
    m_pairs = len(triu_u)

    null_snapshots: list[sp.csr_matrix] = []

    for s in graph.snapshots:
        num_edges = s.nnz // 2
        if num_edges == 0:
            null_snapshots.append(sp.csr_matrix((n, n), dtype=float))
        else:
            if num_edges > m_pairs:
                raise ValueError(
                    f"Snapshot edge count {num_edges} exceeds total pairs {m_pairs}."
                )

            # Sample num_edges distinct pairs uniformly without replacement
            chosen = rng.choice(m_pairs, size=num_edges, replace=False)
            u = triu_u[chosen]
            v = triu_v[chosen]

            # Construct symmetric CSR matrix with weight 1
            row_idx = np.concatenate([u, v])
            col_idx = np.concatenate([v, u])
            data = np.ones(2 * num_edges, dtype=float)

            null_s = sp.csr_matrix(
                (data, (row_idx, col_idx)), shape=(n, n), dtype=float
            )
            null_snapshots.append(null_s)

    return TemporalGraph(
        name=f"{graph.name}_er_null",
        n=n,
        t_res=graph.t_res,
        snapshots=null_snapshots,
        labels=graph.labels,
        start_time=graph.start_time,
    )


def sbm_null_graph(
    graph: TemporalGraph,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> TemporalGraph:
    """Construct a Stochastic Block Model null temporal graph from an empirical graph.

    For every snapshot, preserves the exact number of active edges (nnz // 2).
    Edges are drawn from a block model whose block-pair rates are proportional
    to the interaction counts between communities over the whole graph,
    preserving homophily while destroying temporal memory.
    Each snapshot is symmetric with no self-loops, and edges have weight 1.0.

    Parameters
    ----------
    graph : TemporalGraph
        Empirical temporal graph instance.
    labels : np.ndarray
        1-D array of integer community labels for each node (0 .. K-1).
    rng : np.random.Generator
        Explicit numpy random generator for reproducible randomization.

    Returns
    -------
    TemporalGraph
        New temporal graph with SBM randomized snapshots.
    """
    n = graph.n
    labels = np.asarray(labels, dtype=int)
    k = int(labels.max()) + 1

    # 1. Compute empirical community interaction matrix W over the whole graph
    C = community_interaction_matrix(graph, labels=labels)

    # 2. Count total possible undirected node pairs N_{a, b} between each community pair
    comm_counts = np.bincount(labels, minlength=k).astype(float)
    pair_counts = np.zeros((k, k), dtype=float)
    for a in range(k):
        pair_counts[a, a] = comm_counts[a] * (comm_counts[a] - 1.0) / 2.0
        for b in range(a + 1, k):
            pair_counts[a, b] = comm_counts[a] * comm_counts[b]
            pair_counts[b, a] = pair_counts[a, b]

    # Block-pair edge rate per node pair: w_{a, b} = C_{a, b} / N_{a, b}
    pair_rates = np.zeros((k, k), dtype=float)
    nz_mask = pair_counts > 0
    pair_rates[nz_mask] = C[nz_mask] / pair_counts[nz_mask]

    # 3. Precompute all possible undirected node pairs (u < v)
    triu_u, triu_v = np.triu_indices(n, k=1)
    m_pairs = len(triu_u)

    # Assign sampling probability to each node pair
    pair_weights = pair_rates[labels[triu_u], labels[triu_v]]
    total_weight = float(pair_weights.sum())

    if total_weight <= 0.0:
        pair_probs = np.full(m_pairs, 1.0 / m_pairs)
    else:
        pair_probs = pair_weights / total_weight

    # 4. Generate randomized snapshots
    null_snapshots: list[sp.csr_matrix] = []

    for s in graph.snapshots:
        num_edges = s.nnz // 2
        if num_edges == 0:
            null_snapshots.append(sp.csr_matrix((n, n), dtype=float))
        else:
            if num_edges > m_pairs:
                raise ValueError(
                    f"Snapshot edge count {num_edges} exceeds total pairs {m_pairs}."
                )

            # Sample num_edges distinct pairs according to pair_probs
            chosen = rng.choice(m_pairs, size=num_edges, replace=False, p=pair_probs)
            u = triu_u[chosen]
            v = triu_v[chosen]

            # Construct symmetric CSR matrix with weight 1
            row_idx = np.concatenate([u, v])
            col_idx = np.concatenate([v, u])
            data = np.ones(2 * num_edges, dtype=float)

            null_s = sp.csr_matrix(
                (data, (row_idx, col_idx)), shape=(n, n), dtype=float
            )
            null_snapshots.append(null_s)

    return TemporalGraph(
        name=f"{graph.name}_sbm_null",
        n=n,
        t_res=graph.t_res,
        snapshots=null_snapshots,
        labels=labels,
        start_time=graph.start_time,
    )


def time_shuffle_graph(
    graph: TemporalGraph,
    rng: np.random.Generator,
) -> TemporalGraph:
    """Construct a time-shuffled null temporal graph from an empirical graph.

    Randomly permutes the chronological sequence of snapshots within the day
    while keeping the internal topology of every snapshot completely untouched.

    What it destroys:
      - Temporal ordering and arrow of time
      - Continuity and burstiness of interactions across consecutive snapshots
      - Dynamic causal paths / time-respecting reachability

    What it keeps:
      - The exact static structure of each individual snapshot (degree sequence,
        clustering, community structure, number of edges, connected components)
      - The aggregate static graph (sum of all snapshots)
      - The activity timeline distribution (number of active edges per time step,
        merely reshuffled across time)

    Parameters
    ----------
    graph : TemporalGraph
        Empirical temporal graph instance.
    rng : np.random.Generator
        Explicit numpy random generator for reproducible permutation.

    Returns
    -------
    TemporalGraph
        New temporal graph with randomly permuted snapshots.
    """
    perm = rng.permutation(len(graph.snapshots))
    shuffled_snaps = [graph.snapshots[i] for i in perm]

    return TemporalGraph(
        name=f"{graph.name}_time_shuffled",
        n=graph.n,
        t_res=graph.t_res,
        snapshots=shuffled_snaps,
        labels=graph.labels,
        start_time=graph.start_time,
    )

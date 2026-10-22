"""
nulls.py
========
Null models for temporal networks.
Implements randomized temporal graphs to assess statistical significance of memory.

Functions
---------
er_null_graph
    Erdos-Renyi null model preserving the number of edges per snapshot.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from memory_trp.aggregation import TemporalGraph


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

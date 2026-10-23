"""
diffusion.py
============
Heat diffusion on temporal networks.

Implements a discrete-time heat equation on a TemporalGraph:
    u_{t+1} = (I - beta * L_t) * u_t
where L_t = D_t - A_t is the combinatorial graph Laplacian at snapshot t and
u_t is a probability vector (non-negative, sums to 1).

The update preserves the sum of u because every row of (I - beta * L_t) sums
to 1 (row sums of L_t are zero). Starting from a unit indicator at a single
seed node, u spreads across the network. After all T snapshots the final
normalized Shannon entropy measures how broadly the probability mass has spread.

Functions
---------
run_diffusion
    Run one diffusion trajectory on a TemporalGraph from a random seed node
    and return the normalized Shannon entropy of the final distribution.
"""

from __future__ import annotations

import warnings

import numpy as np

from config import DIFFUSION_CLIP_WARN
from memory_trp.aggregation import TemporalGraph


def run_diffusion(
    graph: TemporalGraph,
    beta: float,
    rng: np.random.Generator,
    clip_warn_threshold: float = DIFFUSION_CLIP_WARN,
) -> float:
    """Run one heat-diffusion trajectory and return normalized Shannon entropy.

    Starting from a unit indicator at a uniformly random seed node s,
    applies the update

        u <- (I - beta * L_t) * u   for each snapshot t in order

    where L_t = D_t - A_t is the combinatorial Laplacian.

    Because every row of (I - beta * L_t) sums to 1, the sum of u is invariant
    and u stays a probability vector throughout, as long as no entry goes
    negative.  Negative entries arise only when beta > 1/d_max (the update
    is not contracting); they are clipped to 0 with a warning so the function
    always returns a valid entropy rather than a NaN.

    Parameters
    ----------
    graph : TemporalGraph
        Temporal graph whose snapshots drive the diffusion.
    beta : float
        Diffusion rate in (0, 1/d_max].  Small values (e.g. 0.03) are safe
        for graphs with average degree ~5.
    rng : np.random.Generator
        Numpy random generator used to pick the seed node.
    clip_warn_threshold : float, default=DIFFUSION_CLIP_WARN
        If the total probability mass clipped to 0 across all steps exceeds
        this fraction of the initial mass (1.0), a UserWarning is emitted.

    Returns
    -------
    float
        Normalized Shannon entropy H in [0, 1].
        H = 0   means all probability is on one node (no spreading).
        H = 1   means probability is spread uniformly across all nodes.
    """
    n = graph.n
    if n <= 1:
        return 0.0

    # Seed: start from a random node
    seed = int(rng.integers(0, n))
    u = np.zeros(n, dtype=float)
    u[seed] = 1.0

    total_clipped = 0.0

    for snap in graph.snapshots:
        if snap.nnz == 0:
            # Empty snapshot: Laplacian is zero, update is identity
            continue

        # Dense degree vector from the CSR adjacency matrix
        # A_t is symmetric so row sums = column sums = degree
        deg = np.asarray(snap.sum(axis=1), dtype=float).ravel()  # shape (n,)

        # u <- (I - beta * L_t) u = u - beta * (D_t - A_t) u
        #                         = u - beta * deg * u + beta * A_t * u
        # A_t * u is a sparse matrix-vector product: A_t @ u
        A_u = snap @ u                     # CSR mat-vec, shape (n,)
        u = u - beta * (deg * u - A_u)    # element-wise: u_i - beta*(d_i*u_i - (Au)_i)

        # Clip tiny negatives caused by beta > 1/d_max instability
        neg_mask = u < 0.0
        total_clipped += float(np.sum(-u[neg_mask]))
        u[neg_mask] = 0.0

        # Re-normalize to keep u a proper probability vector after clipping
        s = u.sum()
        if s > 0.0:
            u /= s

    # Warn if clipping was large (sign of numerical instability)
    if total_clipped > clip_warn_threshold:
        warnings.warn(
            f"run_diffusion: total clipped probability mass = {total_clipped:.4f} "
            f"(threshold = {clip_warn_threshold}). "
            f"Consider reducing beta (currently {beta:.4f}) to beta < 1/d_max.",
            UserWarning,
            stacklevel=2,
        )

    # Normalized Shannon entropy: H = -sum(u_i * ln u_i) / ln(n)
    return _normalized_entropy(u, n)


def _normalized_entropy(u: np.ndarray, n: int) -> float:
    """Compute normalized Shannon entropy of a probability vector u.

    H = -sum_i u_i * ln(u_i) / ln(n)

    Entries with u_i = 0 contribute 0 (limit of x*ln(x) as x->0 is 0).

    Parameters
    ----------
    u : np.ndarray
        Non-negative probability vector (should sum to ~1).
    n : int
        Number of nodes.  Used to normalize so H in [0, 1].

    Returns
    -------
    float
        Normalized entropy in [0, 1].
    """
    if n <= 1:
        return 0.0
    pos = u > 0.0
    if not np.any(pos):
        return 0.0
    raw_entropy = -float(np.sum(u[pos] * np.log(u[pos])))
    return raw_entropy / np.log(n)

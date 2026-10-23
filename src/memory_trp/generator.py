"""
generator.py
============
Synthetic temporal network generator with memory-driven edge formation.

Generates a TemporalGraph where edges at each snapshot are drawn
proportionally to a memory matrix M computed from recent past snapshots,
following a model inspired by Guerrini et al. (EPJ Data Science 2026).

Functions
---------
_lazy_random_walk_matrix
    Compute the lazy random walk matrix L = (D + I)^{-1}(A + I) for a snapshot.
_memory_matrix
    Compute the product M = L_{t-1} * L_{t-2} * ... * L_{t-m_hat} over recent snapshots.
generate_memory_graph
    Generate a synthetic TemporalGraph with memory-driven edge formation.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import scipy.sparse as sp

from config import GEN_M_HAT, GEN_N, GEN_T, T_RES_SECONDS
from memory_trp.aggregation import TemporalGraph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lazy_random_walk_matrix(A: np.ndarray) -> np.ndarray:
    """Compute the lazy random walk matrix L = (D + I)^{-1} * (A + I).

    This is the transition matrix of a random walk that stays at the current
    node with probability 1/(degree+1) and moves to each neighbour with
    probability 1/(degree+1).  Adding the identity I to both A and D ensures
    every node has self-weight 1 even if it is isolated, so the matrix is
    always well-defined.

    Parameters
    ----------
    A : np.ndarray
        Symmetric (n, n) adjacency matrix (non-negative, zero diagonal).

    Returns
    -------
    np.ndarray
        (n, n) row-stochastic matrix L.
    """
    n = A.shape[0]
    d_plus_one = A.sum(axis=1) + 1.0
    L = A.copy()
    np.fill_diagonal(L, 1.0)
    L /= d_plus_one[:, np.newaxis]
    return L


def _memory_matrix(
    past_matrices: list[np.ndarray],
    m_hat: int,
    are_lazy_rw: bool = False,
) -> np.ndarray:
    """Compute the memory matrix M as the product of lazy random walk matrices.

    M = L_{t-1} * L_{t-2} * ... * L_{t-k}
    where k = min(m_hat, number of available past snapshots).

    Parameters
    ----------
    past_matrices : list[np.ndarray]
        List of past matrices in chronological order.
    m_hat : int
        Maximum number of past snapshots to include.
    are_lazy_rw : bool, default=False
        If True, elements of past_matrices are already L matrices;
        if False, elements are adjacency matrices that must be converted.

    Returns
    -------
    np.ndarray
        (n, n) memory matrix M (row-stochastic, all entries in [0, 1]).
    """
    window = past_matrices[-m_hat:]
    if not window:
        return np.eye(past_matrices[0].shape[0], dtype=float)

    if are_lazy_rw:
        M = window[-1]
        for L in reversed(window[:-1]):
            M = M @ L
        return M.copy()
    else:
        n = window[0].shape[0]
        M = np.eye(n, dtype=float)
        for A in reversed(window):
            L = _lazy_random_walk_matrix(A)
            M = M @ L
        return M


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_memory_graph(
    n: int = GEN_N,
    T: int = GEN_T,
    d: float = float(GEN_M_HAT),
    alpha: float = 0.5,
    m_hat: int = GEN_M_HAT,
    rng: np.random.Generator | None = None,
    t_res: int = T_RES_SECONDS,
    name: str = "synthetic",
    verbose: bool = True,
) -> TemporalGraph:
    """Generate a synthetic TemporalGraph with memory-driven edge formation.

    Snapshot 1 is an Erdos-Renyi graph with expected average degree d.
    For each later snapshot t the edge probability between i and j is:

        p_ij = d * [ (1 - alpha) / n  +  alpha / Z * (M_ij + M_ji) ]

    where:
      - M   = L_{t-1} * ... * L_{t-m_hat}  (product of lazy RW matrices
              over the last min(t-1, m_hat) snapshots)
      - L_s = (D_s + I)^{-1} (A_s + I)     (lazy random walk matrix)
      - Z   = (1/n) * sum_{i != j} M_ij    (normalization constant)

    Probabilities are clipped to [0, 1] before Bernoulli sampling.
    Only the upper triangle is sampled; the result is symmetrized.

    Parameters
    ----------
    n : int, default=GEN_N
        Number of nodes.
    T : int, default=GEN_T
        Number of snapshots to generate.
    d : float, default=GEN_M_HAT
        Target expected average degree per snapshot.
    alpha : float
        Memory mixing parameter in [0, 1].  alpha=0 gives Erdos-Renyi at
        every step; alpha=1 makes edges purely driven by past structure.
    m_hat : int, default=GEN_M_HAT
        Memory horizon: number of past snapshots used to build M.
    rng : np.random.Generator or None
        Numpy random generator.  If None, uses the default_rng with no seed.
    t_res : int, default=T_RES_SECONDS
        Aggregation window duration (stored as TemporalGraph metadata).
    name : str, default="synthetic"
        Name identifier for the returned TemporalGraph.
    verbose : bool, default=True
        Whether to print degree verification output on completion.

    Returns
    -------
    TemporalGraph
        Synthetic temporal graph with T CSR snapshots of shape (n, n).

    Warns
    -----
    UserWarning
        If the realized average degree deviates from d by more than 20 %.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}.")
    if m_hat < 1:
        raise ValueError(f"m_hat must be >= 1, got {m_hat}.")

    snapshots: list[sp.csr_matrix] = []
    past_L: list[np.ndarray] = []

    # Precompute upper triangle indices once
    triu_r, triu_c = np.triu_indices(n, k=1)

    # ----- Snapshot 0: Erdos-Renyi with average degree d -----
    p_er = d / (n - 1) if n > 1 else 0.0
    p_er = float(np.clip(p_er, 0.0, 1.0))
    A0 = _sample_upper_triangle(n, p_er, rng, triu_r, triu_c)
    snapshots.append(_dense_to_csr(A0))
    past_L.append(_lazy_random_walk_matrix(A0))

    # Accumulate realized degrees for the average degree warning check
    realized_degrees: list[float] = [_avg_degree(A0)]

    # ----- Snapshots 1 to T-1: memory-driven -----
    for t in range(1, T):
        # Build M from cached past lazy random walk matrices
        window_L = past_L[-m_hat:]
        M = window_L[-1]
        for L in reversed(window_L[:-1]):
            M = M @ L

        # Z = (1/n) * sum_{i != j} M_ij
        Z = (M.sum() - np.trace(M)) / n

        if Z < 1e-12:
            warnings.warn(
                f"Snapshot {t}: normalization constant Z={Z:.2e} is near zero. "
                "Memory matrix is near identity. Falling back to uniform probabilities.",
                UserWarning,
                stacklevel=2,
            )
            p_ij = np.full((n, n), d / (n - 1), dtype=float)
        else:
            mem_contrib = (M + M.T) / Z
            uniform_term = (1.0 - alpha) / n
            p_ij = d * (uniform_term + alpha * mem_contrib / 2.0)

        # Clip to valid probability range
        np.clip(p_ij, 0.0, 1.0, out=p_ij)
        np.fill_diagonal(p_ij, 0.0)

        # Sample only upper triangle, then symmetrize
        A_t = _sample_upper_triangle_from_matrix(p_ij, rng, triu_r, triu_c)

        snapshots.append(_dense_to_csr(A_t))
        past_L.append(_lazy_random_walk_matrix(A_t))
        realized_degrees.append(_avg_degree(A_t))

    # ----- Average degree sanity check -----
    mean_realized = float(np.mean(realized_degrees))
    if abs(mean_realized - d) / max(d, 1e-9) > 0.20:
        warnings.warn(
            f"generate_memory_graph: realized average degree {mean_realized:.3f} "
            f"differs from target d={d:.3f} by more than 20%. "
            f"Consider reducing alpha (currently {alpha:.2f}) or d (currently {d:.3f}) "
            "so that p_ij values before clipping stay mostly below 1. "
            "If this is intentional, suppress this warning.",
            UserWarning,
            stacklevel=2,
        )
    elif verbose:
        print(
            f"generate_memory_graph '{name}': "
            f"target d={d:.2f}, realized avg degree={mean_realized:.3f} "
            f"({100 * mean_realized / max(d, 1e-9):.1f}% of target)",
            flush=True,
        )

    return TemporalGraph(
        name=name,
        n=n,
        t_res=t_res,
        snapshots=snapshots,
        labels=None,
        start_time=0,
    )


# ---------------------------------------------------------------------------
# Private sampling utilities
# ---------------------------------------------------------------------------

def _sample_upper_triangle(
    n: int,
    p: float,
    rng: np.random.Generator,
    triu_r: np.ndarray | None = None,
    triu_c: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a symmetric binary adjacency matrix with uniform edge probability p."""
    if triu_r is None or triu_c is None:
        triu_r, triu_c = np.triu_indices(n, k=1)
    n_edges = len(triu_r)
    draws = (rng.random(n_edges) < p).astype(float)
    A = np.zeros((n, n), dtype=float)
    A[triu_r, triu_c] = draws
    A += A.T
    return A


def _sample_upper_triangle_from_matrix(
    p_matrix: np.ndarray,
    rng: np.random.Generator,
    triu_r: np.ndarray | None = None,
    triu_c: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a symmetric binary adjacency matrix from a probability matrix."""
    n = p_matrix.shape[0]
    if triu_r is None or triu_c is None:
        triu_r, triu_c = np.triu_indices(n, k=1)
    p_upper = p_matrix[triu_r, triu_c]
    draws = (rng.random(len(p_upper)) < p_upper).astype(float)
    A = np.zeros((n, n), dtype=float)
    A[triu_r, triu_c] = draws
    A += A.T
    return A


def _dense_to_csr(A: np.ndarray) -> sp.csr_matrix:
    """Convert a dense adjacency matrix to a CSR sparse matrix."""
    return sp.csr_matrix(A)


def _avg_degree(A: np.ndarray) -> float:
    """Compute the average degree of nodes from a symmetric adjacency matrix.

    Parameters
    ----------
    A : np.ndarray
        Symmetric (n, n) adjacency matrix with zero diagonal.

    Returns
    -------
    float
        Mean degree (sum of all edges * 2 / n).
    """
    n = A.shape[0]
    if n == 0:
        return 0.0
    return float(A.sum()) / n

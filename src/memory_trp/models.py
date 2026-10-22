"""
models.py
=========
Statistical inference of memory in time-respecting paths.
Implements the MEM model and MEM+SBM model from Guerrini et al. (EPJ Data Science 2026).

Classes
-------
FitResult
    Dataclass packaging model parameter estimates, likelihood, and fit statistics.
SBMFitResult
    Dataclass packaging MEM+SBM parameter estimates, likelihood, and fit statistics.

Functions
---------
bic
    Compute the Bayesian Information Criterion (n_params * ln(N) - 2 * loglik).
log_likelihood_mem
    Compute the log-likelihood of the MEM model for parameter p.
fit_mem_fixed_point
    Compute maximum likelihood estimate of p via the paper's fixed-point update.
fit_mem_numeric
    Compute maximum likelihood estimate of p via bounded scalar minimization.
simulate_mem_records
    Generate synthetic transition records from the MEM model for testing.
community_interaction_matrix
    Compute a matrix of total edge weights between community pairs from a TemporalGraph.
log_likelihood_sbm
    Compute the log-likelihood of the MEM+SBM model for parameters p and C.
fit_mem_sbm
    Fit the MEM+SBM model via alternating fixed-point (p) and L-BFGS-B (C) updates.
simulate_mem_sbm_records
    Generate synthetic transition records from the MEM+SBM model for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from config import FIXED_POINT_DAMPING, FIXED_POINT_MAXITER, FIXED_POINT_TOL, P_INIT, SBM_MAXITER
from memory_trp.records import Records


@dataclass
class FitResult:
    """Results from fitting a path memory model.

    Attributes
    ----------
    p : float
        Estimated memory parameter p in [0, 1].
    loglik : float
        Log-likelihood of the data at the estimated parameters.
    n_params : int
        Number of free parameters in the model (1 for MEM).
    n_records : int
        Total number of transition records fitted.
    bic : float
        Bayesian Information Criterion (BIC = k * ln(N) - 2 * loglik).
    iterations : int
        Number of iterations executed during optimization.
    converged : bool
        True if optimization converged within tolerance, False otherwise.
    """

    p: float
    loglik: float
    n_params: int
    n_records: int
    bic: float
    iterations: int
    converged: bool


def bic(loglik: float, n_params: int, n_records: int) -> float:
    """Compute the Bayesian Information Criterion (BIC) for model selection.

    Formula:
        BIC = n_params * ln(n_records) - 2 * loglik

    Parameter counts:
    - MEM has 1 parameter: the return probability p.
    - MEM+SBM has 1 + k(k+1)/2 - 1 parameters, where k is the number of
      communities (the -1 is because C is defined up to scale).
      Specifically:
        * 1 for the memory parameter p.
        * k*(k+1)/2 entries in the symmetric K x K affinity matrix C.
        * -1 degree of freedom because C is scale-invariant and normalized to mean 1.

    Parameters
    ----------
    loglik : float
        Log-likelihood of the data under the fitted model.
    n_params : int
        Number of free model parameters.
    n_records : int
        Total number of transition records (sample size).

    Returns
    -------
    float
        The calculated BIC score (lower indicates a better model trade-off).
    """
    # Parameter counts: MEM has 1; MEM+SBM has 1 + k(k+1)/2 - 1 where k is the
    # number of communities (the -1 is because C is defined up to scale).
    if n_records <= 0:
        return 0.0
    return float(n_params * np.log(n_records) - 2.0 * loglik)


def log_likelihood_mem(p: float, records: Records, n: int) -> float:
    """Compute the log-likelihood of the MEM model for parameter p.

    For each record with n nodes total:
    - If nxt is in the memory set: P = p / |M| + (1 - p) / (n - 2)
    - If nxt is not in memory:      P = (1 - p) / (n - 2)

    Parameters
    ----------
    p : float
        Memory parameter in [0, 1].
    records : Records
        Transition records container.
    n : int
        Total number of nodes in the network (n >= 3).

    Returns
    -------
    float
        Total log-likelihood across all records.
    """
    if n <= 2:
        raise ValueError(f"Total nodes n must be > 2, got {n}.")

    n_records = len(records)
    if n_records == 0:
        return 0.0

    z = float(n - 2)
    p_clamped = float(np.clip(p, 0.0, 1.0))

    in_mask = records.in_memory
    out_mask = ~in_mask

    ll = 0.0

    # Records in memory: P = p / |M| + (1 - p) / (n - 2)
    if np.any(in_mask):
        m_sizes = records.mem_size[in_mask].astype(float)
        p_in = (p_clamped / m_sizes) + ((1.0 - p_clamped) / z)
        # Numerical guard against log(0)
        p_in = np.maximum(p_in, 1e-15)
        ll += float(np.sum(np.log(p_in)))

    # Records out of memory: P = (1 - p) / (n - 2)
    n_out = int(np.sum(out_mask))
    if n_out > 0:
        p_out = (1.0 - p_clamped) / z
        p_out = max(p_out, 1e-15)
        ll += n_out * float(np.log(p_out))

    return ll


def fit_mem_fixed_point(
    records: Records,
    n: int,
    p0: float = P_INIT,
    tol: float = FIXED_POINT_TOL,
    maxiter: int = FIXED_POINT_MAXITER,
    damping: float = FIXED_POINT_DAMPING,
) -> FitResult:
    """Fit the MEM model via the fixed-point iteration from the paper.

    Update rule:
    p_new = 1 - |R_out| / sum_{r in R_in} [ 1 / ( p + 1 / ( (n - 2) / |M_r| - 1 ) ) ]

    Parameters
    ----------
    records : Records
        Transition records container.
    n : int
        Total number of nodes in the network.
    p0 : float, default=P_INIT
        Initial guess for memory parameter p.
    tol : float, default=FIXED_POINT_TOL
        Convergence tolerance for absolute difference |p_new - p|.
    maxiter : int, default=FIXED_POINT_MAXITER
        Maximum fixed-point iterations.
    damping : float, default=FIXED_POINT_DAMPING
        Step relaxation factor in (0, 1] for stable convergence.

    Returns
    -------
    FitResult
        Dataclass containing fitted p, loglik, BIC, and convergence metadata.
    """
    n_records = len(records)
    if n_records == 0:
        return FitResult(
            p=0.0,
            loglik=0.0,
            n_params=1,
            n_records=0,
            bic=0.0,
            iterations=0,
            converged=True,
        )

    z = float(n - 2)
    in_mask = records.in_memory
    out_mask = ~in_mask

    n_in = int(np.sum(in_mask))
    n_out = int(np.sum(out_mask))

    # Corner cases
    if n_in == 0:
        ll = log_likelihood_mem(0.0, records, n)
        bic_val = bic(ll, 1, n_records)
        return FitResult(
            p=0.0,
            loglik=ll,
            n_params=1,
            n_records=n_records,
            bic=bic_val,
            iterations=0,
            converged=True,
        )

    if n_out == 0:
        ll = log_likelihood_mem(1.0, records, n)
        bic_val = bic(ll, 1, n_records)
        return FitResult(
            p=1.0,
            loglik=ll,
            n_params=1,
            n_records=n_records,
            bic=bic_val,
            iterations=0,
            converged=True,
        )

    # Precompute denominators for records in R_in
    # term = 1 / ( (Z / |M_r|) - 1 )
    m_in = records.mem_size[in_mask].astype(float)
    ratio = (z / m_in) - 1.0
    # Guard against division by zero if |M_r| == Z
    ratio = np.maximum(ratio, 1e-12)
    offsets = 1.0 / ratio

    p = float(np.clip(p0, 0.0, 1.0))
    converged = False
    iteration = 0

    for iteration in range(1, maxiter + 1):
        # Denominator in sum: p + 1 / ( (Z / |M_r|) - 1 )
        denom = p + offsets
        sum_terms = float(np.sum(1.0 / denom))

        if sum_terms <= 0.0:
            p_raw = 0.0
        else:
            p_raw = 1.0 - (float(n_out) / sum_terms)

        p_raw = float(np.clip(p_raw, 0.0, 1.0))
        p_new = (1.0 - damping) * p + damping * p_raw

        if abs(p_new - p) < tol:
            p = p_new
            converged = True
            break

        p = p_new


    ll = log_likelihood_mem(p, records, n)
    bic_val = bic(ll, 1, n_records)

    return FitResult(
        p=p,
        loglik=ll,
        n_params=1,
        n_records=n_records,
        bic=bic_val,
        iterations=iteration,
        converged=converged,
    )


def fit_mem_numeric(records: Records, n: int) -> FitResult:
    """Maximize log-likelihood of the MEM model using scalar numerical optimization.

    Uses scipy.optimize.minimize_scalar with bounded method over [0, 1].

    Parameters
    ----------
    records : Records
        Transition records container.
    n : int
        Total number of nodes in the network.

    Returns
    -------
    FitResult
        Dataclass containing fitted p, loglik, BIC, and convergence metadata.
    """
    n_records = len(records)
    if n_records == 0:
        return FitResult(
            p=0.0,
            loglik=0.0,
            n_params=1,
            n_records=0,
            bic=0.0,
            iterations=0,
            converged=True,
        )

    def neg_ll(p_val: float) -> float:
        return -log_likelihood_mem(p_val, records, n)

    res = minimize_scalar(neg_ll, bounds=(0.0, 1.0), method="bounded")
    p_hat = float(np.clip(res.x, 0.0, 1.0))
    ll = float(-res.fun)
    bic_val = bic(ll, 1, n_records)

    return FitResult(
        p=p_hat,
        loglik=ll,
        n_params=1,
        n_records=n_records,
        bic=bic_val,
        iterations=int(res.nfev),
        converged=bool(res.success),
    )


def simulate_mem_records(
    n: int,
    num_records: int,
    p: float,
    rng: np.random.Generator,
    mem_size: int = 5,
) -> Records:
    """Generate synthetic transition records from the MEM model with a known p.

    For each record:
    - cur and prev are chosen uniformly from {0, ..., n-1} with cur != prev.
    - Memory set M of size mem_size is sampled from {0, ..., n-1} \\ {cur, prev}.
    - With probability p: nxt is chosen uniformly from M.
    - With probability 1 - p: nxt is chosen uniformly from {0, ..., n-1} \\ {cur, prev}.

    Parameters
    ----------
    n : int
        Total number of nodes in the network (n >= mem_size + 2).
    num_records : int
        Number of transition records to generate.
    p : float
        True memory parameter in [0, 1].
    rng : np.random.Generator
        Explicit numpy random generator for reproducible generation.
    mem_size : int, default=5
        Number of nodes in the memory set for each record.

    Returns
    -------
    Records
        Container with synthetic transition records.
    """
    if n < mem_size + 2:
        raise ValueError(
            f"Network size n ({n}) must be at least mem_size + 2 ({mem_size + 2})."
        )

    all_nodes = np.arange(n, dtype=int)

    cur_arr = np.empty(num_records, dtype=int)
    prev_arr = np.empty(num_records, dtype=int)
    nxt_arr = np.empty(num_records, dtype=int)
    in_mem_arr = np.empty(num_records, dtype=bool)
    mem_size_arr = np.full(num_records, mem_size, dtype=int)
    mem_sets_arr = np.empty((num_records, mem_size), dtype=int)
    time_idx_arr = np.arange(num_records, dtype=int)

    for r in range(num_records):
        # Pick cur and prev
        cur_node, prev_node = rng.choice(all_nodes, size=2, replace=False)

        # Candidates excluding cur and prev: n - 2 nodes
        allowed = np.delete(all_nodes, [cur_node, prev_node])

        # Pick memory set M from allowed candidates
        m_set = rng.choice(allowed, size=mem_size, replace=False)

        # Sample nxt according to MEM model
        if rng.random() < p:
            # Memory transition: choose uniformly from M
            nxt_node = int(rng.choice(m_set))
        else:
            # Random non-backtracking transition: choose uniformly from all allowed
            nxt_node = int(rng.choice(allowed))

        cur_arr[r] = cur_node
        prev_arr[r] = prev_node
        nxt_arr[r] = nxt_node
        in_mem_arr[r] = nxt_node in m_set
        mem_sets_arr[r] = m_set

    return Records(
        cur=cur_arr,
        prev=prev_arr,
        nxt=nxt_arr,
        in_memory=in_mem_arr,
        mem_size=mem_size_arr,
        mem_sets=mem_sets_arr,
        time_idx=time_idx_arr,
    )


# ---------------------------------------------------------------------------
# MEM + SBM model (Step 7)
# ---------------------------------------------------------------------------


@dataclass
class SBMFitResult:
    """Results from fitting the MEM+SBM model.

    Attributes
    ----------
    p : float
        Estimated memory parameter p in [0, 1].
    C : np.ndarray
        Estimated community affinity matrix (K x K, symmetric, mean entry = 1).
    loglik : float
        Log-likelihood of the data at the estimated parameters.
    n_params : int
        Free parameters: 1 (p) + K*(K+1)/2 - 1 (C entries minus one scale).
    n_records : int
        Total number of transition records fitted.
    bic : float
        Bayesian Information Criterion (BIC = k * ln(N) - 2 * loglik).
    iterations : int
        Number of alternating outer iterations executed.
    converged : bool
        True if the alternating updates converged within tolerance.
    """

    p: float
    C: np.ndarray
    loglik: float
    n_params: int
    n_records: int
    bic: float
    iterations: int
    converged: bool


def community_interaction_matrix(
    graph: "TemporalGraph",  # type: ignore[name-defined]
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Compute total edge-weight between every pair of communities across all snapshots.

    Aggregates edge interactions across all temporal snapshots between node
    communities:
        W_{a, b} = sum_{t} sum_{(u, v) in E_t} weight_t(u, v) * I(labels[u]=a, labels[v]=b)

    Parameters
    ----------
    graph : TemporalGraph
        Temporal graph with snapshots.
    labels : np.ndarray | None
        Community labels per node. If None, graph.labels is used.

    Returns
    -------
    np.ndarray
        K x K normalized affinity matrix C (mean entry == 1.0).

    Raises
    ------
    ValueError
        If both labels and graph.labels are None.
    """
    if labels is None:
        if graph.labels is None:
            raise ValueError("community_interaction_matrix requires graph.labels or labels to be set.")
        labels = graph.labels

    labels = np.asarray(labels, dtype=int)
    k = int(labels.max()) + 1
    C = np.zeros((k, k), dtype=float)

    for snap in graph.snapshots:
        if snap.nnz == 0:
            continue
        coo = snap.tocoo()
        mask = coo.row < coo.col
        if not np.any(mask):
            continue
        r = labels[coo.row[mask]]
        c = labels[coo.col[mask]]
        w = coo.data[mask].astype(float)
        np.add.at(C, (r, c), w)
        np.add.at(C, (c, r), w)

    C = np.maximum(C, 1e-4)
    return _normalise_C(C)


def _normalise_C(C: np.ndarray) -> np.ndarray:
    """Divide C by its mean so that mean(C) == 1.

    C is identifiable only up to an overall scale factor (because multiplying C
    by a constant leaves the MEM+SBM transition probabilities unchanged when Z
    is rescaled correspondingly).  Fixing mean(C) = 1 resolves the ambiguity.
    """
    mean_val = float(C.mean())
    if mean_val <= 0.0:
        return np.ones_like(C)
    return C / mean_val


def _build_C_from_log_vec(
    log_C_vec: np.ndarray,
    triu_idx: tuple[np.ndarray, np.ndarray],
    k: int,
) -> np.ndarray:
    """Reconstruct a symmetric K x K matrix from a flat vector of upper-triangle log entries.

    Exponentiating ensures every entry of C is strictly positive.
    """
    C = np.zeros((k, k), dtype=float)
    C[triu_idx] = np.exp(log_C_vec)
    # Mirror upper triangle to lower (subtract diagonal to avoid double-counting it)
    C = C + C.T - np.diag(np.diag(C))
    return C


def _precompute_sbm_arrays(
    records: Records,
    labels: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-record label indices and per-community node counts.

    Returns
    -------
    lbl_cur : (R,)   community of v_a for each record
    lbl_nxt : (R,)   community of v_{a+1}
    lbl_prev : (R,)  community of v_{a-1}
    comm_counts : (K,) number of nodes in each community
    in_mask : (R,) bool, True when nxt is in the memory set
    """
    lbl_cur  = labels[records.cur]
    lbl_nxt  = labels[records.nxt]
    lbl_prev = labels[records.prev]
    comm_counts = np.bincount(labels, minlength=k).astype(float)
    return lbl_cur, lbl_nxt, lbl_prev, comm_counts, records.in_memory


def _compute_Z_vectorized(
    lbl_cur: np.ndarray,
    lbl_prev: np.ndarray,
    C: np.ndarray,
    comm_counts: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute the normalisation constant Z for every record without looping over n.

    Z(v_a) = sum_{u != v_a, u != v_{a-1}} L(v_a, u)

    Rewritten using community counts:
        Z = sum_{c} count_c * C[lbl_cur, c]
            - C[lbl_cur, lbl_cur]   (remove the v_a contribution: one node)
            - C[lbl_cur, lbl_prev]  (remove the v_{a-1} contribution: one node)

    This avoids a O(n) loop per record — only O(K) work per record.

    Parameters
    ----------
    lbl_cur : (R,)  community of current node v_a.
    lbl_prev : (R,) community of previous node v_{a-1}.
    C : (K, K) affinity matrix.
    comm_counts : (K,) nodes per community.
    k : int  number of communities.

    Returns
    -------
    Z : (R,) array, clipped to >= 1e-15.
    """
    # C_row[r, c] = C[lbl_cur[r], c]  — row of C for each record's current node
    C_row = C[lbl_cur]  # shape (R, K)

    # Weighted sum over all nodes in each community
    Z = (C_row * comm_counts).sum(axis=1)  # shape (R,)

    # Subtract single-node contributions of v_a and v_{a-1}
    Z -= C[lbl_cur, lbl_cur]   # v_a's own L value
    Z -= C[lbl_cur, lbl_prev]  # v_{a-1}'s L value

    return np.maximum(Z, 1e-15)


def log_likelihood_sbm(
    p: float,
    C: np.ndarray,
    records: Records,
    labels: np.ndarray,
    n: int,
) -> float:
    """Compute the log-likelihood of the MEM+SBM model.

    Transition probability for record r at step v_a -> v_{a+1}:
      P = p / |M|  +  (1-p) * L(v_a, nxt) / Z   if nxt in memory set M
      P =              (1-p) * L(v_a, nxt) / Z   otherwise

    where L(u, v) = C[label(u), label(v)] and
          Z = sum_{u != v_a, v_{a-1}} L(v_a, u).

    Z is computed efficiently from community node counts (no loop over n).

    Parameters
    ----------
    p : float
        Memory parameter in [0, 1].
    C : np.ndarray
        K x K community affinity matrix (symmetric, positive entries).
    records : Records
        Transition records container.
    labels : np.ndarray
        Integer community label per node (0-indexed, length n).
    n : int
        Total number of nodes (used only for type consistency; not looped over).

    Returns
    -------
    float
        Total log-likelihood summed over all records.
    """
    if len(records) == 0:
        return 0.0

    p = float(np.clip(p, 0.0, 1.0))
    k = C.shape[0]

    lbl_cur, lbl_nxt, lbl_prev, comm_counts, in_mask = _precompute_sbm_arrays(
        records, labels, k
    )

    # L(v_a, nxt) for each record
    L_nxt = C[lbl_cur, lbl_nxt]  # shape (R,)

    # Normalisation Z for each record
    Z = _compute_Z_vectorized(lbl_cur, lbl_prev, C, comm_counts, k)  # shape (R,)

    # Base SBM probability: (1-p) * L_nxt / Z
    p_base = (1.0 - p) * L_nxt / Z  # shape (R,)

    # Memory bonus only for records where nxt is in the memory set
    mem_bonus = np.where(in_mask, p / records.mem_size.astype(float), 0.0)

    prob = p_base + mem_bonus
    prob = np.maximum(prob, 1e-15)  # numerical guard against log(0)

    return float(np.sum(np.log(prob)))


def fit_mem_sbm(
    records: Records,
    labels: np.ndarray,
    init_C: np.ndarray | None = None,
    p0: float = 0.5,
    tol: float = FIXED_POINT_TOL,
    maxiter: int = SBM_MAXITER,
    damping: float = FIXED_POINT_DAMPING,
) -> SBMFitResult:
    r"""Fit the MEM+SBM model by alternating between updating p and C.

    p-update (paper's Eq. 6 including L and Z):
        The log-likelihood under MEM+SBM is:
            \ell(p) = \sum_{r \in R_{in}} \log[ p/|M_r| + (1-p) L_r / Z_r ]
                    + \sum_{r \in R_{out}} \log[ (1-p) L_r / Z_r ]
        Differentiating with respect to p and setting to zero yields the
        generalized fixed-point update:
            p_new = 1 - |R_out| / \sum_{r \in R_{in}} [ 1 / (p + q_r) ]
        where q_r = 1 / ( Z_r / (|M_r| * L_r) - 1 ).
        We apply Mann damping to guarantee convergence.

    C-update (numeric L-BFGS-B on log(C)):
        NOTE: We use a numeric update for C instead of the paper's implicit Eq. 7.
        The paper's Eq. 7 gives an implicit self-consistency equation per community
        pair, which is not straightforward to solve efficiently. Maximizing the
        log-likelihood over the upper-triangle of log(C) via L-BFGS-B finds the
        exact optimum and ensures positive entries by parameterizing in log-space.
        After each step, C is normalized so mean(C) == 1.

    Parameters
    ----------
    records : Records
        Transition records container.
    labels : np.ndarray
        Integer community label per node (0-indexed, length n).
    init_C : np.ndarray or None
        Initial K x K affinity matrix. None defaults to all-ones (flat prior).
    p0 : float
        Initial memory parameter (default: 0.5).
    tol : float
        Convergence tolerance: stop when |delta log-likelihood| < tol.
    maxiter : int
        Maximum number of outer alternating iterations.
    damping : float
        Mann relaxation factor alpha for the p fixed-point sub-step.

    Returns
    -------
    SBMFitResult
        Fitted parameters, log-likelihood, BIC, and convergence metadata.
    """
    n_records = len(records)
    k = int(labels.max()) + 1

    if n_records == 0:
        c_ones = np.ones((k, k), dtype=float)
        return SBMFitResult(
            p=0.0,
            C=c_ones,
            loglik=0.0,
            n_params=1 + k * (k + 1) // 2 - 1,
            n_records=0,
            bic=0.0,
            iterations=0,
            converged=True,
        )

    # ---------- Initialise C ----------
    if hasattr(init_C, "snapshots") and hasattr(init_C, "labels"):
        init_C = community_interaction_matrix(init_C, labels)

    C = _normalise_C(
        init_C.astype(float).copy()
        if init_C is not None
        else np.ones((k, k), dtype=float)
    )
    p = float(np.clip(p0, 0.0, 1.0))

    # Precompute label arrays (fixed for the whole optimisation)
    lbl_cur, lbl_nxt, lbl_prev, comm_counts, in_mask = _precompute_sbm_arrays(
        records, labels, k
    )
    out_mask = ~in_mask
    mem_size_in = records.mem_size[in_mask].astype(float)
    n_out = float(np.sum(out_mask))
    n_in = float(np.sum(in_mask))

    prev_ll = log_likelihood_sbm(p, C, records, labels, len(labels))
    converged = False
    iteration = 0
    triu_idx = np.triu_indices(k)

    for iteration in range(1, maxiter + 1):
        # ---- 1. p update: paper's Eq. 6 including L and Z --------------------
        L_nxt = C[lbl_cur, lbl_nxt]
        Z = _compute_Z_vectorized(lbl_cur, lbl_prev, C, comm_counts, k)

        if n_in == 0:
            p = 0.0
        elif n_out == 0:
            p = 1.0
        else:
            L_in = L_nxt[in_mask]
            Z_in = Z[in_mask]
            denom = np.maximum(Z_in / (mem_size_in * L_in) - 1.0, 1e-12)
            q = 1.0 / denom

            for _ in range(10):
                sum_inv = np.sum(1.0 / (p + q))
                if sum_inv <= 0:
                    break
                p_next = float(np.clip(1.0 - n_out / sum_inv, 0.0, 1.0))
                p = float(np.clip((1.0 - damping) * p + damping * p_next, 0.0, 1.0))

        # ---- 2. C update: numeric L-BFGS-B on log(C) upper triangle ---------
        # NOTE: We use a numeric update for C via L-BFGS-B on log(C) instead
        # of the paper's implicit Eq. 7.
        if k > 1:
            log_C0 = np.log(np.maximum(C[triu_idx], 1e-15))
            mem_bonus = np.where(in_mask, p / records.mem_size.astype(float), 0.0)

            def neg_ll_C(log_C_vec: np.ndarray) -> float:
                C_trial = _normalise_C(_build_C_from_log_vec(log_C_vec, triu_idx, k))
                L_trial = C_trial[lbl_cur, lbl_nxt]
                C_row = C_trial[lbl_cur]
                Z_trial = (
                    (C_row * comm_counts).sum(axis=1)
                    - C_trial[lbl_cur, lbl_cur]
                    - C_trial[lbl_cur, lbl_prev]
                )
                Z_trial = np.maximum(Z_trial, 1e-15)
                p_base = (1.0 - p) * L_trial / Z_trial
                prob = np.maximum(p_base + mem_bonus, 1e-15)
                return -float(np.sum(np.log(prob)))

            res = minimize(
                neg_ll_C,
                log_C0,
                method="L-BFGS-B",
                options={"maxiter": 20, "ftol": 1e-7, "gtol": 1e-5},
            )
            C = _normalise_C(_build_C_from_log_vec(res.x, triu_idx, k))
        else:
            C = np.ones((1, 1), dtype=float)

        # ---- convergence check on log-likelihood change ----------------------
        curr_ll = log_likelihood_sbm(p, C, records, labels, len(labels))
        if abs(curr_ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = curr_ll

    ll = log_likelihood_sbm(p, C, records, labels, len(labels))
    # Parameter counts: MEM+SBM has 1 + k(k+1)/2 - 1 where k is the number of
    # communities (the -1 is because C is defined up to scale).
    n_params = 1 + k * (k + 1) // 2 - 1
    bic_val = bic(ll, n_params, n_records)

    return SBMFitResult(
        p=p,
        C=C,
        loglik=ll,
        n_params=n_params,
        n_records=n_records,
        bic=bic_val,
        iterations=iteration,
        converged=converged,
    )


def simulate_mem_sbm_records(
    n: int,
    labels: "np.ndarray",
    num_records: int,
    p: float,
    C: "np.ndarray",
    rng: "np.random.Generator",
    mem_size: int = 5,
) -> "Records":
    """Generate synthetic transition records from the MEM+SBM model.

    Vectorized (no Python loop over records) for speed.

    Sampling steps:
    1. (cur, prev) sampled uniformly from distinct node pairs.
    2. Memory set M: size mem_size, uniform from nodes other than cur/prev.
       A large candidate pool is drawn in one call; a tiny Python loop only
       filters out cur and prev per row (the heavy randomness is vectorized).
    3. Memory-transition (probability p): nxt uniform from M.
    4. SBM-transition (probability 1-p): nxt from Categorical(C[lbl_cur, :])
       restricted to nodes other than cur/prev. Uses the Gumbel-max trick to
       replace per-record np.choice with a single vectorized argmax.

    Parameters
    ----------
    n : int
        Total number of nodes.
    labels : np.ndarray
        Integer community label per node (0-indexed, length n).
    num_records : int
        Number of records to generate.
    p : float
        True memory parameter.
    C : np.ndarray
        K x K true community affinity matrix.
    rng : np.random.Generator
        Explicit numpy random generator for reproducibility.
    mem_size : int
        Number of nodes in the memory set per record.

    Returns
    -------
    Records
        Synthetic transition records.
    """
    if n < mem_size + 2:
        raise ValueError(f"n ({n}) must be >= mem_size + 2 ({mem_size + 2}).")

    import numpy as _np
    all_nodes = _np.arange(n, dtype=int)

    # 1. Sample (cur, prev) pairs in batches (avoid cur == prev)
    pairs_list = []
    collected = 0
    while collected < num_records:
        needed = max((num_records - collected) * 2, 128)
        batch = rng.integers(0, n, size=(needed, 2))
        valid = batch[batch[:, 0] != batch[:, 1]]
        take = min(len(valid), num_records - collected)
        pairs_list.append(valid[:take])
        collected += take
    pair_arr = _np.vstack(pairs_list)[:num_records]
    cur_arr  = pair_arr[:, 0]
    prev_arr = pair_arr[:, 1]

    # 2. Sample memory sets: draw a large pool, filter per row in a tiny loop
    oversample = (mem_size + 2) * 5
    cand_pool = rng.integers(0, n, size=(num_records, oversample))
    mem_sets_arr = _np.empty((num_records, mem_size), dtype=int)
    for r in range(num_records):
        c, pv = int(cur_arr[r]), int(prev_arr[r])
        row = cand_pool[r]
        mask = (row != c) & (row != pv)
        uniq = _np.unique(row[mask])
        if len(uniq) < mem_size:
            allowed = all_nodes[(all_nodes != c) & (all_nodes != pv)]
            uniq = rng.choice(allowed, size=mem_size, replace=False)
        mem_sets_arr[r] = uniq[:mem_size]

    # 3. Memory-transition: choose a random slot in each memory set
    mem_col = rng.integers(0, mem_size, size=num_records)
    nxt_from_mem = mem_sets_arr[_np.arange(num_records), mem_col]

    # 4. SBM-transition: Gumbel-max trick for vectorized categorical sampling
    #    W[u, v] = C[label(u), label(v)]; zero out cur and prev columns first.
    W_full = _np.maximum(C[labels[:, None], labels[None, :]], 1e-30)  # (n, n)
    W_rows = W_full[cur_arr].copy()
    W_rows[_np.arange(num_records), cur_arr]  = 0.0
    W_rows[_np.arange(num_records), prev_arr] = 0.0
    log_W  = _np.log(_np.maximum(W_rows, 1e-30))
    gumbel = -_np.log(-_np.log(rng.random(W_rows.shape)))
    nxt_from_sbm = _np.argmax(log_W + gumbel, axis=1)

    # 5. Combine memory and SBM transitions
    use_mem = rng.random(num_records) < p
    nxt_arr = _np.where(use_mem, nxt_from_mem, nxt_from_sbm)

    in_mem_arr   = _np.any(mem_sets_arr == nxt_arr[:, None], axis=1)
    mem_size_arr = _np.full(num_records, mem_size, dtype=int)
    time_idx_arr = _np.arange(num_records, dtype=int)

    return Records(
        cur=cur_arr,
        prev=prev_arr,
        nxt=nxt_arr,
        in_memory=in_mem_arr,
        mem_size=mem_size_arr,
        mem_sets=mem_sets_arr,
        time_idx=time_idx_arr,
    )

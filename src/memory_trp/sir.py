"""
sir.py
======
Susceptible-Infected-Recovered (SIR) epidemic simulation on temporal networks.

Implements a discrete-time synchronous SIR spreading process driven by the
sequence of contact snapshots in a TemporalGraph:
  1. A random seed node is infected at a random start snapshot t_0.
  2. At each snapshot t >= t_0, infected nodes transmit infection to each
     susceptible contact neighbor with probability:
         P(infection) = 1 - exp(-transmission_rate * weight)
     When a susceptible node has multiple infected neighbors, the cumulative
     transmission probability across all edges is 1 - exp(-transmission_rate * W),
     where W is the sum of contact weights from all infected neighbors.
  3. Infected nodes recover after remaining infected for recovery_snapshots
     snapshots and enter state R (immune, cannot be reinfected).
  4. The process terminates when no infected nodes remain or all snapshots
     have elapsed.

Functions
---------
simulate_sir
    Run one SIR simulation trajectory from a random seed node and start snapshot.
calibrate_rate
    Calibrate the transmission rate so the mean final outbreak fraction is
    within the sensitive 30% to 50% critical transition regime.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config import (
    RANDOM_SEED,
    SIR_CALIBRATION_RUNS,
    SIR_RECOVERY_SNAPSHOTS,
    SIR_TARGET_OUTBREAK,
)
from memory_trp.aggregation import TemporalGraph


@dataclass
class SIRResult:
    """Container for the output of an SIR epidemic simulation.

    Fields
    ------
    final_fraction : float
        Fraction of all nodes that were infected during the epidemic (in [0, 1]).
    peak_snapshot : int
        Snapshot index at which the number of concurrently infected nodes was maximal.
    infected_counts : np.ndarray
        Time series of infected node counts at each snapshot from start_snapshot to end.
    start_snapshot : int
        Snapshot index where the epidemic was seeded.
    """

    final_fraction: float
    peak_snapshot: int
    infected_counts: np.ndarray
    start_snapshot: int = 0

    @property
    def time_to_peak(self) -> int:
        """Number of snapshots between start and peak infection count."""
        return self.peak_snapshot - self.start_snapshot

    def __iter__(self):
        """Enable 3-tuple unpacking: final_fraction, peak_snapshot, infected_counts."""
        return iter((self.final_fraction, self.peak_snapshot, self.infected_counts))


def simulate_sir(
    graph: TemporalGraph,
    transmission_rate: float,
    recovery_snapshots: int = SIR_RECOVERY_SNAPSHOTS,
    rng: np.random.Generator | None = None,
    start_snapshot: int | None = None,
    seed_node: int | None = None,
) -> SIRResult:
    """Run one SIR epidemic simulation trajectory on a temporal contact network.

    Parameters
    ----------
    graph : TemporalGraph
        Temporal graph whose snapshot adjacency matrices drive the contacts.
    transmission_rate : float
        Per-contact-weight infection parameter beta > 0.
        Probability of transmission over weight w is 1 - exp(-beta * w).
    recovery_snapshots : int, default=SIR_RECOVERY_SNAPSHOTS
        Number of snapshots an infected node remains infectious before recovering.
    rng : np.random.Generator or None, default=None
        Random number generator. If None, a default generator seeded with
        RANDOM_SEED is used for reproducibility.
    start_snapshot : int or None, default=None
        Snapshot index to seed the infection. If None, chosen uniformly at
        random from [0, len(graph.snapshots) - 1].
    seed_node : int or None, default=None
        Initial infected node index. If None, chosen uniformly at random
        from [0, graph.n - 1].

    Returns
    -------
    SIRResult
        Named container with:
          - final_fraction: fraction of nodes ever infected (I or R) / n
          - peak_snapshot: snapshot index where concurrent infected count peaked
          - infected_counts: 1D numpy array of infected counts per snapshot
        Supports 3-tuple unpacking: `final_frac, peak_snap, counts = simulate_sir(...)`.
    """
    n = graph.n
    total_snaps = len(graph.snapshots)

    if n == 0 or total_snaps == 0:
        return SIRResult(
            final_fraction=0.0,
            peak_snapshot=0,
            infected_counts=np.zeros(1, dtype=int),
            start_snapshot=0,
        )

    if recovery_snapshots < 1:
        raise ValueError(
            f"recovery_snapshots must be >= 1, got {recovery_snapshots}."
        )

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # 1. Random seed node and start snapshot
    if seed_node is None:
        seed_node = int(rng.integers(0, n))
    if start_snapshot is None:
        start_snapshot = int(rng.integers(0, total_snaps))

    # Node state representation: 0 = Susceptible (S), 1 = Infected (I), 2 = Recovered (R)
    state = np.zeros(n, dtype=np.int8)
    state[seed_node] = 1

    # Remaining recovery time (in snapshots) for each infected node
    recovery_timer = np.zeros(n, dtype=int)
    recovery_timer[seed_node] = recovery_snapshots

    counts: list[int] = []
    max_infected = 1
    peak_snapshot = start_snapshot

    # 2. Synchronous discrete-time propagation across snapshots
    for t in range(start_snapshot, total_snaps):
        # Current number of infected nodes
        num_infected = int(np.sum(state == 1))
        counts.append(num_infected)

        if num_infected > max_infected:
            max_infected = num_infected
            peak_snapshot = t

        if num_infected == 0:
            # Epidemic has died out; pad remaining snapshots with zeros
            remaining = (total_snaps - start_snapshot) - len(counts)
            if remaining > 0:
                counts.extend([0] * remaining)
            break

        snap = graph.snapshots[t]

        # 2a. Transmission step
        newly_infected = np.empty(0, dtype=int)
        if snap.nnz > 0:
            inf_mask = (state == 1).astype(float)
            # Total contact weight from infected nodes to each node: snap @ inf_mask
            w_from_inf = np.asarray(snap @ inf_mask).ravel()
            susc_mask = (state == 0) & (w_from_inf > 0)

            if np.any(susc_mask):
                w_susc = w_from_inf[susc_mask]
                p_inf = 1.0 - np.exp(-transmission_rate * w_susc)
                rand_draws = rng.random(len(p_inf))
                newly_infected = np.where(susc_mask)[0][rand_draws < p_inf]

        # 2b. Recovery step for nodes already infected before this snapshot
        currently_infected = np.where(state == 1)[0]
        recovery_timer[currently_infected] -= 1
        recovering_nodes = currently_infected[recovery_timer[currently_infected] <= 0]
        state[recovering_nodes] = 2  # Recovered

        # 2c. Activate newly infected nodes
        if len(newly_infected) > 0:
            state[newly_infected] = 1
            recovery_timer[newly_infected] = recovery_snapshots

    # 3. Final outbreak calculation
    # Total nodes that ever experienced infection: currently infected or recovered
    total_ever_infected = int(np.sum(state != 0))
    final_fraction = float(total_ever_infected / n)

    return SIRResult(
        final_fraction=final_fraction,
        peak_snapshot=peak_snapshot,
        infected_counts=np.array(counts, dtype=int),
        start_snapshot=start_snapshot,
    )


def calibrate_rate(
    graph: TemporalGraph,
    target_final_fraction: float = SIR_TARGET_OUTBREAK,
    recovery_snapshots: int = SIR_RECOVERY_SNAPSHOTS,
    n_calib_runs: int = SIR_CALIBRATION_RUNS,
    rng: np.random.Generator | None = None,
) -> float:
    """Calibrate transmission rate so the mean outbreak size is in the 30-50% regime.

    Rationale for Target Outbreak (30% to 50%):
    -------------------------------------------
    In epidemic modeling on temporal graphs, choosing an appropriate transmission
    rate beta is critical:
      - If beta is too small (sub-critical), almost all outbreaks sputter out
        after 1 or 2 hops (final fraction ~ 1/n), yielding zero statistical
        power to detect differences between empirical and null topologies.
      - If beta is too large (super-critical saturation), the epidemic floods
        the entire connected component (final fraction > 80-90%), erasing any
        sensitivity to temporal ordering, contact memory, or community barriers.
      - A target attack rate of 30% to 50% (default 40%) places the epidemic
        directly in the critical percolation/transition regime where spreading
        dynamics are most sensitive to contact memory, burstiness, and temporal
        ordering.

    The calibration performs a binary search (bisection) over log10(beta) in
    [1e-4, 2.0] until the empirical mean outbreak fraction across `n_calib_runs`
    falls within [target_final_fraction - 0.05, target_final_fraction + 0.05]
    or 7 bisection steps complete.

    Parameters
    ----------
    graph : TemporalGraph
        Empirical temporal graph instance.
    target_final_fraction : float, default=SIR_TARGET_OUTBREAK (0.40)
        Target mean outbreak fraction (recommended between 0.30 and 0.50).
    recovery_snapshots : int, default=SIR_RECOVERY_SNAPSHOTS
        Number of snapshots an infected node stays infectious.
    n_calib_runs : int, default=SIR_CALIBRATION_RUNS
        Number of Monte Carlo simulations evaluated per candidate transmission rate.
    rng : np.random.Generator or None, default=None
        Random number generator.

    Returns
    -------
    float
        Calibrated transmission rate beta > 0.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Search range in log10 space: 1e-4 to 2.0
    log_low = -4.0
    log_high = 0.5
    best_rate = 10.0 ** ((log_low + log_high) / 2.0)
    best_diff = float("inf")

    # Up to 7 bisection iterations
    for _ in range(7):
        log_mid = (log_low + log_high) / 2.0
        candidate_rate = float(10.0 ** log_mid)

        # Run sample simulations
        outbreaks = [
            simulate_sir(
                graph=graph,
                transmission_rate=candidate_rate,
                recovery_snapshots=recovery_snapshots,
                rng=rng,
            ).final_fraction
            for _ in range(n_calib_runs)
        ]
        mean_outbreak = float(np.mean(outbreaks))
        diff = abs(mean_outbreak - target_final_fraction)

        if diff < best_diff:
            best_diff = diff
            best_rate = candidate_rate

        # Stop early if within +/- 0.05 of target (i.e. comfortably within 30-50%)
        if diff <= 0.05:
            break

        if mean_outbreak < target_final_fraction:
            # Transmission rate is too weak, search higher
            log_low = log_mid
        else:
            # Transmission rate is too strong, search lower
            log_high = log_mid

    return best_rate

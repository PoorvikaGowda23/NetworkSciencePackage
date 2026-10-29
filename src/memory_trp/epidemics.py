"""
epidemics.py
============
Generalised compartmental epidemic simulation (SIR, SIS, SIRS) on temporal
contact networks.

This module extends ``memory_trp.sir``. The transmission rule, the snapshot
update order, the pre-removed ("isolated") node mechanism and the random-number
consumption are identical to ``simulate_sir``, so for ``model="sir"`` the two
functions return the same trajectory when given the same generator state,
``start_snapshot`` and ``seed_node``.

Models
------
SIR   S -> I -> R                  recovered nodes are immune for good.
SIS   S -> I -> S                  recovered nodes are immediately susceptible
                                   again (no immunity, reinfection possible).
SIRS  S -> I -> R -> S             recovered nodes are immune for
                                   ``immunity_snapshots`` snapshots, then become
                                   susceptible again (waning immunity).

Transmission (all models)
-------------------------
A susceptible node with total contact weight W to infected nodes in the current
snapshot is infected with probability ``1 - exp(-transmission_rate * W)``.
An infected node stays infectious for ``recovery_snapshots`` snapshots.

Node states
-----------
0 = Susceptible, 1 = Infected, 2 = Recovered/immune, 3 = Pre-removed (isolated).

Why extra outcome metrics?
--------------------------
In SIR the "final outbreak size" (fraction ever infected) fully describes the
epidemic. In SIS and SIRS nodes can be infected several times and the disease
can persist, so the final size alone hides most of the dynamics. Each result
therefore also carries:

  * ``mean_prevalence``   time-averaged fraction infected over the simulated window
                          (the infection *burden*, proportional to person-time infected)
  * ``peak_prevalence``   largest fraction infected at one time
  * ``final_prevalence``  fraction still infected at the end of the window
  * ``persisted``         True if the infection had not died out by the end
  * ``events_per_node``   infection events per node (>= ever-infected fraction;
                          the excess is reinfection)

Functions
---------
simulate_epidemic
    Run one SIR / SIS / SIRS trajectory.
calibrate_rate_model
    Bisection search for the transmission rate giving a target mean
    ever-infected fraction under the chosen model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    RANDOM_SEED,
    SIR_CALIBRATION_RUNS,
    SIR_RECOVERY_SNAPSHOTS,
    SIR_TARGET_OUTBREAK,
    SIRS_IMMUNITY_SNAPSHOTS,
)
from memory_trp.aggregation import TemporalGraph

VALID_MODELS: tuple[str, ...] = ("sir", "sis", "sirs")
MODEL_LABELS: dict[str, str] = {"sir": "SIR", "sis": "SIS", "sirs": "SIRS"}

_S, _I, _R, _REMOVED = 0, 1, 2, 3


@dataclass
class EpidemicResult:
    """Output of one epidemic trajectory.

    Fields
    ------
    model : str
        "sir", "sis" or "sirs".
    final_fraction : float
        Fraction of all nodes infected at least once. Matches
        ``SIRResult.final_fraction`` for SIR. Isolated nodes are never counted.
    peak_snapshot : int
        Snapshot index of the maximum number of concurrently infected nodes.
    infected_counts : np.ndarray
        Number of infected nodes at every snapshot from ``start_snapshot`` to the
        end of the graph (zero-padded after the infection dies out).
    start_snapshot : int
        Snapshot where the epidemic was seeded.
    n_nodes : int
        Total number of nodes (the denominator of every fraction).
    infection_events : int
        Total infection events, including the index case and reinfections.
    final_infected : int
        Number of nodes still infected after the last simulated snapshot.
    """

    model: str
    final_fraction: float
    peak_snapshot: int
    infected_counts: np.ndarray
    start_snapshot: int = 0
    n_nodes: int = 0
    infection_events: int = 0
    final_infected: int = 0

    @property
    def time_to_peak(self) -> int:
        """Snapshots between the seed and the infection peak."""
        return self.peak_snapshot - self.start_snapshot

    @property
    def mean_prevalence(self) -> float:
        """Time-averaged fraction of nodes infected over the simulated window."""
        if self.n_nodes == 0 or len(self.infected_counts) == 0:
            return 0.0
        return float(self.infected_counts.mean() / self.n_nodes)

    @property
    def peak_prevalence(self) -> float:
        """Maximum fraction of nodes infected at any one snapshot."""
        if self.n_nodes == 0 or len(self.infected_counts) == 0:
            return 0.0
        return float(self.infected_counts.max() / self.n_nodes)

    @property
    def final_prevalence(self) -> float:
        """Fraction of nodes infected after the last snapshot."""
        return float(self.final_infected / self.n_nodes) if self.n_nodes else 0.0

    @property
    def persisted(self) -> bool:
        """True if the infection was still circulating at the end of the window."""
        return self.final_infected > 0

    @property
    def events_per_node(self) -> float:
        """Infection events per node (reinfections push this above final_fraction)."""
        return float(self.infection_events / self.n_nodes) if self.n_nodes else 0.0

    def __iter__(self):
        """Allow 3-tuple unpacking like SIRResult."""
        return iter((self.final_fraction, self.peak_snapshot, self.infected_counts))


def simulate_epidemic(
    graph: TemporalGraph,
    transmission_rate: float,
    model: str = "sir",
    recovery_snapshots: int = SIR_RECOVERY_SNAPSHOTS,
    immunity_snapshots: int = SIRS_IMMUNITY_SNAPSHOTS,
    rng: np.random.Generator | None = None,
    start_snapshot: int | None = None,
    seed_node: int | None = None,
    removed_nodes: set[int] | list[int] | np.ndarray | None = None,
) -> EpidemicResult:
    """Run one SIR, SIS or SIRS trajectory on a temporal contact network.

    Parameters
    ----------
    graph : TemporalGraph
        Temporal graph whose snapshot adjacency matrices drive the contacts.
    transmission_rate : float
        Per-contact-weight infection parameter beta > 0.
    model : {"sir", "sis", "sirs"}
        Compartmental model.
    recovery_snapshots : int
        Number of snapshots a node stays infectious.
    immunity_snapshots : int
        SIRS only: number of snapshots a recovered node stays immune before
        returning to the susceptible state. Ignored for SIR and SIS.
    rng : np.random.Generator or None
        Random generator (default: seeded with RANDOM_SEED).
    start_snapshot : int or None
        Snapshot of the seeding. Uniform at random if None.
    seed_node : int or None
        Index case. Uniform among non-isolated nodes if None.
    removed_nodes : collection of int or None
        Nodes isolated from the start. They can neither catch nor spread the
        infection and are never counted as infected.

    Returns
    -------
    EpidemicResult
    """
    model = model.lower()
    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}, got {model!r}.")
    if recovery_snapshots < 1:
        raise ValueError(f"recovery_snapshots must be >= 1, got {recovery_snapshots}.")
    if model == "sirs" and immunity_snapshots < 1:
        raise ValueError(f"immunity_snapshots must be >= 1, got {immunity_snapshots}.")

    n = graph.n
    total_snaps = len(graph.snapshots)

    if n == 0 or total_snaps == 0:
        return EpidemicResult(model, 0.0, 0, np.zeros(1, dtype=int), 0, n, 0, 0)

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # 1. Isolated nodes
    removed_set = set(removed_nodes) if removed_nodes is not None else set()
    available_nodes = [i for i in range(n) if i not in removed_set]

    if not available_nodes:
        s0 = start_snapshot if start_snapshot is not None else 0
        return EpidemicResult(model, 0.0, s0, np.zeros(1, dtype=int), s0, n, 0, 0)

    # 2. Index case and seeding time (same RNG call order as simulate_sir)
    if seed_node is None:
        seed_node = int(rng.choice(available_nodes))
    elif seed_node in removed_set:
        raise ValueError(f"Explicit seed_node {seed_node} cannot be in removed_nodes.")

    if start_snapshot is None:
        start_snapshot = int(rng.integers(0, total_snaps))

    state = np.zeros(n, dtype=np.int8)
    if removed_set:
        state[list(removed_set)] = _REMOVED
    state[seed_node] = _I

    infect_timer = np.zeros(n, dtype=int)   # snapshots of infectiousness left
    immune_timer = np.zeros(n, dtype=int)   # snapshots of immunity left (SIRS)
    infect_timer[seed_node] = recovery_snapshots

    ever_infected = np.zeros(n, dtype=bool)
    ever_infected[seed_node] = True
    infection_events = 1

    counts: list[int] = []
    max_infected = 1
    peak_snapshot = start_snapshot

    # 3. Synchronous discrete-time propagation
    for t in range(start_snapshot, total_snaps):
        num_infected = int(np.count_nonzero(state == _I))
        counts.append(num_infected)

        if num_infected > max_infected:
            max_infected = num_infected
            peak_snapshot = t

        if num_infected == 0:
            # Absorbing state: the infection has died out
            remaining = (total_snaps - start_snapshot) - len(counts)
            if remaining > 0:
                counts.extend([0] * remaining)
            break

        snap = graph.snapshots[t]

        # 3a. Transmission (uses states at the start of the snapshot)
        newly_infected = np.empty(0, dtype=int)
        if snap.nnz > 0:
            inf_mask = (state == _I).astype(float)
            w_from_inf = np.asarray(snap @ inf_mask).ravel()
            susc_mask = (state == _S) & (w_from_inf > 0)

            if np.any(susc_mask):
                w_susc = w_from_inf[susc_mask]
                p_inf = 1.0 - np.exp(-transmission_rate * w_susc)
                rand_draws = rng.random(len(p_inf))
                newly_infected = np.where(susc_mask)[0][rand_draws < p_inf]

        # 3b. SIRS: waning immunity of nodes that were already recovered
        if model == "sirs":
            immune = np.flatnonzero(state == _R)
            if immune.size:
                immune_timer[immune] -= 1
                state[immune[immune_timer[immune] <= 0]] = _S

        # 3c. Recovery of nodes that were infected before this snapshot
        currently_infected = np.flatnonzero(state == _I)
        infect_timer[currently_infected] -= 1
        recovering = currently_infected[infect_timer[currently_infected] <= 0]
        if recovering.size:
            if model == "sis":
                state[recovering] = _S
            else:
                state[recovering] = _R
                if model == "sirs":
                    immune_timer[recovering] = immunity_snapshots

        # 3d. Activate new infections
        if len(newly_infected) > 0:
            state[newly_infected] = _I
            infect_timer[newly_infected] = recovery_snapshots
            ever_infected[newly_infected] = True
            infection_events += int(len(newly_infected))

    final_infected = int(np.count_nonzero(state == _I))
    return EpidemicResult(
        model=model,
        final_fraction=float(ever_infected.sum() / n),
        peak_snapshot=peak_snapshot,
        infected_counts=np.array(counts, dtype=int),
        start_snapshot=start_snapshot,
        n_nodes=n,
        infection_events=infection_events,
        final_infected=final_infected,
    )


def calibrate_rate_model(
    graph: TemporalGraph,
    model: str = "sir",
    target_final_fraction: float = SIR_TARGET_OUTBREAK,
    recovery_snapshots: int = SIR_RECOVERY_SNAPSHOTS,
    immunity_snapshots: int = SIRS_IMMUNITY_SNAPSHOTS,
    n_calib_runs: int = SIR_CALIBRATION_RUNS,
    rng: np.random.Generator | None = None,
    n_steps: int = 7,
    log_low: float = -4.0,
    log_high: float = 0.5,
    tol: float = 0.05,
) -> float:
    """Calibrate beta so the mean ever-infected fraction hits ``target_final_fraction``.

    Same bisection over log10(beta) as ``memory_trp.sir.calibrate_rate`` but for
    any model. Calibrating each model to the same target ever-infected fraction
    gives every model the same *baseline attack rate*, so intervention effects
    are comparable across models. (Use one shared beta instead if you want to
    compare the models at identical transmissibility.)

    Parameters
    ----------
    n_steps : int
        Maximum bisection steps.
    log_low, log_high : float
        Search interval for log10(beta).
    tol : float
        Stop early when the mean outbreak is within ``tol`` of the target.

    Returns
    -------
    float
        Calibrated transmission rate (the best candidate seen).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    best_rate = 10.0 ** ((log_low + log_high) / 2.0)
    best_diff = float("inf")

    for _ in range(n_steps):
        log_mid = (log_low + log_high) / 2.0
        candidate = float(10.0 ** log_mid)

        outbreaks = [
            simulate_epidemic(
                graph=graph,
                transmission_rate=candidate,
                model=model,
                recovery_snapshots=recovery_snapshots,
                immunity_snapshots=immunity_snapshots,
                rng=rng,
            ).final_fraction
            for _ in range(n_calib_runs)
        ]
        mean_outbreak = float(np.mean(outbreaks))
        diff = abs(mean_outbreak - target_final_fraction)

        if diff < best_diff:
            best_diff, best_rate = diff, candidate
        if diff <= tol:
            break

        if mean_outbreak < target_final_fraction:
            log_low = log_mid
        else:
            log_high = log_mid

    return best_rate

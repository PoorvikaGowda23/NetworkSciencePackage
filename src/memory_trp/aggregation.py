"""
aggregation.py
==============
Temporal network aggregation and daily partitioning for time-respecting paths.

Classes
-------
TemporalGraph
    Dataclass holding a temporal network as a sequence of sparse CSR snapshots.

Functions
---------
split_by_day
    Partition raw contact sequence into per-day observation tables.
build_temporal_graph
    Aggregate 20-second contact records into discrete temporal snapshots.
snapshot_densities
    Compute the number of edges present in each snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import DATA_DIR, DAY_SPLIT_GAP_SECONDS, T_RES_SECONDS


@dataclass
class TemporalGraph:
    """Represents a temporal graph aggregated into discrete time snapshots.

    Attributes
    ----------
    name : str
        Name or identifier for the temporal graph.
    n : int
        Number of nodes in the graph (node IDs are indexed 0 .. n-1).
    t_res : int
        Aggregation window duration in seconds.
    snapshots : list[sp.csr_matrix]
        Sequence of symmetric CSR adjacency matrices with no self-loops.
    labels : np.ndarray | None
        1-D array of node community labels (length n), or None if unavailable.
    start_time : int
        Real clock timestamp (seconds) of the start of the first snapshot.
    """

    name: str
    n: int
    t_res: int
    snapshots: list[sp.csr_matrix]
    labels: np.ndarray | None
    start_time: int


def split_by_day(contacts: pd.DataFrame, dataset_name: str) -> list[pd.DataFrame]:
    """Split a contact sequence into distinct daily observation tables.

    Parameters
    ----------
    contacts : pd.DataFrame
        DataFrame with columns 't', 'i', 'j' sorted chronologically.
    dataset_name : str
        Name of the dataset. 'hospital' is treated as a continuous single table.
        'malawi' uses the explicit daily records from data/malawi_tdij.csv.
        Other datasets are split at overnight gaps exceeding DAY_SPLIT_GAP_SECONDS.

    Returns
    -------
    list[pd.DataFrame]
        List of per-day contact DataFrames with reset indices.
    """
    if len(contacts) == 0:
        return []

    dname = dataset_name.lower().strip()

    # Hospital has continuous round-the-clock contacts: treat as a single table
    if dname == "hospital":
        return [contacts.copy().reset_index(drop=True)]

    # Malawi records explicit observation days in the raw data
    if dname == "malawi":
        malawi_path = DATA_DIR / "malawi_tdij.csv"
        if malawi_path.exists():
            raw_meta = pd.read_csv(malawi_path, index_col=0, usecols=[0, 2])
            if len(raw_meta) == len(contacts):
                days = raw_meta["day"].values
                # Preserve chronological order of unique days
                unique_days: list[int] = []
                for d in days:
                    if not unique_days or unique_days[-1] != d:
                        unique_days.append(d)
                day_dfs: list[pd.DataFrame] = []
                for d in unique_days:
                    mask = days == d
                    day_dfs.append(contacts[mask].copy().reset_index(drop=True))
                return day_dfs

    # General case: split at overnight inactivity gaps > DAY_SPLIT_GAP_SECONDS
    diffs = contacts["t"].diff()
    split_indices = np.where(diffs > DAY_SPLIT_GAP_SECONDS)[0]

    day_dfs: list[pd.DataFrame] = []
    prev_idx = 0
    for split_idx in split_indices:
        day_dfs.append(contacts.iloc[prev_idx:split_idx].copy().reset_index(drop=True))
        prev_idx = int(split_idx)
    day_dfs.append(contacts.iloc[prev_idx:].copy().reset_index(drop=True))

    return day_dfs


def build_temporal_graph(
    contacts_day: pd.DataFrame,
    n: int,
    labels: np.ndarray | None,
    t_res: int = T_RES_SECONDS,
    name: str = "",
) -> TemporalGraph:
    """Group raw contacts into time windows of t_res seconds and build a TemporalGraph.

    The weight of edge (i, j) in a window equals the number of raw 20-second
    contact records between nodes i and j occurring inside that window.
    Each snapshot is symmetric with no self-loops.
    Every window between the first and last contact of the day exists,
    even if empty (represented by an empty CSR matrix).

    Parameters
    ----------
    contacts_day : pd.DataFrame
        Contact records for a single day with columns 't', 'i', 'j'.
    n : int
        Number of nodes in the graph (node IDs must range from 0 to n-1).
    labels : np.ndarray | None
        Community labels for each node (length n), or None.
    t_res : int, default=T_RES_SECONDS
        Duration of each aggregation window in seconds.
    name : str, default=""
        Optional identifier name for the graph.

    Returns
    -------
    TemporalGraph
        Aggregated graph structure with CSR snapshots.
    """
    if len(contacts_day) == 0:
        return TemporalGraph(
            name=name,
            n=n,
            t_res=t_res,
            snapshots=[],
            labels=labels,
            start_time=0,
        )

    # Discard self-loops (i == j)
    valid = contacts_day[contacts_day["i"] != contacts_day["j"]].copy()

    t_min = int(contacts_day["t"].min())
    t_max = int(contacts_day["t"].max())
    start_time = t_min

    # Total contiguous windows from first to last contact
    num_windows = (t_max - t_min) // t_res + 1

    # Initialize all snapshots (including empty windows) as empty CSR matrices
    snapshots: list[sp.csr_matrix] = [
        sp.csr_matrix((n, n), dtype=float) for _ in range(num_windows)
    ]

    if len(valid) > 0:
        # Canonicalize undirected edges so u < v
        u_arr = np.minimum(valid["i"].values, valid["j"].values)
        v_arr = np.maximum(valid["i"].values, valid["j"].values)
        t_arr = valid["t"].values

        # Remove duplicate sensor records occurring at the exact same timestamp
        df_edges = pd.DataFrame({"t": t_arr, "u": u_arr, "v": v_arr})
        df_edges = df_edges.drop_duplicates(subset=["t", "u", "v"])

        # Map each contact event to its window index
        w_arr = (df_edges["t"].values - t_min) // t_res
        df_edges["w"] = w_arr

        # Group by window and edge to count raw 20-second occurrences (edge weight)
        grouped = df_edges.groupby(["w", "u", "v"]).size().reset_index(name="weight")

        # Construct symmetric CSR matrix for each active window
        for w, group in grouped.groupby("w"):
            w_idx = int(w)
            u = group["u"].values
            v = group["v"].values
            weights = group["weight"].values.astype(float)

            # Symmetrize: (u, v) and (v, u)
            row_idx = np.concatenate([u, v])
            col_idx = np.concatenate([v, u])
            data = np.concatenate([weights, weights])

            snapshots[w_idx] = sp.csr_matrix(
                (data, (row_idx, col_idx)), shape=(n, n), dtype=float
            )

    return TemporalGraph(
        name=name,
        n=n,
        t_res=t_res,
        snapshots=snapshots,
        labels=labels,
        start_time=start_time,
    )


def snapshot_densities(graph: TemporalGraph) -> np.ndarray:
    """Compute the number of edges present in each snapshot of a temporal graph.

    Since snapshots are symmetric CSR matrices without self-loops,
    the number of undirected edges equals nnz // 2.

    Parameters
    ----------
    graph : TemporalGraph
        Temporal graph instance.

    Returns
    -------
    np.ndarray
        1-D array of integers giving the edge count for each snapshot.
    """
    return np.array([s.nnz // 2 for s in graph.snapshots], dtype=int)

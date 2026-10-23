"""
experiments/run_all.py
======================
Master orchestration script for the memory-in-TRPs research pipeline.

Reproduces all results from Guerrini et al. (EPJ Data Science 2026) for any
subset of the eight supported datasets, running Steps 6-10 and calling the
shared plotting routines in plotting.py to regenerate all paper figures.

Usage examples
--------------
Run everything for all 8 datasets (slow, ~20 min):
    python experiments/run_all.py

Quick smoke test on two datasets, 300 paths, skip step 9:
    python experiments/run_all.py --datasets primaryschool hospital --steps 6 8 10 --num-paths 300

Custom resolution list and disable caching:
    python experiments/run_all.py --datasets primaryschool --t-res 20 60 300 --no-cache

All datasets, steps 6 and 8 only:
    python experiments/run_all.py --steps 6 8

Command-line arguments
-----------------------
--datasets   Space-separated dataset keys.  Default: all 8 datasets.
--steps      Subset of steps to run (6 8 10).  Default: 6 8 10.
             Step 9 (large-scale null study) is intentionally excluded as it
             can take >30 min; run experiments/step09_nulls.py separately.
--num-paths  TRPs sampled per day per dataset.  Default: BIC_NUM_PATHS from config.
--t-res      Aggregation window sizes in seconds for Step 10 heatmaps.
             Default: RESOLUTION_T_RES_LIST from config.
--cache / --no-cache  Whether to reuse cached path arrays.  Default: cache on.

Outputs
-------
results/summary_p_m10_tres20.csv   Summary table: p (MEM) and p (MEM+SBM) at
                                    m=10 and t_res=20 s for every dataset.
results/runall_fig02_bic.png       Figure 2: BIC comparison panel (Step 8).
results/runall_fig03_memory.png    Figure 3: p vs m for labeled datasets.
results/runall_fig05_heatmap_mem.png   Figure 5: MEM heatmap (Step 10, primary school).
results/runall_fig06_heatmap_sbm.png   Figure 6: MEM+SBM heatmap (Step 10, primary school).
results/runall_fig07_tau_mem.png       Figure 7: p(tau) for MEM.
results/runall_fig08_tau_sbm.png       Figure 8: p(tau) for MEM+SBM.
results/runall_tau_curves_combined.png Combined p(tau) comparison.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

# Ensure package modules and config are discoverable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    BIC_NUM_PATHS,
    MEMORY_HORIZONS,
    P_INIT,
    PATH_LENGTH,
    RANDOM_SEED,
    RESOLUTION_T_RES_LIST,
    RESULTS_DIR,
    T_RES_SECONDS,
    TAU_MINUTES_LIST,
)
from memory_trp.aggregation import TemporalGraph, build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.models import (
    community_interaction_matrix,
    fit_mem_fixed_point,
    fit_mem_sbm,
)
from memory_trp.nulls import er_null_graph
from memory_trp.plotting import (
    plot_bic_panels,
    plot_memory_curves,
    plot_resolution_heatmap,
    plot_resolution_heatmaps_side_by_side,
    plot_tau_curves,
)
from memory_trp.records import build_records, build_records_time
from memory_trp.trp import generate_many_trps

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# All 8 supported datasets: (display_name, dataset_key, has_labels)
ALL_DATASETS: list[tuple[str, str, bool]] = [
    ("Primary School", "primaryschool", True),
    ("High School 2011", "hs2011", True),
    ("High School 2012", "hs2012", True),
    ("High School 2013", "hs2013", True),
    ("Workplace", "workplace", True),
    ("Hospital", "hospital", True),
    ("Conference", "conference", False),
    ("Malawi", "malawi", False),
]

ALL_DATASET_KEYS: list[str] = [key for _, key, _ in ALL_DATASETS]

# Cache directory for path arrays
CACHE_DIR = RESULTS_DIR / "cache_runall"


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _cache_path(dataset_key: str, t_res: int, num_paths: int) -> Path:
    """Return filesystem path for a cached path-list pickle file."""
    return CACHE_DIR / f"{dataset_key}_tres{t_res}_np{num_paths}.pkl"


def _save_cache(
    dataset_key: str,
    t_res: int,
    num_paths: int,
    paths: list[tuple[list[int], list[int]]],
    snapshots: list,
    labels: np.ndarray | None,
    n: int,
) -> None:
    """Pickle (paths, snapshots, labels, n) to the cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "paths": paths,
        "snapshots": snapshots,
        "labels": labels if labels is None else labels.tolist(),
        "n": n,
    }
    dest = _cache_path(dataset_key, t_res, num_paths)
    with open(dest, "wb") as f:
        pickle.dump(payload, f, protocol=4)


def _load_cache(
    dataset_key: str,
    t_res: int,
    num_paths: int,
) -> tuple[list, list, np.ndarray | None, int] | None:
    """Load cached (paths, snapshots, labels, n) or return None if missing."""
    dest = _cache_path(dataset_key, t_res, num_paths)
    if not dest.exists():
        return None
    with open(dest, "rb") as f:
        payload = pickle.load(f)
    labels = payload["labels"]
    if labels is not None:
        labels = np.array(labels, dtype=int)
    return payload["paths"], payload["snapshots"], labels, payload["n"]


# ---------------------------------------------------------------------------
# Core pipeline helpers
# ---------------------------------------------------------------------------

def _sample_paths_for_dataset(
    dataset_key: str,
    t_res: int,
    num_paths: int,
    rng: np.random.Generator,
    use_cache: bool = True,
) -> tuple[list[tuple[list[int], list[int]]], list, np.ndarray | None, int]:
    """Load contacts, build temporal graphs, and sample TRPs.

    Returns
    -------
    paths : list of (nodes, times) tuples
    snapshots : flat list of CSR snapshot matrices (all days pooled)
    labels : community label array, or None
    n : number of nodes
    """
    if use_cache:
        cached = _load_cache(dataset_key, t_res, num_paths)
        if cached is not None:
            print(f"    [cache hit] {dataset_key} t_res={t_res}s np={num_paths}", flush=True)
            return cached

    contacts, labels = load_contacts(dataset_key)
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    days = split_by_day(contacts, dataset_key)

    all_paths: list[tuple[list[int], list[int]]] = []
    all_snapshots: list = []

    num_paths_day = num_paths if len(days) > 1 else max(1000, num_paths * 2)

    for d_idx, day_df in enumerate(days):
        g_day = build_temporal_graph(
            day_df,
            n=n,
            labels=labels,
            t_res=t_res,
            name=f"{dataset_key}_tres{t_res}_d{d_idx + 1}",
        )
        all_snapshots.extend(g_day.snapshots)

        day_paths = generate_many_trps(
            g_day,
            num_paths=num_paths_day,
            length=PATH_LENGTH,
            rng=rng,
        )
        all_paths.extend(day_paths)

    if use_cache:
        _save_cache(dataset_key, t_res, num_paths, all_paths, all_snapshots, labels, n)

    return all_paths, all_snapshots, labels, n


def _build_pooled_graph(
    dataset_key: str,
    snapshots: list,
    labels: np.ndarray | None,
    n: int,
    t_res: int,
) -> TemporalGraph:
    """Assemble a pooled TemporalGraph from flat snapshot list for community matrix."""
    return TemporalGraph(
        name=f"{dataset_key}_tres{t_res}_pooled",
        n=n,
        t_res=t_res,
        snapshots=snapshots,
        labels=labels,
        start_time=0,
    )


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_step6(
    datasets_to_run: list[tuple[str, str, bool]],
    num_paths: int,
    rng: np.random.Generator,
    use_cache: bool,
) -> dict[str, dict]:
    """Figure 3: p vs m (MEM + ER null) for the primary school dataset.

    Returns per-dataset dict with keys 'p_mem', 'p_null', 'n'.
    Only runs on datasets that have labels (labeled datasets only, to keep
    the null comparison fair); uses 'primaryschool' as the canonical dataset.
    """
    print("\n" + "=" * 70, flush=True)
    print("STEP 6 -- MEM on Primary School vs. ER null  (Figure 3)", flush=True)
    print("=" * 70, flush=True)

    results: dict[str, dict] = {}

    # For step 6, always use primaryschool as the canonical dataset
    target_key = "primaryschool"
    target_name = "Primary School"

    # Sample empirical paths
    paths_emp, snaps, labels, n = _sample_paths_for_dataset(
        target_key, T_RES_SECONDS, num_paths, rng, use_cache
    )

    # Build null paths for each day
    contacts, _ = load_contacts(target_key)
    days = split_by_day(contacts, target_key)
    all_null_paths: list = []
    num_paths_day = num_paths if len(days) > 1 else max(1000, num_paths * 2)

    for d_idx, day_df in enumerate(days):
        g_emp = build_temporal_graph(
            day_df, n=n, labels=labels, t_res=T_RES_SECONDS,
            name=f"{target_key}_d{d_idx + 1}",
        )
        g_null = er_null_graph(g_emp, rng)
        null_day = generate_many_trps(
            g_null, num_paths=num_paths_day, length=PATH_LENGTH, rng=rng
        )
        all_null_paths.extend(null_day)

    p_emp_list: list[float] = []
    p_null_list: list[float] = []
    for m in MEMORY_HORIZONS:
        rec_emp = build_records(paths_emp, m=m)
        rec_null = build_records(all_null_paths, m=m)
        p_emp_list.append(fit_mem_fixed_point(rec_emp, n=n).p)
        p_null_list.append(fit_mem_fixed_point(rec_null, n=n).p)
        print(f"  m={m:2d} | p_emp={p_emp_list[-1]:.4f} | p_null={p_null_list[-1]:.4f}", flush=True)

    results[target_key] = {"p_mem": p_emp_list, "p_null": p_null_list, "n": n}

    # Figure 3: p vs m, empirical vs ER null
    fig3_path = RESULTS_DIR / "runall_fig03_memory.png"
    plot_memory_curves(
        m_values=MEMORY_HORIZONS,
        p_emp=p_emp_list,
        p_null=p_null_list,
        dataset_name=f"{target_name} (t_res={T_RES_SECONDS}s)",
        output_path=fig3_path,
    )
    print(f"  Saved Figure 3 -> {fig3_path}", flush=True)

    return results


def run_step8(
    datasets_to_run: list[tuple[str, str, bool]],
    num_paths: int,
    rng: np.random.Generator,
    use_cache: bool,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Figure 2: BIC comparison (MEM vs MEM+SBM) for labeled datasets.

    Returns
    -------
    panel_data : dict mapping display_name -> {bic_mem, bic_sbm, k, n}
    df_records : full records DataFrame (all datasets, all m values)
    """
    print("\n" + "=" * 70, flush=True)
    print("STEP 8 -- BIC Comparison: MEM vs. MEM+SBM  (Figure 2)", flush=True)
    print("=" * 70, flush=True)

    labeled = [(name, key, hl) for name, key, hl in datasets_to_run if hl]

    panel_data: dict[str, dict] = {}
    rows: list[dict] = []

    for d_idx, (display_name, dataset_key, _) in enumerate(labeled, 1):
        t0 = time.time()
        print(f"\n  [{d_idx}/{len(labeled)}] {display_name} ({dataset_key})", flush=True)

        paths, snaps, labels, n = _sample_paths_for_dataset(
            dataset_key, T_RES_SECONDS, num_paths, rng, use_cache
        )
        k = int(labels.max()) + 1

        pooled_graph = _build_pooled_graph(dataset_key, snaps, labels, n, T_RES_SECONDS)
        init_C = community_interaction_matrix(pooled_graph, labels=labels)

        bic_mem_list: list[float] = []
        bic_sbm_list: list[float] = []

        for m in MEMORY_HORIZONS:
            records = build_records(paths, m=m)
            res_mem = fit_mem_fixed_point(records, n=n)
            res_sbm = fit_mem_sbm(records, labels=labels, init_C=init_C, p0=P_INIT)

            bic_mem_list.append(res_mem.bic)
            bic_sbm_list.append(res_sbm.bic)

            rows.append({
                "dataset": display_name,
                "dataset_key": dataset_key,
                "n_nodes": n,
                "k_communities": k,
                "m": m,
                "n_records": len(records),
                "p_mem": res_mem.p,
                "bic_mem": res_mem.bic,
                "p_sbm": res_sbm.p,
                "bic_sbm": res_sbm.bic,
                "delta_bic": res_mem.bic - res_sbm.bic,
            })

            print(
                f"    m={m:2d} | p(MEM)={res_mem.p:.4f} bic={res_mem.bic:.1f} |"
                f" p(SBM)={res_sbm.p:.4f} bic={res_sbm.bic:.1f}",
                flush=True,
            )

        panel_data[display_name] = {"bic_mem": bic_mem_list, "bic_sbm": bic_sbm_list, "k": k, "n": n}
        print(f"  Completed {display_name} in {time.time() - t0:.1f}s.", flush=True)

    # Figure 2: BIC panels
    fig2_path = RESULTS_DIR / "runall_fig02_bic.png"
    plot_bic_panels(
        dataset_results=panel_data,
        m_values=MEMORY_HORIZONS,
        output_path=fig2_path,
    )
    print(f"\n  Saved Figure 2 -> {fig2_path}", flush=True)

    df_records = pd.DataFrame(rows)
    bic_csv = RESULTS_DIR / "runall_bic_results.csv"
    df_records.to_csv(bic_csv, index=False)
    print(f"  Saved BIC CSV -> {bic_csv}", flush=True)

    return panel_data, df_records


def run_step10(
    t_res_list: list[int],
    num_paths: int,
    rng: np.random.Generator,
    use_cache: bool,
) -> dict:
    """Figures 5-8: resolution heatmaps and time-based memory curves.

    Uses 'primaryschool' (the paper's canonical dataset for this analysis).

    Returns
    -------
    dict with keys 'p_mem_grid', 'p_sbm_grid', 'p_mem_tau', 'p_sbm_tau'
    """
    print("\n" + "=" * 70, flush=True)
    print("STEP 10 -- Resolution Heatmaps & Time-Based Memory  (Figs 5-8)", flush=True)
    print("=" * 70, flush=True)

    dataset_key = "primaryschool"
    display_name = "Primary School"

    n_res = len(t_res_list)
    n_horizons = len(MEMORY_HORIZONS)
    p_mem_grid = np.zeros((n_res, n_horizons))
    p_sbm_grid = np.zeros((n_res, n_horizons))

    # Cache 20s paths for the tau study
    paths_20s: list | None = None
    init_c_20s: np.ndarray | None = None

    part1_rows: list[dict] = []

    for r_idx, t_res in enumerate(t_res_list):
        print(f"\n  [{r_idx + 1}/{n_res}] t_res = {t_res}s", flush=True)

        paths, snaps, labels, n = _sample_paths_for_dataset(
            dataset_key, t_res, num_paths, rng, use_cache
        )
        pooled = _build_pooled_graph(dataset_key, snaps, labels, n, t_res)
        init_C = community_interaction_matrix(pooled, labels=labels)

        if t_res == T_RES_SECONDS or (r_idx == 0 and paths_20s is None):
            paths_20s = paths
            init_c_20s = init_C

        for m_idx, m in enumerate(MEMORY_HORIZONS):
            records = build_records(paths, m=m)
            if len(records) == 0:
                p_mem_grid[r_idx, m_idx] = np.nan
                p_sbm_grid[r_idx, m_idx] = np.nan
                continue

            p_mem = fit_mem_fixed_point(records, n=n).p
            res_sbm = fit_mem_sbm(records, labels=labels, init_C=init_C)
            p_sbm = res_sbm.p

            p_mem_grid[r_idx, m_idx] = p_mem
            p_sbm_grid[r_idx, m_idx] = p_sbm

            print(f"    m={m:2d} | p(MEM)={p_mem:.4f} | p(MEM+SBM)={p_sbm:.4f}", flush=True)

            part1_rows.append({
                "dataset": dataset_key,
                "t_res_seconds": t_res,
                "m_horizon": m,
                "n_records": len(records),
                "p_mem": p_mem,
                "p_sbm": p_sbm,
            })

    # Save Part 1 CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_p1 = pd.DataFrame(part1_rows)
    df_p1.to_csv(RESULTS_DIR / "runall_resolution_heatmaps.csv", index=False)

    # Figures 5 and 6 -- individual heatmaps
    plot_resolution_heatmap(
        matrix=p_mem_grid,
        t_res_list=t_res_list,
        m_list=MEMORY_HORIZONS,
        model_name="MEM",
        dataset_name=display_name,
        output_path=RESULTS_DIR / "runall_fig05_heatmap_mem.png",
        cmap="Blues",
    )
    print(f"  Saved Figure 5 -> {RESULTS_DIR / 'runall_fig05_heatmap_mem.png'}", flush=True)

    plot_resolution_heatmap(
        matrix=p_sbm_grid,
        t_res_list=t_res_list,
        m_list=MEMORY_HORIZONS,
        model_name="MEM+SBM",
        dataset_name=display_name,
        output_path=RESULTS_DIR / "runall_fig06_heatmap_sbm.png",
        cmap="Reds",
    )
    print(f"  Saved Figure 6 -> {RESULTS_DIR / 'runall_fig06_heatmap_sbm.png'}", flush=True)

    # Side-by-side combined heatmap
    plot_resolution_heatmaps_side_by_side(
        p_mem_matrix=p_mem_grid,
        p_sbm_matrix=p_sbm_grid,
        t_res_list=t_res_list,
        m_list=MEMORY_HORIZONS,
        dataset_name=display_name,
        output_path=RESULTS_DIR / "runall_fig05_06_side_by_side.png",
    )

    # ---- Part 2: time-based memory (tau study) ----
    print("\n  -- Part 2: p(tau) curves --", flush=True)

    p_mem_tau: list[float] = []
    p_sbm_tau: list[float] = []
    tau_rows: list[dict] = []

    assert paths_20s is not None, "20s paths were not populated"
    assert init_c_20s is not None

    # Reload n and labels for 20s run
    contacts_ps, labels_ps = load_contacts(dataset_key)
    n_ps = int(max(contacts_ps["i"].max(), contacts_ps["j"].max())) + 1

    for tau_min in TAU_MINUTES_LIST:
        tau_sec = tau_min * 60.0
        records_tau = build_records_time(paths_20s, tau_seconds=tau_sec, t_res=T_RES_SECONDS)

        if len(records_tau) == 0:
            p_mem_tau.append(np.nan)
            p_sbm_tau.append(np.nan)
            print(f"  tau={tau_min:.1f}min: 0 records, skipped", flush=True)
            continue

        p_m = fit_mem_fixed_point(records_tau, n=n_ps).p
        p_s = fit_mem_sbm(records_tau, labels=labels_ps, init_C=init_c_20s).p

        p_mem_tau.append(p_m)
        p_sbm_tau.append(p_s)

        print(
            f"  tau={tau_min:4.1f}min ({tau_sec:.0f}s) | "
            f"rec={len(records_tau):,} | p(MEM)={p_m:.4f} | p(MEM+SBM)={p_s:.4f}",
            flush=True,
        )

        tau_rows.append({
            "dataset": dataset_key,
            "tau_minutes": tau_min,
            "tau_seconds": tau_sec,
            "n_records": len(records_tau),
            "p_mem": p_m,
            "p_sbm": p_s,
        })

    pd.DataFrame(tau_rows).to_csv(RESULTS_DIR / "runall_time_memory_results.csv", index=False)

    # Figures 7, 8, and combined
    plot_tau_curves(
        tau_minutes=TAU_MINUTES_LIST,
        p_mem=p_mem_tau,
        p_sbm=p_sbm_tau,
        dataset_name=display_name,
        output_path_mem=RESULTS_DIR / "runall_fig07_tau_mem.png",
        output_path_sbm=RESULTS_DIR / "runall_fig08_tau_sbm.png",
        output_path_combined=RESULTS_DIR / "runall_tau_curves_combined.png",
    )
    print(f"  Saved Figures 7, 8, combined tau curves.", flush=True)

    return {
        "p_mem_grid": p_mem_grid,
        "p_sbm_grid": p_sbm_grid,
        "p_mem_tau": p_mem_tau,
        "p_sbm_tau": p_sbm_tau,
    }


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(
    datasets_to_run: list[tuple[str, str, bool]],
    num_paths: int,
    rng: np.random.Generator,
    use_cache: bool,
) -> pd.DataFrame:
    """Compute and save summary table: p_mem and p_sbm at m=10, t_res=20s.

    This is a standalone quick-fit run for each dataset so we always have
    fresh p values even if step 8 was skipped.

    Returns the DataFrame.
    """
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY -- p (MEM & MEM+SBM) at m=10, t_res=20 s, all datasets", flush=True)
    print("=" * 70, flush=True)

    M_SUMMARY = 10  # The m horizon used in the summary table

    rows: list[dict] = []
    for display_name, dataset_key, has_labels in datasets_to_run:
        print(f"  {display_name} ({dataset_key})...", flush=True)

        paths, snaps, labels, n = _sample_paths_for_dataset(
            dataset_key, T_RES_SECONDS, num_paths, rng, use_cache
        )
        records = build_records(paths, m=M_SUMMARY)

        if len(records) == 0:
            rows.append({
                "dataset": display_name, "dataset_key": dataset_key,
                "has_labels": has_labels, "n_nodes": n,
                "m": M_SUMMARY, "t_res_seconds": T_RES_SECONDS,
                "n_records": 0, "p_mem": float("nan"), "p_sbm": float("nan"),
            })
            continue

        p_mem = fit_mem_fixed_point(records, n=n).p
        p_sbm: float = float("nan")

        if has_labels and labels is not None:
            pooled = _build_pooled_graph(dataset_key, snaps, labels, n, T_RES_SECONDS)
            init_C = community_interaction_matrix(pooled, labels=labels)
            p_sbm = fit_mem_sbm(records, labels=labels, init_C=init_C).p

        rows.append({
            "dataset": display_name,
            "dataset_key": dataset_key,
            "has_labels": has_labels,
            "n_nodes": n,
            "m": M_SUMMARY,
            "t_res_seconds": T_RES_SECONDS,
            "n_records": len(records),
            "p_mem": p_mem,
            "p_sbm": p_sbm,
        })

        sbm_str = f"{p_sbm:.4f}" if not np.isnan(p_sbm) else "N/A"
        print(f"    p(MEM)={p_mem:.4f}  p(MEM+SBM)={sbm_str}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "summary_p_m10_tres20.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved summary CSV -> {csv_path}", flush=True)

    # Pretty-print table
    print("\n  Summary table (m=10, t_res=20s):", flush=True)
    print(f"  {'Dataset':<22} {'n':>6}  {'p(MEM)':>8}  {'p(SBM)':>8}", flush=True)
    print("  " + "-" * 50, flush=True)
    for _, row in df.iterrows():
        sbm_str = f"{row['p_sbm']:.4f}" if not np.isnan(row["p_sbm"]) else " N/A"
        print(f"  {row['dataset']:<22} {row['n_nodes']:>6}  {row['p_mem']:>8.4f}  {sbm_str:>8}", flush=True)

    return df


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the populated namespace."""
    parser = argparse.ArgumentParser(
        description="Master orchestration script for the memory-in-TRPs pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASET_KEYS,
        metavar="DATASET",
        help=(
            "Space-separated dataset keys to include. "
            f"Choices: {', '.join(ALL_DATASET_KEYS)}."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[6, 8, 10],
        metavar="STEP",
        help="Steps to run. Subset of {6, 8, 10}.",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=BIC_NUM_PATHS,
        dest="num_paths",
        help="TRPs sampled per day per dataset.",
    )
    parser.add_argument(
        "--t-res",
        nargs="+",
        type=int,
        default=RESOLUTION_T_RES_LIST,
        dest="t_res",
        metavar="SECONDS",
        help="Aggregation window sizes (seconds) for Step 10 heatmaps.",
    )
    parser.add_argument(
        "--cache",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to use and store cached path arrays.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the requested steps for the requested datasets."""
    args = _parse_args()

    # Validate requested dataset keys
    invalid = set(args.datasets) - set(ALL_DATASET_KEYS)
    if invalid:
        raise ValueError(
            f"Unknown dataset keys: {sorted(invalid)}. "
            f"Valid options: {ALL_DATASET_KEYS}"
        )

    # Validate requested steps
    valid_steps = {6, 8, 10}
    invalid_steps = set(args.steps) - valid_steps
    if invalid_steps:
        raise ValueError(
            f"Unknown step numbers: {sorted(invalid_steps)}. "
            f"Valid options: {sorted(valid_steps)}"
        )

    # Build the filtered dataset list preserving registry order
    datasets_to_run = [
        (name, key, hl)
        for name, key, hl in ALL_DATASETS
        if key in set(args.datasets)
    ]

    # Ensure results dir exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)
    start_total = time.time()

    print("=" * 70, flush=True)
    print("run_all.py -- Memory in Time-Respecting Paths (Guerrini et al. 2026)", flush=True)
    print("=" * 70, flush=True)
    print(f"Datasets  : {[key for _, key, _ in datasets_to_run]}", flush=True)
    print(f"Steps     : {sorted(args.steps)}", flush=True)
    print(f"Paths/day : {args.num_paths}", flush=True)
    print(f"t_res list: {args.t_res} s  (Step 10)", flush=True)
    print(f"Cache     : {'enabled' if args.cache else 'disabled'}", flush=True)
    print("=" * 70, flush=True)

    # Run the requested steps
    if 6 in args.steps:
        run_step6(datasets_to_run, args.num_paths, rng, args.cache)

    if 8 in args.steps:
        run_step8(datasets_to_run, args.num_paths, rng, args.cache)

    if 10 in args.steps:
        run_step10(args.t_res, args.num_paths, rng, args.cache)

    # Always write the summary CSV
    write_summary_csv(datasets_to_run, args.num_paths, rng, args.cache)

    elapsed = time.time() - start_total
    print("\n" + "=" * 70, flush=True)
    print(f"All done in {elapsed:.1f}s ({elapsed / 60:.2f} min).", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()

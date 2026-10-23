"""
experiments/step10_resolution.py
================================
Resolution analysis and time-based memory model evaluation.

Reproduces:
1. Figures 5 & 6 from Guerrini et al. (EPJ Data Science 2026):
   - Loops aggregation window t_res in [20 s, 60 s, 300 s, 600 s, 900 s]
   - Loops discrete memory horizon m in MEMORY_HORIZONS ([5, 10, 15, 20])
   - Fits MEM and MEM+SBM models
   - Draws heatmaps of inferred memory parameter p with numbers in cells.
   - Saves CSV table to results/step10_resolution_heatmaps.csv.
   - Saves heatmaps to results/step10_fig05_heatmap_mem.png,
     results/step10_fig06_heatmap_sbm.png, and side-by-side comparison.

2. Figures 7 & 8 from Guerrini et al. (EPJ Data Science 2026):
   - Builds time-based memory records using build_records_time(paths, tau_seconds).
   - Loops tau across minutes: [2, 5, 10, 15, 20, 30, 45, 60].
   - Fits MEM and MEM+SBM models.
   - Plots p vs tau curves.
   - Saves CSV table to results/step10_time_memory_results.csv.
   - Saves curves to results/step10_fig07_tau_mem.png,
     results/step10_fig08_tau_sbm.png, and combined comparison.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure package modules and config are discoverable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    MEMORY_HORIZONS,
    PATH_LENGTH,
    RANDOM_SEED,
    RESOLUTION_DATASET,
    RESOLUTION_NUM_PATHS,
    RESOLUTION_T_RES_LIST,
    RESULTS_DIR,
    TAU_MINUTES_LIST,
    T_RES_SECONDS,
)
from memory_trp.aggregation import TemporalGraph, build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.models import (
    community_interaction_matrix,
    fit_mem_fixed_point,
    fit_mem_sbm,
)
from memory_trp.plotting import (
    plot_resolution_heatmap,
    plot_resolution_heatmaps_side_by_side,
    plot_tau_curves,
)
from memory_trp.records import build_records, build_records_time
from memory_trp.trp import generate_many_trps


def run_step10_pipeline() -> None:
    """Execute the resolution study and time-based memory analysis."""
    print("=" * 76, flush=True)
    print("Step 10 Experiment: Resolution Analysis & Time-Based Memory Modeling", flush=True)
    print(f"Dataset: '{RESOLUTION_DATASET}' | Paths per day: {RESOLUTION_NUM_PATHS:,}", flush=True)
    print(f"Resolutions (t_res): {RESOLUTION_T_RES_LIST} seconds", flush=True)
    print(f"Discrete horizons (m): {MEMORY_HORIZONS} steps", flush=True)
    print(f"Time horizons (tau): {TAU_MINUTES_LIST} minutes", flush=True)
    print("=" * 76, flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    start_total = time.time()

    # Load contact sequences and community labels
    contacts, labels = load_contacts(RESOLUTION_DATASET)
    if labels is None:
        raise ValueError(f"Dataset '{RESOLUTION_DATASET}' requires community labels.")

    n_nodes = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    n_comm = int(labels.max()) + 1
    days = split_by_day(contacts, RESOLUTION_DATASET)
    print(f"Loaded '{RESOLUTION_DATASET}': n={n_nodes}, K={n_comm}, days={len(days)}", flush=True)

    # -------------------------------------------------------------------------
    # PART 1: Aggregation window resolution t_res vs. Memory horizon m
    # -------------------------------------------------------------------------
    print("\n" + "-" * 76, flush=True)
    print("PART 1: Inferred Memory p across Resolutions t_res and Horizons m", flush=True)
    print("-" * 76, flush=True)

    n_res = len(RESOLUTION_T_RES_LIST)
    n_horizons = len(MEMORY_HORIZONS)
    p_mem_grid = np.zeros((n_res, n_horizons), dtype=float)
    p_sbm_grid = np.zeros((n_res, n_horizons), dtype=float)

    part1_rows: list[dict[str, object]] = []

    # Cache for 20s paths and init_C to reuse in Part 2
    paths_baseline_20s: list[tuple[list[int], list[int]]] = []
    init_c_baseline_20s: np.ndarray | None = None

    for r_idx, t_res in enumerate(RESOLUTION_T_RES_LIST):
        t_res_start = time.time()
        print(f"\n[{r_idx + 1}/{n_res}] Processing resolution t_res = {t_res} seconds...", flush=True)

        # Build daily graphs and sample TRPs for this resolution
        res_paths: list[tuple[list[int], list[int]]] = []
        res_snapshots: list = []

        for d_idx, day_df in enumerate(days):
            g_day = build_temporal_graph(
                day_df,
                n=n_nodes,
                labels=labels,
                t_res=t_res,
                name=f"{RESOLUTION_DATASET}_tres{t_res}_d{d_idx + 1}",
            )
            res_snapshots.extend(g_day.snapshots)

            day_paths = generate_many_trps(
                g_day,
                num_paths=RESOLUTION_NUM_PATHS,
                length=PATH_LENGTH,
                rng=rng,
            )
            res_paths.extend(day_paths)

        print(
            f"      Sampled {len(res_paths):,} TRPs across {len(res_snapshots):,} snapshots "
            f"({time.time() - t_res_start:.1f}s)",
            flush=True,
        )

        # Initial C from pooled community interaction matrix
        pooled_graph = TemporalGraph(
            name=f"{RESOLUTION_DATASET}_tres{t_res}_pooled",
            n=n_nodes,
            t_res=t_res,
            snapshots=res_snapshots,
            labels=labels,
            start_time=0,
        )
        init_C = community_interaction_matrix(pooled_graph, labels=labels)

        # Save baseline 20s TRPs and init_C for Part 2
        if t_res == T_RES_SECONDS or (r_idx == 0 and len(paths_baseline_20s) == 0):
            paths_baseline_20s = res_paths
            init_c_baseline_20s = init_C

        # Fit MEM and MEM+SBM across memory horizons m
        for m_idx, m in enumerate(MEMORY_HORIZONS):
            records = build_records(res_paths, m=m)
            n_recs = len(records)

            if n_recs == 0:
                print(f"      m={m}: 0 valid records found. Skipping.", flush=True)
                p_mem_grid[r_idx, m_idx] = np.nan
                p_sbm_grid[r_idx, m_idx] = np.nan
                continue

            # Fit plain MEM
            p_mem = fit_mem_fixed_point(records, n=n_nodes).p
            p_mem_grid[r_idx, m_idx] = p_mem

            # Fit MEM+SBM
            sbm_res = fit_mem_sbm(
                records=records,
                labels=labels,
                init_C=init_C,
            )
            p_sbm = sbm_res.p
            p_sbm_grid[r_idx, m_idx] = p_sbm

            print(
                f"      m={m:2d} | Records: {n_recs:6,d} | p(MEM)={p_mem:.4f} | p(MEM+SBM)={p_sbm:.4f}",
                flush=True,
            )

            part1_rows.append(
                {
                    "dataset": RESOLUTION_DATASET,
                    "t_res_seconds": t_res,
                    "m_horizon": m,
                    "n_records": n_recs,
                    "p_mem": p_mem,
                    "p_sbm": p_sbm,
                }
            )

    # Save Part 1 CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_part1 = pd.DataFrame(part1_rows)
    csv_part1_path = RESULTS_DIR / "step10_resolution_heatmaps.csv"
    df_part1.to_csv(csv_part1_path, index=False)
    print(f"\nSaved resolution table to: {csv_part1_path}", flush=True)

    # Plot Part 1 Heatmaps
    fig05_path = RESULTS_DIR / "step10_fig05_heatmap_mem.png"
    fig06_path = RESULTS_DIR / "step10_fig06_heatmap_sbm.png"
    fig_side_path = RESULTS_DIR / "step10_fig05_06_heatmaps_side_by_side.png"

    display_dataset = "Primary School"
    plot_resolution_heatmap(
        matrix=p_mem_grid,
        t_res_list=RESOLUTION_T_RES_LIST,
        m_list=MEMORY_HORIZONS,
        model_name="MEM",
        dataset_name=display_dataset,
        output_path=fig05_path,
        cmap="Blues",
    )
    print(f"Saved Figure 5 (MEM heatmap) to: {fig05_path}", flush=True)

    plot_resolution_heatmap(
        matrix=p_sbm_grid,
        t_res_list=RESOLUTION_T_RES_LIST,
        m_list=MEMORY_HORIZONS,
        model_name="MEM+SBM",
        dataset_name=display_dataset,
        output_path=fig06_path,
        cmap="Reds",
    )
    print(f"Saved Figure 6 (MEM+SBM heatmap) to: {fig06_path}", flush=True)

    plot_resolution_heatmaps_side_by_side(
        p_mem_matrix=p_mem_grid,
        p_sbm_matrix=p_sbm_grid,
        t_res_list=RESOLUTION_T_RES_LIST,
        m_list=MEMORY_HORIZONS,
        dataset_name=display_dataset,
        output_path=fig_side_path,
    )
    print(f"Saved side-by-side heatmaps to: {fig_side_path}", flush=True)

    # -------------------------------------------------------------------------
    # PART 2: Time-based memory variant p(tau)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 76, flush=True)
    print("PART 2: Time-Based Memory Modeling p(tau) across Duration tau (minutes)", flush=True)
    print("-" * 76, flush=True)

    if init_c_baseline_20s is None:
        raise RuntimeError("Baseline init_C for 20s data was not initialized.")

    part2_rows: list[dict[str, object]] = []
    p_mem_tau_list: list[float] = []
    p_sbm_tau_list: list[float] = []

    for tau_min in TAU_MINUTES_LIST:
        tau_sec = tau_min * 60.0
        records_tau = build_records_time(
            paths=paths_baseline_20s,
            tau_seconds=tau_sec,
            t_res=T_RES_SECONDS,
        )
        n_recs = len(records_tau)

        if n_recs == 0:
            print(f"tau={tau_min:4.1f} min ({tau_sec:5.0f}s): 0 valid records found.", flush=True)
            p_mem_tau_list.append(np.nan)
            p_sbm_tau_list.append(np.nan)
            continue

        p_mem_tau = fit_mem_fixed_point(records_tau, n=n_nodes).p
        sbm_res_tau = fit_mem_sbm(
            records=records_tau,
            labels=labels,
            init_C=init_c_baseline_20s,
        )
        p_sbm_tau = sbm_res_tau.p

        p_mem_tau_list.append(p_mem_tau)
        p_sbm_tau_list.append(p_sbm_tau)

        print(
            f"tau={tau_min:4.1f} min ({tau_sec:5.0f}s) | Records: {n_recs:6,d} | "
            f"p(MEM)={p_mem_tau:.4f} | p(MEM+SBM)={p_sbm_tau:.4f}",
            flush=True,
        )

        part2_rows.append(
            {
                "dataset": RESOLUTION_DATASET,
                "tau_minutes": tau_min,
                "tau_seconds": tau_sec,
                "n_records": n_recs,
                "p_mem": p_mem_tau,
                "p_sbm": p_sbm_tau,
            }
        )

    # Save Part 2 CSV
    df_part2 = pd.DataFrame(part2_rows)
    csv_part2_path = RESULTS_DIR / "step10_time_memory_results.csv"
    df_part2.to_csv(csv_part2_path, index=False)
    print(f"\nSaved time-based memory table to: {csv_part2_path}", flush=True)

    # Plot Part 2 Curves
    fig07_path = RESULTS_DIR / "step10_fig07_tau_mem.png"
    fig08_path = RESULTS_DIR / "step10_fig08_tau_sbm.png"
    fig_tau_comb_path = RESULTS_DIR / "step10_tau_curves_combined.png"

    plot_tau_curves(
        tau_minutes=TAU_MINUTES_LIST,
        p_mem=p_mem_tau_list,
        p_sbm=p_sbm_tau_list,
        dataset_name=display_dataset,
        output_path_mem=fig07_path,
        output_path_sbm=fig08_path,
        output_path_combined=fig_tau_comb_path,
    )
    print(f"Saved Figure 7 (p vs tau for MEM) to: {fig07_path}", flush=True)
    print(f"Saved Figure 8 (p vs tau for MEM+SBM) to: {fig08_path}", flush=True)
    print(f"Saved combined comparison (p vs tau) to: {fig_tau_comb_path}", flush=True)

    print("\n" + "=" * 76, flush=True)
    print(f"Step 10 Pipeline completed successfully in {time.time() - start_total:.1f}s!", flush=True)
    print("=" * 76, flush=True)


if __name__ == "__main__":
    run_step10_pipeline()

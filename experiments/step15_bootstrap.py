"""
experiments/step15_bootstrap.py
================================
Bootstrap uncertainty quantification and time-shuffled null model comparison.

Workflow
--------
1. Part 1: Re-plots p versus m curves with 95% bootstrap confidence bands
   (resampling records with replacement B=N_BOOTSTRAP times) for empirical
   and Erdős-Rényi null temporal networks.
2. Part 2: Compares, for each dataset at a fixed memory horizon (default m=10):
     - Empirical network
     - Time-shuffled null (randomly permutes snapshots, destroying temporal order)
     - Erdős-Rényi null (destroys network structure, preserving snapshot edge count)
   Visualized as a grouped bar chart with 95% bootstrap error bars.
3. Saves numerical results to CSV and generates publication-ready figures.

Outputs
-------
results/step15_p_vs_m_all_datasets.png   Multi-panel p vs m with confidence bands
results/step15_null_comparison_m10.png   Grouped bar chart comparing null models
results/step15_null_comparison_m10.csv   Table of p values with 95% bootstrap CIs
results/step15_p_vs_m_bootstrap.csv     Table of p(m) curves with bootstrap CIs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure package root and src are discoverable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    MEMORY_HORIZONS,
    N_BOOTSTRAP,
    PATH_LENGTH,
    RANDOM_SEED,
    RESULTS_DIR,
    T_RES_SECONDS,
)
from memory_trp.aggregation import build_temporal_graph, split_by_day
from memory_trp.bootstrap import bootstrap_p
from memory_trp.loading import load_contacts
from memory_trp.nulls import er_null_graph, time_shuffle_graph
from memory_trp.plotting import (
    plot_memory_curves_with_ci,
    plot_memory_grid_with_ci,
    plot_null_comparison_barchart,
)
from memory_trp.records import build_records
from memory_trp.trp import generate_many_trps

TARGET_DATASETS: list[tuple[str, str]] = [
    ("Primary School", "primaryschool"),
    ("High School 2011", "hs2011"),
    ("High School 2012", "hs2012"),
    ("High School 2013", "hs2013"),
    ("Workplace", "workplace"),
    ("Hospital", "hospital"),
    ("Conference", "conference"),
    ("Malawi", "malawi"),
]


def run_bootstrap_experiments(
    datasets: list[tuple[str, str]] = TARGET_DATASETS,
    fixed_m: int = 10,
    m_values: list[int] = [1, 2, 3, 5, 7, 10],
    num_paths: int = 1500,
    B: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> None:
    """Run bootstrap confidence intervals and null comparison experiments."""
    print("=" * 70, flush=True)
    print("Step 15 — Bootstrap Uncertainty and Time-Shuffled Null Models", flush=True)
    print(f"Bootstrap replications B: {B} | Fixed m: {fixed_m}", flush=True)
    print(f"Horizons for curves: {m_values} | Paths/day: {num_paths:,}", flush=True)
    print(f"Datasets: {[name for name, _ in datasets]}", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.default_rng(seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Part 1: p versus m curves with 95% bootstrap confidence bands
    # -----------------------------------------------------------------------
    print("\n[Part 1] Computing p vs. m curves with 95% bootstrap bands...", flush=True)
    grid_curves: dict[str, dict[str, list[float]]] = {}
    p_m_csv_rows: list[dict] = []

    for ds_name, ds_key in datasets:
        t0 = time.time()
        contacts, labels = load_contacts(ds_key)
        days = split_by_day(contacts, ds_key)
        n = len(labels) if labels is not None else int(contacts[["i", "j"]].values.max() + 1)

        # Build day 1 graph for clean benchmark curve
        g_emp = build_temporal_graph(days[0], n, labels, t_res=T_RES_SECONDS, name=f"{ds_key}_day1")
        g_er = er_null_graph(g_emp, rng=rng)

        paths_emp = generate_many_trps(g_emp, num_paths=num_paths, length=PATH_LENGTH, rng=rng)
        paths_er = generate_many_trps(g_er, num_paths=num_paths, length=PATH_LENGTH, rng=rng)

        p_emp_list, ci_emp_low, ci_emp_high = [], [], []
        p_null_list, ci_null_low, ci_null_high = [], [], []

        for m_val in m_values:
            rec_emp = build_records(paths_emp, m=m_val)
            rec_er = build_records(paths_er, m=m_val)

            # Bootstrap for empirical
            res_emp = bootstrap_p(rec_emp, n=n, model="MEM", B=B, rng=rng)
            p_emp_list.append(res_emp.mean)
            ci_emp_low.append(res_emp.p2_5)
            ci_emp_high.append(res_emp.p97_5)

            # Bootstrap for ER null
            res_er = bootstrap_p(rec_er, n=n, model="MEM", B=B, rng=rng)
            p_null_list.append(res_er.mean)
            ci_null_low.append(res_er.p2_5)
            ci_null_high.append(res_er.p97_5)

            p_m_csv_rows.append({
                "dataset": ds_key,
                "dataset_name": ds_name,
                "m": m_val,
                "p_emp": res_emp.mean,
                "ci_emp_low": res_emp.p2_5,
                "ci_emp_high": res_emp.p97_5,
                "p_null_er": res_er.mean,
                "ci_null_low": res_er.p2_5,
                "ci_null_high": res_er.p97_5,
            })

        grid_curves[ds_name] = {
            "p_emp": p_emp_list,
            "ci_emp_low": ci_emp_low,
            "ci_emp_high": ci_emp_high,
            "p_null": p_null_list,
            "ci_null_low": ci_null_low,
            "ci_null_high": ci_null_high,
        }

        # Save individual dataset figure
        plot_memory_curves_with_ci(
            m_values=m_values,
            p_emp=p_emp_list,
            ci_emp_low=ci_emp_low,
            ci_emp_high=ci_emp_high,
            p_null=p_null_list,
            ci_null_low=ci_null_low,
            ci_null_high=ci_null_high,
            dataset_name=ds_name,
            output_path=RESULTS_DIR / f"step15_p_vs_m_{ds_key}.png",
        )

        elapsed = time.time() - t0
        print(f"  {ds_name:18s} completed in {elapsed:.1f}s", flush=True)

    # Save multi-panel grid figure
    grid_path = RESULTS_DIR / "step15_p_vs_m_all_datasets.png"
    plot_memory_grid_with_ci(grid_curves, m_values=m_values, output_path=grid_path)
    print(f"Saved p vs m grid plot with 95% CIs -> {grid_path}", flush=True)

    # Save p vs m CSV
    df_pm = pd.DataFrame(p_m_csv_rows)
    pm_csv_path = RESULTS_DIR / "step15_p_vs_m_bootstrap.csv"
    df_pm.to_csv(pm_csv_path, index=False)
    print(f"Saved p vs m CSV -> {pm_csv_path}", flush=True)

    # -----------------------------------------------------------------------
    # Part 2: Fixed-m null comparison (Real vs. Time-Shuffled vs. ER Null)
    # -----------------------------------------------------------------------
    print(f"\n[Part 2] Comparing Real, Time-Shuffled, and ER Null models at m={fixed_m}...", flush=True)
    comp_rows: list[dict] = []

    for ds_name, ds_key in datasets:
        t0 = time.time()
        contacts, labels = load_contacts(ds_key)
        days = split_by_day(contacts, ds_key)
        n = len(labels) if labels is not None else int(contacts[["i", "j"]].values.max() + 1)

        g_emp = build_temporal_graph(days[0], n, labels, t_res=T_RES_SECONDS, name=f"{ds_key}_day1")
        g_shuff = time_shuffle_graph(g_emp, rng=rng)
        g_er = er_null_graph(g_emp, rng=rng)

        # 1. Real Graph
        paths_emp = generate_many_trps(g_emp, num_paths=num_paths, length=PATH_LENGTH, rng=rng)
        rec_emp = build_records(paths_emp, m=fixed_m)
        res_emp = bootstrap_p(rec_emp, n=n, model="MEM", B=B, rng=rng)

        # 2. Time-Shuffled Graph
        paths_shuff = generate_many_trps(g_shuff, num_paths=num_paths, length=PATH_LENGTH, rng=rng)
        rec_shuff = build_records(paths_shuff, m=fixed_m)
        res_shuff = bootstrap_p(rec_shuff, n=n, model="MEM", B=B, rng=rng)

        # 3. ER Null Graph
        paths_er = generate_many_trps(g_er, num_paths=num_paths, length=PATH_LENGTH, rng=rng)
        rec_er = build_records(paths_er, m=fixed_m)
        res_er = bootstrap_p(rec_er, n=n, model="MEM", B=B, rng=rng)

        comp_rows.append({
            "dataset": ds_name,
            "dataset_key": ds_key,
            "m": fixed_m,
            "p_real": res_emp.mean,
            "p_real_low": res_emp.p2_5,
            "p_real_high": res_emp.p97_5,
            "p_shuffled": res_shuff.mean,
            "p_shuffled_low": res_shuff.p2_5,
            "p_shuffled_high": res_shuff.p97_5,
            "p_er": res_er.mean,
            "p_er_low": res_er.p2_5,
            "p_er_high": res_er.p97_5,
        })

        elapsed = time.time() - t0
        print(
            f"  {ds_name:18s}: Real={res_emp.mean:.3f} | Shuffled={res_shuff.mean:.3f} | ER={res_er.mean:.3f} | {elapsed:.1f}s",
            flush=True,
        )

    df_comp = pd.DataFrame(comp_rows)
    comp_csv_path = RESULTS_DIR / f"step15_null_comparison_m{fixed_m}.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"\nSaved null comparison CSV -> {comp_csv_path}", flush=True)

    barchart_path = RESULTS_DIR / f"step15_null_comparison_m{fixed_m}.png"
    plot_null_comparison_barchart(df_comp, fixed_m=fixed_m, output_path=barchart_path)
    print(f"Saved null comparison bar chart -> {barchart_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 15: Bootstrap CI and null comparison.")
    parser.add_argument("--fixed-m", type=int, default=10, help="Fixed horizon m for null comparison (default: 10)")
    parser.add_argument("--num-paths", type=int, default=1500, help="Sampled TRPs per day (default: 1500)")
    parser.add_argument("--bootstrap-b", type=int, default=N_BOOTSTRAP, help="Bootstrap draws B (default: 200)")
    args = parser.parse_args()

    run_bootstrap_experiments(
        fixed_m=args.fixed_m,
        num_paths=args.num_paths,
        B=args.bootstrap_b,
    )

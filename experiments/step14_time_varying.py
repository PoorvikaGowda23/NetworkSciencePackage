"""
experiments/step14_time_varying.py
==================================
Replicates time-resolved memory analysis across clock hours of the day.

Workflow
--------
For each dataset:
  1. Load contacts and community labels.
  2. Partition into distinct days with split_by_day.
  3. For each day:
     a. Build a TemporalGraph.
     b. Sample time-respecting paths (generate_many_trps).
     c. Extract transition records at m = 10.
     d. Group records into clock-time bins (default hourly, also 15-min) using
        fit_p_by_time_bin.
     e. Skip bins with fewer than 200 records.
     f. For MEM+SBM, hold community matrix C fixed from the global fit.
  4. Plot p vs. clock time for each dataset:
     - One line per day (thin).
     - Mean across days (thick bold red line).
  5. Generate multi-panel summary grids and save numerical data to CSV.

Outputs
-------
results/step14_time_varying_hourly.png      Summary grid (hourly bins)
results/step14_time_varying_15min.png       Summary grid (15-min bins)
results/step14_time_varying_hourly.csv      Numerical table of fitted p values
results/step14_time_varying_15min.csv       Numerical table of fitted p values
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
    PATH_LENGTH,
    RANDOM_SEED,
    RESULTS_DIR,
    T_RES_SECONDS,
)
from memory_trp.aggregation import build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.plotting import plot_time_varying_grid, plot_time_varying_p
from memory_trp.records import build_records
from memory_trp.timevarying import fit_p_by_time_bin, fit_p_by_time_bin_df
from memory_trp.trp import generate_many_trps

# Datasets to analyze (default: 6 labeled datasets with day structure)
TARGET_DATASETS: list[tuple[str, str]] = [
    ("Primary School", "primaryschool"),
    ("High School 2011", "hs2011"),
    ("High School 2012", "hs2012"),
    ("High School 2013", "hs2013"),
    ("Workplace", "workplace"),
    ("Hospital", "hospital"),
]


def run_time_varying_experiment(
    datasets: list[tuple[str, str]] = TARGET_DATASETS,
    bin_minutes_list: list[int] = [60, 15],
    model: str = "MEM",
    m: int = 10,
    num_paths: int = 1500,
    seed: int = RANDOM_SEED,
) -> None:
    """Run time-varying memory estimation across datasets and bin sizes."""
    print("=" * 70, flush=True)
    print("Step 14 — Time-Varying Memory Estimation across Clock Time", flush=True)
    print(f"Model: {model} | Horizon m: {m} | Paths/day: {num_paths:,}", flush=True)
    print(f"Bin widths: {bin_minutes_list} minutes", flush=True)
    print(f"Datasets: {[name for name, _ in datasets]}", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.default_rng(seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for bin_min in bin_minutes_list:
        t_bin_start = time.time()
        print(f"\n--- Running sweep for {bin_min}-minute bins ---", flush=True)

        all_grid_curves: dict[str, dict[str, dict[float, float]]] = {}
        csv_rows: list[dict] = []

        for ds_name, ds_key in datasets:
            t0 = time.time()
            contacts, labels = load_contacts(ds_key)
            days = split_by_day(contacts, ds_key)
            n = len(labels) if labels is not None else int(contacts[["i", "j"]].values.max() + 1)

            daily_results: dict[str, dict[float, float]] = {}

            for d_idx, day_df in enumerate(days):
                day_label = f"Day {d_idx + 1}"
                g_day = build_temporal_graph(
                    day_df, n, labels, t_res=T_RES_SECONDS, name=f"{ds_key}_d{d_idx+1}"
                )

                if len(g_day.snapshots) < 5:
                    continue

                paths = generate_many_trps(
                    g_day, num_paths=num_paths, length=PATH_LENGTH, rng=rng
                )
                if not paths:
                    continue

                records = build_records(paths, m=m)
                if len(records) < 200:
                    continue

                p_bins = fit_p_by_time_bin(
                    records=records,
                    graph_info=g_day,
                    bin_minutes=bin_min,
                    model=model,
                    min_records=200,
                )

                if p_bins:
                    daily_results[day_label] = p_bins
                    for hour_c, p_hat in p_bins.items():
                        csv_rows.append({
                            "dataset": ds_key,
                            "dataset_name": ds_name,
                            "day": day_label,
                            "bin_minutes": bin_min,
                            "clock_hour": hour_c,
                            "p_hat": p_hat,
                            "model": model,
                        })

            elapsed = time.time() - t0
            print(
                f"  {ds_name:18s}: {len(daily_results)}/{len(days)} days with valid bins | {elapsed:.1f}s",
                flush=True,
            )

            if daily_results:
                all_grid_curves[ds_name] = daily_results
                # Save individual dataset figure
                plot_time_varying_p(
                    daily_results=daily_results,
                    dataset_name=ds_name,
                    output_path=RESULTS_DIR / f"step14_{ds_key}_{bin_min}min.png",
                    bin_minutes=bin_min,
                    model=model,
                )

        # Save summary grid plot across all datasets
        suffix = "hourly" if bin_min == 60 else f"{bin_min}min"
        grid_png = RESULTS_DIR / f"step14_time_varying_{suffix}.png"
        plot_time_varying_grid(
            dataset_day_curves=all_grid_curves,
            output_path=grid_png,
            bin_minutes=bin_min,
            model=model,
        )
        print(f"Saved {suffix} grid plot -> {grid_png}", flush=True)

        # Save CSV
        csv_path = RESULTS_DIR / f"step14_time_varying_{suffix}.csv"
        df_csv = pd.DataFrame(csv_rows)
        df_csv.to_csv(csv_path, index=False)
        print(f"Saved {suffix} CSV -> {csv_path}", flush=True)

        elapsed_bin = time.time() - t_bin_start
        print(f"Completed {bin_min}-minute sweep in {elapsed_bin:.1f}s.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 14: Time-varying memory analysis.")
    parser.add_argument(
        "--bin-minutes",
        nargs="+",
        type=int,
        default=[60, 15],
        help="Bin widths in minutes (default: 60 15)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MEM",
        choices=["MEM", "MEM+SBM"],
        help="Model to fit (MEM or MEM+SBM)",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=1500,
        help="Sampled TRPs per day (default: 1500)",
    )
    args = parser.parse_args()

    run_time_varying_experiment(
        bin_minutes_list=args.bin_minutes,
        model=args.model,
        num_paths=args.num_paths,
    )

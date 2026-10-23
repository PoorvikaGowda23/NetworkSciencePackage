"""
experiments/step17_sir.py
=========================
Step 17: SIR Epidemic Spreading Dynamics on Real vs. Time-Shuffled Networks.

This experiment evaluates the functional consequence of temporal ordering and
contact memory by simulating an SIR epidemic spreading process on empirical
temporal networks and comparing against their time-shuffled null models:
  1. For each dataset, loads the contact sequence and builds the Day 1 temporal
     graph g_real.
  2. Calibrates transmission rate beta via calibrate_rate so the mean outbreak
     on g_real is in the sensitive critical transition regime (~30% to 50%).
  3. Constructs the time-shuffled null graph g_shuff = time_shuffle_graph(g_real),
     which randomizes the chronological sequence of snapshots within the day
     (destroying temporal order and contact memory) while preserving each snapshot's
     exact internal structure.
  4. Runs N_SIR_RUNS independent simulations on g_real and g_shuff using the exact
     same transmission rate and recovery duration.
  5. Extracts and plots:
     (a) Mean final outbreak size (Real vs. Shuffled)
     (b) Mean time to infection peak (Real vs. Shuffled)
     (c) Empirical outbreak size vs. inferred memory parameter p across datasets.
  6. Saves results table:
       results/step17_sir_results.csv
     and figures:
       results/step17_sir_outbreak_comparison.png
       results/step17_sir_time_to_peak.png
       results/step17_sir_outbreak_vs_memory.png
       results/step17_sir_summary.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make package root and src importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    N_SIR_RUNS,
    RANDOM_SEED,
    RESULTS_DIR,
    SIR_CALIBRATION_RUNS,
    SIR_RECOVERY_SNAPSHOTS,
    SIR_TARGET_OUTBREAK,
    T_RES_SECONDS,
)
from memory_trp.aggregation import build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.nulls import time_shuffle_graph
from memory_trp.plotting import (
    plot_sir_outbreak_comparison,
    plot_sir_outbreak_vs_memory,
    plot_sir_summary_panel,
    plot_sir_time_to_peak,
)
from memory_trp.sir import calibrate_rate, simulate_sir


# All 8 empirical datasets
TARGET_DATASETS: list[tuple[str, str]] = [
    ("Primary School",   "primaryschool"),
    ("High School 2011", "hs2011"),
    ("High School 2012", "hs2012"),
    ("High School 2013", "hs2013"),
    ("Workplace",        "workplace"),
    ("Hospital",         "hospital"),
    ("Conference",       "conference"),
    ("Malawi",           "malawi"),
]


def _load_inferred_memory_map() -> dict[str, float]:
    """Load precomputed empirical memory parameter p (m=10) from Step 15 results.

    Returns a dict mapping dataset_key -> p_real. Falls back to default values
    if the CSV is not found.
    """
    csv_path = RESULTS_DIR / "step15_null_comparison_m10.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return dict(zip(df["dataset_key"], df["p_real"]))

    # Fallback to literature / previously computed approximate values at m=10
    return {
        "primaryschool": 0.511,
        "hs2011": 0.535,
        "hs2012": 0.647,
        "hs2013": 0.643,
        "workplace": 0.607,
        "hospital": 0.642,
        "conference": 0.555,
        "malawi": 0.974,
    }


def run_sir_experiments(
    datasets: list[tuple[str, str]] = TARGET_DATASETS,
    n_runs: int = N_SIR_RUNS,
    recovery_snapshots: int = SIR_RECOVERY_SNAPSHOTS,
    target_outbreak: float = SIR_TARGET_OUTBREAK,
    calib_runs: int = SIR_CALIBRATION_RUNS,
    seed: int = RANDOM_SEED,
) -> None:
    """Run SIR epidemic simulations on empirical and time-shuffled networks.

    Parameters
    ----------
    datasets : list of (display_name, dataset_key)
        Datasets to evaluate.
    n_runs : int
        Number of independent SIR trajectories per graph.
    recovery_snapshots : int
        Snapshots an infected node remains infectious.
    target_outbreak : float
        Target final outbreak fraction for calibration (0.30 - 0.50).
    calib_runs : int
        Number of test simulations per calibration trial.
    seed : int
        Master random seed for reproducibility.
    """
    print("=" * 72, flush=True)
    print("Step 17 -- SIR Epidemic Spreading Dynamics: Real vs. Time-Shuffled", flush=True)
    print(f"Runs/graph: {n_runs} | Recovery: {recovery_snapshots} snapshots ({recovery_snapshots * T_RES_SECONDS / 60:.1f} min)", flush=True)
    print(f"Target outbreak: {target_outbreak * 100:.0f}% | Calib runs: {calib_runs} | Seed: {seed}", flush=True)
    print("=" * 72, flush=True)

    rng = np.random.default_rng(seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p_mem_map = _load_inferred_memory_map()

    summary_rows: list[dict] = []

    for ds_name, ds_key in datasets:
        t0 = time.time()
        print(f"\n[{ds_name}]", flush=True)

        # 1. Load contacts and labels
        contacts, labels = load_contacts(ds_key)
        days = split_by_day(contacts, ds_key)
        n = len(labels) if labels is not None else int(contacts[["i", "j"]].values.max() + 1)

        # Day 1 graph
        g_real = build_temporal_graph(
            days[0], n, labels, t_res=T_RES_SECONDS, name=f"{ds_key}_day1"
        )
        n_snaps = len(g_real.snapshots)
        print(f"  Nodes: {n} | Snapshots: {n_snaps:,}", flush=True)

        # 2. Calibrate transmission rate on real network
        calib_rate = calibrate_rate(
            graph=g_real,
            target_final_fraction=target_outbreak,
            recovery_snapshots=recovery_snapshots,
            n_calib_runs=calib_runs,
            rng=rng,
        )
        print(f"  Calibrated transmission rate beta: {calib_rate:.5f}", flush=True)

        # 3. Create time-shuffled null graph
        g_shuff = time_shuffle_graph(g_real, rng=rng)

        # 4. Simulate SIR on Real Graph
        outbreaks_real: list[float] = []
        peaks_real: list[int] = []

        for _ in range(n_runs):
            res_real = simulate_sir(
                graph=g_real,
                transmission_rate=calib_rate,
                recovery_snapshots=recovery_snapshots,
                rng=rng,
            )
            outbreaks_real.append(res_real.final_fraction)
            peaks_real.append(res_real.time_to_peak)

        # 5. Simulate SIR on Time-Shuffled Graph (same rate and recovery)
        outbreaks_shuff: list[float] = []
        peaks_shuff: list[int] = []

        for _ in range(n_runs):
            res_shuff = simulate_sir(
                graph=g_shuff,
                transmission_rate=calib_rate,
                recovery_snapshots=recovery_snapshots,
                rng=rng,
            )
            outbreaks_shuff.append(res_shuff.final_fraction)
            peaks_shuff.append(res_shuff.time_to_peak)

        # 6. Compute statistics
        mean_out_real = float(np.mean(outbreaks_real))
        std_out_real  = float(np.std(outbreaks_real))
        mean_out_shuff = float(np.mean(outbreaks_shuff))
        std_out_shuff  = float(np.std(outbreaks_shuff))

        mean_peak_real = float(np.mean(peaks_real))
        std_peak_real  = float(np.std(peaks_real))
        mean_peak_shuff = float(np.mean(peaks_shuff))
        std_peak_shuff  = float(np.std(peaks_shuff))

        p_mem = p_mem_map.get(ds_key, 0.50)

        print(
            f"  Outbreak: Real = {mean_out_real*100:.1f}% (+/-{std_out_real*100:.1f}%) | "
            f"Shuffled = {mean_out_shuff*100:.1f}% (+/-{std_out_shuff*100:.1f}%)",
            flush=True,
        )
        print(
            f"  Peak time: Real = {mean_peak_real:.1f} snaps | "
            f"Shuffled = {mean_peak_shuff:.1f} snaps",
            flush=True,
        )

        summary_rows.append({
            "dataset": ds_name,
            "dataset_key": ds_key,
            "n_nodes": n,
            "n_snapshots": n_snaps,
            "calibrated_rate": calib_rate,
            "p_memory": p_mem,
            "mean_outbreak_real": mean_out_real,
            "std_outbreak_real": std_out_real,
            "mean_outbreak_shuff": mean_out_shuff,
            "std_outbreak_shuff": std_out_shuff,
            "mean_time_to_peak_real": mean_peak_real,
            "std_time_to_peak_real": std_peak_real,
            "mean_time_to_peak_shuff": mean_peak_shuff,
            "std_time_to_peak_shuff": std_peak_shuff,
        })

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s", flush=True)

    # -----------------------------------------------------------------------
    # Save CSV Results
    # -----------------------------------------------------------------------
    df_results = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / "step17_sir_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n{'=' * 72}", flush=True)
    print(f"Saved results table -> {csv_path}", flush=True)

    # -----------------------------------------------------------------------
    # Generate Plots
    # -----------------------------------------------------------------------
    # (a) Mean final outbreak size: Real vs. Shuffled
    plot_a = RESULTS_DIR / "step17_sir_outbreak_comparison.png"
    plot_sir_outbreak_comparison(df_results, output_path=plot_a)
    print(f"Saved plot (a) -> {plot_a}", flush=True)

    # (b) Mean time to peak: Real vs. Shuffled
    plot_b = RESULTS_DIR / "step17_sir_time_to_peak.png"
    plot_sir_time_to_peak(df_results, output_path=plot_b)
    print(f"Saved plot (b) -> {plot_b}", flush=True)

    # (c) Outbreak size vs. inferred memory p
    plot_c = RESULTS_DIR / "step17_sir_outbreak_vs_memory.png"
    plot_sir_outbreak_vs_memory(df_results, output_path=plot_c)
    print(f"Saved plot (c) -> {plot_c}", flush=True)

    # Combined 3-panel summary
    plot_summary = RESULTS_DIR / "step17_sir_summary.png"
    plot_sir_summary_panel(df_results, output_path=plot_summary)
    print(f"Saved 3-panel summary -> {plot_summary}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 17: SIR epidemic spreading on real vs. time-shuffled temporal networks."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=N_SIR_RUNS,
        help=f"Number of simulation runs per graph (default: {N_SIR_RUNS})",
    )
    parser.add_argument(
        "--recovery-snapshots",
        type=int,
        default=SIR_RECOVERY_SNAPSHOTS,
        help=f"Recovery duration in snapshots (default: {SIR_RECOVERY_SNAPSHOTS})",
    )
    parser.add_argument(
        "--target-outbreak",
        type=float,
        default=SIR_TARGET_OUTBREAK,
        help=f"Target outbreak fraction for calibration (default: {SIR_TARGET_OUTBREAK})",
    )
    parser.add_argument(
        "--calib-runs",
        type=int,
        default=SIR_CALIBRATION_RUNS,
        help=f"Number of runs per calibration iteration (default: {SIR_CALIBRATION_RUNS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    run_sir_experiments(
        n_runs=args.runs,
        recovery_snapshots=args.recovery_snapshots,
        target_outbreak=args.target_outbreak,
        calib_runs=args.calib_runs,
        seed=args.seed,
    )

"""
experiments/step18_interventions.py
===================================
Step 18: Targeted Isolation Interventions on Temporal Contact Networks.

Evaluates how network topology, temporal activity, and contact memory can be
exploited to design effective epidemic containment strategies under a fixed
isolation budget (default k = 10% of nodes).

Compares 5 intervention strategies:
  1. None: Baseline epidemic without any intervention.
  2. Random Nodes: Uniform random isolation of k% of nodes.
  3. Highest Total Contact Time: Targeted isolation of the most active nodes
     (top k% by aggregate edge weight).
  4. Highest Node-Level Return Rate: Targeted isolation of nodes with the highest
     empirical return probability in transition records (nodes with return rate
     r_u = sum(in_mem) / count(cur=u), with at least 30 records).
  5. Closed Group (labeled datasets only): Closing an entire community, such as
     a whole school class, hospital role, or workplace department, chosen
     closest in size to the k% budget.

For each dataset:
  - Runs N_SIR_RUNS independent simulations per strategy.
  - Computes mean final outbreak fraction and standard deviation.
  - Exports a bar chart with error bars per dataset and across all datasets.
  - Writes results table to results/step18_interventions_all.csv.
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
    DATA_DIR,
    INTERVENTION_BUDGET_PERCENT,
    INTERVENTION_MIN_RECORDS,
    N_SIR_RUNS,
    PATH_LENGTH,
    RANDOM_SEED,
    RESULTS_DIR,
    SIR_CALIBRATION_RUNS,
    SIR_RECOVERY_SNAPSHOTS,
    SIR_TARGET_OUTBREAK,
    STEP18_M,
    STEP18_NUM_PATHS,
    T_RES_SECONDS,
)
from memory_trp.aggregation import build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.plotting import (
    plot_interventions_barchart,
    plot_interventions_grid,
)
from memory_trp.records import build_records
from memory_trp.sir import (
    calibrate_rate,
    get_community_nodes,
    get_highest_contact_nodes,
    get_highest_return_nodes,
    get_random_nodes,
    simulate_sir,
)
from memory_trp.trp import generate_many_trps


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


def _load_group_name(dataset_key: str, group_idx: int) -> str:
    """Helper to retrieve human-readable group name for a community index."""
    try:
        if dataset_key == "primaryschool":
            df = pd.read_csv(DATA_DIR / "primaryschool.csv", sep=",", header=None, usecols=[3])
            unique = sorted(df.iloc[:, 0].dropna().astype(str).unique().tolist())
            return unique[group_idx] if group_idx < len(unique) else f"Group {group_idx}"
        elif dataset_key in ("hs2011", "hs2012"):
            df = pd.read_csv(DATA_DIR / f"{dataset_key}.csv", sep="\t", header=None, usecols=[3])
            unique = sorted(df.iloc[:, 0].dropna().astype(str).unique().tolist())
            return unique[group_idx] if group_idx < len(unique) else f"Group {group_idx}"
        elif dataset_key == "hs2013":
            df = pd.read_csv(DATA_DIR / "hs2013.csv", sep=" ", header=None, usecols=[3])
            unique = sorted(df.iloc[:, 0].dropna().astype(str).unique().tolist())
            return unique[group_idx] if group_idx < len(unique) else f"Group {group_idx}"
        elif dataset_key == "hospital":
            df = pd.read_csv(DATA_DIR / "hospital.csv", sep="\t", header=None, usecols=[3])
            unique = sorted(df.iloc[:, 0].dropna().astype(str).unique().tolist())
            return unique[group_idx] if group_idx < len(unique) else f"Group {group_idx}"
        elif dataset_key == "workplace":
            df = pd.read_csv(DATA_DIR / "workplace_meta.csv", header=None, names=["id", "dept"])
            unique = sorted(df["dept"].dropna().astype(str).unique().tolist())
            return unique[group_idx] if group_idx < len(unique) else f"Group {group_idx}"
    except Exception:
        pass
    return f"Group {group_idx}"


def run_interventions(
    datasets: list[tuple[str, str]] = TARGET_DATASETS,
    k_percent: float = INTERVENTION_BUDGET_PERCENT,
    n_runs: int = N_SIR_RUNS,
    recovery_snapshots: int = SIR_RECOVERY_SNAPSHOTS,
    target_outbreak: float = SIR_TARGET_OUTBREAK,
    m: int = STEP18_M,
    num_paths: int = STEP18_NUM_PATHS,
    min_records: int = INTERVENTION_MIN_RECORDS,
    seed: int = RANDOM_SEED,
) -> None:
    """Run targeted isolation intervention experiments across datasets.

    Parameters
    ----------
    datasets : list of (display_name, dataset_key)
        Datasets to test.
    k_percent : float
        Intervention budget as a percentage of total nodes (default: 10.0%).
    n_runs : int
        Simulation runs per strategy.
    recovery_snapshots : int
        Recovery duration in snapshots.
    target_outbreak : float
        Target final outbreak for rate calibration.
    m : int
        Look-back memory horizon m for transition records.
    num_paths : int
        Paths sampled to estimate node-level return rates.
    min_records : int
        Minimum record count required for return rate calculation.
    seed : int
        Master random seed for reproducibility.
    """
    print("=" * 72, flush=True)
    print("Step 18 -- Targeted Isolation Interventions on Temporal Networks", flush=True)
    print(f"Budget: {k_percent:.1f}% of nodes | Runs/strategy: {n_runs} | Seed: {seed}", flush=True)
    print("=" * 72, flush=True)

    rng = np.random.default_rng(seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    dataset_dfs: dict[str, pd.DataFrame] = {}

    for ds_name, ds_key in datasets:
        t0 = time.time()
        print(f"\n[{ds_name}]", flush=True)

        # 1. Load contacts and build Day 1 graph
        contacts, labels = load_contacts(ds_key)
        days = split_by_day(contacts, ds_key)
        n = len(labels) if labels is not None else int(contacts[["i", "j"]].values.max() + 1)

        g_real = build_temporal_graph(
            days[0], n, labels, t_res=T_RES_SECONDS, name=f"{ds_key}_day1"
        )
        k_nodes = max(1, int(round(n * k_percent / 100.0)))
        print(f"  Nodes n={n} | Budget k={k_percent:.1f}% ({k_nodes} nodes)", flush=True)

        # 2. Calibrate transmission rate
        calib_rate = calibrate_rate(
            graph=g_real,
            target_final_fraction=target_outbreak,
            recovery_snapshots=recovery_snapshots,
            n_calib_runs=SIR_CALIBRATION_RUNS,
            rng=rng,
        )
        print(f"  Calibrated transmission rate beta: {calib_rate:.5f}", flush=True)

        # 3. Sample paths and compute records for return rate calculation
        paths = generate_many_trps(g_real, num_paths=num_paths, length=PATH_LENGTH, rng=rng)
        records = build_records(paths, m=m)

        # 4. Formulate intervention strategies
        strategies: list[tuple[str, list[int]]] = []

        # Strategy 1: None
        strategies.append(("None", []))

        # Strategy 2: Random nodes
        rand_nodes = get_random_nodes(n=n, budget=k_nodes, rng=rng)
        strategies.append(("Random", rand_nodes))

        # Strategy 3: Highest total contact time
        contact_nodes = get_highest_contact_nodes(graph=g_real, budget=k_nodes)
        strategies.append(("Contact Time", contact_nodes))

        # Strategy 4: Highest node-level return rate
        return_nodes = get_highest_return_nodes(
            records=records, budget=k_nodes, n=n, min_records=min_records
        )
        strategies.append(("Return Rate", return_nodes))

        # Strategy 5: Closing one whole community (if labels available)
        if labels is not None:
            grp_idx, comm_nodes = get_community_nodes(labels, target_budget=k_nodes)
            grp_name = _load_group_name(ds_key, grp_idx)
            strat_label = f"Close Group ({grp_name})"
            strategies.append((strat_label, comm_nodes))

        # 5. Run simulations for each strategy
        strat_results: list[dict] = []
        baseline_mean = 0.0

        print(f"  {'Strategy':<26} {'Removed':>8} {'Mean Outbreak':>15} {'Std Dev':>10} {'Reduction':>12}")
        print(f"  {'-' * 75}")

        for strat_name, rem_nodes in strategies:
            outbreaks: list[float] = []
            for _ in range(n_runs):
                res = simulate_sir(
                    graph=g_real,
                    transmission_rate=calib_rate,
                    recovery_snapshots=recovery_snapshots,
                    rng=rng,
                    removed_nodes=rem_nodes,
                )
                outbreaks.append(res.final_fraction)

            mean_out = float(np.mean(outbreaks))
            std_out = float(np.std(outbreaks))

            if strat_name == "None":
                baseline_mean = mean_out
                reduction_pct = 0.0
            else:
                reduction_pct = (
                    (baseline_mean - mean_out) / baseline_mean * 100.0
                    if baseline_mean > 0 else 0.0
                )

            n_r = len(rem_nodes)
            print(
                f"  {strat_name:<26} {n_r:>8d} {mean_out*100:>14.1f}% "
                f"{std_out*100:>9.1f}% {reduction_pct:>11.1f}%"
            )

            row_data = {
                "dataset": ds_name,
                "dataset_key": ds_key,
                "n_nodes": n,
                "k_budget_percent": k_percent,
                "calibrated_rate": calib_rate,
                "strategy": strat_name,
                "n_removed": n_r,
                "fraction_removed": float(n_r / n),
                "mean_outbreak": mean_out,
                "std_outbreak": std_out,
                "reduction_vs_none_percent": reduction_pct,
            }
            strat_results.append(row_data)
            all_rows.append(row_data)

        # 6. Save per-dataset plot
        df_ds = pd.DataFrame(strat_results)
        dataset_dfs[ds_name] = df_ds

        plot_path = RESULTS_DIR / f"step18_interventions_{ds_key}.png"
        plot_interventions_barchart(df_ds, dataset_name=ds_name, output_path=plot_path)
        print(f"  Saved plot -> {plot_path}", flush=True)

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s", flush=True)

    # -----------------------------------------------------------------------
    # Save Combined CSV & Multi-Dataset Grid Plot
    # -----------------------------------------------------------------------
    df_all = pd.DataFrame(all_rows)
    csv_all = RESULTS_DIR / "step18_interventions_all.csv"
    df_all.to_csv(csv_all, index=False)
    print(f"\n{'=' * 72}", flush=True)
    print(f"Saved combined CSV -> {csv_all}", flush=True)

    grid_path = RESULTS_DIR / "step18_interventions_grid.png"
    plot_interventions_grid(dataset_dfs, output_path=grid_path)
    print(f"Saved multi-dataset grid plot -> {grid_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 18: Targeted isolation interventions on temporal contact networks."
    )
    parser.add_argument(
        "--k-percent",
        type=float,
        default=INTERVENTION_BUDGET_PERCENT,
        help=f"Intervention budget percentage of nodes (default: {INTERVENTION_BUDGET_PERCENT})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=N_SIR_RUNS,
        help=f"Number of simulation runs per strategy (default: {N_SIR_RUNS})",
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
        "--min-records",
        type=int,
        default=INTERVENTION_MIN_RECORDS,
        help=f"Min records for node-level return rate (default: {INTERVENTION_MIN_RECORDS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    run_interventions(
        k_percent=args.k_percent,
        n_runs=args.runs,
        recovery_snapshots=args.recovery_snapshots,
        target_outbreak=args.target_outbreak,
        min_records=args.min_records,
        seed=args.seed,
    )

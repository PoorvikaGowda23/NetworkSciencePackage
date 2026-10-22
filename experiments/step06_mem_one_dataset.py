"""
experiments/step06_mem_one_dataset.py
=====================================
Pipeline for estimating the MEM model on the primary school dataset
and comparing against an Erdős-Rényi null model.

Workflow:
1. Load primaryschool dataset (t_res = 20s).
2. Split contacts into separate observation days.
3. For each day, build empirical and Erdős-Rényi null temporal graphs.
4. Sample NUM_PATHS paths per day for both empirical and null graphs.
5. Pool paths across days for each condition.
6. For each memory horizon m in MEMORY_HORIZONS, build records and fit MEM.
7. Plot inferred memory parameter p vs horizon m and save to results/.
"""

import sys
from pathlib import Path

# Ensure package modules and config are discoverable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import (
    MEMORY_HORIZONS,
    NUM_PATHS,
    PATH_LENGTH,
    RANDOM_SEED,
    RESULTS_DIR,
    T_RES_SECONDS,
)
from memory_trp.aggregation import build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.models import fit_mem_fixed_point
from memory_trp.nulls import er_null_graph
from memory_trp.plotting import plot_memory_curves
from memory_trp.records import build_records
from memory_trp.trp import generate_many_trps


def run_step06_pipeline() -> None:
    """Run the MEM estimation experiment on primary school and its ER null."""
    print("=" * 60)
    print("Step 6 Experiment: MEM Model on Primary School vs. ER Null")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)

    # 1. Load dataset
    print("\n[1/5] Loading primaryschool dataset...")
    contacts, labels = load_contacts("primaryschool")
    n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
    print(f"      Loaded {len(contacts)} contacts across {n} nodes.")

    # 2. Split by day
    print("\n[2/5] Partitioning contacts into daily observation periods...")
    days = split_by_day(contacts, "primaryschool")
    print(f"      Found {len(days)} observation days.")

    # 3. Build graphs and generate TRPs per day
    print(
        f"\n[3/5] Generating {NUM_PATHS:,} paths (length {PATH_LENGTH}) per day..."
    )
    all_emp_paths = []
    all_null_paths = []

    for d_idx, day_contacts in enumerate(days):
        print(f"      --- Day {d_idx + 1} ({len(day_contacts):,} contacts) ---")

        # Empirical graph
        g_emp = build_temporal_graph(
            day_contacts,
            n=n,
            labels=labels,
            t_res=T_RES_SECONDS,
            name=f"primary_d{d_idx + 1}",
        )

        # Erdős-Rényi null graph
        g_null = er_null_graph(g_emp, rng)

        print(f"      Sampling empirical paths (day {d_idx + 1})...")
        emp_paths = generate_many_trps(
            g_emp, num_paths=NUM_PATHS, length=PATH_LENGTH, rng=rng
        )
        all_emp_paths.extend(emp_paths)

        print(f"      Sampling null paths (day {d_idx + 1})...")
        null_paths = generate_many_trps(
            g_null, num_paths=NUM_PATHS, length=PATH_LENGTH, rng=rng
        )
        all_null_paths.extend(null_paths)

    total_paths = len(all_emp_paths)
    print(f"\n      Total pooled paths per condition: {total_paths:,}")

    # 4. Fit MEM for each memory horizon m
    print("\n[4/5] Fitting MEM model across horizons...")
    p_emp_list: list[float] = []
    p_null_list: list[float] = []

    print(f"{'m':<5} | {'p (Empirical)':<15} | {'p (ER Null)':<15} | Status")
    print("-" * 50)

    for m in MEMORY_HORIZONS:
        # Empirical records and fit
        rec_emp = build_records(all_emp_paths, m=m)
        fit_emp = fit_mem_fixed_point(rec_emp, n=n)
        p_emp_list.append(fit_emp.p)

        # Null records and fit
        rec_null = build_records(all_null_paths, m=m)
        fit_null = fit_mem_fixed_point(rec_null, n=n)
        p_null_list.append(fit_null.p)

        print(
            f"{m:<5} | {fit_emp.p:<15.4f} | {fit_null.p:<15.4f} | "
            f"converged: {fit_emp.converged and fit_null.converged}"
        )

    # 5. Plot and save results
    print("\n[5/5] Generating comparison plot...")
    output_path = RESULTS_DIR / "step06_primaryschool_mem_vs_null.png"
    plot_memory_curves(
        m_values=MEMORY_HORIZONS,
        p_emp=p_emp_list,
        p_null=p_null_list,
        dataset_name="Primary School (t_res = 20s)",
        output_path=output_path,
    )
    print(f"      Plot successfully saved to: {output_path}")
    print("=" * 60)
    print("Step 6 Complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_step06_pipeline()

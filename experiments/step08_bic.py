"""
experiments/step08_bic.py
=========================
Pipeline for comparing the MEM and MEM+SBM models via BIC on six labeled datasets:
  1. Primary School ('primaryschool')
  2. High School 1 ('hs2011')
  3. High School 2 ('hs2012')
  4. High School 3 ('hs2013')
  5. Office ('workplace')
  6. Hospital ('hospital')

Workflow:
1. For each dataset, load contacts and community labels.
2. Partition contacts into daily observation periods.
3. Build temporal graphs and sample time-respecting paths per day.
4. Pool paths across days for each dataset.
5. Compute the empirical community interaction matrix as initial C.
6. For each memory horizon m in MEMORY_HORIZONS, construct records and fit:
   - Plain MEM model (fit_mem_fixed_point)
   - MEM+SBM model (fit_mem_sbm)
7. Save numerical results to a CSV table in results/.
8. Generate a 2x3 panel plot (Figure 2 style) comparing BIC vs m for both models.
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
    BIC_NUM_PATHS,
    MEMORY_HORIZONS,
    P_INIT,
    PATH_LENGTH,
    RANDOM_SEED,
    RESULTS_DIR,
    T_RES_SECONDS,
)
from memory_trp.aggregation import TemporalGraph, build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.models import (
    community_interaction_matrix,
    fit_mem_fixed_point,
    fit_mem_sbm,
)
from memory_trp.plotting import plot_bic_panels
from memory_trp.records import build_records
from memory_trp.trp import generate_many_trps

# The six labeled datasets specified in the paper
DATASETS = [
    ("Primary", "primaryschool"),
    ("High School 1", "hs2011"),
    ("High School 2", "hs2012"),
    ("High School 3", "hs2013"),
    ("Office", "workplace"),
    ("Hospital", "hospital"),
]


def run_step08_pipeline() -> None:
    """Run the multi-dataset BIC comparison pipeline for MEM vs MEM+SBM."""
    print("=" * 70, flush=True)
    print("Step 8 Experiment: Model Selection (BIC) for MEM vs. MEM+SBM", flush=True)
    print(f"Memory horizons: {MEMORY_HORIZONS} | Base paths per day: {BIC_NUM_PATHS:,}", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.default_rng(RANDOM_SEED)

    records_table: list[dict[str, object]] = []
    panel_data: dict[str, dict[str, list[float] | int]] = {}

    start_total = time.time()

    for d_idx, (display_name, dataset_key) in enumerate(DATASETS, 1):
        print(f"\n[{d_idx}/6] Processing '{display_name}' ({dataset_key})...", flush=True)
        t0 = time.time()

        # 1. Load contacts and community labels
        contacts, labels = load_contacts(dataset_key)
        if labels is None:
            raise ValueError(f"Dataset '{dataset_key}' does not have community labels.")

        n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
        k = int(labels.max()) + 1
        print(f"      Nodes n={n}, Communities K={k}, Total contacts={len(contacts):,}", flush=True)

        # 2. Partition into observation days
        days = split_by_day(contacts, dataset_key)
        print(f"      Observation days: {len(days)}", flush=True)

        # 3. Build daily temporal graphs and sample TRPs
        all_paths = []
        all_snapshots = []

        num_paths_day = BIC_NUM_PATHS if len(days) > 1 else max(1000, BIC_NUM_PATHS * 2)

        for day_idx, day_df in enumerate(days):
            g_day = build_temporal_graph(
                day_df,
                n=n,
                labels=labels,
                t_res=T_RES_SECONDS,
                name=f"{dataset_key}_d{day_idx + 1}",
            )
            all_snapshots.extend(g_day.snapshots)

            day_paths = generate_many_trps(
                g_day,
                num_paths=num_paths_day,
                length=PATH_LENGTH,
                rng=rng,
            )
            all_paths.extend(day_paths)

        print(f"      Sampled {len(all_paths):,} total paths across {len(days)} days.", flush=True)

        # 4. Compute empirical initial interaction matrix across all snapshots
        pooled_graph = TemporalGraph(
            name=f"{dataset_key}_pooled",
            n=n,
            t_res=T_RES_SECONDS,
            snapshots=all_snapshots,
            labels=labels,
            start_time=0,
        )
        init_C = community_interaction_matrix(pooled_graph, labels=labels)

        # 5. Fit MEM and MEM+SBM for each horizon m
        bic_mem_list: list[float] = []
        bic_sbm_list: list[float] = []

        print(f"      {'m':<5} | {'p (MEM)':<8} | {'BIC (MEM)':<14} | {'p (SBM)':<8} | {'BIC (SBM)':<14} | {'delta BIC':<10}", flush=True)
        print("      " + "-" * 72, flush=True)

        for m in MEMORY_HORIZONS:
            records = build_records(all_paths, m=m)

            # Fit plain MEM
            res_mem = fit_mem_fixed_point(records, n=n)
            # Fit MEM+SBM
            res_sbm = fit_mem_sbm(records, labels=labels, init_C=init_C, p0=P_INIT)

            delta_bic = res_mem.bic - res_sbm.bic  # Positive: MEM+SBM preferred

            bic_mem_list.append(res_mem.bic)
            bic_sbm_list.append(res_sbm.bic)

            print(
                f"      {m:<5} | {res_mem.p:<8.4f} | {res_mem.bic:<14.2f} | "
                f"{res_sbm.p:<8.4f} | {res_sbm.bic:<14.2f} | {delta_bic:<10.2f}",
                flush=True,
            )

            records_table.append({
                "dataset": display_name,
                "dataset_key": dataset_key,
                "n_nodes": n,
                "k_communities": k,
                "m": m,
                "n_records": len(records),
                "p_mem": res_mem.p,
                "loglik_mem": res_mem.loglik,
                "n_params_mem": res_mem.n_params,
                "bic_mem": res_mem.bic,
                "p_sbm": res_sbm.p,
                "loglik_sbm": res_sbm.loglik,
                "n_params_sbm": res_sbm.n_params,
                "bic_sbm": res_sbm.bic,
                "delta_bic": delta_bic,
            })

        panel_data[display_name] = {
            "bic_mem": bic_mem_list,
            "bic_sbm": bic_sbm_list,
            "k": k,
            "n": n,
        }

        print(f"      Completed {display_name} in {time.time() - t0:.1f}s.", flush=True)

    # 6. Save CSV of numbers
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "step08_bic_results.csv"
    df_results = pd.DataFrame(records_table)
    df_results.to_csv(csv_path, index=False)
    print(f"\n[+] Saved numerical results to: {csv_path}", flush=True)

    # 7. Generate Figure 2 comparison plot
    plot_path = RESULTS_DIR / "step08_bic_comparison.png"
    plot_bic_panels(
        dataset_results=panel_data,
        m_values=MEMORY_HORIZONS,
        output_path=plot_path,
    )
    print(f"[+] Saved comparison figure to: {plot_path}", flush=True)

    total_time = time.time() - start_total
    print("\n" + "=" * 70, flush=True)
    print(f"Step 8 Pipeline Complete in {total_time:.1f}s ({total_time / 60:.2f} min).", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_step08_pipeline()

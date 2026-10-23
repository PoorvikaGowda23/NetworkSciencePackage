"""
experiments/step09_nulls.py
===========================
Pipeline for assessing statistical significance of memory in temporal networks
via randomized null models (Figure 3 in Guerrini et al., EPJ Data Science 2026).

Workflow:
1. For each dataset:
   - Load contacts and optional community labels.
   - Partition contacts into daily observation periods.
   - Compute empirical MEM and (if labeled) MEM+SBM curves.
   - Run N_NULL_REALIZATIONS null model realizations:
       * SBM null (sbm_null_graph) for labeled datasets, fitted with MEM+SBM.
       * Erdős-Rényi null (er_null_graph) for unlabeled datasets, fitted with MEM.
   - Compute mean and standard deviation of p across realizations for each horizon m.
2. Cache generated paths and fit records to disk (RESULTS_DIR/cache_step09) for fast reruns.
3. Save numerical results table to results/step09_null_results.csv.
4. Generate a multi-panel Figure 3 comparison plot with shaded error bands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

# Ensure package modules and config are discoverable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BIC_NUM_PATHS,
    CACHE_DIR_STEP09,
    MEMORY_HORIZONS,
    N_NULL_REALIZATIONS,
    P_INIT,
    PATH_LENGTH,
    QUICK_NULL_REALIZATIONS,
    QUICK_NUM_PATHS,
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
from memory_trp.nulls import er_null_graph, sbm_null_graph
from memory_trp.records import build_records
from memory_trp.trp import generate_many_trps

# The 8 datasets from Table 1 / Figure 3: 6 labeled and 2 unlabeled
DATASETS = [
    ("Primary", "primaryschool", True),
    ("High School 1", "hs2011", True),
    ("High School 2", "hs2012", True),
    ("High School 3", "hs2013", True),
    ("Office", "workplace", True),
    ("Hospital", "hospital", True),
    ("Conference", "conference", False),
    ("Malawi", "malawi", False),
]


def load_or_generate_empirical_paths(
    days: list[pd.DataFrame],
    n: int,
    labels: np.ndarray | None,
    dataset_key: str,
    num_paths_day: int,
    rng: np.random.Generator,
    recompute: bool = False,
) -> tuple[list[tuple[list[int], list[int]]], list[sp.csr_matrix]]:
    """Load cached empirical paths or generate and cache them to disk.

    Parameters
    ----------
    days : list[pd.DataFrame]
        Daily contact dataframes.
    n : int
        Total number of nodes.
    labels : np.ndarray | None
        Community labels per node, or None.
    dataset_key : str
        Short identifier for the dataset.
    num_paths_day : int
        Number of paths sampled per day.
    rng : np.random.Generator
        Random number generator.
    recompute : bool, default=False
        If True, ignore cache and recompute.

    Returns
    -------
    tuple[list[tuple[list[int], list[int]]], list[sp.csr_matrix]]
        All pooled paths and list of all daily snapshots.
    """
    CACHE_DIR_STEP09.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR_STEP09 / f"emp_paths_{dataset_key}_{num_paths_day}.pkl"

    if cache_path.exists() and not recompute:
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
            return cached_data["paths"], cached_data["snapshots"]

    all_paths = []
    all_snapshots = []

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

    with open(cache_path, "wb") as f:
        pickle.dump({"paths": all_paths, "snapshots": all_snapshots}, f)

    return all_paths, all_snapshots


def compute_or_load_empirical_fits(
    all_paths: list[tuple[list[int], list[int]]],
    all_snapshots: list[sp.csr_matrix],
    n: int,
    labels: np.ndarray | None,
    dataset_key: str,
    is_labeled: bool,
    num_paths_day: int,
    recompute: bool = False,
) -> dict[str, dict[int, float]]:
    """Compute or load empirical MEM and (if labeled) MEM+SBM fits for each horizon m.

    Parameters
    ----------
    all_paths : list[tuple[list[int], list[int]]]
        Empirical paths pooled across days.
    all_snapshots : list[sp.csr_matrix]
        All empirical snapshots across days.
    n : int
        Total node count.
    labels : np.ndarray | None
        Community labels per node, or None.
    dataset_key : str
        Dataset identifier.
    is_labeled : bool
        True if dataset carries community labels.
    num_paths_day : int
        Path sampling budget per day.
    recompute : bool, default=False
        If True, ignore cache.

    Returns
    -------
    dict[str, dict[int, float]]
        Dictionary with keys 'mem' and optionally 'sbm', mapping horizon m to fitted p.
    """
    CACHE_DIR_STEP09.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR_STEP09 / f"emp_fits_{dataset_key}_{num_paths_day}.json"

    if cache_file.exists() and not recompute:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                model: {int(m): float(p) for m, p in fits.items()}
                for model, fits in data.items()
            }

    init_C = None
    if is_labeled and labels is not None:
        pooled_graph = TemporalGraph(
            name=f"{dataset_key}_pooled",
            n=n,
            t_res=T_RES_SECONDS,
            snapshots=all_snapshots,
            labels=labels,
            start_time=0,
        )
        init_C = community_interaction_matrix(pooled_graph, labels=labels)

    mem_fits: dict[int, float] = {}
    sbm_fits: dict[int, float] = {}

    for m in MEMORY_HORIZONS:
        records = build_records(all_paths, m=m)
        res_mem = fit_mem_fixed_point(records, n=n)
        mem_fits[m] = float(res_mem.p)

        if is_labeled and labels is not None:
            res_sbm = fit_mem_sbm(records, labels=labels, init_C=init_C, p0=P_INIT)
            sbm_fits[m] = float(res_sbm.p)

    out: dict[str, dict[int, float]] = {"mem": mem_fits}
    if is_labeled:
        out["sbm"] = sbm_fits

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def run_null_realization(
    days: list[pd.DataFrame],
    n: int,
    labels: np.ndarray | None,
    dataset_key: str,
    is_labeled: bool,
    realiz_idx: int,
    num_paths_day: int,
    rng: np.random.Generator,
    recompute: bool = False,
) -> dict[int, float]:
    """Run a single null model realization, fit the matching model, and return {m: p}.

    Parameters
    ----------
    days : list[pd.DataFrame]
        Daily contact dataframes.
    n : int
        Total number of nodes.
    labels : np.ndarray | None
        Community labels per node.
    dataset_key : str
        Dataset identifier.
    is_labeled : bool
        True if dataset carries community labels.
    realiz_idx : int
        Zero-based realization index.
    num_paths_day : int
        Paths sampled per day.
    rng : np.random.Generator
        Dedicated RNG for this realization.
    recompute : bool, default=False
        If True, ignore cache.

    Returns
    -------
    dict[int, float]
        Dictionary mapping memory horizon m to fitted p.
    """
    CACHE_DIR_STEP09.mkdir(parents=True, exist_ok=True)
    fit_cache_path = (
        CACHE_DIR_STEP09
        / f"null_fit_{dataset_key}_realiz{realiz_idx}_{num_paths_day}.json"
    )

    if fit_cache_path.exists() and not recompute:
        with open(fit_cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return {int(m): float(p) for m, p in raw.items()}

    # Check path cache
    path_cache_path = (
        CACHE_DIR_STEP09
        / f"null_paths_{dataset_key}_realiz{realiz_idx}_{num_paths_day}.pkl"
    )

    all_null_paths = []
    if path_cache_path.exists() and not recompute:
        with open(path_cache_path, "rb") as f:
            all_null_paths = pickle.load(f)
    else:
        for day_idx, day_df in enumerate(days):
            g_day = build_temporal_graph(
                day_df,
                n=n,
                labels=labels,
                t_res=T_RES_SECONDS,
                name=f"{dataset_key}_d{day_idx + 1}",
            )

            # Generate appropriate null graph
            if is_labeled and labels is not None:
                g_null = sbm_null_graph(g_day, labels=labels, rng=rng)
            else:
                g_null = er_null_graph(g_day, rng=rng)

            day_paths = generate_many_trps(
                g_null,
                num_paths=num_paths_day,
                length=PATH_LENGTH,
                rng=rng,
            )
            all_null_paths.extend(day_paths)

        with open(path_cache_path, "wb") as f:
            pickle.dump(all_null_paths, f)

    # Fit matching model for each horizon m
    p_fits: dict[int, float] = {}
    for m in MEMORY_HORIZONS:
        records = build_records(all_null_paths, m=m)
        if is_labeled and labels is not None:
            # Fit MEM+SBM for labeled datasets
            res = fit_mem_sbm(records, labels=labels, p0=P_INIT)
            p_fits[m] = float(res.p)
        else:
            # Fit plain MEM for unlabeled datasets
            res = fit_mem_fixed_point(records, n=n, p0=P_INIT)
            p_fits[m] = float(res.p)

    with open(fit_cache_path, "w", encoding="utf-8") as f:
        json.dump({str(m): p for m, p in p_fits.items()}, f, indent=2)

    return p_fits


def plot_figure3_panels(
    results_dict: dict[str, dict[str, object]],
    m_values: list[int],
    output_path: Path,
) -> None:
    """Plot Figure 3 comparison panels across all datasets with shaded error bands.

    Parameters
    ----------
    results_dict : dict[str, dict[str, object]]
        Dictionary keyed by dataset display name containing:
            - 'is_labeled': bool
            - 'n': int
            - 'k': int or None
            - 'p_emp_mem': list[float]
            - 'p_emp_sbm': list[float] (if labeled)
            - 'p_null_mean': list[float]
            - 'p_null_std': list[float]
    m_values : list[int]
        Memory horizon values.
    output_path : Path
        Path to save output figure.
    """
    n_datasets = len(results_dict)
    n_cols = 4
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows), dpi=300)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (dname, ddata) in enumerate(results_dict.items()):
        ax = axes_flat[idx]
        is_labeled = bool(ddata["is_labeled"])

        # 1. Empirical MEM
        p_emp_mem = [float(x) for x in ddata["p_emp_mem"]]
        ax.plot(
            m_values,
            p_emp_mem,
            marker="s",
            color="#1f77b4",
            linestyle="-",
            linewidth=1.8,
            markersize=5,
            label="MEM (Empirical)",
        )

        # 2. Empirical MEM+SBM (for labeled datasets)
        if is_labeled and "p_emp_sbm" in ddata and ddata["p_emp_sbm"]:
            p_emp_sbm = [float(x) for x in ddata["p_emp_sbm"]]
            ax.plot(
                m_values,
                p_emp_sbm,
                marker="o",
                color="#2ca02c",
                linestyle="-",
                linewidth=1.8,
                markersize=5,
                label="MEM+SBM (Empirical)",
            )

        # 3. Null model curve with shaded standard deviation
        null_mean = np.array(ddata["p_null_mean"], dtype=float)
        null_std = np.array(ddata["p_null_std"], dtype=float)

        if is_labeled:
            null_label = "Null MEM+SBM (SBM null)"
            null_color = "#d62728"
            null_marker = "o"
        else:
            null_label = "Null MEM (ER null)"
            null_color = "#ff7f0e"
            null_marker = "s"

        ax.plot(
            m_values,
            null_mean,
            marker=null_marker,
            color=null_color,
            linestyle="--",
            linewidth=1.8,
            markersize=5,
            markerfacecolor="none",
            label=null_label,
        )

        ax.fill_between(
            m_values,
            np.maximum(0.0, null_mean - null_std),
            null_mean + null_std,
            color=null_color,
            alpha=0.25,
            label=r"$\pm 1\,\sigma$ Null",
        )

        n_val = ddata.get("n", "")
        k_val = ddata.get("k", None)
        title_extra = [f"$n={n_val}$"]
        if k_val is not None:
            title_extra.append(f"$K={k_val}$")
        extra_str = f" ({', '.join(title_extra)})"

        ax.set_title(f"{dname}{extra_str}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Memory length $m$", fontsize=10)
        ax.set_ylabel("Inferred memory $p$", fontsize=10)
        ax.set_xticks(m_values)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=8, loc="upper right")

        # Set consistent y-limits
        y_max = max(max(p_emp_mem), float(np.max(null_mean + null_std)), 0.1) * 1.15
        ax.set_ylim(-0.02, min(1.0, y_max))

    # Hide unused axes
    for idx in range(n_datasets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_step09_pipeline(
    quick: bool = False,
    target_dataset: str | None = None,
    recompute: bool = False,
) -> None:
    """Execute the Step 9 pipeline across datasets and generate Figure 3.

    Parameters
    ----------
    quick : bool, default=False
        If True, use QUICK_NULL_REALIZATIONS (3) and QUICK_NUM_PATHS (300).
    target_dataset : str | None, default=None
        If provided, run only the specified dataset key.
    recompute : bool, default=False
        If True, ignore disk cache and recompute everything.
    """
    n_realizations = QUICK_NULL_REALIZATIONS if quick else N_NULL_REALIZATIONS
    num_paths_base = QUICK_NUM_PATHS if quick else BIC_NUM_PATHS

    mode_label = "QUICK TEST" if quick else "FULL RUN"
    print("=" * 75, flush=True)
    print(f"Step 9 Experiment: Null Models and Memory Significance ({mode_label})", flush=True)
    print(f"Null realizations: {n_realizations} | Paths/day: {num_paths_base:,} | Horizons: {MEMORY_HORIZONS}", flush=True)
    print("=" * 75, flush=True)

    master_rng = np.random.default_rng(RANDOM_SEED)

    datasets_to_run = [
        (name, key, labeled)
        for name, key, labeled in DATASETS
        if target_dataset is None or target_dataset.lower().strip() == key.lower().strip()
    ]

    if not datasets_to_run:
        raise ValueError(f"No matching dataset found for '{target_dataset}'.")

    results_table: list[dict[str, object]] = []
    figure_data: dict[str, dict[str, object]] = {}

    start_total = time.time()

    for d_idx, (display_name, dataset_key, is_labeled) in enumerate(datasets_to_run, 1):
        print(f"\n[{d_idx}/{len(datasets_to_run)}] Processing '{display_name}' ({dataset_key})...", flush=True)
        t0 = time.time()

        contacts, labels = load_contacts(dataset_key)
        n = int(max(contacts["i"].max(), contacts["j"].max())) + 1
        k = int(labels.max()) + 1 if labels is not None else None
        days = split_by_day(contacts, dataset_key)
        num_paths_day = num_paths_base if len(days) > 1 else max(1000, num_paths_base * 2)

        print(f"      Nodes n={n}, Communities K={k}, Days={len(days)}, Contacts={len(contacts):,}", flush=True)

        # 1. Empirical paths & fits
        emp_rng = np.random.default_rng(RANDOM_SEED + d_idx * 100)
        emp_paths, emp_snapshots = load_or_generate_empirical_paths(
            days, n, labels, dataset_key, num_paths_day, emp_rng, recompute=recompute
        )
        emp_fits = compute_or_load_empirical_fits(
            emp_paths, emp_snapshots, n, labels, dataset_key, is_labeled, num_paths_day, recompute=recompute
        )

        p_emp_mem = [emp_fits["mem"][m] for m in MEMORY_HORIZONS]
        p_emp_sbm = [emp_fits["sbm"][m] for m in MEMORY_HORIZONS] if is_labeled else []

        print(f"      Empirical MEM p: {[round(x, 4) for x in p_emp_mem]}", flush=True)
        if is_labeled:
            print(f"      Empirical SBM p: {[round(x, 4) for x in p_emp_sbm]}", flush=True)

        # 2. Null realizations
        null_p_by_m: dict[int, list[float]] = {m: [] for m in MEMORY_HORIZONS}

        print(f"      Running {n_realizations} null realizations...", flush=True)
        for r in range(n_realizations):
            realiz_rng = np.random.default_rng(RANDOM_SEED + d_idx * 10_000 + r)
            r_fits = run_null_realization(
                days=days,
                n=n,
                labels=labels,
                dataset_key=dataset_key,
                is_labeled=is_labeled,
                realiz_idx=r,
                num_paths_day=num_paths_day,
                rng=realiz_rng,
                recompute=recompute,
            )
            for m in MEMORY_HORIZONS:
                null_p_by_m[m].append(r_fits[m])

        null_mean = [float(np.mean(null_p_by_m[m])) for m in MEMORY_HORIZONS]
        null_std = [float(np.std(null_p_by_m[m])) for m in MEMORY_HORIZONS]

        print(f"      Null Mean p:     {[round(x, 4) for x in null_mean]}", flush=True)
        print(f"      Null Std p:      {[round(x, 4) for x in null_std]}", flush=True)

        figure_data[display_name] = {
            "is_labeled": is_labeled,
            "n": n,
            "k": k,
            "p_emp_mem": p_emp_mem,
            "p_emp_sbm": p_emp_sbm,
            "p_null_mean": null_mean,
            "p_null_std": null_std,
        }

        for m_idx, m in enumerate(MEMORY_HORIZONS):
            results_table.append({
                "dataset": display_name,
                "dataset_key": dataset_key,
                "is_labeled": is_labeled,
                "n_nodes": n,
                "k_communities": k if k is not None else 0,
                "m": m,
                "p_emp_mem": p_emp_mem[m_idx],
                "p_emp_sbm": p_emp_sbm[m_idx] if is_labeled else None,
                "p_null_mean": null_mean[m_idx],
                "p_null_std": null_std[m_idx],
                "n_realizations": n_realizations,
                "num_paths_day": num_paths_day,
            })

        print(f"      Finished '{display_name}' in {time.time() - t0:.1f}s.", flush=True)

    # 3. Save numerical CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_filename = "step09_null_results_quick.csv" if quick else "step09_null_results.csv"
    csv_path = RESULTS_DIR / csv_filename
    pd.DataFrame(results_table).to_csv(csv_path, index=False)
    print(f"\n[+] Saved numerical results table to: {csv_path}", flush=True)

    # 4. Save comparison figure
    fig_filename = "step09_null_comparison_quick.png" if quick else "step09_null_comparison.png"
    fig_path = RESULTS_DIR / fig_filename
    plot_figure3_panels(
        results_dict=figure_data,
        m_values=MEMORY_HORIZONS,
        output_path=fig_path,
    )
    print(f"[+] Saved Figure 3 comparison plot to: {fig_path}", flush=True)

    total_time = time.time() - start_total
    print("\n" + "=" * 75, flush=True)
    print(f"Step 9 Pipeline Finished in {total_time:.1f}s ({total_time / 60:.2f} min).", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 9: Null models and statistical significance")
    parser.add_argument("--quick", action="store_true", help="Run quick test (3 realizations, 300 paths)")
    parser.add_argument("--dataset", type=str, default=None, help="Run only a specific dataset key")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation ignoring cache")

    args = parser.parse_args()
    run_step09_pipeline(quick=args.quick, target_dataset=args.dataset, recompute=args.recompute)

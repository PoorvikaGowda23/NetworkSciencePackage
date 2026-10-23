"""
experiments/step16_pc.py
=========================
Per-community memory parameter p_c estimation.

For every dataset that carries community labels (school class, hospital role,
or workplace department) this script:
  1. Loads all available observation days and samples STEP16_NUM_PATHS TRPs
     per day; paths from all days are pooled together.
  2. Builds transition records at memory horizon STEP16_M.
  3. Fits a global MEM+SBM model to obtain the community affinity matrix C.
  4. Calls fit_p_per_group to estimate the scalar memory parameter p_c
     separately for each community, holding C fixed and computing 95%
     bootstrap confidence intervals (B = N_BOOTSTRAP resamples).
  5. Prints a table of per-group record counts and p_c estimates, then saves:
       results/step16_pc_{dataset}.csv   (numerical results)
       results/step16_pc_{dataset}.png   (bar chart with CIs)
  6. Writes a combined CSV across all datasets:
       results/step16_pc_all.csv

Datasets without community labels (conference, malawi) are skipped.

NOTE on pooling days
--------------------
Paths from all days are concatenated before building records.  This maximises
the number of records per group so that more groups exceed MIN_GROUP_RECORDS.
The downside is that the resulting p_c mixes across-day variability; for a
time-resolved view, see step14_time_varying.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the package root and src importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    MIN_GROUP_RECORDS,
    N_BOOTSTRAP,
    PATH_LENGTH,
    RANDOM_SEED,
    RESULTS_DIR,
    STEP16_M,
    STEP16_NUM_PATHS,
    T_RES_SECONDS,
)
from memory_trp.aggregation import build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.models import fit_mem_sbm, fit_p_per_group
from memory_trp.plotting import plot_p_per_group
from memory_trp.records import build_records
from memory_trp.trp import generate_many_trps


# ---------------------------------------------------------------------------
# Datasets that carry community labels
# ---------------------------------------------------------------------------

LABELED_DATASETS: list[tuple[str, str]] = [
    ("Primary School",   "primaryschool"),
    ("High School 2011", "hs2011"),
    ("High School 2012", "hs2012"),
    ("High School 2013", "hs2013"),
    ("Hospital",         "hospital"),
    ("Workplace",        "workplace"),
]
# conference and malawi carry no labels and are excluded.


# ---------------------------------------------------------------------------
# Label name recovery
# ---------------------------------------------------------------------------

def _load_label_names(dataset_name: str) -> dict[int, str] | None:
    """Reconstruct the integer -> string community label mapping for a dataset.

    The loading.py module encodes string labels via:
        class_to_int = {c: idx for idx, c in enumerate(sorted(unique_labels))}
    so integer k always corresponds to the k-th element of sorted unique
    strings.  This function re-reads the same raw column to build the reverse
    dict {k: string_label}.

    Assumption: sorting order here must match loading._remap_nodes exactly
    (Python default lexicographic sort on the raw strings).

    Returns None for datasets without community metadata.
    """
    def _sorted_unique(path: Path, sep: str, col_idx: int) -> list[str]:
        """Read one column from a headerless CSV and return sorted unique values."""
        df = pd.read_csv(path, sep=sep, header=None, usecols=[col_idx])
        return sorted(df.iloc[:, 0].dropna().astype(str).unique().tolist())

    if dataset_name == "primaryschool":
        # Format: t,i,j,class_i,class_j  (comma-sep, no header)
        unique = _sorted_unique(DATA_DIR / "primaryschool.csv", ",", 3)
        return dict(enumerate(unique))

    if dataset_name in ("hs2011", "hs2012"):
        # Format: t\ti\tj\tclass_i\tclass_j  (tab-sep, no header)
        unique = _sorted_unique(DATA_DIR / f"{dataset_name}.csv", "\t", 3)
        return dict(enumerate(unique))

    if dataset_name == "hs2013":
        # Format: t i j class_i class_j  (space-sep, no header)
        unique = _sorted_unique(DATA_DIR / "hs2013.csv", " ", 3)
        return dict(enumerate(unique))

    if dataset_name == "hospital":
        # Format: t\ti\tj\trole_i\trole_j  (tab-sep, no header)
        # Roles are ADM (admin), MED (doctor), NUR (nurse), PAT (patient)
        unique = _sorted_unique(DATA_DIR / "hospital.csv", "\t", 3)
        return dict(enumerate(unique))

    if dataset_name == "workplace":
        # Metadata in a separate file: id,dept  (comma-sep, no header)
        df = pd.read_csv(DATA_DIR / "workplace_meta.csv", header=None,
                         names=["id", "dept"])
        unique = sorted(df["dept"].dropna().astype(str).unique().tolist())
        return dict(enumerate(unique))

    return None


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_per_group(
    datasets: list[tuple[str, str]] = LABELED_DATASETS,
    m: int = STEP16_M,
    num_paths: int = STEP16_NUM_PATHS,
    B: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> None:
    """Fit and plot per-community memory p_c for all labeled datasets.

    Parameters
    ----------
    datasets : list of (display_name, dataset_key) pairs
        Datasets to process; must have community labels.
    m : int
        Memory horizon (steps) used to build transition records.
    num_paths : int
        TRPs sampled per observation day.
    B : int
        Bootstrap resamples for computing 95% confidence intervals.
    seed : int
        Master random seed passed to np.random.default_rng.
    """
    print("=" * 70, flush=True)
    print("Step 16 -- Per-Community Memory Parameter p_c", flush=True)
    print(f"Memory horizon m={m} | Paths/day={num_paths:,} | Bootstrap B={B}", flush=True)
    print(f"Min records per group: {MIN_GROUP_RECORDS}", flush=True)
    print(f"Datasets: {[name for name, _ in datasets]}", flush=True)
    print("=" * 70, flush=True)

    rng = np.random.default_rng(seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for ds_name, ds_key in datasets:
        t0 = time.time()
        print(f"\n{'=' * 60}", flush=True)
        print(f"[{ds_name}]", flush=True)

        # ------------------------------------------------------------------
        # 1. Load contacts and labels
        # ------------------------------------------------------------------
        contacts, labels = load_contacts(ds_key)
        if labels is None:
            print(f"  Skipping -- no community labels.", flush=True)
            continue

        n = int(labels.shape[0])
        days = split_by_day(contacts, ds_key)
        k = int(np.unique(labels).size)
        print(f"  n={n} nodes | {k} communities | {len(days)} day(s)", flush=True)

        # ------------------------------------------------------------------
        # 2. Sample TRPs from every day and pool all paths
        # ------------------------------------------------------------------
        all_paths: list[tuple[list[int], list[int]]] = []
        for d_idx, day_df in enumerate(days):
            g = build_temporal_graph(
                day_df, n, labels, t_res=T_RES_SECONDS,
                name=f"{ds_key}_day{d_idx}",
            )
            try:
                day_paths = generate_many_trps(
                    g, num_paths=num_paths, length=PATH_LENGTH, rng=rng,
                )
                all_paths.extend(day_paths)
            except (ValueError, RuntimeError) as exc:
                print(f"  Day {d_idx}: skipped ({exc})", flush=True)

        n_paths = len(all_paths)
        print(f"  Total paths collected: {n_paths:,}", flush=True)

        if n_paths < 10:
            print(f"  Skipping -- too few paths.", flush=True)
            continue

        # ------------------------------------------------------------------
        # 3. Build transition records at memory horizon m
        # ------------------------------------------------------------------
        records = build_records(all_paths, m=m)
        n_rec_total = len(records)
        print(f"  Total transition records (m={m}): {n_rec_total:,}", flush=True)

        if n_rec_total == 0:
            print(f"  Skipping -- no transition records.", flush=True)
            continue

        # ------------------------------------------------------------------
        # 4. Fit global MEM+SBM to get the community affinity matrix C
        # ------------------------------------------------------------------
        print(f"  Fitting global MEM+SBM ...", flush=True)
        global_fit = fit_mem_sbm(records, labels=labels)
        C = global_fit.C
        print(
            f"  Global: p={global_fit.p:.4f} | loglik={global_fit.loglik:.1f} | "
            f"C shape={C.shape} | converged={global_fit.converged}",
            flush=True,
        )

        # ------------------------------------------------------------------
        # 5. Per-group p_c with 95% bootstrap CIs
        # ------------------------------------------------------------------
        print(f"  Fitting p_c per group (B={B} bootstrap resamples) ...", flush=True)
        df_group = fit_p_per_group(
            records, labels=labels, C=C, n=n, B=B, rng=rng,
        )

        if len(df_group) == 0:
            print(
                f"  No groups exceeded MIN_GROUP_RECORDS={MIN_GROUP_RECORDS}. "
                "Try increasing --num-paths.",
                flush=True,
            )
            continue

        # ------------------------------------------------------------------
        # 6. Print results table with human-readable group names
        # ------------------------------------------------------------------
        label_names = _load_label_names(ds_key)

        print(
            f"\n  {'Idx':>5} {'Name':>10} {'n_records':>10} "
            f"{'p_hat':>8} {'CI_low':>8} {'CI_high':>8}"
        )
        print(f"  {'=' * 55}")
        for _, row in df_group.iterrows():
            g_idx = int(row["group"])
            gname = (
                label_names.get(g_idx, f"Group {g_idx}")
                if label_names else f"Group {g_idx}"
            )
            print(
                f"  {g_idx:>5} {gname:>10} {int(row['n_records']):>10,} "
                f"{row['p_hat']:>8.4f} {row['ci_low']:>8.4f} {row['ci_high']:>8.4f}"
            )

        # ------------------------------------------------------------------
        # 7. Save per-dataset CSV
        # ------------------------------------------------------------------
        df_out = df_group.copy()
        if label_names:
            df_out["group_name"] = df_out["group"].apply(
                lambda g: label_names.get(int(g), f"Group {g}")
            )
        df_out["dataset"]      = ds_key
        df_out["dataset_name"] = ds_name
        df_out["m"]            = m
        df_out["global_p"]     = global_fit.p

        csv_path = RESULTS_DIR / f"step16_pc_{ds_key}.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"\n  Saved CSV  -> {csv_path}", flush=True)

        # ------------------------------------------------------------------
        # 8. Plot bar chart
        # ------------------------------------------------------------------
        plot_path = RESULTS_DIR / f"step16_pc_{ds_key}.png"
        plot_p_per_group(
            df=df_group,
            dataset_name=ds_name,
            group_names=label_names,
            output_path=plot_path,
            global_p=global_fit.p,
        )
        print(f"  Saved plot -> {plot_path}", flush=True)

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s", flush=True)

        all_rows.extend(df_out.to_dict("records"))

    # ----------------------------------------------------------------------
    # Save combined CSV across all datasets
    # ----------------------------------------------------------------------
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        combined_path = RESULTS_DIR / "step16_pc_all.csv"
        df_all.to_csv(combined_path, index=False)
        print(f"\n{'=' * 60}", flush=True)
        print(f"Saved combined CSV -> {combined_path}", flush=True)
    else:
        print("\nNo results produced -- all datasets were skipped.", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 16: Per-community memory parameter p_c."
    )
    parser.add_argument(
        "--m",
        type=int,
        default=STEP16_M,
        help=f"Memory horizon m (default: {STEP16_M})",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=STEP16_NUM_PATHS,
        help=f"TRPs sampled per observation day (default: {STEP16_NUM_PATHS})",
    )
    parser.add_argument(
        "--bootstrap-b",
        type=int,
        default=N_BOOTSTRAP,
        help=f"Bootstrap draw count B (default: {N_BOOTSTRAP})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    run_per_group(
        m=args.m,
        num_paths=args.num_paths,
        B=args.bootstrap_b,
        seed=args.seed,
    )

"""
experiments/step19_low_return_interventions.py
==============================================
Step 19: Isolating the nodes that are LEAST likely to return to already-met nodes.

Extends Step 18. In Step 18 the "Return Rate" strategy isolated the nodes with the
HIGHEST return rate and gave little benefit. This step tests the opposite idea:

    Hypothesis: nodes that keep going back to the same few people (high return
    rate) meet few DIFFERENT people, so they are poor spreaders. Nodes that rarely
    return meet many different people ("browsers") and should be better targets.

Return rate of node u (same definition as Step 18):
    r_u = (# path steps leaving u whose next node is in the memory set)
          / (# path steps leaving u)
estimated from sampled non-backtracking time-respecting paths. Only nodes with at
least `min_records` records are ranked.

Strategies compared (budget = k% of nodes, isolated before the epidemic starts):
  1. None              baseline, no isolation
  2. Random            k% of nodes at random
  3. Contact Time      k% of nodes with the highest total contact time
  4. High Return Rate  k% of nodes with the HIGHEST return rate (Step 18 strategy)
  5. Low Return Rate   k% of nodes with the LOWEST return rate   <-- new

Improvements over Step 18
-------------------------
* Paired runs: every strategy uses the same random start snapshots and the same
  index cases (an index case that is isolated is replaced deterministically), so
  strategy differences are not buried in start-time noise.
* 95% bootstrap confidence intervals on the mean outbreak and on the PAIRED
  reduction versus "None".
* Optional out-of-sample test: choose nodes on one day (--select-day) and simulate
  the epidemic on another (--eval-day), so the strategies do not "see the future".
* Diagnostics explaining WHY a strategy works: how much contact time and how many
  distinct contacts the removed nodes have compared with the average node, and the
  Spearman correlation between return rate and contact time.

Outputs
-------
  results/step19_low_return_all.csv          per dataset and strategy results
  results/step19_return_diagnostics.csv      per dataset diagnostics
  results/step19_low_return_grid.png         outbreak size per strategy, all datasets
  results/step19_low_return_reduction.png    paired % reduction vs None, all datasets

Usage
-----
  python experiments/step19_low_return_interventions.py
  python experiments/step19_low_return_interventions.py --datasets primaryschool hs2013
  python experiments/step19_low_return_interventions.py --select-day 0 --eval-day 1
  python experiments/step19_low_return_interventions.py --recovery-snapshots 180
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make package root and src importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    INTERVENTION_BUDGET_PERCENT,
    INTERVENTION_MIN_RECORDS,
    N_BOOTSTRAP,
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
from memory_trp.records import build_records
from memory_trp.sir import (
    calibrate_rate,
    get_highest_contact_nodes,
    get_highest_return_nodes,
    get_random_nodes,
    simulate_sir,
)
from memory_trp.trp import generate_many_trps


# Local constants (move to config.py if you want them centralised)
MAJOR_OUTBREAK_THRESHOLD: float = 0.05   # a run is a "major outbreak" if >= 5% infected
CALIBRATION_WARN_FRACTION: float = 0.5   # warn if baseline outbreak < 50% of the target

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

STRATEGY_ORDER: list[str] = [
    "None",
    "Random",
    "Contact Time",
    "High Return Rate",
    "Low Return Rate",
]

STRATEGY_COLORS: dict[str, str] = {
    "None":             "#d62728",
    "Random":           "#1f77b4",
    "Contact Time":     "#ff7f0e",
    "High Return Rate": "#2ca02c",
    "Low Return Rate":  "#9467bd",
}


# ---------------------------------------------------------------------------
# Node statistics and the new selection rule
# ---------------------------------------------------------------------------


def node_return_rates(records, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute the per-node return rate and record count.

    Parameters
    ----------
    records : Records
        Transition records (needs `.cur` and `.in_memory`).
    n : int
        Number of nodes.

    Returns
    -------
    rates : np.ndarray, shape (n,)
        Fraction of steps leaving each node that land in the memory set.
        NaN for nodes with zero records.
    counts : np.ndarray, shape (n,)
        Number of records whose current node is u.
    """
    counts = np.bincount(records.cur, minlength=n)
    returns = np.bincount(
        records.cur, weights=records.in_memory.astype(float), minlength=n
    )
    rates = np.full(n, np.nan, dtype=float)
    seen = counts > 0
    rates[seen] = returns[seen] / counts[seen]
    return rates, counts


def get_lowest_return_nodes(
    records,
    budget: int,
    n: int | None = None,
    min_records: int = INTERVENTION_MIN_RECORDS,
) -> list[int]:
    """Select the `budget` nodes with the LOWEST node-level return rate.

    Mirror image of `get_highest_return_nodes`. Only nodes with at least
    `min_records` transition records are ranked, because a node seen only a few
    times can look like a "never returns" node purely by chance. Ties are broken
    in favour of nodes with more records (more reliable estimate). If fewer than
    `budget` nodes are eligible, the remaining slots are filled with the
    ineligible nodes that have the most records.

    Parameters
    ----------
    records : Records
        Transition records from sampled time-respecting paths.
    budget : int
        Number of nodes to select.
    n : int or None
        Total number of nodes. Inferred from the records if None.
    min_records : int
        Minimum record count for a node to be ranked.

    Returns
    -------
    list[int]
        Sorted list of selected node indices.
    """
    if n is None:
        n = int(np.max(records.cur) + 1) if len(records) > 0 else 0
    budget = min(max(budget, 0), n)
    if budget == 0 or len(records) == 0:
        return []

    rates, counts = node_return_rates(records, n)

    eligible = np.where(counts >= min_records)[0]
    # np.lexsort sorts by the LAST key first: primary = rate ascending,
    # secondary = record count descending
    order = eligible[np.lexsort((-counts[eligible], rates[eligible]))]
    chosen = [int(u) for u in order[:budget]]

    if len(chosen) < budget:
        remaining = np.setdiff1d(np.arange(n), np.array(chosen, dtype=int))
        remaining = remaining[np.argsort(-counts[remaining], kind="stable")]
        chosen += [int(u) for u in remaining[: budget - len(chosen)]]

    return sorted(chosen)


def _total_contact_time(graph) -> np.ndarray:
    """Total contact weight incident to each node over all snapshots."""
    total = np.zeros(graph.n, dtype=float)
    for snap in graph.snapshots:
        if snap.nnz > 0:
            total += np.asarray(snap.sum(axis=1)).ravel()
    return total


def _unique_contacts(graph) -> np.ndarray:
    """Number of DISTINCT people each node was ever in contact with."""
    n = graph.n
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for snap in graph.snapshots:
        if snap.nnz > 0:
            coo = snap.tocoo()
            rows.append(coo.row)
            cols.append(coo.col)
    if not rows:
        return np.zeros(n, dtype=float)
    r = np.concatenate(rows).astype(np.int64)
    c = np.concatenate(cols).astype(np.int64)
    pair_ids = np.unique(r * n + c)           # unique directed (u, v) pairs
    return np.bincount(pair_ids // n, minlength=n).astype(float)


# ---------------------------------------------------------------------------
# Paired simulation helpers
# ---------------------------------------------------------------------------


def _pick_seed(base_seed: int, removed: set[int], n: int) -> int:
    """Return the index case for one run, avoiding isolated nodes.

    The same `base_seed` is used by every strategy. If it is isolated under a
    strategy, a deterministic replacement among the non-isolated nodes is used.
    """
    if base_seed not in removed:
        return base_seed
    available = [i for i in range(n) if i not in removed]
    return available[base_seed % len(available)]


def _bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_reduction(
    base: np.ndarray,
    strat: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float, float]:
    """Paired reduction of a strategy versus the baseline.

    Returns
    -------
    abs_reduction : float
        mean(base - strat), in outbreak-fraction units.
    rel_reduction_pct : float
        100 * mean(base - strat) / mean(base).
    rel_ci_low, rel_ci_high : float
        95% bootstrap CI of the relative reduction (runs resampled in pairs).
    """
    base = np.asarray(base, dtype=float)
    strat = np.asarray(strat, dtype=float)
    diff = base - strat

    abs_red = float(diff.mean())
    rel_red = float(diff.mean() / base.mean() * 100.0) if base.mean() > 0 else 0.0

    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_diff = diff[idx].mean(axis=1)
    boot_base = base[idx].mean(axis=1)
    safe_base = np.where(boot_base > 0, boot_base, 1.0)
    boot_rel = np.where(boot_base > 0, boot_diff / safe_base * 100.0, 0.0)

    return (
        abs_red,
        rel_red,
        float(np.percentile(boot_rel, 2.5)),
        float(np.percentile(boot_rel, 97.5)),
    )


def _ratio_to_population(values: np.ndarray, nodes: list[int]) -> float:
    """Mean of `values` over `nodes` divided by the mean over all nodes."""
    if len(nodes) == 0 or values.mean() == 0:
        return float("nan")
    return float(values[nodes].mean() / values.mean())


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_low_return_interventions(
    datasets: list[tuple[str, str]] = TARGET_DATASETS,
    k_percent: float = INTERVENTION_BUDGET_PERCENT,
    n_runs: int = N_SIR_RUNS,
    recovery_snapshots: int = SIR_RECOVERY_SNAPSHOTS,
    target_outbreak: float = SIR_TARGET_OUTBREAK,
    m: int = STEP18_M,
    num_paths: int = STEP18_NUM_PATHS,
    min_records: int = INTERVENTION_MIN_RECORDS,
    select_day: int = 0,
    eval_day: int = 0,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> None:
    """Run the low-return-rate isolation experiment on every dataset.

    Parameters
    ----------
    datasets : list of (display_name, dataset_key)
        Datasets to evaluate.
    k_percent : float
        Isolation budget as a percentage of nodes.
    n_runs : int
        Epidemic runs per strategy (identical start conditions across strategies).
    recovery_snapshots : int
        Infectious period in snapshots.
    target_outbreak : float
        Target mean outbreak used to calibrate the transmission rate.
    m : int
        Memory horizon used to build transition records.
    num_paths : int
        Number of time-respecting paths sampled to estimate return rates.
    min_records : int
        Minimum records for a node to be ranked by return rate.
    select_day, eval_day : int
        Day index used to choose the nodes and day index used to run the
        epidemic. Equal values reproduce Step 18's in-sample design. If a day
        index does not exist for a dataset, day 0 is used instead.
    n_boot : int
        Number of bootstrap resamples for confidence intervals.
    seed : int
        Master random seed.
    """
    print("=" * 76, flush=True)
    print("Step 19 -- Isolating LOW-return-rate nodes", flush=True)
    print(
        f"Budget: {k_percent:.1f}% | Runs/strategy: {n_runs} | "
        f"Select day: {select_day} | Eval day: {eval_day} | Seed: {seed}",
        flush=True,
    )
    print("=" * 76, flush=True)

    rng = np.random.default_rng(seed)
    boot_rng = np.random.default_rng(seed + 1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    diag_rows: list[dict] = []

    for ds_name, ds_key in datasets:
        t0 = time.time()
        print(f"\n[{ds_name}]", flush=True)

        # 1. Load data and build the selection and evaluation graphs
        contacts, labels = load_contacts(ds_key)
        days = split_by_day(contacts, ds_key)
        n = len(labels) if labels is not None else int(contacts[["i", "j"]].values.max() + 1)

        sel_idx = select_day if select_day < len(days) else 0
        ev_idx = eval_day if eval_day < len(days) else 0
        if sel_idx != select_day or ev_idx != eval_day:
            print(
                f"  Note: dataset has {len(days)} day(s); using select={sel_idx}, eval={ev_idx}.",
                flush=True,
            )

        g_sel = build_temporal_graph(
            days[sel_idx], n, labels, t_res=T_RES_SECONDS, name=f"{ds_key}_select"
        )
        g_eval = (
            g_sel
            if ev_idx == sel_idx
            else build_temporal_graph(
                days[ev_idx], n, labels, t_res=T_RES_SECONDS, name=f"{ds_key}_eval"
            )
        )
        k_nodes = max(1, int(round(n * k_percent / 100.0)))
        print(f"  Nodes n={n} | Budget k={k_nodes} ({k_percent:.1f}%)", flush=True)

        # 2. Calibrate the transmission rate on the evaluation graph
        calib_rate = calibrate_rate(
            graph=g_eval,
            target_final_fraction=target_outbreak,
            recovery_snapshots=recovery_snapshots,
            n_calib_runs=SIR_CALIBRATION_RUNS,
            rng=rng,
        )
        print(f"  Calibrated transmission rate beta: {calib_rate:.5f}", flush=True)

        # 3. Node statistics from the SELECTION graph
        paths = generate_many_trps(g_sel, num_paths=num_paths, length=PATH_LENGTH, rng=rng)
        records = build_records(paths, m=m)
        rates, counts = node_return_rates(records, n)
        contact_time = _total_contact_time(g_sel)
        unique_contacts = _unique_contacts(g_sel)

        eligible = counts >= min_records
        if eligible.sum() >= 3:
            rho_ct, _ = spearmanr(rates[eligible], contact_time[eligible])
            rho_uc, _ = spearmanr(rates[eligible], unique_contacts[eligible])
        else:
            rho_ct = rho_uc = float("nan")
        print(
            f"  Spearman(return rate, contact time) = {rho_ct:+.2f} | "
            f"Spearman(return rate, distinct contacts) = {rho_uc:+.2f} "
            f"({int(eligible.sum())}/{n} nodes ranked)",
            flush=True,
        )
        diag_rows.append({
            "dataset": ds_name,
            "dataset_key": ds_key,
            "n_nodes": n,
            "n_ranked_nodes": int(eligible.sum()),
            "spearman_return_vs_contact_time": float(rho_ct),
            "spearman_return_vs_distinct_contacts": float(rho_uc),
        })

        # 4. Choose the nodes to isolate under each strategy
        strategies: list[tuple[str, list[int]]] = [
            ("None", []),
            ("Random", get_random_nodes(n=n, budget=k_nodes, rng=rng)),
            ("Contact Time", get_highest_contact_nodes(graph=g_sel, budget=k_nodes)),
            ("High Return Rate", get_highest_return_nodes(
                records=records, budget=k_nodes, n=n, min_records=min_records)),
            ("Low Return Rate", get_lowest_return_nodes(
                records=records, budget=k_nodes, n=n, min_records=min_records)),
        ]

        # 5. Shared (paired) start conditions for every strategy
        total_snaps = len(g_eval.snapshots)
        starts = rng.integers(0, total_snaps, size=n_runs)
        base_seeds = rng.integers(0, n, size=n_runs)

        outcomes: dict[str, np.ndarray] = {}
        for strat_name, rem_nodes in strategies:
            removed = set(rem_nodes)
            vals = np.empty(n_runs, dtype=float)
            for r in range(n_runs):
                res = simulate_sir(
                    graph=g_eval,
                    transmission_rate=calib_rate,
                    recovery_snapshots=recovery_snapshots,
                    rng=rng,
                    start_snapshot=int(starts[r]),
                    seed_node=_pick_seed(int(base_seeds[r]), removed, n),
                    removed_nodes=rem_nodes,
                )
                vals[r] = res.final_fraction
            outcomes[strat_name] = vals

        base = outcomes["None"]
        if base.mean() < CALIBRATION_WARN_FRACTION * target_outbreak:
            print(
                f"  WARNING: baseline outbreak {base.mean()*100:.1f}% is far below the "
                f"{target_outbreak*100:.0f}% target. The calibration probably hit its "
                f"ceiling (beta={calib_rate:.2f}); try --recovery-snapshots 180 or more.",
                flush=True,
            )

        # 6. Summaries
        print(
            f"  {'Strategy':<18}{'Removed':>8}{'Mean outbreak':>15}"
            f"{'95% CI':>20}{'Reduction vs None':>22}",
            flush=True,
        )
        print(f"  {'-' * 83}", flush=True)

        for strat_name, rem_nodes in strategies:
            vals = outcomes[strat_name]
            ci_lo, ci_hi = _bootstrap_mean_ci(vals, boot_rng, n_boot)
            abs_red, rel_red, rel_lo, rel_hi = _paired_reduction(base, vals, boot_rng, n_boot)

            rem_list = list(rem_nodes)
            mean_ret = float(np.nanmean(rates[rem_list])) if rem_list else float("nan")

            print(
                f"  {strat_name:<18}{len(rem_list):>8d}{vals.mean()*100:>14.1f}%"
                f"{f'[{ci_lo*100:.1f}, {ci_hi*100:.1f}]%':>20}"
                f"{f'{rel_red:+.0f}% [{rel_lo:+.0f}, {rel_hi:+.0f}]':>22}",
                flush=True,
            )

            all_rows.append({
                "dataset": ds_name,
                "dataset_key": ds_key,
                "n_nodes": n,
                "select_day": sel_idx,
                "eval_day": ev_idx,
                "k_budget_percent": k_percent,
                "calibrated_rate": calib_rate,
                "strategy": strat_name,
                "n_removed": len(rem_list),
                "mean_outbreak": float(vals.mean()),
                "std_outbreak": float(vals.std()),
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "p_major_outbreak": float((vals >= MAJOR_OUTBREAK_THRESHOLD).mean()),
                "reduction_abs": abs_red,
                "reduction_vs_none_percent": rel_red,
                "reduction_ci_low_percent": rel_lo,
                "reduction_ci_high_percent": rel_hi,
                "removed_mean_return_rate": mean_ret,
                "removed_contact_time_vs_avg": _ratio_to_population(contact_time, rem_list),
                "removed_distinct_contacts_vs_avg": _ratio_to_population(unique_contacts, rem_list),
            })

        print(f"  Completed in {time.time() - t0:.1f}s", flush=True)

    # -----------------------------------------------------------------------
    # Save tables and figures
    # -----------------------------------------------------------------------
    df_all = pd.DataFrame(all_rows)
    df_diag = pd.DataFrame(diag_rows)

    csv_all = RESULTS_DIR / "step19_low_return_all.csv"
    csv_diag = RESULTS_DIR / "step19_return_diagnostics.csv"
    df_all.to_csv(csv_all, index=False)
    df_diag.to_csv(csv_diag, index=False)

    grid_path = RESULTS_DIR / "step19_low_return_grid.png"
    red_path = RESULTS_DIR / "step19_low_return_reduction.png"
    _plot_grid(df_all, grid_path)
    _plot_reduction(df_all, red_path)

    print(f"\n{'=' * 76}", flush=True)
    print(f"Saved -> {csv_all}", flush=True)
    print(f"Saved -> {csv_diag}", flush=True)
    print(f"Saved -> {grid_path}", flush=True)
    print(f"Saved -> {red_path}", flush=True)

    _print_summary(df_all)


# ---------------------------------------------------------------------------
# Summary and plotting
# ---------------------------------------------------------------------------


def _print_summary(df_all: pd.DataFrame) -> None:
    """Print the average paired reduction per strategy across datasets."""
    print("\nAverage paired reduction vs None across datasets:", flush=True)
    summary = (
        df_all[df_all["strategy"] != "None"]
        .groupby("strategy")["reduction_vs_none_percent"]
        .mean()
        .reindex([s for s in STRATEGY_ORDER if s != "None"])
    )
    for strat, val in summary.items():
        print(f"  {strat:<18}{val:>+8.1f}%", flush=True)

    wide = df_all.pivot(index="dataset_key", columns="strategy", values="mean_outbreak")
    if {"Low Return Rate", "High Return Rate", "Contact Time"} <= set(wide.columns):
        n_ds = len(wide)
        beats_high = int((wide["Low Return Rate"] < wide["High Return Rate"]).sum())
        beats_ct = int((wide["Low Return Rate"] < wide["Contact Time"]).sum())
        print(
            f"\nLow Return Rate has a smaller mean outbreak than High Return Rate in "
            f"{beats_high}/{n_ds} datasets and than Contact Time in {beats_ct}/{n_ds}.",
            flush=True,
        )


def _plot_grid(df_all: pd.DataFrame, output_path: Path) -> None:
    """One panel per dataset: mean outbreak per strategy with 95% bootstrap CI."""
    keys = list(dict.fromkeys(df_all["dataset_key"]))
    ncols = 4
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)

    for ax, key in zip(axes.ravel(), keys):
        sub = df_all[df_all["dataset_key"] == key].set_index("strategy").reindex(STRATEGY_ORDER)
        means = sub["mean_outbreak"].to_numpy()
        yerr = np.vstack([means - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - means])
        x = np.arange(len(STRATEGY_ORDER))
        ax.bar(x, means, yerr=yerr, capsize=3,
               color=[STRATEGY_COLORS[s] for s in STRATEGY_ORDER])
        ax.set_xticks(x)
        ax.set_xticklabels(STRATEGY_ORDER, rotation=40, ha="right", fontsize=8)
        ax.set_title(sub["dataset"].iloc[0], fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean outbreak fraction", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")

    fig.suptitle("Step 19: isolating 10% of nodes by return rate (error bars = 95% bootstrap CI)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_reduction(df_all: pd.DataFrame, output_path: Path) -> None:
    """Grouped bars: paired % reduction vs None for every dataset."""
    strategies = [s for s in STRATEGY_ORDER if s != "None"]
    keys = list(dict.fromkeys(df_all["dataset_key"]))
    names = [df_all[df_all["dataset_key"] == k]["dataset"].iloc[0] for k in keys]

    fig, ax = plt.subplots(figsize=(1.7 * len(keys) + 3, 5))
    width = 0.8 / len(strategies)
    x = np.arange(len(keys))

    for i, strat in enumerate(strategies):
        sub = df_all[df_all["strategy"] == strat].set_index("dataset_key").reindex(keys)
        vals = sub["reduction_vs_none_percent"].to_numpy()
        yerr = np.vstack([
            vals - sub["reduction_ci_low_percent"].to_numpy(),
            sub["reduction_ci_high_percent"].to_numpy() - vals,
        ])
        ax.bar(x + (i - (len(strategies) - 1) / 2) * width, vals, width,
               yerr=yerr, capsize=2, label=strat, color=STRATEGY_COLORS[strat])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Reduction in outbreak size vs None (%)")
    ax.set_title("Paired reduction in outbreak size vs None\n(higher is better, error bars = 95% CI)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 19: isolate the nodes least likely to return to already-met nodes."
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset keys to run (default: all 8), e.g. primaryschool hs2013")
    parser.add_argument("--k-percent", type=float, default=INTERVENTION_BUDGET_PERCENT)
    parser.add_argument("--runs", type=int, default=N_SIR_RUNS)
    parser.add_argument("--recovery-snapshots", type=int, default=SIR_RECOVERY_SNAPSHOTS)
    parser.add_argument("--target-outbreak", type=float, default=SIR_TARGET_OUTBREAK)
    parser.add_argument("--m", type=int, default=STEP18_M)
    parser.add_argument("--num-paths", type=int, default=STEP18_NUM_PATHS)
    parser.add_argument("--min-records", type=int, default=INTERVENTION_MIN_RECORDS)
    parser.add_argument("--select-day", type=int, default=0,
                        help="Day index used to choose the nodes to isolate")
    parser.add_argument("--eval-day", type=int, default=0,
                        help="Day index used to simulate the epidemic")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    chosen = TARGET_DATASETS
    if args.datasets:
        chosen = [(name, key) for name, key in TARGET_DATASETS if key in args.datasets]
        unknown = set(args.datasets) - {key for _, key in TARGET_DATASETS}
        if unknown:
            parser.error(f"Unknown dataset key(s): {sorted(unknown)}")

    run_low_return_interventions(
        datasets=chosen,
        k_percent=args.k_percent,
        n_runs=args.runs,
        recovery_snapshots=args.recovery_snapshots,
        target_outbreak=args.target_outbreak,
        m=args.m,
        num_paths=args.num_paths,
        min_records=args.min_records,
        select_day=args.select_day,
        eval_day=args.eval_day,
        seed=args.seed,
    )
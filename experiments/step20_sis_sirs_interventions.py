"""
experiments/step20_sis_sirs_interventions.py
============================================
Step 20: Targeted isolation interventions under SIR, SIS and SIRS dynamics.

Extends Steps 18 and 19, which evaluated isolation strategies only for an SIR
epidemic. Here the *same* strategies are evaluated for three compartmental models
on the same networks, with the same budget and the same paired start conditions:

    SIR   S -> I -> R         permanent immunity
    SIS   S -> I -> S         no immunity, reinfection, possible endemic persistence
    SIRS  S -> I -> R -> S    immunity that wanes after --immunity-snapshots

Strategies (budget = k% of nodes, isolated before the epidemic starts):
  1. None              baseline
  2. Random            k% of nodes at random
  3. Contact Time      k% of nodes with the highest total contact time
  4. High Return Rate  k% of nodes with the HIGHEST return rate   (Step 18)
  5. Low Return Rate   k% of nodes with the LOWEST  return rate   (Step 19)
  6. Close Group       one whole community closest in size to the budget
                       (labelled datasets only)

Why a second outcome metric?
----------------------------
With reinfection the fraction ever infected saturates and hides how much
infection is circulating. Two outcomes are therefore reported for every model:

  * Outbreak size   fraction of nodes ever infected (the Step 18/19 metric)
  * Mean prevalence time-averaged fraction infected over the simulated day
                    (infection burden / person-time infected)

plus the probability that the infection persists to the end of the window and the
number of infection events per node (reinfection pressure).

Design
------
* Paired runs: all strategies AND all models share the same start snapshots and
  index cases (an isolated index case is replaced deterministically).
* 95% bootstrap CIs on means and on paired reductions versus "None".
* Optional out-of-sample test with --select-day / --eval-day.
* Calibration: by default every model is calibrated to the same baseline attack
  rate (--target-outbreak) so effects are comparable. With --shared-beta a single
  beta calibrated on SIR is used for all models, which compares them at identical
  transmissibility instead.

Reuses the helpers of experiments/step19_low.py (must sit in the same folder).

Outputs (results/)
------------------
  step20_interventions_all_models.csv     all rows (model x dataset x strategy)
  step20_<model>_interventions.csv        one file per model
  step20_summary_by_model.csv             mean reduction per model and strategy
  step20_<model>_grid_outbreak.png        outbreak size per strategy, all datasets
  step20_<model>_grid_prevalence.png      mean prevalence per strategy, all datasets
  step20_reduction_outbreak.png           paired % reduction, one panel per model
  step20_reduction_prevalence.png         paired % reduction in prevalence
  step20_trajectories_<dataset>.png       mean prevalence over time per strategy

Usage
-----
  python experiments/step20_sis_sirs_interventions.py
  python experiments/step20_sis_sirs_interventions.py --models sis sirs
  python experiments/step20_sis_sirs_interventions.py --datasets primaryschool hs2013 --runs 100
  python experiments/step20_sis_sirs_interventions.py --immunity-snapshots 360 --shared-beta
  python experiments/step20_sis_sirs_interventions.py --select-day 0 --eval-day 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make package root, src and this folder importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

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
    SIRS_IMMUNITY_SNAPSHOTS,
    STEP18_M,
    STEP18_NUM_PATHS,
    STEP20_MODELS,
    T_RES_SECONDS,
)
from memory_trp.aggregation import build_temporal_graph, split_by_day
from memory_trp.epidemics import (
    MODEL_LABELS,
    VALID_MODELS,
    calibrate_rate_model,
    simulate_epidemic,
)
from memory_trp.loading import load_contacts
from memory_trp.records import build_records
from memory_trp.sir import (
    get_community_nodes,
    get_highest_contact_nodes,
    get_highest_return_nodes,
    get_random_nodes,
)
from memory_trp.trp import generate_many_trps

# Helpers shared with Step 19 (keep the two experiments statistically identical)
from step19_low import (
    CALIBRATION_WARN_FRACTION,
    MAJOR_OUTBREAK_THRESHOLD,
    TARGET_DATASETS,
    _bootstrap_mean_ci,
    _paired_reduction,
    _pick_seed,
    _total_contact_time,
    get_lowest_return_nodes,
    node_return_rates,
)

STRATEGY_ORDER: list[str] = [
    "None",
    "Random",
    "Contact Time",
    "High Return Rate",
    "Low Return Rate",
    "Close Group",
]

STRATEGY_COLORS: dict[str, str] = {
    "None":             "#d62728",
    "Random":           "#1f77b4",
    "Contact Time":     "#ff7f0e",
    "High Return Rate": "#2ca02c",
    "Low Return Rate":  "#9467bd",
    "Close Group":      "#8c564b",
}

MODEL_COLORS: dict[str, str] = {"sir": "#1f77b4", "sis": "#d62728", "sirs": "#2ca02c"}

# Minimum fraction of runs that must still have data at time t for the mean
# trajectory to be drawn at t (runs seeded late in the day have short curves).
TRAJECTORY_MIN_RUN_FRACTION: float = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean_trajectory(
    curves: list[np.ndarray], n: int, min_frac: float = TRAJECTORY_MIN_RUN_FRACTION
) -> np.ndarray:
    """Mean prevalence versus time since seeding over runs with different lengths."""
    length = max(len(c) for c in curves)
    sums = np.zeros(length)
    cnt = np.zeros(length)
    for c in curves:
        sums[: len(c)] += c
        cnt[: len(c)] += 1
    keep = int(np.count_nonzero(cnt >= min_frac * len(curves)))
    keep = max(keep, 1)
    return sums[:keep] / cnt[:keep] / n


def _simulate_strategy(
    graph,
    model: str,
    rate: float,
    recovery_snapshots: int,
    immunity_snapshots: int,
    removed: list[int],
    starts: np.ndarray,
    base_seeds: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | list]:
    """Run all paired replicates of one (model, strategy) and collect outcomes."""
    n = graph.n
    n_runs = len(starts)
    removed_set = set(removed)

    out = {
        "outbreak": np.empty(n_runs),
        "prevalence": np.empty(n_runs),
        "peak_prev": np.empty(n_runs),
        "final_prev": np.empty(n_runs),
        "events": np.empty(n_runs),
        "persisted": np.empty(n_runs),
        "curves": [],
    }
    for r in range(n_runs):
        res = simulate_epidemic(
            graph=graph,
            transmission_rate=rate,
            model=model,
            recovery_snapshots=recovery_snapshots,
            immunity_snapshots=immunity_snapshots,
            rng=rng,
            start_snapshot=int(starts[r]),
            seed_node=_pick_seed(int(base_seeds[r]), removed_set, n),
            removed_nodes=removed,
        )
        out["outbreak"][r] = res.final_fraction
        out["prevalence"][r] = res.mean_prevalence
        out["peak_prev"][r] = res.peak_prevalence
        out["final_prev"][r] = res.final_prevalence
        out["events"][r] = res.events_per_node
        out["persisted"][r] = float(res.persisted)
        out["curves"].append(res.infected_counts)
    return out


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_epidemic_interventions(
    models: list[str] = STEP20_MODELS,
    datasets: list[tuple[str, str]] = TARGET_DATASETS,
    k_percent: float = INTERVENTION_BUDGET_PERCENT,
    n_runs: int = N_SIR_RUNS,
    recovery_snapshots: int = SIR_RECOVERY_SNAPSHOTS,
    immunity_snapshots: int = SIRS_IMMUNITY_SNAPSHOTS,
    target_outbreak: float = SIR_TARGET_OUTBREAK,
    shared_beta: bool = False,
    m: int = STEP18_M,
    num_paths: int = STEP18_NUM_PATHS,
    min_records: int = INTERVENTION_MIN_RECORDS,
    select_day: int = 0,
    eval_day: int = 0,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    plot_trajectories: bool = True,
) -> None:
    """Evaluate every isolation strategy under every requested epidemic model.

    Parameters
    ----------
    models : list of {"sir", "sis", "sirs"}
        Epidemic models to run.
    datasets : list of (display_name, dataset_key)
        Datasets to evaluate.
    k_percent : float
        Isolation budget as a percentage of nodes.
    n_runs : int
        Paired epidemic runs per (model, strategy).
    recovery_snapshots : int
        Infectious period in snapshots (all models).
    immunity_snapshots : int
        Immunity duration in snapshots (SIRS only).
    target_outbreak : float
        Target mean ever-infected fraction for beta calibration.
    shared_beta : bool
        If True, calibrate beta once on SIR and reuse it for every model.
    m, num_paths, min_records : int
        Memory horizon, number of sampled paths and minimum record count used to
        estimate node-level return rates.
    select_day, eval_day : int
        Day used to choose nodes and day used to simulate (equal = in-sample).
    n_boot : int
        Bootstrap resamples for confidence intervals.
    seed : int
        Master random seed.
    plot_trajectories : bool
        Save one mean-prevalence-over-time figure per dataset.
    """
    models = [mod.lower() for mod in models]
    bad = [mod for mod in models if mod not in VALID_MODELS]
    if bad:
        raise ValueError(f"Unknown model(s) {bad}; choose from {VALID_MODELS}.")

    print("=" * 80, flush=True)
    print("Step 20 -- Isolation interventions under " +
          ", ".join(MODEL_LABELS[x] for x in models), flush=True)
    print(
        f"Budget: {k_percent:.1f}% | Runs/strategy: {n_runs} | Infectious: "
        f"{recovery_snapshots} snaps | Immunity (SIRS): {immunity_snapshots} snaps | "
        f"{'shared beta' if shared_beta else 'per-model beta'} | Seed: {seed}",
        flush=True,
    )
    print("=" * 80, flush=True)

    rng = np.random.default_rng(seed)
    boot_rng = np.random.default_rng(seed + 1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    diag_rows: list[dict] = []

    for ds_name, ds_key in datasets:
        t0 = time.time()
        print(f"\n[{ds_name}]", flush=True)

        # 1. Data and graphs
        contacts, labels = load_contacts(ds_key)
        days = split_by_day(contacts, ds_key)
        n = len(labels) if labels is not None else int(contacts[["i", "j"]].values.max() + 1)

        sel_idx = select_day if select_day < len(days) else 0
        ev_idx = eval_day if eval_day < len(days) else 0
        if sel_idx != select_day or ev_idx != eval_day:
            print(f"  Note: dataset has {len(days)} day(s); using select={sel_idx}, "
                  f"eval={ev_idx}.", flush=True)

        g_sel = build_temporal_graph(
            days[sel_idx], n, labels, t_res=T_RES_SECONDS, name=f"{ds_key}_select"
        )
        g_eval = (
            g_sel if ev_idx == sel_idx
            else build_temporal_graph(
                days[ev_idx], n, labels, t_res=T_RES_SECONDS, name=f"{ds_key}_eval"
            )
        )
        k_nodes = max(1, int(round(n * k_percent / 100.0)))
        print(f"  Nodes n={n} | Budget k={k_nodes} ({k_percent:.1f}%)", flush=True)

        # 2. Node statistics and the isolation sets (independent of the model)
        paths = generate_many_trps(g_sel, num_paths=num_paths, length=PATH_LENGTH, rng=rng)
        records = build_records(paths, m=m)
        rates, counts = node_return_rates(records, n)
        contact_time = _total_contact_time(g_sel)
        eligible = counts >= min_records
        if eligible.sum() >= 3:
            rho_ct, _ = spearmanr(rates[eligible], contact_time[eligible])
        else:
            rho_ct = float("nan")
        diag_rows.append({
            "dataset": ds_name, "dataset_key": ds_key, "n_nodes": n,
            "n_ranked_nodes": int(eligible.sum()),
            "spearman_return_vs_contact_time": float(rho_ct),
        })

        strategies: list[tuple[str, list[int]]] = [
            ("None", []),
            ("Random", get_random_nodes(n=n, budget=k_nodes, rng=rng)),
            ("Contact Time", get_highest_contact_nodes(graph=g_sel, budget=k_nodes)),
            ("High Return Rate", get_highest_return_nodes(
                records=records, budget=k_nodes, n=n, min_records=min_records)),
            ("Low Return Rate", get_lowest_return_nodes(
                records=records, budget=k_nodes, n=n, min_records=min_records)),
        ]
        group_idx = -1
        if labels is not None:
            group_idx, comm_nodes = get_community_nodes(labels, target_budget=k_nodes)
            strategies.append(("Close Group", comm_nodes))

        # 3. Paired start conditions shared by every strategy and every model
        total_snaps = len(g_eval.snapshots)
        starts = rng.integers(0, total_snaps, size=n_runs)
        base_seeds = rng.integers(0, n, size=n_runs)

        # 4. Transmission rate(s)
        betas: dict[str, float] = {}
        if shared_beta:
            shared = calibrate_rate_model(
                g_eval, model="sir", target_final_fraction=target_outbreak,
                recovery_snapshots=recovery_snapshots,
                immunity_snapshots=immunity_snapshots,
                n_calib_runs=SIR_CALIBRATION_RUNS, rng=rng,
            )
            betas = {mod: shared for mod in models}
            print(f"  Shared beta (calibrated on SIR): {shared:.5f}", flush=True)
        else:
            for mod in models:
                betas[mod] = calibrate_rate_model(
                    g_eval, model=mod, target_final_fraction=target_outbreak,
                    recovery_snapshots=recovery_snapshots,
                    immunity_snapshots=immunity_snapshots,
                    n_calib_runs=SIR_CALIBRATION_RUNS, rng=rng,
                )
            print("  Calibrated beta: " + " | ".join(
                f"{MODEL_LABELS[mod]} {betas[mod]:.5f}" for mod in models), flush=True)

        # 5. Simulate every model x strategy
        trajectories: dict[str, dict[str, np.ndarray]] = {}

        for mod in models:
            outcomes = {
                name: _simulate_strategy(
                    g_eval, mod, betas[mod], recovery_snapshots, immunity_snapshots,
                    rem, starts, base_seeds, rng,
                )
                for name, rem in strategies
            }
            base = outcomes["None"]

            if base["outbreak"].mean() < CALIBRATION_WARN_FRACTION * target_outbreak:
                print(
                    f"  WARNING [{MODEL_LABELS[mod]}]: baseline outbreak "
                    f"{base['outbreak'].mean()*100:.1f}% is far below the "
                    f"{target_outbreak*100:.0f}% target (beta={betas[mod]:.2f}). "
                    f"Calibration probably hit its ceiling; try a larger "
                    f"--recovery-snapshots.", flush=True)

            print(f"\n  --- {MODEL_LABELS[mod]} ---", flush=True)
            print(f"  {'Strategy':<18}{'Removed':>8}{'Outbreak':>11}{'Prevalence':>12}"
                  f"{'Persist':>9}{'dOutbreak':>17}{'dPrevalence':>17}", flush=True)
            print(f"  {'-' * 92}", flush=True)

            trajectories[mod] = {}
            for name, rem in strategies:
                o = outcomes[name]
                ob_lo, ob_hi = _bootstrap_mean_ci(o["outbreak"], boot_rng, n_boot)
                pv_lo, pv_hi = _bootstrap_mean_ci(o["prevalence"], boot_rng, n_boot)
                ob_abs, ob_rel, ob_rlo, ob_rhi = _paired_reduction(
                    base["outbreak"], o["outbreak"], boot_rng, n_boot)
                pv_abs, pv_rel, pv_rlo, pv_rhi = _paired_reduction(
                    base["prevalence"], o["prevalence"], boot_rng, n_boot)

                trajectories[mod][name] = _mean_trajectory(o["curves"], n)

                print(
                    f"  {name:<18}{len(rem):>8d}{o['outbreak'].mean()*100:>10.1f}%"
                    f"{o['prevalence'].mean()*100:>11.1f}%"
                    f"{o['persisted'].mean()*100:>8.0f}%"
                    f"{f'{ob_rel:+.0f}% [{ob_rlo:+.0f},{ob_rhi:+.0f}]':>17}"
                    f"{f'{pv_rel:+.0f}% [{pv_rlo:+.0f},{pv_rhi:+.0f}]':>17}",
                    flush=True,
                )

                all_rows.append({
                    "model": mod,
                    "dataset": ds_name,
                    "dataset_key": ds_key,
                    "n_nodes": n,
                    "select_day": sel_idx,
                    "eval_day": ev_idx,
                    "k_budget_percent": k_percent,
                    "recovery_snapshots": recovery_snapshots,
                    "immunity_snapshots": immunity_snapshots if mod == "sirs" else np.nan,
                    "shared_beta": shared_beta,
                    "calibrated_rate": betas[mod],
                    "strategy": name,
                    "n_removed": len(rem),
                    "closed_group_index": group_idx if name == "Close Group" else -1,
                    # outbreak size (fraction ever infected)
                    "mean_outbreak": float(o["outbreak"].mean()),
                    "std_outbreak": float(o["outbreak"].std()),
                    "ci_low": ob_lo,
                    "ci_high": ob_hi,
                    "p_major_outbreak": float((o["outbreak"] >= MAJOR_OUTBREAK_THRESHOLD).mean()),
                    "reduction_abs": ob_abs,
                    "reduction_vs_none_percent": ob_rel,
                    "reduction_ci_low_percent": ob_rlo,
                    "reduction_ci_high_percent": ob_rhi,
                    # prevalence (infection burden)
                    "mean_prevalence": float(o["prevalence"].mean()),
                    "std_prevalence": float(o["prevalence"].std()),
                    "prev_ci_low": pv_lo,
                    "prev_ci_high": pv_hi,
                    "prev_reduction_abs": pv_abs,
                    "prev_reduction_vs_none_percent": pv_rel,
                    "prev_reduction_ci_low_percent": pv_rlo,
                    "prev_reduction_ci_high_percent": pv_rhi,
                    # persistence and reinfection
                    "mean_peak_prevalence": float(o["peak_prev"].mean()),
                    "mean_final_prevalence": float(o["final_prev"].mean()),
                    "p_persist": float(o["persisted"].mean()),
                    "mean_events_per_node": float(o["events"].mean()),
                    # diagnostics
                    "removed_mean_return_rate": (
                        float(np.nanmean(rates[rem])) if len(rem) else float("nan")),
                })

        if plot_trajectories:
            traj_path = RESULTS_DIR / f"step20_trajectories_{ds_key}.png"
            _plot_trajectories(trajectories, models, ds_name, strategies, traj_path)

        print(f"\n  Completed in {time.time() - t0:.1f}s", flush=True)

    # -----------------------------------------------------------------------
    # Save tables and figures
    # -----------------------------------------------------------------------
    df_all = pd.DataFrame(all_rows)
    csv_all = RESULTS_DIR / "step20_interventions_all_models.csv"
    df_all.to_csv(csv_all, index=False)
    pd.DataFrame(diag_rows).to_csv(RESULTS_DIR / "step20_return_diagnostics.csv", index=False)

    print(f"\n{'=' * 80}", flush=True)
    print(f"Saved -> {csv_all}", flush=True)

    for mod in models:
        sub = df_all[df_all["model"] == mod]
        sub.to_csv(RESULTS_DIR / f"step20_{mod}_interventions.csv", index=False)
        _plot_grid(sub, "mean_outbreak", "ci_low", "ci_high",
                   "Fraction ever infected",
                   f"{MODEL_LABELS[mod]}: outbreak size per strategy "
                   f"(error bars = 95% bootstrap CI)",
                   RESULTS_DIR / f"step20_{mod}_grid_outbreak.png")
        _plot_grid(sub, "mean_prevalence", "prev_ci_low", "prev_ci_high",
                   "Mean prevalence (time-averaged fraction infected)",
                   f"{MODEL_LABELS[mod]}: infection burden per strategy "
                   f"(error bars = 95% bootstrap CI)",
                   RESULTS_DIR / f"step20_{mod}_grid_prevalence.png")

    _plot_reduction_by_model(
        df_all, models, "reduction_vs_none_percent",
        "reduction_ci_low_percent", "reduction_ci_high_percent",
        "Reduction in outbreak size vs None (%)",
        "Paired reduction in outbreak size (higher is better, 95% CI)",
        RESULTS_DIR / "step20_reduction_outbreak.png",
    )
    _plot_reduction_by_model(
        df_all, models, "prev_reduction_vs_none_percent",
        "prev_reduction_ci_low_percent", "prev_reduction_ci_high_percent",
        "Reduction in mean prevalence vs None (%)",
        "Paired reduction in infection burden (higher is better, 95% CI)",
        RESULTS_DIR / "step20_reduction_prevalence.png",
    )
    print(f"Saved figures -> {RESULTS_DIR}", flush=True)

    _print_summary(df_all, models)


# ---------------------------------------------------------------------------
# Summary and plotting
# ---------------------------------------------------------------------------


def _print_summary(df_all: pd.DataFrame, models: list[str]) -> None:
    """Print and save the mean paired reduction per model and strategy."""
    summary = (
        df_all[df_all["strategy"] != "None"]
        .groupby(["model", "strategy"])[
            ["reduction_vs_none_percent", "prev_reduction_vs_none_percent"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "step20_summary_by_model.csv", index=False)

    print("\nMean paired reduction vs None, averaged over datasets", flush=True)
    for mod in models:
        print(f"\n  {MODEL_LABELS[mod]}", flush=True)
        print(f"  {'Strategy':<18}{'Outbreak':>10}{'Prevalence':>12}", flush=True)
        sub = summary[summary["model"] == mod].set_index("strategy")
        for strat in [s for s in STRATEGY_ORDER if s in sub.index]:
            r = sub.loc[strat]
            print(f"  {strat:<18}{r['reduction_vs_none_percent']:>+9.1f}%"
                  f"{r['prev_reduction_vs_none_percent']:>+11.1f}%", flush=True)

    # Does the best strategy depend on the model?
    print("\n  Best strategy by prevalence reduction:", flush=True)
    for mod in models:
        sub = summary[summary["model"] == mod]
        if len(sub):
            best = sub.loc[sub["prev_reduction_vs_none_percent"].idxmax()]
            print(f"    {MODEL_LABELS[mod]:<5} {best['strategy']} "
                  f"({best['prev_reduction_vs_none_percent']:+.1f}%)", flush=True)


def _plot_grid(
    df_model: pd.DataFrame, value_col: str, lo_col: str, hi_col: str,
    ylabel: str, title: str, output_path: Path,
) -> None:
    """One panel per dataset: mean of ``value_col`` per strategy with 95% CI."""
    keys = list(dict.fromkeys(df_model["dataset_key"]))
    ncols = 4
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)

    for ax, key in zip(axes.ravel(), keys):
        sub = df_model[df_model["dataset_key"] == key].set_index("strategy")
        order = [s for s in STRATEGY_ORDER if s in sub.index]
        sub = sub.loc[order]
        means = sub[value_col].to_numpy()
        yerr = np.vstack([means - sub[lo_col].to_numpy(), sub[hi_col].to_numpy() - means])
        x = np.arange(len(order))
        ax.bar(x, means, yerr=np.nan_to_num(yerr), capsize=3,
               color=[STRATEGY_COLORS[s] for s in order])
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_title(sub["dataset"].iloc[0], fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_reduction_by_model(
    df_all: pd.DataFrame, models: list[str], red_col: str, lo_col: str, hi_col: str,
    ylabel: str, title: str, output_path: Path,
) -> None:
    """Grouped bars of paired reduction per dataset, one panel per epidemic model."""
    strategies = [s for s in STRATEGY_ORDER
                  if s != "None" and s in set(df_all["strategy"])]
    keys = list(dict.fromkeys(df_all["dataset_key"]))
    names = [df_all[df_all["dataset_key"] == k]["dataset"].iloc[0] for k in keys]

    fig, axes = plt.subplots(
        1, len(models), figsize=(max(6.5, 1.5 * len(keys) + 2) * len(models) / 1.6, 5),
        sharey=True, squeeze=False,
    )
    width = 0.8 / len(strategies)
    x = np.arange(len(keys))

    for ax, mod in zip(axes.ravel(), models):
        dm = df_all[df_all["model"] == mod]
        for i, strat in enumerate(strategies):
            sub = dm[dm["strategy"] == strat].set_index("dataset_key").reindex(keys)
            vals = sub[red_col].to_numpy()
            yerr = np.vstack([vals - sub[lo_col].to_numpy(), sub[hi_col].to_numpy() - vals])
            ax.bar(x + (i - (len(strategies) - 1) / 2) * width, vals, width,
                   yerr=np.nan_to_num(yerr), capsize=2, label=strat,
                   color=STRATEGY_COLORS[strat])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_title(MODEL_LABELS[mod], fontweight="bold", color=MODEL_COLORS[mod])
        ax.grid(axis="y", alpha=0.3)

    axes[0, 0].set_ylabel(ylabel)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=len(strategies), fontsize=9,
               frameon=False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_trajectories(
    trajectories: dict[str, dict[str, np.ndarray]], models: list[str],
    ds_name: str, strategies: list[tuple[str, list[int]]], output_path: Path,
) -> None:
    """Mean fraction infected versus time since seeding, per strategy, per model."""
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4), squeeze=False)
    minutes_per_snap = T_RES_SECONDS / 60.0

    for ax, mod in zip(axes.ravel(), models):
        for name, _ in strategies:
            y = trajectories[mod][name]
            ax.plot(np.arange(len(y)) * minutes_per_snap, y * 100.0,
                    label=name, color=STRATEGY_COLORS[name], linewidth=1.6)
        ax.set_title(MODEL_LABELS[mod], fontweight="bold", color=MODEL_COLORS[mod])
        ax.set_xlabel("Minutes since seeding")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Mean % of nodes infected")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"{ds_name}: mean prevalence over time (late-seeded runs drop out of the "
        f"average at long times)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 20: isolation interventions under SIR, SIS and SIRS dynamics."
    )
    parser.add_argument("--models", nargs="+", default=STEP20_MODELS, choices=list(VALID_MODELS),
                        help="Epidemic models to run (default: sir sis sirs)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset keys to run (default: all 8), e.g. primaryschool hs2013")
    parser.add_argument("--k-percent", type=float, default=INTERVENTION_BUDGET_PERCENT)
    parser.add_argument("--runs", type=int, default=N_SIR_RUNS)
    parser.add_argument("--recovery-snapshots", type=int, default=SIR_RECOVERY_SNAPSHOTS)
    parser.add_argument("--immunity-snapshots", type=int, default=SIRS_IMMUNITY_SNAPSHOTS,
                        help="SIRS immunity duration in snapshots")
    parser.add_argument("--target-outbreak", type=float, default=SIR_TARGET_OUTBREAK)
    parser.add_argument("--shared-beta", action="store_true",
                        help="Use one beta (calibrated on SIR) for all models")
    parser.add_argument("--m", type=int, default=STEP18_M)
    parser.add_argument("--num-paths", type=int, default=STEP18_NUM_PATHS)
    parser.add_argument("--min-records", type=int, default=INTERVENTION_MIN_RECORDS)
    parser.add_argument("--select-day", type=int, default=0,
                        help="Day index used to choose the nodes to isolate")
    parser.add_argument("--eval-day", type=int, default=0,
                        help="Day index used to simulate the epidemic")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-trajectories", action="store_true",
                        help="Skip the per-dataset trajectory figures")
    args = parser.parse_args()

    chosen = TARGET_DATASETS
    if args.datasets:
        unknown = set(args.datasets) - {key for _, key in TARGET_DATASETS}
        if unknown:
            parser.error(f"Unknown dataset key(s): {sorted(unknown)}")
        chosen = [(name, key) for name, key in TARGET_DATASETS if key in args.datasets]

    run_epidemic_interventions(
        models=args.models,
        datasets=chosen,
        k_percent=args.k_percent,
        n_runs=args.runs,
        recovery_snapshots=args.recovery_snapshots,
        immunity_snapshots=args.immunity_snapshots,
        target_outbreak=args.target_outbreak,
        shared_beta=args.shared_beta,
        m=args.m,
        num_paths=args.num_paths,
        min_records=args.min_records,
        select_day=args.select_day,
        eval_day=args.eval_day,
        seed=args.seed,
        plot_trajectories=not args.no_trajectories,
    )

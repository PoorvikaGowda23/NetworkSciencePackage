"""
plotting.py
===========
Plotting and visualization routines for temporal network memory analysis.

Functions
---------
plot_memory_curves
    Plot estimated memory parameter p versus memory horizon m for empirical and null models.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_memory_curves(
    m_values: list[int],
    p_emp: list[float],
    p_null: list[float],
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plot estimated memory parameter p versus memory horizon m.

    Compares the empirical MEM curve against the Erdos-Renyi null model curve,
    matching the format of Figure 3 from Guerrini et al. (2026).

    Parameters
    ----------
    m_values : list[int]
        List of memory horizon values (steps).
    p_emp : list[float]
        Fitted memory parameter p for empirical temporal graphs.
    p_null : list[float]
        Fitted memory parameter p for Erdos-Renyi null temporal graphs.
    dataset_name : str
        Human-readable title/name of the dataset.
    output_path : Path
        Filesystem path where the figure will be saved.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

    # Empirical curve (solid line with squares, matching paper style)
    ax.plot(
        m_values,
        p_emp,
        marker="s",
        color="#1f77b4",
        linestyle="-",
        linewidth=1.8,
        markersize=6,
        label="MEM (Empirical)",
    )

    # Erdos-Renyi null curve (dashed line with open squares, matching paper style)
    ax.plot(
        m_values,
        p_null,
        marker="s",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
        markersize=6,
        markerfacecolor="none",
        label="Null MEM (Erdős-Rényi)",
    )

    ax.set_xlabel("Memory length $m$", fontsize=12)
    ax.set_ylabel("Inferred memory $p$", fontsize=12)
    ax.set_title(f"Memory Parameter vs Horizon: {dataset_name}", fontsize=13)
    ax.set_xticks(m_values)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)

    # Ensure y-axis covers reasonable range
    y_max = max(max(p_emp), max(p_null), 0.1) * 1.15
    ax.set_ylim(-0.02, min(1.0, y_max))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_bic_panels(
    dataset_results: dict[str, dict[str, list[float] | int]],
    m_values: list[int],
    output_path: Path,
) -> None:
    """Plot BIC comparison panels across multiple datasets (Figure 2 style).

    Creates a 2x3 grid of subplots, one for each dataset, comparing the BIC
    curve of the plain MEM model against the MEM+SBM model across memory horizons.

    Parameters
    ----------
    dataset_results : dict[str, dict[str, list[float] | int]]
        Dictionary mapping dataset display name to:
            {"bic_mem": [...], "bic_sbm": [...], "k": int, "n": int}
    m_values : list[int]
        List of memory horizon values (steps).
    output_path : Path
        Filesystem path to save the generated figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    axes_flat = axes.flatten()

    for idx, (dname, data) in enumerate(dataset_results.items()):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]

        ax.plot(
            m_values,
            data["bic_mem"],
            marker="o",
            color="#1f77b4",
            linestyle="-",
            linewidth=1.8,
            markersize=6,
            label="MEM",
        )
        ax.plot(
            m_values,
            data["bic_sbm"],
            marker="s",
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            markersize=6,
            label="MEM+SBM",
        )

        k_val = data.get("k", "")
        n_val = data.get("n", "")
        title_extra = []
        if n_val:
            title_extra.append(f"$n={n_val}$")
        if k_val:
            title_extra.append(f"$K={k_val}$")
        extra_str = f" ({', '.join(title_extra)})" if title_extra else ""

        ax.set_title(f"{dname}{extra_str}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Memory horizon $m$", fontsize=10)
        ax.set_ylabel("BIC", fontsize=10)
        ax.set_xticks(m_values)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Hide any unused axes
    for idx in range(len(dataset_results), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_resolution_heatmap(
    matrix: np.ndarray,
    t_res_list: list[int],
    m_list: list[int],
    model_name: str,
    dataset_name: str,
    output_path: Path,
    cmap: str = "Blues",
) -> None:
    """Plot a heatmap of estimated memory parameter p across resolutions and horizons.

    Produces a heatmap with numerical values printed in each cell, corresponding
    to Figure 5 (MEM) or Figure 6 (MEM+SBM) from Guerrini et al. (2026).

    Parameters
    ----------
    matrix : np.ndarray
        2D array of shape (len(t_res_list), len(m_list)) with inferred p values.
    t_res_list : list[int]
        Temporal resolutions (aggregation windows in seconds) for the rows.
    m_list : list[int]
        Memory horizons (steps) for the columns.
    model_name : str
        Model identifier ('MEM' or 'MEM+SBM').
    dataset_name : str
        Dataset title (e.g. 'Primary School').
    output_path : Path
        Filesystem path to save the generated figure.
    cmap : str, default='Blues'
        Matplotlib colormap name.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=300)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", origin="lower")

    # Set tick labels
    ax.set_xticks(range(len(m_list)))
    ax.set_xticklabels(m_list, fontsize=11)
    ax.set_yticks(range(len(t_res_list)))
    ax.set_yticklabels(t_res_list, fontsize=11)

    ax.set_xlabel("Memory length $m$", fontsize=12)
    ax.set_ylabel("Aggregation window $\\Delta t$ (s)", fontsize=12)
    ax.set_title(f"Inferred Memory $p$ ({model_name}) - {dataset_name}", fontsize=13, fontweight="bold")

    # Annotate numbers in each cell
    max_val = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0
    threshold = max_val / 2.0
    for r in range(len(t_res_list)):
        for c in range(len(m_list)):
            val = matrix[r, c]
            if np.isnan(val):
                text_str = "N/A"
                text_color = "black"
            else:
                text_str = f"{val:.3f}"
                text_color = "white" if val > threshold else "black"
            ax.text(c, r, text_str, ha="center", va="center", color=text_color, fontsize=11, fontweight="semibold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Inferred memory $p$", fontsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_resolution_heatmaps_side_by_side(
    p_mem_matrix: np.ndarray,
    p_sbm_matrix: np.ndarray,
    t_res_list: list[int],
    m_list: list[int],
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plot side-by-side heatmaps of p for MEM (Figure 5) and MEM+SBM (Figure 6).

    Parameters
    ----------
    p_mem_matrix : np.ndarray
        2D array of inferred p for the MEM model across (t_res, m).
    p_sbm_matrix : np.ndarray
        2D array of inferred p for the MEM+SBM model across (t_res, m).
    t_res_list : list[int]
        Temporal resolutions (aggregation windows in seconds) for the rows.
    m_list : list[int]
        Memory horizons (steps) for the columns.
    dataset_name : str
        Dataset title.
    output_path : Path
        Filesystem path to save the generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=300)

    # Use shared color scale for direct visual comparison
    v_min = min(float(np.nanmin(p_mem_matrix)), float(np.nanmin(p_sbm_matrix)))
    v_max = max(float(np.nanmax(p_mem_matrix)), float(np.nanmax(p_sbm_matrix)))

    panels = [
        (axes[0], p_mem_matrix, "MEM (Fig. 5 style)", "Blues"),
        (axes[1], p_sbm_matrix, "MEM+SBM (Fig. 6 style)", "Reds"),
    ]

    for ax, mat, title, cmap in panels:
        im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower", vmin=v_min, vmax=v_max)
        ax.set_xticks(range(len(m_list)))
        ax.set_xticklabels(m_list, fontsize=11)
        ax.set_yticks(range(len(t_res_list)))
        ax.set_yticklabels(t_res_list, fontsize=11)
        ax.set_xlabel("Memory length $m$", fontsize=11)
        ax.set_ylabel("Aggregation window $\\Delta t$ (s)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

        threshold = (v_min + v_max) / 2.0
        for r in range(len(t_res_list)):
            for c in range(len(m_list)):
                val = mat[r, c]
                if np.isnan(val):
                    text_str = "N/A"
                    text_color = "black"
                else:
                    text_str = f"{val:.3f}"
                    text_color = "white" if val > threshold else "black"
                ax.text(c, r, text_str, ha="center", va="center", color=text_color, fontsize=10, fontweight="semibold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Inferred memory $p$", fontsize=10)

    fig.suptitle(f"Resolution & Memory Horizon Comparison: {dataset_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_tau_curves(
    tau_minutes: list[float],
    p_mem: list[float],
    p_sbm: list[float],
    dataset_name: str,
    output_path_mem: Path,
    output_path_sbm: Path,
    output_path_combined: Path,
) -> None:
    """Plot inferred memory parameter p versus time-based memory duration tau in minutes.

    Generates Figure 7 (MEM), Figure 8 (MEM+SBM), and a combined comparison figure.

    Parameters
    ----------
    tau_minutes : list[float]
        Memory look-back horizons in minutes.
    p_mem : list[float]
        Inferred p values under MEM.
    p_sbm : list[float]
        Inferred p values under MEM+SBM.
    dataset_name : str
        Dataset title.
    output_path_mem : Path
        Path to save Figure 7 (MEM).
    output_path_sbm : Path
        Path to save Figure 8 (MEM+SBM).
    output_path_combined : Path
        Path to save combined comparison plot.
    """
    # 1. Figure 7: MEM curve
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    ax.plot(
        tau_minutes,
        p_mem,
        marker="o",
        color="#1f77b4",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        label="MEM",
    )
    ax.set_xlabel("Memory time duration $\\tau$ (min)", fontsize=12)
    ax.set_ylabel("Inferred memory $p$", fontsize=12)
    ax.set_title(f"Time-Based Memory $p(\\tau)$ (MEM) - {dataset_name}", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(-0.02, min(1.0, max(p_mem) * 1.2 if len(p_mem) > 0 and max(p_mem) > 0 else 0.5))
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()
    output_path_mem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_mem)
    plt.close(fig)

    # 2. Figure 8: MEM+SBM curve
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    ax.plot(
        tau_minutes,
        p_sbm,
        marker="s",
        color="#d62728",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        label="MEM+SBM",
    )
    ax.set_xlabel("Memory time duration $\\tau$ (min)", fontsize=12)
    ax.set_ylabel("Inferred memory $p$", fontsize=12)
    ax.set_title(f"Time-Based Memory $p(\\tau)$ (MEM+SBM) - {dataset_name}", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(-0.02, min(1.0, max(p_sbm) * 1.2 if len(p_sbm) > 0 and max(p_sbm) > 0 else 0.5))
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()
    output_path_sbm.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_sbm)
    plt.close(fig)

    # 3. Combined comparison
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=300)
    ax.plot(
        tau_minutes,
        p_mem,
        marker="o",
        color="#1f77b4",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        label="MEM (Fig. 7)",
    )
    ax.plot(
        tau_minutes,
        p_sbm,
        marker="s",
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        markersize=6,
        label="MEM+SBM (Fig. 8)",
    )
    ax.set_xlabel("Memory time duration $\\tau$ (min)", fontsize=12)
    ax.set_ylabel("Inferred memory $p$", fontsize=12)
    ax.set_title(f"Time-Based Memory $p(\\tau)$: {dataset_name}", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    y_max = max(max(p_mem), max(p_sbm), 0.1) * 1.2
    ax.set_ylim(-0.02, min(1.0, y_max))
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()
    output_path_combined.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_combined)
    plt.close(fig)


def plot_diffusion_figure4(
    df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Draw the two-panel Figure 4 analogue (p vs alpha and entropy vs p).

    Left panel:  p_hat versus alpha (median + 5th-95th percentile band).
    Right panel: final entropy H versus p_hat (scatter + running median).
    One color per degree value d.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'd', 'alpha', 'p_hat', and 'avg_entropy'.
    output_path : Path
        Filesystem path for the saved figure.
    title_suffix : str, default=""
        Optional subtitle text detailing simulation parameters.
    """
    d_values = sorted(df["d"].unique())
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(d_values))]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # ---- Left panel: p vs alpha ----
    for d_val, color in zip(d_values, colors):
        sub = df[df["d"] == d_val]
        alphas_sorted = sorted(sub["alpha"].unique())

        medians, p5, p95 = [], [], []
        for alpha in alphas_sorted:
            vals = sub[sub["alpha"] == alpha]["p_hat"].values
            medians.append(float(np.median(vals)))
            p5.append(float(np.percentile(vals, 5)))
            p95.append(float(np.percentile(vals, 95)))

        ax_left.plot(
            alphas_sorted, medians,
            marker="o", markersize=5, linewidth=2.0,
            color=color, label=f"d = {d_val:.0f}",
        )
        ax_left.fill_between(
            alphas_sorted, p5, p95,
            color=color, alpha=0.18,
        )

    ax_left.set_xlabel("Memory mixing parameter $\\alpha$", fontsize=12)
    ax_left.set_ylabel("Inferred memory $\\hat{p}$", fontsize=12)
    ax_left.set_title("Memory $\\hat{p}$ vs. $\\alpha$", fontsize=13, fontweight="bold")
    ax_left.set_xlim(-0.02, 1.02)
    ax_left.set_ylim(-0.02, 1.02)
    ax_left.grid(True, linestyle=":", alpha=0.6)
    ax_left.legend(frameon=True, fontsize=10)

    # ---- Right panel: entropy vs p_hat ----
    for d_val, color in zip(d_values, colors):
        sub = df[df["d"] == d_val].sort_values("p_hat")

        ax_right.scatter(
            sub["p_hat"], sub["avg_entropy"],
            color=color, alpha=0.35, s=18, edgecolors="none",
            label=f"d = {d_val:.0f}",
        )

        p_vals = sub["p_hat"].values
        h_vals = sub["avg_entropy"].values
        if len(p_vals) >= 5:
            bins = np.linspace(p_vals.min(), p_vals.max(), 15)
            bin_idx = np.digitize(p_vals, bins)
            bin_medians_p, bin_medians_h = [], []
            for b in range(1, len(bins)):
                mask = bin_idx == b
                if mask.sum() >= 1:
                    bin_medians_p.append(float(np.median(p_vals[mask])))
                    bin_medians_h.append(float(np.median(h_vals[mask])))
            if len(bin_medians_p) >= 2:
                ax_right.plot(
                    bin_medians_p, bin_medians_h,
                    color=color, linewidth=2.2, zorder=5,
                )

    ax_right.set_xlabel("Inferred memory $\\hat{p}$", fontsize=12)
    ax_right.set_ylabel("Final entropy $H$", fontsize=12)
    ax_right.set_title("Diffusion Entropy $H$ vs. Memory $\\hat{p}$", fontsize=13, fontweight="bold")
    ax_right.set_xlim(-0.02, 1.02)
    ax_right.set_ylim(-0.02, 1.05)
    ax_right.grid(True, linestyle=":", alpha=0.6)
    ax_right.legend(frameon=True, fontsize=10)

    title = "Figure 4 analogue: Memory-driven spreading on synthetic networks"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_time_varying_p(
    daily_results: dict[str, dict[float, float]],
    dataset_name: str,
    output_path: Path,
    bin_minutes: int = 60,
    model: str = "MEM",
) -> None:
    """Plot inferred memory p versus clock time for a single dataset.

    Plots one thin line per day, and a thick bold line for the mean across days.

    Parameters
    ----------
    daily_results : dict[str, dict[float, float]]
        Mapping: day_label -> {clock_hour: p_hat}.
    dataset_name : str
        Human-readable name of the dataset.
    output_path : Path
        Filesystem path for the saved figure.
    bin_minutes : int, default=60
        Time bin resolution in minutes (e.g. 60 for hourly, 15 for 15-min).
    model : str, default="MEM"
        Model label ("MEM" or "MEM+SBM").
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)

    # Collect all unique clock hour points across days
    all_hours: set[float] = set()
    for day_curve in daily_results.values():
        all_hours.update(day_curve.keys())

    sorted_hours = sorted(all_hours)

    # 1. Plot individual days as thin lines
    cmap = plt.get_cmap("Blues")
    n_days = max(len(daily_results), 1)
    day_colors = [cmap(0.4 + 0.5 * (i / max(n_days - 1, 1))) for i in range(n_days)]

    for (day_label, day_curve), col in zip(daily_results.items(), day_colors):
        x = sorted(day_curve.keys())
        y = [day_curve[h] for h in x]
        ax.plot(
            x, y,
            linestyle="-",
            linewidth=1.2,
            alpha=0.6,
            color=col,
            label=f"{day_label} (thin)",
        )

    # 2. Compute mean across days for each clock hour
    mean_x: list[float] = []
    mean_y: list[float] = []
    for h in sorted_hours:
        vals = [daily_results[d][h] for d in daily_results if h in daily_results[d]]
        if vals:
            mean_x.append(h)
            mean_y.append(float(np.mean(vals)))

    # Plot mean as thick bold line
    if mean_x:
        ax.plot(
            mean_x, mean_y,
            marker="o",
            markersize=5,
            linestyle="-",
            linewidth=3.0,
            color="#d62728",
            zorder=10,
            label="Mean across days (thick)",
        )

    bin_label = f"{bin_minutes}-min" if bin_minutes != 60 else "Hourly"
    ax.set_title(
        f"Time-Varying Memory $p(t)$ ({model}, {bin_label}): {dataset_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Clock Time (Hour of Day)", fontsize=12)
    ax.set_ylabel("Inferred memory $p$", fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Format x ticks nicely
    if sorted_hours:
        min_h = int(np.floor(min(sorted_hours)))
        max_h = int(np.ceil(max(sorted_hours)))
        step = 1 if (max_h - min_h) <= 12 else 2
        tick_hours = list(range(min_h, max_h + 1, step))
        tick_labels = [f"{h:02d}:00" for h in tick_hours]
        ax.set_xticks(tick_hours)
        ax.set_xticklabels(tick_labels)

    ax.legend(frameon=True, fontsize=9, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_time_varying_grid(
    dataset_day_curves: dict[str, dict[str, dict[float, float]]],
    output_path: Path,
    bin_minutes: int = 60,
    model: str = "MEM",
) -> None:
    """Plot time-varying memory panels across multiple datasets in a multi-panel grid.

    Parameters
    ----------
    dataset_day_curves : dict
        Mapping: dataset_name -> {day_label: {clock_hour: p}}.
    output_path : Path
        Filesystem path for the saved grid figure.
    bin_minutes : int, default=60
        Bin width in minutes.
    model : str, default="MEM"
        Model label.
    """
    n_datasets = len(dataset_day_curves)
    if n_datasets == 0:
        return

    n_cols = 3 if n_datasets >= 3 else n_datasets
    n_rows = int(np.ceil(n_datasets / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), dpi=300)
    axes_flat = np.array(axes).reshape(-1)

    for idx, (ds_name, daily_results) in enumerate(dataset_day_curves.items()):
        ax = axes_flat[idx]
        all_hours: set[float] = set()
        for day_curve in daily_results.values():
            all_hours.update(day_curve.keys())
        sorted_hours = sorted(all_hours)

        cmap = plt.get_cmap("Blues")
        n_days = max(len(daily_results), 1)
        day_colors = [cmap(0.4 + 0.5 * (i / max(n_days - 1, 1))) for i in range(n_days)]

        for (day_label, day_curve), col in zip(daily_results.items(), day_colors):
            x = sorted(day_curve.keys())
            y = [day_curve[h] for h in x]
            ax.plot(x, y, linestyle="-", linewidth=1.1, alpha=0.5, color=col)

        mean_x, mean_y = [], []
        for h in sorted_hours:
            vals = [daily_results[d][h] for d in daily_results if h in daily_results[d]]
            if vals:
                mean_x.append(h)
                mean_y.append(float(np.mean(vals)))

        if mean_x:
            ax.plot(mean_x, mean_y, marker="o", markersize=4, linestyle="-", linewidth=2.5, color="#d62728", zorder=10)

        ax.set_title(ds_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Clock Time", fontsize=10)
        ax.set_ylabel("Inferred $p$", fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)

        if sorted_hours:
            min_h = int(np.floor(min(sorted_hours)))
            max_h = int(np.ceil(max(sorted_hours)))
            step = 2 if (max_h - min_h) > 8 else 1
            tick_hours = list(range(min_h, max_h + 1, step))
            ax.set_xticks(tick_hours)
            ax.set_xticklabels([f"{h:02d}:00" for h in tick_hours], fontsize=8)

    # Hide extra subplots
    for j in range(n_datasets, len(axes_flat)):
        axes_flat[j].set_visible(False)

    bin_label = f"{bin_minutes}-min" if bin_minutes != 60 else "Hourly"
    fig.suptitle(
        f"Time-Varying Memory $p(t)$ across Datasets ({model}, {bin_label})\n"
        "Thin blue lines = individual days; thick red line = across-day mean",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_memory_curves_with_ci(
    m_values: list[int],
    p_emp: list[float],
    ci_emp_low: list[float],
    ci_emp_high: list[float],
    p_null: list[float],
    ci_null_low: list[float],
    ci_null_high: list[float],
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plot estimated memory parameter p versus memory horizon m with 95% bootstrap bands."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=300)

    # Empirical curve with confidence band
    ax.plot(
        m_values, p_emp,
        marker="s", color="#1f77b4", linestyle="-", linewidth=2.0, markersize=6,
        label="MEM (Empirical)",
    )
    ax.fill_between(
        m_values, ci_emp_low, ci_emp_high,
        color="#1f77b4", alpha=0.2, label="95% Bootstrap CI (Empirical)",
    )

    # Null curve with confidence band
    ax.plot(
        m_values, p_null,
        marker="s", color="#ff7f0e", linestyle="--", linewidth=1.8, markersize=6,
        markerfacecolor="none", label="Null MEM (Erdős-Rényi)",
    )
    ax.fill_between(
        m_values, ci_null_low, ci_null_high,
        color="#ff7f0e", alpha=0.15, label="95% Bootstrap CI (Null)",
    )

    ax.set_xlabel("Memory length $m$", fontsize=12)
    ax.set_ylabel("Inferred memory $p$", fontsize=12)
    ax.set_title(f"Memory Parameter vs Horizon: {dataset_name}", fontsize=13, fontweight="bold")
    ax.set_xticks(m_values)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)

    y_max = max(max(ci_emp_high), max(ci_null_high), 0.1) * 1.15
    ax.set_ylim(-0.02, min(1.0, y_max))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_memory_grid_with_ci(
    dataset_curves: dict[str, dict[str, list[float]]],
    m_values: list[int],
    output_path: Path,
) -> None:
    """Plot p vs m curves with 95% bootstrap bands across all datasets in a multi-panel grid."""
    n_datasets = len(dataset_curves)
    if n_datasets == 0:
        return

    n_cols = 3 if n_datasets >= 3 else n_datasets
    n_rows = int(np.ceil(n_datasets / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows), dpi=300)
    axes_flat = np.array(axes).reshape(-1)

    for idx, (ds_name, curves) in enumerate(dataset_curves.items()):
        ax = axes_flat[idx]
        p_emp = curves["p_emp"]
        ci_emp_low = curves["ci_emp_low"]
        ci_emp_high = curves["ci_emp_high"]
        p_null = curves["p_null"]
        ci_null_low = curves["ci_null_low"]
        ci_null_high = curves["ci_null_high"]

        ax.plot(m_values, p_emp, marker="s", color="#1f77b4", linestyle="-", linewidth=1.8, markersize=5, label="Empirical")
        ax.fill_between(m_values, ci_emp_low, ci_emp_high, color="#1f77b4", alpha=0.2)

        ax.plot(m_values, p_null, marker="s", color="#ff7f0e", linestyle="--", linewidth=1.5, markersize=5, markerfacecolor="none", label="ER Null")
        ax.fill_between(m_values, ci_null_low, ci_null_high, color="#ff7f0e", alpha=0.15)

        ax.set_title(ds_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Memory length $m$", fontsize=10)
        ax.set_ylabel("Inferred $p$", fontsize=10)
        ax.set_xticks(m_values)
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(frameon=True, fontsize=9)

        y_max = max(max(ci_emp_high), max(ci_null_high), 0.1) * 1.15
        ax.set_ylim(-0.02, min(1.0, y_max))

    for j in range(n_datasets, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Memory Parameter $\\hat{p}$ vs. Memory Length $m$ across Datasets\n"
        "(Shaded bands indicate 95% non-parametric bootstrap confidence intervals)",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_null_comparison_barchart(
    df_comparison: pd.DataFrame,
    fixed_m: int,
    output_path: Path,
) -> None:
    """Plot grouped bar chart with bootstrap error bars comparing Real, Time-Shuffled, and ER Null.

    Parameters
    ----------
    df_comparison : pd.DataFrame
        DataFrame with columns:
          - dataset: display name of dataset
          - p_real, p_real_low, p_real_high
          - p_shuffled, p_shuffled_low, p_shuffled_high
          - p_er, p_er_low, p_er_high
    fixed_m : int
        The fixed horizon length m.
    output_path : Path
        Filesystem path for saved bar chart figure.
    """
    datasets = df_comparison["dataset"].values
    n_ds = len(datasets)
    x = np.arange(n_ds)
    bar_width = 0.26

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)

    # 1. Real Graph
    p_real = df_comparison["p_real"].values
    err_real = [
        p_real - df_comparison["p_real_low"].values,
        df_comparison["p_real_high"].values - p_real,
    ]
    ax.bar(
        x - bar_width, p_real,
        width=bar_width, yerr=err_real, capsize=4,
        color="#1f77b4", edgecolor="black", linewidth=0.8,
        label="Empirical Network",
    )

    # 2. Time-Shuffled Null
    p_shuff = df_comparison["p_shuffled"].values
    err_shuff = [
        p_shuff - df_comparison["p_shuffled_low"].values,
        df_comparison["p_shuffled_high"].values - p_shuff,
    ]
    ax.bar(
        x, p_shuff,
        width=bar_width, yerr=err_shuff, capsize=4,
        color="#2ca02c", edgecolor="black", linewidth=0.8,
        label="Time-Shuffled Null (destroys temporal order)",
    )

    # 3. Erdős-Rényi Null
    p_er = df_comparison["p_er"].values
    err_er = [
        p_er - df_comparison["p_er_low"].values,
        df_comparison["p_er_high"].values - p_er,
    ]
    ax.bar(
        x + bar_width, p_er,
        width=bar_width, yerr=err_er, capsize=4,
        color="#ff7f0e", edgecolor="black", linewidth=0.8,
        label="Erdős-Rényi Null (destroys network structure)",
    )

    ax.set_ylabel("Inferred memory parameter $\\hat{p}$", fontsize=12)
    ax.set_title(
        f"Impact of Null Models on Memory $\\hat{{p}}$ at $m={fixed_m}$\n"
        "(Error bars represent 95% bootstrap confidence intervals)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_p_per_group(
    df: "pd.DataFrame",
    dataset_name: str,
    group_names: "dict[int, str] | None",
    output_path: Path,
    global_p: "float | None" = None,
) -> None:
    """Bar chart of per-community memory parameter p_c with 95% bootstrap CIs.

    Each bar shows the fitted p_c for one community group (e.g. a school class,
    a hospital role, or a workplace department).  Error bars span the 95%
    non-parametric bootstrap confidence interval.  Each bar is annotated with
    the number of transition records used so the reader can judge reliability.
    An optional horizontal dashed line marks the population-level global p.

    Parameters
    ----------
    df : pd.DataFrame
        Output of fit_p_per_group: columns group, n_records, p_hat, ci_low,
        ci_high.
    dataset_name : str
        Human-readable dataset name for the plot title.
    group_names : dict[int, str] or None
        Mapping from integer community index to a display label (e.g. class
        name or role).  If None, bars are labelled as 'Group k'.
    output_path : Path
        Filesystem path where the figure is saved.
    global_p : float or None, default=None
        If provided, a horizontal dashed line is drawn at this value to show
        the population-level memory estimate for comparison.
    """
    if len(df) == 0:
        return

    # Figure width scales with the number of groups so labels stay readable
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(df)), 5.2), dpi=300)

    x = np.arange(len(df))
    p_vals = df["p_hat"].values
    err_low  = p_vals - df["ci_low"].values
    err_high = df["ci_high"].values - p_vals
    yerr = [err_low, err_high]

    bars = ax.bar(
        x,
        p_vals,
        yerr=yerr,
        capsize=5,
        color="#4c8be0",
        edgecolor="#1a4a8a",
        linewidth=0.9,
        error_kw={"elinewidth": 1.6, "ecolor": "#333333"},
    )

    # Annotate each bar top with the number of records it was fitted on.
    # Placement: just above the upper error bar so it never overlaps the bar.
    for rect, n_rec, eh in zip(bars, df["n_records"].values, err_high):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + eh + 0.016,
            f"n={n_rec:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )

    # Optional global-p reference line
    if global_p is not None:
        ax.axhline(
            global_p,
            linestyle="--",
            color="#d62728",
            linewidth=1.8,
            label=f"Global $\\hat{{p}}$ = {global_p:.3f}",
        )
        ax.legend(frameon=True, fontsize=10)

    # X-axis tick labels from the supplied name mapping or fallback integers
    if group_names is not None:
        tick_labels = [group_names.get(int(g), f"Group {g}") for g in df["group"].values]
    else:
        tick_labels = [f"Group {g}" for g in df["group"].values]

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Per-community memory $\\hat{p}_c$", fontsize=12)
    ax.set_title(
        f"Per-Community Memory Parameter $\\hat{{p}}_c$: {dataset_name}",
        fontsize=13,
        fontweight="bold",
    )

    # y-axis headroom: above the tallest bar + its upper CI + record-count text
    y_top = float(np.max(p_vals + err_high)) + 0.10
    ax.set_ylim(0.0, min(1.05, max(y_top, 0.15)))
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 17 — SIR epidemic spreading plots
# ---------------------------------------------------------------------------


def plot_sir_outbreak_comparison(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped bar chart comparing final outbreak size: Real vs Time-Shuffled.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
          - dataset: display name of dataset
          - mean_outbreak_real, std_outbreak_real
          - mean_outbreak_shuff, std_outbreak_shuff
    output_path : Path
        Filesystem path where the figure is saved.
    """
    datasets = df["dataset"].values
    n_ds = len(datasets)
    x = np.arange(n_ds)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * n_ds), 5.2), dpi=300)

    # 1. Real Graph
    ax.bar(
        x - bar_width / 2.0,
        df["mean_outbreak_real"].values,
        width=bar_width,
        yerr=df["std_outbreak_real"].values,
        capsize=4,
        color="#1f77b4",
        edgecolor="#0f4c81",
        linewidth=0.9,
        label="Empirical Network",
    )

    # 2. Time-Shuffled Null
    ax.bar(
        x + bar_width / 2.0,
        df["mean_outbreak_shuff"].values,
        width=bar_width,
        yerr=df["std_outbreak_shuff"].values,
        capsize=4,
        color="#2ca02c",
        edgecolor="#1b661b",
        linewidth=0.9,
        label="Time-Shuffled Null (destroys temporal order & memory)",
    )

    ax.set_ylabel("Mean Final Outbreak Size (Fraction)", fontsize=12)
    ax.set_title(
        "SIR Epidemic: Mean Outbreak Size (Real vs. Time-Shuffled)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_sir_time_to_peak(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped bar chart comparing time to infection peak: Real vs Time-Shuffled.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
          - dataset: display name of dataset
          - mean_time_to_peak_real, std_time_to_peak_real
          - mean_time_to_peak_shuff, std_time_to_peak_shuff
    output_path : Path
        Filesystem path where the figure is saved.
    """
    datasets = df["dataset"].values
    n_ds = len(datasets)
    x = np.arange(n_ds)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * n_ds), 5.2), dpi=300)

    # 1. Real Graph
    ax.bar(
        x - bar_width / 2.0,
        df["mean_time_to_peak_real"].values,
        width=bar_width,
        yerr=df["std_time_to_peak_real"].values,
        capsize=4,
        color="#1f77b4",
        edgecolor="#0f4c81",
        linewidth=0.9,
        label="Empirical Network",
    )

    # 2. Time-Shuffled Null
    ax.bar(
        x + bar_width / 2.0,
        df["mean_time_to_peak_shuff"].values,
        width=bar_width,
        yerr=df["std_time_to_peak_shuff"].values,
        capsize=4,
        color="#2ca02c",
        edgecolor="#1b661b",
        linewidth=0.9,
        label="Time-Shuffled Null (destroys temporal order & memory)",
    )

    ax.set_ylabel("Mean Time to Peak (Snapshots)", fontsize=12)
    ax.set_title(
        "SIR Epidemic: Mean Time to Infection Peak (Real vs. Time-Shuffled)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_sir_outbreak_vs_memory(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter plot of final outbreak size vs inferred memory parameter p.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
          - dataset: display name of dataset
          - p_memory: inferred empirical memory parameter p
          - mean_outbreak_real: mean final outbreak fraction on real graph
    output_path : Path
        Filesystem path where the figure is saved.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)

    p_vals = df["p_memory"].values
    outbreaks = df["mean_outbreak_real"].values
    names = df["dataset"].values

    ax.scatter(
        p_vals,
        outbreaks,
        color="#d62728",
        s=80,
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
    )

    # Annotate points with dataset names
    for name, p_val, out_val in zip(names, p_vals, outbreaks):
        ax.annotate(
            name,
            (p_val, out_val),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
            fontweight="medium",
        )

    # Linear trendline if at least 2 datasets
    if len(p_vals) >= 2:
        z = np.polyfit(p_vals, outbreaks, 1)
        p_line = np.linspace(min(p_vals) - 0.03, max(p_vals) + 0.03, 50)
        ax.plot(
            p_line,
            np.polyval(z, p_line),
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label=f"Trend (slope: {z[0]:+.2f})",
        )
        ax.legend(frameon=True, fontsize=10)

    ax.set_xlabel("Inferred Memory Parameter $p$ ($m=10$)", fontsize=12)
    ax.set_ylabel("Empirical Mean Final Outbreak Size", fontsize=12)
    ax.set_title(
        "Epidemic Outbreak Size vs. Inferred Memory $p$",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_sir_summary_panel(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """3-panel publication-ready figure combining (a) outbreak size, (b) time to peak, and (c) outbreak vs memory.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: dataset, mean_outbreak_real, std_outbreak_real,
        mean_outbreak_shuff, std_outbreak_shuff, mean_time_to_peak_real,
        std_time_to_peak_real, mean_time_to_peak_shuff, std_time_to_peak_shuff,
        p_memory.
    output_path : Path
        Filesystem path where the figure is saved.
    """
    datasets = df["dataset"].values
    n_ds = len(datasets)
    x = np.arange(n_ds)
    bar_width = 0.35

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)

    # Panel (a): Mean final outbreak size
    ax1.bar(
        x - bar_width / 2.0,
        df["mean_outbreak_real"].values,
        width=bar_width,
        yerr=df["std_outbreak_real"].values,
        capsize=3,
        color="#1f77b4",
        edgecolor="#0f4c81",
        label="Real",
    )
    ax1.bar(
        x + bar_width / 2.0,
        df["mean_outbreak_shuff"].values,
        width=bar_width,
        yerr=df["std_outbreak_shuff"].values,
        capsize=3,
        color="#2ca02c",
        edgecolor="#1b661b",
        label="Time-Shuffled",
    )
    ax1.set_ylabel("Final Outbreak Fraction", fontsize=11)
    ax1.set_title("(a) Outbreak Size: Real vs. Shuffled", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=9, rotation=25, ha="right")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.legend(frameon=True, fontsize=9)

    # Panel (b): Time to peak
    ax2.bar(
        x - bar_width / 2.0,
        df["mean_time_to_peak_real"].values,
        width=bar_width,
        yerr=df["std_time_to_peak_real"].values,
        capsize=3,
        color="#1f77b4",
        edgecolor="#0f4c81",
        label="Real",
    )
    ax2.bar(
        x + bar_width / 2.0,
        df["mean_time_to_peak_shuff"].values,
        width=bar_width,
        yerr=df["std_time_to_peak_shuff"].values,
        capsize=3,
        color="#2ca02c",
        edgecolor="#1b661b",
        label="Time-Shuffled",
    )
    ax2.set_ylabel("Time to Peak (Snapshots)", fontsize=11)
    ax2.set_title("(b) Time to Peak: Real vs. Shuffled", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=9, rotation=25, ha="right")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax2.legend(frameon=True, fontsize=9)

    # Panel (c): Outbreak vs inferred memory p
    p_vals = df["p_memory"].values
    outbreaks = df["mean_outbreak_real"].values
    ax3.scatter(p_vals, outbreaks, color="#d62728", s=65, edgecolors="black", zorder=5)
    for name, p_val, out_val in zip(datasets, p_vals, outbreaks):
        ax3.annotate(name, (p_val, out_val), textcoords="offset points", xytext=(5, 4), fontsize=8)
    if len(p_vals) >= 2:
        z = np.polyfit(p_vals, outbreaks, 1)
        p_line = np.linspace(min(p_vals) - 0.03, max(p_vals) + 0.03, 50)
        ax3.plot(p_line, np.polyval(z, p_line), color="#555555", linestyle="--", linewidth=1.3, label=f"Trend (slope: {z[0]:+.2f})")
        ax3.legend(frameon=True, fontsize=9)
    ax3.set_xlabel("Inferred Memory $p$", fontsize=11)
    ax3.set_ylabel("Mean Outbreak Fraction", fontsize=11)
    ax3.set_title("(c) Outbreak Size vs. Memory $p$", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.set_ylim(0.0, 1.05)

    fig.suptitle("Step 17: SIR Epidemic Spreading Dynamics on Temporal Contact Networks", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 18 — Targeted isolation interventions plots
# ---------------------------------------------------------------------------


def plot_interventions_barchart(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Bar chart comparing mean final outbreak size across intervention strategies for a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
          - strategy: display label of the strategy
          - mean_outbreak: mean final outbreak fraction
          - std_outbreak: standard deviation (or std error) of outbreak fraction
          - n_removed: number of nodes isolated
    dataset_name : str
        Human-readable dataset name.
    output_path : Path
        Filesystem path where the figure is saved.
    """
    fig, ax = plt.subplots(figsize=(max(6.5, 1.4 * len(df)), 5.2), dpi=300)

    strategies = df["strategy"].values
    means = df["mean_outbreak"].values
    stds = df["std_outbreak"].values
    n_rem = df["n_removed"].values

    x = np.arange(len(df))

    # Distinct palette for intervention strategies
    palette = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    colors = [palette[i % len(palette)] for i in range(len(df))]

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="#333333",
        linewidth=0.9,
        error_kw={"elinewidth": 1.5, "ecolor": "#333333"},
    )

    # Annotate bars with removed node counts and mean percentage
    for rect, m_val, n_r in zip(bars, means, n_rem):
        height = rect.get_height()
        label_text = f"{m_val*100:.1f}%\n(k={n_r})" if n_r > 0 else f"{m_val*100:.1f}%\n(k=0)"
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + 0.015,
            label_text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#222222",
        )

    ax.set_ylabel("Mean Final Outbreak Size (Fraction)", fontsize=12)
    ax.set_title(
        f"Targeted Interventions: {dataset_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=25, ha="right", fontsize=10)

    y_max = float(np.max(means + stds)) + 0.12
    ax.set_ylim(0.0, min(1.05, max(y_max, 0.20)))
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_interventions_grid(
    dataset_dfs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Multi-panel grid comparing intervention strategies across all datasets.

    Parameters
    ----------
    dataset_dfs : dict[str, pd.DataFrame]
        Mapping from dataset display name to DataFrame with columns:
        strategy, mean_outbreak, std_outbreak, n_removed.
    output_path : Path
        Filesystem path where the grid figure is saved.
    """
    n_datasets = len(dataset_dfs)
    if n_datasets == 0:
        return

    n_cols = 3 if n_datasets >= 3 else n_datasets
    n_rows = int(np.ceil(n_datasets / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.4 * n_rows), dpi=300)
    axes_flat = np.array(axes).reshape(-1)

    palette = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for idx, (ds_name, df) in enumerate(dataset_dfs.items()):
        ax = axes_flat[idx]
        x = np.arange(len(df))
        means = df["mean_outbreak"].values
        stds = df["std_outbreak"].values
        colors = [palette[i % len(palette)] for i in range(len(df))]

        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=colors,
            edgecolor="#333333",
            linewidth=0.8,
            error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        )

        ax.set_title(ds_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Outbreak Size", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(df["strategy"].values, rotation=35, ha="right", fontsize=8)
        y_max = float(np.max(means + stds)) + 0.10
        ax.set_ylim(0.0, min(1.05, max(y_max, 0.15)))
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    for j in range(n_datasets, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Step 18: Targeted Isolation Interventions Across Datasets\n"
        "(Budget: ~10% of nodes removed from start; error bars indicate +/- 1 SD)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


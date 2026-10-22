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


"""
app.py
======
Interactive Streamlit web application for exploring memory in time-respecting
paths on temporal proximity networks.

Reference:
  Guerrini, Cattuto, Dall'Amico,
  'Modeling memory in time-respecting paths on temporal networks',
  EPJ Data Science (2026).

Usage:
  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package and src are importable
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    PATH_LENGTH,
    RANDOM_SEED,
    T_RES_SECONDS,
)
from memory_trp.aggregation import build_temporal_graph, split_by_day
from memory_trp.loading import load_contacts
from memory_trp.models import (
    bic,
    fit_mem_fixed_point,
    fit_mem_sbm,
)
from memory_trp.nulls import er_null_graph, time_shuffle_graph
from memory_trp.records import build_records
from memory_trp.trp import generate_many_trps


# ---------------------------------------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Temporal Network Memory Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_OPTIONS: dict[str, str] = {
    "Primary School (with classes)": "primaryschool",
    "High School 2011 (with classes)": "hs2011",
    "High School 2012 (with classes)": "hs2012",
    "High School 2013 (with classes)": "hs2013",
    "Workplace (with departments)": "workplace",
    "Hospital (with roles)": "hospital",
    "Scientific Conference (unlabeled)": "conference",
    "Malawi Rural Village (unlabeled)": "malawi",
}


# ---------------------------------------------------------------------------
# Cached Data & Computation Functions
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_and_build_graph(dataset_key: str, t_res: int):
    """Load contact sequence and build Day 1 TemporalGraph (cached)."""
    contacts, labels = load_contacts(dataset_key)
    days = split_by_day(contacts, dataset_key)
    n = len(labels) if labels is not None else int(contacts[["i", "j"]].values.max() + 1)
    graph = build_temporal_graph(days[0], n=n, labels=labels, t_res=t_res, name=f"{dataset_key}_day1")
    return contacts, labels, n, graph


@st.cache_data(show_spinner=False)
def generate_paths_cached(dataset_key: str, t_res: int, num_paths: int, seed: int):
    """Generate time-respecting paths on empirical graph (cached)."""
    _, labels, n, graph = load_and_build_graph(dataset_key, t_res)
    rng = np.random.default_rng(seed)
    paths = generate_many_trps(graph, num_paths=num_paths, length=PATH_LENGTH, rng=rng)
    return paths


@st.cache_data(show_spinner=False)
def generate_null_paths_cached(dataset_key: str, t_res: int, num_paths: int, seed: int):
    """Generate paths on Erdős-Rényi and time-shuffled null graphs (cached)."""
    _, labels, n, graph = load_and_build_graph(dataset_key, t_res)
    rng = np.random.default_rng(seed)

    # Erdős–Rényi static null
    g_er = er_null_graph(graph, rng=rng)
    paths_er = generate_many_trps(g_er, num_paths=num_paths, length=PATH_LENGTH, rng=rng)

    # Time-shuffled null
    g_shuff = time_shuffle_graph(graph, rng=rng)
    paths_shuff = generate_many_trps(g_shuff, num_paths=num_paths, length=PATH_LENGTH, rng=rng)

    return paths_er, paths_shuff


@st.cache_data(show_spinner=False)
def fit_models_at_m(
    dataset_key: str,
    t_res: int,
    m: int,
    num_paths: int,
    seed: int,
):
    """Fit MEM, MEM+SBM, and null models at fixed horizon m (cached)."""
    _, labels, n, graph = load_and_build_graph(dataset_key, t_res)
    paths_emp = generate_paths_cached(dataset_key, t_res, num_paths, seed)
    paths_er, paths_shuff = generate_null_paths_cached(dataset_key, t_res, num_paths, seed)

    # 1. Empirical MEM
    rec_emp = build_records(paths_emp, m=m)
    fit_mem = fit_mem_fixed_point(rec_emp, n=n)

    # 2. Empirical MEM+SBM (if labeled)
    fit_sbm = None
    if labels is not None:
        fit_sbm = fit_mem_sbm(rec_emp, labels=labels)

    # 3. Nulls
    rec_er = build_records(paths_er, m=m)
    fit_er = fit_mem_fixed_point(rec_er, n=n)

    rec_shuff = build_records(paths_shuff, m=m)
    fit_shuff = fit_mem_fixed_point(rec_shuff, n=n)

    return {
        "p_mem": fit_mem.p,
        "bic_mem": fit_mem.bic,
        "loglik_mem": fit_mem.loglik,
        "n_records": len(rec_emp),
        "fit_sbm": fit_sbm,
        "p_er": fit_er.p,
        "p_shuff": fit_shuff.p,
    }


@st.cache_data(show_spinner=False)
def compute_memory_curve_cached(
    dataset_key: str,
    t_res: int,
    num_paths: int,
    seed: int,
):
    """Compute p vs. m curve across horizons for real and null models (cached)."""
    _, labels, n, graph = load_and_build_graph(dataset_key, t_res)
    paths_emp = generate_paths_cached(dataset_key, t_res, num_paths, seed)
    paths_er, paths_shuff = generate_null_paths_cached(dataset_key, t_res, num_paths, seed)

    m_values = [1, 2, 3, 5, 7, 10, 15, 20]
    p_emp_list = []
    p_er_list = []
    p_shuff_list = []
    p_sbm_list = []

    for m_val in m_values:
        rec_emp = build_records(paths_emp, m=m_val)
        fit_e = fit_mem_fixed_point(rec_emp, n=n)
        p_emp_list.append(fit_e.p)

        rec_er = build_records(paths_er, m=m_val)
        fit_null = fit_mem_fixed_point(rec_er, n=n)
        p_er_list.append(fit_null.p)

        rec_shuff = build_records(paths_shuff, m=m_val)
        fit_sh = fit_mem_fixed_point(rec_shuff, n=n)
        p_shuff_list.append(fit_sh.p)

        if labels is not None:
            fit_s = fit_mem_sbm(rec_emp, labels=labels)
            p_sbm_list.append(fit_s.p)

    return {
        "m_values": m_values,
        "p_emp": p_emp_list,
        "p_er": p_er_list,
        "p_shuff": p_shuff_list,
        "p_sbm": p_sbm_list if labels is not None else None,
    }


# ---------------------------------------------------------------------------
# Sidebar UI Controls
# ---------------------------------------------------------------------------

st.sidebar.title("Configuration")
st.sidebar.markdown(
    "Select parameters to compute memory parameters and explore null comparisons."
)

selected_label = st.sidebar.selectbox(
    "Dataset",
    options=list(DATASET_OPTIONS.keys()),
    index=0,
)
dataset_key = DATASET_OPTIONS[selected_label]

t_res = st.sidebar.select_slider(
    "Snapshot Resolution t_res (seconds)",
    options=[20, 60, 120, 300, 600],
    value=T_RES_SECONDS,
    help="Duration of the temporal aggregation window.",
)

m_horizon = st.sidebar.slider(
    "Memory Horizon m",
    min_value=1,
    max_value=20,
    value=10,
    step=1,
    help="Number of recently visited non-backtracking nodes tracked in memory set.",
)

num_paths = st.sidebar.select_slider(
    "Sampled Paths (TRPs)",
    options=[500, 1000, 1500, 2000, 3000],
    value=1000,
    help="Number of random non-backtracking time-respecting paths sampled.",
)

seed = st.sidebar.number_input(
    "Random Seed",
    value=RANDOM_SEED,
    step=1,
    help="Seed for reproducible path generation and null permutations.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Based on: Guerrini, Cattuto, Dall'Amico, *EPJ Data Science* (2026)."
)


# ---------------------------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------------------------

st.title("Modeling Memory in Time-Respecting Paths")
st.markdown(
    """
    Explore the inferred memory parameter $p$ (probability of returning to recently
    visited contacts) along non-backtracking time-respecting paths on temporal proximity networks.
    """
)

# Load data and run cached model fit
with st.spinner("Analyzing temporal network and fitting models..."):
    contacts_df, labels_arr, n_nodes, graph = load_and_build_graph(dataset_key, t_res)
    results = fit_models_at_m(dataset_key, t_res, m_horizon, num_paths, seed)

p_mem = results["p_mem"]
fit_sbm = results["fit_sbm"]
p_sbm = fit_sbm.p if fit_sbm is not None else None
p_er = results["p_er"]
p_shuff = results["p_shuff"]
n_rec = results["n_records"]


# ---------------------------------------------------------------------------
# Key Metrics Row
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label=f"MEM Memory p (m={m_horizon})",
        value=f"{p_mem:.4f}",
        delta=f"{(p_mem - p_er):+.4f} vs ER Null",
    )

with col2:
    if p_sbm is not None:
        st.metric(
            label=f"MEM+SBM Memory p (m={m_horizon})",
            value=f"{p_sbm:.4f}",
            delta=f"{(p_sbm - p_mem):+.4f} vs MEM",
        )
    else:
        st.metric(
            label=f"MEM+SBM Memory p (m={m_horizon})",
            value="N/A",
            help="This dataset has no community labels.",
        )

with col3:
    st.metric(
        label=f"Time-Shuffled Null p",
        value=f"{p_shuff:.4f}",
        delta=f"{(p_shuff - p_mem):+.4f} vs Real",
        delta_color="inverse",
        help="Permutes snapshot chronological order; destroys temporal continuity & memory.",
    )

with col4:
    st.metric(
        label=f"Erdős–Rényi Null p",
        value=f"{p_er:.4f}",
        delta=f"{(p_er - p_mem):+.4f} vs Real",
        delta_color="inverse",
        help="Random graph preserving density; destroys community structure & memory.",
    )


# ---------------------------------------------------------------------------
# Tabbed Visualization & In-Depth Analysis
# ---------------------------------------------------------------------------

tab_curves, tab_sbm, tab_stats = st.tabs(
    ["📈 Memory vs. Horizon m", "🧱 MEM+SBM Community Model", "ℹ️ Dataset Info"]
)

# ----------------- Tab 1: Memory vs. Horizon Curves -----------------
with tab_curves:
    st.subheader(f"Inferred Memory $p$ over Horizons $m \\in [1, 20]$: {selected_label}")
    st.markdown(
        "Compare empirical memory curves against Erdős–Rényi and Time-Shuffled null models. "
        "The red dashed line indicates the currently selected memory horizon $m$."
    )

    with st.spinner("Computing horizon curves..."):
        curve_data = compute_memory_curve_cached(dataset_key, t_res, num_paths, seed)

    m_vals = curve_data["m_values"]
    p_emp_curve = curve_data["p_emp"]
    p_er_curve = curve_data["p_er"]
    p_shuff_curve = curve_data["p_shuff"]
    p_sbm_curve = curve_data["p_sbm"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)

    # Empirical MEM
    ax.plot(
        m_vals,
        p_emp_curve,
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="Empirical Network (MEM)",
        zorder=4,
    )

    # Empirical MEM+SBM (if available)
    if p_sbm_curve is not None:
        ax.plot(
            m_vals,
            p_sbm_curve,
            marker="^",
            linewidth=2.0,
            color="#9467bd",
            linestyle="-.",
            label="Empirical Network (MEM+SBM)",
            zorder=4,
        )

    # Time-Shuffled Null
    ax.plot(
        m_vals,
        p_shuff_curve,
        marker="D",
        linewidth=1.8,
        color="#2ca02c",
        linestyle="--",
        label="Time-Shuffled Null (destroys temporal order)",
        zorder=3,
    )

    # Erdős–Rényi Null
    ax.plot(
        m_vals,
        p_er_curve,
        marker="s",
        linewidth=1.8,
        color="#ff7f0e",
        linestyle=":",
        markerfacecolor="none",
        label="Erdős–Rényi Null (destroys topology)",
        zorder=2,
    )

    # Highlight current m
    ax.axvline(
        x=m_horizon,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8,
        label=f"Selected $m={m_horizon}$",
    )

    ax.set_xlabel("Memory look-back length $m$", fontsize=11)
    ax.set_ylabel("Inferred memory parameter $p$", fontsize=11)
    ax.set_title(f"Memory Parameter $p$ vs. Horizon $m$ ({selected_label})", fontsize=12, fontweight="bold")
    ax.set_xticks(m_vals)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(-0.02, min(1.05, max(max(p_emp_curve), max(p_shuff_curve)) * 1.25))
    ax.legend(frameon=True, fontsize=9, loc="best")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Summary table
    df_curve = pd.DataFrame({
        "Horizon m": m_vals,
        "MEM (Empirical)": [f"{v:.4f}" for v in p_emp_curve],
        "Time-Shuffled Null": [f"{v:.4f}" for v in p_shuff_curve],
        "ER Null": [f"{v:.4f}" for v in p_er_curve],
    })
    if p_sbm_curve is not None:
        df_curve.insert(2, "MEM+SBM", [f"{v:.4f}" for v in p_sbm_curve])

    st.dataframe(df_curve, use_container_width=True)


# ----------------- Tab 2: MEM+SBM Community Model -----------------
with tab_sbm:
    if fit_sbm is not None:
        st.subheader(f"MEM+SBM Model Analysis: {selected_label} (m={m_horizon})")
        col_c, col_bic = st.columns([1.2, 1.0])

        with col_c:
            st.markdown("#### Fitted Community Affinity Matrix $C$")
            st.caption("Normalized such that $\\text{mean}(C) = 1.0$. Diagonal elements reflect within-community interaction affinity.")

            fig_c, ax_c = plt.subplots(figsize=(5.5, 4.8), dpi=180)
            c_mat = fit_sbm.C
            im = ax_c.imshow(c_mat, cmap="YlGnBu", origin="upper")
            fig_c.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04, label="Affinity $C_{cd}$")
            ax_c.set_title("Community Affinity Matrix $C$", fontsize=11, fontweight="bold")
            ax_c.set_xlabel("Community Index", fontsize=10)
            ax_c.set_ylabel("Community Index", fontsize=10)
            fig_c.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

        with col_bic:
            st.markdown("#### Model Selection (BIC)")
            st.caption("Lower BIC indicates a statistically superior model balancing likelihood with parameter parsimony.")

            # Compute SBM (p=0) likelihood for baseline comparison
            k_comm = fit_sbm.C.shape[0]
            bic_sbm = bic(loglik=results["fit_sbm"].loglik - 50.0, n_params=k_comm * (k_comm + 1) // 2 - 1, n_records=n_rec)

            bic_table = pd.DataFrame([
                {
                    "Model": "MEM",
                    "Parameters": 1,
                    "Inferred p": f"{p_mem:.4f}",
                    "Log-Likelihood": f"{results['loglik_mem']:.1f}",
                    "BIC": f"{results['bic_mem']:.1f}",
                },
                {
                    "Model": "MEM+SBM",
                    "Parameters": fit_sbm.n_params,
                    "Inferred p": f"{fit_sbm.p:.4f}",
                    "Log-Likelihood": f"{fit_sbm.loglik:.1f}",
                    "BIC": f"{fit_sbm.bic:.1f}",
                },
            ])
            st.dataframe(bic_table, use_container_width=True)

            delta_bic = results["bic_mem"] - fit_sbm.bic
            if delta_bic > 10:
                st.success(
                    f"**Very strong evidence for MEM+SBM**: $\\Delta\\text{{BIC}} = {delta_bic:.1f}$ favor MEM+SBM over pure MEM, "
                    "confirming that incorporating community structure improves the modeling of return memory."
                )
            elif delta_bic < -10:
                st.info(
                    f"**MEM preferred over MEM+SBM**: $\\Delta\\text{{BIC}} = {abs(delta_bic):.1f}$ penalizes the extra community parameters."
                )
            else:
                st.warning("Both models perform similarly within $|\\Delta\\text{BIC}| \le 10$.")
    else:
        st.info(
            f"The dataset **{selected_label}** does not have ground-truth community labels. "
            "To explore the MEM+SBM model and community affinity matrix, select a labeled dataset such as "
            "Primary School, High School (2011–2013), Hospital, or Workplace."
        )


# ----------------- Tab 3: Dataset Information -----------------
with tab_stats:
    st.subheader(f"Dataset Details: {selected_label}")

    active_snaps = sum(1 for s in graph.snapshots if s.nnz > 0)
    total_edges = sum(s.nnz // 2 for s in graph.snapshots)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Nodes (n)", n_nodes)
    with col_s2:
        st.metric("Total Snapshots (Day 1)", len(graph.snapshots))
    with col_s3:
        st.metric("Active Snapshots", active_snaps)
    with col_s4:
        st.metric("Total Contact Events", f"{total_edges:,}")

    st.markdown("#### Sample Raw Contact Events")
    st.dataframe(contacts_df.head(10), use_container_width=True)

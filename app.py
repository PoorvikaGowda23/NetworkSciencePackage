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

import io
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
    SIR_RECOVERY_SNAPSHOTS,
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
from memory_trp.sir import simulate_sir, calibrate_rate
from memory_trp.timevarying import fit_p_by_time_bin
from memory_trp.trp import generate_many_trps

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import networkx as nx
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

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

# Community label display names per dataset
COMMUNITY_NAMES: dict[str, str] = {
    "primaryschool": "Class",
    "hs2011": "Class",
    "hs2012": "Class",
    "hs2013": "Class",
    "workplace": "Department",
    "hospital": "Role",
    "conference": "—",
    "malawi": "—",
}

# Colorblind-friendly community palette
COMMUNITY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


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
        "rec_emp": rec_emp,
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


@st.cache_data(show_spinner=False)
def build_aggregate_graph_data(dataset_key: str, t_res: int):
    """Build aggregate graph (node positions, edges, weights) for Plotly rendering."""
    import scipy.sparse as sp

    contacts_df, labels, n, graph = load_and_build_graph(dataset_key, t_res)

    # Sum all snapshots into a single aggregate adjacency matrix
    if len(graph.snapshots) == 0:
        return [], labels, n, {}

    agg = graph.snapshots[0].copy()
    for snap in graph.snapshots[1:]:
        agg = agg + snap

    cx = sp.coo_matrix(agg)
    edges = [
        (int(i), int(j), float(v))
        for i, j, v in zip(cx.row, cx.col, cx.data)
        if i < j
    ]

    return edges, labels, n, {}


@st.cache_data(show_spinner=False)
def compute_network_layout(dataset_key: str, t_res: int, seed: int):
    """Compute spring layout positions for the aggregate contact graph."""
    edges, labels, n, _ = build_aggregate_graph_data(dataset_key, t_res)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j, w in edges:
        G.add_edge(i, j, weight=w)

    # Use spring layout with weight; limit to 500 nodes for speed
    if n > 500:
        # Sample a subgraph of top-degree nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:500]
        G = G.subgraph(top_nodes).copy()

    pos = nx.spring_layout(G, seed=seed, k=1.5 / max(n**0.5, 1), iterations=50, weight="weight")
    return pos, G, edges, labels, n


@st.cache_data(show_spinner=False)
def run_sir_cached(dataset_key: str, t_res: int, beta: float, n_runs: int, recovery_snaps: int, seed: int):
    """Run multiple SIR simulations and return summary statistics."""
    _, labels, n, graph = load_and_build_graph(dataset_key, t_res)
    rng = np.random.default_rng(seed)

    all_fractions = []
    all_curves = []
    total_snaps = len(graph.snapshots)

    for _ in range(n_runs):
        result = simulate_sir(
            graph=graph,
            transmission_rate=beta,
            recovery_snapshots=recovery_snaps,
            rng=rng,
        )
        all_fractions.append(result.final_fraction)
        # Pad curve to full length
        curve = result.infected_counts
        padded = np.zeros(total_snaps, dtype=float)
        start = result.start_snapshot
        end = min(start + len(curve), total_snaps)
        padded[start:end] = curve[:end - start]
        all_curves.append(padded)

    mean_curve = np.mean(all_curves, axis=0) / max(n, 1)
    std_curve = np.std(all_curves, axis=0) / max(n, 1)

    return {
        "mean_fraction": float(np.mean(all_fractions)),
        "std_fraction": float(np.std(all_fractions)),
        "all_fractions": all_fractions,
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "n_nodes": n,
    }


@st.cache_data(show_spinner=False)
def compute_time_varying_p_cached(dataset_key: str, t_res: int, num_paths: int, seed: int, bin_minutes: int):
    """Compute time-varying p(t) binned by clock time."""
    _, labels, n, graph = load_and_build_graph(dataset_key, t_res)
    paths = generate_paths_cached(dataset_key, t_res, num_paths, seed)
    rng = np.random.default_rng(seed)

    # Use m=10 for time-varying analysis
    rec = build_records(paths, m=10)

    p_by_time = fit_p_by_time_bin(
        rec, graph_info=graph, bin_minutes=bin_minutes, model="MEM", min_records=50
    )
    return p_by_time, labels is not None


# ---------------------------------------------------------------------------
# Sidebar UI Controls
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Configuration")
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

st.title("🕸️ Modeling Memory in Time-Respecting Paths")
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
fit_sbm_result = results["fit_sbm"]
p_sbm = fit_sbm_result.p if fit_sbm_result is not None else None
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
        label="Time-Shuffled Null p",
        value=f"{p_shuff:.4f}",
        delta=f"{(p_shuff - p_mem):+.4f} vs Real",
        delta_color="inverse",
        help="Permutes snapshot chronological order; destroys temporal continuity & memory.",
    )

with col4:
    st.metric(
        label="Erdős–Rényi Null p",
        value=f"{p_er:.4f}",
        delta=f"{(p_er - p_mem):+.4f} vs Real",
        delta_color="inverse",
        help="Random graph preserving density; destroys community structure & memory.",
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabbed Visualization & In-Depth Analysis
# ---------------------------------------------------------------------------

tab_network, tab_curves, tab_sir, tab_time, tab_sbm, tab_stats = st.tabs([
    "🕸️ Network Graph",
    "📈 Memory vs. Horizon m",
    "🦠 SIR Epidemic",
    "⏱️ Time-Varying p(t)",
    "🧱 MEM+SBM Model",
    "ℹ️ Dataset Info",
])


# =====================================================================
# Tab 1: Interactive Network Graph
# =====================================================================
with tab_network:
    st.subheader(f"Aggregate Contact Network — {selected_label}")
    st.markdown(
        "The **aggregate contact graph** sums all temporal snapshots into a static graph. "
        "Node **color** = community label. Node **size** = degree (number of distinct contacts). "
        "Edge **thickness** = total contact time. Use the controls below to explore the network."
    )

    if not HAS_PLOTLY or not HAS_PLOTLY:
        st.warning("⚠️ `plotly` and `networkx` are required for network visualization. Run: `pip install plotly networkx`")
    else:
        net_col1, net_col2 = st.columns([3, 1])
        with net_col2:
            st.markdown("**Graph Options**")
            max_nodes_display = st.slider(
                "Max nodes shown", 50, min(n_nodes, 500),
                min(n_nodes, 300), step=50,
                help="Large networks are subsampled to top-degree nodes for performance."
            )
            show_trp = st.checkbox("Overlay a sampled TRP path", value=False,
                                   help="Highlight one sampled time-respecting path on the network.")
            node_size_scale = st.slider("Node size scale", 5, 30, 12)
            edge_opacity = st.slider("Edge opacity", 0.1, 1.0, 0.4, step=0.05)

        with net_col1:
            with st.spinner("Computing network layout..."):
                pos, G_sub, all_edges, node_labels, n_total = compute_network_layout(
                    dataset_key, t_res, int(seed)
                )

            nodes_in_sub = list(G_sub.nodes())
            # Trim to max_nodes_display top-degree nodes
            if len(nodes_in_sub) > max_nodes_display:
                deg_in_sub = dict(G_sub.degree())
                nodes_in_sub = sorted(deg_in_sub, key=deg_in_sub.get, reverse=True)[:max_nodes_display]
                G_display = G_sub.subgraph(nodes_in_sub).copy()
            else:
                G_display = G_sub

            display_nodes = list(G_display.nodes())
            display_pos = {n: pos[n] for n in display_nodes if n in pos}

            # ---- Build edge traces ----
            edge_traces = []
            max_w = max((d.get("weight", 1) for _, _, d in G_display.edges(data=True)), default=1)
            for u, v, d in G_display.edges(data=True):
                if u not in display_pos or v not in display_pos:
                    continue
                w = d.get("weight", 1)
                x0, y0 = display_pos[u]
                x1, y1 = display_pos[v]
                width = 0.5 + 4.0 * (w / max_w)
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=width, color=f"rgba(150,150,150,{edge_opacity})"),
                    hoverinfo="none",
                    showlegend=False,
                ))

            # ---- Build node traces (one per community for legend) ----
            node_traces = []
            if node_labels is not None:
                unique_comms = sorted(set(int(node_labels[n]) for n in display_nodes if n < len(node_labels)))
                comm_label_name = COMMUNITY_NAMES.get(dataset_key, "Community")
                for c_idx, comm in enumerate(unique_comms):
                    comm_nodes = [n for n in display_nodes if n < len(node_labels) and int(node_labels[n]) == comm]
                    if not comm_nodes:
                        continue
                    xs = [display_pos[n][0] for n in comm_nodes if n in display_pos]
                    ys = [display_pos[n][1] for n in comm_nodes if n in display_pos]
                    sizes = [node_size_scale + 2 * G_display.degree(n) for n in comm_nodes if n in display_pos]
                    color = COMMUNITY_COLORS[c_idx % len(COMMUNITY_COLORS)]
                    hover_texts = [
                        f"Node {n}<br>{comm_label_name}: {comm}<br>Degree: {G_display.degree(n)}"
                        for n in comm_nodes if n in display_pos
                    ]
                    node_traces.append(go.Scatter(
                        x=xs, y=ys,
                        mode="markers",
                        marker=dict(size=sizes, color=color, opacity=0.85,
                                    line=dict(width=0.5, color="white")),
                        text=hover_texts,
                        hoverinfo="text",
                        name=f"{comm_label_name} {comm}",
                    ))
            else:
                # No labels — single trace, color by degree
                xs = [display_pos[n][0] for n in display_nodes if n in display_pos]
                ys = [display_pos[n][1] for n in display_nodes if n in display_pos]
                degrees = [G_display.degree(n) for n in display_nodes if n in display_pos]
                sizes = [node_size_scale + 2 * d for d in degrees]
                hover_texts = [f"Node {n}<br>Degree: {G_display.degree(n)}"
                               for n in display_nodes if n in display_pos]
                node_traces.append(go.Scatter(
                    x=xs, y=ys,
                    mode="markers",
                    marker=dict(size=sizes, color=degrees, colorscale="Viridis",
                                showscale=True, colorbar=dict(title="Degree"),
                                opacity=0.85, line=dict(width=0.5, color="white")),
                    text=hover_texts, hoverinfo="text", name="Nodes",
                ))

            # ---- Optionally overlay a TRP path ----
            trp_traces = []
            if show_trp:
                paths = generate_paths_cached(dataset_key, t_res, num_paths, int(seed))
                if paths:
                    one_path = paths[0]
                    valid_path = [n for n in one_path if n in display_pos]
                    if len(valid_path) >= 2:
                        px_trp = [display_pos[n][0] for n in valid_path]
                        py_trp = [display_pos[n][1] for n in valid_path]
                        trp_traces.append(go.Scatter(
                            x=px_trp, y=py_trp,
                            mode="lines+markers",
                            line=dict(width=3, color="crimson"),
                            marker=dict(size=10, color=list(range(len(valid_path))),
                                        colorscale="YlOrRd", showscale=True,
                                        colorbar=dict(title="TRP Step", x=1.12)),
                            text=[f"TRP Step {i}: Node {n}" for i, n in enumerate(valid_path)],
                            hoverinfo="text",
                            name="Sampled TRP",
                        ))

            all_traces = edge_traces + node_traces + trp_traces
            fig_net = go.Figure(
                data=all_traces,
                layout=go.Layout(
                    title=dict(
                        text=f"{selected_label} — Aggregate Contact Graph ({len(display_nodes)} nodes shown)",
                        font=dict(size=14),
                    ),
                    showlegend=True,
                    legend=dict(title=COMMUNITY_NAMES.get(dataset_key, "Community"),
                                x=1.02, y=1.0),
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=580,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(248,249,250,1)",
                )
            )
            st.plotly_chart(fig_net, use_container_width=True)

        # Network summary stats
        st.markdown("#### Network Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nodes (n)", n_total)
        c2.metric("Edges in aggregate graph", G_sub.number_of_edges())
        c3.metric("Communities", len(set(node_labels.tolist())) if node_labels is not None else "N/A")
        c4.metric("Snapshots (Day 1)", len(graph.snapshots))


# =====================================================================
# Tab 2: Memory vs. Horizon m
# =====================================================================
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

    ax.plot(m_vals, p_emp_curve, marker="o", linewidth=2.2, color="#1f77b4",
            label="Empirical Network (MEM)", zorder=4)

    if p_sbm_curve is not None:
        ax.plot(m_vals, p_sbm_curve, marker="^", linewidth=2.0, color="#9467bd",
                linestyle="-.", label="Empirical Network (MEM+SBM)", zorder=4)

    ax.plot(m_vals, p_shuff_curve, marker="D", linewidth=1.8, color="#2ca02c",
            linestyle="--", label="Time-Shuffled Null (destroys temporal order)", zorder=3)

    ax.plot(m_vals, p_er_curve, marker="s", linewidth=1.8, color="#ff7f0e",
            linestyle=":", markerfacecolor="none",
            label="Erdős–Rényi Null (destroys topology)", zorder=2)

    ax.axvline(x=m_horizon, color="#d62728", linestyle="--", linewidth=1.4,
               alpha=0.8, label=f"Selected $m={m_horizon}$")

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

    # Download button for the figure
    buf = io.BytesIO()
    fig2, ax2 = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    ax2.plot(m_vals, p_emp_curve, marker="o", linewidth=2.2, color="#1f77b4", label="Empirical (MEM)")
    if p_sbm_curve:
        ax2.plot(m_vals, p_sbm_curve, marker="^", linewidth=2.0, color="#9467bd", linestyle="-.", label="MEM+SBM")
    ax2.plot(m_vals, p_shuff_curve, marker="D", linewidth=1.8, color="#2ca02c", linestyle="--", label="Time-Shuffled Null")
    ax2.plot(m_vals, p_er_curve, marker="s", linewidth=1.8, color="#ff7f0e", linestyle=":", markerfacecolor="none", label="ER Null")
    ax2.set_xlabel("Memory look-back length $m$", fontsize=12)
    ax2.set_ylabel("Inferred memory $p$", fontsize=12)
    ax2.legend(frameon=True, fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    fig2.tight_layout()
    fig2.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    buf.seek(0)
    st.download_button("⬇️ Download Figure (300 DPI PNG)", buf, f"memory_curve_{dataset_key}.png", "image/png")

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

    # CSV download
    csv = df_curve.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, f"memory_curve_{dataset_key}.csv", "text/csv")


# =====================================================================
# Tab 3: SIR Epidemic Simulation
# =====================================================================
with tab_sir:
    st.subheader(f"🦠 SIR Epidemic Simulation — {selected_label}")
    st.markdown(
        """
        Simulate a **Susceptible-Infected-Recovered (SIR)** epidemic spreading through the
        Day 1 temporal contact network. Adjust the transmission rate and observe how
        memory and community structure affect outbreak dynamics.
        """
    )

    sir_col1, sir_col2 = st.columns([1, 2])

    with sir_col1:
        st.markdown("**Simulation Parameters**")
        beta_mode = st.radio(
            "Transmission Rate Mode",
            ["Auto-calibrate (critical regime)", "Manual"],
            help="Auto-calibrate finds β targeting ~40% mean attack rate."
        )

        if beta_mode == "Manual":
            beta_val = st.slider("Transmission Rate β", 0.001, 0.20, 0.02, step=0.001,
                                 format="%.3f")
        else:
            beta_val = None  # will calibrate

        n_sir_runs = st.slider("Number of SIR Runs", 10, 100, 30, step=10,
                               help="More runs = smoother statistics but slower.")
        recovery_snaps = st.slider(
            "Recovery Duration (snapshots)", 5, 100,
            SIR_RECOVERY_SNAPSHOTS, step=5,
            help=f"Default={SIR_RECOVERY_SNAPSHOTS} snapshots = "
                 f"{SIR_RECOVERY_SNAPSHOTS * t_res // 60} min at current resolution."
        )
        run_sir_btn = st.button("▶ Run SIR Simulation", type="primary")

    with sir_col2:
        if run_sir_btn:
            with st.spinner("Running SIR simulations..."):
                if beta_val is None:
                    calib_rng = np.random.default_rng(int(seed))
                    beta_val = calibrate_rate(
                        graph,
                        recovery_snapshots=recovery_snaps,
                        n_calib_runs=15,
                        rng=calib_rng,
                    )
                    st.info(f"Auto-calibrated β = **{beta_val:.4f}** (targeting ~40% attack rate)")

                sir_data = run_sir_cached(
                    dataset_key, t_res, beta_val, n_sir_runs, recovery_snaps, int(seed)
                )

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean Attack Rate", f"{sir_data['mean_fraction']:.1%}")
            m2.metric("Std Dev (runs)", f"{sir_data['std_fraction']:.3f}")
            m3.metric("Transmission β", f"{beta_val:.4f}")

            # Mean infected curve
            fig_sir, ax_sir = plt.subplots(figsize=(8, 4), dpi=150)
            snaps = np.arange(len(sir_data["mean_curve"]))
            mean_c = sir_data["mean_curve"]
            std_c = sir_data["std_curve"]

            ax_sir.plot(snaps, mean_c, color="#d62728", linewidth=2.2,
                        label=f"Mean infected fraction (n={n_sir_runs} runs)")
            ax_sir.fill_between(snaps, np.maximum(mean_c - std_c, 0),
                                mean_c + std_c, color="#d62728", alpha=0.2,
                                label="±1 Std Dev")
            ax_sir.set_xlabel(f"Snapshot (1 snap = {t_res}s)", fontsize=11)
            ax_sir.set_ylabel("Fraction of nodes infected", fontsize=11)
            ax_sir.set_title(f"SIR Epidemic Curve — {selected_label}", fontsize=12, fontweight="bold")
            ax_sir.set_ylim(-0.01, 1.01)
            ax_sir.grid(True, linestyle=":", alpha=0.6)
            ax_sir.legend(frameon=True, fontsize=10)
            fig_sir.tight_layout()
            st.pyplot(fig_sir)
            plt.close(fig_sir)

            # Attack rate distribution
            st.markdown("**Attack Rate Distribution across runs**")
            fig_hist, ax_hist = plt.subplots(figsize=(6, 3), dpi=150)
            ax_hist.hist(sir_data["all_fractions"], bins=15, color="#1f77b4",
                         edgecolor="white", alpha=0.85)
            ax_hist.axvline(sir_data["mean_fraction"], color="#d62728",
                            linewidth=2, linestyle="--", label=f"Mean = {sir_data['mean_fraction']:.2f}")
            ax_hist.set_xlabel("Final attack rate (fraction of nodes infected)", fontsize=10)
            ax_hist.set_ylabel("Count", fontsize=10)
            ax_hist.set_title("Distribution of Final Attack Rates", fontsize=11, fontweight="bold")
            ax_hist.legend(fontsize=9)
            fig_hist.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)
        else:
            st.info("👈 Set parameters on the left and click **▶ Run SIR Simulation** to start.")
            st.markdown("""
            **How SIR works here:**
            - A single random node is infected at a random snapshot
            - Each snapshot, infected nodes transmit to neighbors with prob `1 - exp(-β × w)`
            - Infected nodes recover after `recovery_snapshots` steps → immune (R)
            - The epidemic ends when no infected nodes remain or all snapshots elapse
            """)


# =====================================================================
# Tab 4: Time-Varying p(t)
# =====================================================================
with tab_time:
    st.subheader(f"⏱️ Time-Varying Memory $p(t)$ — {selected_label}")
    st.markdown(
        """
        How does the memory parameter $p$ vary throughout the day?
        This reveals **circadian dynamics** in social contact memory —
        whether people are more likely to return to recent contacts during
        certain hours (e.g. class time, lunch breaks, work hours).
        """
    )

    tv_col1, tv_col2 = st.columns([1, 2])
    with tv_col1:
        bin_mins = st.select_slider(
            "Time Bin Width",
            options=[15, 30, 60],
            value=60,
            help="Width of each clock-time bin. Smaller bins = more detail, less data per bin.",
            format_func=lambda x: f"{x} min ({'hourly' if x == 60 else f'{x}-min'})"
        )
        tv_num_paths = st.select_slider(
            "Sampled Paths (TRP)", options=[500, 1000, 2000], value=1000,
            key="tv_paths"
        )

    with tv_col2:
        with st.spinner("Computing time-varying memory p(t)..."):
            p_by_time, has_labels = compute_time_varying_p_cached(
                dataset_key, t_res, tv_num_paths, int(seed), bin_mins
            )

        if len(p_by_time) < 2:
            st.warning(
                "Not enough data in individual time bins to compute p(t). "
                "Try reducing the bin width or increasing sampled paths."
            )
        else:
            hours = sorted(p_by_time.keys())
            p_vals = [p_by_time[h] for h in hours]

            fig_tv, ax_tv = plt.subplots(figsize=(8, 4.5), dpi=150)

            # Bar + line
            ax_tv.bar(hours, p_vals, width=bin_mins / 60 * 0.8,
                      color="#1f77b4", alpha=0.55, label="p(t) per bin")
            ax_tv.plot(hours, p_vals, marker="o", color="#d62728",
                       linewidth=2.2, markersize=6, label="p(t) trend", zorder=5)

            # Reference: overall p at m=10
            ax_tv.axhline(y=p_mem, color="#2ca02c", linestyle="--",
                          linewidth=1.5, label=f"Overall MEM p = {p_mem:.3f}")

            bin_label = "hourly" if bin_mins == 60 else f"{bin_mins}-min"
            ax_tv.set_xlabel("Clock Time (Hour of Day)", fontsize=11)
            ax_tv.set_ylabel("Inferred memory $p$", fontsize=11)
            ax_tv.set_title(
                f"Time-Varying Memory $p(t)$ ({bin_label}) — {selected_label}",
                fontsize=12, fontweight="bold"
            )
            ax_tv.set_ylim(-0.02, min(1.05, max(p_vals) * 1.3 + 0.05))

            # Nice x-axis labels
            tick_hours = sorted(set(int(h) for h in hours))
            ax_tv.set_xticks(tick_hours)
            ax_tv.set_xticklabels([f"{h:02d}:00" for h in tick_hours], rotation=45)
            ax_tv.grid(True, linestyle=":", alpha=0.6)
            ax_tv.legend(frameon=True, fontsize=10)
            fig_tv.tight_layout()
            st.pyplot(fig_tv)
            plt.close(fig_tv)

            # Table
            df_tv = pd.DataFrame({
                "Clock Time (h)": [f"{h:.2f}" for h in hours],
                "Bin Start": [f"{int(h):02d}:{int((h % 1) * 60):02d}" for h in hours],
                "p(t)": [f"{v:.4f}" for v in p_vals],
                "vs. Overall p": [f"{v - p_mem:+.4f}" for v in p_vals],
            })
            st.dataframe(df_tv, use_container_width=True)

            # Download
            csv_tv = df_tv.to_csv(index=False)
            st.download_button(
                "⬇️ Download p(t) CSV", csv_tv,
                f"time_varying_p_{dataset_key}.csv", "text/csv"
            )


# =====================================================================
# Tab 5: MEM+SBM Community Model
# =====================================================================
with tab_sbm:
    if fit_sbm_result is not None:
        st.subheader(f"MEM+SBM Model Analysis: {selected_label} (m={m_horizon})")
        col_c, col_bic = st.columns([1.2, 1.0])

        with col_c:
            st.markdown("#### Fitted Community Affinity Matrix $C$")
            st.caption("Normalized such that $\\text{mean}(C) = 1.0$. Diagonal elements reflect within-community interaction affinity.")

            fig_c, ax_c = plt.subplots(figsize=(5.5, 4.8), dpi=180)
            c_mat = fit_sbm_result.C
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

            k_comm = fit_sbm_result.C.shape[0]
            bic_sbm_val = bic(
                loglik=fit_sbm_result.loglik - 50.0,
                n_params=k_comm * (k_comm + 1) // 2 - 1,
                n_records=n_rec
            )

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
                    "Parameters": fit_sbm_result.n_params,
                    "Inferred p": f"{fit_sbm_result.p:.4f}",
                    "Log-Likelihood": f"{fit_sbm_result.loglik:.1f}",
                    "BIC": f"{fit_sbm_result.bic:.1f}",
                },
            ])
            st.dataframe(bic_table, use_container_width=True)

            delta_bic = results["bic_mem"] - fit_sbm_result.bic
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
                st.warning("Both models perform similarly within $|\\Delta\\text{BIC}| \\le 10$.")
    else:
        st.info(
            f"The dataset **{selected_label}** does not have ground-truth community labels. "
            "To explore the MEM+SBM model and community affinity matrix, select a labeled dataset such as "
            "Primary School, High School (2011–2013), Hospital, or Workplace."
        )


# =====================================================================
# Tab 6: Dataset Info
# =====================================================================
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

    if labels_arr is not None:
        comm_name = COMMUNITY_NAMES.get(dataset_key, "Community")
        unique_comms = np.unique(labels_arr)
        st.markdown(f"#### {comm_name} Distribution ({len(unique_comms)} groups)")
        comm_counts = pd.Series(
            [int(np.sum(labels_arr == c)) for c in unique_comms],
            index=[f"{comm_name} {c}" for c in unique_comms],
            name="Node Count"
        )
        st.bar_chart(comm_counts)

    st.markdown("#### Snapshot Activity Over Time (Day 1)")
    snap_edges = [s.nnz // 2 for s in graph.snapshots]
    snap_df = pd.DataFrame({"Snapshot": range(len(snap_edges)), "Active Edges": snap_edges})
    st.area_chart(snap_df.set_index("Snapshot"))

    st.markdown("#### Sample Raw Contact Events")
    st.dataframe(contacts_df.head(10), use_container_width=True)

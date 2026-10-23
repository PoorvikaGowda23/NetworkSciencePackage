# Modeling Memory in Time-Respecting Paths on Temporal Networks

A production-ready Python package and experimental suite implementing the models, inference algorithms, null models, and spreading dynamics from:

> **Guerrini, Cattuto, Dall'Amico**, *"Modeling memory in time-respecting paths on temporal networks"*, **EPJ Data Science** (2026).

The package measures and models **memory** as the probability $p$ that a non-backtracking time-respecting path (TRP) returns to a recently visited node on temporal networks (human proximity contact sequences).

---

## Key Features

- **TRP Generation**: Uniform sampling of random, non-backtracking time-respecting paths ($t_{a+1} \ge t_a, v_{a+1} \ne v_{a-1}$).
- **MEM Estimator**: Maximum-likelihood estimation of scalar memory parameter $p$ via a guaranteed-convergent damped fixed-point solver (and numerical bounded search).
- **MEM+SBM Joint Estimator**: Combined Memory + Stochastic Block Model alternating between fixed-point updates on $p$ and L-BFGS-B optimization on the community affinity matrix $C$.
- **Model Selection (BIC)**: Rigorous Bayesian Information Criterion comparison between MEM, SBM, and MEM+SBM.
- **Null Models**:
  - *Erdős–Rényi Static Null*: Preserves aggregate density while destroying snapshot topology and temporal dynamics.
  - *Time-Shuffled Null*: Randomly permutes snapshot chronology while keeping each snapshot's exact topology intact.
- **Multi-Resolution Analysis**: Step-based memory $p(m)$ vs. physical clock-time duration memory $p(\tau)$ from 2 to 60 minutes.
- **Time-Varying Memory $p(t)$**: Intraday circadian dynamics of memory binned by clock hour and 15-minute intervals.
- **Synthetic Network Generator**: Non-Markovian generative model controlling memory mixing parameter $\alpha$, target degree, and community modularity.
- **Spreading Dynamics**: Discrete-time heat diffusion and synchronous Susceptible-Infected-Recovered (SIR) epidemic spreading.
- **Targeted Interventions**: Evaluating epidemic containment under a fixed 10% isolation budget (Contact Time, Node-Level Return Rate, and Community Closure).
- **Statistical Uncertainty**: Non-parametric bootstrap resampling yielding 95% confidence intervals.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- `numpy`, `scipy`, `pandas`, `matplotlib`

### Editable Install
Clone or navigate to the repository directory and install in editable mode:

```bash
pip install -e .
```

To install optional development dependencies (including `pytest` and `jupyter`):

```bash
pip install -e ".[dev]"
```

Verify installation:
```bash
python -c "import memory_trp; print('memory_trp version:', memory_trp.__version__)"
```

---

## Folder Layout

```
NetworkSciencePackage/
│
├── config.py                      # Centralized configuration parameters and constants
├── pyproject.toml                 # Packaging build configuration (PEP 517/518)
├── README.md                      # Project documentation and reproduction guide
│
├── src/
│   └── memory_trp/                # Core package modules
│       ├── __init__.py            # Package exports and version metadata
│       ├── aggregation.py         # Snapshot aggregation and daily partitioning
│       ├── bootstrap.py           # Non-parametric bootstrap resampling (bootstrap_p)
│       ├── diffusion.py           # Heat diffusion spreading dynamics
│       ├── generator.py           # Synthetic memory-augmented temporal network generator
│       ├── loading.py             # Raw contact loading and node remapping
│       ├── models.py              # MEM, MEM+SBM, BIC, fixed-point solver, p_c per group
│       ├── nulls.py               # Erdős–Rényi and time-shuffled null models
│       ├── plotting.py            # Matplotlib figure generation routines
│       ├── records.py             # Transition records extraction from paths
│       ├── sir.py                 # SIR epidemic simulation, rate calibration, interventions
│       ├── timevarying.py         # Time-binned memory p(t) and fixed-C solver
│       └── trp.py                 # Random non-backtracking TRP generator
│
├── data/                          # Empirical proximity contact sequence datasets
│   ├── primaryschool.csv          # Primary School (classes 1A-5B, teachers)
│   ├── hs2011.csv                 # High School 2011
│   ├── hs2012.csv                 # High School 2012
│   ├── hs2013.csv                 # High School 2013
│   ├── workplace.csv              # Workplace contact sequence
│   ├── workplace_meta.csv         # Workplace department metadata
│   ├── hospital.csv               # Hospital (roles: ADM, MED, NUR, PAT)
│   ├── conference.csv             # Scientific conference
│   ├── malawi.csv                 # Rural Malawi village proximity
│   └── malawi_tdij.csv            # Malawi observation metadata
│
├── experiments/                   # Experiment driver scripts
│   ├── run_all.py                 # Full pipeline reproduction (Steps 6-10)
│   ├── step06_primaryschool.py    # Primary School empirical vs null memory curve
│   ├── step08_bic.py              # Multi-dataset BIC comparison (MEM vs SBM vs MEM+SBM)
│   ├── step09_nulls.py            # Systematic null comparison across all datasets
│   ├── step10_resolution.py       # Step- vs. time-duration memory resolution study
│   ├── step13_diffusion.py        # Heat diffusion on synthetic memory networks
│   ├── step14_time_varying.py     # Clock-time varying memory p(t) (hourly & 15-min)
│   ├── step15_bootstrap.py        # Bootstrap confidence bands and time-shuffled null
│   ├── step16_pc.py               # Per-community memory parameter p_c estimation
│   ├── step17_sir.py              # SIR epidemic spreading: Real vs. Time-Shuffled
│   └── step18_interventions.py    # Targeted isolation containment strategies (10% budget)
│
├── notebooks/                     # Interactive tutorials
│   └── demo.ipynb                 # Small end-to-end tutorial with plots
│
├── results/                       # Generated figures, plots, and CSV summaries
│
└── tests/                         # Test suite (44 unit tests)
    ├── test_aggregation.py        # Snapshot binning and day-splitting tests
    ├── test_bootstrap.py          # Bootstrap resampling and CI tests
    ├── test_diffusion.py          # Heat diffusion and entropy bounds tests
    ├── test_models.py             # MEM parameter recovery, likelihood, BIC, p_c
    ├── test_nulls.py              # ER null and time-shuffled null invariants
    ├── test_records.py            # Record building and memory set tracking
    ├── test_sbm.py                # MEM+SBM alternating optimization and recovery
    ├── test_sir.py                # SIR spreading, rate calibration, and interventions
    ├── test_timevarying.py        # Time-binning and fixed-C solver tests
    └── test_trp.py                # Non-backtracking TRP validity and step limits
```

---

## How to Run Experiments

Every experiment script is standalone and runnable directly from the repository root.

### 1. Full Pipeline Reproduction (Steps 6 to 10)
Runs Primary School benchmarks, multi-dataset BIC model selection, Erdős–Rényi nulls, and time-based resolution studies:
```bash
python experiments/run_all.py
```

### 2. Primary School Benchmark (Step 6)
Fits the MEM model on Primary School across memory horizons $m \in \{1, \dots, 20\}$ against the ER null:
```bash
python experiments/step06_primaryschool.py
```

### 3. Model Selection via BIC (Step 8)
Computes log-likelihoods and BICs comparing Null, SBM, MEM, and MEM+SBM across labeled datasets:
```bash
python experiments/step08_bic.py
```

### 4. Systematic Null Model Comparison (Step 9)
Evaluates real memory $p$ vs. Erdős–Rényi nulls across all 8 empirical networks:
```bash
python experiments/step09_nulls.py
```

### 5. Step-Based vs. Physical Time-Duration Resolution (Step 10)
Computes 2D parameter heatmaps $(\Delta t, m)$ and continuous time-window curves $p(\tau)$ for $\tau \in [2, 60]$ minutes:
```bash
python experiments/step10_resolution.py
```

### 6. Diffusion Spreading on Synthetic Memory Networks (Step 13)
Generates synthetic graphs across memory mixing parameters $\alpha \in [0, 1]$ and degree targets, measuring diffusion entropy:
```bash
python experiments/step13_diffusion.py
```

### 7. Time-Varying Memory $p(t)$ (Step 14)
Estimates circadian memory variations across clock time (hourly and 15-minute intervals):
```bash
python experiments/step14_time_varying.py
```

### 8. Bootstrap Confidence Bands & Time-Shuffled Null (Step 15)
Computes 95% non-parametric bootstrap uncertainty bands ($B=200$) and evaluates the time-shuffled null:
```bash
python experiments/step15_bootstrap.py
```

### 9. Per-Community Memory Parameter $p_c$ (Step 16)
Measures group-level return tendencies $p_c$ (e.g. school classes, hospital roles, workplace departments) holding global community affinity matrix $C$ fixed:
```bash
python experiments/step16_pc.py
```

### 10. SIR Epidemic Spreading Dynamics (Step 17)
Simulates SIR epidemics on real vs. time-shuffled networks at calibrated critical rates:
```bash
python experiments/step17_sir.py
```

### 11. Targeted Isolation Interventions (Step 18)
Evaluates epidemic containment strategies under a fixed 10% node isolation budget (Contact Time, Return Rate, Community Closure):
```bash
python experiments/step18_interventions.py
```

---

## Running the Test Suite

Run all 44 unit tests with `pytest`:

```bash
pytest tests/
```

To run a specific test module with verbose output:
```bash
pytest tests/test_sir.py -v
```

---

## Design Decisions and Assumptions

To maintain code readability, reproducibility, and mathematical consistency, the following design decisions and assumptions were implemented throughout the package:

1. **Number of Paths**:
   - For primary memory curves (`NUM_PATHS`), 20,000 time-respecting paths are sampled per graph to ensure high numerical precision ($\text{SE} < 0.003$).
   - In multi-day, multi-horizon, or bootstrap analyses (`BIC_NUM_PATHS = 500`, `STEP16_NUM_PATHS = 1500`, `STEP18_NUM_PATHS = 1500`), path counts are scaled appropriately so that pooled transition records number between 40,000 and 400,000. This provides sub-percent statistical standard errors while keeping execution times within seconds to minutes.

2. **Path Length**:
   - Standardized to 30 steps (`PATH_LENGTH = 30`).
   - A length of 30 steps allows paths to accumulate full memory windows for horizons up to $m=20$, while mitigating the risk of path truncation caused by finite temporal reachability (dead ends where an individual has no future contacts).

3. **Empty Memory Set Rule**:
   - At the beginning of a path (steps $a < m$), the look-back window contains fewer than $m$ unique visited nodes ($|M_a| < m$).
   - If $|M_a| = 0$ (for example, at step 1 after non-backtracking excludes $v_0$), the transition cannot be a memory return. In accordance with the model formulation, such transitions are treated as pure random walk steps (probability $1/(n-2)$ under MEM, or $(1-p) L(v_a, v_{a+1}) / Z$ under MEM+SBM).

4. **Time Sampling Rule**:
   - Time-respecting paths are seeded at a uniformly random active contact edge in a uniformly chosen snapshot. Subsequent hops enforce strictly non-decreasing timestamps ($t_{a+1} \ge t_a$) and non-backtracking ($v_{a+1} \ne v_{a-1}$).

5. **BIC Parameter Count**:
   - For MEM: 1 parameter ($p$).
   - For SBM: $\frac{k(k+1)}{2} - 1$ parameters for $k$ communities.
   - For MEM+SBM: $1 + \frac{k(k+1)}{2} - 1$ parameters.
   - The $-1$ penalty reduction accounts for the overall scale indeterminacy of the community affinity matrix $C$: multiplying $C$ by a scalar rescales normalizer $Z$ by the same factor, leaving the ratio $L / Z$ invariant. Fixing $\text{mean}(C) = 1$ resolves the scale degree of freedom.

6. **Numeric $C$ Update in MEM+SBM**:
   - In `models.fit_mem_sbm`, the community affinity matrix $C$ is updated numerically using L-BFGS-B on $\log(C)$ (upper triangle) rather than solving the paper's implicit fixed-point equations (Eq. 7).
   - Parameterizing in log-space guarantees that all matrix entries remain strictly positive ($C > 0$), avoids divergence, and monotonically increases the log-likelihood at every iteration.

7. **Generator Normalization & Stability**:
   - In `generator.generate_memory_graph`, transition probabilities are clipped and normalized to ensure valid row-stochastic probability vectors across varying degree distributions.
   - In `diffusion.run_diffusion`, small numerical negatives that arise when the diffusion rate $\beta > 1/d_{\max}$ are clipped to 0, ensuring strict probability conservation ($\sum u = 1$).

8. **SIR Epidemic Rates and Calibration**:
   - In `sir.calibrate_rate`, transmission rate $\beta$ is calibrated via bisection search so that mean unconditioned outbreak sizes fall into the critical percolation transition regime (30% to 50%, target 40%).
   - Below 30%, outbreaks frequently sputter out after 1 hop ($1/n$), masking temporal dynamics; above 50%, outbreaks saturate the network ($>85\%$), erasing sensitivity to contact memory and community boundaries.
   - In `sir.simulate_sir`, nodes isolated from the start (quarantined/removed) cannot be chosen as seed nodes and cannot contract or transmit infection; they are excluded from the numerator when computing final infected attack rates.

---

## Time-Shuffled Null Model Details

To isolate whether memory in time-respecting paths arises from the sequential ordering and temporal continuity of contacts rather than static network topology, `nulls.time_shuffle_graph(graph, rng)` randomly permutes the temporal order of snapshots within each observation day:

$$\{G_1, G_2, \dots, G_T\} \xrightarrow{\pi} \{G_{\pi(1)}, G_{\pi(2)}, \dots, G_{\pi(T)}\}$$

### What the Time-Shuffled Null Destroys
- **Temporal Ordering and Causal Direction**: Randomizing snapshot order destroys the arrow of time and breaks multi-step causal paths requiring $t_1 \le t_2 \le \dots \le t_k$.
- **Interaction Continuity and Burstiness**: Real contacts exhibit bursty clustering across consecutive snapshots; time-shuffling disperses these interactions across the timeline into isolated events.
- **Dynamic Memory**: Correlations between past interactions and future contact choices are broken.

### What the Time-Shuffled Null Preserves
- **Individual Snapshot Topologies**: Every snapshot $G_t = (V, E_t)$ remains intact. Degrees, local clustering, triangles, and instantaneous community structures are 100% preserved.
- **Aggregate Static Graph**: The static aggregate matrix $A_{\text{agg}} = \sum_t A_t$ is preserved.
- **Snapshot Activity Distribution**: The exact sequence of snapshot edge counts $\{|E_t|\}$ is preserved (merely permuted).

---

## Citation

```bibtex
@article{guerrini2026modeling,
  title={Modeling memory in time-respecting paths on temporal networks},
  author={Guerrini, Eleonora and Cattuto, Ciro and Dall'Amico, Lorenzo},
  journal={EPJ Data Science},
  year={2026}
}
```

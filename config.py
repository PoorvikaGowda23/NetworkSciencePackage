"""
config.py
=========
Central configuration file. Every tunable parameter lives here.
No magic numbers should appear anywhere else in the codebase.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Data resolution
T_RES_SECONDS: int = 20          # Temporal resolution of contact data (seconds)
DAY_SPLIT_GAP_SECONDS: int = 14400  # Minimum gap (seconds, 4h) defining day boundaries for daily splitting
RESOLUTION_T_RES_LIST: list[int] = [20, 60, 300, 600, 900]  # Window durations (seconds) for Step 10
RESOLUTION_DATASET: str = "primaryschool"                   # Default dataset for Step 10 resolution study
RESOLUTION_NUM_PATHS: int = 500                             # Paths sampled per day for Step 10
TAU_MINUTES_LIST: list[float] = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0]  # Time-based memory horizons (minutes)

# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------
MEMORY_HORIZONS: list[int] = [5, 10, 15, 20]   # Look-back windows (in units of T_RES_SECONDS) for memory p
P_INIT: float = 0.5              # Initial guess for fixed-point iteration on p

# ---------------------------------------------------------------------------
# Fixed-point solver
# ---------------------------------------------------------------------------
FIXED_POINT_TOL: float = 1e-8   # Convergence tolerance for fixed-point iteration
FIXED_POINT_MAXITER: int = 1000  # Maximum iterations allowed before declaring non-convergence
FIXED_POINT_DAMPING: float = 0.15  # Step damping factor (Mann relaxation) for fixed-point stability
SBM_MAXITER: int = 15            # Max alternating iterations for the MEM+SBM fit (L-BFGS-B per step)

# ---------------------------------------------------------------------------
# Path sampling
# ---------------------------------------------------------------------------
NUM_PATHS: int = 20_000          # Number of random TRPs to sample per measurement
BIC_NUM_PATHS: int = 500         # TRPs sampled per day for multi-dataset BIC comparison (Step 8)
PATH_LENGTH: int = 30            # Number of hops in each sampled time-respecting path
MAX_TRP_RETRIES_PER_PATH: int = 1000  # Max retry attempts allowed to generate each valid path

# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------
N_NULL_REALIZATIONS: int = 50   # Number of null-model reshufflings to generate
QUICK_NULL_REALIZATIONS: int = 3  # Fast test run: null realizations
QUICK_NUM_PATHS: int = 300        # Fast test run: sampled paths per day

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42            # Master seed passed to numpy random Generators

# ---------------------------------------------------------------------------
# Diffusion / spreading
# ---------------------------------------------------------------------------
DIFFUSION_BETA: float = 0.03    # Transmission probability per contact in SI diffusion
DIFFUSION_RUNS: int = 15        # Independent diffusion runs used to estimate spreading speed

# Step 13 — diffusion on synthetic memory graphs
DIFFUSION_ALPHA_GRID: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DIFFUSION_D_VALUES: list[float] = [3.0, 6.0, 10.0]   # Three target average-degree values
DIFFUSION_N_GRAPHS: int = 10    # Synthetic graphs generated per (alpha, d) combination
DIFFUSION_M_FIT: int = 5        # Memory horizon m used when fitting MEM on synthetic graphs
DIFFUSION_CLIP_WARN: float = 0.01  # Warn if total clipped probability mass exceeds this threshold
STEP13_NUM_PATHS: int = 500     # TRPs sampled per graph in Step 13 (smaller than NUM_PATHS to stay fast)


# ---------------------------------------------------------------------------
# Synthetic network generator
# ---------------------------------------------------------------------------
GEN_N: int = 250                 # Number of nodes in the synthetic temporal network
GEN_T: int = 300                 # Number of time steps to generate
GEN_M_HAT: int = 5              # Mean number of contacts per node per time step

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
N_BOOTSTRAP: int = 200           # Number of bootstrap resamples for confidence intervals

# ---------------------------------------------------------------------------
# SIR epidemic model
# ---------------------------------------------------------------------------
N_SIR_RUNS: int = 200           # Number of independent SIR simulation runs
SIR_RECOVERY_SNAPSHOTS: int = 30   # Recovery duration in snapshots (30 snapshots * 20s = 10 min)
SIR_TARGET_OUTBREAK: float = 0.40  # Target outbreak fraction for rate calibration (30% to 50%)
SIR_CALIBRATION_RUNS: int = 25     # Simulation runs per rate-calibration step

# ---------------------------------------------------------------------------
# Filesystem paths  (resolved relative to this file so the package is portable)
# ---------------------------------------------------------------------------
_BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = _BASE_DIR / "data"       # Raw and processed contact-sequence datasets
RESULTS_DIR: Path = _BASE_DIR / "results" # Output figures, tables, and serialised results
CACHE_DIR_STEP09: Path = RESULTS_DIR / "cache_step09"  # Cache for Step 9 null model paths and fits

# ---------------------------------------------------------------------------
# Step 16 — per-community memory
# ---------------------------------------------------------------------------
MIN_GROUP_RECORDS: int = 200   # Skip a community group that has fewer records than this
STEP16_M: int = 10             # Memory horizon m used in the per-group analysis
STEP16_NUM_PATHS: int = 1500   # TRPs sampled per observation day in Step 16

# ---------------------------------------------------------------------------
# Step 18 — targeted isolation interventions
# ---------------------------------------------------------------------------
INTERVENTION_BUDGET_PERCENT: float = 10.0  # Budget: percentage k of nodes to isolate (default 10%)
INTERVENTION_MIN_RECORDS: int = 30         # Minimum records required to compute return rate for a node
STEP18_M: int = 10                         # Look-back window m for transition records in Step 18
STEP18_NUM_PATHS: int = 1500               # Sampled TRPs to estimate node-level return rate

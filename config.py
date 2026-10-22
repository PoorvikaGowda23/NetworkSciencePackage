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

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42            # Master seed passed to numpy random Generators

# ---------------------------------------------------------------------------
# Diffusion / spreading
# ---------------------------------------------------------------------------
DIFFUSION_BETA: float = 0.03    # Transmission probability per contact in SI diffusion
DIFFUSION_RUNS: int = 15        # Independent diffusion runs used to estimate spreading speed

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

# ---------------------------------------------------------------------------
# Filesystem paths  (resolved relative to this file so the package is portable)
# ---------------------------------------------------------------------------
_BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = _BASE_DIR / "data"       # Raw and processed contact-sequence datasets
RESULTS_DIR: Path = _BASE_DIR / "results" # Output figures, tables, and serialised results

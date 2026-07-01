"""
config.py
=========
Central configuration for the BioVentureSim project.

All paths, global constants, simulation parameters, and reproducibility
settings live here so that every other module imports from a single source
of truth. Edit values here rather than scattering "magic numbers" through
the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Project paths (resolved relative to this file so the project is portable)
# ----------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ASSUMPTIONS_DIR = DATA_DIR / "assumptions"

FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Key files
ASSUMPTIONS_CSV = ASSUMPTIONS_DIR / "ai_drug_discovery_market_assumptions.csv"
RAW_TRIALS_JSON = RAW_DIR / "clinicaltrials_ai_drug_discovery.json"
PROCESSED_TRIALS_CSV = PROCESSED_DIR / "clinicaltrials_ai_trials.csv"

SIMULATION_RESULTS_CSV = OUTPUTS_DIR / "simulation_results.csv"
SEGMENT_SUMMARY_CSV = OUTPUTS_DIR / "segment_summary.csv"
SENSITIVITY_CSV = OUTPUTS_DIR / "sensitivity_results.csv"
MODEL_METRICS_CSV = OUTPUTS_DIR / "model_metrics.csv"
PORTFOLIO_CSV = OUTPUTS_DIR / "portfolio_summary.csv"
EFFICIENT_FRONTIER_CSV = OUTPUTS_DIR / "efficient_frontier.csv"

ALL_DIRS = [
    RAW_DIR,
    PROCESSED_DIR,
    ASSUMPTIONS_DIR,
    FIGURES_DIR,
    OUTPUTS_DIR,
    NOTEBOOKS_DIR,
]

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
RANDOM_SEED = 42

# ----------------------------------------------------------------------
# Monte Carlo simulation parameters
# ----------------------------------------------------------------------
N_SIMULATIONS = 10_000          # number of Monte Carlo paths per segment
START_YEAR = 2026
END_YEAR = 2035
HORIZON_YEARS = END_YEAR - START_YEAR + 1   # inclusive -> 10 years

# Event-effect magnitudes (multiplicative shocks applied to a given year).
# These are deliberately simple, transparent assumptions.
BREAKTHROUGH_UPLIFT = 0.25       # +25% in a breakthrough year
FAILURE_SHOCK_DRAWDOWN = 0.30    # -30% in a failure-shock year

# Strength of how each qualitative driver bends the growth rate. Each is a
# small multiplier weight; see monte_carlo.py for how they combine.
REGULATORY_WEIGHT = 0.06         # regulatory acceptance acceleration
CLINICAL_WEIGHT = 0.05           # clinical validation boost
FUNDING_WEIGHT = 0.05            # venture funding sensitivity
PARTNERSHIP_WEIGHT = 0.04        # pharma partnership intensity
COMPUTE_WEIGHT = 0.05            # compute cost drag (subtracts from growth)
COMPETITION_WEIGHT = 0.05        # competition compression (subtracts)

# Funding cycle: a single macro venture-funding factor is sampled per year
# and shared across segments (captures correlated boom/bust dynamics).
FUNDING_CYCLE_MEAN = 1.0
FUNDING_CYCLE_STD = 0.12

# ----------------------------------------------------------------------
# Quant / risk parameters
# ----------------------------------------------------------------------
VAR_CONFIDENCE = 0.95            # 95% Value-at-Risk / Expected Shortfall
RISK_FREE_RATE = 0.04            # used for Sharpe-like ratios

# ----------------------------------------------------------------------
# Portfolio optimisation
# ----------------------------------------------------------------------
N_PORTFOLIOS = 20_000            # random portfolios for the efficient frontier

# ----------------------------------------------------------------------
# Data fetching (ClinicalTrials.gov API v2)
# ----------------------------------------------------------------------
CLINICALTRIALS_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
QUERY_TERMS = [
    "artificial intelligence drug discovery",
    "machine learning drug discovery",
    "AI clinical trial optimization",
    "computational drug discovery",
    "protein design",
    "target discovery",
    "drug repurposing",
]
MAX_RECORDS_PER_QUERY = 200      # keep network requests modest
REQUEST_TIMEOUT = 30             # seconds
USE_NETWORK = True               # if False, data_fetch falls back to synthetic data

# ----------------------------------------------------------------------
# Plotting style
# ----------------------------------------------------------------------
FIG_DPI = 150
FIG_STYLE = "seaborn-v0_8-whitegrid"
PALETTE = "viridis"


def ensure_dirs() -> None:
    """Create every project directory if it does not already exist."""
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)


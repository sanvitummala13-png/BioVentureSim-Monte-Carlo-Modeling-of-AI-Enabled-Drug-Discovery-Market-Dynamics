"""
assumptions.py
==============
Load and validate the manual market-assumptions CSV that drives the Monte
Carlo engine. All values here are clearly-labelled ASSUMPTIONS, not data.

Responsibilities:
* read data/assumptions/ai_drug_discovery_market_assumptions.csv
* validate that every required column is present
* normalise probability / score columns into clean 0-1 fractions
* expose convenience accessors used by the simulation
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.utils import as_fraction, get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "segment",
    "current_market_size_usd",
    "cagr_low",
    "cagr_base",
    "cagr_high",
    "volatility",
    "regulatory_acceptance",
    "clinical_validation_score",
    "funding_sensitivity",
    "partnership_intensity",
    "compute_cost_risk",
    "competition_intensity",
    "breakthrough_probability",
    "failure_shock_probability",
]

# Columns that represent a 0-1 fraction (probability, score, or rate) and
# should be normalised with as_fraction (so 27 and 0.27 both -> 0.27).
FRACTION_COLUMNS = [
    "cagr_low", "cagr_base", "cagr_high", "volatility",
    "regulatory_acceptance", "clinical_validation_score",
    "funding_sensitivity", "partnership_intensity", "compute_cost_risk",
    "competition_intensity", "breakthrough_probability",
    "failure_shock_probability",
]


def load_assumptions(path=config.ASSUMPTIONS_CSV) -> pd.DataFrame:
    """
    Load the assumptions CSV, validate it, and return a normalised DataFrame.

    Raises
    ------
    FileNotFoundError : if the CSV does not exist.
    ValueError        : if required columns are missing or CAGR ordering is
                        inconsistent (low <= base <= high).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Assumptions CSV not found at {path}. "
            "Run main.py (it auto-creates a default) or restore the file."
        )

    df = pd.read_csv(path)

    # --- Validate required columns -----------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Assumptions CSV missing columns: {missing}")

    # --- Normalise fraction columns ----------------------------------
    for col in FRACTION_COLUMNS:
        df[col] = df[col].apply(as_fraction)

    # --- Sanity checks -----------------------------------------------
    df["current_market_size_usd"] = pd.to_numeric(
        df["current_market_size_usd"], errors="coerce"
    )
    bad_cagr = df[(df["cagr_low"] > df["cagr_base"]) |
                  (df["cagr_base"] > df["cagr_high"])]
    if not bad_cagr.empty:
        raise ValueError(
            "CAGR ordering violated (need low <= base <= high) for: "
            f"{bad_cagr['segment'].tolist()}"
        )

    logger.info(
        "Loaded %d market segments from assumptions CSV.", len(df)
    )
    return df.reset_index(drop=True)


def get_segments(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of segment names."""
    return df["segment"].tolist()


if __name__ == "__main__":
    print(load_assumptions().to_string(index=False))

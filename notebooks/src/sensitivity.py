"""
sensitivity.py
==============
One-at-a-time (OAT) sensitivity analysis. For each assumption driver we
perturb it up and down by a fixed relative amount, re-run a (smaller) Monte
Carlo, and measure the change in total expected 2035 market size. The result
feeds a tornado plot showing which assumptions matter most.

Perturbed drivers:
    cagr (base), volatility, regulatory_acceptance, clinical_validation_score,
    funding_sensitivity, partnership_intensity, compute_cost_risk,
    competition_intensity, breakthrough_probability, failure_shock_probability
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.monte_carlo import run_monte_carlo, summarise_segments
from src.utils import get_logger

logger = get_logger(__name__)

PERTURB_PARAMS = [
    "cagr_base",
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


def _total_expected_2035(assumptions: pd.DataFrame, n_sims: int) -> float:
    """Run MC at reduced sim count and return total expected 2035 size."""
    # Temporarily shrink the simulation count for speed.
    original = config.N_SIMULATIONS
    config.N_SIMULATIONS = n_sims
    try:
        res = run_monte_carlo(assumptions, save=False)
        summ = summarise_segments(res, assumptions, save=False)
    finally:
        config.N_SIMULATIONS = original
    return float(summ["expected_2035_market_size"].sum())


def run_sensitivity(assumptions: pd.DataFrame,
                    perturbation: float = 0.20,
                    n_sims: int = 2_000,
                    save: bool = True) -> pd.DataFrame:
    """
    Perform OAT sensitivity analysis.

    Parameters
    ----------
    perturbation : relative +/- change applied to each driver (0.20 = +/-20%).
    n_sims       : reduced Monte Carlo paths per evaluation (for speed).

    Returns a DataFrame (one row per parameter) with low/high/baseline totals
    and the resulting swing, sorted by absolute impact (tornado order).
    """
    baseline = _total_expected_2035(assumptions, n_sims)
    logger.info("Sensitivity baseline total 2035 size: %.3e", baseline)

    records = []
    for param in PERTURB_PARAMS:
        low_df = assumptions.copy()
        high_df = assumptions.copy()

        # For CAGR we shift base but keep low<=base<=high consistent.
        if param == "cagr_base":
            low_df["cagr_base"] = np.clip(
                low_df["cagr_base"] * (1 - perturbation),
                low_df["cagr_low"], low_df["cagr_high"])
            high_df["cagr_base"] = np.clip(
                high_df["cagr_base"] * (1 + perturbation),
                high_df["cagr_low"], high_df["cagr_high"])
        else:
            low_df[param] = (low_df[param] * (1 - perturbation)).clip(0, 1)
            high_df[param] = (high_df[param] * (1 + perturbation)).clip(0, 1)

        low_total = _total_expected_2035(low_df, n_sims)
        high_total = _total_expected_2035(high_df, n_sims)
        swing = abs(high_total - low_total)

        records.append({
            "parameter": param,
            "low_total": low_total,
            "baseline_total": baseline,
            "high_total": high_total,
            "low_delta": low_total - baseline,
            "high_delta": high_total - baseline,
            "swing": swing,
        })
        logger.info("  %-28s swing=%.3e", param, swing)

    df = pd.DataFrame(records).sort_values("swing", ascending=False).reset_index(drop=True)

    if save:
        config.ensure_dirs()
        df.to_csv(config.SENSITIVITY_CSV, index=False)
        logger.info("Saved sensitivity results -> %s", config.SENSITIVITY_CSV)
    return df


if __name__ == "__main__":
    from src.assumptions import load_assumptions

    print(run_sensitivity(load_assumptions()).to_string(index=False))


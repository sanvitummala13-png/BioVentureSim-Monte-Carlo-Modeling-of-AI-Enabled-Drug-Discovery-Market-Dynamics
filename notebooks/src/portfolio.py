"""
portfolio.py
============
Treat the AI drug-discovery subsegments as asset classes and apply
mean-variance portfolio theory.

We derive each segment's annual return series from the Monte Carlo output
(mean annual growth per simulated year), then compute:
    * expected return per segment
    * covariance matrix across segments
    * Sharpe-like ratios
    * thousands of random portfolios (Monte Carlo on the simplex)
    * the efficient frontier
    * the minimum-variance portfolio
    * the maximum risk-adjusted (max-Sharpe) portfolio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def build_return_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (year x segment) matrix of average annual returns.

    For each (segment, year) we average the annual growth across all
    simulations, giving a return time-series per segment that we can treat
    like an asset's historical returns.
    """
    matrix = (results.groupby(["year", "segment"])["annual_growth"]
                     .mean().unstack("segment"))
    return matrix


def portfolio_stats(weights, mean_returns, cov):
    """Return (expected_return, volatility, sharpe) for a weight vector."""
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - config.RISK_FREE_RATE) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def run_portfolio(results: pd.DataFrame, save: bool = True):
    """
    Full portfolio analysis. Returns a dict bundle with:
        return_matrix, mean_returns, cov, frontier (DataFrame),
        min_var (dict), max_sharpe (dict), asset_table (DataFrame).
    """
    rng = np.random.default_rng(config.RANDOM_SEED)

    returns = build_return_matrix(results)
    segments = list(returns.columns)
    mean_returns = returns.mean().values
    cov = returns.cov().values
    n_assets = len(segments)

    # Per-asset Sharpe-like ratios
    asset_vol = returns.std().values
    asset_sharpe = np.where(
        asset_vol > 0, (mean_returns - config.RISK_FREE_RATE) / asset_vol, 0.0
    )
    asset_table = pd.DataFrame({
        "segment": segments,
        "expected_return": mean_returns,
        "volatility": asset_vol,
        "sharpe_like": asset_sharpe,
    }).sort_values("sharpe_like", ascending=False).reset_index(drop=True)

    # --- Random portfolios on the simplex ---------------------------
    n = config.N_PORTFOLIOS
    rets = np.empty(n)
    vols = np.empty(n)
    sharpes = np.empty(n)
    weights_record = np.empty((n, n_assets))

    for i in range(n):
        w = rng.random(n_assets)
        w /= w.sum()
        weights_record[i] = w
        rets[i], vols[i], sharpes[i] = portfolio_stats(w, mean_returns, cov)

    frontier = pd.DataFrame({
        "expected_return": rets,
        "volatility": vols,
        "sharpe": sharpes,
    })

    # Min-variance and max-Sharpe portfolios
    min_idx = int(np.argmin(vols))
    max_idx = int(np.argmax(sharpes))

    min_var = {
        "expected_return": rets[min_idx],
        "volatility": vols[min_idx],
        "sharpe": sharpes[min_idx],
        "weights": dict(zip(segments, weights_record[min_idx].round(4))),
    }
    max_sharpe = {
        "expected_return": rets[max_idx],
        "volatility": vols[max_idx],
        "sharpe": sharpes[max_idx],
        "weights": dict(zip(segments, weights_record[max_idx].round(4))),
    }

    logger.info("Max-Sharpe portfolio sharpe=%.3f", max_sharpe["sharpe"])
    logger.info("Min-variance portfolio vol=%.4f", min_var["volatility"])

    if save:
        config.ensure_dirs()
        frontier.to_csv(config.EFFICIENT_FRONTIER_CSV, index=False)

        # A tidy summary CSV with the two special portfolios + per-asset stats.
        summary_rows = []
        for label, port in [("min_variance", min_var), ("max_sharpe", max_sharpe)]:
            row = {"portfolio": label,
                   "expected_return": port["expected_return"],
                   "volatility": port["volatility"],
                   "sharpe": port["sharpe"]}
            row.update({f"w_{k}": v for k, v in port["weights"].items()})
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(config.PORTFOLIO_CSV, index=False)
        asset_table.to_csv(config.OUTPUTS_DIR / "asset_stats.csv", index=False)
        logger.info("Saved portfolio outputs -> %s", config.PORTFOLIO_CSV)

    return {
        "return_matrix": returns,
        "mean_returns": mean_returns,
        "cov": cov,
        "segments": segments,
        "frontier": frontier,
        "min_var": min_var,
        "max_sharpe": max_sharpe,
        "asset_table": asset_table,
    }


if __name__ == "__main__":
    sim = pd.read_csv(config.SIMULATION_RESULTS_CSV)
    bundle = run_portfolio(sim)
    print(bundle["asset_table"].to_string(index=False))


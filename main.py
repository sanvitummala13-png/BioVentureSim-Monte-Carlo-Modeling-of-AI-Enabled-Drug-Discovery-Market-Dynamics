"""
main.py
=======
BioVentureSim - single-command pipeline runner.

Run from the project root:

    python main.py

It will, in order:
    1. create folders if they do not exist
    2. create the assumptions CSV if it does not exist
    3. fetch or load clinical-trial data
    4. run the Monte Carlo simulation (10,000 paths, 2026-2035)
    5. compute per-segment quant/risk metrics
    6. run one-at-a-time sensitivity analysis
    7. train ML outperformance classifiers
    8. run mean-variance portfolio analysis
    9. generate every figure
   10. save all CSV / JSON outputs

Use --quick for a fast smoke test (fewer simulations).
"""

from __future__ import annotations

import argparse

from src import config
from src.utils import get_logger

logger = get_logger("BioVentureSim.main")

# A minimal default assumptions table written if the CSV is missing.
_DEFAULT_ASSUMPTIONS = """segment,current_market_size_usd,cagr_low,cagr_base,cagr_high,volatility,regulatory_acceptance,clinical_validation_score,funding_sensitivity,partnership_intensity,compute_cost_risk,competition_intensity,breakthrough_probability,failure_shock_probability
AI small molecule discovery,1800000000,0.18,0.27,0.38,0.22,0.55,0.50,0.65,0.70,0.45,0.70,0.12,0.08
AI biologics and protein design,1200000000,0.22,0.32,0.45,0.28,0.45,0.42,0.70,0.65,0.60,0.55,0.15,0.10
AI target discovery,900000000,0.20,0.30,0.42,0.25,0.50,0.48,0.60,0.72,0.40,0.60,0.13,0.07
AI drug repurposing,600000000,0.15,0.22,0.30,0.18,0.60,0.55,0.50,0.55,0.30,0.50,0.10,0.06
AI clinical trial optimization,750000000,0.17,0.25,0.35,0.20,0.48,0.45,0.58,0.62,0.35,0.55,0.11,0.07
AI toxicology and safety prediction,500000000,0.16,0.24,0.33,0.19,0.52,0.50,0.55,0.50,0.38,0.48,0.10,0.06
AI lab automation and robotics,1100000000,0.19,0.28,0.40,0.24,0.58,0.46,0.68,0.58,0.65,0.62,0.12,0.09
"""


def ensure_assumptions_csv() -> None:
    """Write the default assumptions CSV if the user has not provided one."""
    if not config.ASSUMPTIONS_CSV.exists():
        config.ensure_dirs()
        config.ASSUMPTIONS_CSV.write_text(_DEFAULT_ASSUMPTIONS, encoding="utf-8")
        logger.info("Created default assumptions CSV -> %s", config.ASSUMPTIONS_CSV)


def main(quick: bool = False) -> None:
    # Imports kept local so `python main.py --help` is fast.
    from src.assumptions import load_assumptions
    from src.data_cleaning import clean_clinical_trials
    from src.data_fetch import fetch_clinical_trials
    from src.ml_models import train_models
    from src.monte_carlo import run_monte_carlo, summarise_segments
    from src.portfolio import run_portfolio
    from src.sensitivity import run_sensitivity
    from src.visualization import generate_all_figures

    if quick:
        config.N_SIMULATIONS = 1_000
        config.N_PORTFOLIOS = 3_000
        logger.info("QUICK mode: reduced simulation counts.")

    # 1 & 2 -----------------------------------------------------------
    logger.info("[1/9] Ensuring directories and assumptions CSV ...")
    config.ensure_dirs()
    ensure_assumptions_csv()

    # 3 ---------------------------------------------------------------
    logger.info("[2/9] Fetching / loading clinical-trial data ...")
    raw = fetch_clinical_trials()
    trials = clean_clinical_trials(raw)
    logger.info("    -> %d clean trials", len(trials))

    # Load assumptions
    assumptions = load_assumptions()

    # 4 ---------------------------------------------------------------
    logger.info("[3/9] Running Monte Carlo simulation ...")
    results = run_monte_carlo(assumptions)

    # 5 ---------------------------------------------------------------
    logger.info("[4/9] Computing quant / risk summary ...")
    summary = summarise_segments(results, assumptions)

    # 6 ---------------------------------------------------------------
    logger.info("[5/9] Running sensitivity analysis ...")
    sensitivity = run_sensitivity(
        assumptions, n_sims=500 if quick else 2_000
    )

    # 7 ---------------------------------------------------------------
    logger.info("[6/9] Training ML models ...")
    metrics, importance = train_models(results)

    # 8 ---------------------------------------------------------------
    logger.info("[7/9] Running portfolio analysis ...")
    portfolio_bundle = run_portfolio(results)

    # 9 ---------------------------------------------------------------
    logger.info("[8/9] Generating figures ...")
    generate_all_figures(results, summary, sensitivity,
                         portfolio_bundle, importance)

    logger.info("[9/9] Done. Outputs in %s, figures in %s",
                config.OUTPUTS_DIR, config.FIGURES_DIR)
    print("\n=== Segment summary (expected 2035 market size) ===")
    print(summary[["segment", "expected_2035_market_size",
                   "prob_double", "dominance_probability"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BioVentureSim pipeline.")
    parser.add_argument("--quick", action="store_true",
                        help="Run a fast smoke test with fewer simulations.")
    args = parser.parse_args()
    main(quick=args.quick)


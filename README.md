# BioVentureSim
### Monte Carlo Modeling of AI-Enabled Drug Discovery Market Dynamics

A research-style quantitative project that simulates how uncertainty across
**regulatory acceptance, pharma partnerships, clinical-trial productivity,
venture funding, model performance, compute costs, and scientific validation**
shapes the future market for AI-enabled drug discovery. It produces
publication-quality figures, CSV outputs, and reproducible notebooks — **not** a
dashboard.

---

## Project overview

BioVentureSim models seven AI drug-discovery subsegments as stochastic,
interacting "asset classes," runs **10,000 Monte Carlo paths over 2026–2035**,
and then layers quantitative-finance risk metrics, sensitivity analysis,
machine learning, and portfolio optimisation on top of the simulated universe.

## Research question

> How do uncertainty in regulatory acceptance, pharma partnerships, clinical
> trial productivity, venture funding, model performance, compute costs, and
> scientific validation shape the future market dynamics of AI-enabled drug
> discovery?

## Methodology

1. **Data layer** — pull AI/biotech trials from the ClinicalTrials.gov API for
   context; load a manually-curated market-assumptions table.
2. **Monte Carlo engine** — for each segment/year, sample an annual growth rate
   and apply correlated shocks:
   - CAGR — **triangular(low, base, high)**
   - volatility — **normal** noise
   - venture funding — shared macro **funding cycle** scaled by segment sensitivity
   - regulatory acceleration & clinical-validation boost — driver-weighted lifts
   - compute-cost drag — **lognormal**
   - competition compression — **beta**
   - breakthrough / failure events — **Bernoulli** multiplicative shocks
3. **Quant metrics** — expected/percentile 2035 size, probability of doubling or
   declining, annualized volatility, downside risk, **Value-at-Risk**, **Expected
   Shortfall**, risk-adjusted growth, and **dominance probability**.
4. **Sensitivity** — one-at-a-time perturbation of every driver → tornado plot.
5. **Machine learning** — logistic regression, random forest, gradient boosting
   predict whether a segment-year **outperforms the market**.
6. **Portfolio theory** — covariance, Sharpe-like ratios, efficient frontier,
   minimum-variance and maximum-Sharpe portfolios.
7. **Visualization** — 11 figures saved to `figures/`.

## Datasets & sources

| Source | Use in this project |
|---|---|
| **ClinicalTrials.gov API (v2)** | Live fetch of AI/ML drug-discovery trial counts, sponsors, phases, status, dates, conditions, interventions. |
| **AACT database** | Documented as the structured (PostgreSQL) ClinicalTrials.gov option for future SQL-based expansion. |
| **Open Targets Platform** | GraphQL hook (`fetch_open_targets_disease`) for target-disease-drug evidence and therapeutic-area signals. |
| **FDA AI drug-development guidance** | Encoded as manual assumptions for regulatory AI-acceptance trends. |
| **Manual market assumptions CSV** | `data/assumptions/ai_drug_discovery_market_assumptions.csv` — the primary driver table. |

> **Offline-safe:** if the network is unavailable (or a corporate proxy blocks
> ClinicalTrials.gov), `data_fetch.py` automatically falls back to a clearly-
> labelled synthetic dataset so the full pipeline still runs.

## Assumptions

All values in `ai_drug_discovery_market_assumptions.csv` are **illustrative
placeholders**, not forecasts. Segments covered:

- AI small molecule discovery
- AI biologics and protein design
- AI target discovery
- AI drug repurposing
- AI clinical trial optimization
- AI toxicology and safety prediction
- AI lab automation and robotics

Columns: `segment, current_market_size_usd, cagr_low, cagr_base, cagr_high,
volatility, regulatory_acceptance, clinical_validation_score, funding_sensitivity,
partnership_intensity, compute_cost_risk, competition_intensity,
breakthrough_probability, failure_shock_probability`.

## How to run (VS Code)

```bash
# 1. Open the BioVentureSim folder in VS Code
# 2. (Optional) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python main.py

# Fast smoke test (fewer simulations):
python main.py --quick
```

Then open any notebook in `notebooks/` (select the same interpreter) and run.

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `simulation_results.csv` | Full long-format Monte Carlo output (per simulation/year/segment). |
| `segment_summary.csv` | Per-segment quant & risk metrics. |
| `sensitivity_results.csv` | Tornado swings per assumption. |
| `model_metrics.csv` | Accuracy, F1, ROC-AUC, confusion-matrix counts per model. |
| `feature_importance.csv` | Feature importances across models. |
| `portfolio_summary.csv` | Min-variance & max-Sharpe portfolio weights. |
| `efficient_frontier.csv` | Random-portfolio cloud (return, vol, Sharpe). |
| `asset_stats.csv` | Per-segment expected return / volatility / Sharpe. |

## Figures (`figures/`)

1. `01_fan_chart.png` — Monte Carlo fan chart
2. `02_market_2035_distribution.png` — 2035 market-size distribution
3. `03_scenarios.png` — bull / base / bear paths
4. `04_volatility_heatmap.png` — volatility heatmap
5. `05_tornado.png` — tornado sensitivity plot
6. `06_correlation_matrix.png` — segment return correlation matrix
7. `07_efficient_frontier.png` — efficient frontier
8. `08_feature_importance.png` — feature importance
9. `09_regime_clusters.png` — PCA regime clusters
10. `10_value_at_risk.png` — Value-at-Risk bar chart
11. `11_dominance_probability.png` — segment dominance probability

## Project structure

```
BioVentureSim/
  README.md
  requirements.txt
  main.py
  data/{raw,processed,assumptions}/
  src/{config,data_fetch,data_cleaning,assumptions,monte_carlo,
       sensitivity,ml_models,portfolio,visualization,utils}.py
  notebooks/01..05_*.ipynb
  figures/
  outputs/
```

## Future roadmap

- Swap the synthetic fallback for a cached AACT PostgreSQL pull (full SQL queries).
- Enrich segments with live Open Targets evidence scores.
- Add correlated multi-segment shocks via a copula instead of a shared funding factor.
- Bayesian calibration of assumptions against historical biotech funding data.
- Constrained mean-variance optimisation (no-short, sector caps) with `cvxpy`.

## Disclaimer

This project is for **educational and research purposes only**. It is **not
investment advice, medical advice, or a market forecast**. All assumptions are
illustrative placeholders; simulated outputs do not represent real or predicted
market outcomes.

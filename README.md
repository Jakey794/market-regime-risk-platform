# Market Regime + Portfolio Risk Platform

A Python research platform for **portfolio risk measurement**, **market-regime
analysis**, **stress testing**, and **no-look-ahead backtesting**.

This is **not** a stock-prediction app. It does not claim reliable alpha,
guaranteed risk reduction, or investment performance.

The goal is to study how portfolios behave across changing market environments
and to produce interpretable, leakage-aware research outputs.

## Current Status (through Week 5)

| Week | Deliverable | Status |
| --- | --- | --- |
| 1 | Config-driven ETF ingestion, Parquet cache, validation | Complete |
| 2 | Risk / return / drawdown / tail / performance metrics | Complete |
| 3 | Portfolio config, concentration, correlation, beta, risk contribution | Complete |
| 4 | Four-page Streamlit dashboard on the risk engine | Complete |
| 5 | Leakage-safe regime feature engineering + diagnostics page | Complete |
| 6+ | Regime models, stress tests, backtests, memo / v1.0 polish | In progress on `cursor/complete-v1` |

See `PROJECT_COMPLETION.md` for the full v1.0 completion plan and acceptance
checklist. Do not treat Weeks 6–12 as complete until their acceptance tests pass.

## Quickstart

```bash
uv sync
make data        # download and cache adjusted-close prices
make features    # build raw + train-scaled regime features
make feature-check
make dashboard   # launch the Streamlit app
make test
make check       # ruff lint + format check + pytest
```

## Week 5: Regime Feature Engineering

Week 5 adds leakage-safe descriptive features for regime research:

- Trailing volatility, correlation, drawdown, and momentum features
- Chronological train/test split with train-only `StandardScaler` fit
- Persisted raw and scaled Parquet artifacts plus metadata
- Streamlit **Regime Feature Diagnostics** page
- Deterministic unit tests for warm-up NaNs, index preservation, and
  future-mutation leakage resistance

```bash
make features
make feature-check
```

## Dashboard

```bash
make dashboard
```

Pages currently delivered:

1. Portfolio Overview
2. Risk Metrics
3. Correlation & Beta
4. Data Quality
5. Regime Feature Diagnostics

The dashboard reports historical, deterministic estimates for research purposes
only. It is not financial advice.

## Architecture

```text
app (Streamlit pages)
  -> dashboard adapters (src/mrrp/dashboard)
    -> core engines (portfolio, risk, features, models, backtest, reporting)
      -> configs + processed data artifacts
```

Streamlit pages stay thin and call package APIs. Notebooks must call the same
package code rather than reimplement financial logic.

## Repository Structure

```text
configs/       Universe, portfolio, feature, and (upcoming) model configs
app/           Streamlit entrypoint and pages
data/          Raw/processed/sample data (generated artifacts mostly ignored)
notebooks/     Research notebooks
reports/       Notes, memos, architecture, screenshots
scripts/       Reproducible CLI workflows
src/mrrp/      Project package
tests/         Automated tests
.github/       GitHub Actions workflows
```

## Validation Commands

| Command | Purpose |
| --- | --- |
| `make setup` | Install dependencies with `uv sync` |
| `make data` | Build adjusted-close cache |
| `make features` | Build regime feature artifacts |
| `make feature-check` | Validate feature artifacts |
| `make dashboard` | Launch Streamlit |
| `make test` | Run pytest |
| `make lint` | Ruff lint |
| `make check` | Lint + format check + tests |
| `make format` | Apply Ruff formatting |
| `make clean` | Remove local caches |

## Limitations

- Free ETF data is not institutional-grade and may contain missing, revised, or
  inconsistent observations.
- This project is not financial advice.
- This project is not a return prediction system.
- This project does not perform live trading or live order execution.
- Regime features describe historical risk conditions; they are not forecasts.

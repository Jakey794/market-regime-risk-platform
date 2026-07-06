# Market Regime + Portfolio Risk Platform

This project is a portfolio risk, regime detection, stress-testing, and backtesting
platform. It is not a stock prediction app.

The repository is a Python research platform for studying how portfolios behave
across changing market environments and producing interpretable risk outputs.

## Week 1 Status

- Config-driven ETF universe
- Reproducible adjusted-close data ingestion
- Parquet data cache
- Data validation checks and missing-data reporting
- Starter test suite
- GitHub Actions CI
- Starter data audit notebook

## Week 2 Metrics Engine

The portfolio risk engine now includes:

- Simple, logarithmic, cumulative, and weighted portfolio returns
- Annualized return, volatility, rolling volatility, and EWMA volatility
- Drawdown paths, maximum/current drawdown, and drawdown duration
- Sharpe, Sortino, Calmar, tracking-error, and information ratios
- Asset, rolling, and weighted portfolio beta
- Rolling correlation matrices and mean pairwise correlation
- Historical VaR/CVaR, worst-period returns, skewness, and kurtosis
- An integrated numeric portfolio risk summary against a benchmark

## Week 3: Portfolio Risk Engine

Week 3 packages the metric primitives into a reusable portfolio analysis layer:

- Typed YAML portfolio configuration with weight, ticker, benchmark, currency,
  and short-position validation
- Aligned asset and weighted portfolio returns without silent normalization
- Concentration diagnostics including HHI, effective holdings, top-weight
  exposure, entropy, and a concentration-risk label
- Static and rolling correlation analysis with a current correlation-regime
  classification and diversification ratio
- Asset, portfolio, rolling, and up/down beta against a configured benchmark
- Marginal, component, and percentage contribution to portfolio variance
- A typed `PortfolioRiskSummary` plus dashboard-ready exposure, correlation,
  contribution, and summary-card builders

The package API keeps data preparation separate from risk math:

```python
from mrrp.portfolio import (
    build_portfolio_risk_summary,
    build_summary_cards,
    load_portfolio_config,
)
from mrrp.data.cache import load_parquet

config = load_portfolio_config("configs/sample_portfolio.yaml")
prices = load_parquet("data/processed/adjusted_close.parquet")
summary = build_portfolio_risk_summary(prices, config)
cards = build_summary_cards(summary)
```

See `notebooks/02_portfolio_risk_engine.ipynb` for the complete reproducible
workflow and `reports/week_3_portfolio_risk_notes.md` for engineering notes and
limitations.

## How to Run

```bash
uv sync
make data       # download and cache adjusted-close prices
make dashboard  # launch the Streamlit dashboard
make test       # run the automated test suite
make check      # lint, format-check, and test
```

After downloading the data, you can also open `notebooks/01_data_audit.ipynb`
from the repository root to review ticker coverage, missing data, normalized
prices, and recent daily returns.

## Week 4: Dashboard

Week 4 completed a baseline, four-page Streamlit dashboard built entirely on
the Week 2/3 risk engine — no new financial logic was added for the
dashboard itself. Start it with `make dashboard` from the repository root;
the shell reads `data/processed/adjusted_close.parquet` and
`configs/sample_portfolio.yaml` to initialize shared portfolio, benchmark, and
date-range controls used by every page.

- **Portfolio Overview** — headline summary cards (return, volatility,
  drawdown, VaR/CVaR, beta, concentration) plus cumulative return, rolling
  volatility/drawdown/beta, and weight charts.
- **Risk Metrics** — return and volatility statistics, rolling volatility at
  multiple windows, drawdown history and worst-drawdown episodes, and
  historical VaR/CVaR with a return-distribution chart.
- **Correlation & Beta** — pairwise correlation summary and heatmap, rolling
  correlation, portfolio and asset beta against the selected benchmark, and
  an approximate sector/factor proxy exposure chart (labeled as a proxy, not
  a factor model).
- **Data Quality** — selected tickers/benchmark, coverage and missing-value
  reporting per ticker, duplicate-date and alignment checks, data freshness
  relative to today, and a portfolio weight/benchmark validation summary.

Screenshots of each page can be captured by running `make dashboard` and
saving them under `reports/screenshots/` (e.g. `1_portfolio_overview.png`,
`2_risk_metrics.png`, `3_correlation_beta.png`, `4_data_quality.png`) — this
is a manual step, not automated by any script in this repository.

The dashboard reports historical, deterministic estimates for research
purposes only. It does not include regime models, stress tests, backtests,
portfolio editing, or live data refresh, and it is not financial advice.

## Architecture

```text
app (Streamlit pages)
  -> dashboard adapters (src/mrrp/dashboard: loaders, state, formatting, components)
    -> core risk engine (src/mrrp/risk, src/mrrp/portfolio)
      -> processed data (data/processed/*.parquet)
```

Streamlit pages stay thin: they call into the dashboard adapters and the
core risk engine, and never reimplement financial calculations themselves.

## Repository Structure

```text
configs/       ETF universe and portfolio configuration
app/           Streamlit dashboard entrypoint and pages
data/          Raw and processed parquet data
notebooks/     Research and data audit notebooks
reports/       Demo notes and generated research outputs
scripts/       Reproducible command-line workflows
src/mrrp/      Project package
tests/         Automated tests
.github/       GitHub Actions workflows
```

## Project Scope

**Delivered (Weeks 1-4):** ETF data ingestion and validation, the portfolio
risk/return/drawdown/tail-risk metrics engine, concentration diagnostics,
static and rolling correlation with a simple percentile-based correlation
label (a rule, not a statistical model), asset/portfolio beta, variance risk
contribution, and the four-page dashboard described above.

**Planned, not yet started:**

- Volatility and correlation **regime models** (e.g. HMM/GMM-based regime
  detection, distinct from today's simple percentile-based correlation
  label) — begins Week 6.
- **Stress testing** — begins Week 8.
- **Backtesting** (simple, no-lookahead) — begins Week 9.
- Quarterly risk memo generation — no target week set yet.

## Limitations

- Free ETF data is not institutional-grade and may contain missing, revised, or
  inconsistent observations.
- This project is not financial advice.
- This project is not a return prediction system.
- This project does not perform live trading or live order execution.

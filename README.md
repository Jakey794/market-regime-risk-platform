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

## Quickstart

```bash
uv sync
make data
make check
```

After downloading the data, open `notebooks/01_data_audit.ipynb` from the
repository root to review ticker coverage, missing data, normalized prices, and
recent daily returns.

## Repository Structure

```text
configs/       ETF universe and portfolio configuration
data/          Raw and processed parquet data
notebooks/     Research and data audit notebooks
reports/       Demo notes and generated research outputs
scripts/       Reproducible command-line workflows
src/mrrp/      Project package
tests/         Automated tests
.github/       GitHub Actions workflows
```

## Project Scope

- Volatility regime detection
- Correlation regime detection
- Drawdown risk
- Beta vs benchmark
- Portfolio concentration risk
- Sector and factor exposure proxies
- Stress tests
- Regime model comparison
- Simple no-lookahead backtests
- Quarterly risk memo generation

## Limitations

- Free ETF data is not institutional-grade and may contain missing, revised, or
  inconsistent observations.
- This project is not financial advice.
- This project is not a return prediction system.

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

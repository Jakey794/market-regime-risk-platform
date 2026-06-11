# Week 1 Demo Notes

## What Works

- A YAML-configured ETF universe drives reproducible adjusted-close downloads.
- Raw and processed price data are cached as parquet files.
- Price-frame validation checks dates, columns, price values, history, and
  missing-data levels.
- The data audit notebook summarizes coverage and visualizes the downloaded data.
- Tests, Ruff checks, and GitHub Actions CI provide a starter quality baseline.

## Commands To Run

```bash
uv sync
make data
make check
```

## Demo Flow: Under 2 Minutes

1. Open `configs/default_universe.yaml` to show the config-driven ETF universe.
2. Run `make data` and point out the downloaded shape, date range, and parquet
   output paths.
3. Open `notebooks/01_data_audit.ipynb` and run it top-to-bottom.
4. Show ticker coverage, the missing-data table, normalized prices, and recent
   daily returns.
5. Run `make check` to demonstrate linting, formatting, and tests.

## Next Week Scope

- Returns
- Volatility
- Drawdown
- VaR and CVaR
- Beta
- Tracking error

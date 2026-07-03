# Week 3 Portfolio Risk Engine Notes

## What Was Built

Week 3 turns the Week 2 metric primitives into a reusable portfolio-analysis
layer. A validated YAML configuration now drives return aggregation, exposure
analysis, concentration diagnostics, correlation regimes, beta, variance
contributions, and a typed portfolio risk summary. The implementation remains
independent of Streamlit and other presentation frameworks.

## Key Modules

- `mrrp.portfolio.config` validates names, benchmark, currency, shorting policy,
  tickers, and weights.
- `mrrp.portfolio.returns` aligns configured holdings with asset returns and
  builds portfolio-level returns.
- `mrrp.portfolio.exposures` and `mrrp.portfolio.metadata` produce holding and
  proxy-group exposures.
- `mrrp.risk.concentration` measures weight concentration and effective breadth.
- `mrrp.risk.correlation` measures pairwise dependence, rolling regimes, and
  diversification ratio.
- `mrrp.risk.beta` estimates asset and portfolio benchmark sensitivity.
- `mrrp.risk.risk_contribution` decomposes portfolio variance by holding.
- `mrrp.portfolio.summary` composes the engines into `PortfolioRiskSummary` and
  dashboard-ready tables.

## Core Metrics

- Simple, logarithmic, cumulative, and weighted portfolio returns
- Annualized return and volatility, Sharpe ratio, and Sortino ratio
- Current and maximum drawdown plus historical 95% VaR and CVaR
- HHI, effective holdings, top-three weight, entropy, and concentration label
- Mean and maximum pairwise correlation, rolling correlation regime, and
  diversification ratio
- Asset, portfolio, rolling, up-market, and down-market beta
- Portfolio variance plus marginal, component, and percentage risk contribution

Concentration uses gross-normalized absolute weights so short positions affect
concentration without confusing leverage with breadth. Risk contribution uses a
sample covariance matrix and complete observations across configured assets.

## Validation and Tests

- Weight validation rejects missing, duplicated, nonnumeric, nonfinite, and
  incorrectly summed holdings, and enforces the configured shorting policy.
- Return and benchmark calculations align dates explicitly and remove missing
  pairs only after alignment.
- Rolling correlation and beta calculations use observations available through
  each date and include no look-ahead.
- Deterministic unit tests cover every financial metric and integration facade.
- GitHub Actions runs Ruff linting, Ruff formatting checks, and the complete
  pytest suite on Python 3.14.

Run the same quality gates locally with:

```bash
make check
```

## Known Limitations

- Results are historical estimates, not forecasts or investment advice.
- The engine currently assumes fixed weights; it does not model rebalancing,
  turnover, transaction costs, taxes, or liquidity.
- Historical VaR/CVaR and sample covariance can be unstable in short samples and
  do not guarantee future tail behavior.
- Sector, region, style, and asset-class fields are ETF-level proxies rather than
  look-through holdings analysis.
- Daily annualization assumes 252 periods per year, and correlation regime uses a
  63-observation trailing window.
- Cached free-market data can contain missing, revised, or inconsistent values.
- The deterministic synthetic fallback exists only to make the demo runnable
  without cached data; it is not a calibration or validation dataset.

## What Week 4 Will Consume

The dashboard can consume package outputs directly without embedding risk math:

- `PortfolioRiskSummary` for scalar portfolio cards
- `build_summary_cards` for a plain key/value representation
- `build_exposure_table` for holdings and metadata proxy tables
- `build_correlation_table` for the asset correlation matrix
- `build_risk_contribution_table` for holding-level risk decomposition and beta
- `compute_asset_returns` and `compute_portfolio_returns` for time-series charts

The same interfaces can later feed regime features, stress tests, and the memo
generator.

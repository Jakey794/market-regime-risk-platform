# Quarterly Portfolio Risk Memo

## Executive summary

As of **2026-06-04**, the **sample_global_equity_portfolio** portfolio is reviewed
against benchmark **SPY** for risk monitoring and research. This
memo summarises volatility/correlation regimes, drawdown and tail risk,
concentration, stress tests, and historical backtest evidence. It does **not**
provide buy/sell instructions or personalised financial advice.

## Data as-of date

2026-06-04

## Current volatility regime

Normal volatility

## Current correlation regime

High correlation

## Regime-model agreement / disagreement

Models currently agree on 'calm' (threshold=calm, gmm=calm).

## Drawdown and tail risk

- Current / recent drawdown context: -0.52%
- Historical VaR (95%): -1.77%
- Historical CVaR (95%): -2.89%

## Benchmark beta

Portfolio beta vs SPY: **0.97**

## Concentration and risk contribution

Concentration label: **Moderate**

Top risk contributors:

- SPY: 34.44%
- QQQ: 15.93%
- XIU.TO: 15.69%
- EFA: 15.50%
- EEM: 12.96%

## Factor / sector proxy exposure

These are ETF sector/factor **proxies**, not a statistical factor model.

- broad_us: 35.00%
- canada_large_cap: 20.00%
- us_growth_tech_heavy: 15.00%
- developed_ex_us: 15.00%
- emerging_markets: 10.00%
- technology: 5.00%

## Stress-test results

- global_financial_crisis: unavailable
- covid_crash: -32.55%
- inflation_rate_shock_2022: -18.65%
- worst_rolling_21d: -31.47%
- worst_rolling_63d: -28.61%
- worst_rolling_126d: -24.12%
- equity_down_10: -8.80%
- emerging_market_shock: -2.00%
- tech_sector_proxy_shock: -0.75%
- benchmark_beta_shock: -9.71%
- volatility_shock: -3.54%
- correlation_shock: -2.44%

## Backtest evidence

Historical simulation metrics for research comparison only. They are not a
guarantee of future performance or alpha.

- cagr: 9.86%
- annualized_volatility: 19.16%
- sharpe: 0.5146
- sortino: 0.4791
- max_drawdown: -55.25%
- calmar: 0.1785
- tracking_error: 3.99%
- turnover: 0.00%
- transaction_cost_drag: 0.00%
- worst_month: -26.10%
- rolling_12m_return_last: 38.55%
- fraction_outperforming_months: 50.62%
- underperformance_duration_days: 4002

## Review considerations

- Revisit defensive sleeves if high-volatility and high-correlation regimes agree.
- Check whether concentration and risk contribution remain aligned with policy.
- Confirm stress coverage limitations when ETF history is shorter than a scenario.
- Treat backtest out/under-performance as descriptive evidence, not a trading signal.

## Methodology

- Trailing risk metrics and regimes use information available at each date.
- Scalers and models are fit on training periods only.
- Stress tests disclose historical replay vs approximation methodologies.
- Backtests shift signals to avoid same-close execution.

## Limitations

- ETF data may be incomplete, revised, or shorter than requested scenarios.
- Regime labels are descriptive risk states, not forecasts.
- Backtests can overfit research choices and ignore market impact beyond modelled costs.
- This memo is not personalised financial advice.

---
Generated deterministically from platform outputs. Not investment advice.


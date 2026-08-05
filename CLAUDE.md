# Claude Code Instructions

Project: Market Regime + Portfolio Risk Modeling Platform.

This is a risk modeling, regime detection, stress-testing, backtesting, and
research platform. It is not a stock prediction app. Do not make predictive
market claims, alpha guarantees, or investment-advice language.

## Scope

Deliver and maintain a recruiter-ready research platform with:

- ETF data ingestion and validation
- Portfolio risk metrics and concentration analytics
- Leakage-safe regime feature engineering
- Interpretable regime models (threshold, KMeans, GMM, HMM, change-points)
- Historical and deterministic stress testing
- No-look-ahead backtesting
- Thin Streamlit dashboard pages
- Reproducible quarterly memo and research reporting

## Rules

- First inspect the repo before editing.
- Do not create a second package if one already exists (`mrrp` under `src/`).
- Preserve `DatetimeIndex` in all feature and model outputs.
- Use trailing/right-edge rolling or expanding windows only.
- Never use centered rolling windows.
- Never fit scalers or models on the full dataset.
- Fit `StandardScaler` and models only on the training period.
- Enforce chronological train/validation/test separation.
- Use deterministic random seeds for stochastic models.
- No look-ahead and no same-period / same-close execution in backtests.
- Keep core logic separate from notebooks and dashboard pages.
- Thin dashboard pages: adapters and package APIs only; no duplicated finance math.
- Unit tests must be offline and must not depend on ignored local artifacts.
- Add tests for every new financial, feature, model, stress, or backtest function.
- Run tests after changes and report failures.
- Prefer configuration-driven research parameters.
- Do not add dependencies without explicit approval.
- Do not push, merge, force-push, rewrite history, or create release tags without
  explicit approval.

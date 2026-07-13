# Claude Code Instructions

Project: Market Regime + Portfolio Risk Modeling Platform.

This is a risk modeling, regime detection, backtesting, and research platform. It is not a stock prediction app.

Week 5 goal: leakage-safe regime feature engineering.

Rules:
- First inspect the repo before editing.
- Do not create a second package if one already exists.
- Preserve DatetimeIndex in all feature outputs.
- Use trailing/right-edge rolling windows only.
- Never use centered rolling windows.
- Never fit scalers on the full dataset.
- Fit StandardScaler only on the training period.
- Do not implement regime models this week.
- Do not add dashboard changes unless explicitly requested.
- Keep core logic separate from notebooks/dashboard.
- Add tests for every new feature function.
- Run tests after changes and report failures.
- Do not make predictive market claims.

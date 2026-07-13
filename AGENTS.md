# Project Instructions

## Purpose and Scope

- This repository is a market-regime, portfolio-risk, stress-testing, and
  backtesting research platform. It is not a stock prediction app.
- Keep outputs interpretable and focused on risk measurement, regime analysis,
  and portfolio behavior rather than return forecasts or trading signals.
- Week 5 focuses on leakage-safe regime feature engineering.
- Keep implementations production-shaped but simple. Do not change unrelated
  application, backend, or research files.

## Engineering Conventions

- Use Python with pandas, NumPy, and scikit-learn conventions. Use Streamlit and
  Plotly only where dashboard work already calls for them.
- Prefer pure functions with typed signatures and deterministic behavior.
- Keep data loading and validation separate from financial and risk math.
- Keep dashboard and reporting code separate from portfolio, risk, feature, and
  backtest modules.
- Drive rolling windows, thresholds, and similar research parameters through
  configuration rather than hard-coded constants.
- Preserve a `pandas.DatetimeIndex` in every time-series feature output. Do not
  silently replace, reorder, or discard source timestamps.
- Raise clear, actionable errors for invalid inputs, missing columns, bad index
  types, and insufficient history.
- Do not add dependencies unless the user explicitly approves them.

## Leakage-Safe Feature and Backtest Rules

- Build financial features from information available at or before each output
  timestamp. Never use centered rolling windows.
- Use trailing rolling or expanding windows with explicit minimum-history
  behavior. Any shift must be intentional and documented by the feature's
  information-availability assumptions.
- Split data chronologically before fitting any learned preprocessing or model.
  Never fit scalers, imputers, selectors, or models on the full dataset.
- Fit `StandardScaler` only on the training period, then use that fitted scaler
  to transform validation and test periods. Do not refit it on validation or
  test data.
- Preserve chronological order and prevent future observations from affecting
  past features, labels, thresholds, regime assignments, or portfolio weights.
- Backtests and regime analyses must avoid look-ahead bias. Tests should prove
  that changing future observations cannot change earlier feature values.

## Testing and Quality

- Add deterministic unit tests for every financial metric and every feature
  transformation.
- Cover index preservation, expected warm-up `NaN` values, boundary conditions,
  invalid inputs, and leakage-sensitive behavior where applicable.
- Keep test fixtures small and deterministic; do not make unit tests depend on
  network data.
- Run lint and tests before finishing any task. Use `make check` when practical
  to include Ruff's formatting check as well as lint and tests.

## Repository Commands

- Setup: `make setup` (runs `uv sync`).
- Build the configured market-data cache: `make data`.
- Build raw and train-scaled regime features: `make features`.
- Validate persisted regime feature artifacts: `make feature-check`.
- Run tests: `make test` (runs `uv run pytest`).
- Run lint: `make lint` (runs `uv run ruff check .`).
- Run the full validation suite: `make check` (Ruff lint, Ruff format check,
  then pytest).
- Apply formatting: `make format`.
- Remove local Python and tool caches: `make clean`.

There is currently no repository command for launching a dashboard. Streamlit
and Plotly are installed dependencies, but no tracked dashboard entry point is
present. Do not invent a dashboard command; add and document one only when the
corresponding implementation exists.

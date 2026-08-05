# Project Instructions

## Purpose and Scope

- This repository is a market-regime, portfolio-risk, stress-testing, and
  backtesting research platform. It is not a stock prediction app.
- Keep outputs interpretable and focused on risk measurement, regime analysis,
  stress scenarios, and portfolio behavior rather than return forecasts or
  trading signals.
- Do not claim reliable alpha, market prediction, guaranteed risk reduction, or
  investment performance.
- Keep implementations production-shaped but simple. Do not change unrelated
  application, backend, or research files.

## Engineering Conventions

- Use Python with pandas, NumPy, and scikit-learn conventions. Use Streamlit and
  Plotly only where dashboard work already calls for them. Use hmmlearn and
  ruptures only inside the regime-model package.
- Prefer pure functions with typed signatures and deterministic behavior.
  Set and document random seeds for every stochastic model.
- Keep data loading and validation separate from financial and risk math.
- Keep dashboard and reporting code separate from portfolio, risk, feature,
  model, stress, and backtest modules. Streamlit pages must stay thin.
- Drive rolling windows, thresholds, rebalance rules, and similar research
  parameters through configuration rather than hard-coded constants.
- Preserve a `pandas.DatetimeIndex` in every time-series feature or model
  output. Do not silently replace, reorder, or discard source timestamps.
- Raise clear, actionable errors for invalid inputs, missing columns, bad index
  types, and insufficient history.
- Do not add dependencies unless the user explicitly approves them.
- Do not create a second Python package. Extend `mrrp` under `src/`.

## Leakage-Safe Feature, Model, and Backtest Rules

- Build financial features from information available at or before each output
  timestamp. Never use centered rolling windows.
- Use trailing rolling or expanding windows with explicit minimum-history
  behavior. Any shift must be intentional and documented.
- Split data chronologically into train / validation / test before fitting any
  learned preprocessing or model. Never fit scalers, imputers, selectors, or
  models on the full dataset.
- Fit `StandardScaler` and regime models only on the training period, then use
  the fitted objects to transform or infer on validation and test periods.
- Preserve chronological order and prevent future observations from affecting
  past features, labels, thresholds, regime assignments, or portfolio weights.
- Backtests must distinguish signal, decision, execution, and return-realization
  dates. No same-close execution and no look-ahead from future model refits.
- Tests should prove that changing future observations cannot change earlier
  feature values, regime assignments, positions, or realized returns.

## Testing and Quality

- Add deterministic unit tests for every financial metric, feature
  transformation, model, stress scenario, and backtest rule.
- Cover index preservation, expected warm-up `NaN` values, boundary conditions,
  invalid inputs, constant/singular inputs, missing artifacts, and
  leakage-sensitive behavior where applicable.
- Keep test fixtures small and deterministic; do not make unit tests depend on
  network data or developer-local ignored artifacts.
- Prefer temporary synthetic fixtures and dependency-injected artifact paths for
  dashboard AppTests.
- Run lint and tests before finishing any task. Use `make check` when practical.

## Conservative Reporting

- Memos and research reports must disclose methodology, assumptions, data-as-of
  dates, and limitations.
- Do not produce buy/sell instructions or personalized financial advice.
- Generate tables and empirical statements only from actual pipeline outputs.

## Repository Commands

- Setup: `make setup` (runs `uv sync`).
- Build the configured market-data cache: `make data`.
- Launch the Streamlit dashboard: `make dashboard`.
- Build raw and train-scaled regime features: `make features`.
- Validate persisted regime feature artifacts: `make feature-check`.
- Fit and validate regime models: `make models` / `make model-check`
  (added as the model package lands).
- Run stress tests: `make stress` (added with the stress engine).
- Run backtests: `make backtest` (added with the backtest engine).
- Generate reports: `make report` (added with reporting modules).
- Run tests: `make test` (runs `uv run pytest`).
- Run lint: `make lint` (runs `uv run ruff check .`).
- Run the full validation suite: `make check` (Ruff lint, Ruff format check,
  then pytest).
- Apply formatting: `make format`.
- Remove local Python and tool caches: `make clean`.

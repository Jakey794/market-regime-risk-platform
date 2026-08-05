# Interview Explanation

The project is a risk-research platform, not a return predictor. Data is
validated once, portfolio and risk math lives in reusable package modules, and
Streamlit pages remain presentation adapters.

The central engineering constraint is information timing. Features use only
trailing observations. Scaling and regime models fit on chronological training
periods. HMM dashboard inference uses forward-filtered probabilities so future
observations cannot revise earlier displayed states. Backtest signals are
shifted before execution, and tests mutate future prices and features to prove
earlier positions and returns remain unchanged.

Model families answer different descriptive questions, so comparison metrics
are not treated as universally interchangeable. Stress results label historical
replays separately from deterministic or covariance-based estimates. Reports
insert actual artifacts when present and mark missing evidence unavailable.

Current limits include free-data quality, simplified transaction costs,
sample-dependent regime identities, no live execution, and no claim of alpha.

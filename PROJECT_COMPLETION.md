# Project Completion — v1.0

Branch: `cursor/complete-v1`  
Date: 2026-08-05

This document records acceptance evidence for the recruiter-ready v1.0 research
platform. It is not a claim of investment performance.

## Delivered capabilities

| Area | Status |
| --- | --- |
| Config-driven ETF ingestion / validation | Complete |
| Risk / return / drawdown / tail / performance metrics | Complete |
| Portfolio concentration, correlation, beta, risk contribution | Complete |
| Leakage-safe regime features + diagnostics page | Complete |
| Threshold, KMeans, GMM, HMM, change-point models | Complete |
| Historical, deterministic, and regime-conditioned stress tests | Complete |
| Shifted-signal, cost-aware backtesting | Complete |
| Quarterly memo + final research report | Complete |
| Nine-page Streamlit dashboard | Complete |
| Offline unit / AppTest suite | Complete (`make check`) |

## CI and offline fixtures

Dashboard diagnostics tests inject temporary regime-feature artifacts through
`MRRP_FEATURE_DIR` (`tests/conftest.py`). CI does not require generated
`data/processed/regime_features_*.parquet` files. A dedicated missing-artifact
AppTest asserts the diagnostics page fails closed with a clear `make features`
warning.

## Validation evidence

Run locally:

```bash
make check
```

Expected: Ruff lint + format check clean; pytest green.

Supporting research commands:

```bash
make data
make features && make feature-check
make models && make model-check
make stress
make backtest
make report
make dashboard
```

## Honest remaining limits

- Screenshots / demo recordings are optional owner assets (`reports/screenshots/`).
- PDF export of the research memo is optional and environment-dependent.
- License terms remain TBD pending owner approval.
- Deployment (for example Streamlit Cloud) requires an explicit owner action.
- Free ETF data, simplified costs, and descriptive regimes remain research limits,
  not unresolved engineering TODOs.

# Project Completion Plan — v1.0

Branch: `cursor/complete-v1`  
Baseline commit: `93d1954` (Week 5 regime features on `main`)  
Date: 2026-08-05

This document maps the repository’s Week 5 state against the v1.0 acceptance
criteria and defines the exact implementation order. It is not a claim that
Weeks 6–12 are complete.

## Current state summary

| Area | Status | Reusable assets |
| --- | --- | --- |
| Data ingestion / cache / validation | Done (Week 1) | `src/mrrp/data/*`, `scripts/download_data.py`, tracked `data/processed/adjusted_close.parquet` |
| Risk metrics engine | Done (Week 2) | `src/mrrp/risk/{returns,volatility,drawdown,performance,tail,beta,correlation,concentration,risk_contribution,summary}.py` |
| Portfolio risk engine | Done (Week 3) | `src/mrrp/portfolio/*`, `configs/sample_portfolio.yaml`, `configs/asset_metadata.yaml` |
| Streamlit dashboard (pages 1–4) | Done (Week 4) | `app/streamlit_app.py`, pages 1–4, `src/mrrp/dashboard/*`, `make dashboard` |
| Leakage-safe regime features | Done (Week 5) | `src/mrrp/features/*`, `configs/regime_features.yaml`, page 5, `make features` / `make feature-check` |
| Regime models | Missing | Empty `src/mrrp/models/` package shell |
| Stress testing | Missing | Empty risk stubs only |
| Backtesting | Missing | Empty `src/mrrp/backtest/` package shell |
| Reporting / memo | Partial | `src/mrrp/reporting/plots.py` only |
| Dashboard pages 6–9 | Missing | Navigation lists pages 1–5 only |
| Deployment / recruiter polish | Missing | No `CONTRIBUTING.md`, `SECURITY.md`, demo path, release checklist |

## Gaps, risks, and stale documentation

### CI failure (blocking)

- Workflow: `.github/workflows/ci.yml` on Python 3.14 Linux.
- Failure: `tests/test_dashboard.py::test_regime_feature_diagnostics_page_renders`
- Root cause: page 5 hardcodes paths under `data/processed/` for
  `regime_features_{raw,scaled}.parquet` and metadata. Those artifacts are
  gitignored and absent on CI, so the page hits `st.stop()` after a warning and
  never renders the sidebar radio. Locally the test passes only because
  developer-generated artifacts exist.
- Fix approach: injectable artifact paths + deterministic temporary fixtures in
  tests; explicit missing-artifact AppTest; no reliance on ignored local files.

### Stale / incorrect documentation

- `AGENTS.md` and `CLAUDE.md` still describe Week-5-only scope.
- `AGENTS.md` falsely states there is no dashboard command; `Makefile` already
  has `make dashboard`.
- `README.md` status stops at Week 4 in the scope section (Week 5 code exists).
- `README 2.md` is a redundant Day-1 stub; merge any unique phrasing into
  `README.md` and delete.
- `.cursor/rules/quant-platform.mdc` is missing and must be created for the full
  project, not Week 5 only.

### Dependency / Python version risks

| Package | Installed | Declared Requires-Python |
| --- | --- | --- |
| pandas | 3.0.3 | `>=3.11` |
| numpy | 2.4.6 | `>=3.11` |
| scikit-learn | 1.9.0 | `>=3.11` |
| streamlit | 1.58.0 | `>=3.10` |
| hmmlearn | 0.3.3 | `>=3.8` |
| ruptures | 1.1.10 | `>=3.9,<3.14` |
| pyarrow | 24.0.0 | (current lock) |

Evidence:

- Project pin is `requires-python = ">=3.14"` and CI uses 3.14.
- `ruptures` metadata officially excludes 3.14, yet uv installs it and CI
  reaches pytest on 3.14 (metadata not strictly enforced by the current
  resolver path).
- Widening will be decided only after a successful lock + test probe on 3.12
  and 3.13. Candidate target if safe: `>=3.12` with CI matrix
  `{3.12, 3.13, 3.14}`. If 3.14 cannot be made compatible with ruptures under
  a strict resolve, keep 3.14 as the documented primary runtime (current CI
  evidence) and add 3.12/3.13 only where the lock resolves cleanly.

### Repository hygiene

- Duplicate macOS-style copies under `src/mrrp/risk/* 2.py` and
  `notebooks/01_metrics_engine_demo 2.ipynb` should be removed.
- `data/processed/adjusted_close.parquet` is tracked despite
  `data/processed/*.parquet` in `.gitignore` (force-added earlier). Keep for
  offline dashboard/CI until a tracked demo dataset supersedes it; do not
  track generated feature/model/backtest artifacts.
- Empty package shells exist for `models/` and `backtest/` — extend these; do
  not create a second package.

## Existing contracts to preserve

1. Single package: `mrrp` under `src/`.
2. Preserve `DatetimeIndex` on all time-series outputs.
3. Trailing / expanding windows only; never centered.
4. Chronological train/validation/test; train-only scaler and model fitting.
5. Thin Streamlit pages; financial math stays in `mrrp.*`.
6. Deterministic synthetic unit tests; no network in unit tests.
7. Conservative research framing — not prediction or advice.

## Exact implementation order

### Phase 0 — Repair and synchronize baseline

1. Fix page-5 artifact loading (injectable paths) and dashboard tests with
   temp fixtures + missing-artifact coverage.
2. Delete redundant `* 2.py` / notebook duplicates.
3. Update `AGENTS.md`, `CLAUDE.md`, create `.cursor/rules/quant-platform.mdc`.
4. Merge/delete `README 2.md`; update `README.md` through Week 5 only.
5. Evaluate and, if safe, widen Python support + CI matrix; regenerate lock.
6. Run `make check`, `make features`, `make feature-check`.
7. Commit milestone: `fix: stabilize CI and synchronize project baseline`.

### Phase 1 — Regime model framework

Implement under `src/mrrp/models/`:

- `base.py`, `result.py`, `labeling.py`, `threshold.py`, `kmeans.py`,
  `gmm.py`, `compare.py`
- Config: `configs/regime_models.yaml`
- CLI: `python -m mrrp.models.build_models` (+ Makefile targets)
- Tests: result contract, seeds, train-only fit, alignment, probability sums,
  label mapping, invalid `n_components`, insufficient data, constant features,
  future-mutation leakage resistance

Commit: `feat: add threshold, KMeans, and GMM regime model framework`.

### Phase 2 — HMM and change-point detection

- `hmm.py` (`hmmlearn.GaussianHMM`, multi-init by train log-likelihood)
- `changepoint.py` (ruptures PELT + Binseg)
- Extend comparison framework with family-aware metrics
- Page: `app/pages/6_Regime_Detection.py`
- Notebook: `notebooks/04_regime_model_comparison.ipynb`
- Tests for HMM/changepoint determinism, train-only selection, break dates

Commit: `feat: add HMM, change-point detection, and regime dashboard`.

### Phase 3 — Stress-testing engine

- `src/mrrp/risk/stress.py`, `src/mrrp/risk/scenarios.py`
- `configs/stress_scenarios.yaml`
- CLI + `make stress`
- Page: `app/pages/7_Stress_Tests.py`
- Synthetic-portfolio unit tests; coverage disclosures for short ETF history

Commit: `feat: add historical and regime-conditioned stress tests`.

### Phase 4 — No-look-ahead backtesting

- `src/mrrp/backtest/{engine,rules,metrics,costs,walkforward}.py`
- `configs/backtest.yaml`
- CLI + `make backtest`
- Page: `app/pages/8_Backtest_Lab.py`
- Notebook: `notebooks/05_backtest_analysis.ipynb`
- Leakage tests proving future mutations cannot change earlier positions/returns

Commit: `feat: add no-look-ahead backtesting engine`.

### Phase 5 — Reporting

- `src/mrrp/reporting/{memo,research_report}.py`
- Pages/report outputs from computed artifacts only
- `app/pages/9_Quarterly_Memo.py`
- `reports/{quarterly_memo_example,final_research_report,architecture}.md`
- CLI + `make report`

Commit: `feat: add quarterly memo and research report generation`.

### Phase 6 — Dashboard product quality

- Shared loading/state for all nine pages
- Empty/missing-artifact states; data-as-of captions; no duplicated math
- AppTest coverage for every page with temp fixtures

Commit: `feat: complete nine-page dashboard with AppTest coverage`.

### Phase 7 — Deployment, CI, recruiter polish

- Demo/sample data path (synthetic, clearly labeled) for secret-free Cloud deploy
- `packages.txt` / `requirements` or uv export as needed for Streamlit Cloud
- CI: lint, format, pytest, integration smoke; optional mypy if clean
- Recruiter README rewrite; `CONTRIBUTING.md`, `SECURITY.md`, release checklist,
  demo script, resume bullets, interview explanation
- LICENSE only if ownership intent is clear; otherwise document and ask

Commit: `docs: polish v1.0 recruiter materials and deployment prep`.

## Exact tests required (by phase)

| Phase | Test modules (planned) |
| --- | --- |
| 0 | Update `tests/test_dashboard.py` (fixture + missing artifact) |
| 1 | `tests/test_regime_models.py`, `tests/test_regime_labeling.py`, `tests/test_model_compare.py` |
| 2 | `tests/test_hmm.py`, `tests/test_changepoint.py`, dashboard page-6 AppTest |
| 3 | `tests/test_stress.py`, `tests/test_scenarios.py`, page-7 AppTest |
| 4 | `tests/test_backtest_engine.py`, `tests/test_backtest_rules.py`, `tests/test_backtest_leakage.py`, page-8 AppTest |
| 5 | `tests/test_memo.py`, `tests/test_research_report.py`, page-9 AppTest |
| 6 | Expand `tests/test_dashboard.py` for all pages |
| 7 | Integration smoke for demo data path; CI workflow updates |

## Makefile / CLI targets to add

```text
make models          # fit/persist regime models
make model-check     # validate model artifacts
make stress          # run stress scenarios
make backtest        # run configured backtests
make report          # generate memo + research report
make dashboard       # already exists — document it
```

## Definition-of-done checklist

- [ ] GitHub CI green
- [x] `make check` passes (311 tests on 2026-08-05)
- [ ] `make data` works or fails with clear provider message
- [x] `make features` / `make feature-check` pass
- [x] New model / stress / backtest / report commands exist and pass
- [x] Every Streamlit page renders under AppTest
- [ ] Dashboard launches locally
- [x] No-look-ahead tests pass
- [x] No network-dependent unit tests
- [x] README / AGENTS / CLAUDE / Cursor rules agree
- [x] `README 2.md` removed if redundant
- [x] Generated full datasets ignored
- [x] Research report and example memo reproducible from pipeline outputs
- [x] Working tree clean; milestone commits; no push/tag without approval

## Non-goals / framing

- Not a stock-prediction or alpha-generation product.
- No claims of guaranteed risk reduction or investment performance.
- No external deployment or `v1.0.0` tag without explicit approval.

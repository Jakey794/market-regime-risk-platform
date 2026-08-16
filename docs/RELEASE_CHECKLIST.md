# v1.0 Release Checklist

- [x] `make check` passes on supported Python versions (CI matrix 3.12–3.14)
- [x] `make features`, `make models`, `make stress`, `make backtest`, and
  `make report` targets exist and are documented
- [x] Pages 1–9 pass Streamlit AppTest; synthetic fallback launches without
  processed prices
- [x] Future-mutation leakage tests pass
- [x] Generated price, feature, model, backtest, and stress artifacts are
  gitignored; credentials are absent from the repository
- [x] Reports distinguish computed, unavailable, replayed, and estimated values
- [x] README limitations and license-TBD statement are current
- [ ] Screenshot placeholders are replaced only with genuine current screenshots
  (owner/manual)
- [ ] Owner explicitly approves any public deployment and release tag

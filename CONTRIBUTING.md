# Contributing

Use Python 3.12+, `uv sync`, and a focused branch. Keep financial logic in
`src/mrrp`, preserve `DatetimeIndex`, use trailing/expanding windows, and fit
learned transformations on training data only. Add deterministic offline tests
for new behavior.

Before requesting review:

```bash
make check
```

Do not commit provider credentials, generated processed artifacts, or claims of
predictive performance. Explain information timing for every new feature,
model, or backtest rule.

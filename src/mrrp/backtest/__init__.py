"""No-look-ahead portfolio backtesting."""

from mrrp.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from mrrp.backtest.metrics import BacktestMetrics, compute_backtest_metrics
from mrrp.backtest.rules import (
    RuleConfig,
    compute_signal_scalar,
    target_weights_from_scalar,
)

__all__ = [
    "BacktestConfig",
    "BacktestMetrics",
    "BacktestResult",
    "RuleConfig",
    "compute_backtest_metrics",
    "compute_signal_scalar",
    "run_backtest",
    "target_weights_from_scalar",
]

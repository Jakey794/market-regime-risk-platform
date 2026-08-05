"""Run configured no-look-ahead backtests and persist summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mrrp.backtest.engine import BacktestConfig, run_backtest
from mrrp.backtest.rules import RuleConfig
from mrrp.data.cache import load_parquet
from mrrp.portfolio.config import load_portfolio_config
from mrrp.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", default="data/processed/adjusted_close.parquet")
    parser.add_argument(
        "--features", default="data/processed/regime_features_raw.parquet"
    )
    parser.add_argument("--portfolio", default="configs/sample_portfolio.yaml")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--output-dir", default="data/processed/backtest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = load_parquet(args.prices)
    features = load_parquet(args.features)
    portfolio = load_portfolio_config(args.portfolio)
    raw = load_yaml(args.config)
    defensive = pd.Series(raw["defensive_allocation"], dtype=float)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    for rule_name in raw["rules"]:
        rule = RuleConfig(
            name=rule_name,
            base_weights=portfolio.holdings,
            defensive_weights=defensive,
            vol_target=float(raw["vol_target"]),
            high_vol_threshold=float(raw["high_vol_threshold"]),
            high_corr_threshold=float(raw["high_corr_threshold"]),
            drawdown_limit=float(raw["drawdown_limit"]),
            min_risk_scalar=float(raw["min_risk_scalar"]),
        )
        config = BacktestConfig(
            rule=rule,
            rebalance_frequency=raw["rebalance_frequency"],
            transaction_cost_bps=float(raw["transaction_cost_bps"]),
            signal_shift=int(raw["signal_shift"]),
            periods_per_year=int(raw["periods_per_year"]),
        )
        result = run_backtest(
            prices,
            features,
            portfolio.benchmark,
            config,
            config_name=rule_name,
        )
        result.strategy_returns.to_csv(output / f"{rule_name}_returns.csv", header=True)
        result.weights.to_csv(output / f"{rule_name}_weights.csv")
        summaries[rule_name] = {
            "metrics": result.metrics.to_dict(),
            "warnings": list(result.warnings),
        }
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(summaries)} backtests to {output}")


if __name__ == "__main__":
    main()

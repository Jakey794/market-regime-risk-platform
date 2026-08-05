"""Run configured portfolio stress scenarios and persist a JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrrp.data.cache import load_parquet
from mrrp.portfolio.config import load_portfolio_config
from mrrp.risk.scenarios import load_stress_scenario_config
from mrrp.risk.stress import (
    deterministic_asset_shock,
    historical_window_stress,
    worst_rolling_stress,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", default="data/processed/adjusted_close.parquet")
    parser.add_argument("--portfolio", default="configs/sample_portfolio.yaml")
    parser.add_argument("--config", default="configs/stress_scenarios.yaml")
    parser.add_argument("--output", default="data/processed/stress_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = load_parquet(args.prices)
    portfolio = load_portfolio_config(args.portfolio)
    config = load_stress_scenario_config(args.config)
    weights = portfolio.holdings
    results = [
        *(
            historical_window_stress(prices, weights, item)
            for item in config.historical_windows
        ),
        *(
            worst_rolling_stress(prices, weights, window)
            for window in config.worst_rolling_windows
        ),
        *(deterministic_asset_shock(weights, item) for item in config.deterministic),
    ]
    payload = {
        "portfolio": portfolio.name,
        "data_as_of": str(prices.index.max().date()),
        "results": [
            {
                "name": result.name,
                "scenario_type": result.scenario_type,
                "methodology": result.methodology,
                "portfolio_impact": result.portfolio_impact,
                "asset_contribution": result.asset_contribution.to_dict(),
                "assumptions": result.assumptions,
                "warnings": list(result.warnings),
            }
            for result in results
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(results)} stress results to {output}")


if __name__ == "__main__":
    main()

"""Run configured portfolio stress scenarios and persist a JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrrp.data.cache import load_parquet
from mrrp.portfolio.config import load_portfolio_config
from mrrp.portfolio.returns import compute_asset_returns
from mrrp.risk.scenarios import load_stress_scenario_config
from mrrp.risk.stress import (
    benchmark_beta_shock,
    correlation_shock_estimate,
    deterministic_asset_shock,
    historical_window_stress,
    rank_stress_results,
    volatility_shock_estimate,
    worst_rolling_stress,
)
from mrrp.utils.config import load_yaml


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
    raw_config = load_yaml(args.config)
    weights = portfolio.holdings
    asset_prices = prices.loc[:, weights.index.intersection(prices.columns)]
    asset_returns = compute_asset_returns(asset_prices, method="simple").dropna(
        how="any"
    )
    benchmark = portfolio.benchmark
    benchmark_returns = (
        compute_asset_returns(prices.loc[:, [benchmark]], method="simple")[benchmark]
        .reindex(asset_returns.index)
        .dropna()
    )
    aligned_assets = asset_returns.reindex(benchmark_returns.index).dropna(how="any")
    aligned_benchmark = benchmark_returns.reindex(aligned_assets.index)

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
        benchmark_beta_shock(
            aligned_assets,
            aligned_benchmark,
            weights,
            float(raw_config.get("benchmark_shock", -0.10)),
        ),
        volatility_shock_estimate(
            aligned_assets,
            weights,
            float(raw_config.get("volatility_multiplier", 1.5)),
        ),
        correlation_shock_estimate(
            aligned_assets,
            weights,
            float(raw_config.get("target_correlation", 0.90)),
        ),
    ]
    ranking = rank_stress_results(results)
    payload = {
        "portfolio": portfolio.name,
        "data_as_of": str(prices.index.max().date()),
        "ranking": ranking.to_dict(orient="records"),
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

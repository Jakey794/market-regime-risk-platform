"""Portfolio stress-testing engine with explicit methodology tags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from mrrp.portfolio.returns import compute_asset_returns, compute_portfolio_returns
from mrrp.risk.beta import compute_asset_betas
from mrrp.risk.scenarios import (
    DeterministicShockScenario,
    HistoricalWindowScenario,
    window_coverage,
)
from mrrp.risk.tail import historical_var

Methodology = Literal[
    "historical_replay",
    "direct_weighted_return_shock",
    "covariance_volatility_estimate",
    "beta_approximation",
]


@dataclass(frozen=True)
class StressScenarioResult:
    """Typed stress-test output."""

    name: str
    scenario_type: str
    methodology: Methodology
    portfolio_impact: float
    asset_contribution: pd.Series
    assumptions: dict[str, Any]
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


def historical_window_stress(
    prices: pd.DataFrame,
    weights: pd.Series,
    scenario: HistoricalWindowScenario,
) -> StressScenarioResult:
    """Replay exact observed returns inside a historical window."""
    start, end, covered, warning = window_coverage(prices, scenario.start, scenario.end)
    warnings = [warning] if warning else []
    if start > end:
        return StressScenarioResult(
            name=scenario.name,
            scenario_type="historical_window",
            methodology="historical_replay",
            portfolio_impact=float("nan"),
            asset_contribution=pd.Series(dtype=float),
            assumptions={
                "requested_start": scenario.start,
                "requested_end": scenario.end,
                "description": scenario.description,
            },
            warnings=tuple(warnings),
        )

    window_prices = prices.loc[start:end]
    asset_returns = compute_asset_returns(window_prices, method="simple").dropna(
        how="any"
    )
    if asset_returns.empty:
        warnings.append("No complete return observations in the historical window")
        impact = float("nan")
        contrib = pd.Series(0.0, index=weights.index)
    else:
        aligned_weights = weights.reindex(asset_returns.columns).fillna(0.0)
        portfolio = compute_portfolio_returns(asset_returns, aligned_weights)
        impact = float((1.0 + portfolio).prod() - 1.0)
        asset_cum = (1.0 + asset_returns).prod() - 1.0
        contrib = (asset_cum * aligned_weights).rename("contribution")

    return StressScenarioResult(
        name=scenario.name,
        scenario_type="historical_window",
        methodology="historical_replay",
        portfolio_impact=impact,
        asset_contribution=contrib,
        assumptions={
            "requested_start": scenario.start,
            "requested_end": scenario.end,
            "effective_start": str(start.date()),
            "effective_end": str(end.date()),
            "fully_covered": covered,
            "description": scenario.description,
        },
        warnings=tuple(warnings),
        diagnostics={
            "n_returns": int(len(asset_returns)) if "asset_returns" in locals() else 0
        },
    )


def worst_rolling_stress(
    prices: pd.DataFrame,
    weights: pd.Series,
    window: int,
) -> StressScenarioResult:
    """Find the worst historical cumulative return over a trailing window."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    asset_returns = compute_asset_returns(prices, method="simple").dropna(how="any")
    aligned_weights = weights.reindex(asset_returns.columns).fillna(0.0)
    portfolio = compute_portfolio_returns(asset_returns, aligned_weights)
    rolling = (
        (1.0 + portfolio)
        .rolling(window)
        .apply(lambda x: float(np.prod(x) - 1.0), raw=True)
    )
    if rolling.dropna().empty:
        raise ValueError(f"Insufficient history for worst rolling window={window}")
    end_date = rolling.idxmin()
    impact = float(rolling.loc[end_date])
    start_loc = portfolio.index.get_loc(end_date) - window + 1
    start_date = portfolio.index[start_loc]
    window_returns = asset_returns.loc[start_date:end_date]
    asset_cum = (1.0 + window_returns).prod() - 1.0
    contrib = (asset_cum * aligned_weights).rename("contribution")
    return StressScenarioResult(
        name=f"worst_rolling_{window}d",
        scenario_type="worst_rolling",
        methodology="historical_replay",
        portfolio_impact=impact,
        asset_contribution=contrib,
        assumptions={
            "window": window,
            "start": str(start_date.date()),
            "end": str(end_date.date()),
        },
        warnings=(),
    )


def deterministic_asset_shock(
    weights: pd.Series,
    scenario: DeterministicShockScenario,
) -> StressScenarioResult:
    """Apply direct weighted-return shocks to named assets."""
    shocks = pd.Series(scenario.shocks, dtype=float)
    aligned_weights = weights.reindex(weights.index).astype(float)
    contrib = pd.Series(0.0, index=aligned_weights.index, name="contribution")
    warnings: list[str] = []
    for asset, shock in shocks.items():
        if asset not in contrib.index:
            warnings.append(f"Shock asset {asset} not in portfolio weights; ignored")
            continue
        contrib.loc[asset] = float(aligned_weights.loc[asset]) * float(shock)
    impact = float(contrib.sum())
    return StressScenarioResult(
        name=scenario.name,
        scenario_type=scenario.scenario_type,
        methodology="direct_weighted_return_shock",
        portfolio_impact=impact,
        asset_contribution=contrib,
        assumptions={
            "shocks": dict(scenario.shocks),
            "description": scenario.description,
            **scenario.metadata,
        },
        warnings=tuple(warnings),
    )


def benchmark_beta_shock(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    weights: pd.Series,
    benchmark_shock: float,
    *,
    name: str = "benchmark_beta_shock",
) -> StressScenarioResult:
    """Translate a benchmark shock through estimated betas (approximation)."""
    betas = compute_asset_betas(asset_returns, benchmark_returns)
    aligned_weights = weights.reindex(betas.index).fillna(0.0)
    asset_shocks = betas * float(benchmark_shock)
    contrib = (asset_shocks * aligned_weights).rename("contribution")
    impact = float(contrib.sum())
    return StressScenarioResult(
        name=name,
        scenario_type="benchmark_beta_shock",
        methodology="beta_approximation",
        portfolio_impact=impact,
        asset_contribution=contrib,
        assumptions={
            "benchmark_shock": benchmark_shock,
            "beta_estimation": "OLS asset beta vs benchmark over provided sample",
            "limitation": (
                "Beta approximation assumes linear factor response and ignores "
                "residual idiosyncratic shocks."
            ),
        },
        warnings=(
            "Beta-translated shocks are approximations, not historical replays.",
        ),
        diagnostics={"betas": betas.to_dict()},
    )


def volatility_shock_estimate(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    vol_multiplier: float,
    *,
    name: str = "volatility_shock",
    z_score: float = -2.0,
) -> StressScenarioResult:
    """Estimate a one-period shock from scaled covariance (not a replay)."""
    if vol_multiplier <= 0:
        raise ValueError("vol_multiplier must be positive")
    aligned = asset_returns.dropna(how="any")
    w = weights.reindex(aligned.columns).fillna(0.0).to_numpy(dtype=float)
    cov = aligned.cov().to_numpy(dtype=float) * float(vol_multiplier) ** 2
    port_var = float(w.T @ cov @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    impact = float(z_score * port_vol)
    # Approximate asset contributions via marginal variance share * impact.
    if port_var <= 0:
        contrib = pd.Series(0.0, index=aligned.columns, name="contribution")
    else:
        mrc = cov @ w
        pct = (w * mrc) / port_var
        contrib = pd.Series(pct * impact, index=aligned.columns, name="contribution")
    return StressScenarioResult(
        name=name,
        scenario_type="volatility_shock",
        methodology="covariance_volatility_estimate",
        portfolio_impact=impact,
        asset_contribution=contrib,
        assumptions={
            "vol_multiplier": vol_multiplier,
            "z_score": z_score,
            "limitation": (
                "Uses covariance-scaled Gaussian z-score approximation; "
                "not an observed historical path."
            ),
        },
        warnings=(
            "Volatility shock is a covariance-based estimate, not historical replay.",
        ),
        diagnostics={"portfolio_volatility": port_vol},
    )


def regime_conditioned_stress(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    regimes: pd.Series,
    *,
    regime_label: str,
) -> list[StressScenarioResult]:
    """Compute worst day/week/month and tail stats inside one regime label."""
    aligned = pd.concat(
        [
            portfolio_returns.rename("portfolio"),
            regimes.rename("regime"),
            asset_returns,
        ],
        axis=1,
    ).dropna()
    subset = aligned.loc[aligned["regime"] == regime_label]
    warnings: list[str] = []
    if subset.empty:
        warnings.append(f"No observations for regime '{regime_label}'")
        empty = StressScenarioResult(
            name=f"regime_{regime_label}_empty",
            scenario_type="regime_conditioned",
            methodology="historical_replay",
            portfolio_impact=float("nan"),
            asset_contribution=pd.Series(dtype=float),
            assumptions={"regime_label": regime_label},
            warnings=tuple(warnings),
        )
        return [empty]

    port = subset["portfolio"]
    results: list[StressScenarioResult] = []
    for horizon, window in (("day", 1), ("week", 5), ("month", 21)):
        if window == 1:
            series = port
        else:
            series = (
                (1.0 + port)
                .rolling(window)
                .apply(lambda x: float(np.prod(x) - 1.0), raw=True)
            )
        clean = series.dropna()
        if clean.empty:
            warnings.append(f"Insufficient data for regime {regime_label} {horizon}")
            continue
        end = clean.idxmin()
        impact = float(clean.loc[end])
        if window == 1:
            start = end
            window_assets = asset_returns.loc[[end]]
        else:
            loc = port.index.get_loc(end)
            start = port.index[loc - window + 1]
            window_assets = asset_returns.loc[start:end]
        asset_cum = (
            1.0 + window_assets.reindex(columns=weights.index).fillna(0.0)
        ).prod() - 1.0
        contrib = (asset_cum * weights).rename("contribution")
        results.append(
            StressScenarioResult(
                name=f"regime_{regime_label}_worst_{horizon}",
                scenario_type="regime_conditioned",
                methodology="historical_replay",
                portfolio_impact=impact,
                asset_contribution=contrib,
                assumptions={
                    "regime_label": regime_label,
                    "horizon": horizon,
                    "start": str(pd.Timestamp(start).date()),
                    "end": str(pd.Timestamp(end).date()),
                },
                warnings=(),
            )
        )

    # Average drawdown and historical VaR inside regime.
    wealth = (1.0 + port).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    avg_dd = float(drawdown.mean())
    try:
        var_5 = float(historical_var(port, confidence=0.95))
    except Exception:  # noqa: BLE001
        var_5 = float("nan")
        warnings.append("Unable to compute regime historical VaR")
    results.append(
        StressScenarioResult(
            name=f"regime_{regime_label}_tail_summary",
            scenario_type="regime_conditioned",
            methodology="historical_replay",
            portfolio_impact=var_5,
            asset_contribution=pd.Series(dtype=float),
            assumptions={
                "regime_label": regime_label,
                "average_drawdown": avg_dd,
                "historical_var_5pct": var_5,
                "n_obs": int(len(port)),
            },
            warnings=tuple(warnings),
            diagnostics={"average_drawdown": avg_dd},
        )
    )
    return results

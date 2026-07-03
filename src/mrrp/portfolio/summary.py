"""Dashboard-independent portfolio risk summary facade."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from mrrp.portfolio.config import PortfolioConfig
from mrrp.portfolio.exposures import compute_weight_exposure
from mrrp.portfolio.returns import compute_asset_returns, compute_portfolio_returns
from mrrp.portfolio.weights import validate_weights
from mrrp.risk.beta import compute_portfolio_beta
from mrrp.risk.concentration import (
    classify_concentration_risk,
    compute_effective_num_holdings,
    compute_top_n_weight,
)
from mrrp.risk.correlation import (
    build_correlation_summary,
    compute_correlation_matrix,
)
from mrrp.risk.drawdown import current_drawdown, max_drawdown
from mrrp.risk.performance import sharpe_ratio, sortino_ratio
from mrrp.risk.risk_contribution import (
    build_risk_contribution_table as _build_risk_contribution_table,
)
from mrrp.risk.tail import historical_cvar, historical_var
from mrrp.risk.volatility import annualized_return, annualized_volatility


PERIODS_PER_YEAR = 252
CORRELATION_WINDOW = 63
METADATA_FIELDS = ("asset_class", "region", "style", "sector_proxy")


@dataclass(frozen=True)
class PortfolioRiskSummary:
    """Typed scalar summary of portfolio performance and risk."""

    portfolio_name: str
    benchmark: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    current_drawdown: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    portfolio_beta: float
    concentration_label: str
    correlation_regime: str
    effective_holdings: float
    top_3_weight: float
    mean_pairwise_corr: float
    diversification_ratio: float


def build_portfolio_risk_summary(
    prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    benchmark_prices: pd.Series | None = None,
    asset_metadata: dict | None = None,
) -> PortfolioRiskSummary:
    """Build a complete typed portfolio risk summary from prices and config."""
    asset_returns = _asset_returns(prices, portfolio_config)
    complete_returns = asset_returns.dropna()
    if len(complete_returns) < CORRELATION_WINDOW:
        raise ValueError(
            f"Prices must provide at least {CORRELATION_WINDOW} complete returns"
        )

    if asset_metadata is not None:
        build_exposure_table(portfolio_config, asset_metadata)

    portfolio_returns = compute_portfolio_returns(
        asset_returns,
        portfolio_config.holdings,
    )
    benchmark_returns = _benchmark_returns(
        prices,
        portfolio_config,
        benchmark_prices,
    )
    correlation = build_correlation_summary(
        asset_returns,
        portfolio_config.holdings,
        window=CORRELATION_WINDOW,
    ).iloc[0]

    return PortfolioRiskSummary(
        portfolio_name=portfolio_config.name,
        benchmark=portfolio_config.benchmark,
        start_date=pd.Timestamp(prices.index[0]),
        end_date=pd.Timestamp(prices.index[-1]),
        annualized_return=annualized_return(
            portfolio_returns,
            PERIODS_PER_YEAR,
        ),
        annualized_volatility=annualized_volatility(
            portfolio_returns,
            PERIODS_PER_YEAR,
        ),
        sharpe=sharpe_ratio(
            portfolio_returns,
            periods_per_year=PERIODS_PER_YEAR,
        ),
        sortino=sortino_ratio(
            portfolio_returns,
            periods_per_year=PERIODS_PER_YEAR,
        ),
        current_drawdown=current_drawdown(portfolio_returns),
        max_drawdown=max_drawdown(portfolio_returns),
        var_95=historical_var(portfolio_returns, confidence=0.95),
        cvar_95=historical_cvar(portfolio_returns, confidence=0.95),
        portfolio_beta=compute_portfolio_beta(
            portfolio_returns,
            benchmark_returns,
        ),
        concentration_label=classify_concentration_risk(portfolio_config.holdings),
        correlation_regime=str(correlation["correlation_regime"]),
        effective_holdings=compute_effective_num_holdings(portfolio_config.holdings),
        top_3_weight=compute_top_n_weight(portfolio_config.holdings, n=3),
        mean_pairwise_corr=float(correlation["mean_pairwise_corr"]),
        diversification_ratio=float(correlation["diversification_ratio"]),
    )


def build_exposure_table(
    portfolio_config: PortfolioConfig,
    asset_metadata: dict | None = None,
) -> pd.DataFrame:
    """Build a stable holding exposure table with optional asset metadata."""
    _validate_portfolio_config(portfolio_config)
    if asset_metadata is not None and not isinstance(asset_metadata, dict):
        raise ValueError("asset_metadata must be a dictionary")

    exposure = (
        compute_weight_exposure(portfolio_config.holdings)
        .rename_axis("ticker")
        .reset_index()
    )
    metadata = asset_metadata or {}
    for field in METADATA_FIELDS:
        exposure[field] = [
            _metadata_value(metadata, str(ticker), field)
            for ticker in exposure["ticker"]
        ]
    return exposure


def build_risk_contribution_table(
    prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    benchmark_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a holding risk-contribution table directly from prices and config."""
    asset_returns = _asset_returns(prices, portfolio_config)
    benchmark_returns = _benchmark_returns(
        prices,
        portfolio_config,
        benchmark_prices,
    )
    return _build_risk_contribution_table(
        asset_returns,
        portfolio_config.holdings,
        benchmark_returns,
    )


def build_correlation_table(
    prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> pd.DataFrame:
    """Build an asset correlation matrix in configured holding order."""
    return compute_correlation_matrix(_asset_returns(prices, portfolio_config))


def build_summary_cards(summary: PortfolioRiskSummary) -> dict[str, object]:
    """Convert a typed summary into a plain dashboard-ready dictionary."""
    if not isinstance(summary, PortfolioRiskSummary):
        raise ValueError("summary must be a PortfolioRiskSummary")
    return asdict(summary)


def _asset_returns(
    prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
) -> pd.DataFrame:
    _validate_prices(prices)
    _validate_portfolio_config(portfolio_config)
    missing_tickers = [
        ticker
        for ticker in portfolio_config.holdings.index
        if ticker not in prices.columns
    ]
    if missing_tickers:
        raise ValueError(f"Prices are missing portfolio tickers: {missing_tickers}")

    returns = compute_asset_returns(
        prices.loc[:, portfolio_config.holdings.index.tolist()],
        method="simple",
    )
    if len(returns.dropna()) < 2:
        raise ValueError("Prices must provide at least 2 complete returns")
    return returns


def _benchmark_returns(
    prices: pd.DataFrame,
    portfolio_config: PortfolioConfig,
    benchmark_prices: pd.Series | None,
) -> pd.Series:
    if benchmark_prices is None:
        if portfolio_config.benchmark not in prices.columns:
            raise ValueError(
                f"Benchmark prices are missing: {portfolio_config.benchmark}"
            )
        benchmark_prices = prices[portfolio_config.benchmark]
    elif not isinstance(benchmark_prices, pd.Series):
        raise ValueError("benchmark_prices must be a pandas Series")

    benchmark_frame = benchmark_prices.rename(portfolio_config.benchmark).to_frame()
    _validate_prices(benchmark_frame)
    returns = compute_asset_returns(benchmark_frame, method="simple")
    if len(returns.dropna()) < 2:
        raise ValueError("Benchmark prices must provide at least 2 valid returns")
    return returns.iloc[:, 0]


def _validate_prices(prices: pd.DataFrame) -> None:
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Prices must be a pandas DataFrame")
    if prices.empty:
        raise ValueError("Prices must not be empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices index must be a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("Prices index must be unique")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Prices index must be ordered from oldest to newest")
    if prices.columns.has_duplicates:
        raise ValueError("Price columns must be unique")


def _validate_portfolio_config(portfolio_config: PortfolioConfig) -> None:
    if not isinstance(portfolio_config, PortfolioConfig):
        raise ValueError("portfolio_config must be a PortfolioConfig")
    if len(portfolio_config.holdings) < 2:
        raise ValueError("Portfolio must contain at least 2 holdings")
    validate_weights(
        portfolio_config.holdings,
        allow_short=portfolio_config.allow_short,
    )


def _metadata_value(metadata: dict, ticker: str, field: str) -> str:
    attributes = metadata.get(ticker.upper())
    if not isinstance(attributes, dict):
        return "Unknown"
    value = attributes.get(field)
    if not isinstance(value, str) or not value.strip():
        return "Unknown"
    return value.strip()

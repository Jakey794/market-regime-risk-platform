"""Offline change-point detection with ruptures (PELT and binary segmentation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import ruptures as rpt

from mrrp.models.base import validate_feature_matrix
from mrrp.models.result import RegimeModelResult, validate_regime_model_result

MODEL_VERSION = "1.0.0"
MethodName = Literal["pelt", "binseg"]
SignalName = Literal["volatility", "correlation", "multivariate"]


@dataclass(frozen=True)
class ChangePointConfig:
    """Configuration for change-point detection."""

    method: MethodName = "pelt"
    signal: SignalName = "volatility"
    feature_columns: tuple[str, ...] | None = None
    vol_column: str = "portfolio_vol_63d"
    corr_column: str = "mean_corr_63d"
    penalty: float = 10.0
    n_bkps: int | None = None
    model: str = "l2"
    min_size: int = 5


@dataclass(frozen=True)
class ChangePointResult:
    """Typed change-point detection output."""

    method: str
    signal: str
    break_dates: tuple[pd.Timestamp, ...]
    break_indices: tuple[int, ...]
    feature_columns: tuple[str, ...]
    diagnostics: dict[str, Any]
    warnings: tuple[str, ...]

    def to_regime_result(self, features: pd.DataFrame) -> RegimeModelResult:
        """Convert breakpoints into contiguous integer segment labels."""
        validated = validate_feature_matrix(
            features, feature_columns=self.feature_columns
        )
        states = np.zeros(len(validated), dtype=int)
        segment = 0
        break_set = set(self.break_indices)
        for idx in range(len(validated)):
            states[idx] = segment
            # ruptures indices are exclusive end-of-segment positions.
            if (idx + 1) in break_set and (idx + 1) < len(validated):
                segment += 1
        n_states = int(states.max()) + 1 if len(states) else 1
        labels = tuple(f"segment_{idx}" for idx in range(n_states))
        summary = (
            pd.DataFrame({"state": states}, index=validated.index)
            .groupby("state")
            .size()
            .rename("n_obs")
            .to_frame()
        )
        result = RegimeModelResult(
            model_name="changepoint",
            model_version=MODEL_VERSION,
            fitted_parameters={
                "method": self.method,
                "signal": self.signal,
                "break_dates": [ts.isoformat() for ts in self.break_dates],
                "break_indices": list(self.break_indices),
            },
            feature_columns=self.feature_columns,
            fit_start=pd.Timestamp(validated.index.min()),
            fit_end=pd.Timestamp(validated.index.max()),
            dates=validated.index,
            states=states,
            economic_labels=labels,
            state_probabilities=None,
            state_summary=summary,
            diagnostics={
                **self.diagnostics,
                "n_states": n_states,
                "n_breakpoints": len(self.break_dates),
            },
            warnings=self.warnings,
            random_seed=None,
        )
        validate_regime_model_result(result)
        return result


class ChangePointDetector:
    """Detect sorted, valid break dates on configured signals."""

    model_name = "changepoint"
    model_version = MODEL_VERSION

    def __init__(self, config: ChangePointConfig | None = None) -> None:
        self.config = config or ChangePointConfig()

    def detect(self, features: pd.DataFrame) -> ChangePointResult:
        """Run PELT or Binseg and return sorted break dates inside the sample."""
        cfg = self.config
        if cfg.method not in {"pelt", "binseg"}:
            raise ValueError("method must be 'pelt' or 'binseg'")
        if cfg.penalty is not None and cfg.penalty <= 0:
            raise ValueError("penalty must be positive")
        if cfg.n_bkps is not None and (
            isinstance(cfg.n_bkps, bool)
            or not isinstance(cfg.n_bkps, int)
            or cfg.n_bkps < 1
        ):
            raise ValueError("n_bkps must be a positive integer when provided")

        columns = self._resolve_columns(features)
        data = validate_feature_matrix(
            features,
            feature_columns=columns,
            min_observations=max(cfg.min_size * 2, 20),
        )
        signal = data.to_numpy(dtype=float)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        warnings: list[str] = []
        if cfg.method == "pelt":
            algo = rpt.Pelt(model=cfg.model, min_size=cfg.min_size).fit(signal)
            # PELT uses penalty; n_bkps is ignored except as a diagnostic note.
            breakpoints = algo.predict(pen=float(cfg.penalty))
            if cfg.n_bkps is not None:
                warnings.append(
                    "n_bkps is ignored for PELT; penalty controls the number of breaks"
                )
        else:
            algo = rpt.Binseg(model=cfg.model, min_size=cfg.min_size).fit(signal)
            if cfg.n_bkps is not None:
                breakpoints = algo.predict(n_bkps=int(cfg.n_bkps))
            else:
                breakpoints = algo.predict(pen=float(cfg.penalty))

        # ruptures always includes the final index (= n_samples); exclude it.
        n_samples = len(data)
        internal = sorted(
            idx
            for idx in breakpoints
            if isinstance(idx, (int, np.integer)) and 0 < int(idx) < n_samples
        )
        break_indices = tuple(int(idx) for idx in internal)
        break_dates = tuple(pd.Timestamp(data.index[idx]) for idx in break_indices)

        return ChangePointResult(
            method=cfg.method,
            signal=cfg.signal,
            break_dates=break_dates,
            break_indices=break_indices,
            feature_columns=columns,
            diagnostics={
                "penalty": cfg.penalty,
                "n_bkps_requested": cfg.n_bkps,
                "cost_model": cfg.model,
                "n_samples": n_samples,
                "raw_breakpoints": [int(x) for x in breakpoints],
            },
            warnings=tuple(warnings),
        )

    def _resolve_columns(self, features: pd.DataFrame) -> tuple[str, ...]:
        cfg = self.config
        if cfg.feature_columns is not None:
            return tuple(cfg.feature_columns)
        if cfg.signal == "volatility":
            return (cfg.vol_column,)
        if cfg.signal == "correlation":
            return (cfg.corr_column,)
        # multivariate default: prefer both vol and corr when present
        columns = []
        for column in (cfg.vol_column, cfg.corr_column):
            if column in features.columns:
                columns.append(column)
        if not columns:
            columns = [
                str(col) for col in features.columns[: min(4, features.shape[1])]
            ]
        return tuple(columns)

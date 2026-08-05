"""Interpretable percentile-threshold regime baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mrrp.models.base import (
    RegimeModel,
    reject_constant_features,
    validate_feature_matrix,
)
from mrrp.models.labeling import (
    DEFAULT_CORR_COLUMN,
    DEFAULT_VOL_COLUMN,
    apply_label_map,
    build_state_summary,
    map_states_to_economic_labels,
    ordered_economic_labels,
)
from mrrp.models.result import RegimeModelResult, validate_regime_model_result

MODEL_VERSION = "1.0.0"


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for the interpretable threshold baseline."""

    vol_column: str = DEFAULT_VOL_COLUMN
    corr_column: str = DEFAULT_CORR_COLUMN
    high_vol_percentile: float = 0.75
    high_corr_percentile: float = 0.75
    n_states: int = 4


class ThresholdRegimeModel(RegimeModel):
    """Assign regimes from training-period volatility/correlation percentiles.

    States (raw integers before economic mapping):

    0. low vol, low corr
    1. high vol, low corr
    2. low vol, high corr
    3. high vol, high corr

    When ``n_states`` is 2 or 3, states are collapsed after the 2x2 grid is
    formed so comparisons remain interpretable.
    """

    model_name = "threshold"
    model_version = MODEL_VERSION

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self.config = config or ThresholdConfig()
        self._fitted = False
        self._vol_threshold = float("nan")
        self._corr_threshold = float("nan")
        self._feature_columns: tuple[str, ...] = ()
        self._label_map: dict[int, str] = {}
        self._fit_start: pd.Timestamp | None = None
        self._fit_end: pd.Timestamp | None = None
        self._train_summary = pd.DataFrame()

    def fit(self, train_features: pd.DataFrame) -> RegimeModelResult:
        """Fit percentile thresholds on the training period only."""
        cfg = self.config
        if not 0.0 < cfg.high_vol_percentile < 1.0:
            raise ValueError("high_vol_percentile must be in (0, 1)")
        if not 0.0 < cfg.high_corr_percentile < 1.0:
            raise ValueError("high_corr_percentile must be in (0, 1)")
        if cfg.n_states not in {2, 3, 4}:
            raise ValueError("n_states must be 2, 3, or 4")

        columns = (cfg.vol_column, cfg.corr_column)
        train = validate_feature_matrix(
            train_features,
            feature_columns=columns,
            min_observations=max(20, cfg.n_states * 5),
        )
        reject_constant_features(train)

        self._vol_threshold = float(
            train[cfg.vol_column].quantile(cfg.high_vol_percentile)
        )
        self._corr_threshold = float(
            train[cfg.corr_column].quantile(cfg.high_corr_percentile)
        )
        raw_states = self._assign_raw_states(train)
        collapsed = self._collapse_states(raw_states, cfg.n_states)
        summary = build_state_summary(train, collapsed)
        self._label_map = {state: "elevated_risk" for state in range(cfg.n_states)}
        self._label_map[0] = "calm"
        if cfg.n_states >= 3:
            self._label_map[cfg.n_states - 1] = "high_vol_high_corr_risk_off"
        if len(summary) > 1:
            self._label_map.update(map_states_to_economic_labels(summary))
        self._feature_columns = columns
        self._fit_start = pd.Timestamp(train.index.min())
        self._fit_end = pd.Timestamp(train.index.max())
        self._train_summary = summary
        self._fitted = True
        return self._build_result(
            train,
            collapsed,
            probabilities=None,
            diagnostics={
                "vol_threshold": self._vol_threshold,
                "corr_threshold": self._corr_threshold,
                "method": "train_percentile_thresholds",
            },
        )

    def transform(self, features: pd.DataFrame) -> RegimeModelResult:
        """Apply train-fitted thresholds to a feature matrix."""
        self._require_fitted()
        data = validate_feature_matrix(features, feature_columns=self._feature_columns)
        raw_states = self._assign_raw_states(data)
        collapsed = self._collapse_states(raw_states, self.config.n_states)
        return self._build_result(
            data,
            collapsed,
            probabilities=None,
            diagnostics={
                "vol_threshold": self._vol_threshold,
                "corr_threshold": self._corr_threshold,
                "method": "train_percentile_thresholds",
            },
        )

    def _assign_raw_states(self, features: pd.DataFrame) -> np.ndarray:
        high_vol = (
            features[self.config.vol_column].to_numpy(dtype=float)
            >= self._vol_threshold
        )
        high_corr = (
            features[self.config.corr_column].to_numpy(dtype=float)
            >= self._corr_threshold
        )
        return (high_vol.astype(int) + 2 * high_corr.astype(int)).astype(int)

    @staticmethod
    def _collapse_states(raw_states: np.ndarray, n_states: int) -> np.ndarray:
        if n_states == 4:
            return raw_states.astype(int)
        if n_states == 2:
            # 0 = calm-ish (raw 0), 1 = elevated (raw 1/2/3)
            return (raw_states > 0).astype(int)
        # n_states == 3: calm (0), elevated (1 or 2), risk-off (3)
        collapsed = np.zeros_like(raw_states, dtype=int)
        collapsed[raw_states == 0] = 0
        collapsed[(raw_states == 1) | (raw_states == 2)] = 1
        collapsed[raw_states == 3] = 2
        return collapsed

    def _build_result(
        self,
        features: pd.DataFrame,
        states: np.ndarray,
        *,
        probabilities: np.ndarray | None,
        diagnostics: dict[str, Any],
    ) -> RegimeModelResult:
        self._require_fitted()
        labels = ordered_economic_labels(self._label_map)
        summary = build_state_summary(features, states)
        summary["economic_label"] = [
            self._label_map[int(state)] for state in summary.index
        ]
        result = RegimeModelResult(
            model_name=self.model_name,
            model_version=self.model_version,
            fitted_parameters={
                "vol_threshold": self._vol_threshold,
                "corr_threshold": self._corr_threshold,
                "high_vol_percentile": self.config.high_vol_percentile,
                "high_corr_percentile": self.config.high_corr_percentile,
                "n_states": self.config.n_states,
                "label_map": {str(k): v for k, v in self._label_map.items()},
            },
            feature_columns=self._feature_columns,
            fit_start=self._fit_start,  # type: ignore[arg-type]
            fit_end=self._fit_end,  # type: ignore[arg-type]
            dates=features.index,
            states=states.astype(int),
            economic_labels=labels,
            state_probabilities=probabilities,
            state_summary=summary,
            diagnostics=diagnostics,
            warnings=(),
            random_seed=None,
        )
        validate_regime_model_result(result)
        # Ensure label map was derived from training statistics only.
        _ = apply_label_map(states, self._label_map)
        return result

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ThresholdRegimeModel must be fit before transform")

"""KMeans regime clustering with train-only fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mrrp.models.base import (
    RegimeModel,
    reject_constant_features,
    validate_feature_matrix,
    validate_n_states,
)
from mrrp.models.labeling import (
    build_state_summary,
    map_states_to_economic_labels,
    ordered_economic_labels,
)
from mrrp.models.result import RegimeModelResult, validate_regime_model_result

MODEL_VERSION = "1.0.0"


@dataclass(frozen=True)
class KMeansConfig:
    """Configuration for KMeans regime models."""

    n_states: int = 3
    feature_columns: tuple[str, ...] | None = None
    random_seed: int = 42
    n_init: int = 10
    max_iter: int = 300


class KMeansRegimeModel(RegimeModel):
    """Fit KMeans on training features and infer states on later periods."""

    model_name = "kmeans"
    model_version = MODEL_VERSION

    def __init__(self, config: KMeansConfig | None = None) -> None:
        self.config = config or KMeansConfig()
        self._model: KMeans | None = None
        self._feature_columns: tuple[str, ...] = ()
        self._label_map: dict[int, str] = {}
        self._fit_start: pd.Timestamp | None = None
        self._fit_end: pd.Timestamp | None = None
        self._fitted = False

    def fit(self, train_features: pd.DataFrame) -> RegimeModelResult:
        """Fit KMeans using the training period only."""
        n_states = validate_n_states(self.config.n_states)
        columns = self._resolve_columns(train_features)
        train = validate_feature_matrix(
            train_features,
            feature_columns=columns,
            min_observations=max(n_states * 10, 30),
        )
        reject_constant_features(train)

        model = KMeans(
            n_clusters=n_states,
            random_state=self.config.random_seed,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
        )
        states = model.fit_predict(train.to_numpy(dtype=float))
        summary = build_state_summary(train, states)
        self._label_map = map_states_to_economic_labels(summary)
        self._model = model
        self._feature_columns = columns
        self._fit_start = pd.Timestamp(train.index.min())
        self._fit_end = pd.Timestamp(train.index.max())
        self._fitted = True

        diagnostics = self._diagnostics(train, states)
        return self._build_result(train, states, diagnostics=diagnostics)

    def transform(self, features: pd.DataFrame) -> RegimeModelResult:
        """Assign clusters with the train-fitted KMeans model."""
        self._require_fitted()
        assert self._model is not None
        data = validate_feature_matrix(features, feature_columns=self._feature_columns)
        states = self._model.predict(data.to_numpy(dtype=float))
        diagnostics = self._diagnostics(data, states)
        return self._build_result(data, states, diagnostics=diagnostics)

    def _diagnostics(
        self, features: pd.DataFrame, states: np.ndarray
    ) -> dict[str, Any]:
        unique = np.unique(states)
        silhouette: float | None
        if len(unique) < 2 or len(features) <= len(unique):
            silhouette = None
        else:
            silhouette = float(silhouette_score(features.to_numpy(dtype=float), states))
        return {
            "silhouette": silhouette,
            "inertia": float(self._model.inertia_) if self._model is not None else None,
            "n_states": int(self.config.n_states),
        }

    def _build_result(
        self,
        features: pd.DataFrame,
        states: np.ndarray,
        *,
        diagnostics: dict[str, Any],
    ) -> RegimeModelResult:
        self._require_fitted()
        assert self._model is not None
        summary = build_state_summary(features, states)
        summary["economic_label"] = [
            self._label_map[int(state)] for state in summary.index
        ]
        result = RegimeModelResult(
            model_name=self.model_name,
            model_version=self.model_version,
            fitted_parameters={
                "n_states": self.config.n_states,
                "cluster_centers": self._model.cluster_centers_.tolist(),
                "label_map": {str(k): v for k, v in self._label_map.items()},
                "n_init": self.config.n_init,
                "max_iter": self.config.max_iter,
            },
            feature_columns=self._feature_columns,
            fit_start=self._fit_start,  # type: ignore[arg-type]
            fit_end=self._fit_end,  # type: ignore[arg-type]
            dates=features.index,
            states=np.asarray(states, dtype=int),
            economic_labels=ordered_economic_labels(self._label_map),
            state_probabilities=None,
            state_summary=summary,
            diagnostics=diagnostics,
            warnings=(),
            random_seed=self.config.random_seed,
        )
        validate_regime_model_result(result)
        return result

    def _resolve_columns(self, features: pd.DataFrame) -> tuple[str, ...]:
        if self.config.feature_columns is not None:
            return tuple(self.config.feature_columns)
        return tuple(str(column) for column in features.columns)

    def _require_fitted(self) -> None:
        if not self._fitted or self._model is None:
            raise RuntimeError("KMeansRegimeModel must be fit before transform")

"""Gaussian Mixture Model regimes with soft probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

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
CovarianceType = Literal["full", "tied", "diag", "spherical"]
SUPPORTED_COVARIANCE_TYPES = frozenset({"full", "tied", "diag", "spherical"})


@dataclass(frozen=True)
class GMMConfig:
    """Configuration for Gaussian mixture regime models."""

    n_states: int = 3
    covariance_type: CovarianceType = "full"
    feature_columns: tuple[str, ...] | None = None
    random_seed: int = 42
    n_init: int = 5
    max_iter: int = 200
    reg_covar: float = 1e-6


class GMMRegimeModel(RegimeModel):
    """Fit a Gaussian mixture on training features and infer later periods."""

    model_name = "gmm"
    model_version = MODEL_VERSION

    def __init__(self, config: GMMConfig | None = None) -> None:
        self.config = config or GMMConfig()
        self._model: GaussianMixture | None = None
        self._feature_columns: tuple[str, ...] = ()
        self._label_map: dict[int, str] = {}
        self._fit_start: pd.Timestamp | None = None
        self._fit_end: pd.Timestamp | None = None
        self._train_aic: float | None = None
        self._train_bic: float | None = None
        self._fitted = False

    def fit(self, train_features: pd.DataFrame) -> RegimeModelResult:
        """Fit the GMM using the training period only."""
        n_states = validate_n_states(self.config.n_states)
        if self.config.covariance_type not in SUPPORTED_COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {sorted(SUPPORTED_COVARIANCE_TYPES)}"
            )
        columns = self._resolve_columns(train_features)
        train = validate_feature_matrix(
            train_features,
            feature_columns=columns,
            min_observations=max(n_states * 15, 40),
        )
        reject_constant_features(train)

        model = GaussianMixture(
            n_components=n_states,
            covariance_type=self.config.covariance_type,
            random_state=self.config.random_seed,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
            reg_covar=self.config.reg_covar,
        )
        values = train.to_numpy(dtype=float)
        model.fit(values)
        states = model.predict(values)
        probabilities = model.predict_proba(values)
        summary = build_state_summary(train, states)
        self._label_map = map_states_to_economic_labels(summary)
        self._model = model
        self._feature_columns = columns
        self._fit_start = pd.Timestamp(train.index.min())
        self._fit_end = pd.Timestamp(train.index.max())
        self._train_aic = float(model.aic(values))
        self._train_bic = float(model.bic(values))
        self._fitted = True

        warnings: list[str] = []
        if not bool(model.converged_):
            warnings.append("GaussianMixture did not converge within max_iter")
        diagnostics = {
            "aic": self._train_aic,
            "bic": self._train_bic,
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
            "lower_bound": float(model.lower_bound_),
            "covariance_type": self.config.covariance_type,
            "n_states": n_states,
        }
        return self._build_result(
            train,
            states,
            probabilities=probabilities,
            diagnostics=diagnostics,
            warnings=tuple(warnings),
        )

    def transform(self, features: pd.DataFrame) -> RegimeModelResult:
        """Infer hard/soft state assignments with the train-fitted GMM."""
        self._require_fitted()
        assert self._model is not None
        data = validate_feature_matrix(features, feature_columns=self._feature_columns)
        values = data.to_numpy(dtype=float)
        states = self._model.predict(values)
        probabilities = self._model.predict_proba(values)
        diagnostics = {
            "aic": self._train_aic,
            "bic": self._train_bic,
            "converged": bool(self._model.converged_),
            "score": float(self._model.score(values)),
            "covariance_type": self.config.covariance_type,
            "n_states": int(self.config.n_states),
            "note": "AIC/BIC values are from the training fit and are not recomputed",
        }
        return self._build_result(
            data,
            states,
            probabilities=probabilities,
            diagnostics=diagnostics,
            warnings=(),
        )

    def _build_result(
        self,
        features: pd.DataFrame,
        states: np.ndarray,
        *,
        probabilities: np.ndarray,
        diagnostics: dict[str, Any],
        warnings: tuple[str, ...],
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
                "covariance_type": self.config.covariance_type,
                "weights": self._model.weights_.tolist(),
                "means": self._model.means_.tolist(),
                "label_map": {str(k): v for k, v in self._label_map.items()},
                "n_init": self.config.n_init,
                "max_iter": self.config.max_iter,
                "reg_covar": self.config.reg_covar,
            },
            feature_columns=self._feature_columns,
            fit_start=self._fit_start,  # type: ignore[arg-type]
            fit_end=self._fit_end,  # type: ignore[arg-type]
            dates=features.index,
            states=np.asarray(states, dtype=int),
            economic_labels=ordered_economic_labels(self._label_map),
            state_probabilities=np.asarray(probabilities, dtype=float),
            state_summary=summary,
            diagnostics=diagnostics,
            warnings=warnings,
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
            raise RuntimeError("GMMRegimeModel must be fit before transform")

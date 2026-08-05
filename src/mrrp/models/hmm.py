"""Hidden Markov Model regimes via hmmlearn.GaussianHMM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

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


@dataclass(frozen=True)
class HMMConfig:
    """Configuration for GaussianHMM regime models."""

    n_states: int = 3
    covariance_type: CovarianceType = "diag"
    feature_columns: tuple[str, ...] | None = None
    random_seed: int = 42
    n_inits: int = 5
    n_iter: int = 200
    tol: float = 1e-3


class HMMRegimeModel(RegimeModel):
    """Fit GaussianHMM on training data with multi-init train log-likelihood selection."""

    model_name = "hmm"
    model_version = MODEL_VERSION

    def __init__(self, config: HMMConfig | None = None) -> None:
        self.config = config or HMMConfig()
        self._model: GaussianHMM | None = None
        self._feature_columns: tuple[str, ...] = ()
        self._label_map: dict[int, str] = {}
        self._fit_start: pd.Timestamp | None = None
        self._fit_end: pd.Timestamp | None = None
        self._train_log_likelihood: float | None = None
        self._selected_seed: int | None = None
        self._fitted = False

    def fit(self, train_features: pd.DataFrame) -> RegimeModelResult:
        """Fit HMM using training observations only; select best init by train score."""
        n_states = validate_n_states(self.config.n_states)
        if self.config.n_inits < 1:
            raise ValueError("n_inits must be >= 1")
        columns = self._resolve_columns(train_features)
        train = validate_feature_matrix(
            train_features,
            feature_columns=columns,
            min_observations=max(n_states * 20, 50),
        )
        reject_constant_features(train)
        values = train.to_numpy(dtype=float)

        best_model: GaussianHMM | None = None
        best_score = -np.inf
        best_seed = self.config.random_seed
        warnings: list[str] = []

        for init_idx in range(self.config.n_inits):
            seed = self.config.random_seed + init_idx
            candidate = GaussianHMM(
                n_components=n_states,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                tol=self.config.tol,
                random_state=seed,
                init_params="stmc",
                params="stmc",
            )
            try:
                candidate.fit(values)
                score = float(candidate.score(values))
            except Exception as exc:  # noqa: BLE001 - collect init failures as warnings
                warnings.append(f"HMM init seed={seed} failed: {exc}")
                continue
            if not bool(candidate.monitor_.converged):
                warnings.append(f"HMM init seed={seed} did not converge")
            if score > best_score:
                best_score = score
                best_model = candidate
                best_seed = seed

        if best_model is None:
            raise RuntimeError("All HMM initializations failed to fit")

        states = best_model.predict(values)
        probabilities = best_model.predict_proba(values)
        summary = build_state_summary(train, states)
        self._label_map = map_states_to_economic_labels(summary)
        self._model = best_model
        self._feature_columns = columns
        self._fit_start = pd.Timestamp(train.index.min())
        self._fit_end = pd.Timestamp(train.index.max())
        self._train_log_likelihood = best_score
        self._selected_seed = best_seed
        self._fitted = True

        diagnostics = self._diagnostics(train, states, values)
        return self._build_result(
            train,
            states,
            probabilities=probabilities,
            diagnostics=diagnostics,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def transform(self, features: pd.DataFrame) -> RegimeModelResult:
        """Infer states/posteriors with the train-selected HMM."""
        self._require_fitted()
        assert self._model is not None
        data = validate_feature_matrix(features, feature_columns=self._feature_columns)
        values = data.to_numpy(dtype=float)
        states = self._model.predict(values)
        probabilities = self._model.predict_proba(values)
        diagnostics = self._diagnostics(data, states, values)
        diagnostics["note"] = (
            "train_log_likelihood is from the training selection step; "
            "state count was not chosen using test observations"
        )
        return self._build_result(
            data,
            states,
            probabilities=probabilities,
            diagnostics=diagnostics,
            warnings=(),
        )

    def _diagnostics(
        self,
        features: pd.DataFrame,
        states: np.ndarray,
        values: np.ndarray,
    ) -> dict[str, Any]:
        assert self._model is not None
        transmat = np.asarray(self._model.transmat_, dtype=float)
        empirical_duration = _empirical_state_durations(states, self.config.n_states)
        implied_duration = _implied_state_durations(transmat)
        return {
            "n_states": int(self.config.n_states),
            "covariance_type": self.config.covariance_type,
            "train_log_likelihood": self._train_log_likelihood,
            "selected_init_seed": self._selected_seed,
            "converged": bool(self._model.monitor_.converged),
            "transition_matrix": transmat.tolist(),
            "empirical_state_duration": empirical_duration,
            "implied_state_duration": implied_duration,
            "score": float(self._model.score(values)),
            "n_observations": int(len(features)),
        }

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
                "transmat": np.asarray(self._model.transmat_, dtype=float).tolist(),
                "startprob": np.asarray(self._model.startprob_, dtype=float).tolist(),
                "means": np.asarray(self._model.means_, dtype=float).tolist(),
                "label_map": {str(k): v for k, v in self._label_map.items()},
                "n_inits": self.config.n_inits,
                "n_iter": self.config.n_iter,
                "selected_init_seed": self._selected_seed,
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
            raise RuntimeError("HMMRegimeModel must be fit before transform")


def _empirical_state_durations(states: np.ndarray, n_states: int) -> list[float]:
    arr = np.asarray(states, dtype=int)
    durations = [[] for _ in range(n_states)]
    if arr.size == 0:
        return [float("nan")] * n_states
    start = 0
    for idx in range(1, len(arr) + 1):
        if idx == len(arr) or arr[idx] != arr[start]:
            state = int(arr[start])
            durations[state].append(idx - start)
            start = idx
    return [float(np.mean(values)) if values else float("nan") for values in durations]


def _implied_state_durations(transmat: np.ndarray) -> list[float]:
    diag = np.clip(np.diag(np.asarray(transmat, dtype=float)), 0.0, 0.999999)
    return [float(1.0 / (1.0 - p)) if p < 1.0 else float("inf") for p in diag]

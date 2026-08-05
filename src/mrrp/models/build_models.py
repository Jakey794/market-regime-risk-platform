"""Fit configured regime models and persist comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mrrp.data.cache import load_parquet
from mrrp.models.base import chronological_split
from mrrp.models.changepoint import ChangePointConfig, ChangePointDetector
from mrrp.models.compare import build_comparison_table
from mrrp.models.config import load_regime_models_config
from mrrp.models.gmm import GMMConfig, GMMRegimeModel
from mrrp.models.hmm import HMMConfig, HMMRegimeModel
from mrrp.models.kmeans import KMeansConfig, KMeansRegimeModel
from mrrp.models.result import RegimeModelResult
from mrrp.models.threshold import ThresholdConfig, ThresholdRegimeModel

DEFAULT_CONFIG = Path("configs/regime_models.yaml")


def fit_configured_models(
    features: pd.DataFrame,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
) -> list[RegimeModelResult]:
    """Fit enabled models using train-only parameters."""
    config = load_regime_models_config(config_path)
    split = chronological_split(
        features.loc[:, list(config.feature_columns)],
        train_end=config.train_end,
        validation_end=config.validation_end,
    )
    results: list[RegimeModelResult] = []
    raw = config.raw

    threshold_cfg = raw.get("threshold", {})
    if threshold_cfg.get("enabled", True):
        for n_states in threshold_cfg.get("n_states", [4]):
            model = ThresholdRegimeModel(
                ThresholdConfig(
                    vol_column=threshold_cfg.get("vol_column", "portfolio_vol_63d"),
                    corr_column=threshold_cfg.get("corr_column", "mean_corr_63d"),
                    high_vol_percentile=float(
                        threshold_cfg.get("high_vol_percentile", 0.75)
                    ),
                    high_corr_percentile=float(
                        threshold_cfg.get("high_corr_percentile", 0.75)
                    ),
                    n_states=int(n_states),
                )
            )
            model.fit(split.train)
            results.append(model.transform(split.full))

    kmeans_cfg = raw.get("kmeans", {})
    if kmeans_cfg.get("enabled", True):
        for n_states in kmeans_cfg.get("n_states", [2, 3, 4]):
            model = KMeansRegimeModel(
                KMeansConfig(
                    n_states=int(n_states),
                    feature_columns=config.feature_columns,
                    random_seed=config.random_seed,
                    n_init=int(kmeans_cfg.get("n_init", 10)),
                    max_iter=int(kmeans_cfg.get("max_iter", 300)),
                )
            )
            model.fit(split.train)
            results.append(model.transform(split.full))

    gmm_cfg = raw.get("gmm", {})
    if gmm_cfg.get("enabled", True):
        covariance_types = gmm_cfg.get("covariance_types", ["full"])
        for n_states in gmm_cfg.get("n_states", [2, 3, 4]):
            for covariance_type in covariance_types:
                model = GMMRegimeModel(
                    GMMConfig(
                        n_states=int(n_states),
                        covariance_type=covariance_type,
                        feature_columns=config.feature_columns,
                        random_seed=config.random_seed,
                        n_init=int(gmm_cfg.get("n_init", 5)),
                        max_iter=int(gmm_cfg.get("max_iter", 200)),
                        reg_covar=float(gmm_cfg.get("reg_covar", 1e-6)),
                    )
                )
                model.fit(split.train)
                results.append(model.transform(split.full))

    hmm_cfg = raw.get("hmm", {})
    if hmm_cfg.get("enabled", False):
        for n_states in hmm_cfg.get("n_states", [3]):
            model = HMMRegimeModel(
                HMMConfig(
                    n_states=int(n_states),
                    covariance_type=hmm_cfg.get("covariance_type", "diag"),
                    feature_columns=config.feature_columns,
                    random_seed=config.random_seed,
                    n_inits=int(hmm_cfg.get("n_inits", 5)),
                    n_iter=int(hmm_cfg.get("n_iter", 200)),
                    tol=float(hmm_cfg.get("tol", 1e-3)),
                )
            )
            model.fit(split.train)
            results.append(model.transform(split.full))

    changepoint_cfg = raw.get("changepoint", {})
    if changepoint_cfg.get("enabled", False):
        for method in changepoint_cfg.get("methods", ["pelt"]):
            for signal in changepoint_cfg.get("signals", ["volatility"]):
                detector = ChangePointDetector(
                    ChangePointConfig(
                        method=method,
                        signal=signal,
                        penalty=float(changepoint_cfg.get("penalty", 10.0)),
                        n_bkps=changepoint_cfg.get("n_bkps"),
                        model=str(changepoint_cfg.get("model", "l2")),
                        min_size=int(changepoint_cfg.get("min_size", 5)),
                    )
                )
                # Change points are descriptive and are detected on training data
                # only. Persist their train-period segments alongside fitted models.
                detected = detector.detect(split.train)
                results.append(detected.to_regime_result(split.train))

    return results


def persist_model_results(
    results: list[RegimeModelResult],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Persist comparison table and per-model JSON summaries."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison_table(results)
    comparison_path = out / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    summaries = []
    for result in results:
        payload = result.to_dict()
        stem = f"{result.model_name}_{len(result.economic_labels)}"
        if "covariance_type" in result.fitted_parameters:
            stem += f"_{result.fitted_parameters['covariance_type']}"
        if result.model_name == "changepoint":
            stem += (
                f"_{result.fitted_parameters['method']}"
                f"_{result.fitted_parameters['signal']}"
            )
        path = out / f"{stem}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        state_path = out / f"{stem}_states.csv"
        result.state_series().to_csv(state_path, header=True)
        summaries.append({"model": stem, "json": str(path), "states": str(state_path)})

    manifest = {
        "comparison": str(comparison_path),
        "models": summaries,
        "n_models": len(results),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--features",
        default="data/processed/regime_features_scaled.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/regime_models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_parquet(args.features)
    results = fit_configured_models(features, config_path=args.config)
    manifest = persist_model_results(results, args.output_dir)
    print(f"Fitted {manifest['n_models']} regime models")
    print(f"Comparison: {manifest['comparison']}")


if __name__ == "__main__":
    main()

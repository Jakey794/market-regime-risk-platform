from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mrrp.features import (
    EXPECTED_REGIME_FEATURE_COLUMNS,
    clean_feature_matrix,
    date_train_test_split,
    fit_train_scaler,
    scale_train_test_features,
    transform_features,
    validate_feature_frame,
)


def _valid_features(periods: int = 5) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    data = {
        column: [float(i) for i in range(periods)]
        for column in EXPECTED_REGIME_FEATURE_COLUMNS
    }
    return pd.DataFrame(data, index=index)


def test_valid_feature_frame_passes() -> None:
    validate_feature_frame(_valid_features())


def test_non_dataframe_input_rejected() -> None:
    with pytest.raises(ValueError, match="pandas DataFrame"):
        validate_feature_frame([1, 2, 3])


def test_non_datetime_index_rejected() -> None:
    features = _valid_features()
    features.index = range(len(features))

    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_feature_frame(features)


def test_duplicate_dates_rejected() -> None:
    features = _valid_features()
    features.index = features.index[:-1].append(features.index[-2:-1])

    with pytest.raises(ValueError, match="duplicate dates"):
        validate_feature_frame(features)


def test_non_monotonic_index_rejected() -> None:
    features = _valid_features().iloc[::-1]

    with pytest.raises(ValueError, match="monotonic increasing"):
        validate_feature_frame(features)


def test_valid_feature_frame_preserves_datetime_index() -> None:
    features = _valid_features()
    original_index = features.index

    validate_feature_frame(features)

    assert isinstance(features.index, pd.DatetimeIndex)
    assert features.index.equals(original_index)


def _ml_features() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "volatility": [0.0, 2.0, 4.0, 100.0, 200.0, 300.0],
            "correlation": [1.0, 3.0, 5.0, 50.0, 60.0, 70.0],
        },
        index=index,
    )


def test_clean_feature_matrix_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns.*momentum"):
        clean_feature_matrix(_ml_features(), ["volatility", "momentum"])


def test_clean_feature_matrix_replaces_inf_and_drops_required_missing_rows() -> None:
    features = _ml_features()
    features.iloc[1, 0] = np.inf
    features.iloc[3, 1] = -np.inf
    features.iloc[4, 0] = np.nan

    result = clean_feature_matrix(features, ["volatility", "correlation"])

    assert result.index.equals(features.index[[0, 2, 5]])
    assert np.isfinite(result.to_numpy()).all()
    assert np.isinf(features.to_numpy()).any()


def test_clean_feature_matrix_rejects_duplicate_dates() -> None:
    features = _ml_features()
    features.index = features.index[:-1].append(features.index[-2:-1])

    with pytest.raises(ValueError, match="duplicate dates"):
        clean_feature_matrix(features, ["volatility"])


def test_clean_feature_matrix_rejects_non_datetime_index() -> None:
    features = _ml_features().reset_index(drop=True)

    with pytest.raises(ValueError, match="DatetimeIndex"):
        clean_feature_matrix(features, ["volatility"])


def test_date_train_test_split_keeps_test_dates_strictly_after_train() -> None:
    features = _ml_features()

    train, test = date_train_test_split(features, train_end="2024-01-03")

    assert train.index.equals(features.index[:3])
    assert test.index.equals(features.index[3:])
    assert train.index.max() < test.index.min()


def test_fit_train_scaler_uses_only_training_data() -> None:
    features = _ml_features()

    scaled_train, scaled_test, scaler = scale_train_test_features(
        features, train_end="2024-01-03"
    )

    assert scaler.mean_.tolist() == pytest.approx([2.0, 3.0])
    assert scaled_train.mean().tolist() == pytest.approx([0.0, 0.0], abs=1e-12)
    assert not np.allclose(scaled_test.mean().to_numpy(), 0.0)


def test_transform_features_preserves_index_and_columns() -> None:
    features = _ml_features()
    train, test = date_train_test_split(features, train_end="2024-01-03")
    scaler = fit_train_scaler(train)

    result = transform_features(scaler, test)

    assert result.index.equals(test.index)
    assert result.columns.equals(test.columns)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_fit_train_scaler_rejects_infinite_values() -> None:
    features = _ml_features().iloc[:3].copy()
    features.iloc[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        fit_train_scaler(features)

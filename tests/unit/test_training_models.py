"""Tests for RUL model definitions."""

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from predictive_maintenance.training.models import (
    AdaptiveLassoRegressor,
    create_adaptive_lasso,
    create_candidate_models,
    create_lightgbm,
    create_linear_regression,
    create_xgboost,
)


def test_linear_regression_pipeline_scales_and_predicts() -> None:
    model = create_linear_regression()

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["model"], LinearRegression)

    features = np.array([[1.0], [2.0], [3.0], [4.0]])
    targets = np.array([40.0, 30.0, 20.0, 10.0])
    model.fit(features, targets)

    prediction = model.predict(np.array([[5.0]]))

    assert prediction[0] == pytest.approx(0.0, abs=1e-10)


def test_adaptive_lasso_learns_weights_and_predicts() -> None:
    model = create_adaptive_lasso()
    features = np.array(
        [
            [0.0, 1.0],
            [1.0, -1.0],
            [2.0, 1.0],
            [3.0, -1.0],
            [4.0, 1.0],
            [5.0, -1.0],
        ]
    )
    targets = np.array([50.0, 40.0, 30.0, 20.0, 10.0, 0.0])

    model.fit(features, targets)
    adaptive_model = model.named_steps["model"]
    predictions = model.predict(features)

    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(adaptive_model, AdaptiveLassoRegressor)
    assert adaptive_model.adaptive_weights_.shape == (2,)
    assert np.all(adaptive_model.adaptive_weights_ > 0)
    assert predictions.shape == targets.shape
    assert np.all(np.isfinite(predictions))


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("alpha", 0.0, "alpha must be greater than zero"),
        ("initial_ridge_alpha", 0.0, "initial_ridge_alpha must be greater than zero"),
        ("gamma", 0.0, "gamma must be greater than zero"),
        ("epsilon", 0.0, "epsilon must be greater than zero"),
        ("max_iter", 0, "max_iter must be greater than zero"),
    ],
)
def test_adaptive_lasso_rejects_invalid_parameters(
    parameter: str,
    value: float,
    message: str,
) -> None:
    model = AdaptiveLassoRegressor(**{parameter: value})

    with pytest.raises(ValueError, match=message):
        model.fit(np.array([[1.0], [2.0]]), np.array([2.0, 1.0]))


def test_xgboost_has_reproducible_baseline_configuration() -> None:
    model = create_xgboost(random_seed=7)

    assert isinstance(model, XGBRegressor)
    assert model.get_params()["objective"] == "reg:squarederror"
    assert model.get_params()["random_state"] == 7
    assert model.get_params()["n_jobs"] == 1


def test_lightgbm_has_reproducible_baseline_configuration() -> None:
    model = create_lightgbm(random_seed=7)

    assert isinstance(model, LGBMRegressor)
    assert model.get_params()["objective"] == "regression"
    assert model.get_params()["random_state"] == 7
    assert model.get_params()["n_jobs"] == 1


def test_candidate_models_contains_all_four_models() -> None:
    models = create_candidate_models(random_seed=11)

    assert set(models) == {
        "linear_regression",
        "adaptive_lasso",
        "xgboost",
        "lightgbm",
    }

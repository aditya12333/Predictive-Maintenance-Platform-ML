"""Model definitions used by the RUL training pipeline."""

from typing import Protocol, Self

import numpy as np
from lightgbm import LGBMRegressor
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import XGBRegressor

DEFAULT_RANDOM_SEED = 42


class RULRegressor(Protocol):
    """Shared interface required by the model-comparison training loop."""

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Fit the regressor."""

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        """Predict RUL values."""


class AdaptiveLassoRegressor(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """Lasso regression with feature-specific penalties learned from Ridge."""

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        initial_ridge_alpha: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 1e-6,
        max_iter: int = 10_000,
    ) -> None:
        self.alpha = alpha
        self.initial_ridge_alpha = initial_ridge_alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter

    def fit(self, features: ArrayLike, targets: ArrayLike) -> Self:
        """Learn adaptive feature penalties, then fit the weighted Lasso."""

        self._validate_parameters()
        checked_features, checked_targets = check_X_y(
            features,
            targets,
            dtype=np.float64,
            ensure_all_finite=True,
        )

        initial_model = Ridge(alpha=self.initial_ridge_alpha)
        initial_model.fit(checked_features, checked_targets)
        initial_coefficients = np.asarray(initial_model.coef_, dtype=np.float64)

        self.adaptive_weights_ = 1.0 / (np.abs(initial_coefficients) + self.epsilon) ** self.gamma
        self.model_ = Lasso(alpha=self.alpha, max_iter=self.max_iter)
        self.model_.fit(checked_features / self.adaptive_weights_, checked_targets)
        self.n_features_in_ = checked_features.shape[1]
        return self

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        """Predict RUL values using the learned adaptive feature penalties."""

        check_is_fitted(self, attributes=("adaptive_weights_", "model_"))
        checked_features = check_array(
            features,
            dtype=np.float64,
            ensure_all_finite=True,
        )
        if checked_features.shape[1] != self.n_features_in_:
            raise ValueError("features has a different number of columns than the fitted model")

        predictions = self.model_.predict(checked_features / self.adaptive_weights_)
        return np.asarray(predictions, dtype=np.float64)

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Return coefficients on the original, unweighted feature scale."""

        check_is_fitted(self, attributes=("adaptive_weights_", "model_"))
        return np.asarray(self.model_.coef_ / self.adaptive_weights_, dtype=np.float64)

    @property
    def intercept_(self) -> float:
        """Return the fitted intercept."""

        check_is_fitted(self, attributes=("model_",))
        return float(self.model_.intercept_)

    def _validate_parameters(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than zero")
        if self.initial_ridge_alpha <= 0:
            raise ValueError("initial_ridge_alpha must be greater than zero")
        if self.gamma <= 0:
            raise ValueError("gamma must be greater than zero")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be greater than zero")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than zero")


def create_linear_regression() -> Pipeline:
    """Create an unfitted, scaled Linear Regression model."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def create_adaptive_lasso() -> Pipeline:
    """Create an unfitted, scaled Adaptive Lasso model."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", AdaptiveLassoRegressor()),
        ]
    )


def create_xgboost(*, random_seed: int = DEFAULT_RANDOM_SEED) -> XGBRegressor:
    """Create an unfitted XGBoost RUL regressor with reproducible defaults."""

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=random_seed,
        n_jobs=1,
        verbosity=0,
    )


def create_lightgbm(*, random_seed: int = DEFAULT_RANDOM_SEED) -> LGBMRegressor:
    """Create an unfitted LightGBM RUL regressor with reproducible defaults."""

    return LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_seed,
        n_jobs=1,
        verbosity=-1,
    )


def create_candidate_models(*, random_seed: int = DEFAULT_RANDOM_SEED) -> dict[str, RULRegressor]:
    """Create all unfitted models included in the RUL comparison."""

    return {
        "linear_regression": create_linear_regression(),
        "adaptive_lasso": create_adaptive_lasso(),
        "xgboost": create_xgboost(random_seed=random_seed),
        "lightgbm": create_lightgbm(random_seed=random_seed),
    }

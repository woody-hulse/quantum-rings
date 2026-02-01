"""
Random forest model for duration prediction: threshold as input, log2(duration) target.
"""

from typing import Dict, Any
from pathlib import Path

import numpy as np
import joblib

from .gradient_boosting_base import GradientBoostingRegressionModel

from sklearn.ensemble import RandomForestRegressor


class RandomForestModel(GradientBoostingRegressionModel):
    """Random forest for duration prediction: threshold as input, log2(duration) target."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

    @property
    def name(self) -> str:
        return "RandomForest"

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def _create_regressor(self, **params) -> Any:
        return RandomForestRegressor(**params)

    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.runtime_model.fit(X_train, y_train)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.runtime_model, path / "runtime_model.joblib")
        self._save_scaler(path)

    def load(self, path: Path) -> None:
        path = Path(path)
        self.runtime_model = joblib.load(path / "runtime_model.joblib")
        self._load_scaler(path)

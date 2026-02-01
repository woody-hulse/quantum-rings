"""
CatBoost model for duration prediction: threshold as input, log2(duration) target.
"""

from typing import Dict, Any
from pathlib import Path

import numpy as np

from .gradient_boosting_base import GradientBoostingRegressionModel

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class CatBoostModel(GradientBoostingRegressionModel):
    """CatBoost for duration prediction: threshold as input, log2(duration) target."""

    def __init__(
        self,
        depth: int = 6,
        learning_rate: float = 0.1,
        iterations: int = 100,
        l2_leaf_reg: float = 3.0,
        random_state: int = 42,
        verbose: bool = False,
    ):
        if not HAS_CATBOOST:
            raise ImportError("catboost is required. Install with: pip install catboost")
        super().__init__()
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "CatBoost"

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "l2_leaf_reg": self.l2_leaf_reg,
            "loss_function": "RMSE",
            "random_seed": self.random_state,
            "verbose": self.verbose,
            "allow_writing_files": False,
        }

    def _create_regressor(self, **params) -> Any:
        return CatBoostRegressor(**params)

    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.runtime_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.runtime_model.save_model(str(path / "runtime_model.cbm"))
        self._save_scaler(path)

    def load(self, path: Path) -> None:
        path = Path(path)
        self.runtime_model = CatBoostRegressor()
        self.runtime_model.load_model(str(path / "runtime_model.cbm"))
        self._load_scaler(path)

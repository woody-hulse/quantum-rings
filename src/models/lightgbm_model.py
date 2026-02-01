"""
LightGBM model for duration prediction: threshold as input, log2(duration) target.
"""

from typing import Dict, Any
from pathlib import Path
import warnings

import numpy as np

from models.gradient_boosting_base import GradientBoostingRegressionModel

warnings.filterwarnings('ignore', message='X does not have valid feature names')

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class LightGBMModel(GradientBoostingRegressionModel):
    """LightGBM for duration prediction: threshold as input, log2(duration) target."""

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbose: int = -1,
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        super().__init__()
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "LightGBM"

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            "objective": "regression",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "n_jobs": 1,
        }

    def _create_regressor(self, **params) -> Any:
        return lgb.LGBMRegressor(**params)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.runtime_model.booster_.save_model(str(path / "runtime_model.txt"))
        self._save_scaler(path)

    def load(self, path: Path) -> None:
        path = Path(path)
        self.runtime_model = lgb.Booster(model_file=str(path / "runtime_model.txt"))
        self._load_scaler(path)

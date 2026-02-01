"""
XGBoost model for duration prediction: threshold as input, log2(duration) target.
"""

from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np

from models.gradient_boosting_base import GradientBoostingRegressionModel

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class XGBoostModel(GradientBoostingRegressionModel):
    """XGBoost for duration prediction: threshold as input, log2(duration) target."""

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        super().__init__()
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "XGBoost"

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
        }

    def _create_regressor(self, **params) -> Any:
        return xgb.XGBRegressor(**params)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.runtime_model.save_model(str(path / "runtime_model.json"))
        self._save_scaler(path)

    def load(self, path: Path) -> None:
        path = Path(path)
        self.runtime_model = xgb.XGBRegressor()
        self.runtime_model.load_model(str(path / "runtime_model.json"))
        self._load_scaler(path)

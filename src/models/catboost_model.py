"""
CatBoost model implementation for threshold classification and runtime regression.
"""

from typing import Dict, Tuple, Any, Optional
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER
from models.base import BaseModel

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class CatBoostModel(BaseModel):
    """CatBoost model implementing the common interface."""

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
            raise ImportError(
                "catboost is required. Install with: pip install catboost"
            )

        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.verbose = verbose

        self.threshold_model: Optional[CatBoostRegressor] = None
        self.runtime_model: Optional[CatBoostRegressor] = None
        self.scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "CatBoost"

    def _extract_data(
        self, loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_features = []
        all_thresh = []
        all_runtime = []

        for batch in loader:
            all_features.append(batch["features"].numpy())
            all_thresh.extend(batch["threshold_class"].tolist())
            all_runtime.extend(batch["log_runtime"].tolist())

        X = np.vstack(all_features)
        y_thresh = np.array(all_thresh, dtype=np.float32)
        y_runtime = np.array(all_runtime)

        return X, y_thresh, y_runtime

    def _round_to_class(self, pred: np.ndarray) -> np.ndarray:
        """Round regression predictions to nearest valid class index."""
        return np.clip(np.round(pred), 0, len(THRESHOLD_LADDER) - 1).astype(int)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        X_train, y_thresh_train, y_runtime_train = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val = self._extract_data(val_loader)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        thresh_params = {
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "l2_leaf_reg": self.l2_leaf_reg,
            "loss_function": "RMSE",
            "random_seed": self.random_state,
            "verbose": self.verbose,
            "allow_writing_files": False,  # Prevent logs
        }

        if verbose:
            print("Training threshold regressor...")
        self.threshold_model = CatBoostRegressor(**thresh_params)
        self.threshold_model.fit(
            X_train_scaled,
            y_thresh_train,
            eval_set=(X_val_scaled, y_thresh_val),
            verbose=False,
        )

        runtime_params = {
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "l2_leaf_reg": self.l2_leaf_reg,
            "loss_function": "RMSE",
            "random_seed": self.random_state,
            "verbose": self.verbose,
            "allow_writing_files": False,
        }

        if verbose:
            print("Training runtime regressor...")
        self.runtime_model = CatBoostRegressor(**runtime_params)
        self.runtime_model.fit(
            X_train_scaled,
            y_runtime_train,
            eval_set=(X_val_scaled, y_runtime_val),
            verbose=False,
        )

        train_metrics = self._evaluate_internal(
            X_train_scaled, y_thresh_train, y_runtime_train
        )
        val_metrics = self._evaluate_internal(X_val_scaled, y_thresh_val, y_runtime_val)

        if verbose:
            print(
                f"Train Threshold Acc: {train_metrics['threshold_accuracy']:.4f} | "
                f"Train Runtime MSE: {train_metrics['runtime_mse']:.4f}"
            )
            print(
                f"Val Threshold Acc: {val_metrics['threshold_accuracy']:.4f} | "
                f"Val Runtime MSE: {val_metrics['runtime_mse']:.4f}"
            )

        return {"train": train_metrics, "val": val_metrics}

    def _evaluate_internal(
        self,
        X: np.ndarray,
        y_thresh: np.ndarray,
        y_runtime: np.ndarray,
    ) -> Dict[str, float]:
        thresh_pred_raw = self.threshold_model.predict(X)
        thresh_pred = self._round_to_class(thresh_pred_raw)
        runtime_pred = self.runtime_model.predict(X)

        return {
            "threshold_accuracy": accuracy_score(y_thresh.astype(int), thresh_pred),
            "runtime_mse": mean_squared_error(y_runtime, runtime_pred),
            "runtime_mae": mean_absolute_error(y_runtime, runtime_pred),
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        X, y_thresh, y_runtime = self._extract_data(loader)
        X_scaled = self.scaler.transform(X)
        return self._evaluate_internal(X_scaled, y_thresh, y_runtime)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold_model is None or self.runtime_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(features)
        thresh_pred_raw = self.threshold_model.predict(X_scaled)
        thresh_classes = self._round_to_class(thresh_pred_raw)
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])
        runtime_log = self.runtime_model.predict(X_scaled)
        runtime_values = np.expm1(runtime_log)

        return thresh_values, runtime_values

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        if self.threshold_model is None or self.runtime_model is None:
            return None
        return {
            "threshold": self.threshold_model.feature_importances_,
            "runtime": self.runtime_model.feature_importances_,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.threshold_model.save_model(str(path / "threshold_model.cbm"))
        self.runtime_model.save_model(str(path / "runtime_model.cbm"))

        np.save(path / "scaler_mean.npy", self.scaler.mean_)
        np.save(path / "scaler_scale.npy", self.scaler.scale_)

    def load(self, path: Path) -> None:
        path = Path(path)

        self.threshold_model = CatBoostRegressor()
        self.threshold_model.load_model(str(path / "threshold_model.cbm"))

        self.runtime_model = CatBoostRegressor()
        self.runtime_model.load_model(str(path / "runtime_model.cbm"))

        self.scaler.mean_ = np.load(path / "scaler_mean.npy")
        self.scaler.scale_ = np.load(path / "scaler_scale.npy")

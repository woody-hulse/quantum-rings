"""
CatBoost model implementation for threshold classification and runtime regression.
"""

from typing import Dict, Tuple, Any, Optional
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

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
        n_features_to_select: Optional[int] = None,  # NEW: feature selection
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
        self.n_features_to_select = n_features_to_select

        self.threshold_model: Optional[CatBoostRegressor] = None
        self.runtime_model: Optional[CatBoostRegressor] = None
        self.scaler = StandardScaler()
        self.feature_selector: Optional[SelectKBest] = None
        self.selected_features: Optional[np.ndarray] = None

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

        # Feature selection if requested
        if self.n_features_to_select is not None and self.n_features_to_select < X_train.shape[1]:
            if verbose:
                print(f"Selecting top {self.n_features_to_select} features...")

            # Use runtime target for feature selection (since it's harder to predict)
            self.feature_selector = SelectKBest(f_regression, k=self.n_features_to_select)
            X_train = self.feature_selector.fit_transform(X_train, y_runtime_train)
            X_val = self.feature_selector.transform(X_val)
            self.selected_features = self.feature_selector.get_support(indices=True)

            if verbose:
                print(f"Selected features: {self.selected_features}")

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
            early_stopping_rounds=15,  # Stop if no improvement
            verbose=False,
        )

        runtime_params = {
            "depth": max(4, self.depth - 2),  # Reduce depth for runtime
            "learning_rate": self.learning_rate * 0.5,  # Slower learning
            "iterations": self.iterations * 2,  # More iterations with early stopping
            "l2_leaf_reg": self.l2_leaf_reg * 3,  # Stronger regularization
            "loss_function": "MAE",  # MAE is more robust to outliers
            "random_seed": self.random_state,
            "verbose": self.verbose,
            "allow_writing_files": False,
            "subsample": 0.8,  # Bagging to reduce overfitting
            "rsm": 0.8,  # Random subspace method
        }

        if verbose:
            print("Training runtime regressor...")
        self.runtime_model = CatBoostRegressor(**runtime_params)
        self.runtime_model.fit(
            X_train_scaled,
            y_runtime_train,
            eval_set=(X_val_scaled, y_runtime_val),
            early_stopping_rounds=20,  # Stop if no improvement for 20 rounds
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

        # Apply feature selection if used
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)

        X_scaled = self.scaler.transform(X)
        return self._evaluate_internal(X_scaled, y_thresh, y_runtime)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold_model is None or self.runtime_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Apply feature selection if used
        if self.feature_selector is not None:
            features = self.feature_selector.transform(features)

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

        if self.selected_features is not None:
            np.save(path / "selected_features.npy", self.selected_features)

    def load(self, path: Path) -> None:
        path = Path(path)

        self.threshold_model = CatBoostRegressor()
        self.threshold_model.load_model(str(path / "threshold_model.cbm"))

        self.runtime_model = CatBoostRegressor()
        self.runtime_model.load_model(str(path / "runtime_model.cbm"))

        self.scaler.mean_ = np.load(path / "scaler_mean.npy")
        self.scaler.scale_ = np.load(path / "scaler_scale.npy")

        if (path / "selected_features.npy").exists():
            self.selected_features = np.load(path / "selected_features.npy")

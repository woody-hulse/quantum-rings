"""
XGBoost model with per-target top-20 correlation-based feature selection.

For each prediction target (threshold and runtime), computes the absolute
Pearson correlation of every feature with the target on the training set,
then selects the 20 most correlated features. Each XGBoost sub-model
trains only on its own top-20 feature subset.
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
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


TOP_K = 20


class Rohan7Model(BaseModel):
    """XGBoost with per-target top-K correlation feature selection."""

    def __init__(
        self,
        top_k: int = TOP_K,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required. Install with: pip install xgboost")

        self.top_k = top_k
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.threshold_model: Optional[xgb.XGBRegressor] = None
        self.runtime_model: Optional[xgb.XGBRegressor] = None
        self.thresh_scaler = StandardScaler()
        self.runtime_scaler = StandardScaler()

        # Feature indices selected per target (set during fit)
        self.thresh_feature_idx: Optional[np.ndarray] = None
        self.runtime_feature_idx: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "Rohan7"

    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _select_top_k_features(
        self, X: np.ndarray, y: np.ndarray, k: int
    ) -> np.ndarray:
        """Return indices of the k features with highest absolute correlation to y."""
        n_features = X.shape[1]
        correlations = np.zeros(n_features)

        for i in range(n_features):
            col = X[:, i]
            # Skip constant features
            if np.std(col) < 1e-12 or np.std(y) < 1e-12:
                correlations[i] = 0.0
            else:
                valid = np.isfinite(col) & np.isfinite(y)
                if valid.sum() > 2:
                    correlations[i] = abs(np.corrcoef(col[valid], y[valid])[0, 1])
                else:
                    correlations[i] = 0.0

        # Handle NaN correlations
        correlations = np.nan_to_num(correlations, nan=0.0)

        k = min(k, n_features)
        top_idx = np.argsort(correlations)[::-1][:k]
        return np.sort(top_idx)

    def _round_to_class(self, pred: np.ndarray) -> np.ndarray:
        return np.clip(np.round(pred), 0, len(THRESHOLD_LADDER) - 1).astype(int)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        X_train, y_thresh_train, y_runtime_train = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val = self._extract_data(val_loader)

        # Select top-k features per target from training data
        self.thresh_feature_idx = self._select_top_k_features(
            X_train, y_thresh_train, self.top_k
        )
        self.runtime_feature_idx = self._select_top_k_features(
            X_train, y_runtime_train, self.top_k
        )

        if verbose:
            print(f"Threshold features ({len(self.thresh_feature_idx)}): {self.thresh_feature_idx}")
            print(f"Runtime features ({len(self.runtime_feature_idx)}): {self.runtime_feature_idx}")

        # Subset and scale for threshold
        X_train_thresh = self.thresh_scaler.fit_transform(X_train[:, self.thresh_feature_idx])
        X_val_thresh = self.thresh_scaler.transform(X_val[:, self.thresh_feature_idx])

        # Subset and scale for runtime
        X_train_runtime = self.runtime_scaler.fit_transform(X_train[:, self.runtime_feature_idx])
        X_val_runtime = self.runtime_scaler.transform(X_val[:, self.runtime_feature_idx])

        xgb_params = {
            "objective": "reg:squarederror",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
        }

        if verbose:
            print("Training threshold regressor...")
        self.threshold_model = xgb.XGBRegressor(**xgb_params)
        self.threshold_model.fit(
            X_train_thresh, y_thresh_train,
            eval_set=[(X_val_thresh, y_thresh_val)],
            verbose=False,
        )

        if verbose:
            print("Training runtime regressor...")
        self.runtime_model = xgb.XGBRegressor(**xgb_params)
        self.runtime_model.fit(
            X_train_runtime, y_runtime_train,
            eval_set=[(X_val_runtime, y_runtime_val)],
            verbose=False,
        )

        train_metrics = self._evaluate_internal(
            X_train_thresh, y_thresh_train,
            X_train_runtime, y_runtime_train,
        )
        val_metrics = self._evaluate_internal(
            X_val_thresh, y_thresh_val,
            X_val_runtime, y_runtime_val,
        )

        if verbose:
            print(f"Train Threshold Acc: {train_metrics['threshold_accuracy']:.4f} | "
                  f"Train Runtime MSE: {train_metrics['runtime_mse']:.4f}")
            print(f"Val Threshold Acc: {val_metrics['threshold_accuracy']:.4f} | "
                  f"Val Runtime MSE: {val_metrics['runtime_mse']:.4f}")

        return {"train": train_metrics, "val": val_metrics}

    def _evaluate_internal(
        self,
        X_thresh: np.ndarray,
        y_thresh: np.ndarray,
        X_runtime: np.ndarray,
        y_runtime: np.ndarray,
    ) -> Dict[str, float]:
        thresh_pred = self._round_to_class(self.threshold_model.predict(X_thresh))
        runtime_pred = self.runtime_model.predict(X_runtime)

        return {
            "threshold_accuracy": accuracy_score(y_thresh.astype(int), thresh_pred),
            "runtime_mse": mean_squared_error(y_runtime, runtime_pred),
            "runtime_mae": mean_absolute_error(y_runtime, runtime_pred),
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        X, y_thresh, y_runtime = self._extract_data(loader)

        X_thresh = self.thresh_scaler.transform(X[:, self.thresh_feature_idx])
        X_runtime = self.runtime_scaler.transform(X[:, self.runtime_feature_idx])

        return self._evaluate_internal(X_thresh, y_thresh, X_runtime, y_runtime)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold_model is None or self.runtime_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_thresh = self.thresh_scaler.transform(features[:, self.thresh_feature_idx])
        X_runtime = self.runtime_scaler.transform(features[:, self.runtime_feature_idx])

        thresh_pred_raw = self.threshold_model.predict(X_thresh)
        thresh_classes = self._round_to_class(thresh_pred_raw)
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])

        runtime_log = self.runtime_model.predict(X_runtime)
        runtime_values = np.expm1(runtime_log)

        return thresh_values, runtime_values

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        if self.threshold_model is None or self.runtime_model is None:
            return None
        return {
            "threshold": self.threshold_model.feature_importances_,
            "runtime": self.runtime_model.feature_importances_,
            "threshold_feature_indices": self.thresh_feature_idx,
            "runtime_feature_indices": self.runtime_feature_idx,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.threshold_model.save_model(path / "threshold_model.json")
        self.runtime_model.save_model(path / "runtime_model.json")

        np.save(path / "thresh_scaler_mean.npy", self.thresh_scaler.mean_)
        np.save(path / "thresh_scaler_scale.npy", self.thresh_scaler.scale_)
        np.save(path / "runtime_scaler_mean.npy", self.runtime_scaler.mean_)
        np.save(path / "runtime_scaler_scale.npy", self.runtime_scaler.scale_)
        np.save(path / "thresh_feature_idx.npy", self.thresh_feature_idx)
        np.save(path / "runtime_feature_idx.npy", self.runtime_feature_idx)

    def load(self, path: Path) -> None:
        path = Path(path)

        self.threshold_model = xgb.XGBRegressor()
        self.threshold_model.load_model(path / "threshold_model.json")

        self.runtime_model = xgb.XGBRegressor()
        self.runtime_model.load_model(path / "runtime_model.json")

        self.thresh_scaler.mean_ = np.load(path / "thresh_scaler_mean.npy")
        self.thresh_scaler.scale_ = np.load(path / "thresh_scaler_scale.npy")
        self.runtime_scaler.mean_ = np.load(path / "runtime_scaler_mean.npy")
        self.runtime_scaler.scale_ = np.load(path / "runtime_scaler_scale.npy")
        self.thresh_feature_idx = np.load(path / "thresh_feature_idx.npy")
        self.runtime_feature_idx = np.load(path / "runtime_feature_idx.npy")

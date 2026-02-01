"""
Ridge Regression baseline for threshold classification and runtime regression.
Uses significantly fewer parameters than tree-based models (~50 vs ~3000).
"""

from typing import Dict, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import joblib
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER
from models.base import BaseModel


class XGBoostModel(BaseModel):
    """
    A lightweight Ridge Regression model.
    
    Instead of learning complex non-linear trees, this model learns 
    a single weight for each feature. It relies on the hypothesis that:
    1. Log-Runtime scales linearly with complexity features (span, depth).
    2. Threshold scales linearly with 'entanglement pressure' (cuts, crossings).
    """

    def __init__(
        self,
        alpha: float = 1.0,  # L2 Regularization strength
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.random_state = random_state
        
        # We use Ridge because it handles correlated features (like n_cx vs n_2q_gates)
        # better than standard LinearRegression.
        self.threshold_model: Optional[Ridge] = None
        self.runtime_model: Optional[Ridge] = None
        self.scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "LinearBaseline"

    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standard extraction pipeline matching XGBoost/LightGBM implementation."""
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
        """Round continuous regression output to nearest valid threshold class index."""
        return np.clip(np.round(pred), 0, len(THRESHOLD_LADDER) - 1).astype(int)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        X_train, y_thresh_train, y_runtime_train = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val = self._extract_data(val_loader)

        # Scaling is CRITICAL for linear models to ensure regularization works evenly
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        if verbose:
            print(f"Training Ridge Regressors (alpha={self.alpha})...")

        # 1. Train Threshold Predictor (Regression -> Rounding)
        self.threshold_model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.threshold_model.fit(X_train_scaled, y_thresh_train)

        # 2. Train Runtime Predictor (Predicting log_runtime)
        self.runtime_model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.runtime_model.fit(X_train_scaled, y_runtime_train)

        # Evaluation
        train_metrics = self._evaluate_internal(X_train_scaled, y_thresh_train, y_runtime_train)
        val_metrics = self._evaluate_internal(X_val_scaled, y_thresh_val, y_runtime_val)

        if verbose:
            print(f"Train Threshold Acc: {train_metrics['threshold_accuracy']:.4f} | "
                  f"Train Runtime MSE: {train_metrics['runtime_mse']:.4f}")
            print(f"Val Threshold Acc: {val_metrics['threshold_accuracy']:.4f} | "
                  f"Val Runtime MSE: {val_metrics['runtime_mse']:.4f}")

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

        # Threshold Prediction
        thresh_pred_raw = self.threshold_model.predict(X_scaled)
        thresh_classes = self._round_to_class(thresh_pred_raw)
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])

        # Runtime Prediction (Inverse Log)
        runtime_log = self.runtime_model.predict(X_scaled)
        runtime_values = np.expm1(runtime_log)

        return thresh_values, runtime_values

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        """
        For linear models, feature importance is the absolute value of coefficients.
        This directly tells you which features drive the prediction up or down.
        """
        if self.threshold_model is None or self.runtime_model is None:
            return None
        return {
            "threshold": np.abs(self.threshold_model.coef_),
            "runtime": np.abs(self.runtime_model.coef_),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.threshold_model, path / "threshold_model.joblib")
        joblib.dump(self.runtime_model, path / "runtime_model.joblib")

        np.save(path / "scaler_mean.npy", self.scaler.mean_)
        np.save(path / "scaler_scale.npy", self.scaler.scale_)

    def load(self, path: Path) -> None:
        path = Path(path)

        self.threshold_model = joblib.load(path / "threshold_model.joblib")
        self.runtime_model = joblib.load(path / "runtime_model.joblib")

        self.scaler.mean_ = np.load(path / "scaler_mean.npy")
        self.scaler.scale_ = np.load(path / "scaler_scale.npy")
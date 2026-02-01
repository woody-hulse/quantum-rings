"""
Base classes for gradient boosting models (XGBoost, CatBoost, LightGBM).

Eliminates code duplication by providing common data extraction, evaluation,
and prediction logic.
"""

from abc import abstractmethod
from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.base import BaseModel, ThresholdClassBaseModel
from data_loader import THRESHOLD_FEATURE_IDX
from scoring import NUM_THRESHOLD_CLASSES, select_threshold_class_by_expected_score, mean_threshold_score


class GradientBoostingRegressionModel(BaseModel):
    """
    Base class for gradient boosting regression models.
    
    Subclasses only need to implement:
    - _create_regressor() -> regressor instance
    - _get_library_name() -> str for file extensions
    - name property
    - save/load for library-specific serialization
    """

    def __init__(self):
        self.runtime_model = None
        self.scaler = StandardScaler()

    @abstractmethod
    def _create_regressor(self, **params) -> Any:
        """Create and return the underlying regressor model."""
        pass

    @abstractmethod
    def _get_model_params(self) -> Dict[str, Any]:
        """Return parameters for the regressor."""
        pass

    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and log2_runtime targets from a data loader."""
        all_features = []
        all_runtime: List[float] = []
        for batch in loader:
            all_features.append(batch["features"].numpy())
            all_runtime.extend(batch["log2_runtime"].tolist())
        X = np.vstack(all_features)
        y_runtime = np.array(all_runtime)
        return X, y_runtime

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        X_train, y_runtime_train = self._extract_data(train_loader)
        X_val, y_runtime_val = self._extract_data(val_loader)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        if verbose:
            print("Training runtime regressor...")

        self.runtime_model = self._create_regressor(**self._get_model_params())
        self._fit_model(X_train_scaled, y_runtime_train, X_val_scaled, y_runtime_val)

        train_metrics = self._evaluate_internal(X_train_scaled, y_runtime_train)
        val_metrics = self._evaluate_internal(X_val_scaled, y_runtime_val)

        if verbose:
            print(f"Train Runtime MSE: {train_metrics['runtime_mse']:.4f} | "
                  f"Val Runtime MSE: {val_metrics['runtime_mse']:.4f}")

        return {"train": train_metrics, "val": val_metrics}

    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit the model. Override for library-specific fit behavior."""
        self.runtime_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    def _evaluate_internal(self, X: np.ndarray, y_runtime: np.ndarray) -> Dict[str, float]:
        """Evaluate predictions against ground truth."""
        runtime_pred = self.runtime_model.predict(X)
        return {
            "runtime_mse": mean_squared_error(y_runtime, runtime_pred),
            "runtime_mae": mean_absolute_error(y_runtime, runtime_pred),
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        X, y_runtime = self._extract_data(loader)
        X_scaled = self.scaler.transform(X)
        return self._evaluate_internal(X_scaled, y_runtime)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.runtime_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(features)
        runtime_pred = self.runtime_model.predict(X_scaled)
        runtime_values = np.power(2.0, runtime_pred)
        threshold_values = np.round(np.power(2.0, features[:, THRESHOLD_FEATURE_IDX])).astype(int)
        return threshold_values, runtime_values

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        if self.runtime_model is None:
            return None
        return {"runtime": self.runtime_model.feature_importances_}

    def _save_scaler(self, path: Path) -> None:
        """Save scaler state to path."""
        np.save(path / "scaler_mean.npy", self.scaler.mean_)
        np.save(path / "scaler_scale.npy", self.scaler.scale_)

    def _load_scaler(self, path: Path) -> None:
        """Load scaler state from path."""
        self.scaler.mean_ = np.load(path / "scaler_mean.npy")
        self.scaler.scale_ = np.load(path / "scaler_scale.npy")


class GradientBoostingClassificationModel(ThresholdClassBaseModel):
    """
    Base class for gradient boosting threshold-class classification models.
    
    Subclasses only need to implement:
    - _create_classifier() -> classifier instance
    - _get_library_name() -> str for file extensions
    - name property
    - save/load for library-specific serialization
    """

    def __init__(self, use_class_weights: bool = True, conservative_bias: float = 0.0):
        """
        Args:
            use_class_weights: If True, compute class weights from training data
                to address class imbalance.
            conservative_bias: Bias towards higher threshold classes during
                selection. Range [0, 1]. Higher = more conservative.
        """
        self.classifier = None
        self.scaler = StandardScaler()
        self.use_class_weights = use_class_weights
        self.conservative_bias = conservative_bias
        self._class_weights = None

    @abstractmethod
    def _create_classifier(self, **params) -> Any:
        """Create and return the underlying classifier model."""
        pass

    @abstractmethod
    def _get_model_params(self) -> Dict[str, Any]:
        """Return parameters for the classifier."""
        pass

    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and threshold class targets from a data loader."""
        all_features = []
        all_class: List[int] = []
        for batch in loader:
            all_features.append(batch["features"].numpy())
            all_class.extend(batch["threshold_class"].tolist())
        X = np.vstack(all_features)
        y = np.array(all_class, dtype=np.int64)
        return X, y

    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute class weights inversely proportional to class frequencies."""
        unique_classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = NUM_THRESHOLD_CLASSES
        
        weights = np.ones(n_classes)
        for cls, count in zip(unique_classes, counts):
            weights[cls] = n_samples / (n_classes * count)
        
        weights = weights / weights.sum() * n_classes
        return weights

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        X_train, y_train = self._extract_data(train_loader)
        X_val, y_val = self._extract_data(val_loader)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        if self.use_class_weights:
            self._class_weights = self._compute_class_weights(y_train)
            if verbose:
                print(f"Class weights: {self._class_weights}")

        if verbose:
            print("Training threshold-class classifier...")

        self.classifier = self._create_classifier(**self._get_model_params())
        self._fit_model(X_train_scaled, y_train, X_val_scaled, y_val)

        train_metrics = self._evaluate_from_arrays(X_train_scaled, y_train)
        val_metrics = self._evaluate_from_arrays(X_val_scaled, y_val)

        if verbose:
            print(f"Train threshold score: {train_metrics['expected_threshold_score']:.4f} | "
                  f"Val: {val_metrics['expected_threshold_score']:.4f}")

        return {"train": train_metrics, "val": val_metrics}

    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit the model. Override for library-specific fit behavior."""
        self.classifier.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    def _evaluate_from_arrays(self, X: np.ndarray, y_class: np.ndarray) -> Dict[str, float]:
        """Evaluate predictions against ground truth."""
        proba = self.classifier.predict_proba(X)
        chosen = select_threshold_class_by_expected_score(proba, conservative_bias=self.conservative_bias)
        
        n_underpred = np.sum(chosen < y_class)
        n_overpred = np.sum(chosen > y_class)
        
        return {
            "threshold_accuracy": float(np.mean(chosen == y_class)),
            "expected_threshold_score": mean_threshold_score(chosen, y_class),
            "underpred_rate": float(n_underpred / len(y_class)),
            "overpred_rate": float(n_overpred / len(y_class)),
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        X, y = self._extract_data(loader)
        X_scaled = self.scaler.transform(X)
        return self._evaluate_from_arrays(X_scaled, y)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(features)
        return self.classifier.predict_proba(X_scaled)

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        if self.classifier is None:
            return None
        return {"threshold_class": self.classifier.feature_importances_}

    def _save_scaler(self, path: Path) -> None:
        """Save scaler state to path."""
        np.save(path / "scaler_mean.npy", self.scaler.mean_)
        np.save(path / "scaler_scale.npy", self.scaler.scale_)

    def _load_scaler(self, path: Path) -> None:
        """Load scaler state from path."""
        self.scaler.mean_ = np.load(path / "scaler_mean.npy")
        self.scaler.scale_ = np.load(path / "scaler_scale.npy")

"""
Random forest for threshold-class prediction: all features except duration and threshold, output P(class).
"""

from typing import Dict, Any
from pathlib import Path

import numpy as np
import joblib

from .gradient_boosting_base import GradientBoostingClassificationModel
from scoring import NUM_THRESHOLD_CLASSES

from sklearn.ensemble import RandomForestClassifier


class RandomForestThresholdClassModel(GradientBoostingClassificationModel):
    """Random forest for threshold-class prediction: features without duration and threshold, output P(class)."""

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
        use_class_weights: bool = True,
        conservative_bias: float = 0.0,
    ):
        super().__init__(use_class_weights=use_class_weights, conservative_bias=conservative_bias)
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
        return "RandomForestThresholdClass"

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

    def _create_classifier(self, **params) -> Any:
        return RandomForestClassifier(**params)

    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        unique_classes = set(y_train)
        missing_classes = set(range(NUM_THRESHOLD_CLASSES)) - unique_classes

        sample_weight = None
        if self.use_class_weights and self._class_weights is not None:
            sample_weight = np.array([self._class_weights[int(y)] for y in y_train])

        if missing_classes:
            dummy_X = np.zeros((len(missing_classes), X_train.shape[1]))
            dummy_y = np.array(list(missing_classes), dtype=np.int64)
            X_train = np.vstack([X_train, dummy_X])
            y_train = np.concatenate([y_train, dummy_y])
            if sample_weight is not None:
                dummy_weights = np.full(len(missing_classes), 0.001)
                sample_weight = np.concatenate([sample_weight, dummy_weights])

        self.classifier.fit(X_train, y_train, sample_weight=sample_weight)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X_scaled = self.scaler.transform(features)
        proba = self.classifier.predict_proba(X_scaled)
        if proba.shape[1] < NUM_THRESHOLD_CLASSES:
            full_proba = np.zeros((proba.shape[0], NUM_THRESHOLD_CLASSES), dtype=proba.dtype)
            for i, c in enumerate(self.classifier.classes_):
                full_proba[:, c] = proba[:, i]
            return full_proba
        return proba

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, path / "classifier.joblib")
        self._save_scaler(path)

    def load(self, path: Path) -> None:
        path = Path(path)
        self.classifier = joblib.load(path / "classifier.joblib")
        self._load_scaler(path)

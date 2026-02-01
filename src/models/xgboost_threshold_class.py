"""
XGBoost for threshold-class prediction: all features except duration and threshold, output P(class).
"""

from typing import Dict, Any
from pathlib import Path

import numpy as np

from .gradient_boosting_base import GradientBoostingClassificationModel
from scoring import NUM_THRESHOLD_CLASSES

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class XGBoostThresholdClassModel(GradientBoostingClassificationModel):
    """XGBoost for threshold-class prediction: features without duration and threshold, output P(class)."""

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        use_class_weights: bool = True,
        conservative_bias: float = 0.0,
    ):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        super().__init__(use_class_weights=use_class_weights, conservative_bias=conservative_bias)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "XGBoostThresholdClass"

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            "objective": "multi:softprob",
            "num_class": NUM_THRESHOLD_CLASSES,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
        }

    def _create_classifier(self, **params) -> Any:
        return xgb.XGBClassifier(**params)

    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        XGBoost infers num_class from training labels. If training doesn't have
        all 9 classes, predict_proba returns fewer columns causing shape mismatches.
        Fix: add dummy samples for missing classes so XGBoost sees all 9.
        """
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
        
        self.classifier.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(str(path / "classifier.json"))
        self._save_scaler(path)

    def load(self, path: Path) -> None:
        path = Path(path)
        self.classifier = xgb.XGBClassifier()
        self.classifier.load_model(str(path / "classifier.json"))
        self._load_scaler(path)

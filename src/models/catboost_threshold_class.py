"""
CatBoost for threshold-class prediction: all features except duration and threshold, output P(class).
"""

from typing import Dict, Any
from pathlib import Path

import numpy as np

from .gradient_boosting_base import GradientBoostingClassificationModel
from scoring import NUM_THRESHOLD_CLASSES

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class CatBoostThresholdClassModel(GradientBoostingClassificationModel):
    """CatBoost for threshold-class prediction: features without duration and threshold, output P(class)."""

    def __init__(
        self,
        depth: int = 6,
        learning_rate: float = 0.1,
        iterations: int = 100,
        l2_leaf_reg: float = 3.0,
        random_state: int = 42,
        verbose: bool = False,
        use_class_weights: bool = True,
        conservative_bias: float = 0.0,
    ):
        if not HAS_CATBOOST:
            raise ImportError("catboost is required. Install with: pip install catboost")
        super().__init__(use_class_weights=use_class_weights, conservative_bias=conservative_bias)
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "CatBoostThresholdClass"

    def _get_model_params(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "l2_leaf_reg": self.l2_leaf_reg,
            "loss_function": "MultiClass",
            "classes_count": NUM_THRESHOLD_CLASSES,
            "random_seed": self.random_state,
            "verbose": self.verbose,
            "allow_writing_files": False,
        }

    def _create_classifier(self, **params) -> Any:
        return CatBoostClassifier(**params)

    def _fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        CatBoost respects classes_count, but add dummy samples for safety
        to ensure predict_proba always returns (n_samples, 9).
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
        
        self.classifier.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=(X_val, y_val),
            verbose=False,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(str(path / "classifier.cbm"))
        self._save_scaler(path)

    def load(self, path: Path) -> None:
        path = Path(path)
        self.classifier = CatBoostClassifier()
        self.classifier.load_model(str(path / "classifier.cbm"))
        self._load_scaler(path)

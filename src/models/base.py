"""
Abstract base class defining the common interface for all models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader


class BaseModel(ABC):
    """Abstract base class for threshold/runtime prediction models."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name for reporting."""
        pass
    
    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            verbose: Whether to print training progress
            show_progress: Whether to show epoch-level progress bar
            
        Returns:
            Dictionary containing training metrics/history
        """
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on features.

        Args:
            features: Input features array of shape (n_samples, n_features).
                For duration-only formulation, features include threshold (e.g. log2(threshold)).

        Returns:
            Tuple of (threshold_values, runtime_values):
                - threshold_values: For duration-only, the input threshold per sample (from features).
                    Otherwise predicted threshold ladder values (1, 2, 4, ..., 256).
                - runtime_values: Predicted forward wall time in seconds.
        """
        pass

    @abstractmethod
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a data loader.

        Args:
            loader: Data loader to evaluate on. For duration-only, batch has log2_runtime; no threshold_class.

        Returns:
            Dictionary containing evaluation metrics:
                - runtime_mae: MAE in log2 runtime (primary metric for duration-only)
                - runtime_mse: MSE in log2 runtime (optional)
                - threshold_accuracy: (optional) Only for non-duration models
        """
        pass
    
    def save(self, path: Path) -> None:
        """Save the model to disk. Override in subclasses if needed."""
        raise NotImplementedError(f"{self.name} does not support saving")
    
    def load(self, path: Path) -> None:
        """Load the model from disk. Override in subclasses if needed."""
        raise NotImplementedError(f"{self.name} does not support loading")
    
    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        """Return feature importance if available. Override in subclasses."""
        return None


class ThresholdClassBaseModel(ABC):
    """
    Abstract base for threshold-class models: predict P(class), select by max expected score.
    Consumes features without duration and without threshold; outputs probability distribution over classes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return (n_samples, num_classes) probability distribution over threshold classes."""
        pass

    @abstractmethod
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Return dict with threshold_accuracy, expected_threshold_score, etc."""
        pass

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Choose threshold class by max expected score; return (threshold_values, placeholder_runtime)."""
        from data_loader import THRESHOLD_LADDER
        from scoring import select_threshold_class_by_expected_score
        proba = self.predict_proba(features)
        chosen = select_threshold_class_by_expected_score(proba)
        threshold_values = np.array([THRESHOLD_LADDER[c] for c in chosen])
        runtime_values = np.ones_like(threshold_values, dtype=float)
        return threshold_values, runtime_values

    def save(self, path: Path) -> None:
        raise NotImplementedError(f"{self.name} does not support saving")

    def load(self, path: Path) -> None:
        raise NotImplementedError(f"{self.name} does not support loading")

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        return None

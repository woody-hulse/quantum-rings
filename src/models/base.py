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
            features: Input features array of shape (n_samples, n_features)
            
        Returns:
            Tuple of (threshold_values, runtime_values):
                - threshold_values: Predicted threshold ladder values (1, 2, 4, ..., 256)
                - runtime_values: Predicted forward wall time in seconds
        """
        pass
    
    @abstractmethod
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a data loader.
        
        Args:
            loader: Data loader to evaluate on
            
        Returns:
            Dictionary containing evaluation metrics:
                - threshold_accuracy: Accuracy of threshold class predictions
                - runtime_mse: Mean squared error of log runtime
                - runtime_mae: Mean absolute error of log runtime
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

"""
Model-agnostic ensemble for duration prediction.

Threshold is an input (pass-through). Runtime predictions are averaged across members.
"""

from typing import Dict, List, Tuple, Any, Optional, Type
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.base import BaseModel


class EnsembleModel(BaseModel):
    """
    Model-agnostic ensemble: threshold is pass-through (input), runtime is averaged.
    """

    def __init__(
        self,
        models: List[BaseModel],
        inference_strategy: str = "vote",
        softmax_temperature: float = 1.0,
    ):
        """
        Args:
            models: List of trained models to ensemble
            inference_strategy: Unused (kept for API compatibility)
            softmax_temperature: Unused (kept for API compatibility)
        """
        if len(models) == 0:
            raise ValueError("Ensemble requires at least one model")

        self.models = models
        self.inference_strategy = inference_strategy
        self.softmax_temperature = softmax_temperature

    @property
    def name(self) -> str:
        model_names = set(m.name for m in self.models)
        if len(model_names) == 1:
            return f"Ensemble({next(iter(model_names))}×{len(self.models)})"
        return f"Ensemble({'+'.join(model_names)})"

    def _get_member_predictions(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get (threshold, runtime) from each member. Threshold is input pass-through."""
        all_thresh = []
        all_runtime = []
        for model in self.models:
            thresh_values, runtime_values = model.predict(features)
            all_thresh.append(thresh_values)
            all_runtime.append(runtime_values)
        return np.array(all_thresh), np.array(all_runtime)

    def predict(
        self,
        features: np.ndarray,
        strategy: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (threshold_values, runtime_values). Threshold from input (first member), runtime = mean of members.
        """
        member_thresh, member_runtime = self._get_member_predictions(features)
        thresh_values = member_thresh[0]
        runtime_values = member_runtime.mean(axis=0)
        return thresh_values, runtime_values

    def predict_with_uncertainty(
        self,
        features: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Predict with runtime std across members. Threshold is pass-through."""
        member_thresh, member_runtime = self._get_member_predictions(features)
        return {
            "threshold_values": member_thresh[0],
            "runtime_values": member_runtime.mean(axis=0),
            "runtime_std": member_runtime.std(axis=0),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit is a no-op for pre-trained ensembles.
        Use EnsembleTrainer to train members from scratch.
        """
        return {"note": "Ensemble uses pre-trained models"}

    def evaluate(
        self,
        loader: DataLoader,
        strategy: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate on a data loader (log2 runtime MSE/MAE)."""
        all_runtime_preds = []
        all_runtime_labels = []
        for batch in loader:
            features = batch["features"].numpy()
            log2_runtime = batch["log2_runtime"].numpy()
            thresh_values, runtime_values = self.predict(features, strategy=strategy)
            all_runtime_preds.extend(np.log2(np.maximum(runtime_values, 1e-10)).tolist())
            all_runtime_labels.extend(log2_runtime.tolist())
        return {
            "runtime_mse": mean_squared_error(all_runtime_labels, all_runtime_preds),
            "runtime_mae": mean_absolute_error(all_runtime_labels, all_runtime_preds),
        }


class EnsembleTrainer:
    """
    Trains multiple models to create an ensemble.

    This is model-agnostic: it can train any model class that implements BaseModel.
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        n_members: int = 5,
        inference_strategy: str = "decision_theoretic",
        **model_kwargs,
    ):
        """
        Args:
            model_class: The model class to instantiate (e.g., MLPModel, XGBoostModel)
            n_members: Number of ensemble members
            inference_strategy: Inference strategy for the ensemble
            **model_kwargs: Arguments to pass to model constructor
        """
        self.model_class = model_class
        self.n_members = n_members
        self.inference_strategy = inference_strategy
        self.model_kwargs = model_kwargs
        self.ensemble: Optional[EnsembleModel] = None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> EnsembleModel:
        """
        Train all ensemble members and return the ensemble.

        Each member is trained with a different random seed for diversity.
        """
        models = []

        for i in range(self.n_members):
            if verbose:
                print(f"\nTraining ensemble member {i+1}/{self.n_members}")

            # Create model with unique seed
            kwargs = self.model_kwargs.copy()

            # Set random seed for diversity
            seed = kwargs.get("seed", 42) + i * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = self.model_class(**kwargs)
            model.fit(train_loader, val_loader, verbose=False, show_progress=show_progress)
            models.append(model)

        self.ensemble = EnsembleModel(
            models=models,
            inference_strategy=self.inference_strategy,
        )

        return self.ensemble


def create_ensemble_from_models(
    models: List[BaseModel],
    inference_strategy: str = "decision_theoretic",
) -> EnsembleModel:
    """Convenience function to create an ensemble from pre-trained models."""
    return EnsembleModel(models=models, inference_strategy=inference_strategy)

"""
Model-agnostic ensemble for threshold classification and runtime regression.

Supports two modes for threshold prediction:
1. Vote-based: Use empirical distribution from member predictions
2. Logit-based: Average logits/probabilities (only for models that support it)

Both modes support decision-theoretic inference.
"""

from typing import Dict, List, Tuple, Any, Optional, Type
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

from data_loader import THRESHOLD_LADDER
from models.base import BaseModel


def get_score_matrix(num_classes: int = 9) -> np.ndarray:
    """Build the challenge score matrix."""
    score_matrix = np.zeros((num_classes, num_classes))
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if pred_idx < true_idx:
                score_matrix[true_idx, pred_idx] = 0.0
            else:
                steps_over = pred_idx - true_idx
                score_matrix[true_idx, pred_idx] = 2.0 ** (-steps_over)
    return score_matrix


class EnsembleModel(BaseModel):
    """
    Model-agnostic ensemble that combines predictions from multiple models.

    For threshold prediction, builds an empirical probability distribution from
    member predictions and optionally applies decision-theoretic inference.

    For runtime prediction, averages the predictions from all members.
    """

    INFERENCE_ARGMAX = "argmax"
    INFERENCE_VOTE = "vote"  # Majority vote (equivalent to argmax on empirical dist)
    INFERENCE_DECISION_THEORETIC = "decision_theoretic"

    def __init__(
        self,
        models: List[BaseModel],
        inference_strategy: str = "vote",
        softmax_temperature: float = 1.0,
    ):
        """
        Args:
            models: List of trained models to ensemble
            inference_strategy: How to combine threshold predictions
                - "vote": Majority vote among members
                - "decision_theoretic": Maximize expected score using empirical distribution
            softmax_temperature: Temperature for softening empirical distribution
                                 (only used with decision_theoretic)
        """
        if len(models) == 0:
            raise ValueError("Ensemble requires at least one model")

        self.models = models
        self.inference_strategy = inference_strategy
        self.softmax_temperature = softmax_temperature
        self.num_classes = len(THRESHOLD_LADDER)
        self.score_matrix = get_score_matrix(self.num_classes)

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
        """Get predictions from all ensemble members.

        Returns:
            Tuple of:
                - threshold_preds: (n_members, n_samples) array of threshold class indices
                - runtime_preds: (n_members, n_samples) array of runtime predictions
        """
        all_thresh = []
        all_runtime = []

        for model in self.models:
            thresh_values, runtime_values = model.predict(features)
            # Convert threshold values back to class indices
            thresh_classes = np.array([
                THRESHOLD_LADDER.index(t) if t in THRESHOLD_LADDER else 0
                for t in thresh_values
            ])
            all_thresh.append(thresh_classes)
            all_runtime.append(runtime_values)

        return np.array(all_thresh), np.array(all_runtime)

    def _build_empirical_distribution(
        self,
        member_preds: np.ndarray,
    ) -> np.ndarray:
        """Build empirical probability distribution from member predictions.

        Args:
            member_preds: (n_members, n_samples) array of class predictions

        Returns:
            (n_samples, num_classes) array of empirical probabilities
        """
        n_members, n_samples = member_preds.shape
        probs = np.zeros((n_samples, self.num_classes))

        for i in range(n_samples):
            for j in range(n_members):
                pred_class = member_preds[j, i]
                probs[i, pred_class] += 1.0 / n_members

        return probs

    def _decision_theoretic_predict(
        self,
        probs: np.ndarray,
    ) -> np.ndarray:
        """Pick classes that maximize expected challenge score.

        Args:
            probs: (n_samples, num_classes) probability distribution

        Returns:
            (n_samples,) array of predicted class indices
        """
        # Apply temperature softening if not 1.0
        if self.softmax_temperature != 1.0:
            log_probs = np.log(probs + 1e-10) / self.softmax_temperature
            probs = np.exp(log_probs) / np.exp(log_probs).sum(axis=1, keepdims=True)

        # expected_scores[i, j] = expected score if we predict class j for sample i
        expected_scores = probs @ self.score_matrix

        return expected_scores.argmax(axis=1)

    def predict(
        self,
        features: np.ndarray,
        strategy: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the ensemble.

        Args:
            features: Input features array
            strategy: Override inference strategy

        Returns:
            Tuple of (threshold_values, runtime_values)
        """
        strategy = strategy or self.inference_strategy

        # Get all member predictions
        member_thresh, member_runtime = self._get_member_predictions(features)

        # Average runtime predictions
        runtime_values = member_runtime.mean(axis=0)

        # Build empirical distribution for threshold
        probs = self._build_empirical_distribution(member_thresh)

        # Apply inference strategy
        if strategy == self.INFERENCE_DECISION_THEORETIC:
            thresh_classes = self._decision_theoretic_predict(probs)
        else:  # vote / argmax
            thresh_classes = probs.argmax(axis=1)

        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])

        return thresh_values, runtime_values

    def predict_with_uncertainty(
        self,
        features: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions and return uncertainty estimates.

        Returns:
            Dict with:
                - threshold_values: Predicted thresholds
                - runtime_values: Predicted runtimes
                - threshold_probs: Empirical probability distribution
                - runtime_std: Standard deviation of runtime predictions
                - member_agreement: Fraction of members agreeing with majority
        """
        member_thresh, member_runtime = self._get_member_predictions(features)

        probs = self._build_empirical_distribution(member_thresh)
        thresh_classes = probs.argmax(axis=1)
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])
        runtime_values = member_runtime.mean(axis=0)

        return {
            "threshold_values": thresh_values,
            "runtime_values": runtime_values,
            "threshold_probs": probs,
            "runtime_std": member_runtime.std(axis=0),
            "member_agreement": probs.max(axis=1),
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
        """Evaluate the ensemble on a data loader."""
        all_thresh_preds = []
        all_thresh_labels = []
        all_runtime_preds = []
        all_runtime_labels = []

        for batch in loader:
            features = batch["features"].numpy()
            threshold_labels = batch["threshold_class"].tolist()
            runtime_labels = batch["log_runtime"].numpy()

            thresh_values, runtime_values = self.predict(features, strategy=strategy)
            thresh_classes = [
                THRESHOLD_LADDER.index(t) if t in THRESHOLD_LADDER else 0
                for t in thresh_values
            ]

            all_thresh_preds.extend(thresh_classes)
            all_thresh_labels.extend(threshold_labels)
            all_runtime_preds.extend(np.log1p(runtime_values).tolist())
            all_runtime_labels.extend(runtime_labels.tolist())

        return {
            "threshold_accuracy": accuracy_score(all_thresh_labels, all_thresh_preds),
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

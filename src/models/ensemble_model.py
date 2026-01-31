"""
Ensemble model that combines predictions from multiple base models.
"""

from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER
from models.base import BaseModel


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines predictions from multiple base models.

    Supports:
    - Simple averaging (default)
    - Weighted averaging (if weights provided)
    - Threshold voting (majority vote for threshold, average for runtime)
    """

    def __init__(
        self,
        models: List[BaseModel],
        weights: Optional[List[float]] = None,
        strategy: str = "average",  # "average", "weighted_average", or "vote"
    ):
        """
        Args:
            models: List of trained BaseModel instances
            weights: Optional weights for weighted averaging (must sum to 1.0)
            strategy: Ensemble strategy - "average", "weighted_average", or "vote"
        """
        if not models:
            raise ValueError("Must provide at least one model")

        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")

        self.models = models
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        self.strategy = strategy

    @property
    def name(self) -> str:
        model_names = [m.name for m in self.models]
        return f"Ensemble({'+'.join(model_names)})"

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Train all base models in the ensemble.

        Note: If models are already trained, this will retrain them.
        """
        all_train_metrics = []
        all_val_metrics = []

        for i, model in enumerate(self.models):
            if verbose:
                print(f"Training model {i+1}/{len(self.models)}: {model.name}")

            result = model.fit(train_loader, val_loader, verbose=False)

            if isinstance(result, dict) and "train" in result and "val" in result:
                all_train_metrics.append(result["train"])
                all_val_metrics.append(result["val"])

        # Aggregate metrics across all base models
        train_metrics = self._aggregate_metrics(all_train_metrics)
        val_metrics = self._aggregate_metrics(all_val_metrics)

        if verbose:
            print(f"\nEnsemble Train Metrics: {train_metrics}")
            print(f"Ensemble Val Metrics: {val_metrics}")

        return {"train": train_metrics, "val": val_metrics}

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across models (take mean)."""
        if not metrics_list:
            return {}

        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = float(np.mean(values))

        return aggregated

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions by combining base model predictions.

        Args:
            features: Input features array of shape (n_samples, n_features)

        Returns:
            Tuple of (threshold_values, runtime_values)
        """
        all_thresh_preds = []
        all_runtime_preds = []

        for model in self.models:
            thresh, runtime = model.predict(features)
            all_thresh_preds.append(thresh)
            all_runtime_preds.append(runtime)

        # Combine predictions based on strategy
        if self.strategy == "vote":
            # Majority vote for threshold
            thresh_preds = self._majority_vote(all_thresh_preds)
        else:
            # Weighted average for threshold (convert to indices, average, round back)
            thresh_indices = []
            for thresh_pred in all_thresh_preds:
                indices = np.array([THRESHOLD_LADDER.index(t) for t in thresh_pred])
                thresh_indices.append(indices)

            # Weighted average of indices
            avg_indices = np.zeros_like(thresh_indices[0], dtype=float)
            for idx, weight in zip(thresh_indices, self.weights):
                avg_indices += idx * weight

            # Round to nearest valid index
            rounded_indices = np.clip(np.round(avg_indices), 0, len(THRESHOLD_LADDER) - 1).astype(int)
            thresh_preds = np.array([THRESHOLD_LADDER[i] for i in rounded_indices])

        # Weighted average for runtime
        runtime_preds = np.zeros_like(all_runtime_preds[0])
        for runtime_pred, weight in zip(all_runtime_preds, self.weights):
            runtime_preds += runtime_pred * weight

        return thresh_preds, runtime_preds

    def _majority_vote(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Compute majority vote across predictions."""
        n_samples = len(predictions[0])
        result = np.zeros(n_samples, dtype=predictions[0].dtype)

        for i in range(n_samples):
            votes = [pred[i] for pred in predictions]
            # Count occurrences and pick most common
            unique, counts = np.unique(votes, return_counts=True)
            result[i] = unique[np.argmax(counts)]

        return result

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the ensemble on a data loader.

        Args:
            loader: Data loader to evaluate on

        Returns:
            Dictionary containing evaluation metrics
        """
        # Extract features and labels
        all_features = []
        all_thresh = []
        all_runtime = []

        for batch in loader:
            all_features.append(batch["features"].numpy())
            thresh_classes = batch["threshold_class"].tolist()
            thresh_values = [THRESHOLD_LADDER[c] for c in thresh_classes]
            all_thresh.extend(thresh_values)
            all_runtime.extend(np.expm1(batch["log_runtime"].numpy()).tolist())

        X = np.vstack(all_features)
        y_thresh = np.array(all_thresh)
        y_runtime = np.array(all_runtime)

        # Make predictions
        pred_thresh, pred_runtime = self.predict(X)

        # Convert to class indices for accuracy calculation
        y_thresh_classes = np.array([THRESHOLD_LADDER.index(t) for t in y_thresh])
        pred_thresh_classes = np.array([THRESHOLD_LADDER.index(t) for t in pred_thresh])

        # Compute log runtime for MSE/MAE
        y_log_runtime = np.log1p(y_runtime)
        pred_log_runtime = np.log1p(pred_runtime)

        return {
            "threshold_accuracy": accuracy_score(y_thresh_classes, pred_thresh_classes),
            "runtime_mse": mean_squared_error(y_log_runtime, pred_log_runtime),
            "runtime_mae": mean_absolute_error(y_log_runtime, pred_log_runtime),
        }

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Return aggregated feature importance from all models.

        Returns None if not all models support feature importance.
        """
        all_importance = []

        for model in self.models:
            importance = model.get_feature_importance()
            if importance is None:
                return None
            all_importance.append(importance)

        # Average importance across models
        aggregated = {}
        for key in all_importance[0].keys():
            importances = [imp[key] for imp in all_importance if key in imp]
            aggregated[key] = np.mean(importances, axis=0)

        return aggregated

    def save(self, path: Path) -> None:
        """Save all models in the ensemble."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each model in a subdirectory
        for i, model in enumerate(self.models):
            model_path = path / f"model_{i}_{model.name}"
            model.save(model_path)

        # Save ensemble metadata
        import json
        metadata = {
            "n_models": len(self.models),
            "model_names": [m.name for m in self.models],
            "weights": self.weights,
            "strategy": self.strategy,
        }
        with open(path / "ensemble_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: Path) -> None:
        """Load all models in the ensemble."""
        raise NotImplementedError("Ensemble loading requires knowing model types upfront")

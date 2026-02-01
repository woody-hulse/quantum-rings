"""
GNN for threshold-class prediction: all features except duration and threshold, output P(class).
Wraps gnn.model and gnn.train with ThresholdClassBaseModel interface.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from data_loader import THRESHOLD_LADDER
from scoring import select_threshold_class_by_expected_score
from models.base import ThresholdClassBaseModel

from gnn.model import create_gnn_threshold_class_model
from gnn.dataset import NUM_THRESHOLD_CLASSES, GLOBAL_FEAT_DIM_THRESHOLD_CLASS
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from gnn.train import GNNTrainerThresholdClass
from gnn.augmentation import get_train_augmentation


def _is_pyg_loader(loader: Any) -> bool:
    try:
        batch = next(iter(loader))
        return hasattr(batch, "x") and hasattr(batch, "batch") and hasattr(batch, "global_features")
    except StopIteration:
        return False
    except Exception:
        return False


class GNNThresholdClassModel(ThresholdClassBaseModel):
    """
    GNN for threshold-class prediction: graph features without duration and threshold, output P(class).
    fit() and evaluate() expect PyG DataLoaders from create_threshold_class_graph_data_loaders.
    predict_proba(x): x must be a PyG DataLoader (not a raw feature array).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        epochs: int = 100,
        early_stopping_patience: int = 20,
        use_augmentation: bool = True,
        augmentation_strength: float = 0.5,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_augmentation = use_augmentation
        self.augmentation_strength = augmentation_strength
        self.model = create_gnn_threshold_class_model(
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM,
            global_feat_dim=GLOBAL_FEAT_DIM_THRESHOLD_CLASS,
            num_classes=NUM_THRESHOLD_CLASSES,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        augmentation = None
        if use_augmentation:
            augmentation = get_train_augmentation(
                qubit_perm_p=augmentation_strength,
                edge_dropout_p=0.1 * augmentation_strength,
                feature_noise_std=0.1 * augmentation_strength,
                temporal_jitter_std=0.05 * augmentation_strength,
            )
        self.trainer: Optional[GNNTrainerThresholdClass] = None
        self._trainer_augmentation = augmentation

    @property
    def name(self) -> str:
        return "GNNThresholdClass"

    def _get_trainer(self) -> GNNTrainerThresholdClass:
        if self.trainer is None:
            self.trainer = GNNTrainerThresholdClass(
                model=self.model,
                device=self.device,
                lr=self.lr,
                weight_decay=self.weight_decay,
                augmentation=self._trainer_augmentation,
            )
        return self.trainer

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        trainer = self._get_trainer()
        return trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_patience,
            verbose=verbose,
            show_progress=show_progress,
        )

    def predict_proba(self, features: Union[np.ndarray, DataLoader]) -> np.ndarray:
        """For GNN, features must be a PyG DataLoader (from create_threshold_class_graph_data_loaders)."""
        if not _is_pyg_loader(features):
            raise TypeError("GNNThresholdClassModel.predict_proba requires a PyG DataLoader, not a raw feature array.")
        trainer = self._get_trainer()
        return trainer.predict_proba(features)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        trainer = self._get_trainer()
        return trainer.evaluate(loader)

    def predict(self, features: Union[np.ndarray, DataLoader]) -> tuple:
        """For GNN, features must be a PyG DataLoader. Returns (threshold_values, placeholder_runtime)."""
        proba = self.predict_proba(features)
        chosen = select_threshold_class_by_expected_score(proba)
        threshold_values = np.array([THRESHOLD_LADDER[c] for c in chosen])
        runtime_values = np.ones_like(threshold_values, dtype=float)
        return threshold_values, runtime_values

    def save(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "gnn_threshold_class.pt")

    def load(self, path: Path) -> None:
        import torch
        path = Path(path)
        self.model.load_state_dict(torch.load(path / "gnn_threshold_class.pt", map_location=self.device))

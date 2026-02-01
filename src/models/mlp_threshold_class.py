"""
MLP for threshold-class prediction: all features except duration and threshold, output P(class).
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import get_feature_statistics, FEATURE_DIM_WITHOUT_THRESHOLD
from scoring import NUM_THRESHOLD_CLASSES, select_threshold_class_by_expected_score, mean_threshold_score
from models.base import ThresholdClassBaseModel
from models.mlp import build_mlp_encoder


class MLPThresholdClassNetwork(nn.Module):
    """MLP for threshold-class: encoder + classification head (no duration, no threshold in input)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder, enc_dim = build_mlp_encoder(input_dim, hidden_dims, dropout)
        self.class_head = nn.Linear(enc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.class_head(features)


class MLPThresholdClassModel(ThresholdClassBaseModel):
    """
    MLP for threshold-class prediction: features without duration and threshold, output P(class).
    Selection at test time by maximum expected threshold score.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM_WITHOUT_THRESHOLD,
        num_classes: int = NUM_THRESHOLD_CLASSES,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        epochs: int = 150,
        early_stopping_patience: int = 25,
        use_class_weights: bool = True,
        conservative_bias: float = 0.0,
        label_smoothing: float = 0.1,
        grad_clip: float = 1.0,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_class_weights = use_class_weights
        self.conservative_bias = conservative_bias
        self.label_smoothing = label_smoothing
        self.grad_clip = grad_clip
        
        self.network = MLPThresholdClassNetwork(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        self._class_weights: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "MLPThresholdClass"

    def _set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.feature_mean = mean.to(self.device)
        self.feature_std = std.to(self.device)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mean is not None:
            return (x - self.feature_mean) / (self.feature_std + 1e-8)
        return x

    def _compute_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """Compute class weights inversely proportional to class frequencies."""
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch["threshold_class"].tolist())
        
        labels = np.array(all_labels)
        unique_classes, counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        
        weights = np.ones(self.num_classes)
        for cls, count in zip(unique_classes, counts):
            weights[cls] = n_samples / (self.num_classes * count)
        
        weights = weights / weights.sum() * self.num_classes
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.network.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features = batch["features"].to(self.device)
            target = batch["threshold_class"].to(self.device)
            features = self._normalize(features)
            self.optimizer.zero_grad()
            logits = self.network(features)
            loss = self.criterion(logits, target)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return {"loss": total_loss / n_batches}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        mean, std = get_feature_statistics(train_loader)
        self._set_normalization(mean, std)
        
        if self.use_class_weights:
            self._class_weights = self._compute_class_weights(train_loader)
            self.criterion = nn.CrossEntropyLoss(
                weight=self._class_weights,
                label_smoothing=self.label_smoothing,
            )
        
        history = {"train_loss": [], "val_threshold_score": []}
        best_val_score = -1.0
        patience_counter = 0
        best_state = None
        epoch_iter = range(self.epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training", leave=False)
        for epoch in epoch_iter:
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_score = val_metrics["expected_threshold_score"]
            self.scheduler.step(val_score)
            history["train_loss"].append(train_metrics["loss"])
            history["val_threshold_score"].append(val_score)
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                patience_counter += 1
            if show_progress:
                epoch_iter.set_postfix(
                    loss=f"{train_metrics['loss']:.3f}",
                    val_score=f"{val_score:.3f}",
                )
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {train_metrics['loss']:.4f} | Val score: {val_score:.4f}")
            if patience_counter >= self.early_stopping_patience:
                if show_progress:
                    epoch_iter.close()
                break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        return {"history": history}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.network.eval()
        all_proba = []
        all_true = []
        for batch in loader:
            features = batch["features"].to(self.device)
            target = batch["threshold_class"]
            features = self._normalize(features)
            logits = self.network(features).cpu()
            proba = torch.softmax(logits, dim=-1).numpy()
            all_proba.append(proba)
            all_true.extend(target.tolist())
        proba = np.vstack(all_proba)
        true_idx = np.array(all_true, dtype=np.int64)
        chosen = select_threshold_class_by_expected_score(proba, conservative_bias=self.conservative_bias)
        
        n_underpred = np.sum(chosen < true_idx)
        n_overpred = np.sum(chosen > true_idx)
        
        return {
            "threshold_accuracy": float(np.mean(chosen == true_idx)),
            "expected_threshold_score": mean_threshold_score(chosen, true_idx),
            "underpred_rate": float(n_underpred / len(true_idx)),
            "overpred_rate": float(n_overpred / len(true_idx)),
        }

    @torch.no_grad()
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.network.eval()
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        features = features.to(self.device)
        features = self._normalize(features)
        logits = self.network(features).cpu()
        return torch.softmax(logits, dim=-1).numpy()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network_state": self.network.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "config": {
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            },
        }, path / "mlp_threshold_class.pt")

    def load(self, path: Path) -> None:
        path = Path(path)
        ckpt = torch.load(path / "mlp_threshold_class.pt", map_location=self.device)
        self.network.load_state_dict(ckpt["network_state"])
        self.feature_mean = ckpt["feature_mean"]
        self.feature_std = ckpt["feature_std"]

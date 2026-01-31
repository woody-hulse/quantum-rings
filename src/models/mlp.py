"""
MLP model implementation for threshold classification and runtime regression.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER, get_feature_statistics
from models.base import BaseModel


class MLPNetwork(nn.Module):
    """Multi-task MLP for threshold classification and runtime regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        num_threshold_classes: int = 9,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.threshold_head = nn.Linear(prev_dim, num_threshold_classes)
        self.runtime_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        threshold_logits = self.threshold_head(features)
        runtime_pred = self.runtime_head(features).squeeze(-1)
        return threshold_logits, runtime_pred


class MLPModel(BaseModel):
    """MLP model wrapper implementing the common interface."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        epochs: int = 100,
        early_stopping_patience: int = 20,
        threshold_weight: float = 1.0,
        runtime_weight: float = 1.0,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.threshold_weight = threshold_weight
        self.runtime_weight = runtime_weight
        
        self.network = MLPNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_threshold_classes=len(THRESHOLD_LADDER),
            dropout=dropout,
        ).to(device)
        
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.threshold_criterion = nn.CrossEntropyLoss()
        self.runtime_criterion = nn.MSELoss()
        
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
    
    @property
    def name(self) -> str:
        return "MLP"
    
    def _set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.feature_mean = mean.to(self.device)
        self.feature_std = std.to(self.device)
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mean is not None:
            return (x - self.feature_mean) / self.feature_std
        return x
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.network.train()
        total_loss = 0.0
        total_thresh_loss = 0.0
        total_runtime_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            features = batch["features"].to(self.device)
            threshold_labels = batch["threshold_class"].to(self.device)
            runtime_labels = batch["log_runtime"].to(self.device)
            
            features = self._normalize(features)
            
            self.optimizer.zero_grad()
            threshold_logits, runtime_pred = self.network(features)
            
            thresh_loss = self.threshold_criterion(threshold_logits, threshold_labels)
            runtime_loss = self.runtime_criterion(runtime_pred, runtime_labels)
            
            loss = self.threshold_weight * thresh_loss + self.runtime_weight * runtime_loss
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_thresh_loss += thresh_loss.item()
            total_runtime_loss += runtime_loss.item()
            n_batches += 1
        
        return {
            "loss": total_loss / n_batches,
            "threshold_loss": total_thresh_loss / n_batches,
            "runtime_loss": total_runtime_loss / n_batches,
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        mean, std = get_feature_statistics(train_loader)
        self._set_normalization(mean, std)
        
        history = {"train_loss": [], "val_threshold_acc": [], "val_runtime_mse": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            val_loss = (1 - val_metrics["threshold_accuracy"]) + val_metrics["runtime_mse"]
            self.scheduler.step(val_loss)
            
            history["train_loss"].append(train_metrics["loss"])
            history["val_threshold_acc"].append(val_metrics["threshold_accuracy"])
            history["val_runtime_mse"].append(val_metrics["runtime_mse"])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Thresh Acc: {val_metrics['threshold_accuracy']:.4f} | "
                      f"Val Runtime MSE: {val_metrics['runtime_mse']:.4f}")
            
            if patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.network.load_state_dict(best_state)
        
        return {"history": history}
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.network.eval()
        all_thresh_preds = []
        all_thresh_labels = []
        all_runtime_preds = []
        all_runtime_labels = []
        
        for batch in loader:
            features = batch["features"].to(self.device)
            threshold_labels = batch["threshold_class"]
            runtime_labels = batch["log_runtime"]
            
            features = self._normalize(features)
            threshold_logits, runtime_pred = self.network(features)
            
            thresh_preds = threshold_logits.argmax(dim=1).cpu()
            runtime_pred = runtime_pred.cpu()
            
            all_thresh_preds.extend(thresh_preds.tolist())
            all_thresh_labels.extend(threshold_labels.tolist())
            all_runtime_preds.extend(runtime_pred.tolist())
            all_runtime_labels.extend(runtime_labels.tolist())
        
        return {
            "threshold_accuracy": accuracy_score(all_thresh_labels, all_thresh_preds),
            "runtime_mse": mean_squared_error(all_runtime_labels, all_runtime_preds),
            "runtime_mae": mean_absolute_error(all_runtime_labels, all_runtime_preds),
        }
    
    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.network.eval()
        
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        
        features = features.to(self.device)
        features = self._normalize(features)
        threshold_logits, runtime_pred = self.network(features)
        
        thresh_classes = threshold_logits.argmax(dim=1).cpu().numpy()
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])
        runtime_values = np.expm1(runtime_pred.cpu().numpy())
        
        return thresh_values, runtime_values
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "network_state": self.network.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
            }
        }, path / "mlp_model.pt")
    
    def load(self, path: Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path / "mlp_model.pt", map_location=self.device)
        
        self.network.load_state_dict(checkpoint["network_state"])
        self.feature_mean = checkpoint["feature_mean"]
        self.feature_std = checkpoint["feature_std"]

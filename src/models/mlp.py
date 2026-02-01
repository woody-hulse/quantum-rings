"""
MLP model implementation optimized for small datasets and challenge scoring.
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
from losses import ChallengeScoringLoss


class MLPNetwork(nn.Module):
    """Refined Multi-task MLP with Dropout and BatchNorm."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        num_threshold_classes: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
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
    """MLP model with noise injection and Huber loss to prevent overfitting."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.3,
        lr: float = 5e-4,
        weight_decay: float = 1e-2,
        device: str = "cpu",
        epochs: int = 200,
        early_stopping_patience: int = 30,
        threshold_weight: float = 1.0,
        runtime_weight: float = 0.5,
        use_scoring_loss: bool = False,
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
        self.use_scoring_loss = use_scoring_loss
        
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
            self.optimizer, mode='max', factor=0.5, patience=15
        )
        
        if use_scoring_loss:
            self.criterion = ChallengeScoringLoss(
                threshold_weight=threshold_weight,
                runtime_weight=runtime_weight,
                multiplicative=True,
            ).to(device)
        else:
            # Label smoothing prevents the model from being overconfident
            self.threshold_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            # Huber loss is more robust to runtime outliers than MSE
            self.runtime_criterion = nn.HuberLoss(delta=1.0)
        
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
    
    @property
    def name(self) -> str:
        return "MLP_Robust"
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mean is not None:
            x = (x - self.feature_mean) / (self.feature_std + 1e-6)
        
        # Data Augmentation: Add tiny noise during training to stop memorization
        if self.network.training:
            x = x + torch.randn_like(x) * 0.01
        return x
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.network.train()
        total_loss, n_batches = 0.0, 0
        
        for batch in train_loader:
            features = self._normalize(batch["features"].to(self.device))
            threshold_labels = batch["threshold_class"].to(self.device)
            runtime_labels = batch["log_runtime"].to(self.device)
            
            self.optimizer.zero_grad()
            logits, run_pred = self.network(features)
            
            if self.use_scoring_loss:
                loss = self.criterion(logits, run_pred, threshold_labels, runtime_labels)["loss"]
            else:
                t_loss = self.threshold_criterion(logits, threshold_labels)
                r_loss = self.runtime_criterion(run_pred, runtime_labels)
                loss = self.threshold_weight * t_loss + self.runtime_weight * r_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return {"loss": total_loss / n_batches}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = False) -> Dict[str, Any]:
        mean, std = get_feature_statistics(train_loader)
        self.feature_mean, self.feature_std = mean.to(self.device), std.to(self.device)
        
        best_val_acc = -1.0
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            # Primary metric for small data is validation accuracy
            current_acc = val_metrics["threshold_accuracy"]
            self.scheduler.step(current_acc)
            
            if current_acc > best_val_acc:
                best_val_acc = current_acc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1} | Val Acc: {current_acc:.4f} | Val MSE: {val_metrics['runtime_mse']:.4f}")
            
            if patience_counter >= self.early_stopping_patience:
                break
        
        if best_state:
            self.network.load_state_dict(best_state)
        return {"best_val_acc": best_val_acc}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.network.eval()
        y_t_true, y_t_pred, y_r_true, y_r_pred = [], [], [], []
        
        for batch in loader:
            features = self._normalize(batch["features"].to(self.device))
            logits, run_pred = self.network(features)
            
            y_t_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_t_true.extend(batch["threshold_class"].tolist())
            y_r_pred.extend(run_pred.cpu().tolist())
            y_r_true.extend(batch["log_runtime"].tolist())
        
        return {
            "threshold_accuracy": accuracy_score(y_t_true, y_t_pred),
            "runtime_mse": mean_squared_error(y_r_true, y_r_pred),
            "runtime_mae": mean_absolute_error(y_r_true, y_r_pred),
        }

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.network.eval()
        x = torch.as_tensor(features, dtype=torch.float32).to(self.device)
        logits, run_pred = self.network(self._normalize(x))
        
        classes = logits.argmax(dim=1).cpu().numpy()
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in classes])
        runtime_values = np.expm1(run_pred.cpu().numpy())
        
        return thresh_values, runtime_values

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state": self.network.state_dict(),
            "mean": self.feature_mean,
            "std": self.feature_std,
            "config": {"input_dim": self.input_dim, "hidden_dims": self.hidden_dims}
        }, path / "mlp_model.pt")

    def load(self, path: Path) -> None:
        ckpt = torch.load(path / "mlp_model.pt", map_location=self.device)
        self.network.load_state_dict(ckpt["state"])
        self.feature_mean, self.feature_std = ckpt["mean"], ckpt["std"]
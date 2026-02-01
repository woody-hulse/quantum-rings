"""
MLP for duration prediction: threshold as input parameter, predict log2(duration).
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from data_loader import THRESHOLD_FEATURE_IDX, get_feature_statistics
from models.base import BaseModel


def build_mlp_encoder(input_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
    """Shared MLP encoder: Linear->ReLU->BN->Dropout repeated. Returns last hidden dim for head."""
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
    return nn.Sequential(*layers), prev_dim


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = torch.relu(out)
        return out


class MLPNetwork(nn.Module):
    """MLP for duration prediction: threshold as input, output log2(duration)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        
        if use_residual and len(hidden_dims) >= 2:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[0]),
            )
            self.res_blocks = nn.ModuleList([
                ResidualBlock(hidden_dims[0], dropout)
                for _ in range(len(hidden_dims) - 1)
            ])
            self.encoder = None
            enc_dim = hidden_dims[0]
        else:
            self.encoder, enc_dim = build_mlp_encoder(input_dim, hidden_dims, dropout)
            self.input_proj = None
            self.res_blocks = None
            
        self.runtime_head = nn.Linear(enc_dim, 1)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual and self.input_proj is not None:
            features = self.input_proj(x)
            for block in self.res_blocks:
                features = block(features)
        else:
            features = self.encoder(x)
        return self.runtime_head(features).squeeze(-1)


class MLPModel(BaseModel):
    """
    MLP for duration prediction: threshold as input parameter, predict log2(duration).
    
    For small datasets (< 500 samples), use smaller hidden_dims and higher dropout.
    Example for small data: hidden_dims=[32, 16], dropout=0.5
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.4,
        lr: float = 5e-4,
        weight_decay: float = 1e-3,
        device: str = "cpu",
        epochs: int = 200,
        early_stopping_patience: int = 30,
        use_residual: bool = False,
        use_huber_loss: bool = True,
        grad_clip: float = 1.0,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_residual = use_residual
        self.use_huber_loss = use_huber_loss
        self.grad_clip = grad_clip

        self.network = MLPNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_residual=use_residual,
        ).to(device)
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        if use_huber_loss:
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.L1Loss()
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
            return (x - self.feature_mean) / (self.feature_std + 1e-8)
        return x

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.network.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features = batch["features"].to(self.device)
            log2_runtime = batch["log2_runtime"].to(self.device)
            features = self._normalize(features)
            self.optimizer.zero_grad()
            log2_pred = self.network(features)
            loss = self.criterion(log2_pred, log2_runtime)
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
        history = {"train_loss": [], "val_runtime_mae": []}
        best_val_mae = float("inf")
        patience_counter = 0
        best_state = None
        epoch_iter = range(self.epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training", leave=False)
        for epoch in epoch_iter:
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_mae = val_metrics["runtime_mae"]
            self.scheduler.step(val_mae)
            history["train_loss"].append(train_metrics["loss"])
            history["val_runtime_mae"].append(val_mae)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                patience_counter += 1
            if show_progress:
                epoch_iter.set_postfix(
                    loss=f"{train_metrics['loss']:.3f}",
                    val_mae=f"{val_mae:.3f}",
                )
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {train_metrics['loss']:.4f} | Val MAE (log2): {val_mae:.4f}")
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
        all_pred = []
        all_label = []
        for batch in loader:
            features = batch["features"].to(self.device)
            log2_runtime = batch["log2_runtime"]
            features = self._normalize(features)
            log2_pred = self.network(features).cpu()
            all_pred.extend(log2_pred.tolist())
            all_label.extend(log2_runtime.tolist())
        return {
            "runtime_mse": mean_squared_error(all_label, all_pred),
            "runtime_mae": mean_absolute_error(all_label, all_pred),
        }

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.network.eval()
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        features = features.to(self.device)
        features = self._normalize(features)
        log2_pred = self.network(features).cpu().numpy()
        runtime_values = np.power(2.0, log2_pred)
        threshold_values = np.round(np.power(2.0, features.cpu().numpy()[:, THRESHOLD_FEATURE_IDX])).astype(int)
        return threshold_values, runtime_values

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
                "use_residual": self.use_residual,
            },
        }, path / "mlp_model.pt")

    def load(self, path: Path) -> None:
        path = Path(path)
        ckpt = torch.load(path / "mlp_model.pt", map_location=self.device)
        self.network.load_state_dict(ckpt["network_state"])
        self.feature_mean = ckpt["feature_mean"]
        self.feature_std = ckpt["feature_std"]

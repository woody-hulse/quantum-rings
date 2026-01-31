"""
Model implementations for quantum circuit threshold and runtime prediction.

DEPRECATED: This module is kept for backward compatibility.
Use the new modular structure instead:
    - from models import MLPModel, XGBoostModel, BaseModel
    - from scoring import compute_challenge_score
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from data_loader import (
    THRESHOLD_LADDER,
    create_data_loaders,
    get_feature_statistics,
)

from scoring import compute_challenge_score


class MLPModel(nn.Module):
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


class MLPTrainer:
    """Trainer for the MLP model."""
    
    def __init__(
        self,
        model: MLPModel,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        threshold_weight: float = 1.0,
        runtime_weight: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.threshold_weight = threshold_weight
        self.runtime_weight = runtime_weight
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.threshold_criterion = nn.CrossEntropyLoss()
        self.runtime_criterion = nn.MSELoss()
        
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
    
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.feature_mean = mean.to(self.device)
        self.feature_std = std.to(self.device)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mean is not None:
            return (x - self.feature_mean) / self.feature_std
        return x
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_thresh_loss = 0.0
        total_runtime_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            features = batch["features"].to(self.device)
            threshold_labels = batch["threshold_class"].to(self.device)
            runtime_labels = batch["log_runtime"].to(self.device)
            
            features = self.normalize(features)
            
            self.optimizer.zero_grad()
            threshold_logits, runtime_pred = self.model(features)
            
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
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_thresh_preds = []
        all_thresh_labels = []
        all_runtime_preds = []
        all_runtime_labels = []
        
        for batch in val_loader:
            features = batch["features"].to(self.device)
            threshold_labels = batch["threshold_class"]
            runtime_labels = batch["log_runtime"]
            
            features = self.normalize(features)
            threshold_logits, runtime_pred = self.model(features)
            
            thresh_preds = threshold_logits.argmax(dim=1).cpu()
            runtime_pred = runtime_pred.cpu()
            
            all_thresh_preds.extend(thresh_preds.tolist())
            all_thresh_labels.extend(threshold_labels.tolist())
            all_runtime_preds.extend(runtime_pred.tolist())
            all_runtime_labels.extend(runtime_labels.tolist())
        
        thresh_acc = accuracy_score(all_thresh_labels, all_thresh_preds)
        runtime_mse = mean_squared_error(all_runtime_labels, all_runtime_preds)
        runtime_mae = mean_absolute_error(all_runtime_labels, all_runtime_preds)
        
        return {
            "threshold_accuracy": thresh_acc,
            "runtime_mse": runtime_mse,
            "runtime_mae": runtime_mae,
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        mean, std = get_feature_statistics(train_loader)
        self.set_normalization(mean, std)
        
        history = {"train_loss": [], "val_threshold_acc": [], "val_runtime_mse": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            val_loss = (1 - val_metrics["threshold_accuracy"]) + val_metrics["runtime_mse"]
            self.scheduler.step(val_loss)
            
            history["train_loss"].append(train_metrics["loss"])
            history["val_threshold_acc"].append(val_metrics["threshold_accuracy"])
            history["val_runtime_mse"].append(val_metrics["runtime_mse"])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Thresh Acc: {val_metrics['threshold_accuracy']:.4f} | "
                      f"Val Runtime MSE: {val_metrics['runtime_mse']:.4f}")
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return history
    
    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        features = features.to(self.device)
        features = self.normalize(features)
        threshold_logits, runtime_pred = self.model(features)
        
        thresh_classes = threshold_logits.argmax(dim=1).cpu().numpy()
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])
        runtime_values = np.expm1(runtime_pred.cpu().numpy())
        
        return thresh_values, runtime_values


class XGBoostModel:
    """XGBoost models for threshold classification and runtime regression."""
    
    def __init__(
        self,
        threshold_params: Optional[Dict] = None,
        runtime_params: Optional[Dict] = None,
    ):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        
        self.threshold_params = threshold_params or {
            "objective": "multi:softmax",
            "num_class": len(THRESHOLD_LADDER),
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        
        self.runtime_params = runtime_params or {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        
        self.threshold_model: Optional[xgb.XGBRegressor] = None
        self.runtime_model: Optional[xgb.XGBRegressor] = None
        self.scaler = StandardScaler()
    
    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_features = []
        all_thresh = []
        all_runtime = []
        
        for batch in loader:
            all_features.append(batch["features"].numpy())
            all_thresh.extend(batch["threshold_class"].tolist())
            all_runtime.extend(batch["log_runtime"].tolist())
        
        X = np.vstack(all_features)
        y_thresh = np.array(all_thresh, dtype=np.float32)
        y_runtime = np.array(all_runtime)
        
        return X, y_thresh, y_runtime
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        X_train, y_thresh_train, y_runtime_train = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val = self._extract_data(val_loader)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        thresh_params = {k: v for k, v in self.threshold_params.items() 
                         if k not in ["objective", "num_class", "eval_metric"]}
        thresh_params["objective"] = "reg:squarederror"
        
        if verbose:
            print("Training threshold regressor...")
        self.threshold_model = xgb.XGBRegressor(**thresh_params)
        self.threshold_model.fit(
            X_train_scaled, y_thresh_train,
            eval_set=[(X_val_scaled, y_thresh_val)],
            verbose=False,
        )
        
        if verbose:
            print("Training runtime regressor...")
        self.runtime_model = xgb.XGBRegressor(**self.runtime_params)
        self.runtime_model.fit(
            X_train_scaled, y_runtime_train,
            eval_set=[(X_val_scaled, y_runtime_val)],
            verbose=False,
        )
        
        train_metrics = self._evaluate(X_train_scaled, y_thresh_train, y_runtime_train)
        val_metrics = self._evaluate(X_val_scaled, y_thresh_val, y_runtime_val)
        
        if verbose:
            print(f"Train Threshold Acc: {train_metrics['threshold_accuracy']:.4f} | "
                  f"Train Runtime MSE: {train_metrics['runtime_mse']:.4f}")
            print(f"Val Threshold Acc: {val_metrics['threshold_accuracy']:.4f} | "
                  f"Val Runtime MSE: {val_metrics['runtime_mse']:.4f}")
        
        return {"train": train_metrics, "val": val_metrics}
    
    def _round_to_class(self, pred: np.ndarray) -> np.ndarray:
        """Round regression predictions to nearest valid class index."""
        return np.clip(np.round(pred), 0, len(THRESHOLD_LADDER) - 1).astype(int)
    
    def _evaluate(
        self,
        X: np.ndarray,
        y_thresh: np.ndarray,
        y_runtime: np.ndarray,
    ) -> Dict[str, float]:
        thresh_pred_raw = self.threshold_model.predict(X)
        thresh_pred = self._round_to_class(thresh_pred_raw)
        runtime_pred = self.runtime_model.predict(X)
        
        return {
            "threshold_accuracy": accuracy_score(y_thresh.astype(int), thresh_pred),
            "runtime_mse": mean_squared_error(y_runtime, runtime_pred),
            "runtime_mae": mean_absolute_error(y_runtime, runtime_pred),
        }
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        X, y_thresh, y_runtime = self._extract_data(loader)
        X_scaled = self.scaler.transform(X)
        return self._evaluate(X_scaled, y_thresh, y_runtime)
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold_model is None or self.runtime_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(features)
        thresh_pred_raw = self.threshold_model.predict(X_scaled)
        thresh_classes = self._round_to_class(thresh_pred_raw)
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])
        runtime_log = self.runtime_model.predict(X_scaled)
        runtime_values = np.expm1(runtime_log)
        
        return thresh_values, runtime_values
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        return {
            "threshold": self.threshold_model.feature_importances_,
            "runtime": self.runtime_model.feature_importances_,
        }
    
    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.threshold_model.save_model(path / "threshold_model.json")
        self.runtime_model.save_model(path / "runtime_model.json")
        
        np.save(path / "scaler_mean.npy", self.scaler.mean_)
        np.save(path / "scaler_scale.npy", self.scaler.scale_)
    
    def load(self, path: Path):
        path = Path(path)
        
        self.threshold_model = xgb.XGBClassifier()
        self.threshold_model.load_model(path / "threshold_model.json")
        
        self.runtime_model = xgb.XGBRegressor()
        self.runtime_model.load_model(path / "runtime_model.json")
        
        self.scaler.mean_ = np.load(path / "scaler_mean.npy")
        self.scaler.scale_ = np.load(path / "scaler_scale.npy")

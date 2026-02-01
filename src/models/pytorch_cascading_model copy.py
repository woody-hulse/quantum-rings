"""
PyTorch Cascading Model (GPU Enabled).
Replicates the "Two-Stage" logic of the linear model but uses Neural Networks
to run on NVIDIA GPUs.

Architecture:
1. Threshold Net: Inputs -> Hidden -> Threshold Class
2. Runtime Net:   [Inputs + Threshold] -> Hidden -> Runtime
"""

from typing import Dict, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER
from models.base import BaseModel


class PyTorchCascadingModel(BaseModel):
    def __init__(
        self,
        hidden_dim: int = 64,      # Small layer to capture interactions (Span * Cut)
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 128,     # Increase this to 1024+ for GPU speed
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        # We will initialize these in fit() once we know input dimension
        self.threshold_net = None
        self.runtime_net = None
        self.scaler = StandardScaler()

    @property
    def name(self) -> str:
        return f"PyTorch_Cascading_MLP_{self.device}"

    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts data to CPU numpy for scaling first."""
        all_features = []
        all_thresh = []
        all_runtime = []

        for batch in loader:
            all_features.append(batch["features"].numpy())
            all_thresh.extend(batch["threshold_class"].tolist())
            all_runtime.extend(batch["log_runtime"].tolist())

        return np.vstack(all_features), np.array(all_thresh), np.array(all_runtime)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = False, **kwargs) -> Dict[str, Any]:
        if verbose:
            print(f"Training on {self.device.upper()}...")

        # 1. Prepare Data
        X_train, y_thresh_train, y_runtime_train = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val = self._extract_data(val_loader)

        # Scale inputs (Critical for Neural Nets)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Convert to Tensors and move to GPU
        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        y_thresh_t = torch.LongTensor(y_thresh_train).to(self.device)
        y_runtime_t = torch.FloatTensor(y_runtime_train).unsqueeze(1).to(self.device)

        X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)
        
        input_dim = X_train.shape[1]
        num_classes = len(THRESHOLD_LADDER)

        # -----------------------------
        # 2. Define Networks
        # -----------------------------
        
        # Stage 1: Threshold Classifier
        # Input -> Hidden -> Class Logits
        self.threshold_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_classes)
        ).to(self.device)

        # Stage 2: Runtime Regressor
        # [Input + 1 Class Feature] -> Hidden -> Runtime
        self.runtime_net = nn.Sequential(
            nn.Linear(input_dim + 1, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        ).to(self.device)

        # -----------------------------
        # 3. Optimization Loop
        # -----------------------------
        optimizer_thresh = optim.Adam(self.threshold_net.parameters(), lr=self.learning_rate)
        optimizer_runtime = optim.Adam(self.runtime_net.parameters(), lr=self.learning_rate)
        
        criterion_thresh = nn.CrossEntropyLoss()
        criterion_runtime = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_train_t, y_thresh_t, y_runtime_t)
        # Use a larger batch size for GPU efficiency
        gpu_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.threshold_net.train()
            self.runtime_net.train()
            
            total_loss_t = 0
            total_loss_r = 0
            
            for xb, yb_t, yb_r in gpu_loader:
                # A. Train Threshold
                optimizer_thresh.zero_grad()
                logits = self.threshold_net(xb)
                loss_t = criterion_thresh(logits, yb_t)
                loss_t.backward()
                optimizer_thresh.step()
                total_loss_t += loss_t.item()

                # B. Train Runtime (Teacher Forcing)
                # Augment input with TRUE threshold class index
                threshold_feature = yb_t.float().unsqueeze(1) 
                xb_aug = torch.cat([xb, threshold_feature], dim=1)
                
                optimizer_runtime.zero_grad()
                pred_r = self.runtime_net(xb_aug)
                loss_r = criterion_runtime(pred_r, yb_r)
                loss_r.backward()
                optimizer_runtime.step()
                total_loss_r += loss_r.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Loss Thresh: {total_loss_t:.4f} | Loss Runtime: {total_loss_r:.4f}")

        # -----------------------------
        # 4. Evaluation
        # -----------------------------
        self.threshold_net.eval()
        self.runtime_net.eval()
        
        with torch.no_grad():
            # Validation Predictions
            # 1. Predict Threshold
            logits_val = self.threshold_net(X_val_t)
            thresh_preds_val = torch.argmax(logits_val, dim=1)
            
            # 2. Augment with PREDICTED threshold
            thresh_feat_val = thresh_preds_val.float().unsqueeze(1)
            X_val_aug = torch.cat([X_val_t, thresh_feat_val], dim=1)
            
            # 3. Predict Runtime
            runtime_preds_val = self.runtime_net(X_val_aug)
            
            # Move back to CPU for metrics
            t_pred = thresh_preds_val.cpu().numpy()
            r_pred = runtime_preds_val.cpu().numpy()

        val_metrics = {
            "threshold_accuracy": accuracy_score(y_thresh_val, t_pred),
            "runtime_mse": mean_squared_error(y_runtime_val, r_pred)
        }
        
        if verbose:
            print(f"Val Acc: {val_metrics['threshold_accuracy']:.2%} | Val MSE: {val_metrics['runtime_mse']:.4f}")

        return {"val": val_metrics}

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.threshold_net.eval()
        self.runtime_net.eval()
        
        X_scaled = self.scaler.transform(features)
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            # Stage 1
            logits = self.threshold_net(X_t)
            class_indices = torch.argmax(logits, dim=1)
            
            # Stage 2
            class_feat = class_indices.float().unsqueeze(1)
            X_aug = torch.cat([X_t, class_feat], dim=1)
            runtime_log = self.runtime_net(X_aug)
            
        # Convert to final values
        indices_cpu = class_indices.cpu().numpy()
        runtime_cpu = runtime_log.cpu().numpy().flatten()
        
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in indices_cpu])
        runtime_values = np.expm1(runtime_cpu)
        
        return thresh_values, runtime_values

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'thresh_state': self.threshold_net.state_dict(),
            'runtime_state': self.runtime_net.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }, path / "pytorch_model.pt")

    def load(self, path: Path) -> None:
        checkpoint = torch.load(Path(path) / "pytorch_model.pt", map_location=self.device)
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']
        # Note: initialization requires knowing input dim, so full load happens in predict/fit context
        # Ideally, save args or init logic here.

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
        self.threshold_net.eval()
        self.runtime_net.eval()

        all_thresh_preds = []
        all_runtime_preds = []
        all_thresh_true = []
        all_runtime_true = []

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].numpy()
                thresh_true = batch["threshold_class"].numpy()
                runtime_true = batch["log_runtime"].numpy()

                # Scale features
                features_scaled = self.scaler.transform(features)
                features_t = torch.FloatTensor(features_scaled).to(self.device)

                # Predict threshold classes
                logits = self.threshold_net(features_t)
                thresh_preds = torch.argmax(logits, dim=1).cpu().numpy()

                # Augment features with predicted threshold class
                thresh_feat = torch.FloatTensor(thresh_preds).unsqueeze(1).to(self.device)
                features_aug = torch.cat([features_t, thresh_feat], dim=1)

                # Predict runtime
                runtime_preds = self.runtime_net(features_aug).cpu().numpy().flatten()

                # Collect predictions and true values
                all_thresh_preds.extend(thresh_preds)
                all_runtime_preds.extend(runtime_preds)
                all_thresh_true.extend(thresh_true)
                all_runtime_true.extend(runtime_true)

        # Compute metrics
        threshold_accuracy = accuracy_score(all_thresh_true, all_thresh_preds)
        runtime_mse = mean_squared_error(all_runtime_true, all_runtime_preds)
        runtime_mae = np.mean(np.abs(np.array(all_runtime_true) - np.array(all_runtime_preds)))

        return {
            "threshold_accuracy": threshold_accuracy,
            "runtime_mse": runtime_mse,
            "runtime_mae": runtime_mae,
        }
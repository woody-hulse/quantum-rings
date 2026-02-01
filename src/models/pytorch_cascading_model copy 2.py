"""
Deep PyTorch Cascading Model with Threshold Embeddings.
Improved Runtime architecture using learned embeddings and robust loss.
"""

from typing import Dict, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER
from models.base import BaseModel


class PyTorchCascadingModel(BaseModel):
    def __init__(
        self,
        hidden_dim: int = 128,      
        embedding_dim: int = 16,    
        learning_rate: float = 0.001,
        epochs: int = 100,          
        batch_size: int = 512,      
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        self.threshold_net = None
        self.runtime_net = None
        self.thresh_embedding = None
        self.scaler = StandardScaler()

    @property
    def name(self) -> str:
        return f"Deep_PyTorch_Embed_{self.device}"

    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract data to CPU numpy for scaling."""
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
            print(f"Training on {self.device.upper()} with Deep Runtime Architecture...")

        # 1. Prepare Data
        X_train, y_thresh_train, y_runtime_train = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val = self._extract_data(val_loader)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Move to GPU
        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        y_thresh_t = torch.LongTensor(y_thresh_train).to(self.device)
        y_runtime_t = torch.FloatTensor(y_runtime_train).unsqueeze(1).to(self.device)
        
        input_dim = X_train.shape[1]
        num_classes = len(THRESHOLD_LADDER)

        # 2. Define Networks (if not already defined)
        if self.threshold_net is None:
            self.threshold_net = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dim // 2),
                nn.Linear(self.hidden_dim // 2, num_classes)
            ).to(self.device)

            self.thresh_embedding = nn.Embedding(num_classes, self.embedding_dim).to(self.device)

            runtime_input_dim = input_dim + self.embedding_dim
            self.runtime_net = nn.Sequential(
                nn.Linear(runtime_input_dim, self.hidden_dim * 2),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dim * 2),
                nn.Dropout(0.1), 
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_dim),
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1) 
            ).to(self.device)

        # 3. Optimization Loop
        params_thresh = list(self.threshold_net.parameters())
        params_runtime = list(self.runtime_net.parameters()) + list(self.thresh_embedding.parameters())

        opt_thresh = optim.AdamW(params_thresh, lr=self.learning_rate, weight_decay=1e-4)
        opt_runtime = optim.AdamW(params_runtime, lr=self.learning_rate, weight_decay=1e-4)
        
        crit_thresh = nn.CrossEntropyLoss()
        crit_runtime = nn.HuberLoss(delta=1.0) 

        dataset = torch.utils.data.TensorDataset(X_train_t, y_thresh_t, y_runtime_t)
        gpu_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.threshold_net.train()
            self.runtime_net.train()
            self.thresh_embedding.train()
            
            for xb, yb_t, yb_r in gpu_loader:
                # Train Threshold
                opt_thresh.zero_grad()
                logits = self.threshold_net(xb)
                loss_t = crit_thresh(logits, yb_t)
                loss_t.backward()
                opt_thresh.step()

                # Train Runtime
                opt_runtime.zero_grad()
                emb = self.thresh_embedding(yb_t) # Teacher Forcing
                xb_aug = torch.cat([xb, emb], dim=1)
                pred_r = self.runtime_net(xb_aug)
                loss_r = crit_runtime(pred_r, yb_r)
                loss_r.backward()
                opt_runtime.step()

        # 4. Final Validation using the public evaluate method
        val_metrics = self.evaluate(val_loader)
        
        if verbose:
            print(f"\nFinal Val Acc: {val_metrics['threshold_accuracy']:.2%}")
            print(f"Final Val MSE: {val_metrics['runtime_mse']:.4f}")

        return {"val": val_metrics}

    # --- THIS WAS THE MISSING METHOD ---
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Required by BaseModel. Runs full inference pipeline to calculate metrics.
        """
        # 1. Extract Data
        X, y_thresh, y_runtime = self._extract_data(loader)
        
        # 2. Scale
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        
        self.threshold_net.eval()
        self.runtime_net.eval()
        self.thresh_embedding.eval()
        
        with torch.no_grad():
            # 3. Predict Threshold Rung
            logits = self.threshold_net(X_t)
            thresh_preds = torch.argmax(logits, dim=1)
            
            # 4. Embed PREDICTED Threshold (Simulating real inference)
            emb = self.thresh_embedding(thresh_preds)
            
            # 5. Predict Runtime
            X_aug = torch.cat([X_t, emb], dim=1)
            runtime_preds = self.runtime_net(X_aug)
            
            # Move to CPU
            t_pred = thresh_preds.cpu().numpy()
            r_pred = runtime_preds.cpu().numpy().flatten()

        return {
            "threshold_accuracy": accuracy_score(y_thresh, t_pred),
            "runtime_mse": mean_squared_error(y_runtime, r_pred),
            "runtime_mae": mean_absolute_error(y_runtime, r_pred),
        }

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.threshold_net.eval()
        self.runtime_net.eval()
        self.thresh_embedding.eval()
        
        X_scaled = self.scaler.transform(features)
        X_t = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            # 1. Predict Threshold
            logits = self.threshold_net(X_t)
            class_indices = torch.argmax(logits, dim=1)
            
            # 2. Embed
            emb = self.thresh_embedding(class_indices)
            
            # 3. Predict Runtime
            X_aug = torch.cat([X_t, emb], dim=1)
            runtime_log = self.runtime_net(X_aug)
            
        indices_cpu = class_indices.cpu().numpy()
        runtime_cpu = runtime_log.cpu().numpy().flatten()
        
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in indices_cpu])
        runtime_values = np.expm1(runtime_cpu)
        
        return thresh_values, runtime_values

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # We need to save input_dim so we can reconstruct the network on load
        input_dim = self.threshold_net[0].in_features
        
        torch.save({
            'input_dim': input_dim,
            'thresh_state': self.threshold_net.state_dict(),
            'runtime_state': self.runtime_net.state_dict(),
            'emb_state': self.thresh_embedding.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }, path / "deep_pytorch_model.pt")

    def load(self, path: Path) -> None:
        checkpoint = torch.load(Path(path) / "deep_pytorch_model.pt", map_location=self.device)
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']
        
        # Reconstruct networks using saved input_dim
        input_dim = checkpoint['input_dim']
        num_classes = len(THRESHOLD_LADDER)
        
        self.threshold_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, num_classes)
        ).to(self.device)
        
        self.thresh_embedding = nn.Embedding(num_classes, self.embedding_dim).to(self.device)
        
        runtime_input_dim = input_dim + self.embedding_dim
        self.runtime_net = nn.Sequential(
            nn.Linear(runtime_input_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        
        self.threshold_net.load_state_dict(checkpoint['thresh_state'])
        self.runtime_net.load_state_dict(checkpoint['runtime_state'])
        self.thresh_embedding.load_state_dict(checkpoint['emb_state'])
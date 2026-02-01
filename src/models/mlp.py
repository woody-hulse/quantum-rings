"""
MLP model for threshold classification and runtime regression.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER, get_feature_statistics
from models.base import BaseModel


class ScoringAlignedLoss(nn.Module):
    """
    Classification loss that directly matches the challenge scoring.
    
    Challenge scoring:
    - Underprediction (pred < true): score = 0
    - Correct: score = 1.0
    - Overprediction by k steps: score = 2^(-k) (0.5, 0.25, 0.125, ...)
    
    This loss = expected (1 - score), so minimizing loss = maximizing score.
    
    The 'temperature' parameter controls sharpness of the probability distribution
    before computing expected score. Lower temp = sharper predictions.
    """
    
    def __init__(self, num_classes: int = 9, temperature: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        
        # score_matrix[true_class, pred_class] = challenge score for that prediction
        score_matrix = torch.zeros(num_classes, num_classes)
        for t in range(num_classes):
            for p in range(num_classes):
                if p < t:
                    score_matrix[t, p] = 0.0  # underprediction = 0 score
                else:
                    steps_over = p - t
                    score_matrix[t, p] = 2.0 ** (-steps_over)  # 1.0, 0.5, 0.25, ...
        
        self.register_buffer("score_matrix", score_matrix)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply temperature scaling
        probs = F.softmax(logits / self.temperature, dim=1)
        
        # Get score row for each target
        scores = self.score_matrix[targets]  # [batch, num_classes]
        
        # Expected score
        expected_score = (probs * scores).sum(dim=1)
        
        # Loss = 1 - expected_score (so minimizing loss = maximizing score)
        return (1.0 - expected_score).mean()


class ScoringAlignedContinuousLoss(nn.Module):
    """
    Continuous threshold loss that matches the challenge scoring.
    
    The predicted value is rounded up to a rung, then scored:
    - Underprediction: score = 0
    - Correct: score = 1.0  
    - k steps over: score = 2^(-k)
    
    Loss = 1 - score (non-differentiable through rounding, so we use a 
    soft approximation with temperature-controlled sigmoid boundaries).
    
    Args:
        temperature: Controls softness of rung boundaries (default 10.0)
                     Lower = sharper boundaries (more like true scoring)
                     Higher = smoother gradients for training
    """
    
    def __init__(self, temperature: float = 10.0):
        super().__init__()
        self.temperature = temperature
        self.register_buffer("ladder", torch.tensor(THRESHOLD_LADDER, dtype=torch.float32))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ladder = self.ladder
        n_rungs = len(ladder)
        
        # Get target rung indices
        target_idx = torch.searchsorted(ladder, target.clamp(min=ladder[0], max=ladder[-1]))
        target_idx = target_idx.clamp(max=n_rungs - 1)
        
        # Compute soft probability of being in each rung using sigmoid boundaries
        # P(pred rounds to rung i) = sigmoid((pred - lower_i) * temp) * sigmoid((upper_i - pred) * temp)
        
        scores = []
        for i in range(pred.shape[0]):
            p = pred[i]
            t_idx = target_idx[i]
            
            # Compute probability of being at or above each rung
            # Using cumulative soft boundaries
            rung_probs = torch.zeros(n_rungs, device=pred.device)
            
            for r in range(n_rungs):
                lower = ladder[r - 1] if r > 0 else torch.tensor(0.0, device=pred.device)
                upper = ladder[r]
                
                # Soft indicator: prob that pred rounds to rung r
                prob_above_lower = torch.sigmoid((p - lower) * self.temperature)
                prob_below_upper = torch.sigmoid((upper - p) * self.temperature)
                rung_probs[r] = prob_above_lower * prob_below_upper
            
            # Normalize to sum to 1
            rung_probs = rung_probs / (rung_probs.sum() + 1e-8)
            
            # Compute expected score
            rung_scores = torch.zeros(n_rungs, device=pred.device)
            for r in range(n_rungs):
                if r < t_idx:
                    rung_scores[r] = 0.0  # underprediction
                else:
                    steps_over = r - t_idx
                    rung_scores[r] = 2.0 ** (-steps_over.float())
            
            expected_score = (rung_probs * rung_scores).sum()
            scores.append(expected_score)
        
        expected_scores = torch.stack(scores)
        return (1.0 - expected_scores).mean()


def round_up_to_rung(pred: float) -> int:
    """Smallest ladder value >= pred. Used only for scoring, not for loss."""
    for t in THRESHOLD_LADDER:
        if t >= max(pred, 0.0):
            return t
    return THRESHOLD_LADDER[-1]


class MLPNetwork(nn.Module):
    """Multi-task MLP for threshold classification and runtime regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        num_threshold_classes: int = 9,
        dropout: float = 0.1,
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
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        threshold_logits = self.threshold_head(features)
        runtime_pred = self.softplus(self.runtime_head(features).squeeze(-1))
        return threshold_logits, runtime_pred


class MLPModel(BaseModel):
    """
    MLP model for threshold classification and runtime regression.

    - Threshold: Scoring-aligned loss that directly optimizes challenge score
    - Runtime: MAE loss in log1p space

    Inference strategies:
    - "argmax": Standard argmax on logits (default)
    - "decision_theoretic": Pick class with highest expected challenge score
    - "shift": Argmax + constant shift upward
    """

    # Inference strategy constants
    INFERENCE_ARGMAX = "argmax"
    INFERENCE_DECISION_THEORETIC = "decision_theoretic"
    INFERENCE_SHIFT = "shift"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 0,
        device: str = "cpu",
        epochs: int = 100,
        early_stopping_patience: int = 20,
        threshold_weight: float = 1.0,
        runtime_weight: float = 1.0,
        temperature: float = 1.0,
        inference_strategy: str = "argmax",
        inference_shift: int = 0,
    ):
        """
        Args:
            temperature: Controls softmax temperature for threshold loss (default 1.0)
                         Lower = sharper predictions
                         Higher = softer predictions, may help avoid underprediction
            inference_strategy: How to convert logits to predictions at inference time
                - "argmax": Standard argmax (default)
                - "decision_theoretic": Maximize expected challenge score
                - "shift": Argmax + constant shift
            inference_shift: Number of classes to shift up (only used with "shift" strategy)
        """
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
        self.temperature = temperature
        self.inference_strategy = inference_strategy
        self.inference_shift = inference_shift

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

        self.threshold_criterion = ScoringAlignedLoss(
            num_classes=len(THRESHOLD_LADDER),
            temperature=temperature,
        ).to(device)
        self.runtime_criterion = nn.L1Loss()

        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

        # Precompute score matrix for decision-theoretic inference
        # score_matrix[true_class, pred_class] = challenge score
        num_classes = len(THRESHOLD_LADDER)
        score_matrix = torch.zeros(num_classes, num_classes)
        for true_idx in range(num_classes):
            for pred_idx in range(num_classes):
                if pred_idx < true_idx:
                    score_matrix[true_idx, pred_idx] = 0.0
                else:
                    steps_over = pred_idx - true_idx
                    score_matrix[true_idx, pred_idx] = 2.0 ** (-steps_over)
        self.score_matrix = score_matrix.to(device)
    
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
    
    def _init_runtime_bias(self, train_loader: DataLoader):
        """Initialize runtime head bias to training mean for faster convergence."""
        all_log_runtime = []
        for batch in train_loader:
            all_log_runtime.extend(batch["log_runtime"].tolist())
        
        runtime_mean = np.mean(all_log_runtime)
        with torch.no_grad():
            self.network.runtime_head.bias.fill_(runtime_mean)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        mean, std = get_feature_statistics(train_loader)
        self._set_normalization(mean, std)
        self._init_runtime_bias(train_loader)
        
        history = {"train_loss": [], "val_threshold_acc": [], "val_runtime_mse": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        epoch_iter = range(self.epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training", leave=False)
        
        for epoch in epoch_iter:
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
            
            if show_progress:
                epoch_iter.set_postfix({
                    "loss": f"{train_metrics['loss']:.3f}",
                    "val_acc": f"{val_metrics['threshold_accuracy']:.3f}",
                    "val_mse": f"{val_metrics['runtime_mse']:.3f}",
                })
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Thresh Acc: {val_metrics['threshold_accuracy']:.4f} | "
                      f"Val Runtime MSE: {val_metrics['runtime_mse']:.4f}")
            
            if patience_counter >= self.early_stopping_patience:
                if show_progress:
                    epoch_iter.close()
                elif verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.network.load_state_dict(best_state)
        
        return {"history": history}
    
    def _decision_theoretic_predict(self, logits: torch.Tensor) -> np.ndarray:
        """
        Pick the class that maximizes expected challenge score.

        For each possible prediction, compute:
            E[score | pred] = sum over true classes of P(true) * score(pred, true)

        The score(pred, true) = 2^(-(pred - true)) if pred >= true, else 0.

        This naturally biases toward safe overprediction when the model is uncertain,
        because underprediction gives score=0 while overprediction decays gracefully.
        """
        probs = F.softmax(logits, dim=1)  # (batch, num_classes)

        # expected_scores[i, j] = expected score if we predict class j for sample i
        # = sum over k of: probs[i, k] * score_matrix[k, j]
        expected_scores = probs @ self.score_matrix  # (batch, num_classes)

        # Pick the prediction with highest expected score
        return expected_scores.argmax(dim=1).cpu().numpy()

    def _get_threshold_predictions(
        self,
        logits: torch.Tensor,
        strategy: Optional[str] = None,
    ) -> np.ndarray:
        """Convert logits to class predictions using the specified strategy."""
        strategy = strategy or self.inference_strategy

        if strategy == self.INFERENCE_DECISION_THEORETIC:
            return self._decision_theoretic_predict(logits)
        elif strategy == self.INFERENCE_SHIFT:
            preds = logits.argmax(dim=1).cpu().numpy()
            return np.clip(preds + self.inference_shift, 0, len(THRESHOLD_LADDER) - 1)
        else:  # argmax (default)
            return logits.argmax(dim=1).cpu().numpy()

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        strategy: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a data loader.

        Args:
            loader: Data loader to evaluate on
            strategy: Inference strategy override (uses self.inference_strategy if None)
        """
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

            thresh_preds = self._get_threshold_predictions(threshold_logits, strategy)
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
    def predict(
        self,
        features: np.ndarray,
        strategy: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict threshold and runtime values.

        Args:
            features: Input features array
            strategy: Inference strategy override. Options:
                - "argmax": Standard argmax on logits (default)
                - "decision_theoretic": Pick class with highest expected challenge score
                - "shift": Argmax + constant shift upward

        Returns:
            Tuple of (threshold_values, runtime_values)
        """
        self.network.eval()

        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)

        features = features.to(self.device)
        features = self._normalize(features)
        threshold_logits, runtime_pred = self.network(features)

        thresh_classes = self._get_threshold_predictions(threshold_logits, strategy)
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


class MLPContinuousNetwork(nn.Module):
    """MLP that outputs continuous threshold (scalar) and log1p(runtime)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
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
        self.threshold_head = nn.Linear(prev_dim, 1)
        self.runtime_head = nn.Linear(prev_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        threshold_continuous = self.softplus(self.threshold_head(features).squeeze(-1))
        runtime_pred = self.softplus(self.runtime_head(features).squeeze(-1))
        return threshold_continuous, runtime_pred


class MLPContinuousModel(BaseModel):
    """
    MLP that predicts threshold and runtime continuously.
    - Loss: Scoring-aligned loss for threshold, MAE for runtime
    - Scoring: threshold prediction is rounded up to the next ladder rung only when computing score.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 0,
        device: str = "cpu",
        epochs: int = 100,
        early_stopping_patience: int = 20,
        threshold_weight: float = 1.0,
        runtime_weight: float = 1.0,
        temperature: float = 10.0,
    ):
        """
        Args:
            temperature: Controls softness of rung boundaries in loss (default 10.0)
                         Higher = smoother gradients
                         Lower = sharper, closer to true scoring
        """
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
        self.temperature = temperature

        self.network = MLPContinuousNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)

        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.threshold_criterion = ScoringAlignedContinuousLoss(temperature=temperature)
        self.runtime_criterion = nn.L1Loss()

        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "MLPContinuous"

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
            threshold_classes = batch["threshold_class"].to(self.device)
            runtime_labels = batch["log_runtime"].to(self.device)
            ladder = torch.tensor(THRESHOLD_LADDER, dtype=torch.float32, device=self.device)
            threshold_targets = ladder[threshold_classes]

            features = self._normalize(features)
            self.optimizer.zero_grad()
            threshold_pred, runtime_pred = self.network(features)

            thresh_loss = self.threshold_criterion(threshold_pred, threshold_targets)
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

    def _init_runtime_bias(self, train_loader: DataLoader):
        all_log_runtime = []
        for batch in train_loader:
            all_log_runtime.extend(batch["log_runtime"].tolist())
        runtime_mean = np.mean(all_log_runtime)
        with torch.no_grad():
            self.network.runtime_head.bias.fill_(runtime_mean)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        mean, std = get_feature_statistics(train_loader)
        self._set_normalization(mean, std)
        self._init_runtime_bias(train_loader)

        history = {"train_loss": [], "val_threshold_mae": [], "val_runtime_mae": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        epoch_iter = range(self.epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training", leave=False)

        for epoch in epoch_iter:
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["threshold_mae"] + val_metrics["runtime_mae"]
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_metrics["loss"])
            history["val_threshold_mae"].append(val_metrics["threshold_mae"])
            history["val_runtime_mae"].append(val_metrics["runtime_mae"])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                patience_counter += 1

            if show_progress:
                epoch_iter.set_postfix({
                    "loss": f"{train_metrics['loss']:.3f}",
                    "val_thr_mae": f"{val_metrics['threshold_mae']:.3f}",
                    "val_rt_mae": f"{val_metrics['runtime_mae']:.3f}",
                })
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Thresh MAE: {val_metrics['threshold_mae']:.4f} | "
                      f"Val Runtime MAE: {val_metrics['runtime_mae']:.4f}")

            if patience_counter >= self.early_stopping_patience:
                if show_progress:
                    epoch_iter.close()
                elif verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.network.load_state_dict(best_state)

        return {"history": history}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.network.eval()
        all_thresh_continuous = []
        all_thresh_labels = []
        all_runtime_preds = []
        all_runtime_labels = []

        for batch in loader:
            features = batch["features"].to(self.device)
            threshold_labels = batch["threshold_class"]
            runtime_labels = batch["log_runtime"]

            features = self._normalize(features)
            threshold_pred, runtime_pred = self.network(features)

            all_thresh_continuous.extend(threshold_pred.cpu().tolist())
            all_thresh_labels.extend(threshold_labels.tolist())
            all_runtime_preds.extend(runtime_pred.cpu().tolist())
            all_runtime_labels.extend(runtime_labels.tolist())

        pred_rungs = [round_up_to_rung(p) for p in all_thresh_continuous]
        pred_classes = [THRESHOLD_LADDER.index(r) if r in THRESHOLD_LADDER else len(THRESHOLD_LADDER) - 1 for r in pred_rungs]
        threshold_accuracy = accuracy_score(all_thresh_labels, pred_classes)

        return {
            "threshold_accuracy": threshold_accuracy,
            "threshold_mae": mean_absolute_error(
                [THRESHOLD_LADDER[c] for c in all_thresh_labels],
                all_thresh_continuous,
            ),
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
        threshold_continuous, runtime_pred = self.network(features)
        thresh_continuous_np = threshold_continuous.cpu().numpy()
        thresh_values = np.array([round_up_to_rung(float(p)) for p in thresh_continuous_np])
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
        }, path / "mlp_continuous_model.pt")

    def load(self, path: Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path / "mlp_continuous_model.pt", map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.feature_mean = checkpoint["feature_mean"]
        self.feature_std = checkpoint["feature_std"]

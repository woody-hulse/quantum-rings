"""
Advanced Training Module for State-of-the-Art GNN.

Implements modern training techniques:
1. Cosine Annealing with Warm Restarts
2. Gradient Accumulation for larger effective batch sizes
3. Mixed Precision Training (AMP)
4. Exponential Moving Average (EMA) for model weights
5. Deep Supervision with auxiliary losses
6. Label Smoothing for classification
7. Graph Mixup for regularization
8. Multi-objective loss balancing
9. Gradient clipping with adaptive thresholds
10. Learning rate warmup
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import math
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, LambdaLR
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data, Batch
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from tqdm import tqdm


# =============================================================================
# EXPONENTIAL MOVING AVERAGE
# =============================================================================

class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of model weights that are updated as an
    exponential moving average of the training weights.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# GRAPH MIXUP
# =============================================================================

class GraphMixup:
    """
    Mixup augmentation for graphs.
    
    Interpolates node features and targets between pairs of graphs
    within a batch for regularization.
    """
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        batch: Batch,
        targets: torch.Tensor,
    ) -> Tuple[Batch, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch.
        
        Returns:
            Mixed batch, original targets, shuffled targets, lambda value
        """
        if np.random.random() > self.prob:
            return batch, targets, targets, 1.0
        
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # Ensure lambda >= 0.5
        
        # Get batch size
        batch_size = batch.batch.max().item() + 1
        
        # Random permutation
        perm = torch.randperm(batch_size)
        
        # Mix node features within same-position graphs
        # This is a simplified version that mixes global features
        batch.global_features = lam * batch.global_features + (1 - lam) * batch.global_features[perm]
        
        # Return shuffled targets for mixup loss
        targets_shuffled = targets[perm]
        
        return batch, targets, targets_shuffled, lam


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DeepSupervisionLoss(nn.Module):
    """
    Loss function with deep supervision.
    
    Combines main prediction loss with auxiliary losses from intermediate layers,
    with decreasing weights for earlier layers.
    """
    
    def __init__(
        self,
        base_loss: nn.Module = nn.L1Loss(),
        aux_weight: float = 0.3,
        aux_decay: float = 0.7,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.aux_weight = aux_weight
        self.aux_decay = aux_decay
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux_preds: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        main_loss = self.base_loss(pred, target)
        
        if aux_preds is not None and len(aux_preds) > 0:
            aux_loss = 0.0
            weight = self.aux_weight
            
            for aux_pred in reversed(aux_preds):
                aux_loss += weight * self.base_loss(aux_pred, target)
                weight *= self.aux_decay
            
            return main_loss + aux_loss
        
        return main_loss


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for threshold class prediction.
    
    Treats the classification as cumulative probabilities,
    preserving the ordinal nature of threshold classes.
    """
    
    def __init__(self, num_classes: int = 9, label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        ordinal_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: Class logits [B, num_classes]
            ordinal_logits: Cumulative logits [B, num_classes - 1]
            targets: Class indices [B]
        
        Returns:
            Classification loss and ordinal loss
        """
        # Standard cross-entropy with optional label smoothing
        if self.label_smoothing > 0:
            # Create smoothed labels
            n_classes = logits.size(-1)
            one_hot = F.one_hot(targets, n_classes).float()
            smooth = self.label_smoothing / (n_classes - 1)
            one_hot = one_hot * (1 - self.label_smoothing) + smooth * (1 - one_hot)
            ce_loss = -(one_hot * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        else:
            ce_loss = F.cross_entropy(logits, targets)
        
        # Ordinal regression: P(y > k) for k = 0, ..., num_classes - 2
        ordinal_targets = torch.zeros_like(ordinal_logits)
        for i in range(ordinal_logits.size(-1)):
            ordinal_targets[:, i] = (targets > i).float()
        
        ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets)
        
        return ce_loss, ordinal_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces the loss for well-classified examples, focusing training
    on hard, misclassified examples.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# LEARNING RATE SCHEDULERS
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
):
    """
    Cosine learning rate schedule with linear warmup.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# ADVANCED TRAINER
# =============================================================================

class AdvancedGNNTrainer:
    """
    Advanced trainer for State-of-the-Art GNN models.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Exponential moving average
    - Deep supervision
    - Graph mixup
    - Automatic mixed precision
    - Gradient clipping
    - Learning rate warmup
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_amp: bool = False,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_deep_supervision: bool = True,
        aux_loss_weight: float = 0.3,
        use_mixup: bool = True,
        mixup_alpha: float = 0.2,
        warmup_epochs: int = 5,
        label_smoothing: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device != "cpu"
        self.use_ema = use_ema
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_deep_supervision = use_deep_supervision
        self.warmup_epochs = warmup_epochs
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # EMA
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        
        # Loss functions
        self.runtime_criterion = DeepSupervisionLoss(
            base_loss=nn.HuberLoss(delta=1.0),
            aux_weight=aux_loss_weight,
        ) if use_deep_supervision else nn.HuberLoss(delta=1.0)
        
        # Mixup
        self.mixup = GraphMixup(alpha=mixup_alpha) if use_mixup else None
        
        # Scheduler (will be set in fit())
        self.scheduler = None
        
        # Tracking
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.best_state = None
    
    def _forward_with_amp(
        self,
        batch: Batch,
        return_aux: bool = False,
    ):
        """Forward pass with optional AMP."""
        if self.use_amp:
            with autocast():
                return self._forward(batch, return_aux)
        return self._forward(batch, return_aux)
    
    def _forward(self, batch: Batch, return_aux: bool = False):
        """Forward pass."""
        return self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_gate_type=batch.edge_gate_type,
            batch=batch.batch,
            global_features=batch.global_features,
            return_aux=return_aux,
        )
    
    def train_epoch(
        self,
        loader: PyGDataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            targets = batch.log2_runtime
            
            # Apply mixup
            if self.mixup is not None:
                batch, targets, targets_shuffled, lam = self.mixup(batch, targets)
            else:
                lam = 1.0
                targets_shuffled = targets
            
            # Forward pass
            if self.use_deep_supervision:
                pred, aux_preds = self._forward_with_amp(batch, return_aux=True)
                
                # Compute mixed loss
                loss1 = self.runtime_criterion(pred, targets, aux_preds)
                if lam < 1.0:
                    loss2 = self.runtime_criterion(pred, targets_shuffled, aux_preds)
                    loss = lam * loss1 + (1 - lam) * loss2
                else:
                    loss = loss1
            else:
                pred = self._forward_with_amp(batch, return_aux=False)
                
                if isinstance(self.runtime_criterion, DeepSupervisionLoss):
                    loss1 = self.runtime_criterion.base_loss(pred, targets)
                else:
                    loss1 = self.runtime_criterion(pred, targets)
                
                if lam < 1.0:
                    if isinstance(self.runtime_criterion, DeepSupervisionLoss):
                        loss2 = self.runtime_criterion.base_loss(pred, targets_shuffled)
                    else:
                        loss2 = self.runtime_criterion(pred, targets_shuffled)
                    loss = lam * loss1 + (1 - lam) * loss2
                else:
                    loss = loss1
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{total_loss / n_batches:.4f}'})
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {'loss': total_loss / n_batches}
    
    @torch.no_grad()
    def evaluate(
        self,
        loader: PyGDataLoader,
        use_ema: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the model."""
        # Apply EMA weights for evaluation
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        for batch in loader:
            batch = batch.to(self.device)
            pred = self._forward(batch, return_aux=False)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            all_preds.extend(pred.cpu().tolist())
            all_targets.extend(batch.log2_runtime.cpu().tolist())
        
        # Restore original weights
        if use_ema and self.ema is not None:
            self.ema.restore()
        
        return {
            'runtime_mse': mean_squared_error(all_targets, all_preds),
            'runtime_mae': mean_absolute_error(all_targets, all_preds),
        }
    
    @torch.no_grad()
    def predict(
        self,
        loader: PyGDataLoader,
        use_ema: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions."""
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        all_thresh = []
        all_runtime = []
        
        for batch in loader:
            batch = batch.to(self.device)
            pred = self._forward(batch, return_aux=False)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            runtime_values = np.power(2.0, pred.cpu().numpy())
            thresh = getattr(batch, 'threshold', None)
            
            if thresh is not None:
                thresh_values = thresh.cpu().numpy() if hasattr(thresh, 'cpu') else np.array(thresh)
            else:
                thresh_values = np.zeros(len(runtime_values))
            
            all_thresh.extend(thresh_values.tolist())
            all_runtime.extend(runtime_values.tolist())
        
        if use_ema and self.ema is not None:
            self.ema.restore()
        
        return np.array(all_thresh), np.array(all_runtime)
    
    def fit(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Full training loop."""
        # Setup scheduler
        num_training_steps = epochs * len(train_loader) // self.gradient_accumulation_steps
        num_warmup_steps = self.warmup_epochs * len(train_loader) // self.gradient_accumulation_steps
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=0.01,
        )
        
        history = {
            'train_loss': [],
            'val_mae': [],
            'val_mse': [],
            'lr': [],
        }
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Record history
            history['train_loss'].append(train_metrics['loss'])
            history['val_mae'].append(val_metrics['runtime_mae'])
            history['val_mse'].append(val_metrics['runtime_mse'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Check for improvement
            if val_metrics['runtime_mae'] < self.best_val_metric:
                self.best_val_metric = val_metrics['runtime_mae']
                patience_counter = 0
                
                # Save best state (from EMA if available)
                if self.ema is not None:
                    self.best_state = {k: v.clone() for k, v in self.ema.shadow.items()}
                else:
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Loss: {train_metrics['loss']:.4f} | "
                      f"Val MAE: {val_metrics['runtime_mae']:.4f} | "
                      f"Val MSE: {val_metrics['runtime_mse']:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best state
        if self.best_state is not None:
            if self.ema is not None:
                self.ema.shadow = self.best_state
                self.ema.apply_shadow()
            else:
                self.model.load_state_dict(self.best_state)
        
        return {'history': history}
    
    def save(self, path: Path):
        """Save model and training state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'global_step': self.global_step,
        }
        
        if self.ema is not None:
            state['ema_shadow'] = self.ema.shadow
        
        if self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()
        
        torch.save(state, path / 'checkpoint.pt')
    
    def load(self, path: Path):
        """Load model and training state."""
        path = Path(path)
        state = torch.load(path / 'checkpoint.pt', map_location=self.device)
        
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.best_val_metric = state['best_val_metric']
        self.global_step = state['global_step']
        
        if self.ema is not None and 'ema_shadow' in state:
            self.ema.shadow = state['ema_shadow']
        
        if self.scheduler is not None and 'scheduler_state' in state:
            self.scheduler.load_state_dict(state['scheduler_state'])


# =============================================================================
# THRESHOLD CLASS TRAINER
# =============================================================================

class AdvancedThresholdClassTrainer:
    """
    Advanced trainer for threshold class prediction.
    
    Uses ordinal regression loss and focal loss for class imbalance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_amp: bool = False,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_epochs: int = 5,
        label_smoothing: float = 0.1,
        ordinal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        num_classes: int = 9,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device != "cpu"
        self.use_ema = use_ema
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = warmup_epochs
        self.ordinal_weight = ordinal_weight
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # EMA
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        
        # Loss functions
        self.ordinal_loss = OrdinalRegressionLoss(num_classes, label_smoothing)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        
        # Scheduler
        self.scheduler = None
        
        # Tracking
        self.global_step = 0
        self.best_val_metric = 0.0  # Accuracy (higher is better)
        self.best_state = None
    
    def _forward(self, batch: Batch, return_ordinal: bool = True):
        return self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_gate_type=batch.edge_gate_type,
            batch=batch.batch,
            global_features=batch.global_features,
            return_ordinal=return_ordinal,
        )
    
    def train_epoch(
        self,
        loader: PyGDataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_ordinal_loss = 0.0
        n_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            targets = batch.threshold_class
            
            # Forward
            if self.use_amp:
                with autocast():
                    logits, ordinal_logits = self._forward(batch, return_ordinal=True)
                    ce_loss, ordinal_loss = self.ordinal_loss(logits, ordinal_logits, targets)
                    focal = self.focal_loss(logits, targets)
                    loss = ce_loss + self.ordinal_weight * ordinal_loss + 0.5 * focal
            else:
                logits, ordinal_logits = self._forward(batch, return_ordinal=True)
                ce_loss, ordinal_loss = self.ordinal_loss(logits, ordinal_logits, targets)
                focal = self.focal_loss(logits, targets)
                loss = ce_loss + self.ordinal_weight * ordinal_loss + 0.5 * focal
            
            loss = loss / self.gradient_accumulation_steps
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.ema is not None:
                    self.ema.update()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_ce_loss += ce_loss.item()
            total_ordinal_loss += ordinal_loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{total_loss / n_batches:.4f}'})
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {
            'loss': total_loss / n_batches,
            'ce_loss': total_ce_loss / n_batches,
            'ordinal_loss': total_ordinal_loss / n_batches,
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        loader: PyGDataLoader,
        use_ema: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the model."""
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        all_preds = []
        all_targets = []
        
        for batch in loader:
            batch = batch.to(self.device)
            logits = self._forward(batch, return_ordinal=False)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch.threshold_class.cpu().tolist())
        
        if use_ema and self.ema is not None:
            self.ema.restore()
        
        accuracy = accuracy_score(all_targets, all_preds)
        
        # Compute MAE in terms of class distance
        mae_class = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
        
        return {
            'accuracy': accuracy,
            'mae_class': mae_class,
        }
    
    def fit(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Full training loop."""
        num_training_steps = epochs * len(train_loader) // self.gradient_accumulation_steps
        num_warmup_steps = self.warmup_epochs * len(train_loader) // self.gradient_accumulation_steps
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=0.01,
        )
        
        history = {
            'train_loss': [],
            'val_accuracy': [],
            'val_mae_class': [],
        }
        
        patience_counter = 0
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)
            
            history['train_loss'].append(train_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_mae_class'].append(val_metrics['mae_class'])
            
            # Check for improvement (accuracy: higher is better)
            if val_metrics['accuracy'] > self.best_val_metric:
                self.best_val_metric = val_metrics['accuracy']
                patience_counter = 0
                
                if self.ema is not None:
                    self.best_state = {k: v.clone() for k, v in self.ema.shadow.items()}
                else:
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Loss: {train_metrics['loss']:.4f} | "
                      f"Val Acc: {val_metrics['accuracy']:.4f} | "
                      f"Val MAE Class: {val_metrics['mae_class']:.4f}")
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best state
        if self.best_state is not None:
            if self.ema is not None:
                self.ema.shadow = self.best_state
                self.ema.apply_shadow()
            else:
                self.model.load_state_dict(self.best_state)
        
        return {'history': history}


if __name__ == "__main__":
    print("Advanced Training Module loaded successfully!")
    print("\nAvailable components:")
    print("  - EMA: Exponential Moving Average")
    print("  - GraphMixup: Graph-level mixup augmentation")
    print("  - DeepSupervisionLoss: Multi-scale supervision")
    print("  - OrdinalRegressionLoss: Ordinal classification")
    print("  - FocalLoss: Class imbalance handling")
    print("  - AdvancedGNNTrainer: Full-featured trainer for regression")
    print("  - AdvancedThresholdClassTrainer: Trainer for classification")

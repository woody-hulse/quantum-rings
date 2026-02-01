"""
Temporal GNN wrapper models conforming to the base model interface.

These wrappers integrate the cutting-edge Temporal GNN architectures with the
existing training and evaluation infrastructure.
"""

from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from .base import BaseModel, ThresholdClassBaseModel


class TemporalGNNDurationModel(BaseModel):
    """
    Wrapper for Temporal GNN duration prediction model.
    
    Conforms to the BaseModel interface for integration with the
    training and evaluation pipeline.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.15,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 20,
        device: Optional[str] = None,
        use_state_memory: bool = True,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.use_state_memory = use_state_memory
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.history: List[Dict[str, float]] = []
    
    @property
    def name(self) -> str:
        return f"TemporalGNN(h={self.hidden_dim},L={self.num_layers})"
    
    def _build_model(self, sample_batch) -> None:
        """Build model from sample batch to infer dimensions."""
        from gnn.temporal_model import create_temporal_gnn_model
        from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE
        
        node_feat_dim = sample_batch.x.size(-1)
        edge_feat_dim = sample_batch.edge_attr.size(-1)
        global_feat_dim = sample_batch.global_features.size(-1)
        
        self.model = create_temporal_gnn_model(
            model_type="duration",
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            use_state_memory=self.use_state_memory,
        )
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        sample_batch = next(iter(train_loader))
        self._build_model(sample_batch)
        
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        epoch_iter = range(self.epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training")
        
        for epoch in epoch_iter:
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.edge_gate_type,
                    batch.batch,
                    batch.global_features,
                )
                loss = F.l1_loss(pred, batch.log2_runtime)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= max(n_batches, 1)
            
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["runtime_mae"]
            
            self.scheduler.step(val_loss)
            
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae": val_loss,
            })
            
            if verbose:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_mae={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)
        
        return {
            "history": self.history,
            "best_val_mae": best_val_loss,
        }
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "TemporalGNNDurationModel requires graph inputs. Use predict_batch instead."
        )
    
    def predict_batch(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on a PyG data loader."""
        self.model.eval()
        all_thresholds = []
        all_runtimes = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred_log2_runtime = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.edge_gate_type,
                    batch.batch,
                    batch.global_features,
                )
                
                if hasattr(batch, "threshold"):
                    thresholds = batch.threshold.cpu().numpy()
                else:
                    thresholds = np.ones(pred_log2_runtime.size(0))
                
                runtimes = np.power(2, pred_log2_runtime.cpu().numpy())
                
                all_thresholds.extend(thresholds.tolist())
                all_runtimes.extend(runtimes.tolist())
        
        return np.array(all_thresholds), np.array(all_runtimes)
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_l1 = 0.0
        total_l2 = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.edge_gate_type,
                    batch.batch,
                    batch.global_features,
                )
                target = batch.log2_runtime
                
                total_l1 += F.l1_loss(pred, target, reduction="sum").item()
                total_l2 += F.mse_loss(pred, target, reduction="sum").item()
                n_samples += target.size(0)
        
        return {
            "runtime_mae": total_l1 / max(n_samples, 1),
            "runtime_mse": total_l2 / max(n_samples, 1),
        }
    
    def save(self, path: Path) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "use_state_memory": self.use_state_memory,
            },
        }, path)
    
    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint["config"]
        
        from gnn.temporal_model import create_temporal_gnn_model
        self.model = create_temporal_gnn_model(
            model_type="duration",
            **config,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model = self.model.to(self.device)


class TemporalGNNThresholdClassModel(ThresholdClassBaseModel):
    """
    Wrapper for Temporal GNN threshold classification model.
    
    Conforms to the ThresholdClassBaseModel interface.
    """
    
    THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.15,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 20,
        device: Optional[str] = None,
        use_ordinal: bool = True,
        use_state_memory: bool = True,
        conservative_weight: float = 1.5,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.use_ordinal = use_ordinal
        self.use_state_memory = use_state_memory
        self.conservative_weight = conservative_weight
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.history: List[Dict[str, float]] = []
    
    @property
    def name(self) -> str:
        return f"TemporalGNNThreshold(h={self.hidden_dim},L={self.num_layers})"
    
    def _build_model(self, sample_batch) -> None:
        from gnn.temporal_model import create_temporal_gnn_model
        
        node_feat_dim = sample_batch.x.size(-1)
        edge_feat_dim = sample_batch.edge_attr.size(-1)
        global_feat_dim = sample_batch.global_features.size(-1)
        
        self.model = create_temporal_gnn_model(
            model_type="threshold",
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            use_ordinal=self.use_ordinal,
            use_state_memory=self.use_state_memory,
        )
        self.model = self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_ordinal:
            from gnn.improved_model import OrdinalLoss
            loss_fn = OrdinalLoss(
                num_classes=len(self.THRESHOLD_LADDER),
                conservative_weight=self.conservative_weight,
            )
            return loss_fn(logits, targets)
        else:
            return F.cross_entropy(logits, targets)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        sample_batch = next(iter(train_loader))
        self._build_model(sample_batch)
        
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        
        epoch_iter = range(self.epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training")
        
        for epoch in epoch_iter:
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.edge_gate_type,
                    batch.batch,
                    batch.global_features,
                )
                
                loss = self._compute_loss(logits, batch.threshold_class)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= max(n_batches, 1)
            
            val_metrics = self.evaluate(val_loader)
            val_acc = val_metrics["threshold_accuracy"]
            
            self.scheduler.step(val_acc)
            
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
            })
            
            if verbose:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)
        
        return {
            "history": self.history,
            "best_val_accuracy": best_val_acc,
        }
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "TemporalGNNThresholdClassModel requires graph inputs. Use predict_proba_batch instead."
        )
    
    def predict_proba_batch(self, loader: DataLoader) -> np.ndarray:
        """Get class probabilities for a PyG data loader."""
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                probs = self.model.get_class_probs(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.edge_gate_type,
                    batch.batch,
                    batch.global_features,
                )
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    def predict_batch(
        self,
        loader: DataLoader,
        use_safety_margin: bool = True,
        safety_margin: int = 1,
        min_confidence: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict threshold values from a PyG data loader."""
        proba = self.predict_proba_batch(loader)
        
        if use_safety_margin:
            from scoring import select_threshold_with_safety_margin
            chosen = select_threshold_with_safety_margin(
                proba,
                safety_margin=safety_margin,
                min_confidence=min_confidence,
            )
        else:
            from scoring import select_threshold_class_by_expected_score
            chosen = select_threshold_class_by_expected_score(proba)
        
        threshold_values = np.array([self.THRESHOLD_LADDER[c] for c in chosen])
        runtime_values = np.ones_like(threshold_values, dtype=float)
        
        return threshold_values, runtime_values
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        underpred = 0
        overpred = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                probs = self.model.get_class_probs(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.edge_gate_type,
                    batch.batch,
                    batch.global_features,
                )
                pred_class = probs.argmax(dim=-1)
                target_class = batch.threshold_class
                
                correct += (pred_class == target_class).sum().item()
                underpred += (pred_class < target_class).sum().item()
                overpred += (pred_class > target_class).sum().item()
                total += target_class.size(0)
        
        return {
            "threshold_accuracy": correct / max(total, 1),
            "underprediction_rate": underpred / max(total, 1),
            "overprediction_rate": overpred / max(total, 1),
        }
    
    def save(self, path: Path) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "use_ordinal": self.use_ordinal,
                "use_state_memory": self.use_state_memory,
            },
        }, path)
    
    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint["config"]
        
        from gnn.temporal_model import create_temporal_gnn_model
        self.model = create_temporal_gnn_model(
            model_type="threshold",
            **config,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model = self.model.to(self.device)

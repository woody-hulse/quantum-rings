"""
Unified Graph Model Wrappers for Threshold Classification.

This module provides wrapper classes that adapt all GNN architectures to the
ThresholdClassBaseModel interface, enabling consistent training and evaluation.

Available Models:
- BasicGNNThresholdClassModel: Simple message-passing GNN
- ImprovedGNNThresholdClassModel: Attention-based GNN with ordinal regression
- GraphTransformerThresholdClassModel: Full transformer attention
- HeteroGNNThresholdClassModel: Heterogeneous multi-relation GNN
- TemporalGNNThresholdClassModel: Temporal/causal modeling GNN

All models share a common interface for:
- fit(train_loader, val_loader) -> training metrics
- predict_proba(loader) -> class probabilities
- evaluate(loader) -> evaluation metrics
- predict(loader) -> threshold values
"""

from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import ThresholdClassBaseModel


THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
NUM_THRESHOLD_CLASSES = len(THRESHOLD_LADDER)


@dataclass
class GraphModelConfig:
    """Common configuration for all graph models."""
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 20
    use_ordinal: bool = True
    use_augmentation: bool = True
    augmentation_strength: float = 0.5
    device: Optional[str] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseGraphModelWrapper(ThresholdClassBaseModel):
    """Base class for all graph model wrappers."""
    
    def __init__(self, config: GraphModelConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.history: List[Dict[str, float]] = []
        self._loss_fn = None
    
    @property
    def name(self) -> str:
        raise NotImplementedError
    
    def _create_model(self, sample_batch) -> nn.Module:
        """Create the underlying GNN model. Override in subclasses."""
        raise NotImplementedError
    
    def _build_model(self, sample_batch) -> None:
        """Build model from sample batch to infer dimensions."""
        self.model = self._create_model(sample_batch)
        self.model = self.model.to(self.config.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for threshold classification."""
        return F.cross_entropy(logits, targets)
    
    def _get_class_probs(self, batch) -> torch.Tensor:
        """Get class probabilities from model output."""
        logits = self.model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_gate_type,
            batch.batch,
            batch.global_features,
        )
        
        if self.config.use_ordinal:
            cumulative_probs = torch.sigmoid(logits)
            n_classes = NUM_THRESHOLD_CLASSES
            batch_size = logits.size(0)
            
            class_probs = torch.zeros(batch_size, n_classes, device=logits.device)
            class_probs[:, 0] = 1 - cumulative_probs[:, 0]
            for k in range(1, n_classes - 1):
                class_probs[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]
            class_probs[:, -1] = cumulative_probs[:, -1]
            class_probs = class_probs.clamp(min=1e-7)
            class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
            return class_probs
        else:
            return F.softmax(logits, dim=-1)
    
    def _get_augmentation(self):
        """Get data augmentation transform if enabled."""
        if not self.config.use_augmentation:
            return None
        
        from gnn.augmentation import get_train_augmentation
        return get_train_augmentation(
            qubit_perm_p=self.config.augmentation_strength,
            edge_dropout_p=0.1 * self.config.augmentation_strength,
            feature_noise_std=0.1 * self.config.augmentation_strength,
            temporal_jitter_std=0.05 * self.config.augmentation_strength,
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
        
        augmentation = self._get_augmentation()
        
        best_val_score = -float("inf")
        best_state = None
        patience_counter = 0
        
        epoch_iter = range(self.config.epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc=f"Training {self.name}", leave=False)
        
        for epoch in epoch_iter:
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                if augmentation is not None:
                    batch = augmentation(batch)
                batch = batch.to(self.config.device)
                
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
            val_score = val_metrics.get("expected_threshold_score", val_metrics["threshold_accuracy"])
            
            self.scheduler.step(val_score)
            
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["threshold_accuracy"],
                "val_score": val_score,
            })
            
            if show_progress:
                epoch_iter.set_postfix({
                    "loss": f"{train_loss:.3f}",
                    "val_acc": f"{val_metrics['threshold_accuracy']:.3f}",
                    "val_score": f"{val_score:.3f}",
                })
            
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.config.device)
        
        return {
            "history": self.history,
            "best_val_score": best_val_score,
        }
    
    def predict_proba(self, features: Union[np.ndarray, DataLoader]) -> np.ndarray:
        """Get class probabilities. Features must be a PyG DataLoader."""
        if isinstance(features, np.ndarray):
            raise TypeError(f"{self.name} requires a PyG DataLoader, not numpy array.")
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in features:
                batch = batch.to(self.config.device)
                probs = self._get_class_probs(batch)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.config.device)
                probs = self._get_class_probs(batch)
                all_probs.append(probs.cpu().numpy())
                all_targets.extend(batch.threshold_class.cpu().tolist())
        
        probs = np.concatenate(all_probs, axis=0)
        targets = np.array(all_targets)
        
        from scoring import select_threshold_class_by_expected_score, mean_threshold_score
        
        pred_class = select_threshold_class_by_expected_score(probs)
        
        accuracy = float(np.mean(pred_class == targets))
        threshold_score = mean_threshold_score(pred_class, targets)
        underpred = float(np.mean(pred_class < targets))
        overpred = float(np.mean(pred_class > targets))
        
        return {
            "threshold_accuracy": accuracy,
            "expected_threshold_score": threshold_score,
            "underprediction_rate": underpred,
            "overprediction_rate": overpred,
        }
    
    def predict(
        self,
        features: Union[np.ndarray, DataLoader],
        use_safety_margin: bool = True,
        safety_margin: int = 1,
        min_confidence: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict threshold values."""
        probs = self.predict_proba(features)
        
        if use_safety_margin:
            from scoring import select_threshold_with_safety_margin
            chosen = select_threshold_with_safety_margin(
                probs,
                safety_margin=safety_margin,
                min_confidence=min_confidence,
            )
        else:
            from scoring import select_threshold_class_by_expected_score
            chosen = select_threshold_class_by_expected_score(probs)
        
        threshold_values = np.array([THRESHOLD_LADDER[c] for c in chosen])
        runtime_values = np.ones_like(threshold_values, dtype=float)
        
        return threshold_values, runtime_values
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def save(self, path: Path) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config.__dict__,
        }, path)
    
    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model = self.model.to(self.config.device)


class BasicGNNThresholdClassModel(BaseGraphModelWrapper):
    """Basic message-passing GNN for threshold classification.
    
    Note: This model does not support ordinal regression - it uses
    standard cross-entropy loss with softmax outputs.
    """
    
    def __init__(self, config: GraphModelConfig):
        config_copy = GraphModelConfig(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            patience=config.patience,
            use_ordinal=False,  # BasicGNN doesn't support ordinal
            use_augmentation=config.use_augmentation,
            augmentation_strength=config.augmentation_strength,
            device=config.device,
        )
        super().__init__(config_copy)
    
    @property
    def name(self) -> str:
        return f"BasicGNN(h={self.config.hidden_dim},L={self.config.num_layers})"
    
    def _create_model(self, sample_batch) -> nn.Module:
        from gnn.model import create_gnn_threshold_class_model
        
        return create_gnn_threshold_class_model(
            node_feat_dim=sample_batch.x.size(-1),
            edge_feat_dim=sample_batch.edge_attr.size(-1),
            global_feat_dim=sample_batch.global_features.size(-1),
            num_classes=NUM_THRESHOLD_CLASSES,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
    
    def _get_class_probs(self, batch) -> torch.Tensor:
        logits = self.model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_gate_type,
            batch.batch,
            batch.global_features,
        )
        return F.softmax(logits, dim=-1)


class ImprovedGNNThresholdClassModel(BaseGraphModelWrapper):
    """Improved GNN with attention and ordinal regression."""
    
    def __init__(self, config: GraphModelConfig, stochastic_depth: float = 0.1):
        super().__init__(config)
        self.stochastic_depth = stochastic_depth
        self._ordinal_loss = None
    
    @property
    def name(self) -> str:
        return f"ImprovedGNN(h={self.config.hidden_dim},L={self.config.num_layers})"
    
    def _create_model(self, sample_batch) -> nn.Module:
        from gnn.improved_model import ImprovedQuantumCircuitGNNThresholdClass
        
        return ImprovedQuantumCircuitGNNThresholdClass(
            node_feat_dim=sample_batch.x.size(-1),
            edge_feat_dim=sample_batch.edge_attr.size(-1),
            global_feat_dim=sample_batch.global_features.size(-1),
            num_classes=NUM_THRESHOLD_CLASSES,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            stochastic_depth=self.stochastic_depth,
            use_ordinal=self.config.use_ordinal,
        )
    
    def _build_model(self, sample_batch) -> None:
        super()._build_model(sample_batch)
        if self.config.use_ordinal:
            from gnn.improved_model import OrdinalLoss
            self._ordinal_loss = OrdinalLoss(num_classes=NUM_THRESHOLD_CLASSES)
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.use_ordinal and self._ordinal_loss is not None:
            return self._ordinal_loss(logits, targets)
        return F.cross_entropy(logits, targets)
    
    def _get_class_probs(self, batch) -> torch.Tensor:
        return self.model.predict_proba(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_gate_type,
            batch.batch,
            batch.global_features,
        )


class GraphTransformerThresholdClassModel(BaseGraphModelWrapper):
    """Graph Transformer for threshold classification."""
    
    def __init__(self, config: GraphModelConfig, use_positional_encoding: bool = True):
        super().__init__(config)
        self.use_positional_encoding = use_positional_encoding
        self._ordinal_loss = None
    
    @property
    def name(self) -> str:
        return f"GraphTransformer(h={self.config.hidden_dim},L={self.config.num_layers})"
    
    def _create_model(self, sample_batch) -> nn.Module:
        from gnn.transformer import QuantumCircuitGraphTransformerThresholdClass
        
        return QuantumCircuitGraphTransformerThresholdClass(
            node_feat_dim=sample_batch.x.size(-1),
            edge_feat_dim=sample_batch.edge_attr.size(-1),
            global_feat_dim=sample_batch.global_features.size(-1),
            num_classes=NUM_THRESHOLD_CLASSES,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            use_positional_encoding=self.use_positional_encoding,
            use_ordinal=self.config.use_ordinal,
        )
    
    def _build_model(self, sample_batch) -> None:
        super()._build_model(sample_batch)
        if self.config.use_ordinal:
            from gnn.improved_model import OrdinalLoss
            self._ordinal_loss = OrdinalLoss(num_classes=NUM_THRESHOLD_CLASSES)
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.use_ordinal and self._ordinal_loss is not None:
            return self._ordinal_loss(logits, targets)
        return F.cross_entropy(logits, targets)
    
    def _get_class_probs(self, batch) -> torch.Tensor:
        return self.model.predict_proba(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_gate_type,
            batch.batch,
            batch.global_features,
        )


class HeteroGNNThresholdClassModel(BaseGraphModelWrapper):
    """Heterogeneous GNN (QCHGT) for threshold classification.
    
    Note: This model uses standard cross-entropy loss (no ordinal regression).
    """
    
    def __init__(self, config: GraphModelConfig):
        config_copy = GraphModelConfig(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            patience=config.patience,
            use_ordinal=False,  # HeteroGNN doesn't support ordinal
            use_augmentation=config.use_augmentation,
            augmentation_strength=config.augmentation_strength,
            device=config.device,
        )
        super().__init__(config_copy)
    
    @property
    def name(self) -> str:
        return f"HeteroGNN(h={self.config.hidden_dim},L={self.config.num_layers})"
    
    def _create_model(self, sample_batch) -> nn.Module:
        from gnn.hetero_gnn import QuantumCircuitHeteroGNN
        
        return QuantumCircuitHeteroGNN(
            node_feat_dim=sample_batch.x.size(-1),
            edge_feat_dim=sample_batch.edge_attr.size(-1),
            global_feat_dim=sample_batch.global_features.size(-1),
            num_classes=NUM_THRESHOLD_CLASSES,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )
    
    def _get_class_probs(self, batch) -> torch.Tensor:
        logits = self.model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_gate_type,
            batch.batch,
            batch.global_features,
        )
        return F.softmax(logits, dim=-1)


class TemporalGNNThresholdClassModelV2(BaseGraphModelWrapper):
    """Temporal GNN for threshold classification (new wrapper)."""
    
    def __init__(self, config: GraphModelConfig, use_state_memory: bool = True):
        super().__init__(config)
        self.use_state_memory = use_state_memory
        self._ordinal_loss = None
    
    @property
    def name(self) -> str:
        return f"TemporalGNN(h={self.config.hidden_dim},L={self.config.num_layers})"
    
    def _create_model(self, sample_batch) -> nn.Module:
        from gnn.temporal_model import TemporalQuantumCircuitGNNThresholdClass
        
        return TemporalQuantumCircuitGNNThresholdClass(
            node_feat_dim=sample_batch.x.size(-1),
            edge_feat_dim=sample_batch.edge_attr.size(-1),
            global_feat_dim=sample_batch.global_features.size(-1),
            num_classes=NUM_THRESHOLD_CLASSES,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            use_ordinal=self.config.use_ordinal,
            use_state_memory=self.use_state_memory,
        )
    
    def _build_model(self, sample_batch) -> None:
        super()._build_model(sample_batch)
        if self.config.use_ordinal:
            from gnn.improved_model import OrdinalLoss
            self._ordinal_loss = OrdinalLoss(num_classes=NUM_THRESHOLD_CLASSES)
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.use_ordinal and self._ordinal_loss is not None:
            return self._ordinal_loss(logits, targets)
        return F.cross_entropy(logits, targets)
    
    def _get_class_probs(self, batch) -> torch.Tensor:
        return self.model.get_class_probs(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_gate_type,
            batch.batch,
            batch.global_features,
        )


def create_graph_model(
    model_type: str,
    hidden_dim: int = 16,
    num_layers: int = 4,
    num_heads: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    epochs: int = 100,
    patience: int = 20,
    use_ordinal: bool = True,
    use_augmentation: bool = True,
    device: Optional[str] = None,
    **kwargs,
) -> BaseGraphModelWrapper:
    """
    Factory function to create graph models.
    
    Args:
        model_type: One of "basic", "improved", "transformer", "hetero", "temporal"
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        learning_rate: Learning rate
        weight_decay: Weight decay
        epochs: Maximum training epochs
        patience: Early stopping patience
        use_ordinal: Whether to use ordinal regression
        use_augmentation: Whether to use data augmentation
        device: Device (cpu/cuda)
        **kwargs: Additional model-specific arguments
        
    Returns:
        Graph model wrapper instance
    """
    config = GraphModelConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        use_ordinal=use_ordinal,
        use_augmentation=use_augmentation,
        device=device,
    )
    
    model_classes = {
        "basic": BasicGNNThresholdClassModel,
        "improved": ImprovedGNNThresholdClassModel,
        "transformer": GraphTransformerThresholdClassModel,
        "hetero": HeteroGNNThresholdClassModel,
        "temporal": TemporalGNNThresholdClassModelV2,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")
    
    model_class = model_classes[model_type]
    
    if model_type == "improved":
        return model_class(config, stochastic_depth=kwargs.get("stochastic_depth", 0.1))
    elif model_type == "transformer":
        return model_class(config, use_positional_encoding=kwargs.get("use_positional_encoding", True))
    elif model_type == "temporal":
        return model_class(config, use_state_memory=kwargs.get("use_state_memory", True))
    else:
        return model_class(config)


def get_all_model_types() -> List[str]:
    """Return list of all available model types."""
    return ["basic", "improved", "transformer", "hetero", "temporal"]


MODEL_DESCRIPTIONS = {
    "basic": "Simple message-passing GNN with per-gate-type embeddings",
    "improved": "Attention-based GNN with ordinal regression and stochastic depth",
    "transformer": "Graph Transformer with edge-aware attention and positional encoding",
    "hetero": "Heterogeneous GNN with multi-relation edges and meta-path attention",
    "temporal": "Temporal GNN with causal attention and state memory",
}


class BaseGraphDurationModelWrapper:
    """Base class for graph-based duration model wrappers."""
    
    def __init__(self, config: GraphModelConfig):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.history: List[Dict[str, float]] = []
    
    @property
    def name(self) -> str:
        raise NotImplementedError
    
    def _create_model(self, sample_batch) -> nn.Module:
        raise NotImplementedError
    
    def _build_model(self, sample_batch) -> None:
        self.model = self._create_model(sample_batch)
        self.model = self.model.to(self.config.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
    
    def _get_augmentation(self):
        if not self.config.use_augmentation:
            return None
        try:
            from gnn.augmentation import get_train_augmentation
            return get_train_augmentation(
                qubit_perm_p=self.config.augmentation_strength,
                edge_dropout_p=0.1 * self.config.augmentation_strength,
                feature_noise_std=0.1 * self.config.augmentation_strength,
                temporal_jitter_std=0.05 * self.config.augmentation_strength,
            )
        except ImportError:
            return None
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        sample_batch = next(iter(train_loader))
        self._build_model(sample_batch)
        
        augmentation = self._get_augmentation()
        
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        epoch_iter = range(self.config.epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc=f"Training {self.name}", leave=False)
        
        for epoch in epoch_iter:
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                if augmentation is not None:
                    batch = augmentation(batch)
                batch = batch.to(self.config.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.edge_gate_type,
                    batch.batch,
                    batch.global_features,
                )
                
                targets = batch.log2_runtime
                loss = F.mse_loss(predictions, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= max(n_batches, 1)
            
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["runtime_mse"]
            
            self.scheduler.step(val_loss)
            
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae": val_metrics["runtime_mae"],
                "val_mse": val_metrics["runtime_mse"],
            })
            
            if show_progress:
                epoch_iter.set_postfix({
                    "loss": f"{train_loss:.4f}",
                    "val_mae": f"{val_metrics['runtime_mae']:.4f}",
                })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.config.device)
        
        return {
            "history": self.history,
            "best_val_loss": best_val_loss,
        }
    
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.config.device)
                predictions = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.edge_gate_type,
                    batch.batch,
                    batch.global_features,
                )
                all_preds.append(predictions.cpu().numpy())
        
        return np.concatenate(all_preds, axis=0)
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.config.device)
                predictions = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.edge_gate_type,
                    batch.batch,
                    batch.global_features,
                )
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch.log2_runtime.cpu().numpy())
        
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        mae = float(np.mean(np.abs(preds - targets)))
        mse = float(np.mean((preds - targets) ** 2))
        
        return {
            "runtime_mae": mae,
            "runtime_mse": mse,
        }
    
    def count_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class BasicGNNDurationModel(BaseGraphDurationModelWrapper):
    """Basic message-passing GNN for duration prediction."""
    
    def __init__(self, config: GraphModelConfig):
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return f"BasicGNN-Duration(h={self.config.hidden_dim},L={self.config.num_layers})"
    
    def _create_model(self, sample_batch) -> nn.Module:
        from gnn.model import create_gnn_model
        
        return create_gnn_model(
            model_type="basic",
            node_feat_dim=sample_batch.x.size(-1),
            edge_feat_dim=sample_batch.edge_attr.size(-1),
            global_feat_dim=sample_batch.global_features.size(-1),
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )


class ImprovedGNNDurationModel(BaseGraphDurationModelWrapper):
    """Improved GNN with attention for duration prediction."""
    
    def __init__(self, config: GraphModelConfig, stochastic_depth: float = 0.1):
        super().__init__(config)
        self.stochastic_depth = stochastic_depth
    
    @property
    def name(self) -> str:
        return f"ImprovedGNN-Duration(h={self.config.hidden_dim},L={self.config.num_layers})"
    
    def _create_model(self, sample_batch) -> nn.Module:
        from gnn.improved_model import create_improved_gnn_model
        
        return create_improved_gnn_model(
            model_type="duration",
            node_feat_dim=sample_batch.x.size(-1),
            edge_feat_dim=sample_batch.edge_attr.size(-1),
            global_feat_dim=sample_batch.global_features.size(-1),
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            stochastic_depth=self.stochastic_depth,
        )


class GraphTransformerDurationModel(BaseGraphDurationModelWrapper):
    """Graph Transformer for duration prediction."""
    
    def __init__(self, config: GraphModelConfig, use_positional_encoding: bool = True):
        super().__init__(config)
        self.use_positional_encoding = use_positional_encoding
    
    @property
    def name(self) -> str:
        return f"GraphTransformer-Duration(h={self.config.hidden_dim},L={self.config.num_layers})"
    
    def _create_model(self, sample_batch) -> nn.Module:
        from gnn.transformer import create_graph_transformer_model
        
        return create_graph_transformer_model(
            model_type="duration",
            node_feat_dim=sample_batch.x.size(-1),
            edge_feat_dim=sample_batch.edge_attr.size(-1),
            global_feat_dim=sample_batch.global_features.size(-1),
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            use_positional_encoding=self.use_positional_encoding,
        )


class HeteroGNNDurationModel(BaseGraphDurationModelWrapper):
    """Heterogeneous GNN (QCHGT) for duration prediction."""
    
    def __init__(self, config: GraphModelConfig):
        super().__init__(config)
    
    @property
    def name(self) -> str:
        return f"HeteroGNN-Duration(h={self.config.hidden_dim},L={self.config.num_layers})"
    
    def _create_model(self, sample_batch) -> nn.Module:
        from gnn.hetero_gnn import create_hetero_gnn_model
        
        return create_hetero_gnn_model(
            model_type="duration",
            node_feat_dim=sample_batch.x.size(-1),
            edge_feat_dim=sample_batch.edge_attr.size(-1),
            global_feat_dim=sample_batch.global_features.size(-1),
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )


class TemporalGNNDurationModel(BaseGraphDurationModelWrapper):
    """Temporal GNN for duration prediction."""
    
    def __init__(self, config: GraphModelConfig, use_state_memory: bool = True):
        super().__init__(config)
        self.use_state_memory = use_state_memory
    
    @property
    def name(self) -> str:
        return f"TemporalGNN-Duration(h={self.config.hidden_dim},L={self.config.num_layers})"
    
    def _create_model(self, sample_batch) -> nn.Module:
        from gnn.temporal_model import create_temporal_gnn_model
        
        return create_temporal_gnn_model(
            model_type="duration",
            node_feat_dim=sample_batch.x.size(-1),
            edge_feat_dim=sample_batch.edge_attr.size(-1),
            global_feat_dim=sample_batch.global_features.size(-1),
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            use_state_memory=self.use_state_memory,
        )


def create_graph_duration_model(
    model_type: str,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 20,
    use_augmentation: bool = True,
    device: Optional[str] = None,
    **kwargs,
) -> BaseGraphDurationModelWrapper:
    """Factory function to create graph-based duration models."""
    config = GraphModelConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        use_ordinal=False,
        use_augmentation=use_augmentation,
        device=device,
    )
    
    model_classes = {
        "basic": BasicGNNDurationModel,
        "improved": ImprovedGNNDurationModel,
        "transformer": GraphTransformerDurationModel,
        "hetero": HeteroGNNDurationModel,
        "temporal": TemporalGNNDurationModel,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")
    
    model_class = model_classes[model_type]
    
    if model_type == "improved":
        return model_class(config, stochastic_depth=kwargs.get("stochastic_depth", 0.1))
    elif model_type == "transformer":
        return model_class(config, use_positional_encoding=kwargs.get("use_positional_encoding", True))
    elif model_type == "temporal":
        return model_class(config, use_state_memory=kwargs.get("use_state_memory", True))
    else:
        return model_class(config)


DURATION_MODEL_DESCRIPTIONS = {
    "basic": "Simple message-passing GNN for duration regression",
    "improved": "Attention-based GNN with stochastic depth for duration regression",
    "transformer": "Graph Transformer for duration regression",
    "hetero": "Heterogeneous GNN (QCHGT) for duration regression",
    "temporal": "Temporal GNN with state memory for duration regression",
}

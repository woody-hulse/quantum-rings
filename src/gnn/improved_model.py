"""
Improved GNN models for quantum circuit prediction.

Key improvements over the base model:
1. Ordinal regression for threshold classification (exploits ordered nature)
2. Attention-based message passing for better feature aggregation
3. Multi-scale pooling combining local and global information
4. Stronger regularization with stochastic depth and feature dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from typing import Optional, Tuple, List

from .base import (
    BaseGraphModel,
    BaseGraphDurationModel,
    BaseGraphThresholdClassModel,
    GraphModelConfig,
)
from .graph_builder import NUM_GATE_TYPES, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE

NUM_FAMILIES = 20
NUM_THRESHOLD_CLASSES = 9


class AttentiveGateMessagePassing(MessagePassing):
    """
    Attention-based message passing with per-gate-type transformations.
    
    Uses multi-head attention to weight messages from neighbors,
    allowing the model to focus on the most relevant connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_gate_types: int = NUM_GATE_TYPES,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        num_heads: int = 4,
        dropout: float = 0.1,
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        self.gate_type_embed = nn.Embedding(num_gate_types, out_channels)
        
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels + out_channels + edge_feat_dim, out_channels)
        self.v_proj = nn.Linear(in_channels + out_channels + edge_feat_dim, out_channels)
        
        self.msg_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self._attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
    ) -> torch.Tensor:
        gate_embed = self.gate_type_embed(edge_gate_type)
        
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, gate_embed=gate_embed
        )
        
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        out = self.norm2(out)
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        gate_embed: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(x_i)
        kv_input = torch.cat([x_j, gate_embed, edge_attr], dim=-1)
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        
        B = q.shape[0]
        q = q.view(B, self.num_heads, self.head_dim)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)
        
        attn_scores = (q * k).sum(dim=-1) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=0)
        self._attention_weights = attn_weights.detach()
        
        weighted_v = (attn_weights.unsqueeze(-1) * v).view(B, -1)
        
        return self.msg_mlp(weighted_v)


class StochasticDepth(nn.Module):
    """Stochastic depth (layer dropout) for regularization."""
    
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x + residual
        
        keep_prob = 1 - self.drop_prob
        mask = torch.bernoulli(torch.tensor([keep_prob])).item()
        
        if mask == 0:
            return x
        else:
            return x + residual / keep_prob


class ImprovedQuantumCircuitGNN(BaseGraphDurationModel):
    """
    Improved GNN for duration prediction with attention and regularization.
    
    Uses attention-based message passing and stochastic depth for better
    generalization on quantum circuit graphs.
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 1 + 20,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        stochastic_depth: float = 0.1,
    ):
        config = GraphModelConfig(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.pool_dim = 3 * hidden_dim
        
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.mp_layers = nn.ModuleList()
        self.stoch_depth = nn.ModuleList()
        for i in range(num_layers):
            layer_drop = stochastic_depth * (i + 1) / num_layers
            self.mp_layers.append(
                AttentiveGateMessagePassing(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    edge_feat_dim=edge_feat_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
            self.stoch_depth.append(StochasticDepth(layer_drop))
        
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        combined_dim = self.pool_dim + hidden_dim
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.runtime_head = nn.Linear(hidden_dim, 1)
    
    @property
    def name(self) -> str:
        return "ImprovedQuantumCircuitGNN"
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_embed(x)
        for mp_layer, sd in zip(self.mp_layers, self.stoch_depth):
            h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
            h = sd(h, h_new)
        return h
    
    def pool_graph(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h_mean = global_mean_pool(node_embeddings, batch)
        h_max = global_max_pool(node_embeddings, batch)
        h_sum = global_add_pool(node_embeddings, batch)
        return torch.cat([h_mean, h_max, h_sum], dim=-1)
    
    def predict_runtime(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode_nodes(x, edge_index, edge_attr, edge_gate_type, batch)
        h_graph = self.pool_graph(h, batch)
        g = self.global_proj(global_features)
        combined = torch.cat([h_graph, g], dim=-1)
        combined = self.combined_mlp(combined)
        return self.runtime_head(combined).squeeze(-1)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_runtime(
            x, edge_index, edge_attr, edge_gate_type, batch, global_features
        )
    
    def get_attention_weights(self) -> Optional[List[torch.Tensor]]:
        weights = []
        for layer in self.mp_layers:
            if hasattr(layer, '_attention_weights') and layer._attention_weights is not None:
                weights.append(layer._attention_weights)
        return weights if weights else None


class OrdinalRegressionHead(nn.Module):
    """
    Ordinal regression head for threshold classification.
    
    Instead of predicting K independent classes, predicts K-1 cumulative
    probabilities: P(class >= k) for k = 1, ..., K-1.
    
    This naturally captures the ordered nature of thresholds and tends to
    produce more conservative predictions (favoring higher thresholds).
    """
    
    def __init__(self, input_dim: int, num_classes: int = NUM_THRESHOLD_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes - 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cumulative_logits: P(class >= k) logits for k=1..K-1
            class_probs: Probability for each class
        """
        cumulative_logits = self.linear(x)
        cumulative_probs = torch.sigmoid(cumulative_logits)
        
        class_probs = torch.zeros(x.shape[0], self.num_classes, device=x.device)
        class_probs[:, 0] = 1 - cumulative_probs[:, 0]
        for k in range(1, self.num_classes - 1):
            class_probs[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]
        class_probs[:, -1] = cumulative_probs[:, -1]
        
        class_probs = class_probs.clamp(min=1e-7)
        class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
        
        return cumulative_logits, class_probs
    
    def predict_class(self, x: torch.Tensor, conservative_bias: float = 0.0) -> torch.Tensor:
        """
        Predict class with optional conservative bias toward higher classes.
        
        Args:
            x: Input features
            conservative_bias: Shift threshold for cumulative probabilities (0-0.5)
                              Higher = more conservative (predicts higher thresholds)
        """
        cumulative_logits = self.linear(x)
        cumulative_probs = torch.sigmoid(cumulative_logits)
        
        threshold = 0.5 - conservative_bias
        predicted = (cumulative_probs > threshold).sum(dim=1)
        
        return predicted


class OrdinalLoss(nn.Module):
    """
    Loss function for ordinal regression.
    
    Uses binary cross-entropy on cumulative probabilities, which naturally
    encourages predictions to respect the ordering of classes.
    """
    
    def __init__(self, num_classes: int = NUM_THRESHOLD_CLASSES, conservative_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.conservative_weight = conservative_weight
    
    def forward(
        self,
        cumulative_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cumulative_logits: Shape (batch, num_classes - 1)
            targets: Class indices, shape (batch,)
        """
        batch_size = targets.shape[0]
        device = cumulative_logits.device
        
        cumulative_targets = torch.zeros(batch_size, self.num_classes - 1, device=device)
        for k in range(self.num_classes - 1):
            cumulative_targets[:, k] = (targets > k).float()
        
        base_loss = F.binary_cross_entropy_with_logits(
            cumulative_logits, cumulative_targets, reduction='none'
        )
        
        if self.conservative_weight != 1.0:
            underpred_mask = cumulative_targets == 1
            overpred_mask = cumulative_targets == 0
            weights = torch.ones_like(base_loss)
            weights[underpred_mask] = self.conservative_weight
            base_loss = base_loss * weights
        
        return base_loss.mean()


class ImprovedQuantumCircuitGNNThresholdClass(BaseGraphThresholdClassModel):
    """
    Improved GNN for threshold classification with ordinal regression.
    
    Key improvements:
    1. Ordinal regression head (exploits ordered nature of thresholds)
    2. Attention-based message passing
    3. Conservative bias option for inference
    4. Stochastic depth for regularization
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES,
        num_classes: int = NUM_THRESHOLD_CLASSES,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        stochastic_depth: float = 0.1,
        use_ordinal: bool = True,
    ):
        config = GraphModelConfig(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.pool_dim = 3 * hidden_dim
        self.use_ordinal = use_ordinal
        self.num_classes = num_classes
        
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.mp_layers = nn.ModuleList()
        self.stoch_depth = nn.ModuleList()
        for i in range(num_layers):
            layer_drop = stochastic_depth * (i + 1) / num_layers
            self.mp_layers.append(
                AttentiveGateMessagePassing(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    edge_feat_dim=edge_feat_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
            self.stoch_depth.append(StochasticDepth(layer_drop))
        
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        combined_dim = self.pool_dim + hidden_dim
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        if use_ordinal:
            self.head = OrdinalRegressionHead(hidden_dim, num_classes)
        else:
            self.head = nn.Linear(hidden_dim, num_classes)
    
    @property
    def name(self) -> str:
        return "ImprovedQuantumCircuitGNNThresholdClass"
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_embed(x)
        for mp_layer, sd in zip(self.mp_layers, self.stoch_depth):
            h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
            h = sd(h, h_new)
        return h
    
    def pool_graph(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h_mean = global_mean_pool(node_embeddings, batch)
        h_max = global_max_pool(node_embeddings, batch)
        h_sum = global_add_pool(node_embeddings, batch)
        return torch.cat([h_mean, h_max, h_sum], dim=-1)
    
    def predict_logits(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode_nodes(x, edge_index, edge_attr, edge_gate_type, batch)
        h_graph = self.pool_graph(h, batch)
        g = self.global_proj(global_features)
        combined = torch.cat([h_graph, g], dim=-1)
        features = self.combined_mlp(combined)
        
        if self.use_ordinal:
            cumulative_logits, _ = self.head(features)
            return cumulative_logits
        else:
            return self.head(features)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_logits(
            x, edge_index, edge_attr, edge_gate_type, batch, global_features
        )
    
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """Get class probabilities for decision-theoretic inference."""
        h = self.encode_nodes(x, edge_index, edge_attr, edge_gate_type, batch)
        h_graph = self.pool_graph(h, batch)
        g = self.global_proj(global_features)
        combined = torch.cat([h_graph, g], dim=-1)
        features = self.combined_mlp(combined)
        
        if self.use_ordinal:
            _, class_probs = self.head(features)
            return class_probs
        else:
            return F.softmax(self.head(features), dim=-1)
    
    def get_class_probs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for predict_proba for backward compatibility."""
        return self.predict_proba(
            x, edge_index, edge_attr, edge_gate_type, batch, global_features
        )
    
    def get_attention_weights(self) -> Optional[List[torch.Tensor]]:
        weights = []
        for layer in self.mp_layers:
            if hasattr(layer, '_attention_weights') and layer._attention_weights is not None:
                weights.append(layer._attention_weights)
        return weights if weights else None


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Focuses training on hard examples by down-weighting easy (well-classified) examples.
    """
    
    def __init__(
        self,
        num_classes: int = NUM_THRESHOLD_CLASSES,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        
        pt = (probs * targets_one_hot).sum(dim=-1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ConservativeCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with asymmetric label smoothing that favors higher classes.
    
    Instead of uniform label smoothing, smooths toward higher classes to reduce
    underprediction (which gives 0 score in the challenge).
    """
    
    def __init__(
        self,
        num_classes: int = NUM_THRESHOLD_CLASSES,
        base_smoothing: float = 0.1,
        upward_bias: float = 0.2,
    ):
        """
        Args:
            num_classes: Number of classes
            base_smoothing: Base label smoothing amount
            upward_bias: Additional probability mass shifted to higher classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.base_smoothing = base_smoothing
        self.upward_bias = upward_bias
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = targets.shape[0]
        device = logits.device
        
        smooth_labels = torch.full(
            (batch_size, self.num_classes),
            self.base_smoothing / self.num_classes,
            device=device
        )
        
        for i in range(batch_size):
            t = targets[i].item()
            main_weight = 1.0 - self.base_smoothing - self.upward_bias
            smooth_labels[i, t] = main_weight
            
            if t < self.num_classes - 1:
                remaining = self.upward_bias
                for k in range(t + 1, self.num_classes):
                    weight = remaining * 0.5
                    smooth_labels[i, k] += weight
                    remaining -= weight
        
        smooth_labels = smooth_labels / smooth_labels.sum(dim=1, keepdim=True)
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1)
        
        return loss.mean()


def create_improved_gnn_model(
    model_type: str = "duration",
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: Optional[int] = None,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.2,
    stochastic_depth: float = 0.1,
    use_ordinal: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function to create improved GNN models."""
    
    if model_type == "duration":
        if global_feat_dim is None:
            global_feat_dim = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES
        return ImprovedQuantumCircuitGNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
        )
    elif model_type == "threshold":
        if global_feat_dim is None:
            global_feat_dim = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES
        return ImprovedQuantumCircuitGNNThresholdClass(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            use_ordinal=use_ordinal,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

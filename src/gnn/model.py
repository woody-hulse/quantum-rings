"""
Graph Neural Network for quantum circuit threshold and runtime prediction.

Architecture:
- Per-gate-type learnable embeddings for message passing
- Graph-level readout with global feature conditioning
- Multi-task heads: ordinal regression for thresholds, regression for runtime
- Improved regularization for small datasets
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops

from .graph_builder import NUM_GATE_TYPES, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE


class OrdinalRegressionHead(nn.Module):
    """
    Ordinal regression head for threshold prediction.
    
    Instead of treating thresholds as independent classes, ordinal regression
    respects the natural ordering: 1 < 2 < 4 < 8 < ... < 256.
    
    Predicts K-1 cumulative probabilities P(Y > k) for k in 1..K-1.
    The predicted class is the largest k where P(Y > k) > 0.5.
    """
    
    def __init__(self, in_features: int, num_classes: int = 9):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared feature transformation
        self.shared = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # K-1 binary classifiers for cumulative probabilities
        # Each predicts P(Y > k) for k = 0, 1, ..., K-2
        self.cumulative_logits = nn.Linear(in_features, num_classes - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns logits for ordinal regression.
        Shape: [batch_size, num_classes - 1]
        """
        x = self.shared(x)
        return self.cumulative_logits(x)
    
    def predict_class(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert cumulative logits to class predictions.
        """
        # P(Y > k) for each k
        probs = torch.sigmoid(logits)
        
        # Predicted class is number of thresholds exceeded
        # Count how many P(Y > k) > 0.5
        predictions = (probs > 0.5).sum(dim=1)
        
        return predictions
    
    def to_class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert cumulative logits to class probabilities.
        P(Y = k) = P(Y > k-1) - P(Y > k)
        """
        probs_exceed = torch.sigmoid(logits)
        
        # Add boundaries: P(Y > -1) = 1, P(Y > K-1) = 0
        ones = torch.ones(logits.shape[0], 1, device=logits.device)
        zeros = torch.zeros(logits.shape[0], 1, device=logits.device)
        
        probs_exceed_full = torch.cat([ones, probs_exceed, zeros], dim=1)
        
        # P(Y = k) = P(Y > k-1) - P(Y > k)
        class_probs = probs_exceed_full[:, :-1] - probs_exceed_full[:, 1:]
        
        return class_probs


class OrdinalRegressionLoss(nn.Module):
    """
    Loss function for ordinal regression.
    
    Uses binary cross-entropy on cumulative probabilities.
    """
    
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Cumulative logits [batch_size, num_classes - 1]
            targets: Class labels [batch_size] in range [0, num_classes - 1]
        """
        # Create cumulative targets
        # For class k, we want P(Y > j) = 1 for j < k, P(Y > j) = 0 for j >= k
        batch_size = targets.shape[0]
        device = logits.device
        
        cumulative_targets = torch.zeros_like(logits)
        for i in range(self.num_classes - 1):
            cumulative_targets[:, i] = (targets > i).float()
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, cumulative_targets)
        
        return loss


class GateTypeMessagePassing(MessagePassing):
    """
    Message passing layer with per-gate-type learnable transformations.
    
    Each gate type has its own learned weight matrix for message transformation,
    allowing the model to learn how different gates propagate information differently.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_gate_types: int = NUM_GATE_TYPES,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_gate_types = num_gate_types
        
        # Per-gate-type message transformation weights
        self.gate_type_embed = nn.Embedding(num_gate_types, out_channels)
        
        # Shared message MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels + edge_feat_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        
        # Node update
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
    ) -> torch.Tensor:
        # Get gate type embeddings for each edge
        gate_embed = self.gate_type_embed(edge_gate_type)  # [n_edges, out_channels]
        
        # Propagate messages
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, gate_embed=gate_embed
        )
        
        # Update with residual if dimensions match
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        out = self.norm(out)
        
        return out
    
    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        gate_embed: torch.Tensor,
    ) -> torch.Tensor:
        # x_j: source node features [n_edges, in_channels]
        # gate_embed: per-edge gate type embedding [n_edges, out_channels]
        # edge_attr: temporal position and parameters [n_edges, edge_feat_dim]
        
        msg_input = torch.cat([x_j, gate_embed, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)


class QuantumCircuitGNN(nn.Module):
    """
    Full GNN model for quantum circuit prediction.
    
    Architecture:
    1. Node embedding projection
    2. Multiple GateTypeMessagePassing layers with dropout
    3. Graph-level pooling (mean + max + sum)
    4. Concatenation with global features
    5. Task-specific prediction heads (ordinal regression for thresholds)
    """
    
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 20,  # +20 for family one-hot
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_threshold_classes: int = 9,
        dropout: float = 0.1,
        use_ordinal: bool = True,
    ):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_ordinal = use_ordinal
        self.num_threshold_classes = num_threshold_classes
        
        # Initial node projection with dropout
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            GateTypeMessagePassing(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_feat_dim=edge_feat_dim,
            )
            for _ in range(num_layers)
        ])
        
        # Dropout after each message passing layer
        self.mp_dropout = nn.Dropout(dropout)
        
        # Graph pooling produces 3 * hidden_dim features (mean, max, sum)
        pool_dim = 3 * hidden_dim
        
        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Combined representation processing
        combined_dim = pool_dim + hidden_dim
        
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        if use_ordinal:
            self.threshold_head = OrdinalRegressionHead(hidden_dim, num_threshold_classes)
        else:
            self.threshold_head = nn.Linear(hidden_dim, num_threshold_classes)
        
        self.runtime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features [total_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, total_edges]
            edge_attr: Edge features [total_edges, edge_feat_dim]
            edge_gate_type: Gate type indices [total_edges]
            batch: Batch assignment for each node [total_nodes]
            global_features: Global/circuit-level features [batch_size, global_feat_dim]
        
        Returns:
            threshold_logits: [batch_size, num_threshold_classes] or [batch_size, num_classes-1] for ordinal
            runtime_pred: [batch_size]
        """
        # Initial node embedding
        h = self.node_embed(x)
        
        # Message passing with dropout
        for mp_layer in self.mp_layers:
            h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
            h = h + self.mp_dropout(h_new)  # residual connection with dropout
        
        # Graph-level pooling (multiple aggregations for richer representation)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max, h_sum], dim=-1)
        
        # Process global features
        g = self.global_proj(global_features)
        
        # Combine graph and global representations
        combined = torch.cat([h_graph, g], dim=-1)
        combined = self.combined_mlp(combined)
        
        # Task predictions
        threshold_logits = self.threshold_head(combined)
        runtime_pred = self.runtime_head(combined).squeeze(-1)
        
        return threshold_logits, runtime_pred
    
    def predict_threshold_class(self, threshold_logits: torch.Tensor) -> torch.Tensor:
        """Convert threshold logits to class predictions."""
        if self.use_ordinal:
            return self.threshold_head.predict_class(threshold_logits)
        else:
            return threshold_logits.argmax(dim=1)


class QuantumCircuitGNNWithAttention(nn.Module):
    """
    Enhanced GNN with attention-based pooling for better graph-level representations.
    """
    
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 20,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_threshold_classes: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_ordinal: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_ordinal = use_ordinal
        self.num_threshold_classes = num_threshold_classes
        
        # Initial node projection
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            GateTypeMessagePassing(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_feat_dim=edge_feat_dim,
            )
            for _ in range(num_layers)
        ])
        
        self.mp_dropout = nn.Dropout(dropout)
        
        # Attention pooling
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )
        
        pool_dim = num_heads * hidden_dim
        
        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        combined_dim = pool_dim + hidden_dim
        
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Task heads
        if use_ordinal:
            self.threshold_head = OrdinalRegressionHead(hidden_dim, num_threshold_classes)
        else:
            self.threshold_head = nn.Linear(hidden_dim, num_threshold_classes)
        
        self.runtime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.node_embed(x)
        
        for mp_layer in self.mp_layers:
            h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
            h = h + self.mp_dropout(h_new)
        
        attn_scores = self.pool_attention(h)
        
        batch_size = global_features.shape[0]
        num_heads = attn_scores.shape[1]
        
        attn_weights = torch.zeros_like(attn_scores)
        for i in range(batch_size):
            mask = batch == i
            attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)
        
        h_pooled_list = []
        for head in range(num_heads):
            weights = attn_weights[:, head:head+1]
            h_weighted = h * weights
            h_pooled = global_add_pool(h_weighted, batch)
            h_pooled_list.append(h_pooled)
        
        h_graph = torch.cat(h_pooled_list, dim=-1)
        
        g = self.global_proj(global_features)
        
        combined = torch.cat([h_graph, g], dim=-1)
        combined = self.combined_mlp(combined)
        
        threshold_logits = self.threshold_head(combined)
        runtime_pred = self.runtime_head(combined).squeeze(-1)
        
        return threshold_logits, runtime_pred
    
    def predict_threshold_class(self, threshold_logits: torch.Tensor) -> torch.Tensor:
        """Convert threshold logits to class predictions."""
        if self.use_ordinal:
            return self.threshold_head.predict_class(threshold_logits)
        else:
            return threshold_logits.argmax(dim=1)


def create_gnn_model(
    model_type: str = "basic",
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 20,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_threshold_classes: int = 9,
    dropout: float = 0.1,
    use_ordinal: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create GNN models.
    
    Args:
        model_type: "basic" or "attention"
        node_feat_dim: Node feature dimension
        edge_feat_dim: Edge feature dimension
        global_feat_dim: Global feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of message passing layers
        num_threshold_classes: Number of threshold classes (9)
        dropout: Dropout rate
        use_ordinal: Use ordinal regression for threshold prediction
    """
    if model_type == "basic":
        return QuantumCircuitGNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_threshold_classes=num_threshold_classes,
            dropout=dropout,
            use_ordinal=use_ordinal,
        )
    elif model_type == "attention":
        return QuantumCircuitGNNWithAttention(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_threshold_classes=num_threshold_classes,
            dropout=dropout,
            use_ordinal=use_ordinal,
            num_heads=kwargs.get("num_heads", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

"""
Graph Neural Network for quantum circuit threshold and runtime prediction.

Architecture:
- Per-gate-type learnable embeddings for message passing
- Graph-level readout with global feature conditioning
- Multi-task heads for threshold classification and runtime regression
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops

from .graph_builder import NUM_GATE_TYPES, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE


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
    2. Multiple GateTypeMessagePassing layers
    3. Graph-level pooling (mean + max + sum)
    4. Concatenation with global features
    5. Task-specific prediction heads
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
    ):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial node projection
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
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
        
        # Graph pooling produces 3 * hidden_dim features (mean, max, sum)
        pool_dim = 3 * hidden_dim
        
        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Combined representation processing
        combined_dim = pool_dim + hidden_dim
        
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Task-specific heads
        self.threshold_head = nn.Linear(hidden_dim, num_threshold_classes)
        self.runtime_head = nn.Linear(hidden_dim, 1)
    
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
            threshold_logits: [batch_size, num_threshold_classes]
            runtime_pred: [batch_size]
        """
        # Initial node embedding
        h = self.node_embed(x)
        
        # Message passing
        for mp_layer in self.mp_layers:
            h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
            h = h + h_new  # residual connection
        
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
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial node projection
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
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
        
        # Attention pooling
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads, bias=False),
        )
        
        # After attention: num_heads * hidden_dim
        pool_dim = num_heads * hidden_dim
        
        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Combined representation
        combined_dim = pool_dim + hidden_dim
        
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Task heads
        self.threshold_head = nn.Linear(hidden_dim, num_threshold_classes)
        self.runtime_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial embedding
        h = self.node_embed(x)
        
        # Message passing with residual
        for mp_layer in self.mp_layers:
            h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
            h = h + h_new
        
        # Attention-based pooling
        attn_scores = self.pool_attention(h)  # [total_nodes, num_heads]
        
        # Compute attention weights per graph
        batch_size = global_features.shape[0]
        num_heads = attn_scores.shape[1]
        
        # Softmax within each graph
        attn_weights = torch.zeros_like(attn_scores)
        for i in range(batch_size):
            mask = batch == i
            attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)
        
        # Weighted sum pooling for each head
        h_pooled_list = []
        for head in range(num_heads):
            weights = attn_weights[:, head:head+1]  # [total_nodes, 1]
            h_weighted = h * weights
            h_pooled = global_add_pool(h_weighted, batch)  # [batch_size, hidden_dim]
            h_pooled_list.append(h_pooled)
        
        h_graph = torch.cat(h_pooled_list, dim=-1)  # [batch_size, num_heads * hidden_dim]
        
        # Global features
        g = self.global_proj(global_features)
        
        # Combine
        combined = torch.cat([h_graph, g], dim=-1)
        combined = self.combined_mlp(combined)
        
        # Predictions
        threshold_logits = self.threshold_head(combined)
        runtime_pred = self.runtime_head(combined).squeeze(-1)
        
        return threshold_logits, runtime_pred


def create_gnn_model(
    model_type: str = "basic",
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 20,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_threshold_classes: int = 9,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """Factory function to create GNN models."""
    if model_type == "basic":
        return QuantumCircuitGNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_threshold_classes=num_threshold_classes,
            dropout=dropout,
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
            num_heads=kwargs.get("num_heads", 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

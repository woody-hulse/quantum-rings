"""
Graph Neural Network for quantum circuit duration prediction.

Threshold as input (in global features), output log2(duration).
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool

from .graph_builder import NUM_GATE_TYPES, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE

NUM_FAMILIES = 20


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
    GNN for duration prediction: threshold as input (in global features), output log2(duration).
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 1 + 20,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.mp_layers = nn.ModuleList([
            GateTypeMessagePassing(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_feat_dim=edge_feat_dim,
            )
            for _ in range(num_layers)
        ])
        self.mp_dropout = nn.Dropout(dropout)
        pool_dim = 3 * hidden_dim
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
        self.runtime_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_embed(x)
        for mp_layer in self.mp_layers:
            h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
            h = h + self.mp_dropout(h_new)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max, h_sum], dim=-1)
        g = self.global_proj(global_features)
        combined = torch.cat([h_graph, g], dim=-1)
        combined = self.combined_mlp(combined)
        return self.runtime_head(combined).squeeze(-1)


def create_gnn_model(
    model_type: str = "basic",
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 1 + 20,
    hidden_dim: int = 64,
    num_layers: int = 4,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """Create GNN model for duration prediction (threshold as input, log2(duration) output)."""
    return QuantumCircuitGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        global_feat_dim=global_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


GLOBAL_FEAT_DIM_THRESHOLD_CLASS = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES


class QuantumCircuitGNNThresholdClass(nn.Module):
    """
    GNN for threshold-class prediction: global features without log2(threshold) and duration, output class logits.
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_THRESHOLD_CLASS,
        num_classes: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.mp_layers = nn.ModuleList([
            GateTypeMessagePassing(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_feat_dim=edge_feat_dim,
            )
            for _ in range(num_layers)
        ])
        self.mp_dropout = nn.Dropout(dropout)
        pool_dim = 3 * hidden_dim
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
        self.class_head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_embed(x)
        for mp_layer in self.mp_layers:
            h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
            h = h + self.mp_dropout(h_new)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max, h_sum], dim=-1)
        g = self.global_proj(global_features)
        combined = torch.cat([h_graph, g], dim=-1)
        combined = self.combined_mlp(combined)
        return self.class_head(combined)


def create_gnn_threshold_class_model(
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: int = GLOBAL_FEAT_DIM_THRESHOLD_CLASS,
    num_classes: int = 9,
    hidden_dim: int = 64,
    num_layers: int = 4,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    """Create GNN model for threshold-class prediction (no duration, no threshold in features)."""
    return QuantumCircuitGNNThresholdClass(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        global_feat_dim=global_feat_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

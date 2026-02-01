"""
Temporal Graph Neural Network for Quantum Circuit Simulation Prediction.

This module implements cutting-edge temporal GNN architectures that model quantum
circuits as temporal sequences of operations, capturing:

1. **State Evolution**: How quantum state complexity grows through gate application
2. **Causal Structure**: Later operations depend on earlier state transformations
3. **Entanglement Dynamics**: Track how entanglement spreads across qubits over time
4. **Multi-Scale Temporal Patterns**: Capture both local gate sequences and global circuit structure

Key architectural innovations:
- Temporal Positional Encoding with learnable time embeddings
- Causal Message Passing that respects gate ordering
- State Memory GRU that tracks qubit state evolution
- Multi-Resolution Temporal Aggregation
- Entanglement-Aware Graph Attention
"""

from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from torch_geometric.utils import softmax, degree

from .graph_builder import NUM_GATE_TYPES, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE
from .base import BaseGraphModel, BaseGraphDurationModel, BaseGraphThresholdClassModel, GraphModelConfig

NUM_FAMILIES = 20
NUM_THRESHOLD_CLASSES = 9


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal positions."""
    
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:dim // 2])
        self.register_buffer("pe", pe)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: [n_edges] normalized positions in [0, 1]"""
        indices = (positions * 999).long().clamp(0, 999)
        return self.pe[indices]


class LearnableTemporalEncoding(nn.Module):
    """Learnable temporal encoding that combines multiple time scales."""
    
    def __init__(self, dim: int, num_buckets: int = 32):
        super().__init__()
        self.num_buckets = num_buckets
        self.bucket_embed = nn.Embedding(num_buckets, dim)
        self.continuous_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.combine = nn.Linear(dim * 2, dim)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: [n_edges] normalized in [0, 1]"""
        bucket_idx = (positions * (self.num_buckets - 1)).long().clamp(0, self.num_buckets - 1)
        discrete_embed = self.bucket_embed(bucket_idx)
        continuous_embed = self.continuous_proj(positions.unsqueeze(-1))
        combined = torch.cat([discrete_embed, continuous_embed], dim=-1)
        return self.combine(combined)


class TemporalGateEmbedding(nn.Module):
    """
    Rich gate embedding that combines gate type, parameters, and temporal context.
    
    For MPS simulation, different gates have different entanglement effects:
    - CX/CZ: Can increase bond dimension
    - SWAP: Reorders qubits but preserves entanglement
    - Single-qubit: Doesn't change entanglement structure
    - Controlled rotations: Conditional entanglement based on parameter
    """
    
    def __init__(
        self,
        num_gate_types: int,
        hidden_dim: int,
        edge_feat_dim: int,
    ):
        super().__init__()
        self.gate_embed = nn.Embedding(num_gate_types, hidden_dim)
        
        # Gate property embeddings (learned properties)
        self.is_entangling = nn.Embedding(num_gate_types, hidden_dim // 4)
        self.gate_complexity = nn.Embedding(num_gate_types, hidden_dim // 4)
        
        # Temporal encoding
        self.temporal_enc = LearnableTemporalEncoding(hidden_dim // 2)
        
        # Parameter encoding (rotation angles matter for simulation fidelity)
        self.param_enc = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
        )
        
        # Edge feature projection
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim // 4)
        
        # Combine all
        total_dim = hidden_dim + (hidden_dim // 4) * 2 + (hidden_dim // 2) + (hidden_dim // 4) * 2
        self.combine = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        gate_types: torch.Tensor,
        edge_attr: torch.Tensor,
        temporal_pos: torch.Tensor,
    ) -> torch.Tensor:
        gate_emb = self.gate_embed(gate_types)
        entangle_emb = self.is_entangling(gate_types)
        complex_emb = self.gate_complexity(gate_types)
        temporal_emb = self.temporal_enc(temporal_pos)
        param_emb = self.param_enc(edge_attr[:, 1:2] if edge_attr.size(1) > 1 else torch.zeros_like(edge_attr[:, :1]))
        edge_emb = self.edge_proj(edge_attr)
        
        combined = torch.cat([
            gate_emb, entangle_emb, complex_emb,
            temporal_emb, param_emb, edge_emb
        ], dim=-1)
        
        return self.combine(combined)


class CausalTemporalAttention(MessagePassing):
    """
    Temporal attention mechanism that respects causal ordering of gates.
    
    Unlike standard graph attention, this layer considers:
    1. Temporal distance between gates (closer gates have stronger influence)
    2. Causal relationships (earlier gates can influence later ones)
    3. Entanglement connectivity (qubits that have interacted share information)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        causal_decay: float = 0.1,
    ):
        super().__init__(aggr="add", flow="source_to_target")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.causal_decay = causal_decay
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Temporal bias for attention
        self.temporal_bias = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.Tanh(),
        )
        
        # Relative position encoding
        self.rel_pos_enc = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.Tanh(),
        )
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self._attn_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_pos: torch.Tensor,
        gate_embed: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        out = self.propagate(
            edge_index, x=x, temporal_pos=temporal_pos, gate_embed=gate_embed
        )
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(h + out)
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        temporal_pos: torch.Tensor,
        gate_embed: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        B = x_i.shape[0]
        
        q = self.q_proj(x_i).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(x_j + gate_embed).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(x_j + gate_embed).view(B, self.num_heads, self.head_dim)
        
        attn_scores = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)
        
        # Add temporal bias
        temporal_bias = self.temporal_bias(temporal_pos.unsqueeze(-1))
        attn_scores = attn_scores + temporal_bias
        
        attn_weights = softmax(attn_scores, index, dim=0)
        self._attn_weights = attn_weights.detach()
        
        attn_weights = self.dropout(attn_weights)
        
        out = (attn_weights.unsqueeze(-1) * v).view(B, -1)
        return out


class StateMemoryGRU(nn.Module):
    """
    GRU-based state memory that tracks qubit state evolution through the circuit.
    
    This models how each qubit's state complexity evolves as gates are applied,
    capturing the cumulative effect of entanglement on simulation difficulty.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.gate_transform = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        gate_embed: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """
        Update node states based on incoming gates.
        
        Args:
            node_states: Current node representations [n_nodes, hidden_dim]
            edge_index: [2, n_edges] connectivity
            gate_embed: Gate embeddings [n_edges, hidden_dim]
            n_nodes: Number of nodes
        """
        src, dst = edge_index
        
        # Aggregate gate information to destination nodes
        gate_msg = self.gate_transform(gate_embed)
        gate_agg = torch.zeros(n_nodes, gate_msg.size(-1), device=node_states.device)
        gate_agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(gate_msg), gate_msg)
        
        # Count incoming edges for normalization
        edge_count = torch.zeros(n_nodes, device=node_states.device)
        edge_count.scatter_add_(0, dst, torch.ones(dst.size(0), device=node_states.device))
        edge_count = edge_count.clamp(min=1)
        gate_agg = gate_agg / edge_count.unsqueeze(-1)
        
        # GRU update
        gru_input = torch.cat([gate_agg, node_states], dim=-1)
        new_states = self.gru(gru_input, node_states)
        new_states = self.dropout(new_states)
        new_states = self.norm(new_states)
        
        return new_states


class EntanglementAwareConv(MessagePassing):
    """
    Message passing layer that explicitly models entanglement spread.
    
    For MPS simulation, the key insight is:
    - 2-qubit gates between distant qubits create long-range entanglement
    - This entanglement must be tracked through bond dimensions
    - Consecutive gates on overlapping qubits compound entanglement effects
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(aggr="add")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Entanglement-aware message
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Distance-based attention (long-range gates are harder)
        self.distance_attention = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.Sigmoid(),
        )
        
        # Node update with gating
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.update_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        gate_embed: torch.Tensor,
        qubit_distance: torch.Tensor,
    ) -> torch.Tensor:
        out = self.propagate(
            edge_index, x=x, gate_embed=gate_embed, qubit_distance=qubit_distance
        )
        
        gate = self.update_gate(torch.cat([x, out], dim=-1))
        update = self.update_transform(torch.cat([x, out], dim=-1))
        out = x * (1 - gate) + update * gate
        out = self.norm(out)
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        gate_embed: torch.Tensor,
        qubit_distance: torch.Tensor,
    ) -> torch.Tensor:
        msg_input = torch.cat([x_i, x_j, gate_embed], dim=-1)
        msg = self.msg_mlp(msg_input)
        
        # Weight by qubit distance (long-range interactions are more impactful)
        dist_weight = self.distance_attention(qubit_distance.unsqueeze(-1))
        dist_weight = dist_weight.mean(dim=-1, keepdim=True)
        
        # Emphasize long-range gates (they require higher bond dimension)
        msg = msg * (1 + dist_weight)
        
        return msg


class MultiScaleTemporalPooling(nn.Module):
    """
    Multi-scale temporal pooling that captures patterns at different time scales.
    
    Aggregates node features using:
    1. Global pooling (overall circuit complexity)
    2. Early/Late split (temporal asymmetry)
    3. Temporal attention pooling (learned importance weighting)
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Learn to weight different temporal regions
        self.early_weight = nn.Parameter(torch.ones(1))
        self.late_weight = nn.Parameter(torch.ones(1))
        
        # Combine multiple pooling strategies
        # mean + max + sum + temporal_attn + early + late = 6 * hidden_dim
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        temporal_pos: torch.Tensor,
    ) -> torch.Tensor:
        # Standard pooling
        h_mean = global_mean_pool(x, batch)
        h_max = global_max_pool(x, batch)
        h_sum = global_add_pool(x, batch)
        
        # Temporal attention pooling
        attn_input = torch.cat([x, temporal_pos.unsqueeze(-1)], dim=-1)
        attn_scores = self.temporal_attn(attn_input)
        attn_weights = softmax(attn_scores.squeeze(-1), batch)
        h_attn = global_add_pool(x * attn_weights.unsqueeze(-1), batch)
        
        # Early/Late temporal split
        early_mask = (temporal_pos < 0.33).float().unsqueeze(-1)
        late_mask = (temporal_pos > 0.66).float().unsqueeze(-1)
        
        h_early = global_mean_pool(x * early_mask, batch) * self.early_weight
        h_late = global_mean_pool(x * late_mask, batch) * self.late_weight
        
        combined = torch.cat([h_mean, h_max, h_sum, h_attn, h_early, h_late], dim=-1)
        return self.combine(combined)


class TemporalGraphTransformerLayer(nn.Module):
    """
    Transformer layer adapted for temporal graph processing.
    
    Combines:
    - Multi-head self-attention over graph structure
    - Temporal position encoding
    - Gate-aware message passing
    - Feed-forward network with gating
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        ff_dim = ff_dim or hidden_dim * 4
        
        self.temporal_attn = CausalTemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.entangle_conv = EntanglementAwareConv(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Gating mechanism for residual connections
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        gate_embed: torch.Tensor,
        temporal_pos: torch.Tensor,
        qubit_distance: torch.Tensor,
    ) -> torch.Tensor:
        # Temporal attention
        h = self.temporal_attn(x, edge_index, temporal_pos, gate_embed)
        
        # Entanglement-aware convolution
        h = self.entangle_conv(h, edge_index, gate_embed, qubit_distance)
        
        # Feed-forward with gating
        h_ff = self.ff(self.norm1(h))
        gate = self.gate(torch.cat([h, h_ff], dim=-1))
        h = h + gate * h_ff
        h = self.norm2(h)
        
        return h


class TemporalQuantumCircuitGNN(BaseGraphDurationModel):
    """
    Cutting-edge Temporal GNN for quantum circuit runtime prediction.
    
    Architecture overview:
    1. Rich node/edge embedding with temporal encoding
    2. Stack of Temporal Graph Transformer layers
    3. State Memory GRU for tracking qubit evolution
    4. Multi-scale temporal pooling
    5. Output head with global context integration
    
    Key features for quantum circuit simulation:
    - Models sequential gate execution
    - Tracks entanglement dynamics across time
    - Captures both local gate effects and global circuit patterns
    - Uses physics-inspired attention (distance-weighted for MPS simulation)
    """
    
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.15,
        use_state_memory: bool = True,
        config: Optional[GraphModelConfig] = None,
    ):
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_state_memory = use_state_memory
        
        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Gate embedding with temporal encoding
        self.gate_embed = TemporalGateEmbedding(
            num_gate_types=NUM_GATE_TYPES,
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
        )
        
        # Temporal transformer layers
        self.layers = nn.ModuleList([
            TemporalGraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # State memory (optional)
        if use_state_memory:
            self.state_memory = StateMemoryGRU(hidden_dim, dropout)
        
        # Multi-scale pooling
        self.pool = MultiScaleTemporalPooling(hidden_dim, dropout)
        
        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    @property
    def name(self) -> str:
        return "TemporalQuantumCircuitGNN"
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # Extract temporal position from edge attributes (first column)
        temporal_pos = edge_attr[:, 0] if edge_attr.size(1) > 0 else torch.zeros(edge_attr.size(0), device=x.device)
        qubit_distance = edge_attr[:, 2] if edge_attr.size(1) > 2 else torch.zeros(edge_attr.size(0), device=x.device)
        
        # Embed nodes and gates
        h = self.node_embed(x)
        gate_emb = self.gate_embed(edge_gate_type, edge_attr, temporal_pos)
        
        # Apply temporal transformer layers
        for layer in self.layers:
            h = layer(h, edge_index, gate_emb, temporal_pos, qubit_distance)
        
        # State memory update
        if self.use_state_memory:
            h = self.state_memory(h, edge_index, gate_emb, h.size(0))
        
        return h
    
    def pool_graph(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # Use simple pooling for the abstract interface
        # Full temporal pooling needs edge temporal positions
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
        return self.forward(x, edge_index, edge_attr, edge_gate_type, batch, global_features)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        # Extract temporal features from edge attributes
        temporal_pos = edge_attr[:, 0] if edge_attr.size(1) > 0 else torch.zeros(edge_attr.size(0), device=x.device)
        qubit_distance = edge_attr[:, 2] if edge_attr.size(1) > 2 else torch.zeros(edge_attr.size(0), device=x.device)
        
        # Embed nodes and gates
        h = self.node_embed(x)
        gate_emb = self.gate_embed(edge_gate_type, edge_attr, temporal_pos)
        
        # Apply temporal transformer layers
        for layer in self.layers:
            h = layer(h, edge_index, gate_emb, temporal_pos, qubit_distance)
        
        # State memory update
        if self.use_state_memory:
            h = self.state_memory(h, edge_index, gate_emb, h.size(0))
        
        # Compute per-node temporal positions for pooling
        node_temporal_pos = torch.zeros(h.size(0), device=h.device)
        if edge_index.size(1) > 0:
            src, dst = edge_index
            node_temporal_pos.scatter_reduce_(
                0, dst, temporal_pos, reduce="mean", include_self=False
            )
        
        # Multi-scale temporal pooling
        h_graph = self.pool(h, batch, node_temporal_pos)
        
        # Global context
        g = self.global_proj(global_features)
        
        # Combine and predict
        combined = torch.cat([h_graph, g], dim=-1)
        return self.output_head(combined).squeeze(-1)


class TemporalQuantumCircuitGNNThresholdClass(BaseGraphThresholdClassModel):
    """
    Temporal GNN variant for threshold class prediction.
    
    Uses the same temporal architecture but with ordinal regression head
    for predicting the discrete threshold classes.
    """
    
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES,
        num_classes: int = NUM_THRESHOLD_CLASSES,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.15,
        use_ordinal: bool = True,
        use_state_memory: bool = True,
        config: Optional[GraphModelConfig] = None,
    ):
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_ordinal = use_ordinal
        self.use_state_memory = use_state_memory
        
        # Node embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Gate embedding
        self.gate_embed = TemporalGateEmbedding(
            num_gate_types=NUM_GATE_TYPES,
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
        )
        
        # Temporal transformer layers
        self.layers = nn.ModuleList([
            TemporalGraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # State memory
        if use_state_memory:
            self.state_memory = StateMemoryGRU(hidden_dim, dropout)
        
        # Multi-scale pooling
        self.pool = MultiScaleTemporalPooling(hidden_dim, dropout)
        
        # Global projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Classification head
        head_input_dim = hidden_dim * 2
        if use_ordinal:
            self.class_head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes - 1),
            )
        else:
            self.class_head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
    
    @property
    def name(self) -> str:
        return "TemporalQuantumCircuitGNNThresholdClass"
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        temporal_pos = edge_attr[:, 0] if edge_attr.size(1) > 0 else torch.zeros(edge_attr.size(0), device=x.device)
        qubit_distance = edge_attr[:, 2] if edge_attr.size(1) > 2 else torch.zeros(edge_attr.size(0), device=x.device)
        
        h = self.node_embed(x)
        gate_emb = self.gate_embed(edge_gate_type, edge_attr, temporal_pos)
        
        for layer in self.layers:
            h = layer(h, edge_index, gate_emb, temporal_pos, qubit_distance)
        
        if self.use_state_memory:
            h = self.state_memory(h, edge_index, gate_emb, h.size(0))
        
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
        return self.forward(x, edge_index, edge_attr, edge_gate_type, batch, global_features)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        temporal_pos = edge_attr[:, 0] if edge_attr.size(1) > 0 else torch.zeros(edge_attr.size(0), device=x.device)
        qubit_distance = edge_attr[:, 2] if edge_attr.size(1) > 2 else torch.zeros(edge_attr.size(0), device=x.device)
        
        h = self.node_embed(x)
        gate_emb = self.gate_embed(edge_gate_type, edge_attr, temporal_pos)
        
        for layer in self.layers:
            h = layer(h, edge_index, gate_emb, temporal_pos, qubit_distance)
        
        if self.use_state_memory:
            h = self.state_memory(h, edge_index, gate_emb, h.size(0))
        
        node_temporal_pos = torch.zeros(h.size(0), device=h.device)
        if edge_index.size(1) > 0:
            src, dst = edge_index
            node_temporal_pos.scatter_reduce_(
                0, dst, temporal_pos, reduce="mean", include_self=False
            )
        
        h_graph = self.pool(h, batch, node_temporal_pos)
        g = self.global_proj(global_features)
        
        combined = torch.cat([h_graph, g], dim=-1)
        return self.class_head(combined)
    
    def get_class_probs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(x, edge_index, edge_attr, edge_gate_type, batch, global_features)
        
        if self.use_ordinal:
            cumulative_probs = torch.sigmoid(logits)
            class_probs = torch.zeros(logits.size(0), self.num_classes, device=logits.device)
            class_probs[:, 0] = 1 - cumulative_probs[:, 0]
            for k in range(1, self.num_classes - 1):
                class_probs[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]
            class_probs[:, -1] = cumulative_probs[:, -1]
            class_probs = class_probs.clamp(min=1e-7)
            class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
            return class_probs
        else:
            return F.softmax(logits, dim=-1)


def create_temporal_gnn_model(
    model_type: str = "duration",
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: Optional[int] = None,
    hidden_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.15,
    use_ordinal: bool = True,
    use_state_memory: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function to create Temporal GNN models."""
    
    if model_type == "duration":
        if global_feat_dim is None:
            global_feat_dim = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES
        return TemporalQuantumCircuitGNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_state_memory=use_state_memory,
        )
    elif model_type == "threshold":
        if global_feat_dim is None:
            global_feat_dim = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES
        return TemporalQuantumCircuitGNNThresholdClass(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_ordinal=use_ordinal,
            use_state_memory=use_state_memory,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Compact configuration presets
TEMPORAL_GNN_CONFIGS = {
    "small": {
        "hidden_dim": 64,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
    },
    "medium": {
        "hidden_dim": 128,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.15,
    },
    "large": {
        "hidden_dim": 256,
        "num_layers": 8,
        "num_heads": 8,
        "dropout": 0.2,
    },
}

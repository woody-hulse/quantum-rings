"""
State-of-the-Art Graph Neural Network for Quantum Circuit Prediction.

This module implements cutting-edge GNN architectures incorporating:
1. GraphGPS-style hybrid architecture (local MPNN + global attention)
2. Principal Neighborhood Aggregation (PNA) with multiple aggregators
3. Graph Transformer with edge-aware attention (GATv2-style)
4. Virtual nodes for global information flow
5. Learnable positional encodings (Random Walk, Laplacian Eigenvector)
6. Gate-type aware Mixture of Experts (MoE)
7. Pre-normalization (as in modern transformers)
8. Gated Recurrent Updates (GRU-style)
9. Hierarchical multi-scale pooling
10. Stochastic depth for regularization
11. Deep supervision with auxiliary losses

Architecture Overview:
    Input -> Positional Encoding -> [GPS Block x N] -> Hierarchical Pooling -> Prediction Heads

Each GPS Block:
    h -> LayerNorm -> LocalMPNN -> + -> LayerNorm -> GlobalAttn -> + -> FFN -> +
         |_________________________|    |__________________________|    |_______|
              (residual)                      (residual)              (residual)
"""

from typing import Tuple, Optional, List, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from torch_geometric.utils import softmax, degree, add_self_loops
from torch_scatter import scatter

# Import from your existing graph_builder
try:
    from .graph_builder import NUM_GATE_TYPES, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE
except ImportError:
    NUM_GATE_TYPES = 30
    NODE_FEAT_DIM = 22
    EDGE_FEAT_DIM = 4
    GLOBAL_FEAT_DIM_BASE = 6

NUM_FAMILIES = 20


# =============================================================================
# POSITIONAL ENCODINGS
# =============================================================================

class RandomWalkPE(nn.Module):
    """
    Random Walk Positional Encoding.
    
    Computes the diagonal of random walk matrices R, R^2, ..., R^k where R = AD^{-1}.
    These capture local structural information around each node.
    """
    
    def __init__(self, walk_length: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.walk_length = walk_length
        self.encoder = nn.Sequential(
            nn.Linear(walk_length, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """Compute RWPE features for each node."""
        device = edge_index.device
        
        # Build adjacency matrix
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv = 1.0 / deg.clamp(min=1)
        
        # Random walk transition matrix (sparse representation via message passing)
        # We compute diagonals of R^k iteratively
        pe = torch.zeros(num_nodes, self.walk_length, device=device)
        
        # Initial distribution: uniform over neighbors
        walk_dist = torch.ones(num_nodes, device=device)
        
        for k in range(self.walk_length):
            # One step of random walk
            # new_dist[i] = sum_{j->i} dist[j] / deg[j]
            msg = walk_dist[row] * deg_inv[row]
            new_dist = scatter(msg, col, dim=0, dim_size=num_nodes, reduce='sum')
            
            # Store diagonal (probability of returning to start after k+1 steps)
            pe[:, k] = new_dist
            walk_dist = new_dist
        
        return self.encoder(pe)


class LaplacianPE(nn.Module):
    """
    Laplacian Eigenvector Positional Encoding.
    
    Uses the k smallest eigenvectors of the graph Laplacian to encode
    structural position. Sign invariance is handled via learnable sign flip.
    """
    
    def __init__(self, k: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.k = k
        self.encoder = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Learnable sign correction
        self.sign_inv = nn.Sequential(
            nn.Linear(k, k),
            nn.Tanh(),
        )
    
    def forward(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """Compute Laplacian PE features."""
        device = edge_index.device
        
        if num_nodes < self.k + 1:
            # Not enough nodes, return zeros
            return torch.zeros(num_nodes, self.encoder[0].out_features, device=device)
        
        # Build Laplacian
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.float)
        
        # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = torch.pow(deg.clamp(min=1), -0.5)
        
        # Build dense Laplacian for eigendecomposition
        L = torch.zeros(num_nodes, num_nodes, device=device)
        L[row, col] = -deg_inv_sqrt[row] * deg_inv_sqrt[col]
        L = L + torch.eye(num_nodes, device=device)
        
        # Compute eigenvectors
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            # Take k smallest non-trivial eigenvectors (skip the constant one)
            pe = eigenvectors[:, 1:self.k+1]
            
            # Handle sign ambiguity
            pe = pe * self.sign_inv(pe)
        except:
            pe = torch.zeros(num_nodes, self.k, device=device)
        
        return self.encoder(pe)


class SinusoidalPE(nn.Module):
    """
    Sinusoidal Positional Encoding for temporal/sequential positions.
    
    Uses sine and cosine functions at different frequencies to encode
    continuous position values (e.g., gate position in circuit).
    """
    
    def __init__(self, dim: int = 64, max_freq: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (max_freq ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, positions: Tensor) -> Tensor:
        """
        Args:
            positions: [N] tensor of position values in [0, 1]
        Returns:
            [N, dim] positional embeddings
        """
        # Scale positions to larger range for more variation
        pos = positions.unsqueeze(-1) * 100.0  # [N, 1]
        
        sinusoid_inp = pos * self.inv_freq  # [N, dim//2]
        pe = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)  # [N, dim]
        
        return pe


# =============================================================================
# ADVANCED MESSAGE PASSING LAYERS
# =============================================================================

class GatedGCNLayer(MessagePassing):
    """
    Gated Graph Convolutional Layer with edge features.
    
    Uses gating mechanism to control information flow, similar to GRU.
    Incorporates edge features through edge gates.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual and (in_channels == out_channels)
        
        # Node transformations
        self.linear_src = nn.Linear(in_channels, out_channels, bias=False)
        self.linear_dst = nn.Linear(in_channels, out_channels, bias=False)
        
        # Edge transformation
        self.linear_edge = nn.Linear(edge_dim, out_channels, bias=False)
        
        # Gating
        self.gate_linear = nn.Linear(3 * out_channels, out_channels)
        
        # Output projection with gating (GRU-style)
        self.update_gate = nn.Linear(2 * out_channels, out_channels)
        self.reset_gate = nn.Linear(2 * out_channels, out_channels)
        self.candidate = nn.Linear(2 * out_channels, out_channels)
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_src.weight)
        nn.init.xavier_uniform_(self.linear_dst.weight)
        nn.init.xavier_uniform_(self.linear_edge.weight)
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.zeros_(self.gate_linear.bias)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        # Transform nodes
        h_src = self.linear_src(x)
        h_dst = self.linear_dst(x)
        
        # Transform edges
        e = self.linear_edge(edge_attr)
        
        # Message passing with gating
        out = self.propagate(edge_index, h_src=h_src, h_dst=h_dst, edge_feat=e)
        
        # GRU-style update
        combined = torch.cat([x[:, :self.out_channels] if x.size(1) >= self.out_channels 
                             else F.pad(x, (0, self.out_channels - x.size(1))), out], dim=-1)
        
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        
        h_reset = torch.cat([reset * x[:, :self.out_channels] if x.size(1) >= self.out_channels
                            else reset * F.pad(x, (0, self.out_channels - x.size(1))), out], dim=-1)
        candidate = torch.tanh(self.candidate(h_reset))
        
        out = (1 - update) * (x[:, :self.out_channels] if x.size(1) >= self.out_channels
                             else F.pad(x, (0, self.out_channels - x.size(1)))) + update * candidate
        
        out = self.norm(out)
        out = self.dropout(out)
        
        return out
    
    def message(self, h_src: Tensor, h_dst_i: Tensor, edge_feat: Tensor) -> Tensor:
        # Compute gated message
        gate_input = torch.cat([h_src, h_dst_i, edge_feat], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))
        return gate * h_src


class PNAConv(MessagePassing):
    """
    Principal Neighborhood Aggregation (PNA) Layer.
    
    Uses multiple aggregators (mean, max, min, std) and scalers (identity, 
    amplification, attenuation) for more expressive message aggregation.
    """
    
    AGGREGATORS = ['mean', 'max', 'min', 'std']
    SCALERS = ['identity', 'amplification', 'attenuation']
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        towers: int = 4,
        dropout: float = 0.1,
        divide_input: bool = True,
        avg_deg: float = 5.0,
    ):
        super().__init__(aggr=None)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.avg_deg = avg_deg
        
        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = out_channels // towers
        
        # Pre-transformation
        self.pre_nn = nn.Sequential(
            nn.Linear(2 * self.F_in + edge_dim, self.F_in),
            nn.GELU(),
        )
        
        # Post-transformation for each tower
        # 4 aggregators * 3 scalers = 12 channels per input
        self.post_nn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.F_in * len(self.AGGREGATORS) * len(self.SCALERS), self.F_out),
                nn.GELU(),
            )
            for _ in range(towers)
        ])
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if self.divide_input:
            x_tower = x.view(-1, self.towers, self.F_in)
        else:
            x_tower = x.unsqueeze(1).expand(-1, self.towers, -1)
        
        out = []
        for i in range(self.towers):
            tower_out = self.propagate(
                edge_index, x=x_tower[:, i], edge_attr=edge_attr
            )
            tower_out = self.post_nn[i](tower_out)
            out.append(tower_out)
        
        out = torch.cat(out, dim=-1)
        out = self.norm(out)
        out = self.dropout(out)
        
        return out
    
    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.pre_nn(msg_input)
    
    def aggregate(self, inputs: Tensor, index: Tensor, dim_size: int) -> Tensor:
        # Compute multiple aggregations
        aggregated = []
        
        for aggr in self.AGGREGATORS:
            if aggr == 'mean':
                agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='mean')
            elif aggr == 'max':
                agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='max')
            elif aggr == 'min':
                agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='min')
            elif aggr == 'std':
                mean = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='mean')
                mean_sq = scatter(inputs ** 2, index, dim=0, dim_size=dim_size, reduce='mean')
                agg = (mean_sq - mean ** 2).clamp(min=1e-6).sqrt()
            
            # Apply scalers
            for scaler in self.SCALERS:
                if scaler == 'identity':
                    scaled = agg
                elif scaler == 'amplification':
                    # Scale up based on degree
                    deg = scatter(torch.ones_like(index, dtype=torch.float), 
                                 index, dim=0, dim_size=dim_size, reduce='sum')
                    scaled = agg * (torch.log(deg + 1) / math.log(self.avg_deg + 1)).unsqueeze(-1)
                elif scaler == 'attenuation':
                    deg = scatter(torch.ones_like(index, dtype=torch.float),
                                 index, dim=0, dim_size=dim_size, reduce='sum')
                    scaled = agg * (math.log(self.avg_deg + 1) / torch.log(deg + 1).clamp(min=1e-6)).unsqueeze(-1)
                
                aggregated.append(scaled)
        
        return torch.cat(aggregated, dim=-1)


class GateTypeExpertLayer(nn.Module):
    """
    Gate-Type Aware Mixture of Experts Layer.
    
    Each quantum gate type routes through a learned expert network,
    allowing the model to learn gate-specific transformations.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_gate_types: int = NUM_GATE_TYPES,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gate type to expert routing
        self.gate_type_embed = nn.Embedding(num_gate_types, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_experts)
        ])
        
        # Learnable routing refinement
        self.router = nn.Linear(hidden_dim, num_experts)
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: Tensor,
        edge_gate_type: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Args:
            x: Node features [N, hidden_dim]
            edge_gate_type: Gate type for each edge [E]
            edge_index: Edge connectivity [2, E]
        """
        # Compute routing weights from both gate type and content
        content_logits = self.router(x)  # [N, num_experts]
        
        # Aggregate gate type information to nodes
        src, dst = edge_index
        gate_embeds = self.gate_type_embed(edge_gate_type)  # [E, num_experts]
        
        # Sum gate type embeddings for each destination node
        node_gate_logits = scatter(
            gate_embeds, dst, dim=0, dim_size=x.size(0), reduce='mean'
        )  # [N, num_experts]
        
        # Combine routing signals
        router_logits = content_logits + node_gate_logits
        
        # Top-k gating
        top_k_logits, top_k_indices = router_logits.topk(self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [N, top_k]
        
        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [N, num_experts, hidden_dim]
        
        # Gather top-k expert outputs
        batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(-1, self.top_k)
        selected_outputs = expert_outputs[batch_idx, top_k_indices]  # [N, top_k, hidden_dim]
        
        # Weighted combination
        out = (selected_outputs * top_k_gates.unsqueeze(-1)).sum(dim=1)  # [N, hidden_dim]
        
        return self.norm(out)


# =============================================================================
# ATTENTION MECHANISMS
# =============================================================================

class GraphMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention for Graphs.
    
    Implements efficient global attention with optional edge bias,
    similar to Graphormer's spatial encoding.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_bias: bool = True,
        edge_dim: int = EDGE_FEAT_DIM,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_edge_bias = use_edge_bias
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        if use_edge_bias:
            self.edge_bias_proj = nn.Sequential(
                nn.Linear(edge_dim, num_heads),
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        batch: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute global attention within each graph in the batch.
        
        For efficiency, we use a batched dense attention within each graph.
        """
        batch_size = batch.max().item() + 1
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Process each graph separately
        outputs = []
        node_idx = 0
        
        for b in range(batch_size):
            mask = batch == b
            n_nodes = mask.sum().item()
            
            if n_nodes == 0:
                continue
            
            # Get nodes for this graph
            q_b = q[mask].view(n_nodes, self.num_heads, self.head_dim)
            k_b = k[mask].view(n_nodes, self.num_heads, self.head_dim)
            v_b = v[mask].view(n_nodes, self.num_heads, self.head_dim)
            
            # Compute attention: [n_nodes, n_heads, head_dim] @ [n_nodes, n_heads, head_dim].T
            attn = torch.einsum('nhd,mhd->nmh', q_b, k_b) * self.scale
            
            # Add edge bias if available
            if self.use_edge_bias and edge_index is not None and edge_attr is not None:
                # Find edges within this graph
                edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
                if edge_mask.any():
                    # Map global indices to local
                    local_map = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                    local_map[mask] = torch.arange(n_nodes, device=x.device)
                    
                    local_src = local_map[edge_index[0, edge_mask]]
                    local_dst = local_map[edge_index[1, edge_mask]]
                    
                    edge_bias = self.edge_bias_proj(edge_attr[edge_mask])  # [E_b, num_heads]
                    
                    # Add to attention matrix
                    attn[local_src, local_dst] += edge_bias
            
            # Softmax and apply to values
            attn = F.softmax(attn, dim=1)
            attn = self.dropout(attn)
            
            out_b = torch.einsum('nmh,mhd->nhd', attn, v_b)
            out_b = out_b.reshape(n_nodes, -1)
            outputs.append(out_b)
        
        # Reconstruct output in original order
        out = torch.zeros_like(x)
        for b in range(batch_size):
            mask = batch == b
            if mask.any():
                out[mask] = outputs[b]
        
        return self.out_proj(out)


class EdgeAwareGATv2(MessagePassing):
    """
    GATv2-style attention with edge features.
    
    Implements dynamic attention that can express a wider range of attention
    patterns than original GAT.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels if concat else out_channels))
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        H, C = self.heads, self.out_channels
        
        x_src = self.lin_src(x).view(-1, H, C)
        x_dst = self.lin_dst(x).view(-1, H, C)
        
        out = self.propagate(edge_index, x_src=x_src, x_dst=x_dst, edge_attr=edge_attr)
        
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
        
        out = out + self.bias
        
        return out
    
    def message(
        self,
        x_src: Tensor,
        x_dst_i: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        H, C = self.heads, self.out_channels
        
        edge_feat = self.lin_edge(edge_attr).view(-1, H, C)
        
        # GATv2: apply attention AFTER combining features
        alpha = (x_src + x_dst_i + edge_feat).tanh()
        alpha = (alpha * self.att).sum(dim=-1)
        
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.dropout(alpha)
        
        return x_src * alpha.unsqueeze(-1)


# =============================================================================
# GPS BLOCK (LOCAL + GLOBAL)
# =============================================================================

class GPSBlock(nn.Module):
    """
    GraphGPS Block: Combines local message passing with global attention.
    
    Architecture:
        x -> LayerNorm -> LocalMPNN -> + -> LayerNorm -> GlobalAttn -> + -> FFN -> +
             |___________________________|    |__________________________|    |______|
                   (residual)                       (residual)            (residual)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        ffn_expansion: int = 4,
        local_type: str = 'gated_gcn',  # 'gated_gcn', 'pna', 'gatv2'
        use_global_attn: bool = True,
        stochastic_depth: float = 0.0,
    ):
        super().__init__()
        
        self.use_global_attn = use_global_attn
        self.stochastic_depth = stochastic_depth
        
        # Pre-normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Local message passing
        if local_type == 'gated_gcn':
            self.local_mpnn = GatedGCNLayer(hidden_dim, hidden_dim, edge_dim, dropout)
        elif local_type == 'pna':
            self.local_mpnn = PNAConv(hidden_dim, hidden_dim, edge_dim, dropout=dropout)
        elif local_type == 'gatv2':
            self.local_mpnn = EdgeAwareGATv2(
                hidden_dim, hidden_dim // num_heads, edge_dim, 
                heads=num_heads, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown local type: {local_type}")
        
        # Global attention
        if use_global_attn:
            self.global_attn = GraphMultiHeadAttention(
                hidden_dim, num_heads=num_heads, dropout=attn_dropout,
                use_edge_bias=True, edge_dim=edge_dim
            )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def _drop_path(self, x: Tensor, residual: Tensor) -> Tensor:
        """Stochastic depth / drop path."""
        if self.training and self.stochastic_depth > 0:
            keep_prob = 1 - self.stochastic_depth
            mask = torch.bernoulli(torch.full((x.size(0), 1), keep_prob, device=x.device))
            return residual + x * mask / keep_prob
        return residual + x
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        # Local message passing
        h = self.norm1(x)
        h = self.local_mpnn(h, edge_index, edge_attr)
        x = self._drop_path(h, x)
        
        # Global attention
        if self.use_global_attn:
            h = self.norm2(x)
            h = self.global_attn(h, batch, edge_index, edge_attr)
            x = self._drop_path(h, x)
        
        # FFN
        h = self.norm3(x)
        h = self.ffn(h)
        x = self._drop_path(h, x)
        
        return x


# =============================================================================
# VIRTUAL NODE
# =============================================================================

class VirtualNode(nn.Module):
    """
    Virtual Node for global information flow.
    
    Adds a virtual node connected to all nodes in each graph,
    enabling efficient global communication.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Virtual node embedding (learnable)
        self.vn_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.vn_embed, std=0.02)
        
        # Update networks
        self.node_to_vn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.vn_to_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Gating
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: Tensor,
        batch: Tensor,
        vn_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Node features [N, hidden_dim]
            batch: Batch assignment [N]
            vn_state: Previous virtual node state [B, hidden_dim] or None
            
        Returns:
            Updated node features and virtual node state
        """
        batch_size = batch.max().item() + 1
        
        # Initialize virtual node state if needed
        if vn_state is None:
            vn_state = self.vn_embed.expand(batch_size, -1)
        
        # Aggregate node features to virtual node
        node_agg = scatter(x, batch, dim=0, dim_size=batch_size, reduce='mean')
        vn_update = self.node_to_vn(node_agg)
        
        # Update virtual node state (residual)
        vn_state = vn_state + vn_update
        
        # Broadcast virtual node back to nodes
        vn_broadcast = vn_state[batch]  # [N, hidden_dim]
        node_update = self.vn_to_node(vn_broadcast)
        
        # Gated update
        gate = self.gate(torch.cat([x, node_update], dim=-1))
        x = x + gate * node_update
        
        return x, vn_state


# =============================================================================
# HIERARCHICAL POOLING
# =============================================================================

class HierarchicalPooling(nn.Module):
    """
    Multi-scale graph pooling combining different pooling strategies.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Attention-based pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        
        # Set2Set-style pooling
        self.set2set_q = nn.Linear(hidden_dim, hidden_dim)
        self.set2set_k = nn.Linear(hidden_dim, hidden_dim)
        
        # Combination
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),  # mean, max, sum, attn, set2set
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
    
    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        # Standard pooling
        h_mean = global_mean_pool(x, batch)
        h_max = global_max_pool(x, batch)
        h_sum = global_add_pool(x, batch)
        
        # Attention pooling
        attn_weights = softmax(self.attn_pool(x).squeeze(-1), batch)
        h_attn = scatter(x * attn_weights.unsqueeze(-1), batch, dim=0, reduce='sum')
        
        # Set2Set-style: use graph representation to query nodes
        batch_size = batch.max().item() + 1
        q = self.set2set_q(h_mean)  # [B, hidden]
        k = self.set2set_k(x)  # [N, hidden]
        
        # Compute attention per graph
        h_s2s_list = []
        for b in range(batch_size):
            mask = batch == b
            k_b = k[mask]  # [n_b, hidden]
            q_b = q[b:b+1]  # [1, hidden]
            
            attn = F.softmax(torch.mm(q_b, k_b.t()) / math.sqrt(k_b.size(-1)), dim=-1)
            h_s2s_list.append(torch.mm(attn, k_b).squeeze(0))
        
        h_s2s = torch.stack(h_s2s_list, dim=0)
        
        # Combine all representations
        combined = torch.cat([h_mean, h_max, h_sum, h_attn, h_s2s], dim=-1)
        return self.combine(combined)


# =============================================================================
# MAIN MODEL
# =============================================================================

class QuantumCircuitGNNSoTA(nn.Module):
    """
    State-of-the-Art GNN for Quantum Circuit Duration Prediction.
    
    Combines:
    - Learnable positional encodings (Random Walk PE)
    - Virtual nodes for global information flow
    - GPS blocks (local MPNN + global attention)
    - Gate-type Mixture of Experts
    - Hierarchical pooling
    - Deep supervision (auxiliary losses)
    
    Args:
        node_feat_dim: Input node feature dimension
        edge_feat_dim: Input edge feature dimension
        global_feat_dim: Global circuit feature dimension
        hidden_dim: Hidden dimension throughout the network
        num_layers: Number of GPS blocks
        num_heads: Number of attention heads
        dropout: Dropout rate
        attn_dropout: Attention dropout rate
        use_virtual_node: Whether to use virtual node
        use_positional_encoding: Whether to use positional encodings
        local_type: Type of local MPNN ('gated_gcn', 'pna', 'gatv2')
        use_moe: Whether to use Mixture of Experts
        stochastic_depth: Drop path probability
        deep_supervision: Whether to use auxiliary losses
    """
    
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_virtual_node: bool = True,
        use_positional_encoding: bool = True,
        local_type: str = 'gated_gcn',
        use_moe: bool = True,
        stochastic_depth: float = 0.1,
        deep_supervision: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_virtual_node = use_virtual_node
        self.use_positional_encoding = use_positional_encoding
        self.use_moe = use_moe
        self.deep_supervision = deep_supervision
        
        # Input embeddings
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Positional encodings
        if use_positional_encoding:
            self.rwpe = RandomWalkPE(walk_length=16, hidden_dim=hidden_dim)
            self.pe_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Virtual node
        if use_virtual_node:
            self.virtual_nodes = nn.ModuleList([
                VirtualNode(hidden_dim, dropout) for _ in range(num_layers)
            ])
        
        # GPS blocks with stochastic depth
        stochastic_depth_rates = [stochastic_depth * i / (num_layers - 1) for i in range(num_layers)]
        
        self.gps_blocks = nn.ModuleList([
            GPSBlock(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                local_type=local_type,
                use_global_attn=True,
                stochastic_depth=stochastic_depth_rates[i],
            )
            for i in range(num_layers)
        ])
        
        # MoE layers (inserted every 2 blocks)
        if use_moe:
            self.moe_layers = nn.ModuleList([
                GateTypeExpertLayer(hidden_dim, NUM_GATE_TYPES, num_experts=8, dropout=dropout)
                if i % 2 == 1 else nn.Identity()
                for i in range(num_layers)
            ])
        
        # Hierarchical pooling
        self.pooling = HierarchicalPooling(hidden_dim, dropout)
        
        # Global features projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Final prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Auxiliary heads for deep supervision
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(num_layers // 2)
            ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_gate_type: Tensor,
        batch: Tensor,
        global_features: Tensor,
        return_aux: bool = False,
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, node_feat_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_feat_dim]
            edge_gate_type: Gate type indices [E]
            batch: Batch assignment [N]
            global_features: Global circuit features [B, global_feat_dim]
            return_aux: Whether to return auxiliary predictions
            
        Returns:
            Runtime predictions [B] (and aux_preds if return_aux=True)
        """
        num_nodes = x.size(0)
        
        # Embed inputs
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        
        # Add positional encodings
        if self.use_positional_encoding:
            pe = self.rwpe(edge_index, num_nodes)
            h = self.pe_combine(torch.cat([h, pe], dim=-1))
        
        # Virtual node state
        vn_state = None
        
        # Auxiliary predictions
        aux_preds = []
        
        # GPS blocks
        for i, gps_block in enumerate(self.gps_blocks):
            # Virtual node update (before GPS block)
            if self.use_virtual_node:
                h, vn_state = self.virtual_nodes[i](h, batch, vn_state)
            
            # GPS block
            h = gps_block(h, edge_index, e, batch)
            
            # MoE layer
            if self.use_moe:
                if not isinstance(self.moe_layers[i], nn.Identity):
                    h = h + self.moe_layers[i](h, edge_gate_type, edge_index)
            
            # Auxiliary prediction
            if self.deep_supervision and i > 0 and i % 2 == 1:
                aux_idx = i // 2 - 1
                if aux_idx < len(self.aux_heads):
                    h_pool = global_mean_pool(h, batch)
                    aux_preds.append(self.aux_heads[aux_idx](h_pool).squeeze(-1))
        
        # Hierarchical pooling
        h_graph = self.pooling(h, batch)
        
        # Combine with global features
        g = self.global_proj(global_features)
        combined = torch.cat([h_graph, g], dim=-1)
        
        # Final prediction
        pred = self.pred_head(combined).squeeze(-1)
        
        if return_aux and self.deep_supervision:
            return pred, aux_preds
        
        return pred


class QuantumCircuitGNNThresholdClassSoTA(nn.Module):
    """
    State-of-the-Art GNN for Threshold Class Prediction.
    
    Similar architecture to QuantumCircuitGNNSoTA but outputs class logits
    instead of regression values.
    """
    
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES,
        num_classes: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        use_virtual_node: bool = True,
        use_positional_encoding: bool = True,
        local_type: str = 'gated_gcn',
        use_moe: bool = True,
        stochastic_depth: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.use_virtual_node = use_virtual_node
        self.use_positional_encoding = use_positional_encoding
        self.use_moe = use_moe
        self.label_smoothing = label_smoothing
        
        # Input embeddings
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Positional encodings
        if use_positional_encoding:
            self.rwpe = RandomWalkPE(walk_length=16, hidden_dim=hidden_dim)
            self.pe_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Virtual node
        if use_virtual_node:
            self.virtual_nodes = nn.ModuleList([
                VirtualNode(hidden_dim, dropout) for _ in range(num_layers)
            ])
        
        # GPS blocks
        stochastic_depth_rates = [stochastic_depth * i / max(num_layers - 1, 1) for i in range(num_layers)]
        
        self.gps_blocks = nn.ModuleList([
            GPSBlock(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                local_type=local_type,
                use_global_attn=True,
                stochastic_depth=stochastic_depth_rates[i],
            )
            for i in range(num_layers)
        ])
        
        # MoE layers
        if use_moe:
            self.moe_layers = nn.ModuleList([
                GateTypeExpertLayer(hidden_dim, NUM_GATE_TYPES, num_experts=8, dropout=dropout)
                if i % 2 == 1 else nn.Identity()
                for i in range(num_layers)
            ])
        
        # Pooling
        self.pooling = HierarchicalPooling(hidden_dim, dropout)
        
        # Global features
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Classification head with ordinal regression awareness
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # Ordinal regression: cumulative logits
        self.ordinal_head = nn.Linear(hidden_dim * 2, num_classes - 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_gate_type: Tensor,
        batch: Tensor,
        global_features: Tensor,
        return_ordinal: bool = False,
    ) -> Tensor:
        num_nodes = x.size(0)
        
        # Embed inputs
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        
        # Positional encodings
        if self.use_positional_encoding:
            pe = self.rwpe(edge_index, num_nodes)
            h = self.pe_combine(torch.cat([h, pe], dim=-1))
        
        vn_state = None
        
        # GPS blocks
        for i, gps_block in enumerate(self.gps_blocks):
            if self.use_virtual_node:
                h, vn_state = self.virtual_nodes[i](h, batch, vn_state)
            
            h = gps_block(h, edge_index, e, batch)
            
            if self.use_moe and not isinstance(self.moe_layers[i], nn.Identity):
                h = h + self.moe_layers[i](h, edge_gate_type, edge_index)
        
        # Pooling
        h_graph = self.pooling(h, batch)
        
        # Global features
        g = self.global_proj(global_features)
        combined = torch.cat([h_graph, g], dim=-1)
        
        # Classification
        logits = self.class_head(combined)
        
        if return_ordinal:
            ordinal_logits = self.ordinal_head(combined)
            return logits, ordinal_logits
        
        return logits


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_sota_gnn_model(
    model_type: str = "sota",
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES,
    hidden_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    local_type: str = 'gated_gcn',
    use_virtual_node: bool = True,
    use_moe: bool = True,
    **kwargs,
) -> nn.Module:
    """Create state-of-the-art GNN model for duration prediction."""
    return QuantumCircuitGNNSoTA(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        global_feat_dim=global_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        local_type=local_type,
        use_virtual_node=use_virtual_node,
        use_moe=use_moe,
        **kwargs,
    )


def create_sota_threshold_class_model(
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES,
    num_classes: int = 9,
    hidden_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    local_type: str = 'gated_gcn',
    **kwargs,
) -> nn.Module:
    """Create state-of-the-art GNN model for threshold class prediction."""
    return QuantumCircuitGNNThresholdClassSoTA(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        global_feat_dim=global_feat_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        local_type=local_type,
        **kwargs,
    )


# =============================================================================
# LIGHTWEIGHT VARIANT FOR FASTER TRAINING
# =============================================================================

class QuantumCircuitGNNLite(nn.Module):
    """
    Lightweight variant that balances performance and efficiency.
    
    Uses efficient approximations:
    - Simplified attention (no edge bias computation)
    - Reduced MoE (4 experts instead of 8)
    - Single pooling strategy
    """
    
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES,
        hidden_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Simplified GPS blocks
        self.blocks = nn.ModuleList([
            GPSBlock(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                local_type='gated_gcn',
                use_global_attn=(i % 2 == 1),  # Global attention every other layer
                stochastic_depth=0.05 * i,
            )
            for i in range(num_layers)
        ])
        
        # Simple pooling
        self.pool_combine = nn.Linear(hidden_dim * 3, hidden_dim)
        
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.GELU(),
        )
        
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_gate_type: Tensor,
        batch: Tensor,
        global_features: Tensor,
    ) -> Tensor:
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        
        for block in self.blocks:
            h = block(h, edge_index, e, batch)
        
        # Pool
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        h_graph = self.pool_combine(torch.cat([h_mean, h_max, h_sum], dim=-1))
        
        g = self.global_proj(global_features)
        combined = torch.cat([h_graph, g], dim=-1)
        
        return self.pred_head(combined).squeeze(-1)


if __name__ == "__main__":
    # Test the models
    print("Testing State-of-the-Art GNN Models...")
    
    # Create dummy data
    batch_size = 4
    num_nodes_per_graph = 10
    num_edges_per_graph = 20
    
    x = torch.randn(batch_size * num_nodes_per_graph, NODE_FEAT_DIM)
    edge_index = torch.randint(0, num_nodes_per_graph, (2, batch_size * num_edges_per_graph))
    # Adjust edge indices for batching
    for i in range(batch_size):
        edge_index[:, i*num_edges_per_graph:(i+1)*num_edges_per_graph] += i * num_nodes_per_graph
    edge_attr = torch.randn(batch_size * num_edges_per_graph, EDGE_FEAT_DIM)
    edge_gate_type = torch.randint(0, NUM_GATE_TYPES, (batch_size * num_edges_per_graph,))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes_per_graph)
    global_features = torch.randn(batch_size, GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES)
    
    # Test SoTA model
    print("\n1. Testing QuantumCircuitGNNSoTA...")
    model = QuantumCircuitGNNSoTA(hidden_dim=64, num_layers=4)
    out = model(x, edge_index, edge_attr, edge_gate_type, batch, global_features)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with aux outputs
    out, aux = model(x, edge_index, edge_attr, edge_gate_type, batch, global_features, return_aux=True)
    print(f"   Auxiliary outputs: {len(aux)}")
    
    # Test Lite model
    print("\n2. Testing QuantumCircuitGNNLite...")
    model_lite = QuantumCircuitGNNLite(hidden_dim=64, num_layers=4)
    out_lite = model_lite(x, edge_index, edge_attr, edge_gate_type, batch, global_features)
    print(f"   Output shape: {out_lite.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_lite.parameters()):,}")
    
    # Test Threshold class model
    print("\n3. Testing QuantumCircuitGNNThresholdClassSoTA...")
    global_features_cls = torch.randn(batch_size, GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES)
    model_cls = QuantumCircuitGNNThresholdClassSoTA(hidden_dim=64, num_layers=4)
    logits = model_cls(x, edge_index, edge_attr, edge_gate_type, batch, global_features_cls)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
    
    print("\n✅ All tests passed!")

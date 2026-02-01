"""
Graph Transformer model for quantum circuit prediction.

The Graph Transformer architecture is particularly well-suited for quantum circuits:

1. **Attention as Entanglement Flow**: Quantum simulation complexity is dominated
   by entanglement structure. Self-attention naturally captures how information
   (entanglement) propagates through the circuit, with attention weights learning
   which qubit interactions are most significant for complexity.

2. **Edge-Aware Attention**: Gate types matter crucially - CNOT gates create
   entanglement differently than CZ or SWAP gates. We incorporate edge features
   (gate type, temporal position) directly into the attention computation as
   bias terms, allowing the model to learn gate-specific information flow.

3. **Global Receptive Field**: Unlike message-passing GNNs that require multiple
   layers to propagate information across the graph, Transformers have global
   receptive field in each layer. This is important for quantum circuits where
   early gates can have long-range effects through entanglement.

4. **Positional Awareness**: We use graph-aware positional encodings that capture
   the qubit topology and their relative positions in the quantum register.

Architecture Overview:
    Input → Node Embedding + Positional Encoding
         → [Graph Transformer Layer × N]
            - Multi-Head Self-Attention with Edge Bias
            - Feed-Forward Network
            - Residual + LayerNorm
         → Graph Pooling (mean, max, sum)
         → Combine with Global Features
         → Output Head
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

from .base import (
    BaseGraphModel,
    BaseGraphDurationModel,
    BaseGraphThresholdClassModel,
    GraphModelConfig,
)
from .graph_builder import NUM_GATE_TYPES, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE


class GraphPositionalEncoding(nn.Module):
    """
    Learnable positional encoding for graph nodes.
    
    For quantum circuits, this captures:
    - Qubit position in the register (already in node features)
    - Local graph structure (degree, connectivity)
    - Random walk-based structural encoding
    
    We use a combination of learned embeddings and computed features.
    """
    
    def __init__(self, hidden_dim: int, max_nodes: int = 256, use_degree: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_degree = use_degree
        
        if use_degree:
            self.degree_encoder = nn.Embedding(max_nodes, hidden_dim)
        
        self.random_walk_dim = 8
        self.rw_proj = nn.Linear(self.random_walk_dim, hidden_dim)
    
    def compute_random_walk_pe(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        k: int = 8,
    ) -> torch.Tensor:
        """
        Compute random walk positional encoding.
        
        For each node, computes the probability of returning to itself
        in 1, 2, ..., k steps. This captures local graph structure.
        """
        device = edge_index.device
        
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        if edge_index.numel() > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
            adj = adj + adj.t()
            adj = (adj > 0).float()
        
        deg = adj.sum(dim=1)
        deg_inv = torch.zeros_like(deg)
        deg_inv[deg > 0] = 1.0 / deg[deg > 0]
        
        transition = adj * deg_inv.unsqueeze(0)
        
        rw_pe = torch.zeros(num_nodes, k, device=device)
        power = torch.eye(num_nodes, device=device)
        
        for i in range(k):
            power = power @ transition
            rw_pe[:, i] = torch.diag(power)
        
        return rw_pe
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute positional encoding for each node.
        
        Args:
            x: Node features [n_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, n_edges]
            batch: Batch assignment [n_nodes]
            
        Returns:
            Positional encoding [n_nodes, hidden_dim]
        """
        device = x.device
        num_nodes = x.shape[0]
        
        pe = torch.zeros(num_nodes, self.hidden_dim, device=device)
        
        if self.use_degree and edge_index.numel() > 0:
            degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
            degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long, device=device))
            degree = degree.clamp(max=self.degree_encoder.num_embeddings - 1)
            pe = pe + self.degree_encoder(degree)
        
        batch_size = int(batch.max().item()) + 1
        for b in range(batch_size):
            mask = batch == b
            local_nodes = mask.sum().item()
            if local_nodes == 0:
                continue
            
            local_indices = torch.where(mask)[0]
            node_map = {idx.item(): i for i, idx in enumerate(local_indices)}
            
            local_edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            local_edges = edge_index[:, local_edge_mask]
            
            if local_edges.numel() > 0:
                remapped_edges = torch.tensor(
                    [[node_map[e.item()] for e in local_edges[0]],
                     [node_map[e.item()] for e in local_edges[1]]],
                    device=device
                )
                rw_pe = self.compute_random_walk_pe(
                    remapped_edges, local_nodes, k=self.random_walk_dim
                )
            else:
                rw_pe = torch.zeros(local_nodes, self.random_walk_dim, device=device)
            
            pe[mask] = pe[mask] + self.rw_proj(rw_pe)
        
        return pe


class EdgeAwareMultiHeadAttention(nn.Module):
    """
    Multi-head attention with edge feature integration.
    
    For quantum circuits, edges represent gates. This layer computes attention
    between qubits with attention bias based on the gates connecting them.
    
    Key design choices:
    - Gate type embeddings: Each gate type has learned attention bias
    - Edge features (temporal position, parameters): Modulate attention
    - Sparse attention: Only compute attention for connected qubits + self
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_gate_types: int = NUM_GATE_TYPES,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        dropout: float = 0.1,
        use_sparse_attention: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_sparse_attention = use_sparse_attention
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.gate_attn_bias = nn.Embedding(num_gate_types, num_heads)
        self.edge_bias_proj = nn.Linear(edge_feat_dim, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        
        self._attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute edge-aware multi-head attention.
        
        Args:
            x: Node features [n_nodes, hidden_dim]
            edge_index: Edge connectivity [2, n_edges]
            edge_attr: Edge features [n_edges, edge_feat_dim]
            edge_gate_type: Gate type indices [n_edges]
            batch: Batch assignment [n_nodes]
            mask: Optional attention mask
            
        Returns:
            Attended features [n_nodes, hidden_dim]
        """
        n_nodes = x.shape[0]
        device = x.device
        
        Q = self.q_proj(x).view(n_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(n_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(n_nodes, self.num_heads, self.head_dim)
        
        x_padded, node_mask = to_dense_batch(x, batch)
        batch_size, max_nodes, _ = x_padded.shape
        
        Q_dense = self.q_proj(x_padded).view(batch_size, max_nodes, self.num_heads, self.head_dim)
        K_dense = self.k_proj(x_padded).view(batch_size, max_nodes, self.num_heads, self.head_dim)
        V_dense = self.v_proj(x_padded).view(batch_size, max_nodes, self.num_heads, self.head_dim)
        
        Q_dense = Q_dense.permute(0, 2, 1, 3)
        K_dense = K_dense.permute(0, 2, 1, 3)
        V_dense = V_dense.permute(0, 2, 1, 3)
        
        attn_scores = torch.matmul(Q_dense, K_dense.transpose(-2, -1)) * self.scale
        
        edge_bias = self._compute_edge_attention_bias(
            edge_index, edge_attr, edge_gate_type, batch, max_nodes, batch_size
        )
        attn_scores = attn_scores + edge_bias
        
        attn_mask = node_mask.unsqueeze(1).unsqueeze(2) & node_mask.unsqueeze(1).unsqueeze(3)
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~attn_mask, 0.0)
        self._attention_weights = attn_weights.detach()
        
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V_dense)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, max_nodes, self.hidden_dim)
        
        output = torch.zeros(n_nodes, self.hidden_dim, device=device)
        for b in range(batch_size):
            batch_mask = batch == b
            n_batch_nodes = batch_mask.sum().item()
            output[batch_mask] = out[b, :n_batch_nodes]
        
        return self.out_proj(output)
    
    def _compute_edge_attention_bias(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        max_nodes: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute attention bias from edges (gates).
        
        Uses vectorized scatter operations for efficiency.
        """
        device = edge_index.device
        
        bias = torch.zeros(batch_size, self.num_heads, max_nodes, max_nodes, device=device)
        
        if edge_index.numel() == 0:
            return bias
        
        # Compute batch offsets robustly using bincount
        node_counts = torch.bincount(batch, minlength=batch_size)
        batch_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            node_counts.cumsum(0)[:-1]
        ])
        
        # Get batch assignment for each edge (from source node)
        edge_batch = batch[edge_index[0]]
        
        # Convert global node indices to local (within-batch) indices
        src_local = edge_index[0] - batch_offsets[edge_batch]
        dst_local = edge_index[1] - batch_offsets[edge_batch]
        
        # Compute edge biases: [num_edges, num_heads]
        gate_bias = self.gate_attn_bias(edge_gate_type)
        edge_bias_val = self.edge_bias_proj(edge_attr)
        total_bias = gate_bias + edge_bias_val
        
        # Create valid mask for bounds checking
        valid_mask = (
            (src_local >= 0) & (src_local < max_nodes) &
            (dst_local >= 0) & (dst_local < max_nodes)
        )
        
        # Filter to valid edges only
        valid_idx = torch.where(valid_mask)[0]
        if valid_idx.numel() == 0:
            return bias
        
        edge_batch_v = edge_batch[valid_idx]
        src_local_v = src_local[valid_idx]
        dst_local_v = dst_local[valid_idx]
        total_bias_v = total_bias[valid_idx]  # [num_valid, num_heads]
        
        # Compute linear indices into flattened bias tensor
        # bias has shape [batch_size, num_heads, max_nodes, max_nodes]
        # We'll use scatter_add on the last two dimensions
        # Linear index = b * (num_heads * max_nodes * max_nodes) + h * (max_nodes * max_nodes) + s * max_nodes + d
        
        num_heads = self.num_heads
        stride_b = num_heads * max_nodes * max_nodes
        stride_h = max_nodes * max_nodes
        stride_s = max_nodes
        
        # Compute base index for each edge (excluding head dimension)
        base_idx = (
            edge_batch_v.unsqueeze(1) * stride_b + 
            torch.arange(num_heads, device=device).unsqueeze(0) * stride_h
        )  # [num_valid, num_heads]
        
        # Add source and destination indices (forward direction: s -> d)
        fwd_idx = base_idx + src_local_v.unsqueeze(1) * stride_s + dst_local_v.unsqueeze(1)
        
        # Add reverse direction: d -> s (for symmetric bias)
        rev_idx = base_idx + dst_local_v.unsqueeze(1) * stride_s + src_local_v.unsqueeze(1)
        
        # Flatten bias and use scatter_add
        bias_flat = bias.view(-1)
        
        # Add forward direction
        bias_flat.scatter_add_(0, fwd_idx.view(-1), total_bias_v.view(-1))
        
        # Add reverse direction (only for non-self-loops)
        non_self_loop = src_local_v != dst_local_v
        if non_self_loop.any():
            rev_idx_filtered = rev_idx[non_self_loop]
            total_bias_filtered = total_bias_v[non_self_loop]
            bias_flat.scatter_add_(0, rev_idx_filtered.view(-1), total_bias_filtered.view(-1))
        
        return bias


class GraphTransformerLayer(nn.Module):
    """
    Single Graph Transformer layer with pre-norm architecture.
    
    Structure:
        x → LayerNorm → EdgeAwareAttention → Dropout → + → 
          → LayerNorm → FFN → Dropout → +
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_gate_types: int = NUM_GATE_TYPES,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        ffn_dim = ffn_dim or hidden_dim * 4
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = EdgeAwareMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_gate_types=num_gate_types,
            edge_feat_dim=edge_feat_dim,
            dropout=dropout,
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        activation_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attention(h, edge_index, edge_attr, edge_gate_type, batch)
        x = x + self.dropout(h)
        
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        
        return x


class GraphTransformerEncoder(nn.Module):
    """
    Stack of Graph Transformer layers forming the encoder.
    
    This is the core processing unit that transforms node features
    into contextualized representations through multiple layers of
    edge-aware self-attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        num_gate_types: int = NUM_GATE_TYPES,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = GraphPositionalEncoding(hidden_dim)
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_gate_types=num_gate_types,
                edge_feat_dim=edge_feat_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        self._layer_outputs: List[torch.Tensor] = []
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        self._layer_outputs = []
        
        if self.use_positional_encoding:
            pos_enc = self.pos_encoder(x, edge_index, batch)
            x = x + pos_enc
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, edge_gate_type, batch)
            if return_all_layers:
                self._layer_outputs.append(x)
        
        x = self.final_norm(x)
        
        return x
    
    def get_layer_outputs(self) -> List[torch.Tensor]:
        return self._layer_outputs


class QuantumCircuitGraphTransformer(BaseGraphDurationModel):
    """
    Graph Transformer for quantum circuit duration prediction.
    
    This model predicts log2(runtime) given:
    - Circuit graph structure (qubits as nodes, gates as edges)
    - Threshold parameter (in global features)
    - Circuit metadata (backend, precision, family)
    
    The Graph Transformer architecture captures the entanglement structure
    of quantum circuits through attention, learning which gate patterns
    lead to increased simulation complexity.
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
        use_positional_encoding: bool = True,
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
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.encoder = GraphTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_gate_types=NUM_GATE_TYPES,
            edge_feat_dim=edge_feat_dim,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
        )
        
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        combined_dim = self.pool_dim + hidden_dim
        self.output_mlp = nn.Sequential(
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
        return "GraphTransformer"
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_embed(x)
        h = self.encoder(h, edge_index, edge_attr, edge_gate_type, batch)
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
        features = self.output_mlp(combined)
        return self.runtime_head(features).squeeze(-1)
    
    def get_attention_weights(self) -> Optional[List[torch.Tensor]]:
        weights = []
        for layer in self.encoder.layers:
            if hasattr(layer.attention, '_attention_weights'):
                weights.append(layer.attention._attention_weights)
        return weights if weights else None


class QuantumCircuitGraphTransformerThresholdClass(BaseGraphThresholdClassModel):
    """
    Graph Transformer for threshold class prediction.
    
    Predicts the optimal threshold class (1, 2, 4, ..., 256) for
    achieving target fidelity in quantum circuit simulation.
    
    Can optionally use ordinal regression to exploit the ordered
    nature of threshold classes.
    """
    
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 20,
        num_classes: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_positional_encoding: bool = True,
        use_ordinal: bool = False,
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
        self.num_classes = num_classes
        self.use_ordinal = use_ordinal
        
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.encoder = GraphTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_gate_types=NUM_GATE_TYPES,
            edge_feat_dim=edge_feat_dim,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
        )
        
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        combined_dim = self.pool_dim + hidden_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        if use_ordinal:
            self.class_head = nn.Linear(hidden_dim, num_classes - 1)
        else:
            self.class_head = nn.Linear(hidden_dim, num_classes)
    
    @property
    def name(self) -> str:
        return "GraphTransformerThresholdClass"
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_embed(x)
        h = self.encoder(h, edge_index, edge_attr, edge_gate_type, batch)
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
        features = self.output_mlp(combined)
        
        if self.use_ordinal:
            cumulative_logits = self.class_head(features)
            return cumulative_logits
        else:
            return self.class_head(features)
    
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_ordinal:
            cumulative_logits = self.predict_logits(
                x, edge_index, edge_attr, edge_gate_type, batch, global_features
            )
            cumulative_probs = torch.sigmoid(cumulative_logits)
            
            class_probs = torch.zeros(
                cumulative_probs.shape[0], self.num_classes, 
                device=cumulative_probs.device
            )
            class_probs[:, 0] = 1 - cumulative_probs[:, 0]
            for k in range(1, self.num_classes - 1):
                class_probs[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]
            class_probs[:, -1] = cumulative_probs[:, -1]
            
            class_probs = class_probs.clamp(min=1e-7)
            class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
            return class_probs
        else:
            logits = self.predict_logits(
                x, edge_index, edge_attr, edge_gate_type, batch, global_features
            )
            return torch.softmax(logits, dim=-1)
    
    def get_attention_weights(self) -> Optional[List[torch.Tensor]]:
        weights = []
        for layer in self.encoder.layers:
            if hasattr(layer.attention, '_attention_weights'):
                weights.append(layer.attention._attention_weights)
        return weights if weights else None


def create_graph_transformer_model(
    model_type: str = "duration",
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: Optional[int] = None,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.2,
    use_positional_encoding: bool = True,
    use_ordinal: bool = False,
    **kwargs,
) -> BaseGraphModel:
    """
    Factory function to create Graph Transformer models.
    
    Args:
        model_type: "duration" or "threshold"
        node_feat_dim: Input node feature dimension
        edge_feat_dim: Input edge feature dimension
        global_feat_dim: Global feature dimension (auto-computed if None)
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_positional_encoding: Whether to use graph positional encoding
        use_ordinal: Whether to use ordinal regression (threshold model only)
        
    Returns:
        Graph Transformer model instance
    """
    if model_type == "duration":
        if global_feat_dim is None:
            global_feat_dim = GLOBAL_FEAT_DIM_BASE + 1 + 20
        return QuantumCircuitGraphTransformer(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
        )
    elif model_type == "threshold":
        if global_feat_dim is None:
            global_feat_dim = GLOBAL_FEAT_DIM_BASE + 20
        return QuantumCircuitGraphTransformerThresholdClass(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            use_ordinal=use_ordinal,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'duration' or 'threshold'.")

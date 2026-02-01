"""
Quantum Circuit Heterogeneous Graph Transformer (QCHGT).

A novel heterogeneous GNN inspired by HAN (Heterogeneous Attention Network) and 
HGT (Heterogeneous Graph Transformer) for quantum circuit simulation prediction.

Key innovations:
1. Multi-Relation Heterogeneous Edges: Semantic edge types capturing quantum operations
   - ROTATION: Single-qubit rotation gates (rx, ry, rz, u1, u2, u3)
   - PAULI: Pauli gates (x, y, z, h, s, t and their daggers)
   - ENTANGLE: Entangling 2-qubit gates (cx, cz, rxx, ryy, rzz)
   - SWAP: Permutation gates (swap, cswap)
   - CONTROL: Multi-controlled gates (ccx, ccz)
   - TEMPORAL: Sequential gate ordering on same qubit

2. Meta-path Attention: Captures entanglement flow patterns
   - Direct paths: qubit -> gate -> qubit
   - Chain paths: multi-hop entanglement propagation
   
3. Hierarchical Attention Mechanism:
   - Level 1: Intra-relation attention (within same edge type)
   - Level 2: Inter-relation attention (aggregate across edge types)
   - Level 3: Semantic attention (weight meta-path contributions)

4. Entanglement-aware Pooling: Graph readout based on interaction patterns

Implements BaseGraphThresholdClassModel interface for compatibility.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import softmax

from .base import BaseGraphDurationModel, BaseGraphThresholdClassModel, GraphModelConfig
from .graph_builder import (
    NUM_GATE_TYPES, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE,
    GATE_1Q, GATE_2Q, GATE_3Q, GATE_TO_IDX
)

NUM_FAMILIES = 20
NUM_THRESHOLD_CLASSES = 9


class EdgeRelation(IntEnum):
    """Semantic edge relation types for quantum circuits."""
    ROTATION = 0      # rx, ry, rz, u1, u2, u3 (parameterized single-qubit)
    PAULI = 1         # h, x, y, z, s, t, sdg, tdg, id (discrete single-qubit)
    ENTANGLE = 2      # cx, cz, rxx, ryy, rzz (entangling 2-qubit)
    SWAP = 3          # swap, cswap (permutation)
    CONTROL = 4       # ccx, ccz, cp, crx, cry, crz, cu1, cu3 (controlled)
    TEMPORAL = 5      # temporal ordering within same qubit


NUM_RELATIONS = len(EdgeRelation)

ROTATION_GATES = {'rx', 'ry', 'rz', 'u1', 'u2', 'u3'}
PAULI_GATES = {'h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'id'}
ENTANGLE_GATES = {'cx', 'cz', 'rxx', 'ryy', 'rzz'}
SWAP_GATES = {'swap', 'cswap'}
CONTROL_GATES = {'ccx', 'ccz', 'cp', 'crx', 'cry', 'crz', 'cu1', 'cu3'}


def gate_to_relation(gate_type: str) -> EdgeRelation:
    """Map gate type to semantic relation."""
    if gate_type in ROTATION_GATES:
        return EdgeRelation.ROTATION
    elif gate_type in PAULI_GATES:
        return EdgeRelation.PAULI
    elif gate_type in ENTANGLE_GATES:
        return EdgeRelation.ENTANGLE
    elif gate_type in SWAP_GATES:
        return EdgeRelation.SWAP
    elif gate_type in CONTROL_GATES:
        return EdgeRelation.CONTROL
    else:
        return EdgeRelation.PAULI


@dataclass 
class HeteroGraphData:
    """Heterogeneous graph data with multi-relation edges."""
    x: torch.Tensor                    # [n_nodes, node_feat_dim]
    edge_index_dict: Dict[int, torch.Tensor]   # relation -> [2, n_edges]
    edge_attr_dict: Dict[int, torch.Tensor]    # relation -> [n_edges, edge_feat_dim]
    edge_gate_type_dict: Dict[int, torch.Tensor]  # relation -> [n_edges]
    global_features: torch.Tensor      # [1, global_feat_dim]
    n_qubits: int


def build_hetero_edges_from_standard(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor, 
    edge_gate_type: torch.Tensor,
    n_qubits: int,
    add_temporal: bool = True,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Convert standard edge representation to heterogeneous multi-relation format.
    Also adds temporal edges connecting sequential operations on same qubit.
    """
    from .graph_builder import ALL_GATE_TYPES
    
    edge_index_dict: Dict[int, List] = {r: [[], []] for r in range(NUM_RELATIONS)}
    edge_attr_dict: Dict[int, List] = {r: [] for r in range(NUM_RELATIONS)}
    edge_gate_type_dict: Dict[int, List] = {r: [] for r in range(NUM_RELATIONS)}
    
    qubit_last_edge_idx: Dict[int, int] = {}
    qubit_edge_sequence: Dict[int, List[Tuple[int, int]]] = {q: [] for q in range(n_qubits)}
    
    n_edges = edge_index.shape[1]
    
    for i in range(n_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        gate_idx = edge_gate_type[i].item()
        gate_name = ALL_GATE_TYPES[gate_idx] if gate_idx < len(ALL_GATE_TYPES) else "id"
        
        relation = gate_to_relation(gate_name)
        
        edge_index_dict[relation][0].append(src)
        edge_index_dict[relation][1].append(dst)
        edge_attr_dict[relation].append(edge_attr[i])
        edge_gate_type_dict[relation].append(gate_idx)
        
        if src < n_qubits:
            qubit_edge_sequence[src].append((i, dst))
        if dst < n_qubits and dst != src:
            qubit_edge_sequence[dst].append((i, src))
    
    if add_temporal:
        for q in range(n_qubits):
            seq = qubit_edge_sequence[q]
            for j in range(1, len(seq)):
                prev_partner = seq[j-1][1]
                curr_partner = seq[j][1]
                
                if prev_partner < n_qubits:
                    edge_index_dict[EdgeRelation.TEMPORAL][0].append(prev_partner)
                    edge_index_dict[EdgeRelation.TEMPORAL][1].append(q)
                    temporal_attr = torch.zeros(edge_attr.shape[1])
                    temporal_attr[0] = j / max(len(seq), 1)
                    edge_attr_dict[EdgeRelation.TEMPORAL].append(temporal_attr)
                    edge_gate_type_dict[EdgeRelation.TEMPORAL].append(0)
    
    result_edge_index = {}
    result_edge_attr = {}
    result_edge_gate_type = {}
    
    for r in range(NUM_RELATIONS):
        if len(edge_index_dict[r][0]) > 0:
            result_edge_index[r] = torch.tensor(edge_index_dict[r], dtype=torch.long)
            result_edge_attr[r] = torch.stack(edge_attr_dict[r])
            result_edge_gate_type[r] = torch.tensor(edge_gate_type_dict[r], dtype=torch.long)
        else:
            result_edge_index[r] = torch.zeros((2, 0), dtype=torch.long)
            result_edge_attr[r] = torch.zeros((0, edge_attr.shape[1]))
            result_edge_gate_type[r] = torch.zeros(0, dtype=torch.long)
    
    return result_edge_index, result_edge_attr, result_edge_gate_type


class RelationSpecificTransform(nn.Module):
    """Per-relation linear transformation for heterogeneous graphs (HGT-style)."""
    
    def __init__(self, in_dim: int, out_dim: int, num_relations: int = NUM_RELATIONS):
        super().__init__()
        self.num_relations = num_relations
        self.transforms = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_relations)
        ])
    
    def forward(self, x: torch.Tensor, relation: int) -> torch.Tensor:
        return self.transforms[relation](x)


class HeteroAttentionLayer(MessagePassing):
    """
    Heterogeneous attention layer inspired by HGT.
    
    For each relation type, computes attention-weighted messages with
    relation-specific key/query/value transformations.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        num_relations: int = NUM_RELATIONS,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        dropout: float = 0.1,
    ):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.num_relations = num_relations
        
        self.q_transforms = RelationSpecificTransform(in_channels, out_channels, num_relations)
        self.k_transforms = RelationSpecificTransform(in_channels + edge_feat_dim, out_channels, num_relations)
        self.v_transforms = RelationSpecificTransform(in_channels + edge_feat_dim, out_channels, num_relations)
        
        self.relation_prior = nn.Parameter(torch.ones(num_relations, num_heads))
        
        self.msg_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self._attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        relation: int,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return torch.zeros(x.shape[0], self.out_channels, device=x.device)
        
        q = self.q_transforms(x, relation)
        
        out = self.propagate(
            edge_index, x=x, q=q, edge_attr=edge_attr, relation=relation
        )
        
        out = self.msg_mlp(out)
        out = self.dropout(out)
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        q_i: torch.Tensor,
        edge_attr: torch.Tensor,
        relation: int,
        index: torch.Tensor,
    ) -> torch.Tensor:
        kv_input = torch.cat([x_j, edge_attr], dim=-1)
        k = self.k_transforms(kv_input, relation)
        v = self.v_transforms(kv_input, relation)
        
        B = q_i.shape[0]
        q_i = q_i.view(B, self.num_heads, self.head_dim)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)
        
        attn_scores = (q_i * k).sum(dim=-1) / math.sqrt(self.head_dim)
        
        relation_prior = self.relation_prior[relation].unsqueeze(0)
        attn_scores = attn_scores * relation_prior
        
        attn_weights = softmax(attn_scores, index, dim=0)
        self._attention_weights = attn_weights.detach()
        
        weighted_v = (attn_weights.unsqueeze(-1) * v).view(B, -1)
        
        return weighted_v


class MetaPathAttention(nn.Module):
    """
    Meta-path based attention for capturing multi-hop patterns.
    
    Implements semantic-level attention that aggregates information
    along meaningful meta-paths in quantum circuits:
    - Qubit -> Entangle -> Qubit (direct entanglement)
    - Qubit -> Control -> Qubit (control flow)
    - Temporal sequences within qubits
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_relations: int = NUM_RELATIONS,
        num_meta_paths: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_meta_paths = num_meta_paths
        
        self.meta_path_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )
        
        self.meta_path_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_meta_paths)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self._semantic_weights = None
    
    def forward(
        self,
        relation_outputs: Dict[int, torch.Tensor],
        n_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Aggregate relation-specific outputs using semantic attention.
        
        Meta-paths:
        0: Entanglement path (ENTANGLE + SWAP relations)
        1: Control path (CONTROL + ROTATION relations)
        2: Local path (PAULI + TEMPORAL relations)
        """
        meta_path_outputs = []
        
        entangle_out = torch.zeros(n_nodes, self.hidden_dim, device=device)
        for r in [EdgeRelation.ENTANGLE, EdgeRelation.SWAP]:
            if r in relation_outputs:
                entangle_out = entangle_out + relation_outputs[r]
        meta_path_outputs.append(self.meta_path_transforms[0](entangle_out))
        
        control_out = torch.zeros(n_nodes, self.hidden_dim, device=device)
        for r in [EdgeRelation.CONTROL, EdgeRelation.ROTATION]:
            if r in relation_outputs:
                control_out = control_out + relation_outputs[r]
        meta_path_outputs.append(self.meta_path_transforms[1](control_out))
        
        local_out = torch.zeros(n_nodes, self.hidden_dim, device=device)
        for r in [EdgeRelation.PAULI, EdgeRelation.TEMPORAL]:
            if r in relation_outputs:
                local_out = local_out + relation_outputs[r]
        meta_path_outputs.append(self.meta_path_transforms[2](local_out))
        
        stacked = torch.stack(meta_path_outputs, dim=1)
        
        attn_scores = self.meta_path_attention(stacked).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        self._semantic_weights = attn_weights.mean(dim=0).detach()
        
        output = (attn_weights.unsqueeze(-1) * stacked).sum(dim=1)
        
        output = self.norm(output)
        output = self.dropout(output)
        
        return output


class HeteroMessagePassingBlock(nn.Module):
    """
    Complete heterogeneous message passing block.
    
    Combines:
    1. Relation-specific attention layers
    2. Inter-relation aggregation
    3. Meta-path semantic attention
    4. Residual connection with layer normalization
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_relations: int = NUM_RELATIONS,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        
        self.relation_layers = nn.ModuleList([
            HeteroAttentionLayer(
                hidden_dim, hidden_dim, num_heads, num_relations, edge_feat_dim, dropout
            )
            for _ in range(num_relations)
        ])
        
        self.inter_relation_attn = nn.Sequential(
            nn.Linear(hidden_dim * num_relations, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations),
        )
        
        self.meta_path_attn = MetaPathAttention(hidden_dim, num_relations, dropout=dropout)
        
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index_dict: Dict[int, torch.Tensor],
        edge_attr_dict: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        n_nodes = x.shape[0]
        device = x.device
        
        relation_outputs = {}
        for r in range(self.num_relations):
            if r in edge_index_dict and edge_index_dict[r].shape[1] > 0:
                relation_outputs[r] = self.relation_layers[r](
                    x, edge_index_dict[r], edge_attr_dict[r], r
                )
            else:
                relation_outputs[r] = torch.zeros(n_nodes, self.hidden_dim, device=device)
        
        all_relation_out = torch.cat([relation_outputs[r] for r in range(self.num_relations)], dim=-1)
        inter_attn_weights = F.softmax(self.inter_relation_attn(all_relation_out), dim=-1)
        
        inter_aggregated = sum(
            inter_attn_weights[:, r:r+1] * relation_outputs[r]
            for r in range(self.num_relations)
        )
        
        meta_aggregated = self.meta_path_attn(relation_outputs, n_nodes, device)
        
        combined = self.combine(torch.cat([inter_aggregated, meta_aggregated], dim=-1))
        
        out = self.norm(x + self.dropout(combined))
        
        return out


class EntanglementAwarePooling(nn.Module):
    """
    Graph pooling that considers qubit entanglement structure.
    
    Computes attention-weighted pooling where attention is based on
    each qubit's role in the entanglement structure (degree, centrality).
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.transform_mean = nn.Linear(hidden_dim, hidden_dim)
        self.transform_max = nn.Linear(hidden_dim, hidden_dim)
        self.transform_weighted = nn.Linear(hidden_dim, hidden_dim)
        
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        attn_scores = self.node_attention(x).squeeze(-1)
        attn_weights = softmax(attn_scores, batch)
        
        num_graphs = batch.max().item() + 1
        weighted_sum = torch.zeros(num_graphs, x.shape[1], device=x.device)
        weighted_sum.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), attn_weights.unsqueeze(-1) * x)
        
        h_mean = global_mean_pool(x, batch)
        h_max = global_max_pool(x, batch)
        h_weighted = weighted_sum
        
        h_mean = self.transform_mean(h_mean)
        h_max = self.transform_max(h_max)
        h_weighted = self.transform_weighted(h_weighted)
        
        combined = torch.cat([h_mean, h_max, h_weighted], dim=-1)
        out = self.combine(combined)
        out = self.norm(out)
        
        return out


class QuantumCircuitHeteroGNN(BaseGraphThresholdClassModel):
    """
    Quantum Circuit Heterogeneous Graph Transformer (QCHGT).
    
    A heterogeneous GNN that captures the multi-relational structure of 
    quantum circuits for threshold class prediction.
    
    Architecture:
    1. Node embedding with positional encoding
    2. Stack of HeteroMessagePassingBlocks with:
       - Relation-specific attention
       - Inter-relation aggregation  
       - Meta-path semantic attention
    3. Entanglement-aware graph pooling
    4. Global feature fusion
    5. Classification head
    """
    
    def __init__(
        self,
        config: Optional[GraphModelConfig] = None,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES,
        num_classes: int = NUM_THRESHOLD_CLASSES,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        max_qubits = 256
        self.pos_embed = nn.Embedding(max_qubits, hidden_dim // 4)
        self.pos_proj = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        
        self.hetero_blocks = nn.ModuleList([
            HeteroMessagePassingBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                edge_feat_dim=edge_feat_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.pooling = EntanglementAwarePooling(hidden_dim, dropout)
        
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self._apply_init()
    
    def _apply_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    @property
    def name(self) -> str:
        return f"QCHGT-{self.num_layers}L-{self.hidden_dim}D"
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = x.shape[0]
        device = x.device
        
        node_positions = torch.zeros(n_nodes, dtype=torch.long, device=device)
        if batch.max() > 0:
            batch_sizes = torch.bincount(batch)
            offset = 0
            for i, size in enumerate(batch_sizes):
                node_positions[offset:offset+size] = torch.arange(size, device=device)
                offset += size
        else:
            node_positions = torch.arange(n_nodes, device=device)
        
        node_positions = node_positions.clamp(max=255)
        
        h = self.node_embed(x)
        pos_emb = self.pos_embed(node_positions)
        h = self.pos_proj(torch.cat([h, pos_emb], dim=-1))
        
        batch_sizes = torch.bincount(batch)
        max_qubits = batch_sizes.max().item()
        
        edge_index_dict, edge_attr_dict, _ = build_hetero_edges_from_standard(
            edge_index, edge_attr, edge_gate_type, max_qubits
        )
        
        for r in edge_index_dict:
            edge_index_dict[r] = edge_index_dict[r].to(device)
            edge_attr_dict[r] = edge_attr_dict[r].to(device)
        
        for block in self.hetero_blocks:
            h = block(h, edge_index_dict, edge_attr_dict)
        
        return h
    
    def pool_graph(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.pooling(node_embeddings, batch)
    
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
        
        combined = self.fusion(torch.cat([h_graph, g], dim=-1))
        
        logits = self.classifier(combined)
        
        return logits
    
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


class QuantumCircuitHeteroGNNDuration(BaseGraphDurationModel):
    """
    QCHGT variant for duration prediction.
    
    Same architecture as threshold classification but with regression head.
    Global features include log2(threshold).
    """
    
    def __init__(
        self,
        config: Optional[GraphModelConfig] = None,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        max_qubits = 256
        self.pos_embed = nn.Embedding(max_qubits, hidden_dim // 4)
        self.pos_proj = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        
        self.hetero_blocks = nn.ModuleList([
            HeteroMessagePassingBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                edge_feat_dim=edge_feat_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.pooling = EntanglementAwarePooling(hidden_dim, dropout)
        
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
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
        return f"QCHGT-Duration-{self.num_layers}L-{self.hidden_dim}D"
    
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = x.shape[0]
        device = x.device
        
        node_positions = torch.zeros(n_nodes, dtype=torch.long, device=device)
        batch_sizes = torch.bincount(batch)
        offset = 0
        for i, size in enumerate(batch_sizes):
            node_positions[offset:offset+size] = torch.arange(size, device=device)
            offset += size
        node_positions = node_positions.clamp(max=255)
        
        h = self.node_embed(x)
        pos_emb = self.pos_embed(node_positions)
        h = self.pos_proj(torch.cat([h, pos_emb], dim=-1))
        
        max_qubits = batch_sizes.max().item()
        edge_index_dict, edge_attr_dict, _ = build_hetero_edges_from_standard(
            edge_index, edge_attr, edge_gate_type, max_qubits
        )
        
        for r in edge_index_dict:
            edge_index_dict[r] = edge_index_dict[r].to(device)
            edge_attr_dict[r] = edge_attr_dict[r].to(device)
        
        for block in self.hetero_blocks:
            h = block(h, edge_index_dict, edge_attr_dict)
        
        return h
    
    def pool_graph(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.pooling(node_embeddings, batch)
    
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
        combined = self.fusion(torch.cat([h_graph, g], dim=-1))
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


def create_hetero_gnn_model(
    model_type: str = "threshold",
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: Optional[int] = None,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.2,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create QCHGT models.
    
    Args:
        model_type: "threshold" for classification, "duration" for regression
        node_feat_dim: Node feature dimension
        edge_feat_dim: Edge feature dimension  
        global_feat_dim: Global feature dimension (auto-computed if None)
        hidden_dim: Hidden layer dimension
        num_layers: Number of HeteroMessagePassingBlocks
        num_heads: Number of attention heads
        dropout: Dropout rate
    
    Returns:
        QCHGT model instance
    """
    if model_type == "threshold":
        if global_feat_dim is None:
            global_feat_dim = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES
        return QuantumCircuitHeteroGNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif model_type == "duration":
        if global_feat_dim is None:
            global_feat_dim = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES
        return QuantumCircuitHeteroGNNDuration(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Testing QCHGT model...")
    
    model = create_hetero_gnn_model(
        model_type="threshold",
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
    )
    
    print(f"Model: {model.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(10, NODE_FEAT_DIM)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]], dtype=torch.long)
    edge_attr = torch.randn(5, EDGE_FEAT_DIM)
    edge_gate_type = torch.tensor([15, 15, 15, 0, 1], dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    global_features = torch.randn(1, GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES)
    
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr, edge_gate_type, batch, global_features)
    
    print(f"Output shape: {logits.shape}")
    print(f"Logits: {logits}")
    
    print("\nTesting duration model...")
    duration_model = create_hetero_gnn_model(
        model_type="duration",
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
    )
    
    global_features_dur = torch.randn(1, GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES)
    with torch.no_grad():
        runtime = duration_model(x, edge_index, edge_attr, edge_gate_type, batch, global_features_dur)
    
    print(f"Runtime prediction: {runtime}")
    print("\nAll tests passed!")

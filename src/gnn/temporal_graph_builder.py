"""
Enhanced graph builder with rich temporal and structural features for Temporal GNN.

This module extends the base graph builder with:
1. Layer-wise temporal encoding (gates grouped by parallel execution)
2. Entanglement trajectory features (how entanglement spreads over time)
3. Causal dependency edges (which gates must execute before others)
4. Rich gate sequence features (patterns, repetitions, algorithmic signatures)
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import numpy as np

from .graph_builder import (
    GATE_1Q, GATE_2Q, GATE_3Q, ALL_GATE_TYPES, GATE_TO_IDX, NUM_GATE_TYPES,
    ParsedGate, parse_qasm,
    NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE,
)


@dataclass
class TemporalGate:
    """Gate with enhanced temporal information."""
    gate_type: str
    qubits: List[int]
    params: List[float]
    position: int
    layer: int = 0
    cumulative_2q_before: int = 0
    cumulative_entanglement: float = 0.0
    is_long_range: bool = False


def compute_circuit_layers(gates: List[ParsedGate], n_qubits: int) -> List[int]:
    """
    Assign gates to execution layers based on qubit dependencies.
    
    Gates that operate on different qubits can execute in parallel (same layer).
    This is critical for understanding circuit depth and parallelism.
    """
    if not gates:
        return []
    
    qubit_last_layer = [-1] * n_qubits
    gate_layers = []
    
    for gate in gates:
        max_dep_layer = -1
        for q in gate.qubits:
            if q < n_qubits:
                max_dep_layer = max(max_dep_layer, qubit_last_layer[q])
        
        gate_layer = max_dep_layer + 1
        gate_layers.append(gate_layer)
        
        for q in gate.qubits:
            if q < n_qubits:
                qubit_last_layer[q] = gate_layer
    
    return gate_layers


def compute_entanglement_trajectory(
    gates: List[ParsedGate],
    n_qubits: int,
) -> Tuple[List[float], List[Set[int]]]:
    """
    Track how entanglement spreads through the circuit over time.
    
    Uses a simplified model where qubits become "entangled" when connected
    by 2-qubit gates. Returns cumulative entanglement score and connected
    components at each gate.
    """
    if not gates:
        return [], []
    
    # Union-Find for tracking connected components
    parent = list(range(n_qubits))
    
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x: int, y: int) -> bool:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    cumulative_entanglement = []
    connected_components = []
    entanglement_score = 0.0
    
    for gate in gates:
        if gate.gate_type in GATE_2Q or gate.gate_type in GATE_3Q:
            qubits = [q for q in gate.qubits if q < n_qubits]
            
            for i in range(len(qubits)):
                for j in range(i + 1, len(qubits)):
                    if union(qubits[i], qubits[j]):
                        span = abs(qubits[i] - qubits[j])
                        entanglement_score += 1.0 + 0.1 * span / n_qubits
        
        cumulative_entanglement.append(entanglement_score)
        
        components: Set[int] = set()
        for q in range(n_qubits):
            components.add(find(q))
        connected_components.append(components)
    
    max_ent = max(cumulative_entanglement) if cumulative_entanglement else 1.0
    cumulative_entanglement = [e / max(max_ent, 1.0) for e in cumulative_entanglement]
    
    return cumulative_entanglement, connected_components


def build_temporal_graph(
    qasm_text: str,
    backend: str = "CPU",
    precision: str = "single",
    family: Optional[str] = None,
    family_to_idx: Optional[Dict[str, int]] = None,
    num_families: int = 20,
    log2_threshold: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build enhanced temporal graph representation.
    
    Extends the base graph with:
    - Layer-wise temporal encoding
    - Entanglement trajectory features
    - Long-range gate indicators
    - Richer edge features for temporal modeling
    """
    n_qubits, gates = parse_qasm(qasm_text)
    
    if n_qubits == 0:
        n_qubits = 1
    
    gate_layers = compute_circuit_layers(gates, n_qubits)
    entanglement_traj, _ = compute_entanglement_trajectory(gates, n_qubits)
    max_layer = max(gate_layers) if gate_layers else 1
    
    node_1q_counts = torch.zeros(n_qubits, len(GATE_1Q))
    node_2q_counts = torch.zeros(n_qubits, 1)
    node_first_2q_pos = torch.ones(n_qubits, 1)
    node_last_2q_pos = torch.zeros(n_qubits, 1)
    node_unique_neighbors = [set() for _ in range(n_qubits)]
    node_total_span = torch.zeros(n_qubits, 1)
    node_layer_range = torch.zeros(n_qubits, 2)  # [first_layer, last_layer]
    node_layer_range[:, 0] = float('inf')
    node_entanglement_exposure = torch.zeros(n_qubits, 1)
    
    edge_src = []
    edge_dst = []
    edge_gate_type = []
    edge_position = []
    edge_params = []
    edge_qubit_dist = []
    edge_cumulative_idx = []
    edge_layer = []
    edge_entanglement = []
    edge_is_longrange = []
    
    total_gates = len(gates) if gates else 1
    cumulative_2q = 0
    longrange_threshold = n_qubits // 4
    
    for i, gate in enumerate(gates):
        gate_idx = GATE_TO_IDX.get(gate.gate_type, 0)
        norm_pos = gate.position / total_gates
        layer_norm = gate_layers[i] / max(max_layer, 1) if gate_layers else 0
        ent_score = entanglement_traj[i] if i < len(entanglement_traj) else 0
        
        if gate.gate_type in GATE_1Q:
            if len(gate.qubits) >= 1 and gate.qubits[0] < n_qubits:
                q = gate.qubits[0]
                type_idx = GATE_1Q.index(gate.gate_type)
                node_1q_counts[q, type_idx] += 1
                node_layer_range[q, 0] = min(node_layer_range[q, 0], gate_layers[i])
                node_layer_range[q, 1] = max(node_layer_range[q, 1], gate_layers[i])
                
                edge_src.append(q)
                edge_dst.append(q)
                edge_gate_type.append(gate_idx)
                edge_position.append(norm_pos)
                edge_params.append(gate.params[0] if gate.params else 0.0)
                edge_qubit_dist.append(0.0)
                edge_cumulative_idx.append(cumulative_2q / max(total_gates, 1))
                edge_layer.append(layer_norm)
                edge_entanglement.append(ent_score)
                edge_is_longrange.append(0.0)
        
        elif gate.gate_type in GATE_2Q:
            if len(gate.qubits) >= 2:
                q_ctrl, q_targ = gate.qubits[0], gate.qubits[1]
                if q_ctrl < n_qubits and q_targ < n_qubits:
                    span = abs(q_targ - q_ctrl)
                    is_longrange = span >= longrange_threshold
                    
                    node_2q_counts[q_ctrl] += 1
                    node_2q_counts[q_targ] += 1
                    
                    node_first_2q_pos[q_ctrl] = min(node_first_2q_pos[q_ctrl].item(), norm_pos)
                    node_first_2q_pos[q_targ] = min(node_first_2q_pos[q_targ].item(), norm_pos)
                    node_last_2q_pos[q_ctrl] = max(node_last_2q_pos[q_ctrl].item(), norm_pos)
                    node_last_2q_pos[q_targ] = max(node_last_2q_pos[q_targ].item(), norm_pos)
                    
                    node_unique_neighbors[q_ctrl].add(q_targ)
                    node_unique_neighbors[q_targ].add(q_ctrl)
                    
                    node_total_span[q_ctrl] += span
                    node_total_span[q_targ] += span
                    
                    node_layer_range[q_ctrl, 0] = min(node_layer_range[q_ctrl, 0], gate_layers[i])
                    node_layer_range[q_ctrl, 1] = max(node_layer_range[q_ctrl, 1], gate_layers[i])
                    node_layer_range[q_targ, 0] = min(node_layer_range[q_targ, 0], gate_layers[i])
                    node_layer_range[q_targ, 1] = max(node_layer_range[q_targ, 1], gate_layers[i])
                    
                    node_entanglement_exposure[q_ctrl] += ent_score
                    node_entanglement_exposure[q_targ] += ent_score
                    
                    cumulative_2q += 1
                    
                    edge_src.append(q_ctrl)
                    edge_dst.append(q_targ)
                    edge_gate_type.append(gate_idx)
                    edge_position.append(norm_pos)
                    edge_params.append(gate.params[0] if gate.params else 0.0)
                    edge_qubit_dist.append(span / max(n_qubits - 1, 1))
                    edge_cumulative_idx.append(cumulative_2q / max(total_gates, 1))
                    edge_layer.append(layer_norm)
                    edge_entanglement.append(ent_score)
                    edge_is_longrange.append(1.0 if is_longrange else 0.0)
        
        elif gate.gate_type in GATE_3Q:
            if len(gate.qubits) >= 3:
                q1, q2, q_targ = gate.qubits[0], gate.qubits[1], gate.qubits[2]
                for q_ctrl in [q1, q2]:
                    if q_ctrl < n_qubits and q_targ < n_qubits:
                        span = abs(q_targ - q_ctrl)
                        is_longrange = span >= longrange_threshold
                        
                        node_2q_counts[q_ctrl] += 1
                        node_2q_counts[q_targ] += 1
                        
                        node_first_2q_pos[q_ctrl] = min(node_first_2q_pos[q_ctrl].item(), norm_pos)
                        node_first_2q_pos[q_targ] = min(node_first_2q_pos[q_targ].item(), norm_pos)
                        node_last_2q_pos[q_ctrl] = max(node_last_2q_pos[q_ctrl].item(), norm_pos)
                        node_last_2q_pos[q_targ] = max(node_last_2q_pos[q_targ].item(), norm_pos)
                        
                        node_unique_neighbors[q_ctrl].add(q_targ)
                        node_unique_neighbors[q_targ].add(q_ctrl)
                        
                        node_total_span[q_ctrl] += span
                        node_total_span[q_targ] += span
                        
                        node_layer_range[q_ctrl, 0] = min(node_layer_range[q_ctrl, 0], gate_layers[i])
                        node_layer_range[q_ctrl, 1] = max(node_layer_range[q_ctrl, 1], gate_layers[i])
                        node_layer_range[q_targ, 0] = min(node_layer_range[q_targ, 0], gate_layers[i])
                        node_layer_range[q_targ, 1] = max(node_layer_range[q_targ, 1], gate_layers[i])
                        
                        node_entanglement_exposure[q_ctrl] += ent_score
                        node_entanglement_exposure[q_targ] += ent_score
                        
                        cumulative_2q += 1
                        
                        edge_src.append(q_ctrl)
                        edge_dst.append(q_targ)
                        edge_gate_type.append(gate_idx)
                        edge_position.append(norm_pos)
                        edge_params.append(gate.params[0] if gate.params else 0.0)
                        edge_qubit_dist.append(span / max(n_qubits - 1, 1))
                        edge_cumulative_idx.append(cumulative_2q / max(total_gates, 1))
                        edge_layer.append(layer_norm)
                        edge_entanglement.append(ent_score)
                        edge_is_longrange.append(1.0 if is_longrange else 0.0)
    
    node_positions = torch.arange(n_qubits, dtype=torch.float32) / max(n_qubits - 1, 1)
    node_positions = node_positions.unsqueeze(1)
    
    node_1q_log = torch.log1p(node_1q_counts)
    node_2q_log = torch.log1p(node_2q_counts)
    
    node_degree = torch.tensor(
        [len(neighbors) for neighbors in node_unique_neighbors],
        dtype=torch.float32
    ).unsqueeze(1) / max(n_qubits - 1, 1)
    
    node_avg_span = node_total_span / (node_2q_counts + 1)
    node_activity_window = node_last_2q_pos - node_first_2q_pos
    
    node_layer_range[:, 0] = torch.where(
        node_layer_range[:, 0] == float('inf'),
        torch.zeros_like(node_layer_range[:, 0]),
        node_layer_range[:, 0] / max(max_layer, 1)
    )
    node_layer_range[:, 1] = node_layer_range[:, 1] / max(max_layer, 1)
    node_layer_span = (node_layer_range[:, 1] - node_layer_range[:, 0]).unsqueeze(1)
    
    node_entanglement_exposure = node_entanglement_exposure / (node_2q_counts + 1)
    
    x = torch.cat([
        node_1q_log,              # 15
        node_2q_log,              # 1
        node_positions,           # 1
        node_first_2q_pos,        # 1
        node_last_2q_pos,         # 1
        node_activity_window,     # 1
        node_degree,              # 1
        node_avg_span,            # 1
        node_layer_span,          # 1 (new)
        node_entanglement_exposure,  # 1 (new)
    ], dim=1)  # Total: 24
    
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_gate_type_tensor = torch.tensor(edge_gate_type, dtype=torch.long)
        
        edge_attr = torch.stack([
            torch.tensor(edge_position, dtype=torch.float32),
            torch.tensor(edge_params, dtype=torch.float32),
            torch.tensor(edge_qubit_dist, dtype=torch.float32),
            torch.tensor(edge_cumulative_idx, dtype=torch.float32),
            torch.tensor(edge_layer, dtype=torch.float32),
            torch.tensor(edge_entanglement, dtype=torch.float32),
            torch.tensor(edge_is_longrange, dtype=torch.float32),
        ], dim=1)
    else:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_gate_type_tensor = torch.tensor([0], dtype=torch.long)
        edge_attr = torch.zeros(1, 7)
    
    backend_idx = 1.0 if backend == "GPU" else 0.0
    precision_idx = 1.0 if precision == "double" else 0.0
    
    n_2q_gates = cumulative_2q
    gate_density = n_2q_gates / max(n_qubits, 1)
    
    n_longrange = sum(1 for lr in edge_is_longrange if lr > 0)
    longrange_ratio = n_longrange / max(len(gates), 1)
    
    circuit_depth = max_layer + 1
    depth_normalized = circuit_depth / max(len(gates), 1)
    
    final_entanglement = entanglement_traj[-1] if entanglement_traj else 0.0
    
    global_feats = [
        float(n_qubits) / 130.0,
        float(len(gates)) / 1000.0,
        float(n_2q_gates) / 500.0,
        gate_density / 10.0,
        backend_idx,
        precision_idx,
        longrange_ratio,
        depth_normalized,
        final_entanglement,
    ]
    
    if log2_threshold is not None:
        global_feats.append(log2_threshold / 8.0)
    
    if family and family_to_idx:
        family_onehot = [0.0] * num_families
        if family in family_to_idx:
            family_onehot[family_to_idx[family]] = 1.0
        global_feats.extend(family_onehot)
    
    global_features = torch.tensor(global_feats, dtype=torch.float32).unsqueeze(0)
    
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_gate_type": edge_gate_type_tensor,
        "global_features": global_features,
        "n_qubits": n_qubits,
        "n_edges": edge_index.shape[1],
        "n_layers": circuit_depth,
        "n_longrange": n_longrange,
    }


def build_temporal_graph_from_file(
    qasm_path: Path,
    backend: str = "CPU",
    precision: str = "single",
    family: Optional[str] = None,
    family_to_idx: Optional[Dict[str, int]] = None,
    num_families: int = 20,
    log2_threshold: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Build temporal graph from a QASM file path."""
    text = qasm_path.read_text(encoding="utf-8")
    return build_temporal_graph(
        text, backend, precision, family, family_to_idx, num_families,
        log2_threshold=log2_threshold,
    )


TEMPORAL_NODE_FEAT_DIM = 24
TEMPORAL_EDGE_FEAT_DIM = 7
TEMPORAL_GLOBAL_FEAT_DIM_BASE = 9


if __name__ == "__main__":
    test_qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg c[4];
    h q[0];
    cx q[0],q[1];
    cx q[1],q[2];
    cx q[0],q[3];
    rz(0.5) q[3];
    cx q[2],q[3];
    h q[0];
    measure q[0] -> c[0];
    """
    
    result = build_temporal_graph(test_qasm, "CPU", "single")
    print(f"Nodes: {result['x'].shape}")
    print(f"Edges: {result['edge_index'].shape}")
    print(f"Edge attr: {result['edge_attr'].shape}")
    print(f"Global features: {result['global_features'].shape}")
    print(f"Circuit layers: {result['n_layers']}")
    print(f"Long-range gates: {result['n_longrange']}")

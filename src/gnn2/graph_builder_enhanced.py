"""
Enhanced Graph Builder for Quantum Circuits.

Builds richer graph representations with:
1. Advanced node features (spectral, topological)
2. Enhanced edge features (gate properties, temporal encoding)
3. Circuit-level structural features
4. Learnable feature engineering
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import math

import torch
import numpy as np


# Gate definitions (extended)
GATE_1Q = ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'id', 'sx', 'sxdg']
GATE_2Q = ['cx', 'cz', 'swap', 'cp', 'crx', 'cry', 'crz', 'cu1', 'cu3', 'rxx', 'ryy', 'rzz', 'ecr', 'iswap']
GATE_3Q = ['ccx', 'cswap', 'ccz', 'c3x', 'rccx']

ALL_GATE_TYPES = GATE_1Q + GATE_2Q + GATE_3Q
GATE_TO_IDX = {g: i for i, g in enumerate(ALL_GATE_TYPES)}
NUM_GATE_TYPES = len(ALL_GATE_TYPES)

# Gate properties
GATE_PROPERTIES = {
    # 1Q gates
    'h': {'clifford': True, 'rotation': False, 'diagonal': False},
    'x': {'clifford': True, 'rotation': False, 'diagonal': False},
    'y': {'clifford': True, 'rotation': False, 'diagonal': False},
    'z': {'clifford': True, 'rotation': False, 'diagonal': True},
    's': {'clifford': True, 'rotation': False, 'diagonal': True},
    'sdg': {'clifford': True, 'rotation': False, 'diagonal': True},
    't': {'clifford': False, 'rotation': False, 'diagonal': True},
    'tdg': {'clifford': False, 'rotation': False, 'diagonal': True},
    'rx': {'clifford': False, 'rotation': True, 'diagonal': False},
    'ry': {'clifford': False, 'rotation': True, 'diagonal': False},
    'rz': {'clifford': False, 'rotation': True, 'diagonal': True},
    'u1': {'clifford': False, 'rotation': True, 'diagonal': True},
    'u2': {'clifford': False, 'rotation': True, 'diagonal': False},
    'u3': {'clifford': False, 'rotation': True, 'diagonal': False},
    'id': {'clifford': True, 'rotation': False, 'diagonal': True},
    'sx': {'clifford': True, 'rotation': False, 'diagonal': False},
    'sxdg': {'clifford': True, 'rotation': False, 'diagonal': False},
    # 2Q gates
    'cx': {'clifford': True, 'rotation': False, 'diagonal': False, 'symmetric': False},
    'cz': {'clifford': True, 'rotation': False, 'diagonal': True, 'symmetric': True},
    'swap': {'clifford': True, 'rotation': False, 'diagonal': False, 'symmetric': True},
    'cp': {'clifford': False, 'rotation': True, 'diagonal': True, 'symmetric': True},
    'crx': {'clifford': False, 'rotation': True, 'diagonal': False, 'symmetric': False},
    'cry': {'clifford': False, 'rotation': True, 'diagonal': False, 'symmetric': False},
    'crz': {'clifford': False, 'rotation': True, 'diagonal': True, 'symmetric': False},
    'cu1': {'clifford': False, 'rotation': True, 'diagonal': True, 'symmetric': False},
    'cu3': {'clifford': False, 'rotation': True, 'diagonal': False, 'symmetric': False},
    'rxx': {'clifford': False, 'rotation': True, 'diagonal': False, 'symmetric': True},
    'ryy': {'clifford': False, 'rotation': True, 'diagonal': False, 'symmetric': True},
    'rzz': {'clifford': False, 'rotation': True, 'diagonal': True, 'symmetric': True},
    'ecr': {'clifford': True, 'rotation': False, 'diagonal': False, 'symmetric': False},
    'iswap': {'clifford': False, 'rotation': False, 'diagonal': False, 'symmetric': True},
    # 3Q gates
    'ccx': {'clifford': False, 'rotation': False, 'diagonal': False},
    'cswap': {'clifford': False, 'rotation': False, 'diagonal': False},
    'ccz': {'clifford': False, 'rotation': False, 'diagonal': True},
    'c3x': {'clifford': False, 'rotation': False, 'diagonal': False},
    'rccx': {'clifford': False, 'rotation': False, 'diagonal': False},
}


@dataclass
class ParsedGate:
    gate_type: str
    qubits: List[int]
    params: List[float]
    position: int
    depth: int  # Circuit depth at this point


def parse_qasm_enhanced(text: str) -> Tuple[int, List[ParsedGate], Dict[str, Any]]:
    """
    Enhanced QASM parser with circuit analysis.
    
    Returns:
        n_qubits: Number of qubits
        gates: List of parsed gates
        circuit_stats: Dictionary of circuit statistics
    """
    lines = text.splitlines()
    
    n_qubits = 0
    qreg_match = re.search(r"qreg\s+\w+\[(\d+)\]", text)
    if qreg_match:
        n_qubits = int(qreg_match.group(1))
    
    gate_pattern = re.compile(
        r"\b(" + "|".join(ALL_GATE_TYPES) + r")\b"
        r"(?:\s*\(([^)]*)\))?"
        r"\s+(.+?);"
    )
    qubit_ref = re.compile(r"\w+\[(\d+)\]")
    
    gates = []
    position = 0
    
    # Track qubit depths for computing circuit depth
    qubit_depths = defaultdict(int)
    
    # Track circuit statistics
    stats = {
        'n_1q_gates': 0,
        'n_2q_gates': 0,
        'n_3q_gates': 0,
        'n_clifford': 0,
        'n_rotation': 0,
        'n_diagonal': 0,
        'gate_type_counts': defaultdict(int),
        'depth': 0,
        'critical_path': [],
    }
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM") or \
           line.startswith("include") or line.startswith("qreg") or \
           line.startswith("creg") or line.startswith("barrier") or \
           line.startswith("measure"):
            continue
        
        match = gate_pattern.search(line)
        if match:
            gate_type = match.group(1)
            param_str = match.group(2) or ""
            qubit_str = match.group(3)
            
            # Parse parameters
            params = []
            if param_str:
                for p in param_str.split(","):
                    p = p.strip()
                    try:
                        if "pi" in p:
                            p = p.replace("pi", str(np.pi))
                            params.append(float(eval(p)))
                        else:
                            params.append(float(p))
                    except:
                        params.append(0.0)
            
            qubits = [int(m.group(1)) for m in qubit_ref.finditer(qubit_str)]
            
            if qubits:
                # Compute depth at this gate
                max_depth = max(qubit_depths[q] for q in qubits) if qubits else 0
                current_depth = max_depth + 1
                
                # Update qubit depths
                for q in qubits:
                    qubit_depths[q] = current_depth
                
                gates.append(ParsedGate(
                    gate_type=gate_type,
                    qubits=qubits,
                    params=params,
                    position=position,
                    depth=current_depth,
                ))
                position += 1
                
                # Update statistics
                props = GATE_PROPERTIES.get(gate_type, {})
                
                if gate_type in GATE_1Q:
                    stats['n_1q_gates'] += 1
                elif gate_type in GATE_2Q:
                    stats['n_2q_gates'] += 1
                elif gate_type in GATE_3Q:
                    stats['n_3q_gates'] += 1
                
                if props.get('clifford', False):
                    stats['n_clifford'] += 1
                if props.get('rotation', False):
                    stats['n_rotation'] += 1
                if props.get('diagonal', False):
                    stats['n_diagonal'] += 1
                
                stats['gate_type_counts'][gate_type] += 1
    
    stats['depth'] = max(qubit_depths.values()) if qubit_depths else 0
    
    return n_qubits, gates, stats


def compute_spectral_features(edge_index: torch.Tensor, num_nodes: int, k: int = 4) -> torch.Tensor:
    """
    Compute spectral graph features based on Laplacian eigenvectors.
    
    Returns node features based on the first k non-trivial eigenvectors.
    """
    if num_nodes < 2:
        return torch.zeros(num_nodes, k)
    
    device = edge_index.device
    
    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    if edge_index.numel() > 0:
        row, col = edge_index
        adj[row, col] = 1
        adj[col, row] = 1  # Make symmetric
    
    # Compute degree matrix
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.where(deg > 0, 1.0 / torch.sqrt(deg), torch.zeros_like(deg))
    
    # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    L_norm = torch.eye(num_nodes, device=device) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    # Compute eigenvectors
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
        # Take first k non-trivial eigenvectors (skip the first constant one)
        features = eigenvectors[:, 1:k+1]
        
        # Pad if not enough eigenvectors
        if features.shape[1] < k:
            features = torch.cat([
                features, 
                torch.zeros(num_nodes, k - features.shape[1], device=device)
            ], dim=1)
    except:
        features = torch.zeros(num_nodes, k, device=device)
    
    return features


def compute_connectivity_features(
    n_qubits: int,
    gates: List[ParsedGate],
) -> torch.Tensor:
    """
    Compute qubit connectivity features.
    
    Returns features describing each qubit's role in the circuit topology.
    """
    features = torch.zeros(n_qubits, 8)
    
    # Track interactions
    interaction_counts = defaultdict(int)
    interaction_times = defaultdict(list)
    neighbors = defaultdict(set)
    
    for gate in gates:
        if gate.gate_type in GATE_2Q and len(gate.qubits) >= 2:
            q1, q2 = gate.qubits[0], gate.qubits[1]
            if q1 < n_qubits and q2 < n_qubits:
                interaction_counts[(q1, q2)] += 1
                interaction_counts[(q2, q1)] += 1
                interaction_times[q1].append(gate.position)
                interaction_times[q2].append(gate.position)
                neighbors[q1].add(q2)
                neighbors[q2].add(q1)
    
    total_gates = len(gates) if gates else 1
    
    for q in range(n_qubits):
        # Degree (number of unique neighbors)
        features[q, 0] = len(neighbors[q]) / max(n_qubits - 1, 1)
        
        # Total interaction count
        total_interactions = sum(interaction_counts[(q, other)] for other in neighbors[q])
        features[q, 1] = math.log1p(total_interactions)
        
        # Interaction times statistics
        times = interaction_times[q]
        if times:
            features[q, 2] = min(times) / total_gates  # First interaction
            features[q, 3] = max(times) / total_gates  # Last interaction
            features[q, 4] = np.mean(times) / total_gates  # Mean interaction time
            features[q, 5] = np.std(times) / total_gates if len(times) > 1 else 0  # Std
        
        # Average neighbor distance
        if neighbors[q]:
            avg_dist = np.mean([abs(other - q) for other in neighbors[q]])
            features[q, 6] = avg_dist / max(n_qubits - 1, 1)
        
        # Clustering coefficient (fraction of neighbors connected to each other)
        if len(neighbors[q]) >= 2:
            neighbor_list = list(neighbors[q])
            connected = 0
            total = 0
            for i, n1 in enumerate(neighbor_list):
                for n2 in neighbor_list[i+1:]:
                    total += 1
                    if n2 in neighbors[n1]:
                        connected += 1
            features[q, 7] = connected / total if total > 0 else 0
    
    return features


def sinusoidal_encoding(position: float, dim: int = 16) -> torch.Tensor:
    """Generate sinusoidal positional encoding."""
    encoding = torch.zeros(dim)
    for i in range(0, dim, 2):
        freq = 1.0 / (10000 ** (i / dim))
        encoding[i] = math.sin(position * freq)
        if i + 1 < dim:
            encoding[i + 1] = math.cos(position * freq)
    return encoding


def build_graph_enhanced(
    qasm_text: str,
    backend: str = "CPU",
    precision: str = "single",
    family: Optional[str] = None,
    family_to_idx: Optional[Dict[str, int]] = None,
    num_families: int = 20,
    log2_threshold: Optional[float] = None,
    include_spectral: bool = True,
    include_sinusoidal_pe: bool = True,
    sinusoidal_dim: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Build an enhanced graph representation from QASM.
    
    Features enhanced node and edge representations compared to the basic version.
    """
    n_qubits, gates, stats = parse_qasm_enhanced(qasm_text)
    
    if n_qubits == 0:
        n_qubits = 1
    
    # =========================================================================
    # NODE FEATURES
    # =========================================================================
    
    # Basic features: per-gate-type counts
    node_1q_counts = torch.zeros(n_qubits, len(GATE_1Q))
    node_2q_counts = torch.zeros(n_qubits, 1)
    
    # Temporal features
    node_first_gate_pos = torch.ones(n_qubits, 1)
    node_last_gate_pos = torch.zeros(n_qubits, 1)
    node_first_2q_pos = torch.ones(n_qubits, 1)
    node_last_2q_pos = torch.zeros(n_qubits, 1)
    
    # Depth features
    node_max_depth = torch.zeros(n_qubits, 1)
    node_avg_depth = torch.zeros(n_qubits, 1)
    node_depth_counts = torch.zeros(n_qubits, 1)
    
    # Gate property features
    node_clifford_count = torch.zeros(n_qubits, 1)
    node_rotation_count = torch.zeros(n_qubits, 1)
    node_diagonal_count = torch.zeros(n_qubits, 1)
    
    # Process gates
    total_gates = len(gates) if gates else 1
    
    for gate in gates:
        props = GATE_PROPERTIES.get(gate.gate_type, {})
        norm_pos = gate.position / total_gates
        
        for q in gate.qubits:
            if q >= n_qubits:
                continue
            
            # Update temporal features
            node_first_gate_pos[q] = min(node_first_gate_pos[q].item(), norm_pos)
            node_last_gate_pos[q] = max(node_last_gate_pos[q].item(), norm_pos)
            
            # Update depth features
            node_max_depth[q] = max(node_max_depth[q].item(), gate.depth)
            node_avg_depth[q] += gate.depth
            node_depth_counts[q] += 1
            
            # Update property counts
            if props.get('clifford', False):
                node_clifford_count[q] += 1
            if props.get('rotation', False):
                node_rotation_count[q] += 1
            if props.get('diagonal', False):
                node_diagonal_count[q] += 1
        
        if gate.gate_type in GATE_1Q and len(gate.qubits) >= 1:
            q = gate.qubits[0]
            if q < n_qubits:
                type_idx = GATE_1Q.index(gate.gate_type)
                node_1q_counts[q, type_idx] += 1
        
        elif gate.gate_type in GATE_2Q and len(gate.qubits) >= 2:
            q1, q2 = gate.qubits[0], gate.qubits[1]
            if q1 < n_qubits:
                node_2q_counts[q1] += 1
                node_first_2q_pos[q1] = min(node_first_2q_pos[q1].item(), norm_pos)
                node_last_2q_pos[q1] = max(node_last_2q_pos[q1].item(), norm_pos)
            if q2 < n_qubits:
                node_2q_counts[q2] += 1
                node_first_2q_pos[q2] = min(node_first_2q_pos[q2].item(), norm_pos)
                node_last_2q_pos[q2] = max(node_last_2q_pos[q2].item(), norm_pos)
    
    # Normalize depth features
    node_avg_depth = node_avg_depth / (node_depth_counts + 1e-6)
    node_max_depth = node_max_depth / max(stats['depth'], 1)
    node_avg_depth = node_avg_depth / max(stats['depth'], 1)
    
    # Position encoding
    node_positions = torch.arange(n_qubits, dtype=torch.float32) / max(n_qubits - 1, 1)
    node_positions = node_positions.unsqueeze(1)
    
    # Log-transform counts
    node_1q_log = torch.log1p(node_1q_counts)
    node_2q_log = torch.log1p(node_2q_counts)
    node_clifford_log = torch.log1p(node_clifford_count)
    node_rotation_log = torch.log1p(node_rotation_count)
    node_diagonal_log = torch.log1p(node_diagonal_count)
    
    # Activity window
    node_activity_window = node_last_gate_pos - node_first_gate_pos
    node_2q_activity_window = node_last_2q_pos - node_first_2q_pos
    
    # Connectivity features
    connectivity_feats = compute_connectivity_features(n_qubits, gates)
    
    # Combine base node features
    node_features = [
        node_1q_log,                # [n, 17] - 1Q gate counts per type
        node_2q_log,                # [n, 1] - 2Q gate count
        node_positions,             # [n, 1] - position in register
        node_first_gate_pos,        # [n, 1] - first gate position
        node_last_gate_pos,         # [n, 1] - last gate position
        node_first_2q_pos,          # [n, 1] - first 2Q gate position
        node_last_2q_pos,           # [n, 1] - last 2Q gate position
        node_activity_window,       # [n, 1] - activity duration
        node_2q_activity_window,    # [n, 1] - 2Q activity duration
        node_max_depth,             # [n, 1] - max depth
        node_avg_depth,             # [n, 1] - avg depth
        node_clifford_log,          # [n, 1] - clifford gate count
        node_rotation_log,          # [n, 1] - rotation gate count
        node_diagonal_log,          # [n, 1] - diagonal gate count
        connectivity_feats,         # [n, 8] - connectivity features
    ]
    
    x = torch.cat(node_features, dim=1)
    
    # =========================================================================
    # EDGE FEATURES
    # =========================================================================
    
    edge_src = []
    edge_dst = []
    edge_gate_type = []
    edge_features_list = []
    
    for gate in gates:
        gate_idx = GATE_TO_IDX.get(gate.gate_type, 0)
        norm_pos = gate.position / total_gates
        norm_depth = gate.depth / max(stats['depth'], 1)
        props = GATE_PROPERTIES.get(gate.gate_type, {})
        
        # Gate property features
        is_clifford = float(props.get('clifford', False))
        is_rotation = float(props.get('rotation', False))
        is_diagonal = float(props.get('diagonal', False))
        is_symmetric = float(props.get('symmetric', False))
        
        # Parameter features
        param_val = gate.params[0] if gate.params else 0.0
        param_sin = math.sin(param_val) if gate.params else 0.0
        param_cos = math.cos(param_val) if gate.params else 0.0
        
        if gate.gate_type in GATE_1Q and len(gate.qubits) >= 1:
            q = gate.qubits[0]
            if q < n_qubits:
                edge_src.append(q)
                edge_dst.append(q)  # Self-loop
                edge_gate_type.append(gate_idx)
                
                edge_feat = [
                    norm_pos,       # Temporal position
                    norm_depth,     # Circuit depth
                    param_val,      # Raw parameter
                    param_sin,      # Sin of parameter
                    param_cos,      # Cos of parameter
                    0.0,            # Qubit distance (0 for self-loop)
                    is_clifford,    # Is Clifford gate
                    is_rotation,    # Is rotation gate
                    is_diagonal,    # Is diagonal gate
                    0.0,            # Is symmetric (N/A for 1Q)
                    1.0,            # Is 1Q gate
                    0.0,            # Is 2Q gate
                ]
                
                # Add sinusoidal encoding
                if include_sinusoidal_pe:
                    edge_feat.extend(sinusoidal_encoding(norm_pos, sinusoidal_dim).tolist())
                
                edge_features_list.append(edge_feat)
        
        elif gate.gate_type in GATE_2Q and len(gate.qubits) >= 2:
            q_ctrl, q_targ = gate.qubits[0], gate.qubits[1]
            if q_ctrl < n_qubits and q_targ < n_qubits:
                qubit_dist = abs(q_targ - q_ctrl) / max(n_qubits - 1, 1)
                
                edge_src.append(q_ctrl)
                edge_dst.append(q_targ)
                edge_gate_type.append(gate_idx)
                
                edge_feat = [
                    norm_pos,
                    norm_depth,
                    param_val,
                    param_sin,
                    param_cos,
                    qubit_dist,
                    is_clifford,
                    is_rotation,
                    is_diagonal,
                    is_symmetric,
                    0.0,            # Is 1Q gate
                    1.0,            # Is 2Q gate
                ]
                
                if include_sinusoidal_pe:
                    edge_feat.extend(sinusoidal_encoding(norm_pos, sinusoidal_dim).tolist())
                
                edge_features_list.append(edge_feat)
                
                # Add reverse edge for symmetric gates
                if is_symmetric:
                    edge_src.append(q_targ)
                    edge_dst.append(q_ctrl)
                    edge_gate_type.append(gate_idx)
                    edge_features_list.append(edge_feat.copy())
        
        elif gate.gate_type in GATE_3Q and len(gate.qubits) >= 3:
            qubits = gate.qubits[:3]
            # Add edges from each control to target
            for i, q_ctrl in enumerate(qubits[:-1]):
                q_targ = qubits[-1]
                if q_ctrl < n_qubits and q_targ < n_qubits:
                    qubit_dist = abs(q_targ - q_ctrl) / max(n_qubits - 1, 1)
                    
                    edge_src.append(q_ctrl)
                    edge_dst.append(q_targ)
                    edge_gate_type.append(gate_idx)
                    
                    edge_feat = [
                        norm_pos,
                        norm_depth,
                        param_val,
                        param_sin,
                        param_cos,
                        qubit_dist,
                        is_clifford,
                        is_rotation,
                        is_diagonal,
                        0.0,        # Is symmetric
                        0.0,        # Is 1Q gate
                        0.0,        # Is 2Q gate (it's 3Q)
                    ]
                    
                    if include_sinusoidal_pe:
                        edge_feat.extend(sinusoidal_encoding(norm_pos, sinusoidal_dim).tolist())
                    
                    edge_features_list.append(edge_feat)
    
    # Build edge tensors
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_gate_type_tensor = torch.tensor(edge_gate_type, dtype=torch.long)
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float32)
    else:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_gate_type_tensor = torch.tensor([0], dtype=torch.long)
        edge_feat_dim = 12 + (sinusoidal_dim if include_sinusoidal_pe else 0)
        edge_attr = torch.zeros(1, edge_feat_dim)
    
    # Add spectral features to nodes
    if include_spectral:
        spectral_feats = compute_spectral_features(edge_index, n_qubits, k=4)
        x = torch.cat([x, spectral_feats], dim=1)
    
    # =========================================================================
    # GLOBAL FEATURES
    # =========================================================================
    
    backend_idx = 1.0 if backend == "GPU" else 0.0
    precision_idx = 1.0 if precision == "double" else 0.0
    
    global_feats = [
        float(n_qubits) / 130.0,
        float(len(gates)) / 1000.0,
        float(stats['n_2q_gates']) / 500.0,
        float(stats['n_2q_gates']) / max(n_qubits, 1) / 10.0,  # Gate density
        backend_idx,
        precision_idx,
        # Additional statistics
        float(stats['depth']) / 1000.0,
        float(stats['n_clifford']) / max(len(gates), 1),
        float(stats['n_rotation']) / max(len(gates), 1),
        float(stats['n_diagonal']) / max(len(gates), 1),
        float(stats['n_1q_gates']) / max(len(gates), 1),
        float(stats['n_3q_gates']) / max(len(gates), 1),
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
        "circuit_stats": stats,
    }


def build_graph_from_file_enhanced(
    qasm_path: Path,
    backend: str = "CPU",
    precision: str = "single",
    family: Optional[str] = None,
    family_to_idx: Optional[Dict[str, int]] = None,
    num_families: int = 20,
    log2_threshold: Optional[float] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Build enhanced graph from a QASM file path."""
    text = qasm_path.read_text(encoding="utf-8")
    return build_graph_enhanced(
        text, backend, precision, family, family_to_idx, num_families,
        log2_threshold=log2_threshold, **kwargs
    )


# Feature dimensions with enhanced features
NODE_FEAT_DIM_ENHANCED = (
    len(GATE_1Q) +  # 1Q gate counts (17)
    1 +             # 2Q gate count
    1 +             # Position
    1 +             # First gate pos
    1 +             # Last gate pos
    1 +             # First 2Q pos
    1 +             # Last 2Q pos
    1 +             # Activity window
    1 +             # 2Q activity window
    1 +             # Max depth
    1 +             # Avg depth
    1 +             # Clifford count
    1 +             # Rotation count
    1 +             # Diagonal count
    8 +             # Connectivity features
    4               # Spectral features
)  # Total: 41

EDGE_FEAT_DIM_ENHANCED = 12 + 16  # Base features + sinusoidal PE = 28

GLOBAL_FEAT_DIM_ENHANCED = 12  # Base global features


if __name__ == "__main__":
    # Test enhanced parser
    test_qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg c[4];
    h q[0];
    cx q[0],q[1];
    t q[1];
    cx q[1],q[2];
    cx q[2],q[3];
    rz(0.5) q[3];
    cz q[0],q[3];
    measure q -> c;
    """
    
    result = build_graph_enhanced(test_qasm, "CPU", "single")
    
    print("Enhanced Graph Builder Test:")
    print(f"  Node features: {result['x'].shape}")
    print(f"  Edge index: {result['edge_index'].shape}")
    print(f"  Edge features: {result['edge_attr'].shape}")
    print(f"  Edge gate types: {result['edge_gate_type'].shape}")
    print(f"  Global features: {result['global_features'].shape}")
    print(f"\nCircuit Statistics:")
    for k, v in result['circuit_stats'].items():
        if k != 'gate_type_counts':
            print(f"  {k}: {v}")
    print(f"\nExpected dimensions:")
    print(f"  NODE_FEAT_DIM_ENHANCED: {NODE_FEAT_DIM_ENHANCED}")
    print(f"  EDGE_FEAT_DIM_ENHANCED: {EDGE_FEAT_DIM_ENHANCED}")

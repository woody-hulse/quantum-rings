"""
Parse QASM files and build PyTorch Geometric graph representations.

Graph structure:
- Nodes: Qubits
- Edges: Gates (directed for 2-qubit gates, self-loops for 1-qubit gates)
- Edge features include gate type embedding and temporal position
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import numpy as np


GATE_1Q = ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'id']
GATE_2Q = ['cx', 'cz', 'swap', 'cp', 'crx', 'cry', 'crz', 'cu1', 'cu3', 'rxx', 'ryy', 'rzz']
GATE_3Q = ['ccx', 'cswap', 'ccz']

ALL_GATE_TYPES = GATE_1Q + GATE_2Q + GATE_3Q
GATE_TO_IDX = {g: i for i, g in enumerate(ALL_GATE_TYPES)}
NUM_GATE_TYPES = len(ALL_GATE_TYPES)


@dataclass
class ParsedGate:
    gate_type: str
    qubits: List[int]
    params: List[float]
    position: int  # temporal position in circuit


def parse_qasm(text: str) -> Tuple[int, List[ParsedGate]]:
    """
    Parse a QASM string and extract gates with their qubits.
    
    Returns:
        n_qubits: Number of qubits in the circuit
        gates: List of parsed gates in order of appearance
    """
    lines = text.splitlines()
    
    n_qubits = 0
    qreg_match = re.search(r"qreg\s+\w+\[(\d+)\]", text)
    if qreg_match:
        n_qubits = int(qreg_match.group(1))
    
    gate_pattern = re.compile(
        r"\b(" + "|".join(ALL_GATE_TYPES) + r")\b"
        r"(?:\s*\(([^)]*)\))?"  # optional parameters
        r"\s+(.+?);"  # qubit arguments
    )
    qubit_ref = re.compile(r"\w+\[(\d+)\]")
    
    gates = []
    position = 0
    
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
                gates.append(ParsedGate(
                    gate_type=gate_type,
                    qubits=qubits,
                    params=params,
                    position=position,
                ))
                position += 1
    
    return n_qubits, gates


def build_graph_from_qasm(
    qasm_text: str,
    backend: str = "CPU",
    precision: str = "single",
    family: Optional[str] = None,
    family_to_idx: Optional[Dict[str, int]] = None,
    num_families: int = 20,
    rich_features: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Build a graph representation from QASM text.
    
    Returns a dictionary with:
        - x: Node features [n_nodes, node_feat_dim]
        - edge_index: Edge connectivity [2, n_edges]
        - edge_attr: Edge features [n_edges, edge_feat_dim]
        - global_features: Circuit-level features [global_feat_dim]
        - n_qubits: Number of qubits
    """
    n_qubits, gates = parse_qasm(qasm_text)
    
    if n_qubits == 0:
        n_qubits = 1  # fallback
    
    # Node features: per-qubit gate counts and position
    node_1q_counts = torch.zeros(n_qubits, len(GATE_1Q))
    node_2q_counts = torch.zeros(n_qubits, 1)
    
    # Rich node features: temporal info per qubit
    node_first_2q_pos = torch.ones(n_qubits, 1)  # first 2Q gate position (1.0 = never)
    node_last_2q_pos = torch.zeros(n_qubits, 1)   # last 2Q gate position (0.0 = never)
    node_unique_neighbors = [set() for _ in range(n_qubits)]  # unique qubits interacted with
    node_total_span = torch.zeros(n_qubits, 1)  # sum of interaction distances
    
    # Edge lists
    edge_src = []
    edge_dst = []
    edge_gate_type = []
    edge_position = []
    edge_params = []
    edge_qubit_dist = []  # distance between qubits
    edge_cumulative_idx = []  # cumulative edge count at this position
    
    total_gates = len(gates) if gates else 1
    cumulative_2q = 0
    
    for gate in gates:
        gate_idx = GATE_TO_IDX.get(gate.gate_type, 0)
        norm_pos = gate.position / total_gates
        
        if gate.gate_type in GATE_1Q:
            if len(gate.qubits) >= 1 and gate.qubits[0] < n_qubits:
                q = gate.qubits[0]
                type_idx = GATE_1Q.index(gate.gate_type)
                node_1q_counts[q, type_idx] += 1
                
                # Self-loop edge for 1-qubit gate
                edge_src.append(q)
                edge_dst.append(q)
                edge_gate_type.append(gate_idx)
                edge_position.append(norm_pos)
                edge_params.append(gate.params[0] if gate.params else 0.0)
                edge_qubit_dist.append(0.0)
                edge_cumulative_idx.append(cumulative_2q / max(total_gates, 1))
        
        elif gate.gate_type in GATE_2Q:
            if len(gate.qubits) >= 2:
                q_ctrl, q_targ = gate.qubits[0], gate.qubits[1]
                if q_ctrl < n_qubits and q_targ < n_qubits:
                    node_2q_counts[q_ctrl] += 1
                    node_2q_counts[q_targ] += 1
                    
                    # Update temporal features
                    node_first_2q_pos[q_ctrl] = min(node_first_2q_pos[q_ctrl].item(), norm_pos)
                    node_first_2q_pos[q_targ] = min(node_first_2q_pos[q_targ].item(), norm_pos)
                    node_last_2q_pos[q_ctrl] = max(node_last_2q_pos[q_ctrl].item(), norm_pos)
                    node_last_2q_pos[q_targ] = max(node_last_2q_pos[q_targ].item(), norm_pos)
                    
                    # Update neighbor sets
                    node_unique_neighbors[q_ctrl].add(q_targ)
                    node_unique_neighbors[q_targ].add(q_ctrl)
                    
                    # Update span
                    span = abs(q_targ - q_ctrl)
                    node_total_span[q_ctrl] += span
                    node_total_span[q_targ] += span
                    
                    cumulative_2q += 1
                    
                    # Directed edge: control -> target
                    edge_src.append(q_ctrl)
                    edge_dst.append(q_targ)
                    edge_gate_type.append(gate_idx)
                    edge_position.append(norm_pos)
                    edge_params.append(gate.params[0] if gate.params else 0.0)
                    edge_qubit_dist.append(span / max(n_qubits - 1, 1))
                    edge_cumulative_idx.append(cumulative_2q / max(total_gates, 1))
        
        elif gate.gate_type in GATE_3Q:
            if len(gate.qubits) >= 3:
                q1, q2, q_targ = gate.qubits[0], gate.qubits[1], gate.qubits[2]
                for q_ctrl in [q1, q2]:
                    if q_ctrl < n_qubits and q_targ < n_qubits:
                        node_2q_counts[q_ctrl] += 1
                        node_2q_counts[q_targ] += 1
                        
                        node_first_2q_pos[q_ctrl] = min(node_first_2q_pos[q_ctrl].item(), norm_pos)
                        node_first_2q_pos[q_targ] = min(node_first_2q_pos[q_targ].item(), norm_pos)
                        node_last_2q_pos[q_ctrl] = max(node_last_2q_pos[q_ctrl].item(), norm_pos)
                        node_last_2q_pos[q_targ] = max(node_last_2q_pos[q_targ].item(), norm_pos)
                        
                        node_unique_neighbors[q_ctrl].add(q_targ)
                        node_unique_neighbors[q_targ].add(q_ctrl)
                        
                        span = abs(q_targ - q_ctrl)
                        node_total_span[q_ctrl] += span
                        node_total_span[q_targ] += span
                        
                        cumulative_2q += 1
                        
                        edge_src.append(q_ctrl)
                        edge_dst.append(q_targ)
                        edge_gate_type.append(gate_idx)
                        edge_position.append(norm_pos)
                        edge_params.append(gate.params[0] if gate.params else 0.0)
                        edge_qubit_dist.append(span / max(n_qubits - 1, 1))
                        edge_cumulative_idx.append(cumulative_2q / max(total_gates, 1))
    
    # Compute derived node features
    node_positions = torch.arange(n_qubits, dtype=torch.float32) / max(n_qubits - 1, 1)
    node_positions = node_positions.unsqueeze(1)
    
    node_1q_log = torch.log1p(node_1q_counts)
    node_2q_log = torch.log1p(node_2q_counts)
    
    # Unique neighbor count (normalized by n_qubits)
    node_degree = torch.tensor(
        [len(neighbors) for neighbors in node_unique_neighbors], 
        dtype=torch.float32
    ).unsqueeze(1) / max(n_qubits - 1, 1)
    
    # Normalize total span
    node_avg_span = node_total_span / (node_2q_counts + 1)
    
    # Activity window (duration qubit is involved in 2Q gates)
    node_activity_window = node_last_2q_pos - node_first_2q_pos
    
    # Combine node features
    if rich_features:
        x = torch.cat([
            node_1q_log,           # [n_qubits, 15] - 1Q gate counts per type
            node_2q_log,           # [n_qubits, 1] - 2Q involvement count
            node_positions,        # [n_qubits, 1] - normalized position in register
            node_first_2q_pos,     # [n_qubits, 1] - when first entangled
            node_last_2q_pos,      # [n_qubits, 1] - when last entangled  
            node_activity_window,  # [n_qubits, 1] - duration of activity
            node_degree,           # [n_qubits, 1] - unique interaction partners
            node_avg_span,         # [n_qubits, 1] - average interaction distance
        ], dim=1)
    else:
        x = torch.cat([
            node_1q_log,
            node_2q_log,
            node_positions,
        ], dim=1)
    
    # Build edge tensors
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_gate_type_tensor = torch.tensor(edge_gate_type, dtype=torch.long)
        edge_position_tensor = torch.tensor(edge_position, dtype=torch.float32).unsqueeze(1)
        edge_params_tensor = torch.tensor(edge_params, dtype=torch.float32).unsqueeze(1)
        edge_dist_tensor = torch.tensor(edge_qubit_dist, dtype=torch.float32).unsqueeze(1)
        edge_cumul_tensor = torch.tensor(edge_cumulative_idx, dtype=torch.float32).unsqueeze(1)
        
        if rich_features:
            edge_attr = torch.cat([
                edge_position_tensor,   # temporal position
                edge_params_tensor,     # gate parameter
                edge_dist_tensor,       # qubit distance (normalized)
                edge_cumul_tensor,      # cumulative 2Q gate count
            ], dim=1)
        else:
            edge_attr = torch.cat([
                edge_position_tensor,
                edge_params_tensor,
            ], dim=1)
    else:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_gate_type_tensor = torch.tensor([0], dtype=torch.long)
        edge_attr = torch.zeros(1, 4 if rich_features else 2)
    
    # Global features
    backend_idx = 1.0 if backend == "GPU" else 0.0
    precision_idx = 1.0 if precision == "double" else 0.0
    
    # Compute graph-level statistics
    n_2q_gates = cumulative_2q
    gate_density = n_2q_gates / max(n_qubits, 1)
    
    global_feats = [
        float(n_qubits) / 130.0,
        float(len(gates)) / 1000.0,
        float(n_2q_gates) / 500.0,
        gate_density / 10.0,
        backend_idx,
        precision_idx,
    ]
    
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
    }


def build_graph_from_file(
    qasm_path: Path,
    backend: str = "CPU",
    precision: str = "single",
    family: Optional[str] = None,
    family_to_idx: Optional[Dict[str, int]] = None,
    num_families: int = 20,
) -> Dict[str, torch.Tensor]:
    """Build graph from a QASM file path."""
    text = qasm_path.read_text(encoding="utf-8")
    return build_graph_from_qasm(
        text, backend, precision, family, family_to_idx, num_families
    )


# Feature dimensions for external reference (with rich_features=True)
NODE_FEAT_DIM = len(GATE_1Q) + 1 + 1 + 5  # 1Q counts + 2Q count + position + 5 temporal = 22
NODE_FEAT_DIM_BASIC = len(GATE_1Q) + 1 + 1  # without rich features = 17
EDGE_FEAT_DIM = 4  # temporal position + param + qubit_dist + cumulative
EDGE_FEAT_DIM_BASIC = 2  # without rich features
GLOBAL_FEAT_DIM_BASE = 6  # n_qubits, n_gates, n_2q_gates, gate_density, backend, precision


if __name__ == "__main__":
    # Test parsing
    test_qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg c[4];
    h q[0];
    cx q[0],q[1];
    cx q[1],q[2];
    cx q[2],q[3];
    rz(0.5) q[3];
    measure q[0] -> c[0];
    """
    
    result = build_graph_from_qasm(test_qasm, "CPU", "single")
    print(f"Nodes: {result['x'].shape}")
    print(f"Edges: {result['edge_index'].shape}")
    print(f"Edge attr: {result['edge_attr'].shape}")
    print(f"Edge gate types: {result['edge_gate_type']}")
    print(f"Global features: {result['global_features'].shape}")

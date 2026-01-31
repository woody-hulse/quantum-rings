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
    # [n_1q_gates_per_type..., n_2q_gates_involved, normalized_position]
    node_1q_counts = torch.zeros(n_qubits, len(GATE_1Q))
    node_2q_counts = torch.zeros(n_qubits, 1)
    
    # Edge lists
    edge_src = []
    edge_dst = []
    edge_gate_type = []
    edge_position = []
    edge_params = []
    
    total_gates = len(gates) if gates else 1
    
    for gate in gates:
        gate_idx = GATE_TO_IDX.get(gate.gate_type, 0)
        
        if gate.gate_type in GATE_1Q:
            if len(gate.qubits) >= 1 and gate.qubits[0] < n_qubits:
                q = gate.qubits[0]
                type_idx = GATE_1Q.index(gate.gate_type)
                node_1q_counts[q, type_idx] += 1
                
                # Self-loop edge for 1-qubit gate
                edge_src.append(q)
                edge_dst.append(q)
                edge_gate_type.append(gate_idx)
                edge_position.append(gate.position / total_gates)
                edge_params.append(gate.params[0] if gate.params else 0.0)
        
        elif gate.gate_type in GATE_2Q:
            if len(gate.qubits) >= 2:
                q_ctrl, q_targ = gate.qubits[0], gate.qubits[1]
                if q_ctrl < n_qubits and q_targ < n_qubits:
                    node_2q_counts[q_ctrl] += 1
                    node_2q_counts[q_targ] += 1
                    
                    # Directed edge: control -> target
                    edge_src.append(q_ctrl)
                    edge_dst.append(q_targ)
                    edge_gate_type.append(gate_idx)
                    edge_position.append(gate.position / total_gates)
                    edge_params.append(gate.params[0] if gate.params else 0.0)
        
        elif gate.gate_type in GATE_3Q:
            # Decompose into pairs: ctrl1->target, ctrl2->target
            if len(gate.qubits) >= 3:
                q1, q2, q_targ = gate.qubits[0], gate.qubits[1], gate.qubits[2]
                for q_ctrl in [q1, q2]:
                    if q_ctrl < n_qubits and q_targ < n_qubits:
                        node_2q_counts[q_ctrl] += 1
                        node_2q_counts[q_targ] += 1
                        
                        edge_src.append(q_ctrl)
                        edge_dst.append(q_targ)
                        edge_gate_type.append(gate_idx)
                        edge_position.append(gate.position / total_gates)
                        edge_params.append(gate.params[0] if gate.params else 0.0)
    
    # Normalize node features
    node_positions = torch.arange(n_qubits, dtype=torch.float32) / max(n_qubits - 1, 1)
    node_positions = node_positions.unsqueeze(1)
    
    # Log-transform counts
    node_1q_log = torch.log1p(node_1q_counts)
    node_2q_log = torch.log1p(node_2q_counts)
    
    # Combine node features
    x = torch.cat([
        node_1q_log,           # [n_qubits, 15] - 1Q gate counts per type
        node_2q_log,           # [n_qubits, 1] - 2Q involvement count
        node_positions,        # [n_qubits, 1] - normalized position
    ], dim=1)
    
    # Build edge tensors
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_gate_type_tensor = torch.tensor(edge_gate_type, dtype=torch.long)
        edge_position_tensor = torch.tensor(edge_position, dtype=torch.float32).unsqueeze(1)
        edge_params_tensor = torch.tensor(edge_params, dtype=torch.float32).unsqueeze(1)
        
        # Edge attr: [gate_type_idx, temporal_position, param_value]
        # Gate type will be embedded by the model
        edge_attr = torch.cat([
            edge_position_tensor,
            edge_params_tensor,
        ], dim=1)
    else:
        # No edges - create dummy self-loop on node 0
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_gate_type_tensor = torch.tensor([0], dtype=torch.long)
        edge_attr = torch.zeros(1, 2)
    
    # Global features
    backend_idx = 1.0 if backend == "GPU" else 0.0
    precision_idx = 1.0 if precision == "double" else 0.0
    
    global_feats = [
        float(n_qubits) / 130.0,  # normalized n_qubits (max ~130)
        float(len(gates)) / 1000.0,  # normalized gate count
        backend_idx,
        precision_idx,
    ]
    
    # Add family one-hot if available
    if family and family_to_idx:
        family_onehot = [0.0] * num_families
        if family in family_to_idx:
            family_onehot[family_to_idx[family]] = 1.0
        global_feats.extend(family_onehot)
    
    # Store as 2D tensor [1, feat_dim] so PyG stacks correctly during batching
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


# Feature dimensions for external reference
NODE_FEAT_DIM = len(GATE_1Q) + 1 + 1  # 1Q counts + 2Q count + position = 17
EDGE_FEAT_DIM = 2  # temporal position + param value
GLOBAL_FEAT_DIM_BASE = 4  # n_qubits, n_gates, backend, precision


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

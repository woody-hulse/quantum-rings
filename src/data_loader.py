"""
PyTorch-compatible data loading for the quantum circuit fingerprint challenge.

Provides Dataset classes for training threshold prediction and runtime estimation models.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BACKEND_MAP = {"CPU": 0, "GPU": 1}
PRECISION_MAP = {"single": 0, "double": 1}


# input
@dataclass
class CircuitInfo:
    file: str
    family: str
    n_qubits: int
    source_name: str = ""
    source_url: str = ""

# output subclass
@dataclass
class ThresholdSweepEntry:
    threshold: int
    sdk_get_fidelity: Optional[float]
    p_return_zero: Optional[float]
    run_wall_s: Optional[float]
    peak_rss_mb: Optional[float]
    returncode: int
    note: str

# output
@dataclass
class ResultEntry:
    file: str
    backend: str
    precision: str
    status: str
    selected_threshold: Optional[int] = None
    target_fidelity: Optional[float] = None # always 0.99
    threshold_sweep: List[ThresholdSweepEntry] = field(default_factory=list)
    forward_wall_s: Optional[float] = None
    forward_shots: Optional[int] = None


def _compute_graph_features(qubit_pairs: set, n_qubits: int) -> Dict[str, float]:
    """Compute interaction graph features from 2-qubit gate pairs."""
    if not qubit_pairs or n_qubits == 0: # TODO check that these are the right defaults
        return {
            "max_degree": 0,
            "avg_degree": 0.0,
            "degree_entropy": 0.0,
            "n_connected_components": 0,
            "clustering_coeff": 0.0,
            "max_component_size": 1.0 if n_qubits > 0 else 0.0,
            "component_entropy": 0.0,
        }
    
    degree = [0] * n_qubits # TODO check that we don't want 2^n
    adjacency = {i: set() for i in range(n_qubits)} # for sets duplicates are removed automatically
    
    for q1, q2 in qubit_pairs:
        if q1 < n_qubits and q2 < n_qubits:
            degree[q1] += 1
            degree[q2] += 1
            adjacency[q1].add(q2)
            adjacency[q2].add(q1)
    
    max_degree = max(degree)
    avg_degree = np.mean(degree)
    
    degree_counts = np.array(degree, dtype=float)
    degree_counts = degree_counts[degree_counts > 0]
    if len(degree_counts) > 0:
        probs = degree_counts / degree_counts.sum()
        raw_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(n_qubits) if n_qubits > 1 else 1.0
        degree_entropy = raw_entropy / max_entropy
    else:
        degree_entropy = 0.0
    
    visited = set()
    n_components = 0
    max_component_size = 1.0 if n_qubits > 0 else 0.0
    component_sizes = []
    for start in range(n_qubits):
        if start in visited or not adjacency[start]:
            continue
        n_components += 1
        component_size = 0
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component_size += 1
            stack.extend(adjacency[node] - visited)
        max_component_size = max(max_component_size, float(component_size))
        component_sizes.append(component_size)
    
    if component_sizes:
        sizes = np.array(component_sizes, dtype=float)
        probs = sizes / sizes.sum()
        raw_comp_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_ent = np.log2(n_qubits) if n_qubits > 1 else 1.0
        component_entropy = raw_comp_entropy / max_ent
    else:
        component_entropy = 0.0
    
    triangles = 0
    triplets = 0
    for node in range(n_qubits):
        neighbors = list(adjacency[node])
        k = len(neighbors)
        if k >= 2:
            triplets += k * (k - 1) // 2
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in adjacency[neighbors[i]]:
                        triangles += 1
    clustering_coeff = triangles / triplets if triplets > 0 else 0.0
    
    return {
        "max_degree": max_degree,
        "avg_degree": avg_degree,
        "degree_entropy": degree_entropy,
        "n_connected_components": n_components,
        "clustering_coeff": clustering_coeff,
        "max_component_size": max_component_size,
        "component_entropy": component_entropy,
    }


def _estimate_depth(text: str, n_qubits: int) -> Dict[str, float]:
    """Estimate circuit depth using a simple layer heuristic."""
    if n_qubits == 0:
        return {"estimated_depth": 0, "depth_per_qubit": 0.0}
    
    qubit_depth = [0] * n_qubits
    
    gate_pattern = re.compile(
        r"\b(cx|cz|swap|ccx|h|x|y|z|s|sdg|t|tdg|rx|ry|rz|u1|u2|u3)\b[^;]*;"
    )
    qubit_ref = re.compile(r"\w+\[(\d+)\]")
    
    for match in gate_pattern.finditer(text):
        gate_text = match.group(0)
        qubits = [int(m.group(1)) for m in qubit_ref.finditer(gate_text)]
        qubits = [q for q in qubits if q < n_qubits]
        if qubits:
            max_layer = max(qubit_depth[q] for q in qubits)
            for q in qubits:
                qubit_depth[q] = max_layer + 1
    
    estimated_depth = max(qubit_depth) if qubit_depth else 0
    depth_per_qubit = estimated_depth / n_qubits if n_qubits > 0 else 0.0
    
    return {"estimated_depth": estimated_depth, "depth_per_qubit": depth_per_qubit}


def _compute_cut_features(qubit_pairs: set, all_2q_ops: list, n_qubits: int) -> Dict[str, float]:
    """Compute entanglement cut pressure features."""
    if n_qubits < 2 or not all_2q_ops:
        return {
            "middle_cut_crossings": 0,
            "cut_crossing_ratio": 0.0,
            "max_cut_crossings": 0,
        }
    
    middle = n_qubits // 2
    middle_crossings = sum(1 for q1, q2 in all_2q_ops if (q1 < middle) != (q2 < middle))
    
    cut_counts = []
    for cut_pos in range(1, n_qubits):
        crossings = sum(1 for q1, q2 in all_2q_ops if (q1 < cut_pos) != (q2 < cut_pos))
        cut_counts.append(crossings)
    
    max_cut = max(cut_counts) if cut_counts else 0
    total_ops = len(all_2q_ops)
    
    return {
        "middle_cut_crossings": middle_crossings,
        "cut_crossing_ratio": middle_crossings / total_ops if total_ops > 0 else 0.0,
        "max_cut_crossings": max_cut,
    }


def _compute_graph_bandwidth(qubit_pairs: set, n_qubits: int) -> Dict[str, float]:
    """
    Compute graph bandwidth - a key metric for MPS simulation complexity.
    
    Bandwidth measures how "spread out" the interaction graph is along the 1D qubit chain.
    Lower bandwidth = easier for MPS. This is related to the minimum bond dimension needed.
    """
    if not qubit_pairs or n_qubits == 0:
        return {
            "graph_bandwidth": 0,
            "normalized_bandwidth": 0.0,
            "bandwidth_squared": 0,
        }
    
    bandwidth = max(abs(q2 - q1) for q1, q2 in qubit_pairs)
    normalized = bandwidth / n_qubits if n_qubits > 0 else 0.0
    
    return {
        "graph_bandwidth": bandwidth,
        "normalized_bandwidth": normalized,
        "bandwidth_squared": bandwidth ** 2,
    }


def _compute_temporal_features(text: str, n_qubits: int) -> Dict[str, float]:
    """
    Analyze temporal structure of gate placement.
    
    Key insight: Long-range gates early in the circuit cause entanglement to persist
    and accumulate. Late long-range gates may have less impact on total entanglement.
    """
    if n_qubits == 0:
        return {
            "early_longrange_ratio": 0.0,
            "late_longrange_ratio": 0.0,
            "longrange_temporal_center": 0.5,
            "entanglement_velocity": 0.0,
        }
    
    gate_2q_pattern = re.compile(r"\b(cx|cz|swap)\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]")
    
    gate_positions = []
    gate_spans = []
    
    for i, match in enumerate(gate_2q_pattern.finditer(text)):
        q1, q2 = int(match.group(2)), int(match.group(3))
        span = abs(q2 - q1)
        gate_positions.append(i)
        gate_spans.append(span)
    
    if not gate_positions:
        return {
            "early_longrange_ratio": 0.0,
            "late_longrange_ratio": 0.0,
            "longrange_temporal_center": 0.5,
            "entanglement_velocity": 0.0,
        }
    
    total_gates = len(gate_positions)
    threshold_span = n_qubits // 4  # "long-range" = spans > 25% of qubit count
    
    early_cutoff = total_gates // 3
    late_cutoff = 2 * total_gates // 3
    
    early_longrange = sum(1 for i, s in enumerate(gate_spans) if i < early_cutoff and s > threshold_span)
    late_longrange = sum(1 for i, s in enumerate(gate_spans) if i >= late_cutoff and s > threshold_span)
    
    early_total = max(early_cutoff, 1)
    late_total = max(total_gates - late_cutoff, 1)
    
    longrange_positions = [i for i, s in enumerate(gate_spans) if s > threshold_span]
    if longrange_positions:
        temporal_center = np.mean(longrange_positions) / total_gates
    else:
        temporal_center = 0.5
    
    cumulative_entanglement = np.cumsum(gate_spans)
    if len(cumulative_entanglement) > 1:
        velocity = np.mean(np.diff(cumulative_entanglement))
    else:
        velocity = gate_spans[0] if gate_spans else 0.0
    
    return {
        "early_longrange_ratio": early_longrange / early_total,
        "late_longrange_ratio": late_longrange / late_total,
        "longrange_temporal_center": temporal_center,
        "entanglement_velocity": velocity,
    }


def _compute_qubit_activity_features(text: str, n_qubits: int) -> Dict[str, float]:
    """
    Analyze qubit activity distribution.
    
    Uniform activity often indicates structured algorithms (QFT, Grover).
    Concentrated activity may indicate simpler circuits or local operations.
    """
    if n_qubits == 0:
        return {
            "qubit_activity_entropy": 0.0,
            "qubit_activity_variance": 0.0,
            "qubit_activity_max_ratio": 0.0,
            "active_qubit_fraction": 0.0,
        }
    
    qubit_ref = re.compile(r"\w+\[(\d+)\]")
    activity = [0] * n_qubits
    
    for match in qubit_ref.finditer(text):
        q = int(match.group(1))
        if q < n_qubits:
            activity[q] += 1
    
    total = sum(activity)
    if total == 0:
        return {
            "qubit_activity_entropy": 0.0,
            "qubit_activity_variance": 0.0,
            "qubit_activity_max_ratio": 0.0,
            "active_qubit_fraction": 0.0,
        }
    
    probs = np.array(activity, dtype=float) / total
    probs_nonzero = probs[probs > 0]
    entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
    max_entropy = np.log2(n_qubits) if n_qubits > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    variance = np.var(activity)
    max_activity = max(activity)
    max_ratio = max_activity / total
    active_fraction = sum(1 for a in activity if a > 0) / n_qubits
    
    return {
        "qubit_activity_entropy": normalized_entropy,
        "qubit_activity_variance": variance,
        "qubit_activity_max_ratio": max_ratio,
        "active_qubit_fraction": active_fraction,
    }


def _compute_gate_pattern_features(text: str) -> Dict[str, float]:
    """
    Detect common gate sequence patterns (n-grams) that indicate circuit structure.
    
    Certain patterns are signatures of specific algorithms:
    - H-CX patterns: Bell states, GHZ preparation
    - CX-RZ-CX: Variational circuits (VQE/QAOA)
    - Repeated CX chains: Data encoding, entanglement layers
    """
    gate_pattern = re.compile(r"\b(cx|cz|h|x|y|z|rx|ry|rz|s|t|swap|u1|u2|u3)\b")
    gates = [m.group(1) for m in gate_pattern.finditer(text)]
    
    if len(gates) < 2:
        return {
            "cx_chain_max_length": 0,
            "h_cx_pattern_count": 0,
            "cx_rz_cx_pattern_count": 0,
            "rotation_density": 0.0,
            "gate_type_entropy": 0.0,
            "cx_h_ratio": 0.0,
        }
    
    cx_chain_length = 0
    max_cx_chain = 0
    for g in gates:
        if g == "cx":
            cx_chain_length += 1
            max_cx_chain = max(max_cx_chain, cx_chain_length)
        else:
            cx_chain_length = 0
    
    h_cx_count = 0
    cx_rz_cx_count = 0
    for i in range(len(gates) - 1):
        if gates[i] == "h" and gates[i+1] == "cx":
            h_cx_count += 1
    for i in range(len(gates) - 2):
        if gates[i] == "cx" and gates[i+1] in ("rz", "rx", "ry", "u1") and gates[i+2] == "cx":
            cx_rz_cx_count += 1
    
    rotation_gates = {"rx", "ry", "rz", "u1", "u2", "u3"}
    n_rotations = sum(1 for g in gates if g in rotation_gates)
    rotation_density = n_rotations / len(gates) if gates else 0.0
    
    from collections import Counter
    gate_counts = Counter(gates)
    total = sum(gate_counts.values())
    probs = np.array(list(gate_counts.values()), dtype=float) / total
    gate_entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    n_cx = gate_counts.get("cx", 0)
    n_h = gate_counts.get("h", 0)
    cx_h_ratio = n_cx / (n_h + 1)
    
    return {
        "cx_chain_max_length": max_cx_chain,
        "h_cx_pattern_count": h_cx_count,
        "cx_rz_cx_pattern_count": cx_rz_cx_count,
        "rotation_density": rotation_density,
        "gate_type_entropy": gate_entropy,
        "cx_h_ratio": cx_h_ratio,
    }


def _compute_light_cone_features(all_2q_ops: list, n_qubits: int) -> Dict[str, float]:
    """
    Compute light cone spread - how quickly information propagates across qubits.
    
    Fast light cone expansion = rapid entanglement growth = harder for MPS.
    This simulates information spread from the middle qubit.
    """
    if n_qubits == 0 or not all_2q_ops:
        return {
            "light_cone_spread_rate": 0.0,
            "light_cone_half_coverage_depth": 0.0,
            "final_light_cone_size": 0.0,
        }
    
    start_qubit = n_qubits // 2
    reached = {start_qubit}
    
    spread_history = [1]
    
    for q1, q2 in all_2q_ops:
        if q1 in reached or q2 in reached:
            reached.add(q1)
            reached.add(q2)
        spread_history.append(len(reached))
    
    if len(spread_history) > 1:
        spread_rate = (spread_history[-1] - spread_history[0]) / len(all_2q_ops)
    else:
        spread_rate = 0.0
    
    half_coverage = n_qubits // 2
    half_depth = len(all_2q_ops)
    for i, size in enumerate(spread_history):
        if size >= half_coverage:
            half_depth = i
            break
    normalized_half_depth = half_depth / len(all_2q_ops) if all_2q_ops else 0.0
    
    final_coverage = len(reached) / n_qubits
    
    return {
        "light_cone_spread_rate": spread_rate,
        "light_cone_half_coverage_depth": normalized_half_depth,
        "final_light_cone_size": final_coverage,
    }


def _compute_entanglement_structure_features(all_2q_ops: list, n_qubits: int) -> Dict[str, float]:
    """
    Analyze entanglement structure patterns.
    
    Key features:
    - Nearest-neighbor ratio: High = linear/local structure (easier for MPS)
    - All-to-all ratio: High = highly connected (harder for MPS)
    - Layer structure: Regular layers vs random placement
    """
    if n_qubits < 2 or not all_2q_ops:
        return {
            "nearest_neighbor_ratio": 0.0,
            "long_range_ratio": 0.0,
            "span_gini_coefficient": 0.0,
            "weighted_span_sum": 0.0,
        }
    
    spans = [abs(q2 - q1) for q1, q2 in all_2q_ops]
    
    nn_count = sum(1 for s in spans if s == 1)
    nn_ratio = nn_count / len(spans)
    
    long_threshold = n_qubits // 3
    long_count = sum(1 for s in spans if s >= long_threshold)
    long_ratio = long_count / len(spans)
    
    sorted_spans = np.sort(spans)
    n = len(sorted_spans)
    cumulative = np.cumsum(sorted_spans)
    gini = (2 * np.sum((np.arange(1, n+1, dtype=np.float64) * sorted_spans))) / (n * cumulative[-1]) - (n + 1) / n
    gini = max(0, min(1, gini))
    
    weights = np.array(spans, dtype=np.float64) ** 2
    weighted_sum = np.sum(weights) / (n_qubits ** 2) if n_qubits > 0 else 0.0
    
    return {
        "nearest_neighbor_ratio": nn_ratio,
        "long_range_ratio": long_ratio,
        "span_gini_coefficient": gini,
        "weighted_span_sum": weighted_sum,
    }


def _compute_circuit_regularity_features(text: str, n_qubits: int) -> Dict[str, float]:
    """
    Detect circuit regularity and repetition patterns.
    
    Regular/repetitive circuits often have predictable entanglement behavior.
    This helps identify variational ansatz patterns and structured algorithms.
    """
    gate_pattern = re.compile(r"\b(cx|cz|h|rx|ry|rz|swap)\s+\w+\[(\d+)\]")
    
    gate_sequence = []
    for match in gate_pattern.finditer(text):
        gate_type = match.group(1)
        qubit = int(match.group(2))
        gate_sequence.append((gate_type, qubit % 4))  # mod 4 for pattern detection
    
    if len(gate_sequence) < 10:
        return {
            "pattern_repetition_score": 0.0,
            "barrier_regularity": 0.0,
            "layer_uniformity": 0.0,
        }
    
    pattern_counts = {}
    window_size = 4
    for i in range(len(gate_sequence) - window_size):
        pattern = tuple(gate_sequence[i:i+window_size])
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    if pattern_counts:
        max_repeat = max(pattern_counts.values())
        repetition_score = max_repeat / (len(gate_sequence) - window_size)
    else:
        repetition_score = 0.0
    
    barrier_positions = [m.start() for m in re.finditer(r"\bbarrier\b", text)]
    if len(barrier_positions) > 1:
        gaps = np.diff(barrier_positions)
        barrier_regularity = 1.0 - (np.std(gaps) / (np.mean(gaps) + 1))
        barrier_regularity = max(0, min(1, barrier_regularity))
    else:
        barrier_regularity = 0.0
    
    return {
        "pattern_repetition_score": repetition_score,
        "barrier_regularity": barrier_regularity,
        "layer_uniformity": 0.5,  # placeholder for more complex analysis
    }


def _compute_treewidth_features(qubit_pairs: set, n_qubits: int) -> Dict[str, float]:
    """
    Estimate treewidth using the Min-Degree heuristic.
    
    Treewidth is a key parameter for tensor network contraction complexity.
    Exact computation is NP-hard, so we compute an upper bound using
    the Min-Degree elimination ordering heuristic.
    """
    if n_qubits == 0:
        return {"treewidth_min_degree": 0.0}
    
    # Build adjacency list
    adj = {i: set() for i in range(n_qubits)}
    for q1, q2 in qubit_pairs:
        if q1 < n_qubits and q2 < n_qubits:
            adj[q1].add(q2)
            adj[q2].add(q1)
            
    max_degree_at_elimination = 0
    active_nodes = set(range(n_qubits))
    
    # Greedy elimination
    for _ in range(n_qubits):
        # Find node with min degree
        min_deg = n_qubits + 1
        node_to_eliminate = -1
        
        for node in active_nodes:
            deg = len(adj[node])
            if deg < min_deg:
                min_deg = deg
                node_to_eliminate = node
        
        if node_to_eliminate == -1:
            break
            
        max_degree_at_elimination = max(max_degree_at_elimination, min_deg)
        
        # Add fill-in edges (clique on neighbors)
        neighbors = list(adj[node_to_eliminate])
        for i in range(len(neighbors)):
            u = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                v = neighbors[j]
                if v not in adj[u]:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Remove node
        active_nodes.remove(node_to_eliminate)
        for u in neighbors:
            adj[u].remove(node_to_eliminate)
            
    return {"treewidth_min_degree": float(max_degree_at_elimination)}


def extract_qasm_features(qasm_path: Path) -> Dict[str, Any]:
    """Extract features from a QASM file for model input."""
    if not qasm_path.exists():
        return {}
    
    text = qasm_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    
    non_empty_lines = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("//"))
    n_qubits = 0
    qreg_match = re.search(r"qreg\s+\w+\[(\d+)\]", text)
    if qreg_match:
        n_qubits = int(qreg_match.group(1))
    
    n_cx = len(re.findall(r"\bcx\b", text))
    n_cz = len(re.findall(r"\bcz\b", text))
    n_swap = len(re.findall(r"\bswap\b", text))
    n_ccx = len(re.findall(r"\bccx\b", text))
    n_2q_gates = n_cx + n_cz + n_swap + n_ccx
    
    n_h = len(re.findall(r"\bh\b", text))
    n_x = len(re.findall(r"\bx\b", text))
    n_y = len(re.findall(r"\by\b", text))
    n_z = len(re.findall(r"\bz\b", text))
    n_s = len(re.findall(r"\bs\b", text))
    n_t = len(re.findall(r"\bt\b", text))
    n_rx = len(re.findall(r"\brx\b", text))
    n_ry = len(re.findall(r"\bry\b", text))
    n_rz = len(re.findall(r"\brz\b", text))
    n_u1 = len(re.findall(r"\bu1\b", text))
    n_u2 = len(re.findall(r"\bu2\b", text))
    n_u3 = len(re.findall(r"\bu3\b", text))
    n_1q_gates = n_h + n_x + n_y + n_z + n_s + n_t + n_rx + n_ry + n_rz + n_u1 + n_u2 + n_u3
    
    n_measure = len(re.findall(r"\bmeasure\b", text))
    n_barrier = len(re.findall(r"\bbarrier\b", text))
    n_custom_gates = len(re.findall(r"\bgate\s+\w+", text))
    
    qubit_pairs = set()
    all_2q_ops = []
    for match in re.finditer(r"\bcx\s+(\w+)\[(\d+)\]\s*,\s*(\w+)\[(\d+)\]", text):
        q1, q2 = int(match.group(2)), int(match.group(4))
        qubit_pairs.add((min(q1, q2), max(q1, q2)))
        all_2q_ops.append((q1, q2))
    for match in re.finditer(r"\bcz\s+(\w+)\[(\d+)\]\s*,\s*(\w+)\[(\d+)\]", text):
        q1, q2 = int(match.group(2)), int(match.group(4))
        qubit_pairs.add((min(q1, q2), max(q1, q2)))
        all_2q_ops.append((q1, q2))
    for match in re.finditer(r"\bswap\s+(\w+)\[(\d+)\]\s*,\s*(\w+)\[(\d+)\]", text):
        q1, q2 = int(match.group(2)), int(match.group(4))
        qubit_pairs.add((min(q1, q2), max(q1, q2)))
        all_2q_ops.append((q1, q2))
    
    n_unique_pairs = len(qubit_pairs)
    
    if qubit_pairs:
        spans = [abs(q2 - q1) for q1, q2 in qubit_pairs]
        avg_span = np.mean(spans)
        max_span = max(spans)
        min_span = min(spans)
        span_std = np.std(spans) if len(spans) > 1 else 0.0
    else:
        avg_span = 0.0
        max_span = 0
        min_span = 0
        span_std = 0.0
    
    gate_density = n_2q_gates / max(n_qubits, 1)
    total_gates = n_2q_gates + n_1q_gates
    gate_ratio_2q = n_2q_gates / max(total_gates, 1)
    
    graph_features = _compute_graph_features(qubit_pairs, n_qubits)
    depth_features = _estimate_depth(text, n_qubits)
    cut_features = _compute_cut_features(qubit_pairs, all_2q_ops, n_qubits)
    bandwidth_features = _compute_graph_bandwidth(qubit_pairs, n_qubits)
    temporal_features = _compute_temporal_features(text, n_qubits)
    activity_features = _compute_qubit_activity_features(text, n_qubits)
    pattern_features = _compute_gate_pattern_features(text)
    lightcone_features = _compute_light_cone_features(all_2q_ops, n_qubits)
    structure_features = _compute_entanglement_structure_features(all_2q_ops, n_qubits)
    regularity_features = _compute_circuit_regularity_features(text, n_qubits)
    treewidth_features = _compute_treewidth_features(qubit_pairs, n_qubits)
    
    return {
        "n_lines": non_empty_lines,
        "n_qubits": n_qubits,
        "n_cx": n_cx,
        "n_cz": n_cz,
        "n_swap": n_swap,
        "n_ccx": n_ccx,
        "n_2q_gates": n_2q_gates,
        "n_1q_gates": n_1q_gates,
        "n_measure": n_measure,
        "n_barrier": n_barrier,
        "n_custom_gates": n_custom_gates,
        "n_unique_pairs": n_unique_pairs,
        "avg_span": avg_span,
        "max_span": max_span,
        "min_span": min_span,
        "span_std": span_std,
        "gate_density": gate_density,
        "gate_ratio_2q": gate_ratio_2q,
        "n_h": n_h,
        "n_rx": n_rx,
        "n_ry": n_ry,
        "n_rz": n_rz,
        **graph_features,
        **depth_features,
        **cut_features,
        **bandwidth_features,
        **temporal_features,
        **activity_features,
        **pattern_features,
        **lightcone_features,
        **structure_features,
        **regularity_features,
        **treewidth_features,
    }


def load_hackathon_data(data_path: Path) -> Tuple[List[CircuitInfo], List[ResultEntry]]:
    """Load the hackathon public dataset."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    circuits = []
    for c in data["circuits"]:
        circuits.append(CircuitInfo(
            file=c["file"],
            family=c["family"],
            n_qubits=c["n_qubits"],
            source_name=c.get("source", {}).get("name", ""),
            source_url=c.get("source", {}).get("url", "") or "",
        ))
    
    results = []
    for r in data["results"]:
        sweep = []
        for s in r.get("threshold_sweep", []):
            sweep.append(ThresholdSweepEntry(
                threshold=s["threshold"],
                sdk_get_fidelity=s.get("sdk_get_fidelity"),
                p_return_zero=s.get("p_return_zero"),
                run_wall_s=s.get("run_wall_s"),
                peak_rss_mb=s.get("peak_rss_mb"),
                returncode=s.get("returncode", 0),
                note=s.get("note", ""),
            ))
        
        selection = r.get("selection", {})
        forward = r.get("forward", {})
        
        results.append(ResultEntry(
            file=r["file"],
            backend=r["backend"],
            precision=r["precision"],
            status=r["status"],
            selected_threshold=selection.get("selected_threshold"),
            target_fidelity=selection.get("target"),
            threshold_sweep=sweep,
            forward_wall_s=forward.get("run_wall_s") if forward else None,
            forward_shots=forward.get("shots") if forward else None,
        ))
    
    return circuits, results


def compute_min_threshold(sweep: List[ThresholdSweepEntry], target: float = 0.99) -> Optional[int]:
    """Find the minimum threshold that meets the target fidelity."""
    for entry in sorted(sweep, key=lambda x: x.threshold):
        fid = entry.sdk_get_fidelity
        if fid is not None and fid >= target:
            return entry.threshold
    return None


class QuantumCircuitDataset(Dataset):
    """PyTorch Dataset for quantum circuit threshold/runtime prediction."""
    
    FAMILY_CATEGORIES = [
        "Amplitude_Estimation", "CutBell", "Deutsch_Jozsa", "GHZ", "GraphState",
        "Ground_State", "Grover_NoAncilla", "Grover_V_Chain", "Portfolio_QAOA",
        "Portfolio_VQE", "Pricing_Call", "QAOA", "QFT", "QFT_Entangled", "QNN",
        "QPE_Exact", "Shor", "TwoLocalRandom", "VQE", "W_State"
    ]
    
    def __init__(
        self,
        data_path: Path,
        circuits_dir: Path,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        filter_ok_only: bool = True,
    ):
        """
        Args:
            data_path: Path to hackathon_public.json
            circuits_dir: Path to circuits directory
            split: "train" or "val"
            val_fraction: Fraction of circuits for validation (split by circuit file)
            seed: Random seed for split
            filter_ok_only: Only include results with status="ok"
        """
        self.circuits_dir = circuits_dir
        self.split = split
        
        circuits, results = load_hackathon_data(data_path)
        
        self.circuit_info = {c.file: c for c in circuits}
        self.family_to_idx = {f: i for i, f in enumerate(self.FAMILY_CATEGORIES)}
        
        if filter_ok_only:
            results = [r for r in results if r.status == "ok"]
        
        circuit_files = list(set(r.file for r in results))
        rng = np.random.RandomState(seed)
        rng.shuffle(circuit_files)
        
        n_val = int(len(circuit_files) * val_fraction)
        val_files = set(circuit_files[:n_val])
        train_files = set(circuit_files[n_val:])
        
        if split == "train":
            self.results = [r for r in results if r.file in train_files]
        else:
            self.results = [r for r in results if r.file in val_files]
        
        self._feature_cache: Dict[str, Dict] = {}
    
    def __len__(self) -> int:
        return len(self.results)
    
    def _get_circuit_features(self, file: str) -> Dict[str, Any]:
        if file not in self._feature_cache:
            qasm_path = self.circuits_dir / file
            self._feature_cache[file] = extract_qasm_features(qasm_path)
        return self._feature_cache[file]
    
    def _get_threshold_label(self, result: ResultEntry) -> int:
        """Get the minimum threshold meeting fidelity target as a class index."""
        min_thresh = compute_min_threshold(result.threshold_sweep, target=0.99)
        if min_thresh is None:
            return len(THRESHOLD_LADDER) - 1
        try:
            return THRESHOLD_LADDER.index(min_thresh)
        except ValueError:
            for i, t in enumerate(THRESHOLD_LADDER):
                if t >= min_thresh:
                    return i
            return len(THRESHOLD_LADDER) - 1
    
    NUMERIC_FEATURE_KEYS = [
        # Basic gate counts
        "n_qubits", "n_lines", "n_cx", "n_cz", "n_swap", "n_ccx",
        "n_2q_gates", "n_1q_gates", "n_unique_pairs",
        "n_custom_gates", "n_measure", "n_barrier",
        "n_h", "n_rx", "n_ry", "n_rz",
        # Span features
        "avg_span", "max_span", "min_span", "span_std",
        "gate_density", "gate_ratio_2q",
        # Graph structure
        "max_degree", "avg_degree", "degree_entropy",
        "n_connected_components", "clustering_coeff", "max_component_size", "component_entropy",
        # Depth
        "estimated_depth", "depth_per_qubit",
        # Cut features
        "middle_cut_crossings", "cut_crossing_ratio", "max_cut_crossings",
        # Bandwidth (MPS complexity)
        "graph_bandwidth", "normalized_bandwidth", "bandwidth_squared",
        # Temporal features (when do entangling gates occur)
        "early_longrange_ratio", "late_longrange_ratio",
        "longrange_temporal_center", "entanglement_velocity",
        # Qubit activity distribution
        "qubit_activity_entropy", "qubit_activity_variance",
        "qubit_activity_max_ratio", "active_qubit_fraction",
        # Gate pattern features
        "cx_chain_max_length", "h_cx_pattern_count", "cx_rz_cx_pattern_count",
        "rotation_density", "gate_type_entropy", "cx_h_ratio",
        # Light cone spread
        "light_cone_spread_rate", "light_cone_half_coverage_depth", "final_light_cone_size",
        # Entanglement structure
        "nearest_neighbor_ratio", "long_range_ratio",
        "span_gini_coefficient", "weighted_span_sum",
        # Circuit regularity
        "pattern_repetition_score", "barrier_regularity", "layer_uniformity",
        # Treewidth
        "treewidth_min_degree",
    ]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = self.results[idx]
        circuit = self.circuit_info.get(result.file)
        qasm_features = self._get_circuit_features(result.file)
        
        family_idx = self.family_to_idx.get(circuit.family if circuit else "", 0)
        family_onehot = torch.zeros(len(self.FAMILY_CATEGORIES))
        family_onehot[family_idx] = 1.0
        
        backend_idx = BACKEND_MAP.get(result.backend, 0)
        precision_idx = PRECISION_MAP.get(result.precision, 0)
        
        numeric_values = [qasm_features.get(k, 0.0) for k in self.NUMERIC_FEATURE_KEYS]
        numeric_values.extend([backend_idx, precision_idx])
        numeric_features = torch.tensor(numeric_values, dtype=torch.float32)
        
        features = torch.cat([numeric_features, family_onehot])
        
        threshold_class = self._get_threshold_label(result)
        
        forward_time = result.forward_wall_s if result.forward_wall_s else 0.0
        log_runtime = np.log1p(forward_time)
        
        return {
            "features": features,
            "threshold_class": torch.tensor(threshold_class, dtype=torch.long),
            "log_runtime": torch.tensor(log_runtime, dtype=torch.float32),
            "file": result.file,
            "backend": result.backend,
            "precision": result.precision,
        }
    
    @property
    def feature_dim(self) -> int:
        return len(self.NUMERIC_FEATURE_KEYS) + 2 + len(self.FAMILY_CATEGORIES)
    
    @property
    def num_threshold_classes(self) -> int:
        return len(THRESHOLD_LADDER)


class HoldoutDataset(Dataset):
    """Dataset for holdout predictions (no labels)."""
    
    def __init__(
        self,
        holdout_path: Path,
        circuits_dir: Path,
        circuit_id_map: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            holdout_path: Path to holdout_public.json
            circuits_dir: Path to circuits directory
            circuit_id_map: Optional mapping from task ID to circuit file
        """
        self.circuits_dir = circuits_dir
        self.circuit_id_map = circuit_id_map or {}
        
        with open(holdout_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.tasks = data["tasks"]
        self._feature_cache: Dict[str, Dict] = {}
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task = self.tasks[idx]
        task_id = task["id"]
        processor = task["processor"]
        precision = task["precision"]
        
        circuit_file = self.circuit_id_map.get(task_id, "")
        
        return {
            "task_id": task_id,
            "processor": processor,
            "precision": precision,
            "circuit_file": circuit_file,
        }


def create_data_loaders(
    data_path: Path,
    circuits_dir: Path,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders with deterministic shuffling."""
    train_dataset = QuantumCircuitDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="train",
        val_fraction=val_fraction,
        seed=seed,
    )
    
    val_dataset = QuantumCircuitDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="val",
        val_fraction=val_fraction,
        seed=seed,
    )
    
    # Use a generator for deterministic shuffling
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=train_generator,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function to handle string fields."""
    features = torch.stack([item["features"] for item in batch])
    threshold_class = torch.stack([item["threshold_class"] for item in batch])
    log_runtime = torch.stack([item["log_runtime"] for item in batch])
    
    return {
        "features": features,
        "threshold_class": threshold_class,
        "log_runtime": log_runtime,
        "file": [item["file"] for item in batch],
        "backend": [item["backend"] for item in batch],
        "precision": [item["precision"] for item in batch],
    }


class KFoldQuantumCircuitDataset(Dataset):
    """K-fold cross-validation dataset that splits by circuit file."""
    
    FAMILY_CATEGORIES = QuantumCircuitDataset.FAMILY_CATEGORIES
    NUMERIC_FEATURE_KEYS = QuantumCircuitDataset.NUMERIC_FEATURE_KEYS
    
    def __init__(
        self,
        data_path: Path,
        circuits_dir: Path,
        fold: int,
        n_folds: int = 5,
        is_train: bool = True,
        seed: int = 42,
        filter_ok_only: bool = True,
    ):
        """
        Args:
            data_path: Path to hackathon_public.json
            circuits_dir: Path to circuits directory
            fold: Current fold index (0 to n_folds-1)
            n_folds: Total number of folds
            is_train: If True, use all folds except current; if False, use current fold only
            seed: Random seed for shuffling circuits before splitting
            filter_ok_only: Only include results with status="ok"
        """
        self.circuits_dir = circuits_dir
        self.fold = fold
        self.n_folds = n_folds
        self.is_train = is_train
        
        circuits, results = load_hackathon_data(data_path)
        
        self.circuit_info = {c.file: c for c in circuits}
        self.family_to_idx = {f: i for i, f in enumerate(self.FAMILY_CATEGORIES)}
        
        if filter_ok_only:
            results = [r for r in results if r.status == "ok"]
        
        # Get unique circuit files and shuffle deterministically
        circuit_files = sorted(list(set(r.file for r in results)))
        rng = np.random.RandomState(seed)
        rng.shuffle(circuit_files)
        
        # Split into k folds
        fold_size = len(circuit_files) // n_folds
        fold_starts = [i * fold_size for i in range(n_folds)]
        fold_starts.append(len(circuit_files))  # End marker
        
        # Get files for this fold
        val_files = set(circuit_files[fold_starts[fold]:fold_starts[fold + 1]])
        train_files = set(circuit_files) - val_files
        
        if is_train:
            self.results = [r for r in results if r.file in train_files]
        else:
            self.results = [r for r in results if r.file in val_files]
        
        self._feature_cache: Dict[str, Dict] = {}
    
    def __len__(self) -> int:
        return len(self.results)
    
    def _get_circuit_features(self, file: str) -> Dict[str, Any]:
        if file not in self._feature_cache:
            qasm_path = self.circuits_dir / file
            self._feature_cache[file] = extract_qasm_features(qasm_path)
        return self._feature_cache[file]
    
    def _get_threshold_label(self, result: ResultEntry) -> int:
        """Get the minimum threshold meeting fidelity target as a class index."""
        min_thresh = compute_min_threshold(result.threshold_sweep, target=0.99)
        if min_thresh is None:
            return len(THRESHOLD_LADDER) - 1
        try:
            return THRESHOLD_LADDER.index(min_thresh)
        except ValueError:
            for i, t in enumerate(THRESHOLD_LADDER):
                if t >= min_thresh:
                    return i
            return len(THRESHOLD_LADDER) - 1
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = self.results[idx]
        circuit = self.circuit_info.get(result.file)
        qasm_features = self._get_circuit_features(result.file)
        
        family_idx = self.family_to_idx.get(circuit.family if circuit else "", 0)
        family_onehot = torch.zeros(len(self.FAMILY_CATEGORIES))
        family_onehot[family_idx] = 1.0
        
        backend_idx = BACKEND_MAP.get(result.backend, 0)
        precision_idx = PRECISION_MAP.get(result.precision, 0)
        
        numeric_values = [qasm_features.get(k, 0.0) for k in self.NUMERIC_FEATURE_KEYS]
        numeric_values.extend([backend_idx, precision_idx])
        numeric_features = torch.tensor(numeric_values, dtype=torch.float32)
        
        features = torch.cat([numeric_features, family_onehot])
        
        threshold_class = self._get_threshold_label(result)
        
        forward_time = result.forward_wall_s if result.forward_wall_s else 0.0
        log_runtime = np.log1p(forward_time)
        
        return {
            "features": features,
            "threshold_class": torch.tensor(threshold_class, dtype=torch.long),
            "log_runtime": torch.tensor(log_runtime, dtype=torch.float32),
            "file": result.file,
            "backend": result.backend,
            "precision": result.precision,
        }
    
    @property
    def feature_dim(self) -> int:
        return len(self.NUMERIC_FEATURE_KEYS) + 2 + len(self.FAMILY_CATEGORIES)
    
    @property
    def num_threshold_classes(self) -> int:
        return len(THRESHOLD_LADDER)


def create_kfold_data_loaders(
    data_path: Path,
    circuits_dir: Path,
    n_folds: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader]]:
    """Create k-fold cross-validation data loaders.
    
    Returns:
        List of (train_loader, val_loader) tuples, one per fold.
    """
    fold_loaders = []
    
    for fold in range(n_folds):
        train_dataset = KFoldQuantumCircuitDataset(
            data_path=data_path,
            circuits_dir=circuits_dir,
            fold=fold,
            n_folds=n_folds,
            is_train=True,
            seed=seed,
        )
        
        val_dataset = KFoldQuantumCircuitDataset(
            data_path=data_path,
            circuits_dir=circuits_dir,
            fold=fold,
            n_folds=n_folds,
            is_train=False,
            seed=seed,
        )
        
        train_generator = torch.Generator()
        train_generator.manual_seed(seed + fold)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            generator=train_generator,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders


def get_feature_statistics(data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of features for normalization."""
    all_features = []
    for batch in data_loader:
        all_features.append(batch["features"])
    
    all_features = torch.cat(all_features, dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)
    std[std == 0] = 1.0
    
    return mean, std


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    train_loader, val_loader = create_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=16,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Features shape: {batch['features'].shape}")
    print(f"  Threshold classes: {batch['threshold_class']}")
    print(f"  Log runtimes: {batch['log_runtime']}")
    print(f"  Files: {batch['file'][:3]}...")
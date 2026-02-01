"""
QASM file feature extraction for quantum circuit analysis.

This module provides functions to extract various structural and statistical
features from QASM circuit files for use in machine learning models.
"""

import re
from pathlib import Path
from typing import Dict, Any
from collections import Counter

import numpy as np


def compute_graph_features(qubit_pairs: set, n_qubits: int) -> Dict[str, float]:
    """Compute interaction graph features from 2-qubit gate pairs."""
    if not qubit_pairs or n_qubits == 0:
        return {
            "max_degree": 0,
            "avg_degree": 0.0,
            "degree_entropy": 0.0,
            "n_connected_components": 0,
            "clustering_coeff": 0.0,
            "max_component_size": 1.0 if n_qubits > 0 else 0.0,
            "component_entropy": 0.0,
        }
    
    degree = [0] * n_qubits
    adjacency = {i: set() for i in range(n_qubits)}
    
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


def estimate_depth(text: str, n_qubits: int) -> Dict[str, float]:
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


def compute_cut_features(qubit_pairs: set, all_2q_ops: list, n_qubits: int) -> Dict[str, float]:
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


def compute_graph_bandwidth(qubit_pairs: set, n_qubits: int) -> Dict[str, float]:
    """
    Compute graph bandwidth - a key metric for MPS simulation complexity.
    
    Bandwidth measures how "spread out" the interaction graph is along the 1D qubit chain.
    Lower bandwidth = easier for MPS.
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


def compute_temporal_features(text: str, n_qubits: int) -> Dict[str, float]:
    """
    Analyze temporal structure of gate placement.
    
    Key insight: Long-range gates early in the circuit cause entanglement to persist
    and accumulate.
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
    threshold_span = n_qubits // 4
    
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


def compute_qubit_activity_features(text: str, n_qubits: int) -> Dict[str, float]:
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


def compute_gate_pattern_features(text: str) -> Dict[str, float]:
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


def compute_light_cone_features(all_2q_ops: list, n_qubits: int) -> Dict[str, float]:
    """
    Compute light cone spread - how quickly information propagates across qubits.
    
    Fast light cone expansion = rapid entanglement growth = harder for MPS.
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


def compute_entanglement_structure_features(all_2q_ops: list, n_qubits: int) -> Dict[str, float]:
    """
    Analyze entanglement structure patterns.
    
    Key features:
    - Nearest-neighbor ratio: High = linear/local structure (easier for MPS)
    - All-to-all ratio: High = highly connected (harder for MPS)
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
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_spans))) / (n * cumulative[-1]) - (n + 1) / n
    gini = max(0, min(1, gini))
    
    weights = np.array(spans) ** 2
    weighted_sum = np.sum(weights) / (n_qubits ** 2) if n_qubits > 0 else 0.0
    
    return {
        "nearest_neighbor_ratio": nn_ratio,
        "long_range_ratio": long_ratio,
        "span_gini_coefficient": gini,
        "weighted_span_sum": weighted_sum,
    }


def compute_circuit_regularity_features(text: str, n_qubits: int) -> Dict[str, float]:
    """
    Detect circuit regularity and repetition patterns.
    
    Regular/repetitive circuits often have predictable entanglement behavior.
    """
    gate_pattern = re.compile(r"\b(cx|cz|h|rx|ry|rz|swap)\s+\w+\[(\d+)\]")
    
    gate_sequence = []
    for match in gate_pattern.finditer(text):
        gate_type = match.group(1)
        qubit = int(match.group(2))
        gate_sequence.append((gate_type, qubit % 4))
    
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
        "layer_uniformity": 0.5,
    }


def compute_treewidth_features(qubit_pairs: set, n_qubits: int) -> Dict[str, float]:
    """
    Estimate treewidth using the Min-Degree heuristic.
    
    Treewidth is a key parameter for tensor network contraction complexity.
    """
    if n_qubits == 0:
        return {"treewidth_min_degree": 0.0}
    
    adj = {i: set() for i in range(n_qubits)}
    for q1, q2 in qubit_pairs:
        if q1 < n_qubits and q2 < n_qubits:
            adj[q1].add(q2)
            adj[q2].add(q1)
            
    max_degree_at_elimination = 0
    active_nodes = set(range(n_qubits))
    
    for _ in range(n_qubits):
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
        
        neighbors = list(adj[node_to_eliminate])
        for i in range(len(neighbors)):
            u = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                v = neighbors[j]
                if v not in adj[u]:
                    adj[u].add(v)
                    adj[v].add(u)
        
        active_nodes.remove(node_to_eliminate)
        for u in neighbors:
            adj[u].remove(node_to_eliminate)
            
    return {"treewidth_min_degree": float(max_degree_at_elimination)}


def extract_qasm_features(qasm_path: Path) -> Dict[str, Any]:
    """Extract all features from a QASM file for model input."""
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
    
    graph_features = compute_graph_features(qubit_pairs, n_qubits)
    depth_features = estimate_depth(text, n_qubits)
    cut_features = compute_cut_features(qubit_pairs, all_2q_ops, n_qubits)
    bandwidth_features = compute_graph_bandwidth(qubit_pairs, n_qubits)
    temporal_features = compute_temporal_features(text, n_qubits)
    activity_features = compute_qubit_activity_features(text, n_qubits)
    pattern_features = compute_gate_pattern_features(text)
    lightcone_features = compute_light_cone_features(all_2q_ops, n_qubits)
    structure_features = compute_entanglement_structure_features(all_2q_ops, n_qubits)
    regularity_features = compute_circuit_regularity_features(text, n_qubits)
    treewidth_features = compute_treewidth_features(qubit_pairs, n_qubits)
    
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

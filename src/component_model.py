"""
Component-Based Cost Model for quantum circuit threshold/runtime prediction.

Instead of treating the circuit as a black box, this model:
1. Parses the circuit into individual gate operations
2. Assigns learned costs to each gate based on type, span, position
3. Aggregates costs to predict threshold and runtime

Key insight: MPS simulation cost is driven by:
- Gate type (2Q gates are expensive, 1Q gates are cheap)
- Gate span (long-range gates require more bond dimension)
- Gate position (early entangling gates compound)
- Cut crossings (gates crossing the middle are expensive)
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]


@dataclass
class GateInfo:
    """Information about a single gate operation."""
    gate_type: str
    qubits: List[int]
    position: int  # sequential position in circuit
    span: int  # max qubit distance (0 for 1Q gates)
    crosses_middle: bool  # does it cross n_qubits/2?
    layer: int  # estimated layer/depth


def parse_circuit_gates(qasm_text: str, n_qubits: int) -> List[GateInfo]:
    """Parse a QASM circuit into a list of gate operations."""
    gates = []
    position = 0
    qubit_depth = [0] * max(n_qubits, 1)
    middle = n_qubits // 2 if n_qubits > 0 else 0
    
    gate_2q = re.compile(r"\b(cx|cz|swap)\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]")
    gate_1q = re.compile(r"\b(h|x|y|z|s|sdg|t|tdg|rx|ry|rz|u1|u2|u3)\s+\w+\[(\d+)\]")
    gate_3q = re.compile(r"\b(ccx|cswap)\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]")
    
    for match in gate_2q.finditer(qasm_text):
        gate_type = match.group(1)
        q1, q2 = int(match.group(2)), int(match.group(3))
        qubits = [q1, q2]
        span = abs(q2 - q1)
        crosses = (min(q1, q2) < middle <= max(q1, q2)) if n_qubits > 1 else False
        
        layer = max(qubit_depth[q] for q in qubits if q < len(qubit_depth))
        for q in qubits:
            if q < len(qubit_depth):
                qubit_depth[q] = layer + 1
        
        gates.append(GateInfo(
            gate_type=gate_type,
            qubits=qubits,
            position=position,
            span=span,
            crosses_middle=crosses,
            layer=layer,
        ))
        position += 1
    
    for match in gate_1q.finditer(qasm_text):
        gate_type = match.group(1)
        q = int(match.group(2))
        
        layer = qubit_depth[q] if q < len(qubit_depth) else 0
        if q < len(qubit_depth):
            qubit_depth[q] = layer + 1
        
        gates.append(GateInfo(
            gate_type=gate_type,
            qubits=[q],
            position=position,
            span=0,
            crosses_middle=False,
            layer=layer,
        ))
        position += 1
    
    for match in gate_3q.finditer(qasm_text):
        gate_type = match.group(1)
        q1, q2, q3 = int(match.group(2)), int(match.group(3)), int(match.group(4))
        qubits = [q1, q2, q3]
        span = max(qubits) - min(qubits)
        crosses = (min(qubits) < middle <= max(qubits)) if n_qubits > 1 else False
        
        layer = max(qubit_depth[q] for q in qubits if q < len(qubit_depth))
        for q in qubits:
            if q < len(qubit_depth):
                qubit_depth[q] = layer + 1
        
        gates.append(GateInfo(
            gate_type=gate_type,
            qubits=qubits,
            position=position,
            span=span,
            crosses_middle=crosses,
            layer=layer,
        ))
        position += 1
    
    gates.sort(key=lambda g: g.position)
    return gates


class AnalyticalCostModel:
    """
    Analytical model that assigns costs to gates based on physical intuition.
    
    Cost factors:
    1. Base cost per gate type (2Q > 1Q)
    2. Span penalty (long-range gates are expensive)
    3. Cut crossing penalty (gates crossing middle increase bond dim)
    4. Early gate multiplier (early entanglement compounds)
    5. Position decay (later gates have less marginal impact)
    """
    
    def __init__(self, calibrated: bool = True):
        self.calibrated = calibrated
        
        if calibrated:
            self.gate_base_cost = {
                "cx": 0.15,
                "cz": 0.15,
                "swap": 0.25,
                "ccx": 0.4,
                "cswap": 0.5,
                "h": 0.001,
                "x": 0.001,
                "y": 0.001,
                "z": 0.001,
                "s": 0.001,
                "sdg": 0.001,
                "t": 0.001,
                "tdg": 0.001,
                "rx": 0.002,
                "ry": 0.002,
                "rz": 0.002,
                "u1": 0.002,
                "u2": 0.002,
                "u3": 0.003,
            }
            self.span_exponent = 1.2
            self.cut_crossing_multiplier = 1.3
            self.early_gate_boost = 1.2
            self.early_fraction = 0.25
            self.threshold_boundaries = [1.0, 4.0, 12.0, 35.0, 100.0, 300.0, 800.0, 2000.0]
        else:
            self.gate_base_cost = {
                "cx": 1.0, "cz": 1.0, "swap": 1.5, "ccx": 2.0, "cswap": 2.5,
                "h": 0.01, "x": 0.01, "y": 0.01, "z": 0.01, "s": 0.01,
                "sdg": 0.01, "t": 0.01, "tdg": 0.01, "rx": 0.02, "ry": 0.02,
                "rz": 0.02, "u1": 0.02, "u2": 0.02, "u3": 0.03,
            }
            self.span_exponent = 1.5
            self.cut_crossing_multiplier = 2.0
            self.early_gate_boost = 1.5
            self.early_fraction = 0.3
            self.threshold_boundaries = [0.5, 2, 8, 25, 80, 200, 500, 1000]
    
    def compute_gate_cost(self, gate: GateInfo, n_qubits: int, total_gates: int) -> float:
        """Compute the cost of a single gate."""
        base = self.gate_base_cost.get(gate.gate_type, 0.1)
        
        if gate.span > 0 and n_qubits > 1:
            normalized_span = gate.span / n_qubits
            span_penalty = 1.0 + (normalized_span ** self.span_exponent) * 3.0
        else:
            span_penalty = 1.0
        
        cut_penalty = self.cut_crossing_multiplier if gate.crosses_middle else 1.0
        
        if total_gates > 0:
            relative_position = gate.position / total_gates
            if relative_position < self.early_fraction:
                position_factor = self.early_gate_boost
            else:
                position_factor = 1.0 - 0.2 * (relative_position - self.early_fraction)
                position_factor = max(0.6, position_factor)
        else:
            position_factor = 1.0
        
        return base * span_penalty * cut_penalty * position_factor
    
    def compute_circuit_cost(self, gates: List[GateInfo], n_qubits: int) -> Dict[str, float]:
        """Compute aggregate costs for the entire circuit."""
        if not gates:
            return {
                "total_cost": 0.0,
                "entanglement_cost": 0.0,
                "max_layer_cost": 0.0,
                "cut_crossing_cost": 0.0,
            }
        
        total_gates = len(gates)
        gate_costs = [self.compute_gate_cost(g, n_qubits, total_gates) for g in gates]
        
        total_cost = sum(gate_costs)
        
        entanglement_cost = sum(
            cost for g, cost in zip(gates, gate_costs) 
            if g.gate_type in ("cx", "cz", "swap", "ccx", "cswap")
        )
        
        layer_costs = {}
        for g, cost in zip(gates, gate_costs):
            if g.layer not in layer_costs:
                layer_costs[g.layer] = 0.0
            layer_costs[g.layer] += cost
        max_layer_cost = max(layer_costs.values()) if layer_costs else 0.0
        
        cut_crossing_cost = sum(
            cost for g, cost in zip(gates, gate_costs) if g.crosses_middle
        )
        
        return {
            "total_cost": total_cost,
            "entanglement_cost": entanglement_cost,
            "max_layer_cost": max_layer_cost,
            "cut_crossing_cost": cut_crossing_cost,
        }
    
    def predict_threshold_index(self, costs: Dict[str, float]) -> int:
        """Map costs to a threshold index (0-8)."""
        total = costs["total_cost"]
        
        for i, thresh in enumerate(self.threshold_boundaries):
            if total < thresh:
                return i
        return len(THRESHOLD_LADDER) - 1
    
    def predict_runtime(self, costs: Dict[str, float], backend: str, precision: str) -> float:
        """Predict runtime in seconds based on costs."""
        base_runtime = 0.01 + costs["total_cost"] * 0.001
        
        backend_factor = 0.3 if backend == "GPU" else 1.0
        precision_factor = 1.5 if precision == "double" else 1.0
        
        return base_runtime * backend_factor * precision_factor
    
    def calibrate(self, circuits_data: List[Dict]):
        """
        Auto-calibrate threshold boundaries from training data.
        
        circuits_data: list of dicts with 'costs' and 'true_threshold_idx'
        """
        if not circuits_data:
            return
        
        costs = np.array([d["costs"]["total_cost"] for d in circuits_data])
        thresholds = np.array([d["true_threshold_idx"] for d in circuits_data])
        
        new_boundaries = []
        for i in range(len(THRESHOLD_LADDER) - 1):
            mask_low = thresholds <= i
            mask_high = thresholds > i
            
            if mask_low.sum() > 0 and mask_high.sum() > 0:
                max_low = costs[mask_low].max()
                min_high = costs[mask_high].min()
                boundary = (max_low + min_high) / 2
            elif mask_low.sum() > 0:
                boundary = costs[mask_low].max() * 1.5
            else:
                boundary = self.threshold_boundaries[i]
            
            new_boundaries.append(boundary)
        
        self.threshold_boundaries = new_boundaries
        self.calibrated = True


class BondDimensionTracker:
    """
    Track estimated bond dimension growth through the circuit.
    
    Key insight: MPS bond dimension grows when entangling gates cross cuts.
    We simulate this cheaply to estimate the max bond dimension needed.
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.bond_dims = [1] * (n_qubits - 1) if n_qubits > 1 else [1]
    
    def apply_gate(self, gate: GateInfo):
        """Update bond dimensions after applying a gate."""
        if gate.span == 0 or self.n_qubits < 2:
            return
        
        q_min, q_max = min(gate.qubits), max(gate.qubits)
        
        for cut in range(q_min, min(q_max, len(self.bond_dims))):
            growth_factor = 1.5 + 0.5 * (gate.span / self.n_qubits)
            self.bond_dims[cut] = min(
                self.bond_dims[cut] * growth_factor,
                2 ** min(cut + 1, self.n_qubits - cut - 1)
            )
    
    def get_max_bond_dim(self) -> float:
        return max(self.bond_dims)
    
    def get_middle_bond_dim(self) -> float:
        if not self.bond_dims:
            return 1.0
        middle = len(self.bond_dims) // 2
        return self.bond_dims[middle]


class EntanglementBudgetModel:
    """
    Model fidelity degradation as an entanglement budget.
    
    Each gate "spends" from the fidelity budget based on:
    - How much entanglement it creates
    - Current threshold (higher threshold = larger budget)
    
    When budget is depleted, fidelity drops below target.
    """
    
    def __init__(self):
        self.threshold_budgets = {
            1: 5.0,
            2: 10.0,
            4: 25.0,
            8: 60.0,
            16: 150.0,
            32: 400.0,
            64: 1000.0,
            128: 2500.0,
            256: 6000.0,
        }
    
    def compute_gate_fidelity_cost(self, gate: GateInfo, n_qubits: int) -> float:
        """How much fidelity budget does this gate consume?"""
        if gate.span == 0:
            return 0.01
        
        base_cost = 1.0 if gate.gate_type in ("cx", "cz") else 1.5
        span_factor = 1.0 + (gate.span / max(n_qubits, 1)) ** 2
        cross_factor = 2.0 if gate.crosses_middle else 1.0
        
        return base_cost * span_factor * cross_factor
    
    def find_minimum_threshold(self, gates: List[GateInfo], n_qubits: int) -> int:
        """Find the minimum threshold that can handle this circuit."""
        total_cost = sum(self.compute_gate_fidelity_cost(g, n_qubits) for g in gates)
        
        for threshold in THRESHOLD_LADDER:
            if total_cost <= self.threshold_budgets[threshold]:
                return threshold
        
        return THRESHOLD_LADDER[-1]


class LearnedComponentModel:
    """
    Learnable version of the component model using simple regression.
    
    Learns optimal weights for:
    - Gate type costs
    - Span penalty exponent
    - Position weighting
    - Aggregation method
    """
    
    def __init__(self):
        self.gate_weights = None
        self.span_weight = 1.0
        self.cross_weight = 1.0
        self.position_weight = 1.0
        self.threshold_boundaries = None
        self.runtime_coeffs = None
        self.fitted = False
    
    def extract_component_features(
        self, 
        gates: List[GateInfo], 
        n_qubits: int,
        backend: str = "CPU",
        precision: str = "single",
    ) -> np.ndarray:
        """
        Extract features based on gate components.
        
        Includes backend/precision for runtime prediction.
        """
        if not gates:
            return np.zeros(24)
        
        n_gates = len(gates)
        
        gate_type_counts = {}
        for g in gates:
            gate_type_counts[g.gate_type] = gate_type_counts.get(g.gate_type, 0) + 1
        
        n_cx = gate_type_counts.get("cx", 0)
        n_cz = gate_type_counts.get("cz", 0)
        n_swap = gate_type_counts.get("swap", 0)
        n_1q = sum(gate_type_counts.get(g, 0) for g in 
                   ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "u1", "u2", "u3"])
        
        spans = [g.span for g in gates if g.span > 0]
        if spans:
            avg_span = np.mean(spans)
            max_span = max(spans)
            span_sum = sum(spans)
            span_squared_sum = sum(s**2 for s in spans)
        else:
            avg_span = max_span = span_sum = span_squared_sum = 0
        
        n_crossing = sum(1 for g in gates if g.crosses_middle)
        crossing_ratio = n_crossing / n_gates if n_gates > 0 else 0
        
        early_entangling = sum(
            1 for g in gates 
            if g.position < n_gates * 0.3 and g.span > 0
        )
        late_entangling = sum(
            1 for g in gates 
            if g.position > n_gates * 0.7 and g.span > 0
        )
        
        max_layer = max(g.layer for g in gates) if gates else 0
        
        weighted_span = sum(
            g.span * (1 + g.position / n_gates) 
            for g in gates if g.span > 0
        ) if n_gates > 0 else 0
        
        is_gpu = 1.0 if backend == "GPU" else 0.0
        is_double = 1.0 if precision == "double" else 0.0
        
        n_2q = n_cx + n_cz + n_swap
        complexity_proxy = n_2q * max_span * max_layer / max(n_qubits, 1)
        
        return np.array([
            n_qubits,
            n_gates,
            n_cx,
            n_cz,
            n_swap,
            n_1q,
            avg_span,
            max_span,
            span_sum,
            span_squared_sum,
            n_crossing,
            crossing_ratio,
            early_entangling,
            late_entangling,
            max_layer,
            weighted_span,
            n_2q,
            n_2q / max(n_qubits, 1),
            max_span / max(n_qubits, 1) if n_qubits > 0 else 0,
            span_squared_sum / max(n_qubits**2, 1),
            is_gpu,
            is_double,
            complexity_proxy,
            np.log1p(n_gates),
        ], dtype=np.float32)
    
    def fit(self, circuits_data: List[Dict], y_threshold: np.ndarray, y_runtime: np.ndarray):
        """Fit the model using least squares on component features."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        X = np.array([d["features"] for d in circuits_data])
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.threshold_model = Ridge(alpha=1.0)
        self.threshold_model.fit(X_scaled, y_threshold)
        
        self.runtime_model = Ridge(alpha=1.0)
        self.runtime_model.fit(X_scaled, np.log1p(y_runtime))
        
        self.fitted = True
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict threshold index and runtime."""
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        
        X = features.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        thresh_pred = self.threshold_model.predict(X_scaled)[0]
        thresh_idx = int(np.clip(np.round(thresh_pred), 0, len(THRESHOLD_LADDER) - 1))
        
        runtime_log = self.runtime_model.predict(X_scaled)[0]
        runtime = np.expm1(runtime_log)
        
        return thresh_idx, runtime


def analyze_circuit_components(qasm_path: Path) -> Dict[str, Any]:
    """
    Full component analysis of a circuit.
    
    Returns multiple cost estimates from different models.
    """
    text = qasm_path.read_text(encoding="utf-8")
    
    qreg_match = re.search(r"qreg\s+\w+\[(\d+)\]", text)
    n_qubits = int(qreg_match.group(1)) if qreg_match else 0
    
    gates = parse_circuit_gates(text, n_qubits)
    
    analytical = AnalyticalCostModel()
    costs = analytical.compute_circuit_cost(gates, n_qubits)
    predicted_threshold_idx = analytical.predict_threshold_index(costs)
    
    tracker = BondDimensionTracker(n_qubits)
    for gate in gates:
        tracker.apply_gate(gate)
    max_bond_dim = tracker.get_max_bond_dim()
    
    budget = EntanglementBudgetModel()
    min_threshold = budget.find_minimum_threshold(gates, n_qubits)
    
    learnable = LearnedComponentModel()
    component_features = learnable.extract_component_features(gates, n_qubits)
    
    return {
        "n_qubits": n_qubits,
        "n_gates": len(gates),
        "analytical_costs": costs,
        "predicted_threshold_idx": predicted_threshold_idx,
        "predicted_threshold": THRESHOLD_LADDER[predicted_threshold_idx],
        "max_bond_dim_estimate": max_bond_dim,
        "budget_min_threshold": min_threshold,
        "component_features": component_features,
        "gates": gates,
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        qasm_path = Path(sys.argv[1])
    else:
        project_root = Path(__file__).parent.parent
        circuits = list((project_root / "circuits").glob("*.qasm"))
        if circuits:
            qasm_path = circuits[0]
        else:
            print("No circuits found")
            sys.exit(1)
    
    print(f"Analyzing: {qasm_path.name}")
    print("=" * 60)
    
    analysis = analyze_circuit_components(qasm_path)
    
    print(f"Qubits: {analysis['n_qubits']}")
    print(f"Gates: {analysis['n_gates']}")
    print()
    
    print("Analytical Cost Model:")
    for k, v in analysis['analytical_costs'].items():
        print(f"  {k}: {v:.4f}")
    print(f"  Predicted threshold: {analysis['predicted_threshold']}")
    print()
    
    print(f"Bond Dimension Tracker:")
    print(f"  Max bond dim estimate: {analysis['max_bond_dim_estimate']:.2f}")
    print()
    
    print(f"Entanglement Budget Model:")
    print(f"  Minimum threshold: {analysis['budget_min_threshold']}")
    print()
    
    print("Component Features (first 10):")
    for i, v in enumerate(analysis['component_features'][:10]):
        print(f"  [{i}]: {v:.4f}")

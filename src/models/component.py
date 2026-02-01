"""
Component-based models implementing the BaseModel interface.

Wraps the analytical/physics-based models to work with the standard
evaluation infrastructure (cross-validation, multiple runs, etc.).
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader

from .base import BaseModel
from component_model import (
    parse_circuit_gates,
    AnalyticalCostModel as _AnalyticalCostModel,
    BondDimensionTracker,
    EntanglementBudgetModel as _EntanglementBudgetModel,
    LearnedComponentModel as _LearnedComponentModel,
    THRESHOLD_LADDER,
)


def _extract_qasm_text_and_qubits(loader: DataLoader) -> List[Dict]:
    """Extract QASM file contents and metadata. Uses threshold and log2_runtime from dataset."""
    dataset = loader.dataset
    circuits_dir = dataset.circuits_dir
    data = []
    for i in range(len(dataset)):
        item = dataset[i]
        file = item["file"]
        qasm_path = circuits_dir / file
        if qasm_path.exists():
            text = qasm_path.read_text()
            qreg_match = re.search(r"qreg\s+\w+\[(\d+)\]", text)
            n_qubits = int(qreg_match.group(1)) if qreg_match else 0
        else:
            text = ""
            n_qubits = 0
        data.append({
            "file": file,
            "text": text,
            "n_qubits": n_qubits,
            "features": item["features"].numpy(),
            "threshold": item["threshold"],
            "log2_runtime": item["log2_runtime"].item(),
            "backend": item["backend"],
            "precision": item["precision"],
        })
    return data


class AnalyticalCostModelWrapper(BaseModel):
    """
    Wrapper for AnalyticalCostModel implementing BaseModel interface.
    
    Uses physics-based gate costs with data-driven threshold boundary calibration.
    Uses learned regression for runtime prediction.
    """
    
    def __init__(self, calibrate: bool = True, **kwargs):
        self.calibrate = calibrate
        self.model = _AnalyticalCostModel(calibrated=not calibrate)
        self.runtime_model = None
        self.runtime_scaler = None
        self._train_data = None
    
    @property
    def name(self) -> str:
        suffix = "(calibrated)" if self.calibrate else "(fixed)"
        return f"AnalyticalCost {suffix}"
    
    def _extract_cost_features(self, costs: Dict, backend: str, precision: str, log2_threshold: float) -> np.ndarray:
        """Extract features from cost dict for runtime regression (includes log2_threshold as input)."""
        is_gpu = 1.0 if backend == "GPU" else 0.0
        is_double = 1.0 if precision == "double" else 0.0
        return np.array([
            costs["total_cost"],
            costs["entanglement_cost"],
            costs["max_layer_cost"],
            costs["cut_crossing_cost"],
            np.log1p(costs["total_cost"]),
            is_gpu,
            is_double,
            costs["total_cost"] * (1.5 if is_double else 1.0),
            costs["total_cost"] * (0.3 if is_gpu else 1.0),
            log2_threshold,
        ], dtype=np.float32)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        train_data = _extract_qasm_text_and_qubits(train_loader)
        self._train_data = train_data
        
        calibration_data = []
        runtime_features = []
        runtime_targets = []
        
        for d in train_data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            costs = self.model.compute_circuit_cost(gates, d["n_qubits"])
            thresh = d["threshold"]
            log2_thresh = np.log2(max(thresh, 1))
            calibration_data.append({
                "costs": costs,
                "true_threshold_idx": THRESHOLD_LADDER.index(thresh) if thresh in THRESHOLD_LADDER else 0,
            })
            cost_features = self._extract_cost_features(costs, d["backend"], d["precision"], log2_thresh)
            runtime_features.append(cost_features)
            runtime_targets.append(d["log2_runtime"])
        
        if self.calibrate:
            self.model.calibrate(calibration_data)
        
        X_runtime = np.array(runtime_features)
        y_runtime = np.array(runtime_targets)
        self.runtime_scaler = StandardScaler()
        X_scaled = self.runtime_scaler.fit_transform(X_runtime)
        self.runtime_model = Ridge(alpha=1.0)
        self.runtime_model.fit(X_scaled, y_runtime)
        return {"calibrated": self.calibrate, "n_train": len(train_data)}
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "AnalyticalCostModel requires QASM text, not features. "
            "Use predict_from_loader instead."
        )
    
    def predict_from_loader(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Predict: threshold from input, runtime from model (log2 target)."""
        data = _extract_qasm_text_and_qubits(loader)
        thresh_preds = []
        runtime_preds = []
        for d in data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            costs = self.model.compute_circuit_cost(gates, d["n_qubits"])
            threshold = d["threshold"]
            log2_thresh = np.log2(max(threshold, 1))
            cost_features = self._extract_cost_features(costs, d["backend"], d["precision"], log2_thresh)
            X_scaled = self.runtime_scaler.transform(cost_features.reshape(1, -1))
            pred_log2_runtime = self.runtime_model.predict(X_scaled)[0]
            pred_runtime = max(np.power(2.0, pred_log2_runtime), 1e-10)
            thresh_preds.append(threshold)
            runtime_preds.append(pred_runtime)
        return np.array(thresh_preds), np.array(runtime_preds)
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        data = _extract_qasm_text_and_qubits(loader)
        true_log2 = []
        pred_log2 = []
        for d in data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            costs = self.model.compute_circuit_cost(gates, d["n_qubits"])
            log2_thresh = np.log2(max(d["threshold"], 1))
            cost_features = self._extract_cost_features(costs, d["backend"], d["precision"], log2_thresh)
            X_scaled = self.runtime_scaler.transform(cost_features.reshape(1, -1))
            pred_log2.append(self.runtime_model.predict(X_scaled)[0])
            true_log2.append(d["log2_runtime"])
        true_log2 = np.array(true_log2)
        pred_log2 = np.array(pred_log2)
        return {
            "runtime_mse": float(np.mean((true_log2 - pred_log2) ** 2)),
            "runtime_mae": float(np.mean(np.abs(true_log2 - pred_log2))),
        }


class LearnedComponentModelWrapper(BaseModel):
    """
    Wrapper for LearnedComponentModel: threshold as input, predict log2(duration).
    Uses component features + log2(threshold) for runtime regression only.
    """

    def __init__(self, **kwargs):
        self.model = _LearnedComponentModel()
        self.runtime_model = None
        self.runtime_scaler = None

    @property
    def name(self) -> str:
        return "LearnedComponent"

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        train_data = _extract_qasm_text_and_qubits(train_loader)
        X_list = []
        y_list = []
        for d in train_data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            feats = self.model.extract_component_features(
                gates, d["n_qubits"], d["backend"], d["precision"]
            )
            log2_thresh = np.log2(max(d["threshold"], 1))
            X_list.append(np.concatenate([feats, [log2_thresh]]))
            y_list.append(d["log2_runtime"])
        X = np.array(X_list)
        y = np.array(y_list)
        self.runtime_scaler = StandardScaler()
        X_scaled = self.runtime_scaler.fit_transform(X)
        self.runtime_model = Ridge(alpha=1.0)
        self.runtime_model.fit(X_scaled, y)
        return {"n_train_samples": len(train_data)}

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "LearnedComponentModel requires QASM text. Use predict_from_loader instead."
        )

    def predict_from_loader(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Predict: threshold from input, runtime from model."""
        data = _extract_qasm_text_and_qubits(loader)
        thresh_preds = []
        runtime_preds = []
        for d in data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            feats = self.model.extract_component_features(
                gates, d["n_qubits"], d["backend"], d["precision"]
            )
            log2_thresh = np.log2(max(d["threshold"], 1))
            X = np.concatenate([feats, [log2_thresh]]).reshape(1, -1)
            X_scaled = self.runtime_scaler.transform(X)
            pred_log2 = self.runtime_model.predict(X_scaled)[0]
            thresh_preds.append(d["threshold"])
            runtime_preds.append(max(np.power(2.0, pred_log2), 1e-10))
        return np.array(thresh_preds), np.array(runtime_preds)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        data = _extract_qasm_text_and_qubits(loader)
        true_log2 = []
        pred_log2 = []
        for d in data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            feats = self.model.extract_component_features(
                gates, d["n_qubits"], d["backend"], d["precision"]
            )
            log2_thresh = np.log2(max(d["threshold"], 1))
            X = np.concatenate([feats, [log2_thresh]]).reshape(1, -1)
            X_scaled = self.runtime_scaler.transform(X)
            pred_log2.append(self.runtime_model.predict(X_scaled)[0])
            true_log2.append(d["log2_runtime"])
        true_log2 = np.array(true_log2)
        pred_log2 = np.array(pred_log2)
        return {
            "runtime_mse": float(np.mean((true_log2 - pred_log2) ** 2)),
            "runtime_mae": float(np.mean(np.abs(true_log2 - pred_log2))),
        }


class BondDimensionModelWrapper(BaseModel):
    """
    Wrapper for BondDimensionTracker implementing BaseModel interface.
    
    Physics-based model that simulates bond dimension growth through the circuit.
    Uses learned regression for runtime prediction.
    """
    
    def __init__(self, **kwargs):
        self._threshold_boundaries = None
        self.runtime_model = None
        self.runtime_scaler = None
    
    @property
    def name(self) -> str:
        return "BondDimension"
    
    def _extract_bd_features(self, gates, tracker, backend: str, precision: str, log2_threshold: float) -> np.ndarray:
        """Extract features for runtime prediction (includes log2_threshold as input)."""
        max_bd = min(float(tracker.get_max_bond_dim()), 1e6)
        mid_bd = min(float(tracker.get_middle_bond_dim()), 1e6)
        n_gates = float(len(gates))
        n_2q = float(sum(1 for g in gates if g.span > 0))
        is_gpu = 1.0 if backend == "GPU" else 0.0
        is_double = 1.0 if precision == "double" else 0.0
        return np.array([
            np.log1p(max_bd),
            np.log1p(mid_bd),
            np.log1p(n_gates),
            np.log1p(n_2q),
            np.log1p(max_bd * n_gates / 1000),
            is_gpu,
            is_double,
            log2_threshold,
        ], dtype=np.float32)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        train_data = _extract_qasm_text_and_qubits(train_loader)
        
        bd_to_thresh = {}
        runtime_features = []
        runtime_targets = []
        
        for d in train_data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            tracker = BondDimensionTracker(d["n_qubits"])
            for gate in gates:
                tracker.apply_gate(gate)
            max_bd = tracker.get_max_bond_dim()
            thresh = d["threshold"]
            true_idx = THRESHOLD_LADDER.index(thresh) if thresh in THRESHOLD_LADDER else 0
            if max_bd not in bd_to_thresh:
                bd_to_thresh[max_bd] = []
            bd_to_thresh[max_bd].append(true_idx)
            log2_thresh = np.log2(max(thresh, 1))
            features = self._extract_bd_features(gates, tracker, d["backend"], d["precision"], log2_thresh)
            runtime_features.append(features)
            runtime_targets.append(d["log2_runtime"])
        
        self._threshold_boundaries = []
        for i in range(len(THRESHOLD_LADDER) - 1):
            matching_bds = [bd for bd, indices in bd_to_thresh.items() 
                           if np.mean(indices) <= i + 0.5]
            if matching_bds:
                self._threshold_boundaries.append(max(matching_bds))
            else:
                self._threshold_boundaries.append(2 ** (i + 1))
        
        X = np.array(runtime_features)
        y = np.array(runtime_targets)
        
        self.runtime_scaler = StandardScaler()
        X_scaled = self.runtime_scaler.fit_transform(X)
        
        self.runtime_model = Ridge(alpha=1.0)
        self.runtime_model.fit(X_scaled, y)
        
        return {"n_unique_bond_dims": len(bd_to_thresh)}
    
    def _bd_to_threshold_idx(self, max_bd: float) -> int:
        for i, boundary in enumerate(self._threshold_boundaries):
            if max_bd <= boundary:
                return i
        return len(THRESHOLD_LADDER) - 1
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "BondDimensionModel requires QASM text, not features. "
            "Use predict_from_loader instead."
        )
    
    def predict_from_loader(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        data = _extract_qasm_text_and_qubits(loader)
        thresh_preds = []
        runtime_preds = []
        for d in data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            tracker = BondDimensionTracker(d["n_qubits"])
            for gate in gates:
                tracker.apply_gate(gate)
            threshold = d["threshold"]
            log2_thresh = np.log2(max(threshold, 1))
            features = self._extract_bd_features(gates, tracker, d["backend"], d["precision"], log2_thresh)
            X_scaled = self.runtime_scaler.transform(features.reshape(1, -1))
            pred_log2 = self.runtime_model.predict(X_scaled)[0]
            thresh_preds.append(threshold)
            runtime_preds.append(max(np.power(2.0, pred_log2), 1e-10))
        return np.array(thresh_preds), np.array(runtime_preds)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        data = _extract_qasm_text_and_qubits(loader)
        true_log2 = []
        pred_log2 = []
        for d in data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            tracker = BondDimensionTracker(d["n_qubits"])
            for gate in gates:
                tracker.apply_gate(gate)
            log2_thresh = np.log2(max(d["threshold"], 1))
            features = self._extract_bd_features(gates, tracker, d["backend"], d["precision"], log2_thresh)
            X_scaled = self.runtime_scaler.transform(features.reshape(1, -1))
            pred_log2.append(self.runtime_model.predict(X_scaled)[0])
            true_log2.append(d["log2_runtime"])
        true_log2 = np.array(true_log2)
        pred_log2 = np.array(pred_log2)
        return {
            "runtime_mse": float(np.mean((true_log2 - pred_log2) ** 2)),
            "runtime_mae": float(np.mean(np.abs(true_log2 - pred_log2))),
        }


class EntanglementBudgetModelWrapper(BaseModel):
    """
    Wrapper for EntanglementBudgetModel implementing BaseModel interface.
    
    Models fidelity as a budget consumed by gates.
    Uses learned regression for runtime prediction.
    """
    
    def __init__(self, **kwargs):
        self.model = _EntanglementBudgetModel()
        self.runtime_model = None
        self.runtime_scaler = None
    
    @property
    def name(self) -> str:
        return "EntanglementBudget"
    
    def _extract_budget_features(self, gates, total_cost, n_qubits, backend, precision, log2_threshold: float):
        """Extract features for runtime prediction (includes log2_threshold as input)."""
        n_gates = len(gates)
        n_2q = sum(1 for g in gates if g.span > 0)
        is_gpu = 1.0 if backend == "GPU" else 0.0
        is_double = 1.0 if precision == "double" else 0.0
        return np.array([
            total_cost,
            np.log1p(total_cost),
            n_gates,
            n_2q,
            n_qubits,
            np.log1p(n_gates),
            total_cost / max(n_qubits, 1),
            is_gpu,
            is_double,
            log2_threshold,
        ], dtype=np.float32)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        train_data = _extract_qasm_text_and_qubits(train_loader)
        
        cost_by_threshold = {t: [] for t in THRESHOLD_LADDER}
        runtime_features = []
        runtime_targets = []
        for d in train_data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            total_cost = sum(
                self.model.compute_gate_fidelity_cost(g, d["n_qubits"])
                for g in gates
            )
            thresh = d["threshold"]
            if thresh in cost_by_threshold:
                cost_by_threshold[thresh].append(total_cost)
            log2_thresh = np.log2(max(thresh, 1))
            features = self._extract_budget_features(
                gates, total_cost, d["n_qubits"], d["backend"], d["precision"], log2_thresh
            )
            runtime_features.append(features)
            runtime_targets.append(d["log2_runtime"])
        new_budgets = {}
        cumulative_max = 0
        default_budgets = getattr(self.model, "threshold_budgets", {})
        for thresh in THRESHOLD_LADDER:
            if cost_by_threshold[thresh]:
                cumulative_max = max(cumulative_max, max(cost_by_threshold[thresh]) * 1.1)
            new_budgets[thresh] = cumulative_max if cumulative_max > 0 else default_budgets.get(thresh, 1.0)
        self.model.threshold_budgets = new_budgets
        
        X = np.array(runtime_features)
        y = np.array(runtime_targets)
        
        self.runtime_scaler = StandardScaler()
        X_scaled = self.runtime_scaler.fit_transform(X)
        
        self.runtime_model = Ridge(alpha=1.0)
        self.runtime_model.fit(X_scaled, y)
        
        return {"calibrated_budgets": new_budgets}
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "EntanglementBudgetModel requires QASM text, not features. "
            "Use predict_from_loader instead."
        )
    
    def predict_from_loader(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        data = _extract_qasm_text_and_qubits(loader)
        thresh_preds = []
        runtime_preds = []
        for d in data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            total_cost = sum(
                self.model.compute_gate_fidelity_cost(g, d["n_qubits"])
                for g in gates
            )
            threshold = d["threshold"]
            log2_thresh = np.log2(max(threshold, 1))
            features = self._extract_budget_features(
                gates, total_cost, d["n_qubits"], d["backend"], d["precision"], log2_thresh
            )
            X_scaled = self.runtime_scaler.transform(features.reshape(1, -1))
            pred_log2 = self.runtime_model.predict(X_scaled)[0]
            thresh_preds.append(threshold)
            runtime_preds.append(max(np.power(2.0, pred_log2), 1e-10))
        return np.array(thresh_preds), np.array(runtime_preds)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        data = _extract_qasm_text_and_qubits(loader)
        true_log2 = []
        pred_log2 = []
        for d in data:
            gates = parse_circuit_gates(d["text"], d["n_qubits"])
            total_cost = sum(
                self.model.compute_gate_fidelity_cost(g, d["n_qubits"])
                for g in gates
            )
            log2_thresh = np.log2(max(d["threshold"], 1))
            features = self._extract_budget_features(
                gates, total_cost, d["n_qubits"], d["backend"], d["precision"], log2_thresh
            )
            X_scaled = self.runtime_scaler.transform(features.reshape(1, -1))
            pred_log2.append(self.runtime_model.predict(X_scaled)[0])
            true_log2.append(d["log2_runtime"])
        true_log2 = np.array(true_log2)
        pred_log2 = np.array(pred_log2)
        return {
            "runtime_mse": float(np.mean((true_log2 - pred_log2) ** 2)),
            "runtime_mae": float(np.mean(np.abs(true_log2 - pred_log2))),
        }

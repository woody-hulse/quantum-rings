#!/usr/bin/env python3
"""
Test component-based models against the actual dataset.

Compares:
1. Analytical Cost Model (hand-tuned physics-based)
2. Bond Dimension Tracker (MPS simulation proxy)
3. Entanglement Budget Model (fidelity-based)
4. Learned Component Model (regression on component features)
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from component_model import (
    parse_circuit_gates,
    AnalyticalCostModel,
    BondDimensionTracker,
    EntanglementBudgetModel,
    LearnedComponentModel,
    THRESHOLD_LADDER,
)


def load_ground_truth(data_path: Path, circuits_dir: Path) -> list:
    """Load circuits and their ground truth thresholds."""
    with open(data_path, "r") as f:
        data = json.load(f)
    
    circuit_info = {c["file"]: c for c in data["circuits"]}
    
    results = []
    for r in data["results"]:
        if r["status"] != "ok":
            continue
        
        sweep = r.get("threshold_sweep", [])
        if not sweep:
            continue
        
        true_min_threshold = None
        for entry in sorted(sweep, key=lambda x: x["threshold"]):
            fid = entry.get("sdk_get_fidelity")
            if fid is not None and fid >= 0.99:
                true_min_threshold = entry["threshold"]
                break
        
        if true_min_threshold is None:
            true_min_threshold = THRESHOLD_LADDER[-1]
        
        forward = r.get("forward", {})
        runtime = forward.get("run_wall_s", 0.0) if forward else 0.0
        
        qasm_path = circuits_dir / r["file"]
        if not qasm_path.exists():
            continue
        
        results.append({
            "file": r["file"],
            "backend": r["backend"],
            "precision": r["precision"],
            "true_threshold": true_min_threshold,
            "true_threshold_idx": THRESHOLD_LADDER.index(true_min_threshold),
            "true_runtime": runtime,
            "qasm_path": qasm_path,
            "n_qubits": circuit_info.get(r["file"], {}).get("n_qubits", 0),
        })
    
    return results


def evaluate_analytical_model(results: list, calibrated: bool = True) -> dict:
    """Evaluate the analytical cost model."""
    model = AnalyticalCostModel(calibrated=calibrated)
    model_name = "Analytical Cost Model" + (" (calibrated)" if calibrated else " (uncalibrated)")
    
    predictions = []
    for r in results:
        text = r["qasm_path"].read_text()
        gates = parse_circuit_gates(text, r["n_qubits"])
        costs = model.compute_circuit_cost(gates, r["n_qubits"])
        pred_idx = model.predict_threshold_index(costs)
        pred_threshold = THRESHOLD_LADDER[pred_idx]
        
        predictions.append({
            "true_idx": r["true_threshold_idx"],
            "pred_idx": pred_idx,
            "true_threshold": r["true_threshold"],
            "pred_threshold": pred_threshold,
            "costs": costs,
        })
    
    return compute_metrics(predictions, model_name)


def evaluate_calibrated_analytical_model(results: list, val_fraction: float = 0.2) -> dict:
    """Evaluate analytical model with data-driven calibration."""
    model = AnalyticalCostModel(calibrated=False)
    
    np.random.seed(42)
    files = list(set(r["file"] for r in results))
    np.random.shuffle(files)
    n_val = int(len(files) * val_fraction)
    val_files = set(files[:n_val])
    
    train_results = [r for r in results if r["file"] not in val_files]
    val_results = [r for r in results if r["file"] in val_files]
    
    train_data = []
    for r in train_results:
        text = r["qasm_path"].read_text()
        gates = parse_circuit_gates(text, r["n_qubits"])
        costs = model.compute_circuit_cost(gates, r["n_qubits"])
        train_data.append({
            "costs": costs,
            "true_threshold_idx": r["true_threshold_idx"],
        })
    
    model.calibrate(train_data)
    
    predictions = []
    for r in val_results:
        text = r["qasm_path"].read_text()
        gates = parse_circuit_gates(text, r["n_qubits"])
        costs = model.compute_circuit_cost(gates, r["n_qubits"])
        pred_idx = model.predict_threshold_index(costs)
        
        predictions.append({
            "true_idx": r["true_threshold_idx"],
            "pred_idx": pred_idx,
            "true_threshold": r["true_threshold"],
            "pred_threshold": THRESHOLD_LADDER[pred_idx],
        })
    
    return compute_metrics(predictions, "Analytical + Auto-Calibration")


def evaluate_bond_dim_model(results: list) -> dict:
    """Evaluate the bond dimension tracker."""
    predictions = []
    
    for r in results:
        text = r["qasm_path"].read_text()
        gates = parse_circuit_gates(text, r["n_qubits"])
        
        tracker = BondDimensionTracker(r["n_qubits"])
        for gate in gates:
            tracker.apply_gate(gate)
        
        max_bd = tracker.get_max_bond_dim()
        
        if max_bd < 2:
            pred_idx = 0
        elif max_bd < 4:
            pred_idx = 1
        elif max_bd < 8:
            pred_idx = 2
        elif max_bd < 16:
            pred_idx = 3
        elif max_bd < 32:
            pred_idx = 4
        elif max_bd < 64:
            pred_idx = 5
        elif max_bd < 128:
            pred_idx = 6
        elif max_bd < 256:
            pred_idx = 7
        else:
            pred_idx = 8
        
        predictions.append({
            "true_idx": r["true_threshold_idx"],
            "pred_idx": pred_idx,
            "true_threshold": r["true_threshold"],
            "pred_threshold": THRESHOLD_LADDER[pred_idx],
            "max_bond_dim": max_bd,
        })
    
    return compute_metrics(predictions, "Bond Dimension Tracker")


def evaluate_budget_model(results: list) -> dict:
    """Evaluate the entanglement budget model."""
    model = EntanglementBudgetModel()
    predictions = []
    
    for r in results:
        text = r["qasm_path"].read_text()
        gates = parse_circuit_gates(text, r["n_qubits"])
        
        pred_threshold = model.find_minimum_threshold(gates, r["n_qubits"])
        pred_idx = THRESHOLD_LADDER.index(pred_threshold)
        
        predictions.append({
            "true_idx": r["true_threshold_idx"],
            "pred_idx": pred_idx,
            "true_threshold": r["true_threshold"],
            "pred_threshold": pred_threshold,
        })
    
    return compute_metrics(predictions, "Entanglement Budget Model")


def evaluate_learned_model(results: list, val_fraction: float = 0.2) -> dict:
    """Evaluate the learned component model with train/val split."""
    model = LearnedComponentModel()
    
    circuit_data = []
    for r in results:
        text = r["qasm_path"].read_text()
        gates = parse_circuit_gates(text, r["n_qubits"])
        features = model.extract_component_features(gates, r["n_qubits"])
        circuit_data.append({
            "features": features,
            "true_idx": r["true_threshold_idx"],
            "true_runtime": r["true_runtime"],
            "file": r["file"],
        })
    
    np.random.seed(42)
    files = list(set(d["file"] for d in circuit_data))
    np.random.shuffle(files)
    n_val = int(len(files) * val_fraction)
    val_files = set(files[:n_val])
    
    train_data = [d for d in circuit_data if d["file"] not in val_files]
    val_data = [d for d in circuit_data if d["file"] in val_files]
    
    if not train_data or not val_data:
        print("Not enough data for train/val split")
        return {}
    
    y_train_thresh = np.array([d["true_idx"] for d in train_data])
    y_train_runtime = np.array([d["true_runtime"] for d in train_data])
    
    model.fit(train_data, y_train_thresh, y_train_runtime)
    
    predictions = []
    for d in val_data:
        pred_idx, pred_runtime = model.predict(d["features"])
        predictions.append({
            "true_idx": d["true_idx"],
            "pred_idx": pred_idx,
            "true_threshold": THRESHOLD_LADDER[d["true_idx"]],
            "pred_threshold": THRESHOLD_LADDER[pred_idx],
            "true_runtime": d["true_runtime"],
            "pred_runtime": pred_runtime,
        })
    
    metrics = compute_metrics(predictions, "Learned Component Model")
    
    if predictions and "true_runtime" in predictions[0]:
        true_rt = np.array([p["true_runtime"] for p in predictions])
        pred_rt = np.array([p["pred_runtime"] for p in predictions])
        
        mask = true_rt > 0
        if mask.sum() > 0:
            log_mse = np.mean((np.log1p(true_rt[mask]) - np.log1p(pred_rt[mask]))**2)
            metrics["runtime_log_mse"] = log_mse
    
    return metrics


def compute_metrics(predictions: list, model_name: str) -> dict:
    """Compute evaluation metrics."""
    if not predictions:
        return {}
    
    true_idx = np.array([p["true_idx"] for p in predictions])
    pred_idx = np.array([p["pred_idx"] for p in predictions])
    
    accuracy = np.mean(true_idx == pred_idx)
    
    mae = np.mean(np.abs(true_idx - pred_idx))
    
    underpredict = np.sum(pred_idx < true_idx)
    overpredict = np.sum(pred_idx > true_idx)
    
    off_by_one = np.mean(np.abs(true_idx - pred_idx) <= 1)
    
    challenge_scores = []
    for p in predictions:
        if p["pred_idx"] < p["true_idx"]:
            challenge_scores.append(0.0)
        else:
            distance = p["pred_idx"] - p["true_idx"]
            challenge_scores.append(1.0 / (1.0 + distance))
    challenge_score = np.mean(challenge_scores)
    
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"MAE (rungs):        {mae:.4f}")
    print(f"Within 1 rung:      {off_by_one:.4f}")
    print(f"Underpredictions:   {underpredict} ({100*underpredict/len(predictions):.1f}%)")
    print(f"Overpredictions:    {overpredict} ({100*overpredict/len(predictions):.1f}%)")
    print(f"Challenge Score:    {challenge_score:.4f}")
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "mae": mae,
        "off_by_one": off_by_one,
        "underpredict": underpredict,
        "overpredict": overpredict,
        "challenge_score": challenge_score,
    }


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    print("Loading ground truth data...")
    results = load_ground_truth(data_path, circuits_dir)
    print(f"Loaded {len(results)} valid results")
    
    print("\n" + "="*60)
    print("COMPONENT-BASED MODEL COMPARISON")
    print("="*60)
    
    metrics = []
    
    metrics.append(evaluate_analytical_model(results, calibrated=False))
    metrics.append(evaluate_analytical_model(results, calibrated=True))
    metrics.append(evaluate_calibrated_analytical_model(results))
    metrics.append(evaluate_bond_dim_model(results))
    metrics.append(evaluate_budget_model(results))
    metrics.append(evaluate_learned_model(results))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'Acc':>8} {'MAE':>8} {'Score':>8}")
    print("-" * 56)
    for m in metrics:
        if m:
            print(f"{m['model']:<30} {m['accuracy']:>8.4f} {m['mae']:>8.4f} {m['challenge_score']:>8.4f}")
    
    best = max(metrics, key=lambda x: x.get("challenge_score", 0) if x else 0)
    print(f"\nBest model: {best['model']} (score: {best['challenge_score']:.4f})")


if __name__ == "__main__":
    main()

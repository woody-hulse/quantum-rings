#!/usr/bin/env python3
"""
Submission script for the Circuit Fingerprint Challenge (iQuHACK 2026).

This script loads pre-trained models from the artifacts directory and makes
predictions on holdout circuits.

Modes:
    --mode threshold  : Predict threshold only (given QASM, backend, precision)
    --mode duration   : Predict duration only (given QASM, backend, precision, threshold)
    --mode both       : Predict threshold first, then duration using predicted threshold (default)

Usage:
    # Threshold prediction only
    python predict.py --mode threshold --qasm circuit.qasm --backend gpu --precision double

    # Duration prediction only (threshold provided)
    python predict.py --mode duration --qasm circuit.qasm --backend gpu --precision double --threshold 16

    # Batch prediction from tasks file
    python predict.py --tasks <TASKS_JSON> --circuits <CIRCUITS_DIR> --id-map <ID_MAP> --out predictions.json

The models must be trained first using train_submission_models.py.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "src"))

from qasm_features import extract_qasm_features
from gnn.graph_builder import build_graph_from_file, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE
from gnn.transformer import QuantumCircuitGraphTransformer
from data_loader import (
    THRESHOLD_LADDER,
    NUMERIC_FEATURE_KEYS,
    FAMILY_CATEGORIES,
    BACKEND_MAP,
    PRECISION_MAP,
)
from scoring import select_threshold_class_by_expected_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_CATEGORIES)}
NUM_FAMILIES = len(FAMILY_CATEGORIES)
GLOBAL_FEAT_DIM = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES


def load_id_map(path: Path) -> Dict[str, str]:
    """Load the ID map from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["id"]: entry["qasm_file"] for entry in data.get("entries", [])}


def load_holdout_tasks(path: Path) -> List[Dict[str, str]]:
    """Load holdout tasks from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tasks", [])


def extract_threshold_features(
    qasm_path: Path,
    backend: str,
    precision: str,
) -> np.ndarray:
    """Extract features for threshold prediction."""
    qasm_features = extract_qasm_features(qasm_path)
    
    numeric_values = [qasm_features.get(k, 0.0) for k in NUMERIC_FEATURE_KEYS]
    backend_idx = BACKEND_MAP.get(backend, 0)
    precision_idx = PRECISION_MAP.get(precision, 0)
    numeric_values.extend([float(backend_idx), float(precision_idx)])
    
    family_onehot = [0.0] * len(FAMILY_CATEGORIES)
    
    return np.array(numeric_values + family_onehot, dtype=np.float32)


def build_duration_graph(
    qasm_path: Path,
    backend: str,
    precision: str,
    log2_threshold: float,
    family: Optional[str] = None,
) -> Data:
    """Build a graph for duration prediction."""
    graph_dict = build_graph_from_file(
        qasm_path,
        backend=backend,
        precision=precision,
        family=family if family else "unknown",
        family_to_idx=FAMILY_TO_IDX,
        num_families=NUM_FAMILIES,
        log2_threshold=log2_threshold,
    )
    
    return Data(
        x=graph_dict["x"],
        edge_index=graph_dict["edge_index"],
        edge_attr=graph_dict["edge_attr"],
        edge_gate_type=graph_dict["edge_gate_type"],
        global_features=graph_dict["global_features"],
    )


class XGBoostThresholdPredictor:
    """XGBoost threshold predictor (inference only)."""
    
    def __init__(self, conservative_bias: float = 0.1):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        self.classifier = None
        self.scaler = StandardScaler()
        self.conservative_bias = conservative_bias
    
    def load(self, path: Path) -> None:
        """Load model and scaler from directory."""
        path = Path(path)
        self.classifier = xgb.XGBClassifier()
        self.classifier.load_model(str(path / "xgb_threshold.json"))
        self.scaler.mean_ = np.load(path / "scaler_mean.npy")
        self.scaler.scale_ = np.load(path / "scaler_scale.npy")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict threshold class indices."""
        X_scaled = self.scaler.transform(X)
        proba = self.classifier.predict_proba(X_scaled)
        return select_threshold_class_by_expected_score(proba, conservative_bias=self.conservative_bias)


class TransformerGNNPredictor:
    """TransformerGNN duration predictor (inference only)."""
    
    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 4,
        num_heads: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.model = None
    
    def _create_model(self) -> QuantumCircuitGraphTransformer:
        return QuantumCircuitGraphTransformer(
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM,
            global_feat_dim=GLOBAL_FEAT_DIM,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
    
    def load(self, path: Path) -> None:
        """Load model from directory."""
        path = Path(path)
        self.model = self._create_model().to(self.device)
        self.model.load_state_dict(torch.load(path / "gnn_duration.pt", map_location=self.device, weights_only=True))
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, data: Data) -> float:
        """Predict log2(runtime) for a single graph."""
        data = data.to(self.device)
        
        if data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        
        pred = self.model.predict_runtime(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            edge_gate_type=data.edge_gate_type,
            batch=data.batch,
            global_features=data.global_features,
        )
        
        return pred.cpu().item()


def predict_threshold_single(
    qasm_path: Path,
    backend: str,
    precision: str,
    threshold_predictor: XGBoostThresholdPredictor,
) -> int:
    """Predict threshold for a single circuit."""
    features = extract_threshold_features(qasm_path, backend, precision)
    threshold_class_idx = threshold_predictor.predict(features.reshape(1, -1))[0]
    return int(THRESHOLD_LADDER[threshold_class_idx])


def predict_duration_single(
    qasm_path: Path,
    backend: str,
    precision: str,
    threshold: int,
    duration_predictor: TransformerGNNPredictor,
) -> float:
    """Predict duration for a single circuit given threshold."""
    log2_threshold = np.log2(max(threshold, 1))
    graph = build_duration_graph(qasm_path, backend, precision, log2_threshold)
    log2_runtime = duration_predictor.predict(graph)
    return float(2.0 ** log2_runtime)


def main():
    parser = argparse.ArgumentParser(description="Circuit Fingerprint Challenge Prediction")
    parser.add_argument("--mode", type=str, default="both", choices=["threshold", "duration", "both"],
                        help="Prediction mode: threshold, duration, or both (default: both)")
    parser.add_argument("--qasm", type=str, default=None, help="Path to single QASM file (single prediction mode)")
    parser.add_argument("--backend", type=str, default=None, help="Backend: gpu or cpu (single prediction mode)")
    parser.add_argument("--precision", type=str, default=None, help="Precision: single or double (single prediction mode)")
    parser.add_argument("--threshold", type=int, default=None, help="Threshold value (required for duration mode)")
    parser.add_argument("--tasks", type=str, default=None, help="Path to holdout tasks JSON (batch mode)")
    parser.add_argument("--circuits", type=str, default=None, help="Path to holdout circuits directory (batch mode)")
    parser.add_argument("--id-map", type=str, default=None, help="Path to ID map JSON (batch mode)")
    parser.add_argument("--out", type=str, default=None, help="Output predictions JSON path (batch mode)")
    parser.add_argument("--artifacts", type=str, default=None, help="Path to model artifacts (default: ./artifacts)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for GNN (cpu/cuda/mps)")
    parser.add_argument("--verbose", action="store_true", help="Print prediction progress")
    args = parser.parse_args()

    if args.artifacts:
        artifacts_path = Path(args.artifacts)
    else:
        artifacts_path = Path(__file__).parent / "artifacts"

    if not artifacts_path.exists():
        print(f"Error: Artifacts directory not found at {artifacts_path}")
        print("Please run train_submission_models.py first to train and save the models.")
        sys.exit(1)

    # Single prediction mode (for live presentation)
    if args.qasm is not None:
        qasm_path = Path(args.qasm)
        if not qasm_path.exists():
            print(f"Error: QASM file not found: {qasm_path}")
            sys.exit(1)
        if args.backend is None or args.precision is None:
            print("Error: --backend and --precision are required for single prediction mode")
            sys.exit(1)

        if args.mode == "threshold":
            print(f"Loading threshold model from {artifacts_path}...")
            threshold_predictor = XGBoostThresholdPredictor(conservative_bias=0.1)
            threshold_predictor.load(artifacts_path)

            predicted_threshold = predict_threshold_single(
                qasm_path, args.backend, args.precision, threshold_predictor
            )
            print(f"\n{'='*50}")
            print(f"THRESHOLD PREDICTION")
            print(f"{'='*50}")
            print(f"  QASM:      {qasm_path.name}")
            print(f"  Backend:   {args.backend}")
            print(f"  Precision: {args.precision}")
            print(f"  Target Fidelity: 0.75")
            print(f"{'='*50}")
            print(f"  PREDICTED THRESHOLD: {predicted_threshold}")
            print(f"{'='*50}")

        elif args.mode == "duration":
            if args.threshold is None:
                print("Error: --threshold is required for duration mode")
                sys.exit(1)

            print(f"Loading duration model from {artifacts_path}...")
            duration_predictor = TransformerGNNPredictor(
                hidden_dim=32,
                num_layers=4,
                num_heads=2,
                dropout=0.1,
                device=args.device,
            )
            duration_predictor.load(artifacts_path)

            predicted_duration = predict_duration_single(
                qasm_path, args.backend, args.precision, args.threshold, duration_predictor
            )
            print(f"\n{'='*50}")
            print(f"DURATION PREDICTION")
            print(f"{'='*50}")
            print(f"  QASM:      {qasm_path.name}")
            print(f"  Backend:   {args.backend}")
            print(f"  Precision: {args.precision}")
            print(f"  Threshold: {args.threshold}")
            print(f"{'='*50}")
            print(f"  PREDICTED DURATION: {predicted_duration:.6f} seconds")
            print(f"{'='*50}")

        else:  # both
            print(f"Loading models from {artifacts_path}...")
            threshold_predictor = XGBoostThresholdPredictor(conservative_bias=0.1)
            threshold_predictor.load(artifacts_path)

            duration_predictor = TransformerGNNPredictor(
                hidden_dim=32,
                num_layers=4,
                num_heads=2,
                dropout=0.1,
                device=args.device,
            )
            duration_predictor.load(artifacts_path)

            predicted_threshold = predict_threshold_single(
                qasm_path, args.backend, args.precision, threshold_predictor
            )
            predicted_duration = predict_duration_single(
                qasm_path, args.backend, args.precision, predicted_threshold, duration_predictor
            )

            print(f"\n{'='*50}")
            print(f"COMBINED PREDICTION")
            print(f"{'='*50}")
            print(f"  QASM:      {qasm_path.name}")
            print(f"  Backend:   {args.backend}")
            print(f"  Precision: {args.precision}")
            print(f"{'='*50}")
            print(f"  PREDICTED THRESHOLD: {predicted_threshold}")
            print(f"  PREDICTED DURATION:  {predicted_duration:.6f} seconds")
            print(f"{'='*50}")

        return

    # Batch prediction mode (original behavior)
    if args.tasks is None or args.circuits is None or args.id_map is None or args.out is None:
        print("Error: For batch mode, provide --tasks, --circuits, --id-map, and --out")
        print("       For single prediction, provide --qasm, --backend, and --precision")
        sys.exit(1)

    tasks_path = Path(args.tasks)
    circuits_dir = Path(args.circuits)
    id_map_path = Path(args.id_map)
    out_path = Path(args.out)

    print("Loading holdout tasks and ID map...")
    tasks = load_holdout_tasks(tasks_path)
    id_map = load_id_map(id_map_path)
    print(f"Found {len(tasks)} holdout tasks")

    print(f"\nLoading pre-trained models from {artifacts_path}...")

    threshold_predictor = XGBoostThresholdPredictor(conservative_bias=0.1)
    threshold_predictor.load(artifacts_path)
    print("  XGBoost threshold model loaded")

    duration_predictor = TransformerGNNPredictor(
        hidden_dim=32,
        num_layers=4,
        num_heads=2,
        dropout=0.1,
        device=args.device,
    )
    duration_predictor.load(artifacts_path)
    print("  TransformerGNN duration model loaded")

    print("\nMaking predictions on holdout tasks...")
    predictions = []

    for task in tasks:
        task_id = task["id"]
        processor = task["processor"]
        precision = task["precision"]

        qasm_file = id_map.get(task_id)
        if qasm_file is None:
            print(f"Warning: No circuit file found for task {task_id}")
            predictions.append({
                "id": task_id,
                "predicted_threshold_min": 16,
                "predicted_forward_wall_s": 10.0,
            })
            continue

        qasm_path = circuits_dir / qasm_file
        if not qasm_path.exists():
            print(f"Warning: Circuit file not found: {qasm_path}")
            predictions.append({
                "id": task_id,
                "predicted_threshold_min": 16,
                "predicted_forward_wall_s": 10.0,
            })
            continue

        try:
            predicted_threshold = predict_threshold_single(
                qasm_path, processor, precision, threshold_predictor
            )
            predicted_runtime = predict_duration_single(
                qasm_path, processor, precision, predicted_threshold, duration_predictor
            )

            predictions.append({
                "id": task_id,
                "predicted_threshold_min": predicted_threshold,
                "predicted_forward_wall_s": predicted_runtime,
            })

            if args.verbose:
                print(f"  {task_id}: threshold={predicted_threshold}, runtime={predicted_runtime:.3f}s")

        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            predictions.append({
                "id": task_id,
                "predicted_threshold_min": 16,
                "predicted_forward_wall_s": 10.0,
            })

    output_data = {"predictions": predictions}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nPredictions saved to {out_path}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()

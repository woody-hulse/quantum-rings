#!/usr/bin/env python3
"""
Training script for submission models.

Trains XGBoost for threshold prediction and TransformerGNN for duration prediction,
then saves them to the artifacts directory for use by predict.py.

Usage:
    python train_submission_models.py [--artifacts artifacts] [--device cpu] [--no-validation] [--verbose]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent / "src"))

from qasm_features import extract_qasm_features
from gnn.graph_builder import build_graph_from_file, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE
from gnn.transformer import QuantumCircuitGraphTransformer
from data_loader import (
    load_hackathon_data,
    compute_min_threshold,
    threshold_to_class,
    THRESHOLD_LADDER,
    NUMERIC_FEATURE_KEYS,
    FAMILY_CATEGORIES,
    BACKEND_MAP,
    PRECISION_MAP,
)
from scoring import select_threshold_class_by_expected_score, NUM_THRESHOLD_CLASSES

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_CATEGORIES)}
NUM_FAMILIES = len(FAMILY_CATEGORIES)
GLOBAL_FEAT_DIM = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class XGBoostThresholdTrainer:
    """Trains XGBoost for threshold class prediction."""
    
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        conservative_bias: float = 0.1,
    ):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required")
        
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.conservative_bias = conservative_bias
        
        self.classifier = None
        self.scaler = StandardScaler()
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = False,
    ) -> None:
        """Train the XGBoost classifier."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        unique_classes = set(y_train)
        missing_classes = set(range(NUM_THRESHOLD_CLASSES)) - unique_classes
        
        if missing_classes:
            dummy_X = np.zeros((len(missing_classes), X_scaled.shape[1]))
            dummy_y = np.array(list(missing_classes), dtype=np.int64)
            X_scaled = np.vstack([X_scaled, dummy_X])
            y_train = np.concatenate([y_train, dummy_y])
        
        self.classifier = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_THRESHOLD_CLASSES,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
        )
        
        self.classifier.fit(X_scaled, y_train, verbose=False)
        if verbose:
            print(f"XGBoost threshold classifier trained on {len(y_train)} samples")
    
    def save(self, path: Path) -> None:
        """Save model and scaler to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(str(path / "xgb_threshold.json"))
        np.save(path / "scaler_mean.npy", self.scaler.mean_)
        np.save(path / "scaler_scale.npy", self.scaler.scale_)
        print(f"  Saved XGBoost model to {path}")


class TransformerGNNTrainer:
    """Trains Graph Transformer for duration prediction."""
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
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
    
    def fit(
        self,
        data_list: List[Data],
        epochs: int = 100,
        batch_size: int = 16,
        early_stopping_patience: int = 20,
        use_validation: bool = True,
        verbose: bool = False,
    ) -> None:
        """Train the Graph Transformer model."""
        self.model = self._create_model().to(self.device)

        if use_validation:
            n_val = max(1, int(len(data_list) * 0.1))
            train_data = data_list[n_val:]
            val_data = data_list[:n_val]
            val_loader = PyGDataLoader(val_data, batch_size=batch_size, shuffle=False)
        else:
            train_data = data_list
            val_loader = None

        train_loader = PyGDataLoader(
            train_data, batch_size=batch_size, shuffle=True, drop_last=len(train_data) > batch_size
        )
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        criterion = nn.L1Loss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                pred = self.model.predict_runtime(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    edge_gate_type=batch.edge_gate_type,
                    batch=batch.batch,
                    global_features=batch.global_features,
                )

                loss = criterion(pred, batch.log2_runtime)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / max(n_batches, 1)
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        pred = self.model.predict_runtime(
                            x=batch.x,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                            edge_gate_type=batch.edge_gate_type,
                            batch=batch.batch,
                            global_features=batch.global_features,
                        )
                        val_loss += criterion(pred, batch.log2_runtime).item()
                        n_val_batches += 1
                avg_val_loss = val_loss / max(n_val_batches, 1)
                scheduler.step(avg_val_loss)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                scheduler.step(avg_train_loss)
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if verbose and val_loader is not None:
            print(f"  Best validation loss: {best_val_loss:.4f}")
    
    def save(self, path: Path) -> None:
        """Save model to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "gnn_duration.pt")
        print(f"  Saved TransformerGNN model to {path}")


def prepare_training_data(
    data_path: Path,
    circuits_dir: Path,
    fidelity_target: float = 0.75,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Data]]:
    """Prepare training data for both models."""
    circuits, results = load_hackathon_data(data_path)
    circuit_info = {c.file: c for c in circuits}
    
    X_threshold_list = []
    y_threshold_list = []
    duration_graphs = []
    
    ok_results = [r for r in results if r.status == "ok"]
    
    for result in ok_results:
        qasm_path = circuits_dir / result.file
        if not qasm_path.exists():
            continue
        
        min_thr = compute_min_threshold(result.threshold_sweep, target=fidelity_target)
        if min_thr is not None:
            features = extract_threshold_features(qasm_path, result.backend, result.precision)
            X_threshold_list.append(features)
            y_threshold_list.append(threshold_to_class(min_thr))
        
        if result.forward_wall_s is not None and result.forward_wall_s > 0 and result.selected_threshold is not None:
            log2_threshold = np.log2(max(result.selected_threshold, 1))
            log2_runtime = np.log2(max(float(result.forward_wall_s), 1e-10))
            
            circuit = circuit_info.get(result.file)
            family = circuit.family if circuit else ""
            
            try:
                graph_dict = build_graph_from_file(
                    qasm_path,
                    backend=result.backend,
                    precision=result.precision,
                    family=family,
                    family_to_idx=FAMILY_TO_IDX,
                    num_families=NUM_FAMILIES,
                    log2_threshold=log2_threshold,
                )
                
                data = Data(
                    x=graph_dict["x"],
                    edge_index=graph_dict["edge_index"],
                    edge_attr=graph_dict["edge_attr"],
                    edge_gate_type=graph_dict["edge_gate_type"],
                    global_features=graph_dict["global_features"],
                    log2_runtime=torch.tensor(log2_runtime, dtype=torch.float32),
                )
                duration_graphs.append(data)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not process {result.file}: {e}")
                continue
    
    X_threshold = np.vstack(X_threshold_list) if X_threshold_list else np.zeros((0, len(NUMERIC_FEATURE_KEYS) + 2 + len(FAMILY_CATEGORIES)))
    y_threshold = np.array(y_threshold_list, dtype=np.int64)
    
    return X_threshold, y_threshold, duration_graphs


def main():
    parser = argparse.ArgumentParser(description="Train submission models")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Path to save model artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device for GNN training (cpu/cuda/mps)")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs for GNN")
    parser.add_argument("--no-validation", action="store_true", help="Use all samples for training (no held-out validation)")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    args = parser.parse_args()
    
    set_all_seeds(args.seed)
    
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    artifacts_path = Path(args.artifacts)
    
    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("TRAINING SUBMISSION MODELS")
    print("=" * 60)
    
    print("\nPreparing training data...")
    X_threshold, y_threshold, duration_graphs = prepare_training_data(
        data_path, circuits_dir, verbose=args.verbose
    )
    print(f"  Threshold samples: {len(y_threshold)}")
    print(f"  Duration samples: {len(duration_graphs)}")
    
    print("\n" + "-" * 60)
    print("Training XGBoost threshold classifier...")
    print("-" * 60)
    threshold_trainer = XGBoostThresholdTrainer(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        conservative_bias=0.1,
        random_state=args.seed,
    )
    threshold_trainer.fit(X_threshold, y_threshold, verbose=args.verbose)
    threshold_trainer.save(artifacts_path)
    
    print("\n" + "-" * 60)
    print("Training TransformerGNN duration model...")
    print("-" * 60)
    duration_trainer = TransformerGNNTrainer(
        hidden_dim=32,
        num_layers=4,
        num_heads=2,
        dropout=0.1,
        lr=1e-3,
        weight_decay=3e-4,
        device=args.device,
    )
    duration_trainer.fit(
        duration_graphs,
        epochs=args.epochs,
        batch_size=8,
        early_stopping_patience=20,
        use_validation=not args.no_validation,
        verbose=args.verbose,
    )
    duration_trainer.save(artifacts_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nArtifacts saved to: {artifacts_path.absolute()}")
    print("\nContents:")
    for f in artifacts_path.iterdir():
        print(f"  - {f.name}")
    
    print("\nTo make predictions, run:")
    print(f"  python predict.py --tasks <TASKS_JSON> --circuits <CIRCUITS_DIR> --id-map <ID_MAP> --out predictions.json")


if __name__ == "__main__":
    main()

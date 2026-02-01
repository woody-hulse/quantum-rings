#!/usr/bin/env python3
"""
Training script for the Quantum Circuit GNN.

Uses the same evaluation metrics as the main codebase for fair comparison.
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Any, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn.model import create_gnn_model, QuantumCircuitGNN
from gnn.dataset import (
    create_graph_data_loaders,
    create_kfold_graph_data_loaders,
    THRESHOLD_LADDER,
    GLOBAL_FEAT_DIM,
)
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from scoring import compute_challenge_score


class GNNTrainer:
    """Trainer class for the Quantum Circuit GNN."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        threshold_weight: float = 1.0,
        runtime_weight: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.threshold_weight = threshold_weight
        self.runtime_weight = runtime_weight
        
        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        
        self.threshold_criterion = nn.CrossEntropyLoss()
        self.runtime_criterion = nn.MSELoss()
    
    def train_epoch(self, loader: PyGDataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_thresh_loss = 0.0
        total_runtime_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            threshold_logits, runtime_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            
            thresh_loss = self.threshold_criterion(
                threshold_logits, batch.threshold_class
            )
            runtime_loss = self.runtime_criterion(
                runtime_pred, batch.log_runtime
            )
            
            loss = self.threshold_weight * thresh_loss + self.runtime_weight * runtime_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_thresh_loss += thresh_loss.item()
            total_runtime_loss += runtime_loss.item()
            n_batches += 1
        
        return {
            "loss": total_loss / n_batches,
            "threshold_loss": total_thresh_loss / n_batches,
            "runtime_loss": total_runtime_loss / n_batches,
        }
    
    @torch.no_grad()
    def evaluate(self, loader: PyGDataLoader) -> Dict[str, float]:
        """Evaluate on a data loader."""
        self.model.eval()
        
        all_thresh_preds = []
        all_thresh_labels = []
        all_runtime_preds = []
        all_runtime_labels = []
        
        for batch in loader:
            batch = batch.to(self.device)
            
            threshold_logits, runtime_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            
            thresh_preds = threshold_logits.argmax(dim=1).cpu()
            runtime_pred = runtime_pred.cpu()
            
            all_thresh_preds.extend(thresh_preds.tolist())
            all_thresh_labels.extend(batch.threshold_class.cpu().tolist())
            all_runtime_preds.extend(runtime_pred.tolist())
            all_runtime_labels.extend(batch.log_runtime.cpu().tolist())
        
        return {
            "threshold_accuracy": accuracy_score(all_thresh_labels, all_thresh_preds),
            "runtime_mse": mean_squared_error(all_runtime_labels, all_runtime_preds),
            "runtime_mae": mean_absolute_error(all_runtime_labels, all_runtime_preds),
        }
    
    @torch.no_grad()
    def predict(self, loader: PyGDataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for computing challenge scores."""
        self.model.eval()
        
        all_thresh_values = []
        all_runtime_values = []
        
        for batch in loader:
            batch = batch.to(self.device)
            
            threshold_logits, runtime_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            
            thresh_classes = threshold_logits.argmax(dim=1).cpu().numpy()
            thresh_values = [THRESHOLD_LADDER[c] for c in thresh_classes]
            runtime_values = np.expm1(runtime_pred.cpu().numpy())
            
            all_thresh_values.extend(thresh_values)
            all_runtime_values.extend(runtime_values)
        
        return np.array(all_thresh_values), np.array(all_runtime_values)
    
    def fit(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        history = {
            "train_loss": [],
            "val_threshold_acc": [],
            "val_runtime_mse": [],
        }
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            val_loss = (1 - val_metrics["threshold_accuracy"]) + val_metrics["runtime_mse"]
            self.scheduler.step(val_loss)
            
            history["train_loss"].append(train_metrics["loss"])
            history["val_threshold_acc"].append(val_metrics["threshold_accuracy"])
            history["val_runtime_mse"].append(val_metrics["runtime_mse"])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Thresh Acc: {val_metrics['threshold_accuracy']:.4f} | "
                    f"Val Runtime MSE: {val_metrics['runtime_mse']:.4f}"
                )
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {"history": history}


def extract_labels(loader: PyGDataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ground truth labels from a PyG data loader."""
    all_thresh = []
    all_runtime = []
    
    for batch in loader:
        thresh_values = [THRESHOLD_LADDER[c] for c in batch.threshold_class.tolist()]
        all_thresh.extend(thresh_values)
        all_runtime.extend(np.expm1(batch.log_runtime.numpy()).tolist())
    
    return np.array(all_thresh), np.array(all_runtime)


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_evaluation(
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    model_type: str = "basic",
    hidden_dim: int = 64,
    num_layers: int = 4,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    early_stopping_patience: int = 20,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate a single GNN model."""
    model = create_gnn_model(
        model_type=model_type,
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        global_feat_dim=GLOBAL_FEAT_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    
    trainer = GNNTrainer(
        model=model,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
    )
    
    start_time = time.time()
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
    )
    train_time = time.time() - start_time
    
    # Evaluate
    train_metrics = trainer.evaluate(train_loader)
    val_metrics = trainer.evaluate(val_loader)
    
    # Get predictions and ground truth for challenge scoring
    pred_thresh, pred_runtime = trainer.predict(val_loader)
    true_thresh, true_runtime = extract_labels(val_loader)
    
    challenge_scores = compute_challenge_score(
        pred_thresh, true_thresh, pred_runtime, true_runtime, validate=True
    )
    
    return {
        "train_time": train_time,
        "train_metrics": {
            "train_threshold_accuracy": train_metrics["threshold_accuracy"],
            "train_runtime_mse": train_metrics["runtime_mse"],
            "train_runtime_mae": train_metrics["runtime_mae"],
        },
        "val_metrics": val_metrics,
        "challenge_scores": challenge_scores,
    }


def aggregate_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple runs."""
    if not all_results:
        return {}
    
    aggregated = {}
    
    metric_keys = [
        ("train_time", None),
        ("train_threshold_accuracy", "train_metrics"),
        ("train_runtime_mse", "train_metrics"),
        ("train_runtime_mae", "train_metrics"),
        ("threshold_accuracy", "val_metrics"),
        ("runtime_mse", "val_metrics"),
        ("runtime_mae", "val_metrics"),
        ("threshold_score", "challenge_scores"),
        ("runtime_score", "challenge_scores"),
        ("combined_score", "challenge_scores"),
    ]
    
    for key, parent in metric_keys:
        values = []
        for result in all_results:
            if parent:
                val = result.get(parent, {}).get(key)
            else:
                val = result.get(key)
            if val is not None:
                values.append(val)
        
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    
    return aggregated


def print_report(results: Dict[str, Any]) -> None:
    """Print evaluation report."""
    n_runs = results["n_runs"]
    metrics = results["aggregated_metrics"]
    
    print("\n" + "=" * 60)
    print("GNN MODEL REPORT")
    
    is_kfold = "n_folds" in results
    if is_kfold:
        print(f"{results['n_folds']}-fold cross-validation, {results['n_runs_per_fold']} runs per fold")
        print(f"Total evaluations: {n_runs}")
    else:
        print(f"Based on {n_runs} runs with different initializations")
    print("=" * 60)
    
    # Overfitting check
    print("\n--- Overfitting Check (Train vs Val) ---")
    print(f"{'Metric':<25} {'Train Mean':>12} {'Val Mean':>12} {'Gap':>12}")
    print("-" * 61)
    
    overfit_metrics = [
        ("Threshold Accuracy", "train_threshold_accuracy", "threshold_accuracy", True),
        ("Runtime MSE", "train_runtime_mse", "runtime_mse", False),
        ("Runtime MAE", "train_runtime_mae", "runtime_mae", False),
    ]
    
    for display_name, train_key, val_key, higher_is_better in overfit_metrics:
        if train_key in metrics and val_key in metrics:
            train_val = metrics[train_key]["mean"]
            val_val = metrics[val_key]["mean"]
            gap = train_val - val_val if higher_is_better else val_val - train_val
            print(f"{display_name:<25} {train_val:>12.4f} {val_val:>12.4f} {gap:>+12.4f}")
    
    # Full metrics
    print(f"\n--- Validation Metrics (Aggregated) ---")
    print(f"{'Metric':<30} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 78)
    
    metric_display = [
        ("Training Time (s)", "train_time"),
        ("Val Threshold Accuracy", "threshold_accuracy"),
        ("Val Runtime MSE", "runtime_mse"),
        ("Val Runtime MAE", "runtime_mae"),
        ("Challenge Threshold Score", "threshold_score"),
        ("Challenge Runtime Score", "runtime_score"),
        ("Challenge Combined Score", "combined_score"),
    ]
    
    for display_name, key in metric_display:
        if key in metrics:
            m = metrics[key]
            print(
                f"{display_name:<30} {m['mean']:>12.4f} {m['std']:>12.4f} "
                f"{m['min']:>12.4f} {m['max']:>12.4f}"
            )
    
    print("\n" + "-" * 60)
    if "combined_score" in metrics:
        m = metrics["combined_score"]
        print(f"Final Score: {m['mean']:.4f} ± {m['std']:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Quantum Circuit GNN"
    )
    parser.add_argument("--model-type", type=str, default="basic",
                        choices=["basic", "attention"],
                        help="GNN model type (default: basic)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension (default: 64)")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of message passing layers (default: 4)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of runs (default: 10)")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Validation fraction (default: 0.2)")
    parser.add_argument("--kfold", type=int, default=0,
                        help="Number of folds for k-fold CV (0=disabled)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda/mps)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print training progress")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("QUANTUM CIRCUIT GNN EVALUATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model type: {args.model_type}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of runs: {args.n_runs}")
    print(f"  Device: {args.device}")
    
    use_kfold = args.kfold > 1
    
    if use_kfold:
        print(f"  Cross-validation: {args.kfold}-fold")
        n_runs_per_fold = min(args.n_runs, 5)
        print(f"  Runs per fold: {n_runs_per_fold}")
        
        set_all_seeds(args.seed)
        fold_loaders = create_kfold_graph_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            n_folds=args.kfold,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        
        all_results = []
        for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
            print(f"\nFold {fold_idx + 1}/{args.kfold}")
            for run_idx in tqdm(range(n_runs_per_fold), desc=f"Fold {fold_idx+1}"):
                seed = args.seed + fold_idx * 1000 + run_idx
                set_all_seeds(seed)
                
                result = run_single_evaluation(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model_type=args.model_type,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    device=args.device,
                    verbose=args.verbose,
                )
                all_results.append(result)
        
        aggregated = aggregate_metrics(all_results)
        output = {
            "model": "GNN",
            "n_folds": args.kfold,
            "n_runs_per_fold": n_runs_per_fold,
            "n_runs": len(all_results),
            "aggregated_metrics": aggregated,
        }
    
    else:
        print(f"  Validation fraction: {args.val_fraction}")
        
        set_all_seeds(args.seed)
        train_loader, val_loader = create_graph_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        
        print(f"\nDataset info:")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        
        all_results = []
        for i in tqdm(range(args.n_runs), desc="Training GNN models"):
            seed = args.seed + i
            set_all_seeds(seed)
            
            result = run_single_evaluation(
                train_loader=train_loader,
                val_loader=val_loader,
                model_type=args.model_type,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                device=args.device,
                verbose=args.verbose,
            )
            all_results.append(result)
        
        aggregated = aggregate_metrics(all_results)
        output = {
            "model": "GNN",
            "n_runs": args.n_runs,
            "aggregated_metrics": aggregated,
        }
    
    print_report(output)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

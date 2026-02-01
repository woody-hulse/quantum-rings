#!/usr/bin/env python3
"""
Training script for the Quantum Circuit GNN.

Duration-only: threshold as input, predict log2(duration).
Improvements: data augmentation, richer node/edge features, regularization.
"""

from pathlib import Path
import argparse
from typing import Dict, List, Any, Tuple, Optional, Callable
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from gnn.model import create_gnn_model, QuantumCircuitGNN, create_gnn_threshold_class_model
from gnn.dataset import (
    create_graph_data_loaders,
    create_kfold_graph_data_loaders,
    create_threshold_class_graph_data_loaders,
    THRESHOLD_LADDER,
    GLOBAL_FEAT_DIM,
    GLOBAL_FEAT_DIM_THRESHOLD_CLASS,
    NUM_THRESHOLD_CLASSES,
)
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from gnn.augmentation import get_train_augmentation, AugmentedDataset
from scoring import compute_challenge_score, select_threshold_class_by_expected_score, mean_threshold_score


class GNNTrainer:
    """Trainer class for the Quantum Circuit GNN."""

    # Inference strategies
    INFERENCE_ARGMAX = "argmax"
    INFERENCE_DECISION_THEORETIC = "decision_theoretic"

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        augmentation: Optional[Callable] = None,
        inference_strategy: str = "argmax",
    ):
        self.model = model.to(device)
        self.device = device
        self.augmentation = augmentation
        self.inference_strategy = inference_strategy

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        self.runtime_criterion = nn.L1Loss()

    def train_epoch(self, loader: PyGDataLoader) -> Dict[str, float]:
        """Train for one epoch with optional augmentation."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            if self.augmentation is not None:
                batch = self.augmentation(batch)
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            runtime_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            runtime_loss = self.runtime_criterion(runtime_pred, batch.log2_runtime)
            runtime_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += runtime_loss.item()
            n_batches += 1
        return {"loss": total_loss / n_batches}
    
    @torch.no_grad()
    def evaluate(self, loader: PyGDataLoader) -> Dict[str, float]:
        """Evaluate on a data loader."""
        self.model.eval()
        all_runtime_preds = []
        all_runtime_labels = []
        for batch in loader:
            batch = batch.to(self.device)
            runtime_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            runtime_pred = runtime_pred.cpu()
            all_runtime_preds.extend(runtime_pred.tolist())
            all_runtime_labels.extend(batch.log2_runtime.cpu().tolist())
        return {
            "runtime_mse": mean_squared_error(all_runtime_labels, all_runtime_preds),
            "runtime_mae": mean_absolute_error(all_runtime_labels, all_runtime_preds),
        }

    @torch.no_grad()
    def predict(
        self,
        loader: PyGDataLoader,
        strategy: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for computing challenge scores."""
        self.model.eval()
        all_thresh_values = []
        all_runtime_values = []
        for batch in loader:
            batch = batch.to(self.device)
            runtime_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            runtime_values = np.power(2.0, runtime_pred.cpu().numpy())
            thresh = getattr(batch, "threshold", None)
            if thresh is not None:
                thresh_values = thresh.cpu().numpy().tolist() if hasattr(thresh, "cpu") else list(thresh)
            else:
                thresh_values = [0] * len(runtime_values)
            all_thresh_values.extend(thresh_values)
            all_runtime_values.extend(runtime_values.tolist())
        return np.array(all_thresh_values), np.array(all_runtime_values)
    
    def fit(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping (best model by val MAE in log space)."""
        history = {
            "train_loss": [],
            "val_runtime_mae": [],
        }
        best_val_mae = float("inf")
        patience_counter = 0
        best_state = None
        epoch_iter = range(epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training", leave=False)
        for epoch in epoch_iter:
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_mae = val_metrics["runtime_mae"]
            self.scheduler.step(val_mae)
            history["train_loss"].append(train_metrics["loss"])
            history["val_runtime_mae"].append(val_mae)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
            if show_progress:
                epoch_iter.set_postfix({
                    "loss": f"{train_metrics['loss']:.3f}",
                    "val_mae": f"{val_mae:.3f}",
                })
            elif verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val MAE (log2): {val_mae:.4f}"
                )
            
            if patience_counter >= early_stopping_patience:
                if show_progress:
                    epoch_iter.close()
                elif verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {"history": history}


class GNNTrainerThresholdClass:
    """Trainer for GNN threshold-class model: CrossEntropyLoss, select by max expected score."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        augmentation: Optional[Callable] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.augmentation = augmentation
        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader: PyGDataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            if self.augmentation is not None:
                batch = self.augmentation(batch)
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            loss = self.criterion(logits, batch.threshold_class)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return {"loss": total_loss / n_batches}

    @torch.no_grad()
    def evaluate(self, loader: PyGDataLoader) -> Dict[str, float]:
        self.model.eval()
        all_proba = []
        all_true = []
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            proba = torch.softmax(logits.cpu(), dim=-1).numpy()
            all_proba.append(proba)
            all_true.extend(batch.threshold_class.cpu().tolist())
        proba = np.vstack(all_proba)
        true_idx = np.array(all_true, dtype=np.int64)
        chosen = select_threshold_class_by_expected_score(proba)
        return {
            "threshold_accuracy": float(np.mean(chosen == true_idx)),
            "expected_threshold_score": mean_threshold_score(chosen, true_idx),
        }

    @torch.no_grad()
    def predict_proba(self, loader: PyGDataLoader) -> np.ndarray:
        self.model.eval()
        all_proba = []
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            proba = torch.softmax(logits.cpu(), dim=-1).numpy()
            all_proba.append(proba)
        return np.vstack(all_proba)

    def fit(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        history = {"train_loss": [], "val_threshold_score": []}
        best_val_score = -1.0
        patience_counter = 0
        best_state = None
        epoch_iter = range(epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training", leave=False)
        for epoch in epoch_iter:
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_score = val_metrics["expected_threshold_score"]
            self.scheduler.step(val_score)
            history["train_loss"].append(train_metrics["loss"])
            history["val_threshold_score"].append(val_score)
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            if show_progress:
                epoch_iter.set_postfix({
                    "loss": f"{train_metrics['loss']:.3f}",
                    "val_score": f"{val_score:.3f}",
                })
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {train_metrics['loss']:.4f} | Val score: {val_score:.4f}")
            if patience_counter >= early_stopping_patience:
                if show_progress:
                    epoch_iter.close()
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"history": history}


def extract_labels_duration(loader: PyGDataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ground truth from duration-only loader: threshold and 2^log2_runtime (seconds)."""
    all_thresh = []
    all_runtime = []
    for batch in loader:
        thresh = getattr(batch, "threshold", None)
        if thresh is not None:
            t = thresh.cpu().numpy().tolist() if hasattr(thresh, "cpu") else list(thresh)
            all_thresh.extend(t)
        all_runtime.extend(np.power(2.0, batch.log2_runtime.numpy()).tolist())
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
    use_augmentation: bool = True,
    augmentation_strength: float = 0.5,
    inference_strategy: str = "argmax",
) -> Dict[str, Any]:
    """Train and evaluate a single GNN model (threshold as input, log2(duration) output)."""
    model = create_gnn_model(
        model_type=model_type,
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        global_feat_dim=GLOBAL_FEAT_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    augmentation = None
    if use_augmentation:
        augmentation = get_train_augmentation(
            qubit_perm_p=augmentation_strength,
            edge_dropout_p=0.1 * augmentation_strength,
            feature_noise_std=0.1 * augmentation_strength,
            temporal_jitter_std=0.05 * augmentation_strength,
        )
    trainer = GNNTrainer(
        model=model,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        augmentation=augmentation,
        inference_strategy=inference_strategy,
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
    train_metrics = trainer.evaluate(train_loader)
    val_metrics = trainer.evaluate(val_loader)
    pred_thresh, pred_runtime = trainer.predict(val_loader)
    true_thresh, true_runtime = extract_labels_duration(val_loader)
    challenge_scores = compute_challenge_score(
        pred_thresh, true_thresh, pred_runtime, true_runtime
    )
    return {
        "train_time": train_time,
        "train_metrics": {
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
        ("train_runtime_mse", "train_metrics"),
        ("train_runtime_mae", "train_metrics"),
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
        ("Runtime MAE (log2)", "train_runtime_mae", "runtime_mae", False),
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
        ("Val Runtime MAE (log2)", "runtime_mae"),
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
        description="Train and evaluate Quantum Circuit GNN with improvements"
    )
    # Model architecture
    parser.add_argument("--model-type", type=str, default="basic",
                        choices=["basic", "attention"],
                        help="GNN model type (default: basic)")
    parser.add_argument("--hidden-dim", type=int, default=48,
                        help="Hidden dimension (default: 48, reduced for regularization)")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of message passing layers (default: 3)")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="Dropout rate (default: 0.25, increased for regularization)")
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help="Weight decay (default: 1e-3, increased for regularization)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    
    # Improvements
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--aug-strength", type=float, default=0.5,
                        help="Augmentation strength 0-1 (default: 0.5)")
    parser.add_argument("--inference-strategy", type=str, default="argmax",
                        choices=["argmax", "decision_theoretic"],
                        help="Inference strategy (default: argmax)")
    
    # Evaluation
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
    
    use_augmentation = not args.no_augmentation
    
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
    print(f"  Data augmentation: {use_augmentation}")
    if use_augmentation:
        print(f"  Augmentation strength: {args.aug_strength}")
    print(f"  Inference strategy: {args.inference_strategy}")
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
            for run_idx in range(n_runs_per_fold):
                print(f"  Run {run_idx + 1}/{n_runs_per_fold}")
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
                    use_augmentation=use_augmentation,
                    augmentation_strength=args.aug_strength,
                    inference_strategy=args.inference_strategy,
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
        for i in range(args.n_runs):
            print(f"\nRun {i + 1}/{args.n_runs}")
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
                use_augmentation=use_augmentation,
                augmentation_strength=args.aug_strength,
                inference_strategy=args.inference_strategy,
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

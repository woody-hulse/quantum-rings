#!/usr/bin/env python3
"""
Training script for separate Task 1 and Task 2 models.

Task 1: Threshold prediction
Task 2: Runtime prediction (given threshold)
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn.separate_models import (
    ThresholdPredictionGNN,
    RuntimePredictionGNN,
    create_threshold_model,
    create_runtime_model,
)
from gnn.model import OrdinalRegressionLoss
from gnn.dataset import (
    create_graph_data_loaders,
    create_kfold_graph_data_loaders,
    THRESHOLD_LADDER,
    GLOBAL_FEAT_DIM,
)
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from scoring import compute_threshold_score, compute_runtime_score


class ThresholdTrainer:
    """
    Trainer for Task 1: Threshold prediction with expected-score optimization.

    Trains with CrossEntropyLoss to learn probability distributions.
    At inference, uses expected-score decision rule to maximize expected points.
    """

    def __init__(
        self,
        model: ThresholdPredictionGNN,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        # Use CrossEntropyLoss to learn good probability distributions
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader: PyGDataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            threshold_logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )

            loss = self.criterion(threshold_logits, batch.threshold_class)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / n_batches}

    @torch.no_grad()
    def evaluate(self, loader: PyGDataLoader) -> Dict[str, float]:
        """Evaluate on a data loader."""
        self.model.eval()

        all_preds = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)

            threshold_logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )

            preds = self.model.predict_threshold_class(threshold_logits).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch.threshold_class.cpu().tolist())

        return {"threshold_accuracy": accuracy_score(all_labels, all_preds)}

    @torch.no_grad()
    def predict(self, loader: PyGDataLoader) -> np.ndarray:
        """Get threshold predictions."""
        self.model.eval()
        all_thresh_values = []

        for batch in loader:
            batch = batch.to(self.device)

            threshold_logits = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )

            thresh_classes = self.model.predict_threshold_class(threshold_logits).cpu().numpy()
            thresh_values = [THRESHOLD_LADDER[c] for c in thresh_classes]
            all_thresh_values.extend(thresh_values)

        return np.array(all_thresh_values)

    def fit(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        history = {"train_loss": [], "val_threshold_acc": []}

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        epoch_iter = range(epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training Threshold Model", leave=False)

        for epoch in epoch_iter:
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            val_loss = 1 - val_metrics["threshold_accuracy"]
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_metrics["loss"])
            history["val_threshold_acc"].append(val_metrics["threshold_accuracy"])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if show_progress:
                epoch_iter.set_postfix({
                    "loss": f"{train_metrics['loss']:.3f}",
                    "val_acc": f"{val_metrics['threshold_accuracy']:.3f}",
                })
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {train_metrics['loss']:.4f} | Val Acc: {val_metrics['threshold_accuracy']:.4f}")

            if patience_counter >= early_stopping_patience:
                if show_progress:
                    epoch_iter.close()
                elif verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {"history": history}


class RuntimeTrainer:
    """Trainer for Task 2: Runtime prediction."""

    def __init__(
        self,
        model: RuntimePredictionGNN,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        self.criterion = nn.HuberLoss(delta=1.0)

    def train_epoch(self, loader: PyGDataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            runtime_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
                threshold_class=batch.threshold_class,  # Ground truth threshold
            )

            loss = self.criterion(runtime_pred, batch.log_runtime)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / n_batches}

    @torch.no_grad()
    def evaluate(self, loader: PyGDataLoader) -> Dict[str, float]:
        """Evaluate on a data loader."""
        self.model.eval()

        all_preds = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)

            runtime_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
                threshold_class=batch.threshold_class,  # Ground truth threshold
            )

            all_preds.extend(runtime_pred.cpu().tolist())
            all_labels.extend(batch.log_runtime.cpu().tolist())

        return {
            "runtime_mse": mean_squared_error(all_labels, all_preds),
            "runtime_mae": mean_absolute_error(all_labels, all_preds),
        }

    @torch.no_grad()
    def predict(self, loader: PyGDataLoader, threshold_classes: torch.Tensor) -> np.ndarray:
        """Get runtime predictions for given thresholds."""
        self.model.eval()
        all_runtime_values = []

        sample_idx = 0
        for batch in loader:
            batch = batch.to(self.device)
            batch_size = batch.batch.max().item() + 1

            batch_threshold_classes = threshold_classes[sample_idx:sample_idx + batch_size].to(self.device)

            runtime_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
                threshold_class=batch_threshold_classes,
            )

            runtime_values = np.expm1(runtime_pred.cpu().numpy())
            all_runtime_values.extend(runtime_values)
            sample_idx += batch_size

        return np.array(all_runtime_values)

    def fit(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        history = {"train_loss": [], "val_runtime_mse": []}

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        epoch_iter = range(epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="Training Runtime Model", leave=False)

        for epoch in epoch_iter:
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            val_loss = val_metrics["runtime_mse"]
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_metrics["loss"])
            history["val_runtime_mse"].append(val_metrics["runtime_mse"])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if show_progress:
                epoch_iter.set_postfix({
                    "loss": f"{train_metrics['loss']:.3f}",
                    "val_mse": f"{val_metrics['runtime_mse']:.3f}",
                })
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {train_metrics['loss']:.4f} | Val MSE: {val_metrics['runtime_mse']:.4f}")

            if patience_counter >= early_stopping_patience:
                if show_progress:
                    epoch_iter.close()
                elif verbose:
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


def main():
    parser = argparse.ArgumentParser(
        description="Train separate Task 1 and Task 2 models"
    )
    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=48,
                        help="Hidden dimension (default: 48)")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of message passing layers (default: 3)")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="Dropout rate (default: 0.25)")

    # Training
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help="Weight decay (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")

    # Evaluation
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of runs (default: 5)")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Validation fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda/mps)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print training progress")

    # Task selection
    parser.add_argument("--task", type=str, default="both", choices=["1", "2", "both"],
                        help="Which task to train: 1 (threshold), 2 (runtime), or both (default: both)")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"

    print("=" * 60)
    if args.task == "1":
        print("TRAINING TASK 1: THRESHOLD PREDICTION")
    elif args.task == "2":
        print("TRAINING TASK 2: RUNTIME PREDICTION")
    else:
        print("SEPARATE TASK 1 & TASK 2 MODELS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Task: {args.task}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of runs: {args.n_runs}")
    print(f"  Device: {args.device}")

    # Create data loaders
    print("\nLoading data...")
    set_all_seeds(args.seed)
    train_loader, val_loader = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Extract ground truth
    true_thresh, true_runtime = extract_labels(val_loader)

    # Run multiple training runs
    all_task1_scores = []
    all_task2_scores = []

    for run_idx in range(args.n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{args.n_runs}")
        print(f"{'='*60}")

        seed = args.seed + run_idx
        set_all_seeds(seed)

        task1_score = None
        task2_score = None

        # ==================== TASK 1: THRESHOLD PREDICTION ====================
        if args.task in ["1", "both"]:
            print("\n--- Training Task 1: Threshold Prediction ---")
            print("Using expected-score optimization for threshold prediction")
            threshold_model = create_threshold_model(
                node_feat_dim=NODE_FEAT_DIM,
                edge_feat_dim=EDGE_FEAT_DIM,
                global_feat_dim=GLOBAL_FEAT_DIM,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )

            threshold_trainer = ThresholdTrainer(
                model=threshold_model,
                device=args.device,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            start = time.time()
            threshold_trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                early_stopping_patience=20,
                verbose=args.verbose,
                show_progress=True,
            )
            task1_time = time.time() - start

            # Evaluate Task 1
            pred_thresh = threshold_trainer.predict(val_loader)
            task1_scores = compute_threshold_score(pred_thresh, true_thresh)
            task1_score = task1_scores['threshold_score']

            print(f"\nTask 1 Results:")
            print(f"  Training time: {task1_time:.2f}s")
            print(f"  Threshold Score: {task1_score:.4f}")

        # ==================== TASK 2: RUNTIME PREDICTION ====================
        if args.task in ["2", "both"]:
            print("\n--- Training Task 2: Runtime Prediction ---")
            runtime_model = create_runtime_model(
                node_feat_dim=NODE_FEAT_DIM,
                edge_feat_dim=EDGE_FEAT_DIM,
                global_feat_dim=GLOBAL_FEAT_DIM,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )

            runtime_trainer = RuntimeTrainer(
                model=runtime_model,
                device=args.device,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            start = time.time()
            runtime_trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                early_stopping_patience=20,
                verbose=args.verbose,
                show_progress=True,
            )
            task2_time = time.time() - start

            # Evaluate Task 2 (using ground truth thresholds)
            true_thresh_classes = torch.tensor([
                THRESHOLD_LADDER.index(t) for t in true_thresh
            ])
            pred_runtime = runtime_trainer.predict(val_loader, true_thresh_classes)
            task2_scores = compute_runtime_score(pred_runtime, true_runtime)
            task2_score = task2_scores['runtime_score']

            print(f"\nTask 2 Results:")
            print(f"  Training time: {task2_time:.2f}s")
            print(f"  Runtime Score: {task2_score:.4f}")

        if task1_score is not None:
            all_task1_scores.append(task1_score)
        if task2_score is not None:
            all_task2_scores.append(task2_score)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Target Fidelity = 0.75)")
    print("=" * 60)
    print(f"\nBased on {args.n_runs} runs:")

    if all_task1_scores:
        print(f"\nTask 1 (Threshold Prediction):")
        print(f"  Mean Score: {np.mean(all_task1_scores):.4f} ± {np.std(all_task1_scores):.4f}")
        print(f"  Min/Max: {np.min(all_task1_scores):.4f} / {np.max(all_task1_scores):.4f}")

    if all_task2_scores:
        print(f"\nTask 2 (Runtime Prediction):")
        print(f"  Mean Score: {np.mean(all_task2_scores):.4f} ± {np.std(all_task2_scores):.4f}")
        print(f"  Min/Max: {np.min(all_task2_scores):.4f} / {np.max(all_task2_scores):.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

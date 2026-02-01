#!/usr/bin/env python3
"""
Comprehensive GNN evaluation script for diagnosing performance issues.

Evaluates both duration prediction GNN and threshold class GNN with:
- Train/val loss curves
- Overfitting analysis
- Per-class accuracy for threshold classification
- Error distribution analysis
- Comparison to simpler baselines
"""

import sys
from pathlib import Path
import json
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model import create_gnn_model, create_gnn_threshold_class_model
from gnn.dataset import (
    create_graph_data_loaders,
    create_threshold_class_graph_data_loaders,
    THRESHOLD_LADDER,
    GLOBAL_FEAT_DIM,
    GLOBAL_FEAT_DIM_THRESHOLD_CLASS,
    NUM_THRESHOLD_CLASSES,
)
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from gnn.augmentation import get_train_augmentation
from scoring import (
    compute_challenge_score,
    select_threshold_class_by_expected_score,
    mean_threshold_score,
)


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DetailedGNNTrainer:
    """Trainer with detailed logging for diagnostics."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        augmentation=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.augmentation = augmentation
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        self.criterion = nn.L1Loss()

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []
        
        for batch in loader:
            if self.augmentation is not None:
                batch = self.augmentation(batch)
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            loss = self.criterion(pred, batch.log2_runtime)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(pred.detach().cpu().numpy())
            all_labels.extend(batch.log2_runtime.cpu().numpy())
        
        return {
            "loss": total_loss / n_batches,
            "mae": mean_absolute_error(all_labels, all_preds),
            "mse": mean_squared_error(all_labels, all_preds),
        }

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, Any]:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_files = []
        all_thresholds = []
        
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.log2_runtime.cpu().numpy())
            all_files.extend(batch.file)
            if hasattr(batch, "threshold"):
                all_thresholds.extend(batch.threshold.cpu().numpy() if hasattr(batch.threshold, "cpu") else batch.threshold)
        
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        errors = preds - labels
        
        return {
            "mae": mean_absolute_error(labels, preds),
            "mse": mean_squared_error(labels, preds),
            "rmse": np.sqrt(mean_squared_error(labels, preds)),
            "predictions": preds,
            "labels": labels,
            "errors": errors,
            "files": all_files,
            "thresholds": all_thresholds if all_thresholds else None,
            "error_percentiles": {
                "p50": np.percentile(np.abs(errors), 50),
                "p90": np.percentile(np.abs(errors), 90),
                "p95": np.percentile(np.abs(errors), 95),
            },
        }

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
    ) -> Dict[str, Any]:
        history = {
            "train_loss": [],
            "train_mae": [],
            "val_loss": [],
            "val_mae": [],
            "lr": [],
        }
        best_val_mae = float("inf")
        patience_counter = 0
        best_state = None

        pbar = tqdm(range(epochs), desc="Training Duration GNN")
        for epoch in pbar:
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["mae"])
            
            history["train_loss"].append(train_metrics["loss"])
            history["train_mae"].append(train_metrics["mae"])
            history["val_loss"].append(train_metrics["loss"])
            history["val_mae"].append(val_metrics["mae"])
            history["lr"].append(current_lr)
            
            pbar.set_postfix({
                "train_mae": f"{train_metrics['mae']:.3f}",
                "val_mae": f"{val_metrics['mae']:.3f}",
                "lr": f"{current_lr:.1e}",
            })
            
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {"history": history, "best_val_mae": best_val_mae}


class DetailedThresholdClassTrainer:
    """Trainer for threshold classification with detailed logging."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        augmentation=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.augmentation = augmentation
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        all_proba = []
        all_true = []
        
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
            all_proba.append(torch.softmax(logits.detach().cpu(), dim=-1).numpy())
            all_true.extend(batch.threshold_class.cpu().tolist())
        
        proba = np.vstack(all_proba)
        true_idx = np.array(all_true, dtype=np.int64)
        chosen = select_threshold_class_by_expected_score(proba)
        
        return {
            "loss": total_loss / n_batches,
            "accuracy": float(np.mean(chosen == true_idx)),
            "threshold_score": mean_threshold_score(chosen, true_idx),
        }

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, Any]:
        self.model.eval()
        all_proba = []
        all_true = []
        all_files = []
        
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
            all_files.extend(batch.file)
        
        proba = np.vstack(all_proba)
        true_idx = np.array(all_true, dtype=np.int64)
        chosen = select_threshold_class_by_expected_score(proba)
        argmax_chosen = np.argmax(proba, axis=1)
        
        # Per-class accuracy
        per_class_acc = {}
        per_class_count = {}
        for c in range(NUM_THRESHOLD_CLASSES):
            mask = true_idx == c
            if mask.sum() > 0:
                per_class_acc[c] = float(np.mean(chosen[mask] == true_idx[mask]))
                per_class_count[c] = int(mask.sum())
        
        # Confusion matrix
        cm = confusion_matrix(true_idx, chosen, labels=list(range(NUM_THRESHOLD_CLASSES)))
        
        # Error analysis
        underpred_mask = chosen < true_idx
        overpred_mask = chosen > true_idx
        correct_mask = chosen == true_idx
        
        return {
            "accuracy": float(np.mean(chosen == true_idx)),
            "argmax_accuracy": float(np.mean(argmax_chosen == true_idx)),
            "threshold_score": mean_threshold_score(chosen, true_idx),
            "proba": proba,
            "true_idx": true_idx,
            "chosen": chosen,
            "files": all_files,
            "per_class_accuracy": per_class_acc,
            "per_class_count": per_class_count,
            "confusion_matrix": cm,
            "underpred_rate": float(underpred_mask.mean()),
            "overpred_rate": float(overpred_mask.mean()),
            "correct_rate": float(correct_mask.mean()),
        }

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
    ) -> Dict[str, Any]:
        history = {
            "train_loss": [],
            "train_acc": [],
            "train_score": [],
            "val_acc": [],
            "val_score": [],
            "lr": [],
        }
        best_val_score = -1.0
        patience_counter = 0
        best_state = None

        pbar = tqdm(range(epochs), desc="Training Threshold GNN")
        for epoch in pbar:
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["threshold_score"])
            
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["train_score"].append(train_metrics["threshold_score"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_score"].append(val_metrics["threshold_score"])
            history["lr"].append(current_lr)
            
            pbar.set_postfix({
                "train_score": f"{train_metrics['threshold_score']:.3f}",
                "val_score": f"{val_metrics['threshold_score']:.3f}",
                "underpred": f"{val_metrics['underpred_rate']:.2f}",
            })
            
            if val_metrics["threshold_score"] > best_val_score:
                best_val_score = val_metrics["threshold_score"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {"history": history, "best_val_score": best_val_score}


def compute_baselines(train_loader, val_loader) -> Dict[str, float]:
    """Compute simple baselines for comparison."""
    # Extract train/val labels
    train_labels = []
    val_labels = []
    
    for batch in train_loader:
        train_labels.extend(batch.log2_runtime.numpy())
    for batch in val_loader:
        val_labels.extend(batch.log2_runtime.numpy())
    
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    
    # Mean baseline
    mean_pred = np.mean(train_labels)
    mean_mae = mean_absolute_error(val_labels, [mean_pred] * len(val_labels))
    
    # Median baseline
    median_pred = np.median(train_labels)
    median_mae = mean_absolute_error(val_labels, [median_pred] * len(val_labels))
    
    return {
        "mean_baseline_mae": mean_mae,
        "median_baseline_mae": median_mae,
        "train_std": float(np.std(train_labels)),
        "val_std": float(np.std(val_labels)),
        "train_range": float(np.ptp(train_labels)),
        "val_range": float(np.ptp(val_labels)),
    }


def evaluate_duration_gnn(
    data_path: Path,
    circuits_dir: Path,
    n_runs: int = 3,
    hidden_dim: int = 64,
    num_layers: int = 4,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    batch_size: int = 16,
    use_augmentation: bool = True,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate duration prediction GNN with detailed diagnostics."""
    print("\n" + "=" * 70)
    print("DURATION GNN EVALUATION")
    print("=" * 70)
    
    set_all_seeds(seed)
    train_loader, val_loader = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=batch_size,
        val_fraction=0.2,
        seed=seed,
    )
    
    print(f"\nDataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Config: hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout}")
    print(f"        lr={lr}, wd={weight_decay}, aug={use_augmentation}")
    
    # Compute baselines first
    baselines = compute_baselines(train_loader, val_loader)
    print(f"\nBaselines:")
    print(f"  Mean baseline MAE: {baselines['mean_baseline_mae']:.4f}")
    print(f"  Median baseline MAE: {baselines['median_baseline_mae']:.4f}")
    print(f"  Val target range: {baselines['val_range']:.2f}, std: {baselines['val_std']:.2f}")
    
    all_results = []
    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        run_seed = seed + run
        set_all_seeds(run_seed)
        
        model = create_gnn_model(
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM,
            global_feat_dim=GLOBAL_FEAT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        augmentation = None
        if use_augmentation:
            augmentation = get_train_augmentation()
        
        trainer = DetailedGNNTrainer(
            model=model,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            augmentation=augmentation,
        )
        
        fit_result = trainer.fit(train_loader, val_loader, epochs=epochs)
        
        # Final evaluation
        train_eval = trainer.evaluate(train_loader)
        val_eval = trainer.evaluate(val_loader)
        
        # Compute challenge scores if thresholds available
        if val_eval["thresholds"] is not None:
            pred_runtime = np.power(2.0, val_eval["predictions"])
            true_runtime = np.power(2.0, val_eval["labels"])
            pred_thresh = np.array(val_eval["thresholds"])
            true_thresh = pred_thresh  # same threshold used
            challenge = compute_challenge_score(pred_thresh, true_thresh, pred_runtime, true_runtime)
        else:
            challenge = {"runtime_score": 0}
        
        result = {
            "run": run,
            "train_mae": train_eval["mae"],
            "val_mae": val_eval["mae"],
            "val_rmse": val_eval["rmse"],
            "overfitting_gap": val_eval["mae"] - train_eval["mae"],
            "error_percentiles": val_eval["error_percentiles"],
            "history": fit_result["history"],
            "challenge_runtime_score": challenge.get("runtime_score", 0),
        }
        all_results.append(result)
        
        print(f"  Train MAE: {train_eval['mae']:.4f}, Val MAE: {val_eval['mae']:.4f}")
        print(f"  Overfit gap: {result['overfitting_gap']:.4f}")
        print(f"  P90 error: {val_eval['error_percentiles']['p90']:.2f}")
    
    # Aggregate results
    avg_val_mae = np.mean([r["val_mae"] for r in all_results])
    std_val_mae = np.std([r["val_mae"] for r in all_results])
    avg_overfit = np.mean([r["overfitting_gap"] for r in all_results])
    
    summary = {
        "model": "Duration GNN",
        "n_runs": n_runs,
        "config": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "use_augmentation": use_augmentation,
        },
        "baselines": baselines,
        "results": {
            "val_mae_mean": avg_val_mae,
            "val_mae_std": std_val_mae,
            "avg_overfit_gap": avg_overfit,
            "improvement_over_mean_baseline": baselines["mean_baseline_mae"] - avg_val_mae,
        },
        "runs": all_results,
    }
    
    print(f"\n--- Duration GNN Summary ---")
    print(f"Val MAE: {avg_val_mae:.4f} ± {std_val_mae:.4f}")
    print(f"Baseline MAE: {baselines['mean_baseline_mae']:.4f}")
    print(f"Improvement: {summary['results']['improvement_over_mean_baseline']:.4f}")
    print(f"Avg overfit gap: {avg_overfit:.4f}")
    
    return summary


def evaluate_threshold_class_gnn(
    data_path: Path,
    circuits_dir: Path,
    n_runs: int = 3,
    hidden_dim: int = 64,
    num_layers: int = 4,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    batch_size: int = 16,
    use_augmentation: bool = True,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate threshold classification GNN with detailed diagnostics."""
    print("\n" + "=" * 70)
    print("THRESHOLD CLASS GNN EVALUATION")
    print("=" * 70)
    
    set_all_seeds(seed)
    train_loader, val_loader = create_threshold_class_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=batch_size,
        val_fraction=0.2,
        seed=seed,
    )
    
    print(f"\nDataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Config: hidden_dim={hidden_dim}, layers={num_layers}, dropout={dropout}")
    
    # Compute class distribution baseline
    train_classes = []
    for batch in train_loader:
        train_classes.extend(batch.threshold_class.numpy())
    train_classes = np.array(train_classes)
    
    class_counts = np.bincount(train_classes, minlength=NUM_THRESHOLD_CLASSES)
    majority_class = np.argmax(class_counts)
    
    # Compute baseline (most frequent class per decision-theoretic selection)
    val_classes = []
    for batch in val_loader:
        val_classes.extend(batch.threshold_class.numpy())
    val_classes = np.array(val_classes)
    
    random_baseline_score = mean_threshold_score(
        np.array([majority_class] * len(val_classes)), val_classes
    )
    
    print(f"\nClass distribution (train):")
    for c in range(NUM_THRESHOLD_CLASSES):
        print(f"  Class {c} (thr={THRESHOLD_LADDER[c]:3d}): {class_counts[c]:4d} ({100*class_counts[c]/len(train_classes):.1f}%)")
    print(f"\nMajority baseline score: {random_baseline_score:.4f}")
    
    all_results = []
    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        run_seed = seed + run
        set_all_seeds(run_seed)
        
        model = create_gnn_threshold_class_model(
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM,
            global_feat_dim=GLOBAL_FEAT_DIM_THRESHOLD_CLASS,
            num_classes=NUM_THRESHOLD_CLASSES,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        augmentation = None
        if use_augmentation:
            augmentation = get_train_augmentation()
        
        trainer = DetailedThresholdClassTrainer(
            model=model,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            augmentation=augmentation,
        )
        
        fit_result = trainer.fit(train_loader, val_loader, epochs=epochs)
        
        # Final evaluation
        train_eval = trainer.evaluate(train_loader)
        val_eval = trainer.evaluate(val_loader)
        
        result = {
            "run": run,
            "train_acc": train_eval["accuracy"],
            "train_score": train_eval["threshold_score"],
            "val_acc": val_eval["accuracy"],
            "val_score": val_eval["threshold_score"],
            "val_argmax_acc": val_eval["argmax_accuracy"],
            "underpred_rate": val_eval["underpred_rate"],
            "overpred_rate": val_eval["overpred_rate"],
            "per_class_accuracy": val_eval["per_class_accuracy"],
            "per_class_count": val_eval["per_class_count"],
            "confusion_matrix": val_eval["confusion_matrix"].tolist(),
            "history": fit_result["history"],
        }
        all_results.append(result)
        
        print(f"  Train score: {train_eval['threshold_score']:.4f}, Val score: {val_eval['threshold_score']:.4f}")
        print(f"  Val accuracy: {val_eval['accuracy']:.4f} (argmax: {val_eval['argmax_accuracy']:.4f})")
        print(f"  Underpred: {val_eval['underpred_rate']:.2%}, Overpred: {val_eval['overpred_rate']:.2%}")
    
    # Aggregate results
    avg_val_score = np.mean([r["val_score"] for r in all_results])
    std_val_score = np.std([r["val_score"] for r in all_results])
    avg_underpred = np.mean([r["underpred_rate"] for r in all_results])
    
    summary = {
        "model": "Threshold Class GNN",
        "n_runs": n_runs,
        "config": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "use_augmentation": use_augmentation,
        },
        "class_distribution": class_counts.tolist(),
        "majority_baseline_score": random_baseline_score,
        "results": {
            "val_score_mean": avg_val_score,
            "val_score_std": std_val_score,
            "avg_underpred_rate": avg_underpred,
            "improvement_over_baseline": avg_val_score - random_baseline_score,
        },
        "runs": all_results,
    }
    
    print(f"\n--- Threshold GNN Summary ---")
    print(f"Val Score: {avg_val_score:.4f} ± {std_val_score:.4f}")
    print(f"Baseline Score: {random_baseline_score:.4f}")
    print(f"Improvement: {summary['results']['improvement_over_baseline']:.4f}")
    print(f"Avg underprediction rate: {avg_underpred:.2%}")
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive GNN evaluation")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs per config")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of MP layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--duration-only", action="store_true", help="Only evaluate duration GNN")
    parser.add_argument("--threshold-only", action="store_true", help="Only evaluate threshold GNN")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    results = {}
    
    if not args.threshold_only:
        duration_results = evaluate_duration_gnn(
            data_path=data_path,
            circuits_dir=circuits_dir,
            n_runs=args.n_runs,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_augmentation=not args.no_augmentation,
            seed=args.seed,
            device=args.device,
        )
        results["duration"] = duration_results
    
    if not args.duration_only:
        threshold_results = evaluate_threshold_class_gnn(
            data_path=data_path,
            circuits_dir=circuits_dir,
            n_runs=args.n_runs,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_augmentation=not args.no_augmentation,
            seed=args.seed,
            device=args.device,
        )
        results["threshold"] = threshold_results
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = project_root / "results" / f"gnn_comprehensive_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    results_serializable = convert_numpy(results)
    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    if "duration" in results:
        d = results["duration"]["results"]
        print(f"\nDuration GNN:")
        print(f"  Val MAE: {d['val_mae_mean']:.4f} ± {d['val_mae_std']:.4f}")
        print(f"  Improvement over baseline: {d['improvement_over_mean_baseline']:.4f}")
    if "threshold" in results:
        t = results["threshold"]["results"]
        print(f"\nThreshold GNN:")
        print(f"  Val Score: {t['val_score_mean']:.4f} ± {t['val_score_std']:.4f}")
        print(f"  Improvement over baseline: {t['improvement_over_baseline']:.4f}")
        print(f"  Underprediction rate: {t['avg_underpred_rate']:.2%}")


if __name__ == "__main__":
    main()

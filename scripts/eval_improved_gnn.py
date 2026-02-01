#!/usr/bin/env python3
"""
Evaluate improved GNN models and compare with baseline.

Tests improvements:
1. Attention-based message passing
2. Ordinal regression for threshold classification
3. Conservative/focal losses
4. Stochastic depth regularization
"""

import sys
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model import create_gnn_model, create_gnn_threshold_class_model
from gnn.improved_model import (
    create_improved_gnn_model,
    OrdinalLoss,
    FocalLoss,
    ConservativeCrossEntropyLoss,
)
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
from scoring import select_threshold_class_by_expected_score, mean_threshold_score


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImprovedThresholdTrainer:
    """Trainer for improved threshold GNN with ordinal or focal loss."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        augmentation=None,
        loss_type: str = "ordinal",
        conservative_weight: float = 2.0,
        conservative_bias: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.augmentation = augmentation
        self.loss_type = loss_type
        self.conservative_bias = conservative_bias
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
        )
        
        if loss_type == "ordinal":
            self.criterion = OrdinalLoss(
                num_classes=NUM_THRESHOLD_CLASSES,
                conservative_weight=conservative_weight,
            )
        elif loss_type == "focal":
            self.criterion = FocalLoss(gamma=2.0)
        elif loss_type == "conservative_ce":
            self.criterion = ConservativeCrossEntropyLoss(
                base_smoothing=0.1, upward_bias=0.2
            )
        else:
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
            
            output = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            
            if self.loss_type == "ordinal":
                loss = self.criterion(output, batch.threshold_class)
                proba = self._ordinal_to_proba(output).detach().cpu().numpy()
            else:
                loss = self.criterion(output, batch.threshold_class)
                proba = torch.softmax(output.detach().cpu(), dim=-1).numpy()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            all_proba.append(proba)
            all_true.extend(batch.threshold_class.cpu().tolist())
        
        proba = np.vstack(all_proba)
        true_idx = np.array(all_true, dtype=np.int64)
        chosen = self._select_with_bias(proba)
        
        return {
            "loss": total_loss / n_batches,
            "accuracy": float(np.mean(chosen == true_idx)),
            "threshold_score": mean_threshold_score(chosen, true_idx),
        }

    def _ordinal_to_proba(self, cumulative_logits: torch.Tensor) -> torch.Tensor:
        """Convert ordinal logits to class probabilities."""
        cumulative_probs = torch.sigmoid(cumulative_logits)
        
        batch_size = cumulative_logits.shape[0]
        class_probs = torch.zeros(batch_size, NUM_THRESHOLD_CLASSES, device=cumulative_logits.device)
        
        class_probs[:, 0] = 1 - cumulative_probs[:, 0]
        for k in range(1, NUM_THRESHOLD_CLASSES - 1):
            class_probs[:, k] = cumulative_probs[:, k - 1] - cumulative_probs[:, k]
        class_probs[:, -1] = cumulative_probs[:, -1]
        
        class_probs = class_probs.clamp(min=1e-7)
        class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
        
        return class_probs

    def _select_with_bias(self, proba: np.ndarray) -> np.ndarray:
        """Select threshold with optional conservative bias."""
        chosen = select_threshold_class_by_expected_score(
            proba, conservative_bias=self.conservative_bias
        )
        return chosen

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, Any]:
        self.model.eval()
        all_proba = []
        all_true = []
        
        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_gate_type=batch.edge_gate_type,
                batch=batch.batch,
                global_features=batch.global_features,
            )
            
            if self.loss_type == "ordinal":
                proba = self._ordinal_to_proba(output).cpu().numpy()
            else:
                proba = torch.softmax(output.cpu(), dim=-1).numpy()
            
            all_proba.append(proba)
            all_true.extend(batch.threshold_class.cpu().tolist())
        
        proba = np.vstack(all_proba)
        true_idx = np.array(all_true, dtype=np.int64)
        
        chosen = self._select_with_bias(proba)
        argmax_chosen = np.argmax(proba, axis=1)
        
        underpred_mask = chosen < true_idx
        
        return {
            "accuracy": float(np.mean(chosen == true_idx)),
            "argmax_accuracy": float(np.mean(argmax_chosen == true_idx)),
            "threshold_score": mean_threshold_score(chosen, true_idx),
            "underpred_rate": float(underpred_mask.mean()),
        }

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
    ) -> Dict[str, Any]:
        history = {"train_loss": [], "train_score": [], "val_score": []}
        best_val_score = -1.0
        patience_counter = 0
        best_state = None

        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            self.scheduler.step(val_metrics["threshold_score"])
            
            history["train_loss"].append(train_metrics["loss"])
            history["train_score"].append(train_metrics["threshold_score"])
            history["val_score"].append(val_metrics["threshold_score"])
            
            pbar.set_postfix({
                "train": f"{train_metrics['threshold_score']:.3f}",
                "val": f"{val_metrics['threshold_score']:.3f}",
                "underpred": f"{val_metrics['underpred_rate']:.2f}",
            })
            
            if val_metrics["threshold_score"] > best_val_score:
                best_val_score = val_metrics["threshold_score"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                pbar.close()
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {"history": history, "best_val_score": best_val_score}


class ImprovedDurationTrainer:
    """Trainer for improved duration GNN."""

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
        }

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        
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
        
        return {"mae": mean_absolute_error(all_labels, all_preds)}

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
    ) -> Dict[str, Any]:
        history = {"train_loss": [], "val_mae": []}
        best_val_mae = float("inf")
        patience_counter = 0
        best_state = None

        pbar = tqdm(range(epochs), desc="Training Duration")
        for epoch in pbar:
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            self.scheduler.step(val_metrics["mae"])
            
            history["train_loss"].append(train_metrics["loss"])
            history["val_mae"].append(val_metrics["mae"])
            
            pbar.set_postfix({
                "train_mae": f"{train_metrics['mae']:.3f}",
                "val_mae": f"{val_metrics['mae']:.3f}",
            })
            
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                pbar.close()
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return {"history": history, "best_val_mae": best_val_mae}


def evaluate_threshold_models(
    data_path: Path,
    circuits_dir: Path,
    n_runs: int = 3,
    epochs: int = 100,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compare baseline and improved threshold classification GNNs."""
    
    print("\n" + "=" * 70)
    print("THRESHOLD GNN COMPARISON")
    print("=" * 70)
    
    set_all_seeds(seed)
    train_loader, val_loader = create_threshold_class_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=16,
        val_fraction=0.2,
        seed=seed,
    )
    
    print(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    
    configs = [
        {
            "name": "Baseline (CE)",
            "improved": False,
            "loss_type": "ce",
            "use_ordinal": False,
        },
        {
            "name": "Improved + Ordinal",
            "improved": True,
            "loss_type": "ordinal",
            "use_ordinal": True,
            "conservative_weight": 2.0,
        },
        {
            "name": "Improved + Focal",
            "improved": True,
            "loss_type": "focal",
            "use_ordinal": False,
        },
        {
            "name": "Improved + Conservative CE",
            "improved": True,
            "loss_type": "conservative_ce",
            "use_ordinal": False,
        },
    ]
    
    results = {}
    augmentation = get_train_augmentation()
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        
        config_results = []
        for run in range(n_runs):
            run_seed = seed + run
            set_all_seeds(run_seed)
            
            if config["improved"]:
                model = create_improved_gnn_model(
                    model_type="threshold",
                    hidden_dim=64,
                    num_layers=4,
                    num_heads=4,
                    dropout=0.2,
                    stochastic_depth=0.1,
                    use_ordinal=config.get("use_ordinal", False),
                )
            else:
                model = create_gnn_threshold_class_model(
                    hidden_dim=64,
                    num_layers=4,
                    dropout=0.1,
                )
            
            trainer = ImprovedThresholdTrainer(
                model=model,
                device=device,
                lr=1e-3,
                weight_decay=1e-4,
                augmentation=augmentation,
                loss_type=config["loss_type"],
                conservative_weight=config.get("conservative_weight", 1.0),
                conservative_bias=0.1,
            )
            
            fit_result = trainer.fit(train_loader, val_loader, epochs=epochs)
            val_eval = trainer.evaluate(val_loader)
            
            config_results.append({
                "val_score": val_eval["threshold_score"],
                "underpred_rate": val_eval["underpred_rate"],
                "accuracy": val_eval["accuracy"],
            })
            
            print(f"  Run {run+1}: score={val_eval['threshold_score']:.4f}, underpred={val_eval['underpred_rate']:.2%}")
        
        avg_score = np.mean([r["val_score"] for r in config_results])
        std_score = np.std([r["val_score"] for r in config_results])
        avg_underpred = np.mean([r["underpred_rate"] for r in config_results])
        
        results[config["name"]] = {
            "val_score_mean": avg_score,
            "val_score_std": std_score,
            "avg_underpred_rate": avg_underpred,
            "runs": config_results,
        }
        
        print(f"  Summary: {avg_score:.4f} ± {std_score:.4f}, underpred: {avg_underpred:.2%}")
    
    return results


def evaluate_duration_models(
    data_path: Path,
    circuits_dir: Path,
    n_runs: int = 3,
    epochs: int = 100,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compare baseline and improved duration prediction GNNs."""
    
    print("\n" + "=" * 70)
    print("DURATION GNN COMPARISON")
    print("=" * 70)
    
    set_all_seeds(seed)
    train_loader, val_loader = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=16,
        val_fraction=0.2,
        seed=seed,
    )
    
    print(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    
    configs = [
        {"name": "Baseline", "improved": False},
        {"name": "Improved (Attention + StochDepth)", "improved": True},
    ]
    
    results = {}
    augmentation = get_train_augmentation()
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        
        config_results = []
        for run in range(n_runs):
            run_seed = seed + run
            set_all_seeds(run_seed)
            
            if config["improved"]:
                model = create_improved_gnn_model(
                    model_type="duration",
                    hidden_dim=64,
                    num_layers=4,
                    num_heads=4,
                    dropout=0.2,
                    stochastic_depth=0.1,
                )
            else:
                model = create_gnn_model(
                    hidden_dim=64,
                    num_layers=4,
                    dropout=0.1,
                )
            
            trainer = ImprovedDurationTrainer(
                model=model,
                device=device,
                lr=1e-3,
                weight_decay=1e-4,
                augmentation=augmentation,
            )
            
            fit_result = trainer.fit(train_loader, val_loader, epochs=epochs)
            
            train_eval = trainer.evaluate(train_loader)
            val_eval = trainer.evaluate(val_loader)
            
            config_results.append({
                "train_mae": train_eval["mae"],
                "val_mae": val_eval["mae"],
                "overfit_gap": val_eval["mae"] - train_eval["mae"],
            })
            
            print(f"  Run {run+1}: train_mae={train_eval['mae']:.4f}, val_mae={val_eval['mae']:.4f}")
        
        avg_mae = np.mean([r["val_mae"] for r in config_results])
        std_mae = np.std([r["val_mae"] for r in config_results])
        avg_gap = np.mean([r["overfit_gap"] for r in config_results])
        
        results[config["name"]] = {
            "val_mae_mean": avg_mae,
            "val_mae_std": std_mae,
            "avg_overfit_gap": avg_gap,
            "runs": config_results,
        }
        
        print(f"  Summary: {avg_mae:.4f} ± {std_mae:.4f}, overfit_gap: {avg_gap:.4f}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate improved GNN models")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--duration-only", action="store_true")
    parser.add_argument("--threshold-only", action="store_true")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    results = {}
    
    if not args.duration_only:
        threshold_results = evaluate_threshold_models(
            data_path=data_path,
            circuits_dir=circuits_dir,
            n_runs=args.n_runs,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
        )
        results["threshold"] = threshold_results
    
    if not args.threshold_only:
        duration_results = evaluate_duration_models(
            data_path=data_path,
            circuits_dir=circuits_dir,
            n_runs=args.n_runs,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
        )
        results["duration"] = duration_results
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = project_root / "results" / f"improved_gnn_comparison_{timestamp}.json"
    
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
    
    with open(output_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)
    
    if "threshold" in results:
        print("\nThreshold Classification:")
        for name, r in results["threshold"].items():
            print(f"  {name:35s}: {r['val_score_mean']:.4f} ± {r['val_score_std']:.4f} (underpred: {r['avg_underpred_rate']:.2%})")
    
    if "duration" in results:
        print("\nDuration Prediction:")
        for name, r in results["duration"].items():
            print(f"  {name:35s}: {r['val_mae_mean']:.4f} ± {r['val_mae_std']:.4f} (overfit: {r['avg_overfit_gap']:.4f})")


if __name__ == "__main__":
    main()

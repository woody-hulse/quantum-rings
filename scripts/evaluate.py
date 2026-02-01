#!/usr/bin/env python3
"""
Model-agnostic evaluation script for quantum circuit prediction models.

Trains multiple randomly initialized models and reports mean ± std for all metrics.
Supports k-fold cross-validation for robust evaluation on small datasets.
Supports any model implementing the BaseModel interface.
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Any, Type, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    create_data_loaders,
    create_kfold_data_loaders,
    create_threshold_class_data_loaders,
    create_kfold_threshold_class_data_loaders,
    THRESHOLD_LADDER,
    FEATURE_DIM_WITHOUT_THRESHOLD,
)
from models.base import BaseModel, ThresholdClassBaseModel
from models.mlp import MLPModel
from models.mlp_threshold_class import MLPThresholdClassModel
from models.xgboost_model import XGBoostModel
from models.xgboost_threshold_class import XGBoostThresholdClassModel
from models.catboost_model import CatBoostModel
from models.catboost_threshold_class import CatBoostThresholdClassModel
from models.lightgbm_model import LightGBMModel
from scoring import compute_challenge_score

try:
    from models.gnn_threshold_class import GNNThresholdClassModel
    from gnn.dataset import create_threshold_class_graph_data_loaders
    HAS_GNN_THRESHOLD = True
except ImportError:
    GNNThresholdClassModel = None
    create_threshold_class_graph_data_loaders = None
    HAS_GNN_THRESHOLD = False


AVAILABLE_MODELS = {
    "mlp": MLPModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "lightgbm": LightGBMModel,
}

AVAILABLE_MODELS_THRESHOLD_CLASS = {
    "mlp": MLPThresholdClassModel,
    "xgboost": XGBoostThresholdClassModel,
    "catboost": CatBoostThresholdClassModel,
}
if HAS_GNN_THRESHOLD:
    AVAILABLE_MODELS_THRESHOLD_CLASS["gnn"] = GNNThresholdClassModel


def extract_labels(loader) -> tuple:
    """Extract ground truth: threshold (input) and runtime in seconds from log2_runtime."""
    all_thresh = []
    all_runtime = []
    for batch in loader:
        all_thresh.extend(batch["threshold"])
        all_runtime.extend(np.power(2.0, batch["log2_runtime"].numpy()).tolist())
    return np.array(all_thresh), np.array(all_runtime)


def extract_features(loader) -> np.ndarray:
    """Extract features from a data loader."""
    all_features = []
    for batch in loader:
        all_features.append(batch["features"].numpy())
    return np.vstack(all_features)


def aggregate_metrics(all_results: List[Dict[str, Any]], threshold_class: bool = False) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple runs, computing mean and std."""
    if not all_results:
        return {}
    
    aggregated = {}
    
    if threshold_class:
        metric_keys = [
            ("train_time", None),
            ("train_threshold_accuracy", "train_metrics"),
            ("train_expected_threshold_score", "train_metrics"),
            ("threshold_accuracy", "val_metrics"),
            ("expected_threshold_score", "val_metrics"),
        ]
    else:
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


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(
    model_class: Type,
    input_dim: int,
    seed: int,
    threshold_class: bool = False,
    **kwargs,
) -> BaseModel:
    """Factory function to create a model with the given seed."""
    set_all_seeds(seed)
    
    if threshold_class:
        if model_class == MLPThresholdClassModel:
            return MLPThresholdClassModel(
                input_dim=input_dim,
                hidden_dims=kwargs.get("hidden_dims", [128, 64, 32]),
                dropout=kwargs.get("dropout", 0.2),
                lr=kwargs.get("lr", 1e-3),
                weight_decay=kwargs.get("weight_decay", 0),
                device=kwargs.get("device", "cpu"),
                epochs=kwargs.get("epochs", 100),
                early_stopping_patience=kwargs.get("early_stopping_patience", 20),
            )
        elif model_class == XGBoostThresholdClassModel:
            return XGBoostThresholdClassModel(
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                n_estimators=kwargs.get("n_estimators", 100),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                random_state=seed,
            )
        elif model_class == CatBoostThresholdClassModel:
            return CatBoostThresholdClassModel(
                depth=kwargs.get("depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                iterations=kwargs.get("iterations", 100),
                l2_leaf_reg=kwargs.get("l2_leaf_reg", 3.0),
                random_state=seed,
                verbose=False,
            )
        elif HAS_GNN_THRESHOLD and model_class == GNNThresholdClassModel:
            return GNNThresholdClassModel(
                hidden_dim=kwargs.get("hidden_dim", 64),
                num_layers=kwargs.get("num_layers", 4),
                dropout=kwargs.get("dropout", 0.1),
                lr=kwargs.get("lr", 1e-3),
                weight_decay=kwargs.get("weight_decay", 1e-4),
                device=kwargs.get("device", "cpu"),
                epochs=kwargs.get("epochs", 100),
                early_stopping_patience=kwargs.get("early_stopping_patience", 20),
            )
        else:
            raise ValueError(f"Unknown threshold-class model class: {model_class}")
    
    if model_class == MLPModel:
        return MLPModel(
            input_dim=input_dim,
            hidden_dims=kwargs.get("hidden_dims", [128, 64, 32]),
            dropout=kwargs.get("dropout", 0.2),
            lr=kwargs.get("lr", 1e-3),
            weight_decay=kwargs.get("weight_decay", 1e-4),
            device=kwargs.get("device", "cpu"),
            epochs=kwargs.get("epochs", 100),
            early_stopping_patience=kwargs.get("early_stopping_patience", 20),
        )
    elif model_class == XGBoostModel:
        return XGBoostModel(
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            n_estimators=kwargs.get("n_estimators", 100),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            random_state=seed,
        )
    elif model_class == CatBoostModel:
        return CatBoostModel(
            depth=kwargs.get("depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            iterations=kwargs.get("iterations", 100),
            l2_leaf_reg=kwargs.get("l2_leaf_reg", 3.0),
            random_state=seed,
            verbose=False,
        )
    elif model_class == LightGBMModel:
        return LightGBMModel(
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            n_estimators=kwargs.get("n_estimators", 100),
            num_leaves=kwargs.get("num_leaves", 31),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            random_state=seed,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")


def run_single_evaluation(
    model: BaseModel,
    train_loader,
    val_loader,
    threshold_class: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate a single model instance."""
    import time
    
    start_time = time.time()
    model.fit(train_loader, val_loader, verbose=False)
    train_time = time.time() - start_time
    
    train_eval = model.evaluate(train_loader)
    
    if threshold_class:
        train_metrics = {
            "train_threshold_accuracy": train_eval["threshold_accuracy"],
            "train_expected_threshold_score": train_eval["expected_threshold_score"],
        }
        val_metrics = model.evaluate(val_loader)
        result = {
            "train_time": train_time,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
    else:
        train_metrics = {
            "train_runtime_mse": train_eval["runtime_mse"],
            "train_runtime_mae": train_eval["runtime_mae"],
        }
        val_metrics = model.evaluate(val_loader)
        features = extract_features(val_loader)
        pred_thresh, pred_runtime = model.predict(features)
        true_thresh, true_runtime = extract_labels(val_loader)
        challenge_scores = compute_challenge_score(
            pred_thresh, true_thresh, pred_runtime, true_runtime
        )
        result = {
            "train_time": train_time,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "challenge_scores": challenge_scores,
        }
    
    importance = model.get_feature_importance()
    if importance is not None:
        result["feature_importance"] = importance
    
    return result


def evaluate_model(
    model_class: Type,
    data_path: Path,
    circuits_dir: Path,
    input_dim: int,
    n_runs: int = 100,
    base_seed: int = 42,
    batch_size: int = 8,
    val_fraction: float = 0.2,
    threshold_class: bool = False,
    use_gnn_loaders: bool = False,
    **model_kwargs,
) -> Dict[str, Any]:
    """Evaluate a model class with multiple random initializations.
    
    Creates fresh data loaders for each run to ensure full reproducibility.
    """
    model_name = model_class.__name__.replace("Model", "").upper()
    
    print("\n" + "="*60)
    mode = "THRESHOLD-CLASS" if threshold_class else "DURATION"
    print(f"{model_name} MODEL EVALUATION ({mode})")
    print("="*60)
    print(f"\nTraining {n_runs} models with different initializations...")
    
    all_results = []
    
    for i in range(n_runs):
        print(f"\nRun {i + 1}/{n_runs}")
        seed = base_seed + i
        
        set_all_seeds(seed)
        if threshold_class:
            if use_gnn_loaders and create_threshold_class_graph_data_loaders is not None:
                train_loader, val_loader = create_threshold_class_graph_data_loaders(
                    data_path=data_path,
                    circuits_dir=circuits_dir,
                    batch_size=batch_size,
                    val_fraction=val_fraction,
                    seed=base_seed,
                )
            else:
                train_loader, val_loader = create_threshold_class_data_loaders(
                    data_path=data_path,
                    circuits_dir=circuits_dir,
                    batch_size=batch_size,
                    val_fraction=val_fraction,
                    seed=base_seed,
                )
        else:
            train_loader, val_loader = create_data_loaders(
                data_path=data_path,
                circuits_dir=circuits_dir,
                batch_size=batch_size,
                val_fraction=val_fraction,
                seed=base_seed,
            )
        
        model = create_model(model_class, input_dim, seed, threshold_class=threshold_class, **model_kwargs)
        result = run_single_evaluation(model, train_loader, val_loader, threshold_class=threshold_class)
        all_results.append(result)
    
    aggregated = aggregate_metrics(all_results, threshold_class=threshold_class)
    
    output = {
        "model": model_name,
        "n_runs": n_runs,
        "aggregated_metrics": aggregated,
        "all_results": all_results,
        "threshold_class": threshold_class,
    }
    
    if not threshold_class and all_results and "feature_importance" in all_results[0]:
        output["avg_feature_importance"] = {
            "threshold": np.mean(
                [r["feature_importance"]["threshold"] for r in all_results], axis=0
            ),
            "runtime": np.mean(
                [r["feature_importance"]["runtime"] for r in all_results], axis=0
            ),
        }
    elif threshold_class and all_results and "feature_importance" in all_results[0]:
        key = next(iter(all_results[0]["feature_importance"].keys()), None)
        if key:
            output["avg_feature_importance"] = {
                key: np.mean([r["feature_importance"][key] for r in all_results], axis=0)
            }
    
    return output


def evaluate_model_kfold(
    model_class: Type,
    data_path: Path,
    circuits_dir: Path,
    input_dim: int,
    n_folds: int = 5,
    n_runs_per_fold: int = 20,
    base_seed: int = 42,
    batch_size: int = 32,
    threshold_class: bool = False,
    use_gnn_loaders: bool = False,
    **model_kwargs,
) -> Dict[str, Any]:
    """Evaluate a model using k-fold cross-validation.
    
    For each fold, trains n_runs_per_fold models with different initializations.
    Reports metrics aggregated across all folds and runs.
    GNN threshold-class does not support k-fold (use single split).
    """
    model_name = model_class.__name__.replace("Model", "").upper()
    mode = "THRESHOLD-CLASS" if threshold_class else "DURATION"
    
    print("\n" + "="*60)
    print(f"{model_name} MODEL EVALUATION ({n_folds}-FOLD CV, {mode})")
    print("="*60)
    print(f"\n{n_folds} folds × {n_runs_per_fold} runs = {n_folds * n_runs_per_fold} total evaluations")
    
    all_results = []
    fold_results = []
    
    set_all_seeds(base_seed)
    if threshold_class and use_gnn_loaders:
        raise ValueError("GNN threshold-class does not support k-fold CV; use single split (--kfold 0).")
    
    if threshold_class:
        fold_loaders = create_kfold_threshold_class_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            n_folds=n_folds,
            batch_size=batch_size,
            seed=base_seed,
        )
    else:
        fold_loaders = create_kfold_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            n_folds=n_folds,
            batch_size=batch_size,
            seed=base_seed,
        )
    
    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        fold_run_results = []
        
        print(f"\nFold {fold_idx + 1}/{n_folds}")
        for run_idx in range(n_runs_per_fold):
            print(f"  Run {run_idx + 1}/{n_runs_per_fold}")
            seed = base_seed + fold_idx * 1000 + run_idx
            set_all_seeds(seed)
            
            model = create_model(model_class, input_dim, seed, threshold_class=threshold_class, **model_kwargs)
            result = run_single_evaluation(model, train_loader, val_loader, threshold_class=threshold_class)
            result["fold"] = fold_idx
            result["run"] = run_idx
            
            fold_run_results.append(result)
            all_results.append(result)
        
        fold_aggregated = aggregate_metrics(fold_run_results, threshold_class=threshold_class)
        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "metrics": fold_aggregated,
        })
    
    aggregated = aggregate_metrics(all_results, threshold_class=threshold_class)
    
    output = {
        "model": model_name,
        "n_folds": n_folds,
        "n_runs_per_fold": n_runs_per_fold,
        "n_runs": n_folds * n_runs_per_fold,
        "aggregated_metrics": aggregated,
        "fold_results": fold_results,
        "all_results": all_results,
        "threshold_class": threshold_class,
    }
    
    if all_results and "feature_importance" in all_results[0]:
        fi = all_results[0]["feature_importance"]
        keys = list(fi.keys())
        output["avg_feature_importance"] = {
            k: np.mean([r["feature_importance"][k] for r in all_results], axis=0)
            for k in keys
        }
    
    return output


def print_model_report(results: Dict[str, Any]) -> None:
    """Print a detailed report for a model evaluation."""
    model_name = results["model"]
    n_runs = results["n_runs"]
    metrics = results["aggregated_metrics"]
    threshold_class = results.get("threshold_class", False)
    is_kfold = "n_folds" in results
    
    print("\n" + "="*60)
    mode = "THRESHOLD-CLASS" if threshold_class else "DURATION"
    print(f"{model_name} MODEL REPORT ({mode})")
    if is_kfold:
        n_folds = results["n_folds"]
        n_runs_per_fold = results["n_runs_per_fold"]
        print(f"{n_folds}-fold cross-validation, {n_runs_per_fold} runs per fold")
        print(f"Total evaluations: {n_runs}")
    else:
        print(f"Based on {n_runs} runs with different initializations")
    print("="*60)
    
    if is_kfold and "fold_results" in results:
        print("\n--- Per-Fold Results ---")
        if threshold_class:
            print(f"{'Fold':<6} {'Train':>8} {'Val':>6} {'Acc':>10} {'Score':>10}")
            print("-" * 44)
            for fold_info in results["fold_results"]:
                fold_metrics = fold_info["metrics"]
                acc = fold_metrics.get("threshold_accuracy", {}).get("mean", 0)
                score = fold_metrics.get("expected_threshold_score", {}).get("mean", 0)
                print(f"{fold_info['fold']+1:<6} {fold_info['n_train']:>8} {fold_info['n_val']:>6} {acc:>10.4f} {score:>10.4f}")
        else:
            print(f"{'Fold':<6} {'Train':>8} {'Val':>6} {'MAE (log2)':>12} {'Combined':>12}")
            print("-" * 50)
            for fold_info in results["fold_results"]:
                fold_metrics = fold_info["metrics"]
                runtime_mae = fold_metrics.get("runtime_mae", {}).get("mean", 0)
                combined = fold_metrics.get("combined_score", {}).get("mean", 0)
                print(f"{fold_info['fold']+1:<6} {fold_info['n_train']:>8} {fold_info['n_val']:>6} {runtime_mae:>12.4f} {combined:>12.4f}")
        print()
    
    print("--- Overfitting Check (Train vs Val) ---")
    print(f"{'Metric':<30} {'Train Mean':>12} {'Val Mean':>12} {'Gap':>12}")
    print("-" * 68)
    
    if threshold_class:
        overfit_metrics = [
            ("Threshold accuracy", "train_threshold_accuracy", "threshold_accuracy", True),
            ("Expected threshold score", "train_expected_threshold_score", "expected_threshold_score", True),
        ]
    else:
        overfit_metrics = [
            ("Runtime MAE (log2)", "train_runtime_mae", "runtime_mae", False),
        ]
    
    for display_name, train_key, val_key, higher_is_better in overfit_metrics:
        if train_key in metrics and val_key in metrics:
            train_val = metrics[train_key]["mean"]
            val_val = metrics[val_key]["mean"]
            gap = (train_val - val_val) if higher_is_better else (val_val - train_val)
            print(f"{display_name:<30} {train_val:>12.4f} {val_val:>12.4f} {gap:>+12.4f}")
    
    print(f"\n--- Validation Metrics (Aggregated) ---")
    print(f"{'Metric':<35} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 83)
    
    if threshold_class:
        metric_display = [
            ("Training Time (s)", "train_time"),
            ("Val Threshold Accuracy", "threshold_accuracy"),
            ("Val Expected Threshold Score", "expected_threshold_score"),
        ]
    else:
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
            print(f"{display_name:<35} {m['mean']:>12.4f} {m['std']:>12.4f} "
                  f"{m['min']:>12.4f} {m['max']:>12.4f}")
    
    print("\n" + "-"*60)
    if threshold_class and "expected_threshold_score" in metrics:
        m = metrics["expected_threshold_score"]
        print(f"Final Expected Threshold Score: {m['mean']:.4f} ± {m['std']:.4f}")
    elif not threshold_class and "combined_score" in metrics:
        m = metrics["combined_score"]
        print(f"Final Challenge Score: {m['mean']:.4f} ± {m['std']:.4f}")
    print("="*60)
    
    if "avg_feature_importance" in results:
        importance = results["avg_feature_importance"]
        key = "runtime" if "runtime" in importance else next(iter(importance.keys()), None)
        if key:
            label = "threshold_class" if threshold_class else "runtime"
            print(f"\nTop 10 features ({label}, averaged across runs):")
            arr = importance[key]
            top_idx = np.argsort(arr)[::-1][: min(10, len(arr))]
            for i, idx in enumerate(top_idx):
                print(f"  {i+1:2d}. Feature {idx:3d}: {arr[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models with multiple random initializations and optional k-fold CV"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Which model to evaluate (mlp, xgboost, catboost, lightgbm; with --threshold-class also: gnn)"
    )
    parser.add_argument(
        "--threshold-class", action="store_true",
        help="Use threshold-class models (no duration/threshold in features; predict P(class), select by max expected score)"
    )
    parser.add_argument("--n-runs", type=int, default=100, 
                        help="Number of models to train per fold (default: 100, or 20 if using k-fold)")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="MLP training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size (default: 32)")
    parser.add_argument("--val-fraction", type=float, default=0.2, 
                        help="Validation fraction for single split (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Base random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Device for MLP (cpu/cuda/mps)")
    parser.add_argument("--kfold", type=int, default=0,
                        help="Number of folds for cross-validation (0=disabled, default: 0)")
    parser.add_argument("--scoring-loss", action="store_true",
                        help="Use challenge-aligned scoring loss (multiplicative) instead of standard losses")
    parser.add_argument("--asymmetric-loss", action="store_true",
                        help="Use asymmetric log2 loss (penalizes threshold underprediction more steeply)")
    parser.add_argument("--log2-mse-loss", action="store_true",
                        help="Use CrossEntropy + MSE in log2 space for runtime (well-aligned with scoring)")
    parser.add_argument("--underprediction-steepness", type=float, default=5.0,
                        help="Steepness multiplier for underprediction penalty (default: 5.0, only with --asymmetric-loss)")
    parser.add_argument("--underprediction-penalty", type=float, default=2.0,
                        help="Penalty multiplier for threshold underprediction (default: 2.0, only for continuous_mlp)")
    parser.add_argument("--threshold-weight", type=float, default=1.0,
                        help="Weight for threshold loss component (default: 1.0)")
    parser.add_argument("--runtime-weight", type=float, default=1.0,
                        help="Weight for runtime loss component (default: 1.0)")
    parser.add_argument("--inference-strategy", type=str, default="argmax",
                        choices=["argmax", "decision_theoretic", "shift"],
                        help="Inference strategy for threshold prediction (default: argmax). "
                             "'decision_theoretic' maximizes expected challenge score, "
                             "'shift' adds a constant offset to argmax predictions.")
    parser.add_argument("--inference-shift", type=int, default=1,
                        help="Number of classes to shift up (only used with --inference-strategy shift, default: 1)")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    threshold_class = args.threshold_class
    use_kfold = args.kfold > 1
    
    if threshold_class:
        model_choices = AVAILABLE_MODELS_THRESHOLD_CLASS
        if args.model not in model_choices:
            print(f"Error: with --threshold-class, model must be one of: {list(model_choices.keys())}")
            sys.exit(1)
        if args.model == "gnn" and use_kfold:
            print("Error: GNN threshold-class does not support k-fold; use --kfold 0")
            sys.exit(1)
    else:
        model_choices = AVAILABLE_MODELS
        if args.model not in model_choices:
            print(f"Error: model must be one of: {list(model_choices.keys())}")
            sys.exit(1)
    
    model_class = model_choices[args.model]
    use_gnn_loaders = threshold_class and args.model == "gnn"
    
    print("="*60)
    print("QUANTUM CIRCUIT MODEL EVALUATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Mode: {'threshold-class' if threshold_class else 'duration'}")
    if use_kfold:
        n_runs_per_fold = min(args.n_runs, 20) if args.n_runs == 100 else args.n_runs
        print(f"  Cross-validation: {args.kfold}-fold")
        print(f"  Runs per fold: {n_runs_per_fold}")
        print(f"  Total evaluations: {args.kfold * n_runs_per_fold}")
    else:
        print(f"  Number of runs: {args.n_runs}")
        print(f"  Validation fraction: {args.val_fraction}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Base seed: {args.seed}")
    print(f"  Device: {args.device}")
    if not threshold_class:
        if args.log2_mse_loss:
            print(f"  Loss: CrossEntropy + MSE in log2 space")
        elif args.asymmetric_loss:
            print(f"  Loss: Asymmetric log2 loss (steepness={args.underprediction_steepness})")
        elif args.scoring_loss:
            print(f"  Loss: Challenge-aligned scoring loss (multiplicative)")
        else:
            print(f"  Loss: Standard (CrossEntropy + MSE)")
        if args.inference_strategy == "decision_theoretic":
            print(f"  Inference: Decision-theoretic (maximize expected score)")
        elif args.inference_strategy == "shift":
            print(f"  Inference: Shift (argmax + {args.inference_shift})")
        else:
            print(f"  Inference: Argmax")
    
    set_all_seeds(args.seed)
    
    print("\nLoading data...")
    
    if threshold_class and not use_gnn_loaders:
        sample_loader, _ = create_threshold_class_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        input_dim = FEATURE_DIM_WITHOUT_THRESHOLD
    elif use_gnn_loaders:
        input_dim = 0
    else:
        sample_loader, _ = create_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        sample_batch = next(iter(sample_loader))
        input_dim = sample_batch["features"].shape[1]
    
    print(f"\nDataset info:")
    print(f"  Feature dimension: {input_dim}")
    
    model_kwargs = {
        "device": args.device,
        "epochs": args.epochs,
        "early_stopping_patience": 20,
    }
    if not threshold_class:
        model_kwargs.update({
            "use_scoring_loss": args.scoring_loss,
            "use_asymmetric_loss": args.asymmetric_loss,
            "use_log2_mse_loss": args.log2_mse_loss,
            "underprediction_steepness": args.underprediction_steepness,
            "underprediction_penalty": args.underprediction_penalty,
            "threshold_weight": args.threshold_weight,
            "runtime_weight": args.runtime_weight,
            "inference_strategy": args.inference_strategy,
            "inference_shift": args.inference_shift,
        })
    
    if use_kfold:
        n_runs_per_fold = min(args.n_runs, 20) if args.n_runs == 100 else args.n_runs
        results = evaluate_model_kfold(
            model_class=model_class,
            data_path=data_path,
            circuits_dir=circuits_dir,
            input_dim=input_dim,
            n_folds=args.kfold,
            n_runs_per_fold=n_runs_per_fold,
            base_seed=args.seed,
            batch_size=args.batch_size,
            threshold_class=threshold_class,
            use_gnn_loaders=use_gnn_loaders,
            **model_kwargs,
        )
    else:
        if not use_gnn_loaders:
            if threshold_class:
                train_loader, val_loader = create_threshold_class_data_loaders(
                    data_path=data_path,
                    circuits_dir=circuits_dir,
                    batch_size=args.batch_size,
                    val_fraction=args.val_fraction,
                    seed=args.seed,
                )
            else:
                train_loader, val_loader = create_data_loaders(
                    data_path=data_path,
                    circuits_dir=circuits_dir,
                    batch_size=args.batch_size,
                    val_fraction=args.val_fraction,
                    seed=args.seed,
                )
            print(f"  Train samples: {len(train_loader.dataset)}")
            print(f"  Val samples: {len(val_loader.dataset)}")
            if len(val_loader.dataset) < 50 and not threshold_class:
                print(f"\n  WARNING: Small validation set ({len(val_loader.dataset)} samples).")
                print(f"           Consider using --kfold 5 for more robust evaluation.")
        
        results = evaluate_model(
            model_class=model_class,
            data_path=data_path,
            circuits_dir=circuits_dir,
            input_dim=input_dim,
            n_runs=args.n_runs,
            base_seed=args.seed,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            threshold_class=threshold_class,
            use_gnn_loaders=use_gnn_loaders,
            **model_kwargs,
        )
    
    print_model_report(results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

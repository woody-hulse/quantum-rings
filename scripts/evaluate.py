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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import create_data_loaders, create_kfold_data_loaders, THRESHOLD_LADDER
from models.base import BaseModel
from models.mlp import MLPModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.lightgbm_model import LightGBMModel
from scoring import compute_challenge_score


AVAILABLE_MODELS = {
    "mlp": MLPModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "lightgbm": LightGBMModel,
}


def extract_labels(loader) -> tuple:
    """Extract ground truth labels from a data loader."""
    all_thresh = []
    all_runtime = []
    
    for batch in loader:
        thresh_classes = batch["threshold_class"].tolist()
        thresh_values = [THRESHOLD_LADDER[c] for c in thresh_classes]
        all_thresh.extend(thresh_values)
        all_runtime.extend(np.expm1(batch["log_runtime"].numpy()).tolist())
    
    return np.array(all_thresh), np.array(all_runtime)


def extract_features(loader) -> np.ndarray:
    """Extract features from a data loader."""
    all_features = []
    for batch in loader:
        all_features.append(batch["features"].numpy())
    return np.vstack(all_features)


def aggregate_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple runs, computing mean and std."""
    if not all_results:
        return {}
    
    aggregated = {}
    
    metric_keys = [
        ("train_time", None),
        # Training metrics (for overfitting detection)
        ("train_threshold_accuracy", "train_metrics"),
        ("train_runtime_mse", "train_metrics"),
        ("train_runtime_mae", "train_metrics"),
        # Validation metrics
        ("threshold_accuracy", "val_metrics"),
        ("runtime_mse", "val_metrics"),
        ("runtime_mae", "val_metrics"),
        # Challenge scores (validation only)
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
    model_class: Type[BaseModel],
    input_dim: int,
    seed: int,
    **kwargs,
) -> BaseModel:
    """Factory function to create a model with the given seed."""
    set_all_seeds(seed)
    
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
            use_scoring_loss=kwargs.get("use_scoring_loss", False),
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
) -> Dict[str, Any]:
    """Train and evaluate a single model instance."""
    import time
    
    start_time = time.time()
    model.fit(train_loader, val_loader, verbose=False)
    train_time = time.time() - start_time
    
    # Evaluate on training set (for overfitting detection)
    train_eval = model.evaluate(train_loader)
    train_metrics = {
        "train_threshold_accuracy": train_eval["threshold_accuracy"],
        "train_runtime_mse": train_eval["runtime_mse"],
        "train_runtime_mae": train_eval["runtime_mae"],
    }
    
    # Evaluate on validation set
    val_metrics = model.evaluate(val_loader)
    
    # Compute challenge scores on validation set only
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
    model_class: Type[BaseModel],
    data_path: Path,
    circuits_dir: Path,
    input_dim: int,
    n_runs: int = 100,
    base_seed: int = 42,
    batch_size: int = 8,
    val_fraction: float = 0.2,
    **model_kwargs,
) -> Dict[str, Any]:
    """Evaluate a model class with multiple random initializations.
    
    Creates fresh data loaders for each run to ensure full reproducibility.
    """
    model_name = model_class.__name__.replace("Model", "").upper()
    
    print("\n" + "="*60)
    print(f"{model_name} MODEL EVALUATION")
    print("="*60)
    print(f"\nTraining {n_runs} models with different initializations...")
    
    all_results = []
    
    for i in tqdm(range(n_runs), desc=f"Training {model_name} models"):
        seed = base_seed + i
        
        # Create fresh loaders for each run to ensure reproducibility
        # Each run uses a different seed for model init, but same seed for data split
        set_all_seeds(seed)
        train_loader, val_loader = create_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=batch_size,
            val_fraction=val_fraction,
            seed=base_seed,  # Keep data split consistent across runs
        )
        
        model = create_model(model_class, input_dim, seed, **model_kwargs)
        result = run_single_evaluation(model, train_loader, val_loader)
        all_results.append(result)
    
    aggregated = aggregate_metrics(all_results)
    
    output = {
        "model": model_name,
        "n_runs": n_runs,
        "aggregated_metrics": aggregated,
        "all_results": all_results,
    }
    
    if all_results and "feature_importance" in all_results[0]:
        output["avg_feature_importance"] = {
            "threshold": np.mean(
                [r["feature_importance"]["threshold"] for r in all_results], axis=0
            ),
            "runtime": np.mean(
                [r["feature_importance"]["runtime"] for r in all_results], axis=0
            ),
        }
    
    return output


def evaluate_model_kfold(
    model_class: Type[BaseModel],
    data_path: Path,
    circuits_dir: Path,
    input_dim: int,
    n_folds: int = 5,
    n_runs_per_fold: int = 20,
    base_seed: int = 42,
    batch_size: int = 32,
    **model_kwargs,
) -> Dict[str, Any]:
    """Evaluate a model using k-fold cross-validation.
    
    For each fold, trains n_runs_per_fold models with different initializations.
    Reports metrics aggregated across all folds and runs.
    
    Args:
        model_class: Model class to evaluate
        data_path: Path to data JSON
        circuits_dir: Path to circuits directory
        input_dim: Feature dimension
        n_folds: Number of cross-validation folds
        n_runs_per_fold: Number of model runs per fold
        base_seed: Base random seed
        batch_size: Batch size for data loaders
        **model_kwargs: Additional model arguments
    """
    model_name = model_class.__name__.replace("Model", "").upper()
    
    print("\n" + "="*60)
    print(f"{model_name} MODEL EVALUATION ({n_folds}-FOLD CV)")
    print("="*60)
    print(f"\n{n_folds} folds × {n_runs_per_fold} runs = {n_folds * n_runs_per_fold} total evaluations")
    
    all_results = []
    fold_results = []
    
    # Create all fold loaders once
    set_all_seeds(base_seed)
    fold_loaders = create_kfold_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        n_folds=n_folds,
        batch_size=batch_size,
        seed=base_seed,
    )
    
    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        fold_run_results = []
        
        desc = f"Fold {fold_idx+1}/{n_folds}"
        for run_idx in tqdm(range(n_runs_per_fold), desc=desc):
            seed = base_seed + fold_idx * 1000 + run_idx
            set_all_seeds(seed)
            
            model = create_model(model_class, input_dim, seed, **model_kwargs)
            result = run_single_evaluation(model, train_loader, val_loader)
            result["fold"] = fold_idx
            result["run"] = run_idx
            
            fold_run_results.append(result)
            all_results.append(result)
        
        fold_aggregated = aggregate_metrics(fold_run_results)
        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_loader.dataset),
            "n_val": len(val_loader.dataset),
            "metrics": fold_aggregated,
        })
    
    # Aggregate across all folds and runs
    aggregated = aggregate_metrics(all_results)
    
    output = {
        "model": model_name,
        "n_folds": n_folds,
        "n_runs_per_fold": n_runs_per_fold,
        "n_runs": n_folds * n_runs_per_fold,
        "aggregated_metrics": aggregated,
        "fold_results": fold_results,
        "all_results": all_results,
    }
    
    if all_results and "feature_importance" in all_results[0]:
        output["avg_feature_importance"] = {
            "threshold": np.mean(
                [r["feature_importance"]["threshold"] for r in all_results], axis=0
            ),
            "runtime": np.mean(
                [r["feature_importance"]["runtime"] for r in all_results], axis=0
            ),
        }
    
    return output


def print_model_report(results: Dict[str, Any]) -> None:
    """Print a detailed report for a model evaluation."""
    model_name = results["model"]
    n_runs = results["n_runs"]
    metrics = results["aggregated_metrics"]
    
    # Check if this is a k-fold CV result
    is_kfold = "n_folds" in results
    
    print("\n" + "="*60)
    print(f"{model_name} MODEL REPORT")
    if is_kfold:
        n_folds = results["n_folds"]
        n_runs_per_fold = results["n_runs_per_fold"]
        print(f"{n_folds}-fold cross-validation, {n_runs_per_fold} runs per fold")
        print(f"Total evaluations: {n_runs}")
    else:
        print(f"Based on {n_runs} runs with different initializations")
    print("="*60)
    
    # Per-fold summary for k-fold CV
    if is_kfold and "fold_results" in results:
        print("\n--- Per-Fold Results ---")
        print(f"{'Fold':<6} {'Train':>8} {'Val':>6} {'Thresh Acc':>12} {'Combined':>12}")
        print("-" * 50)
        for fold_info in results["fold_results"]:
            fold_idx = fold_info["fold"]
            n_train = fold_info["n_train"]
            n_val = fold_info["n_val"]
            fold_metrics = fold_info["metrics"]
            thresh_acc = fold_metrics.get("threshold_accuracy", {}).get("mean", 0)
            combined = fold_metrics.get("combined_score", {}).get("mean", 0)
            print(f"{fold_idx+1:<6} {n_train:>8} {n_val:>6} {thresh_acc:>12.4f} {combined:>12.4f}")
        print()
    
    # Training vs Validation comparison (for overfitting detection)
    print("--- Overfitting Check (Train vs Val) ---")
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
            if higher_is_better:
                gap = train_val - val_val
            else:
                gap = val_val - train_val
            gap_str = f"{gap:+.4f}"
            print(f"{display_name:<25} {train_val:>12.4f} {val_val:>12.4f} {gap_str:>12}")
    
    # Full metrics table
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
            print(f"{display_name:<30} {m['mean']:>12.4f} {m['std']:>12.4f} "
                  f"{m['min']:>12.4f} {m['max']:>12.4f}")
    
    print("\n" + "-"*60)
    if "combined_score" in metrics:
        m = metrics["combined_score"]
        print(f"Final Score: {m['mean']:.4f} ± {m['std']:.4f}")
    print("="*60)
    
    if "avg_feature_importance" in results:
        importance = results["avg_feature_importance"]
        print(f"\nTop 10 features (threshold model, averaged across runs):")
        top_idx = np.argsort(importance["threshold"])[::-1][:10]
        for i, idx in enumerate(top_idx):
            print(f"  {i+1:2d}. Feature {idx:3d}: {importance['threshold'][idx]:.4f}")
        
        print(f"\nTop 10 features (runtime model, averaged across runs):")
        top_idx = np.argsort(importance["runtime"])[::-1][:10]
        for i, idx in enumerate(top_idx):
            print(f"  {i+1:2d}. Feature {idx:3d}: {importance['runtime'][idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models with multiple random initializations and optional k-fold CV"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(AVAILABLE_MODELS.keys()),
        help="Which model to evaluate"
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
                        help="Use challenge-aligned scoring loss (multiplicative) instead of standard losses") # bad
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    use_kfold = args.kfold > 1
    
    print("="*60)
    print("QUANTUM CIRCUIT MODEL EVALUATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    if use_kfold:
        n_runs_per_fold = min(args.n_runs, 20) if args.n_runs == 100 else args.n_runs
        print(f"  Cross-validation: {args.kfold}-fold")
        print(f"  Runs per fold: {n_runs_per_fold}")
        print(f"  Total evaluations: {args.kfold * n_runs_per_fold}")
    else:
        print(f"  Number of runs: {args.n_runs}")
        print(f"  Validation fraction: {args.val_fraction}")
    print(f"  MLP epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Base seed: {args.seed}")
    print(f"  Device: {args.device}")
    if args.scoring_loss:
        print(f"  Loss: Challenge-aligned scoring loss (multiplicative)")
    else:
        print(f"  Loss: Standard (CrossEntropy + MSE)")
    
    # Set global seeds for reproducibility
    set_all_seeds(args.seed)
    
    print("\nLoading data...")
    
    # Get input dimension from a sample loader
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
    
    model_class = AVAILABLE_MODELS[args.model]
    
    model_kwargs = {
        "device": args.device,
        "epochs": args.epochs,
        "use_scoring_loss": args.scoring_loss,
    }
    
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
            **model_kwargs,
        )
    else:
        # Show dataset split info for non-kfold
        train_loader, val_loader = create_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        
        if len(val_loader.dataset) < 50:
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
            **model_kwargs,
        )
    
    print_model_report(results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

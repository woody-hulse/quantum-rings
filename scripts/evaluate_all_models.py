#!/usr/bin/env python3
"""
Unified evaluation script for all model types with cross-validation support.

Supports:
- ML models (MLP, XGBoost) that use feature vectors
- Component models (Analytical, BondDimension, etc.) that use QASM parsing

Reports mean ± std across multiple runs/folds for fair comparison.
"""

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Any, Type
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    create_data_loaders,
    create_kfold_threshold_class_data_loaders,
    THRESHOLD_LADDER,
    QuantumCircuitDataset,
    FEATURE_DIM_WITHOUT_THRESHOLD,
)
from torch.utils.data import DataLoader

from models.base import BaseModel
from models.component import (
    AnalyticalCostModelWrapper,
    LearnedComponentModelWrapper,
    BondDimensionModelWrapper,
    EntanglementBudgetModelWrapper,
)

try:
    from models.mlp import MLPModel
    from models.mlp_threshold_class import MLPThresholdClassModel
    HAS_MLP = True
except ImportError:
    MLPThresholdClassModel = None
    HAS_MLP = False

try:
    from models.xgboost_model import XGBoostModel
    from models.xgboost_threshold_class import XGBoostThresholdClassModel
    HAS_XGB = True
except ImportError:
    XGBoostThresholdClassModel = None
    HAS_XGB = False

try:
    from models.catboost_model import CatBoostModel
    from models.catboost_threshold_class import CatBoostThresholdClassModel
    HAS_CATBOOST = True
except ImportError:
    CatBoostThresholdClassModel = None
    HAS_CATBOOST = False

from scoring import compute_challenge_score


def collate_fn(batch):
    """Collate for duration-only: features, log2_runtime, threshold, file, backend, precision."""
    import torch
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "log2_runtime": torch.stack([item["log2_runtime"] for item in batch]),
        "threshold": [item["threshold"] for item in batch],
        "file": [item["file"] for item in batch],
        "backend": [item["backend"] for item in batch],
        "precision": [item["precision"] for item in batch],
    }


def create_fold_loaders_threshold_class(
    data_path: Path,
    circuits_dir: Path,
    fold_idx: int,
    n_folds: int,
    batch_size: int = 32,
    seed: int = 42,
):
    """Create train/val loaders for a specific fold (threshold-class)."""
    fold_loaders = create_kfold_threshold_class_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        n_folds=n_folds,
        batch_size=batch_size,
        seed=seed,
    )
    return fold_loaders[fold_idx]


def create_fold_loaders(
    data_path: Path,
    circuits_dir: Path,
    fold_idx: int,
    n_folds: int,
    batch_size: int = 32,
    seed: int = 42,
):
    """Create train/val loaders for a specific fold (duration)."""
    full_dataset = QuantumCircuitDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="train",
        val_fraction=0.0,
        seed=seed,
    )
    
    circuit_files = list(set(r.file for r in full_dataset.results))
    np.random.seed(seed)
    np.random.shuffle(circuit_files)
    
    fold_size = len(circuit_files) // n_folds
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < n_folds - 1 else len(circuit_files)
    
    val_files = set(circuit_files[val_start:val_end])
    train_files = set(circuit_files) - val_files
    
    train_results = [r for r in full_dataset.results if r.file in train_files]
    val_results = [r for r in full_dataset.results if r.file in val_files]
    
    class FoldDataset(QuantumCircuitDataset):
        def __init__(self, base_dataset, results_subset):
            self.circuits_dir = base_dataset.circuits_dir
            self.circuit_info = base_dataset.circuit_info
            self.family_to_idx = base_dataset.family_to_idx
            self.results = results_subset
            self._feature_cache = base_dataset._feature_cache
            self._samples = [
                r for r in results_subset
                if r.forward_wall_s is not None and r.forward_wall_s > 0 and r.selected_threshold is not None
            ]
    
    train_dataset = FoldDataset(full_dataset, train_results)
    val_dataset = FoldDataset(full_dataset, val_results)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def extract_labels(loader) -> tuple:
    """Extract ground truth: threshold (input) and runtime in seconds."""
    all_thresh = []
    all_runtime = []
    for batch in loader:
        all_thresh.extend(batch["threshold"])
        all_runtime.extend(np.power(2.0, batch["log2_runtime"].numpy()).tolist())
    return np.array(all_thresh), np.array(all_runtime)


def extract_features(loader) -> np.ndarray:
    """Extract features from loader."""
    all_features = []
    for batch in loader:
        all_features.append(batch["features"].numpy())
    return np.vstack(all_features)


def run_single_fold(
    model: BaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    is_component_model: bool = False,
    threshold_class: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate a model on a single fold."""
    start_time = time.time()
    model.fit(train_loader, val_loader, verbose=False)
    train_time = time.time() - start_time
    
    train_metrics = model.evaluate(train_loader)
    val_metrics = model.evaluate(val_loader)
    
    if threshold_class:
        return {
            "train_time": train_time,
            "train_metrics": {
                "train_threshold_accuracy": train_metrics["threshold_accuracy"],
                "train_expected_threshold_score": train_metrics["expected_threshold_score"],
            },
            "val_metrics": val_metrics,
            "challenge_scores": {
                "threshold_score": val_metrics["expected_threshold_score"],
                "runtime_score": 0.0,
                "combined_score": val_metrics["expected_threshold_score"],
            },
        }
    
    if is_component_model:
        pred_thresh, pred_runtime = model.predict_from_loader(val_loader)
    else:
        features = extract_features(val_loader)
        pred_thresh, pred_runtime = model.predict(features)
    
    true_thresh, true_runtime = extract_labels(val_loader)
    
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


def create_model(model_name: str, input_dim: int, seed: int = 42, threshold_class: bool = False) -> tuple:
    """Create a model by name. Returns (model, is_component_model)."""
    np.random.seed(seed)
    
    if threshold_class:
        if model_name == "mlp" and HAS_MLP and MLPThresholdClassModel is not None:
            import torch
            torch.manual_seed(seed)
            return MLPThresholdClassModel(
                input_dim=input_dim,
                hidden_dims=[128, 64, 32],
                dropout=0.2,
                device="cpu",
                epochs=100,
                early_stopping_patience=20,
            ), False
        elif model_name == "xgboost" and HAS_XGB and XGBoostThresholdClassModel is not None:
            return XGBoostThresholdClassModel(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=seed,
            ), False
        elif model_name == "catboost" and HAS_CATBOOST and CatBoostThresholdClassModel is not None:
            return CatBoostThresholdClassModel(
                depth=6,
                learning_rate=0.1,
                iterations=100,
                random_state=seed,
            ), False
        else:
            raise ValueError(f"Unknown or unavailable threshold-class model: {model_name}")
    
    if model_name == "analytical":
        return AnalyticalCostModelWrapper(calibrate=True), True
    elif model_name == "analytical_fixed":
        return AnalyticalCostModelWrapper(calibrate=False), True
    elif model_name == "learned_component":
        return LearnedComponentModelWrapper(), True
    elif model_name == "bond_dimension":
        return BondDimensionModelWrapper(), True
    elif model_name == "entanglement_budget":
        return EntanglementBudgetModelWrapper(), True
    elif model_name == "mlp" and HAS_MLP:
        import torch
        torch.manual_seed(seed)
        return MLPModel(
            input_dim=input_dim,
            hidden_dims=[128, 64, 32],
            dropout=0.2,
            device="cpu",
            epochs=100,
            early_stopping_patience=20,
        ), False
    elif model_name == "xgboost" and HAS_XGB:
        return XGBoostModel(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=seed,
        ), False
    elif model_name == "catboost" and HAS_CATBOOST:
        return CatBoostModel(
            depth=6,
            learning_rate=0.1,
            iterations=100,
            random_state=seed,
        ), False
    else:
        raise ValueError(f"Unknown model: {model_name}")


def aggregate_results(all_results: List[Dict], threshold_class: bool = False) -> Dict[str, Dict[str, float]]:
    """Compute mean ± std across runs."""
    metrics = {}
    
    if threshold_class:
        keys = [
            ("train_time", None),
            ("train_threshold_accuracy", "train_metrics"),
            ("train_expected_threshold_score", "train_metrics"),
            ("threshold_accuracy", "val_metrics"),
            ("expected_threshold_score", "val_metrics"),
            ("combined_score", "challenge_scores"),
        ]
    else:
        keys = [
            ("train_time", None),
            ("train_runtime_mse", "train_metrics"),
            ("train_runtime_mae", "train_metrics"),
            ("runtime_mse", "val_metrics"),
            ("runtime_mae", "val_metrics"),
            ("threshold_score", "challenge_scores"),
            ("runtime_score", "challenge_scores"),
            ("combined_score", "challenge_scores"),
        ]
    
    for key, parent in keys:
        values = []
        for r in all_results:
            if parent:
                val = r.get(parent, {}).get(key)
            else:
                val = r.get(key)
            if val is not None:
                values.append(val)
        
        if values:
            metrics[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
    
    return metrics


def evaluate_model_cv(
    model_name: str,
    data_path: Path,
    circuits_dir: Path,
    n_folds: int = 5,
    n_runs: int = 3,
    batch_size: int = 32,
    base_seed: int = 42,
    threshold_class: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a model with cross-validation and multiple runs.
    
    Total evaluations = n_folds * n_runs
    """
    mode = "THRESHOLD-CLASS" if threshold_class else "DURATION"
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} MODEL EVALUATION ({mode})")
    print(f"{'='*60}")
    print(f"Running {n_folds}-fold CV with {n_runs} runs each...")
    
    all_results = []
    input_dim = FEATURE_DIM_WITHOUT_THRESHOLD if threshold_class else None
    
    create_fold_fn = create_fold_loaders_threshold_class if threshold_class else create_fold_loaders
    
    for run in range(n_runs):
        run_seed = base_seed + run * 1000
        
        for fold in range(n_folds):
            train_loader, val_loader = create_fold_fn(
                data_path=data_path,
                circuits_dir=circuits_dir,
                fold_idx=fold,
                n_folds=n_folds,
                batch_size=batch_size,
                seed=run_seed,
            )
            
            if input_dim is None:
                first_batch = next(iter(train_loader))
                input_dim = first_batch["features"].shape[1]
            
            model, is_component = create_model(model_name, input_dim, run_seed + fold, threshold_class=threshold_class)
            result = run_single_fold(model, train_loader, val_loader, is_component, threshold_class=threshold_class)
            all_results.append(result)
            
            if threshold_class:
                print(f"  Run {run+1}/{n_runs}, Fold {fold+1}/{n_folds}: "
                      f"Acc={result['val_metrics']['threshold_accuracy']:.3f}, "
                      f"Score={result['val_metrics']['expected_threshold_score']:.3f}")
            else:
                print(f"  Run {run+1}/{n_runs}, Fold {fold+1}/{n_folds}: "
                      f"MAE={result['val_metrics']['runtime_mae']:.3f}, "
                      f"Score={result['challenge_scores']['combined_score']:.3f}")
    
    aggregated = aggregate_results(all_results, threshold_class=threshold_class)
    
    print(f"\nAggregated Results ({len(all_results)} evaluations):")
    if threshold_class:
        print(f"  Threshold accuracy:         {aggregated['threshold_accuracy']['mean']:.4f} ± {aggregated['threshold_accuracy']['std']:.4f}")
        print(f"  Expected threshold score: {aggregated['expected_threshold_score']['mean']:.4f} ± {aggregated['expected_threshold_score']['std']:.4f}")
    else:
        print(f"  Runtime MAE (log2): {aggregated['runtime_mae']['mean']:.4f} ± {aggregated['runtime_mae']['std']:.4f}")
        print(f"  Challenge Score:   {aggregated['combined_score']['mean']:.4f} ± {aggregated['combined_score']['std']:.4f}")
    
    return {
        "model": model_name,
        "n_folds": n_folds,
        "n_runs": n_runs,
        "n_evaluations": len(all_results),
        "aggregated": aggregated,
        "all_results": all_results,
        "threshold_class": threshold_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Unified model evaluation with CV")
    parser.add_argument("--models", nargs="+", default=["mlp", "xgboost"],
                        help="Models to evaluate (e.g., mlp, xgboost, catboost, analytical)")
    parser.add_argument("--threshold-class", action="store_true",
                        help="Use threshold-class models (no duration/threshold in features; predict P(class), select by max expected score)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs per fold")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    threshold_class = args.threshold_class
    
    if threshold_class:
        all_model_names = []
        if HAS_MLP and MLPThresholdClassModel is not None:
            all_model_names.append("mlp")
        if HAS_XGB and XGBoostThresholdClassModel is not None:
            all_model_names.append("xgboost")
        if HAS_CATBOOST and CatBoostThresholdClassModel is not None:
            all_model_names.append("catboost")
    else:
        all_model_names = [
            "analytical", "analytical_fixed", "learned_component",
            "bond_dimension", "entanglement_budget",
        ]
        if HAS_MLP:
            all_model_names.append("mlp")
        if HAS_XGB:
            all_model_names.append("xgboost")
        if HAS_CATBOOST:
            all_model_names.append("catboost")
    
    if "all" in args.models:
        models_to_run = all_model_names
    else:
        models_to_run = args.models
    
    results = {}
    for model_name in models_to_run:
        try:
            results[model_name] = evaluate_model_cv(
                model_name=model_name,
                data_path=data_path,
                circuits_dir=circuits_dir,
                n_folds=args.n_folds,
                n_runs=args.n_runs,
                batch_size=args.batch_size,
                base_seed=args.seed,
                threshold_class=threshold_class,
            )
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    mode = "THRESHOLD-CLASS" if threshold_class else "DURATION"
    print(f"FINAL COMPARISON ({mode})")
    print("="*80)
    
    if threshold_class:
        print(f"{'Model':<25} {'Accuracy':>15} {'Expected Score':>20}")
        print("-"*62)
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]["aggregated"].get("expected_threshold_score", {}).get("mean", 0),
            reverse=True,
        )
        for model_name, result in sorted_models:
            agg = result["aggregated"]
            acc = agg.get("threshold_accuracy", {})
            score = agg.get("expected_threshold_score", {})
            acc_str = f"{acc.get('mean', 0):.3f}±{acc.get('std', 0):.3f}"
            score_str = f"{score.get('mean', 0):.3f}±{score.get('std', 0):.3f}"
            print(f"{model_name:<25} {acc_str:>15} {score_str:>20}")
        if sorted_models:
            best_model = sorted_models[0][0]
            best_score = sorted_models[0][1]["aggregated"]["expected_threshold_score"]["mean"]
            print(f"\nBest model: {best_model} (expected threshold score: {best_score:.4f})")
    else:
        print(f"{'Model':<25} {'MAE (log2)':>15} {'Challenge Score':>20}")
        print("-"*62)
        sorted_models = sorted(
            results.items(),
            key=lambda x: x[1]["aggregated"].get("combined_score", {}).get("mean", 0),
            reverse=True,
        )
        for model_name, result in sorted_models:
            agg = result["aggregated"]
            mae = agg.get("runtime_mae", {})
            score = agg.get("combined_score", {})
            mae_str = f"{mae.get('mean', 0):.3f}±{mae.get('std', 0):.3f}"
            score_str = f"{score.get('mean', 0):.3f}±{score.get('std', 0):.3f}"
            print(f"{model_name:<25} {mae_str:>15} {score_str:>20}")
        if sorted_models:
            best_model = sorted_models[0][0]
            best_score = sorted_models[0][1]["aggregated"]["combined_score"]["mean"]
            print(f"\nBest model: {best_model} (score: {best_score:.4f})")


if __name__ == "__main__":
    main()

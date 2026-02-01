#!/usr/bin/env python3
"""
Comprehensive evaluation script for all models:
- Duration models: MLP, XGBoost, CatBoost, LightGBM
- Threshold-class models: MLP, XGBoost, CatBoost

Uses k-fold cross-validation and reports detailed metrics.
"""

import sys
from pathlib import Path
import json
from typing import Dict, List, Any, Tuple
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    create_kfold_data_loaders,
    create_kfold_threshold_class_data_loaders,
    THRESHOLD_LADDER,
    collate_fn,
    collate_fn_threshold_class,
)
from scoring import (
    compute_challenge_score,
    mean_threshold_score,
    select_threshold_class_by_expected_score,
)


def extract_duration_data(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features, log2_runtime, and threshold from duration loader."""
    all_features = []
    all_log2_runtime = []
    all_threshold = []
    for batch in loader:
        all_features.append(batch["features"].numpy())
        all_log2_runtime.extend(batch["log2_runtime"].tolist())
        all_threshold.extend(batch["threshold"])
    return np.vstack(all_features), np.array(all_log2_runtime), np.array(all_threshold)


def extract_threshold_class_data(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and threshold_class from threshold-class loader."""
    all_features = []
    all_class = []
    for batch in loader:
        all_features.append(batch["features"].numpy())
        all_class.extend(batch["threshold_class"].tolist())
    return np.vstack(all_features), np.array(all_class)


def evaluate_duration_model(model, train_loader, val_loader, verbose=False) -> Dict[str, float]:
    """Evaluate a duration prediction model."""
    start_time = time.time()
    model.fit(train_loader, val_loader, verbose=verbose, show_progress=False)
    train_time = time.time() - start_time
    
    X_val, y_log2_runtime_val, y_threshold_val = extract_duration_data(val_loader)
    
    pred_threshold, pred_runtime = model.predict(X_val)
    true_runtime = np.power(2.0, y_log2_runtime_val)
    
    log2_pred = np.log2(np.maximum(pred_runtime, 1e-10))
    mae_log2 = np.mean(np.abs(log2_pred - y_log2_runtime_val))
    mse_log2 = np.mean((log2_pred - y_log2_runtime_val) ** 2)
    
    challenge_scores = compute_challenge_score(
        pred_threshold, y_threshold_val, pred_runtime, true_runtime
    )
    
    return {
        "train_time": train_time,
        "mae_log2": mae_log2,
        "mse_log2": mse_log2,
        "rmse_log2": np.sqrt(mse_log2),
        "threshold_score": challenge_scores["threshold_score"],
        "runtime_score": challenge_scores["runtime_score"],
        "combined_score": challenge_scores["combined_score"],
    }


def evaluate_threshold_class_model(model, train_loader, val_loader, verbose=False) -> Dict[str, float]:
    """Evaluate a threshold-class prediction model."""
    start_time = time.time()
    model.fit(train_loader, val_loader, verbose=verbose, show_progress=False)
    train_time = time.time() - start_time
    
    X_val, y_class_val = extract_threshold_class_data(val_loader)
    
    proba = model.predict_proba(X_val)
    chosen = select_threshold_class_by_expected_score(proba)
    
    accuracy = np.mean(chosen == y_class_val)
    threshold_score = mean_threshold_score(chosen, y_class_val)
    
    argmax_chosen = np.argmax(proba, axis=1)
    argmax_accuracy = np.mean(argmax_chosen == y_class_val)
    argmax_threshold_score = mean_threshold_score(argmax_chosen, y_class_val)
    
    n_underpred = np.sum(chosen < y_class_val)
    n_overpred = np.sum(chosen > y_class_val)
    n_exact = np.sum(chosen == y_class_val)
    
    return {
        "train_time": train_time,
        "accuracy": accuracy,
        "threshold_score": threshold_score,
        "argmax_accuracy": argmax_accuracy,
        "argmax_threshold_score": argmax_threshold_score,
        "n_underpred": int(n_underpred),
        "n_overpred": int(n_overpred),
        "n_exact": int(n_exact),
        "underpred_rate": n_underpred / len(y_class_val),
        "overpred_rate": n_overpred / len(y_class_val),
    }


def run_kfold_duration_evaluation(
    model_class,
    model_kwargs: Dict[str, Any],
    data_path: Path,
    circuits_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run k-fold cross-validation for a duration model."""
    fold_loaders = create_kfold_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        n_folds=n_folds,
        batch_size=32,
        seed=seed,
    )
    
    all_metrics = []
    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        if "input_dim" in model_kwargs and model_kwargs["input_dim"] is None:
            first_batch = next(iter(train_loader))
            model_kwargs["input_dim"] = first_batch["features"].shape[1]
        
        model = model_class(**model_kwargs)
        metrics = evaluate_duration_model(model, train_loader, val_loader)
        all_metrics.append(metrics)
        print(f"  Fold {fold+1}/{n_folds}: MAE={metrics['mae_log2']:.4f}, Score={metrics['combined_score']:.4f}")
    
    agg = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        agg[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    return {"fold_metrics": all_metrics, "aggregated": agg}


def run_kfold_threshold_class_evaluation(
    model_class,
    model_kwargs: Dict[str, Any],
    data_path: Path,
    circuits_dir: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run k-fold cross-validation for a threshold-class model."""
    fold_loaders = create_kfold_threshold_class_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        n_folds=n_folds,
        batch_size=32,
        seed=seed,
    )
    
    all_metrics = []
    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        if "input_dim" in model_kwargs and model_kwargs["input_dim"] is None:
            first_batch = next(iter(train_loader))
            model_kwargs["input_dim"] = first_batch["features"].shape[1]
        
        model = model_class(**model_kwargs)
        metrics = evaluate_threshold_class_model(model, train_loader, val_loader)
        all_metrics.append(metrics)
        print(f"  Fold {fold+1}/{n_folds}: Acc={metrics['accuracy']:.4f}, Score={metrics['threshold_score']:.4f}, Underpred={metrics['underpred_rate']:.2%}")
    
    agg = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        agg[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    return {"fold_metrics": all_metrics, "aggregated": agg}


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    n_folds = 3  # Use 3 folds for faster evaluation
    seed = 42
    
    results = {"duration_models": {}, "threshold_class_models": {}}
    
    print("\n" + "=" * 80)
    print("DURATION MODEL EVALUATION (predict log2(runtime) given threshold)")
    print("=" * 80)
    
    try:
        from models.xgboost_model import XGBoostModel
        print("\n[XGBoost Duration]")
        results["duration_models"]["xgboost"] = run_kfold_duration_evaluation(
            XGBoostModel,
            {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 50, "random_state": seed},
            data_path, circuits_dir, n_folds, seed
        )
    except ImportError as e:
        print(f"XGBoost not available: {e}")
    
    try:
        from models.catboost_model import CatBoostModel
        print("\n[CatBoost Duration]")
        results["duration_models"]["catboost"] = run_kfold_duration_evaluation(
            CatBoostModel,
            {"depth": 6, "learning_rate": 0.1, "iterations": 50, "random_state": seed},
            data_path, circuits_dir, n_folds, seed
        )
    except ImportError as e:
        print(f"CatBoost not available: {e}")
    
    try:
        from models.lightgbm_model import LightGBMModel
        print("\n[LightGBM Duration]")
        results["duration_models"]["lightgbm"] = run_kfold_duration_evaluation(
            LightGBMModel,
            {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 50, "random_state": seed},
            data_path, circuits_dir, n_folds, seed
        )
    except ImportError as e:
        print(f"LightGBM not available: {e}")
    
    try:
        from models.mlp import MLPModel
        print("\n[MLP Duration]")
        results["duration_models"]["mlp"] = run_kfold_duration_evaluation(
            MLPModel,
            {"input_dim": None, "hidden_dims": [128, 64, 32], "dropout": 0.2, "epochs": 50, "early_stopping_patience": 10},
            data_path, circuits_dir, n_folds, seed
        )
    except ImportError as e:
        print(f"MLP not available: {e}")
    
    print("\n" + "=" * 80)
    print("THRESHOLD-CLASS MODEL EVALUATION (predict threshold class)")
    print("=" * 80)
    
    try:
        from models.xgboost_threshold_class import XGBoostThresholdClassModel
        print("\n[XGBoost Threshold-Class]")
        results["threshold_class_models"]["xgboost"] = run_kfold_threshold_class_evaluation(
            XGBoostThresholdClassModel,
            {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 50, "random_state": seed},
            data_path, circuits_dir, n_folds, seed
        )
    except ImportError as e:
        print(f"XGBoost Threshold-Class not available: {e}")
    
    try:
        from models.catboost_threshold_class import CatBoostThresholdClassModel
        print("\n[CatBoost Threshold-Class]")
        results["threshold_class_models"]["catboost"] = run_kfold_threshold_class_evaluation(
            CatBoostThresholdClassModel,
            {"depth": 6, "learning_rate": 0.1, "iterations": 50, "random_state": seed},
            data_path, circuits_dir, n_folds, seed
        )
    except ImportError as e:
        print(f"CatBoost Threshold-Class not available: {e}")
    
    try:
        from models.mlp_threshold_class import MLPThresholdClassModel
        print("\n[MLP Threshold-Class]")
        results["threshold_class_models"]["mlp"] = run_kfold_threshold_class_evaluation(
            MLPThresholdClassModel,
            {"input_dim": None, "hidden_dims": [128, 64, 32], "dropout": 0.2, "epochs": 50, "early_stopping_patience": 10},
            data_path, circuits_dir, n_folds, seed
        )
    except ImportError as e:
        print(f"MLP Threshold-Class not available: {e}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nDuration Models (lower MAE is better, higher score is better):")
    print(f"{'Model':<15} {'MAE (log2)':>15} {'Combined Score':>18}")
    print("-" * 50)
    for name, res in sorted(results["duration_models"].items(), 
                           key=lambda x: x[1]["aggregated"]["combined_score"]["mean"], 
                           reverse=True):
        mae = res["aggregated"]["mae_log2"]
        score = res["aggregated"]["combined_score"]
        print(f"{name:<15} {mae['mean']:.4f}±{mae['std']:.3f}   {score['mean']:.4f}±{score['std']:.3f}")
    
    print("\nThreshold-Class Models (higher score is better, lower underpred is better):")
    print(f"{'Model':<15} {'Threshold Score':>18} {'Accuracy':>12} {'Underpred%':>12}")
    print("-" * 60)
    for name, res in sorted(results["threshold_class_models"].items(),
                           key=lambda x: x[1]["aggregated"]["threshold_score"]["mean"],
                           reverse=True):
        score = res["aggregated"]["threshold_score"]
        acc = res["aggregated"]["accuracy"]
        underpred = res["aggregated"]["underpred_rate"]
        print(f"{name:<15} {score['mean']:.4f}±{score['std']:.3f}   {acc['mean']:.4f}±{acc['std']:.3f}   {underpred['mean']:.1%}±{underpred['std']:.1%}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"comprehensive_eval_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

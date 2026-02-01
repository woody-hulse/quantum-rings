"""
Shared evaluation utilities for model training and evaluation scripts.

Consolidates common patterns used across evaluate.py, evaluate_all_models.py,
and other evaluation scripts.
"""

from typing import Dict, List, Any, Tuple, Type
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import THRESHOLD_LADDER
from scoring import compute_challenge_score


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_labels_duration(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ground truth from duration-prediction loader.
    
    Returns:
        (threshold_values, runtime_seconds) - threshold from input, runtime in seconds from log2_runtime
    """
    all_thresh = []
    all_runtime = []
    for batch in loader:
        all_thresh.extend(batch["threshold"])
        all_runtime.extend(np.power(2.0, batch["log2_runtime"].numpy()).tolist())
    return np.array(all_thresh), np.array(all_runtime)


def extract_features(loader: DataLoader) -> np.ndarray:
    """Extract features from a data loader."""
    all_features = []
    for batch in loader:
        all_features.append(batch["features"].numpy())
    return np.vstack(all_features)


def aggregate_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple runs, computing mean, std, min, max.
    
    Args:
        all_results: List of result dictionaries from run_single_evaluation
        
    Returns:
        Dictionary mapping metric names to {mean, std, min, max} stats
    """
    if not all_results:
        return {}
    
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
    
    aggregated = {}
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


def run_single_evaluation(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    is_component_model: bool = False,
) -> Dict[str, Any]:
    """
    Train and evaluate a single model instance.
    
    Args:
        model: Model implementing fit/evaluate/predict interface
        train_loader: Training data loader
        val_loader: Validation data loader
        is_component_model: If True, use predict_from_loader instead of predict
        
    Returns:
        Dictionary with train_time, train_metrics, val_metrics, challenge_scores
    """
    start_time = time.time()
    model.fit(train_loader, val_loader, verbose=False)
    train_time = time.time() - start_time
    
    train_eval = model.evaluate(train_loader)
    train_metrics = {
        "train_runtime_mse": train_eval["runtime_mse"],
        "train_runtime_mae": train_eval["runtime_mae"],
    }
    
    val_metrics = model.evaluate(val_loader)
    
    if is_component_model:
        pred_thresh, pred_runtime = model.predict_from_loader(val_loader)
    else:
        features = extract_features(val_loader)
        pred_thresh, pred_runtime = model.predict(features)
    
    true_thresh, true_runtime = extract_labels_duration(val_loader)
    
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


def print_metrics_table(
    aggregated: Dict[str, Dict[str, float]],
    title: str = "Validation Metrics",
) -> None:
    """Print a formatted table of aggregated metrics."""
    metric_display = [
        ("Training Time (s)", "train_time"),
        ("Train Runtime MSE", "train_runtime_mse"),
        ("Train Runtime MAE", "train_runtime_mae"),
        ("Val Runtime MSE", "runtime_mse"),
        ("Val Runtime MAE", "runtime_mae"),
        ("Challenge Threshold Score", "threshold_score"),
        ("Challenge Runtime Score", "runtime_score"),
        ("Challenge Combined Score", "combined_score"),
    ]
    
    print(f"\n--- {title} ---")
    print(f"{'Metric':<30} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 78)
    
    for display_name, key in metric_display:
        if key in aggregated:
            m = aggregated[key]
            print(f"{display_name:<30} {m['mean']:>12.4f} {m['std']:>12.4f} "
                  f"{m['min']:>12.4f} {m['max']:>12.4f}")


def print_comparison_table(
    results: Dict[str, Dict[str, Any]],
    metric_key: str = "combined_score",
    title: str = "Model Comparison",
) -> None:
    """Print a comparison table of multiple models sorted by a metric."""
    print(f"\n{'='*80}")
    print(title)
    print("="*80)
    print(f"{'Model':<25} {'Runtime MSE':>15} {'Challenge Score':>20}")
    print("-"*62)
    
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get("aggregated", x[1]).get(metric_key, {}).get("mean", 0),
        reverse=True,
    )
    
    for model_name, result in sorted_models:
        agg = result.get("aggregated", result)
        mse = agg.get("runtime_mse", {})
        score = agg.get(metric_key, {})
        mse_str = f"{mse.get('mean', 0):.3f}±{mse.get('std', 0):.3f}"
        score_str = f"{score.get('mean', 0):.3f}±{score.get('std', 0):.3f}"
        print(f"{model_name:<25} {mse_str:>15} {score_str:>20}")
    
    if sorted_models:
        best_model = sorted_models[0][0]
        best_score = sorted_models[0][1].get("aggregated", sorted_models[0][1]).get(metric_key, {}).get("mean", 0)
        print(f"\nBest model: {best_model} (score: {best_score:.4f})")

#!/usr/bin/env python3
"""
Compare XGBoost and MLP models on the quantum circuit prediction task.
"""

import sys
from pathlib import Path
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import create_data_loaders, THRESHOLD_LADDER
from models import MLPModel, MLPTrainer, XGBoostModel, compute_challenge_score, HAS_XGBOOST


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


def run_mlp_experiment(
    train_loader,
    val_loader,
    input_dim: int,
    epochs: int = 100,
    device: str = "cpu",
) -> dict:
    """Train and evaluate the MLP model."""
    print("\n" + "="*60)
    print("MLP MODEL")
    print("="*60)
    
    model = MLPModel(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        num_threshold_classes=len(THRESHOLD_LADDER),
        dropout=0.2,
    )
    
    trainer = MLPTrainer(
        model=model,
        device=device,
        lr=1e-3,
        weight_decay=1e-4,
    )
    
    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden layers: [128, 64, 32]")
    print(f"  Threshold classes: {len(THRESHOLD_LADDER)}")
    print(f"  Device: {device}")
    
    start_time = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=20,
        verbose=True,
    )
    train_time = time.time() - start_time
    
    val_metrics = trainer.evaluate(val_loader)
    
    all_features = []
    for batch in val_loader:
        all_features.append(batch["features"])
    features = torch.cat(all_features, dim=0)
    
    pred_thresh, pred_runtime = trainer.predict(features)
    true_thresh, true_runtime = extract_labels(val_loader)
    
    challenge_scores = compute_challenge_score(
        pred_thresh, true_thresh, pred_runtime, true_runtime
    )
    
    print(f"\nMLP Results:")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Validation Threshold Accuracy: {val_metrics['threshold_accuracy']:.4f}")
    print(f"  Validation Runtime MSE: {val_metrics['runtime_mse']:.4f}")
    print(f"  Validation Runtime MAE: {val_metrics['runtime_mae']:.4f}")
    print(f"  Challenge Threshold Score: {challenge_scores['threshold_score']:.4f}")
    print(f"  Challenge Runtime Score: {challenge_scores['runtime_score']:.4f}")
    print(f"  Challenge Combined Score: {challenge_scores['combined_score']:.4f}")
    
    return {
        "model": "MLP",
        "train_time": train_time,
        "val_metrics": val_metrics,
        "challenge_scores": challenge_scores,
        "history": history,
    }


def run_xgboost_experiment(train_loader, val_loader) -> dict:
    """Train and evaluate the XGBoost model."""
    print("\n" + "="*60)
    print("XGBOOST MODEL")
    print("="*60)
    
    if not HAS_XGBOOST:
        print("XGBoost not installed. Skipping.")
        return None
    
    model = XGBoostModel(
        threshold_params={
            "objective": "multi:softmax",
            "num_class": len(THRESHOLD_LADDER),
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "mlogloss",
        },
        runtime_params={
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    )
    
    print(f"\nModel configuration:")
    print(f"  Max depth: 6")
    print(f"  Learning rate: 0.1")
    print(f"  N estimators: 100")
    
    start_time = time.time()
    metrics = model.fit(train_loader, val_loader, verbose=True)
    train_time = time.time() - start_time
    
    all_features = []
    for batch in val_loader:
        all_features.append(batch["features"].numpy())
    features = np.vstack(all_features)
    
    pred_thresh, pred_runtime = model.predict(features)
    true_thresh, true_runtime = extract_labels(val_loader)
    
    challenge_scores = compute_challenge_score(
        pred_thresh, true_thresh, pred_runtime, true_runtime
    )
    
    importance = model.get_feature_importance()
    
    print(f"\nXGBoost Results:")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Validation Threshold Accuracy: {metrics['val']['threshold_accuracy']:.4f}")
    print(f"  Validation Runtime MSE: {metrics['val']['runtime_mse']:.4f}")
    print(f"  Validation Runtime MAE: {metrics['val']['runtime_mae']:.4f}")
    print(f"  Challenge Threshold Score: {challenge_scores['threshold_score']:.4f}")
    print(f"  Challenge Runtime Score: {challenge_scores['runtime_score']:.4f}")
    print(f"  Challenge Combined Score: {challenge_scores['combined_score']:.4f}")
    
    print(f"\nTop 5 features (threshold model):")
    top_idx = np.argsort(importance["threshold"])[::-1][:5]
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. Feature {idx}: {importance['threshold'][idx]:.4f}")
    
    return {
        "model": "XGBoost",
        "train_time": train_time,
        "val_metrics": metrics["val"],
        "challenge_scores": challenge_scores,
        "feature_importance": importance,
    }


def print_comparison(mlp_results: dict, xgb_results: dict):
    """Print side-by-side comparison of results."""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'MLP':>12} {'XGBoost':>12} {'Winner':>10}")
    print("-" * 64)
    
    metrics = [
        ("Training Time (s)", "train_time", False),
        ("Threshold Accuracy", ("val_metrics", "threshold_accuracy"), True),
        ("Runtime MSE", ("val_metrics", "runtime_mse"), False),
        ("Runtime MAE", ("val_metrics", "runtime_mae"), False),
        ("Challenge Threshold Score", ("challenge_scores", "threshold_score"), True),
        ("Challenge Runtime Score", ("challenge_scores", "runtime_score"), True),
        ("Challenge Combined Score", ("challenge_scores", "combined_score"), True),
    ]
    
    for name, key, higher_is_better in metrics:
        if isinstance(key, tuple):
            mlp_val = mlp_results[key[0]][key[1]]
            xgb_val = xgb_results[key[0]][key[1]] if xgb_results else float("nan")
        else:
            mlp_val = mlp_results[key]
            xgb_val = xgb_results[key] if xgb_results else float("nan")
        
        if xgb_results:
            if higher_is_better:
                winner = "MLP" if mlp_val > xgb_val else "XGBoost"
            else:
                winner = "MLP" if mlp_val < xgb_val else "XGBoost"
        else:
            winner = "N/A"
        
        print(f"{name:<30} {mlp_val:>12.4f} {xgb_val:>12.4f} {winner:>10}")
    
    print("\n" + "="*60)
    if xgb_results:
        mlp_score = mlp_results["challenge_scores"]["combined_score"]
        xgb_score = xgb_results["challenge_scores"]["combined_score"]
        if mlp_score > xgb_score:
            print(f"WINNER: MLP (combined score: {mlp_score:.4f} vs {xgb_score:.4f})")
        else:
            print(f"WINNER: XGBoost (combined score: {xgb_score:.4f} vs {mlp_score:.4f})")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Compare MLP and XGBoost models")
    parser.add_argument("--epochs", type=int, default=100, help="MLP training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device for MLP (cpu/cuda/mps)")
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["features"].shape[1]
    
    print(f"\nDataset info:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Feature dimension: {input_dim}")
    print(f"  Batch size: {args.batch_size}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    mlp_results = run_mlp_experiment(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=input_dim,
        epochs=args.epochs,
        device=args.device,
    )
    
    xgb_results = run_xgboost_experiment(train_loader, val_loader)
    
    print_comparison(mlp_results, xgb_results)


if __name__ == "__main__":
    main()

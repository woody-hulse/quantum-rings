#!/usr/bin/env python3
"""
Hyperparameter sweep for the temperature parameter in MLP models.

The temperature parameter controls the softness of the scoring-aligned loss.
Lower temperature = sharper predictions (closer to hard classification)
Higher temperature = softer predictions (may help avoid underprediction)

Supports both:
- MLPModel (classification-based threshold)
- MLPContinuousModel (continuous threshold prediction)
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import create_kfold_data_loaders, THRESHOLD_LADDER
from models.mlp import MLPModel, MLPContinuousModel
from scoring import compute_challenge_score


AVAILABLE_MODELS = {
    "classification": MLPModel,
    "continuous": MLPContinuousModel,
}


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_labels(loader) -> tuple:
    all_thresh = []
    all_runtime = []
    for batch in loader:
        all_thresh.extend(batch["threshold"])
        all_runtime.extend(np.power(2.0, batch["log2_runtime"].numpy()).tolist())
    return np.array(all_thresh), np.array(all_runtime)


def extract_features(loader) -> np.ndarray:
    all_features = []
    for batch in loader:
        all_features.append(batch["features"].numpy())
    return np.vstack(all_features)


def evaluate_temperature(
    temperature: float,
    model_type: str,
    data_path: Path,
    circuits_dir: Path,
    n_folds: int = 5,
    n_seeds: int = 2,
    epochs: int = 50,
    device: str = "cpu",
) -> dict:
    """Evaluate a single temperature value with k-fold cross-validation."""
    
    model_class = AVAILABLE_MODELS[model_type]
    
    all_scores = []
    all_thresh_scores = []
    all_runtime_scores = []
    
    for seed in range(n_seeds):
        set_seeds(seed)
        
        fold_loaders = create_kfold_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            n_folds=n_folds,
            batch_size=32,
            seed=seed,
        )
        
        for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
            sample_batch = next(iter(train_loader))
            input_dim = sample_batch["features"].shape[1]
            
            model = model_class(
                input_dim=input_dim,
                hidden_dims=[128, 64, 32],
                dropout=0.2,
                lr=1e-3,
                weight_decay=1e-4,
                device=device,
                epochs=epochs,
                early_stopping_patience=20,
                temperature=temperature,
            )
            
            model.fit(train_loader, val_loader, show_progress=False)
            
            val_features = extract_features(val_loader)
            true_thresh, true_runtime = extract_labels(val_loader)
            pred_thresh, pred_runtime = model.predict(val_features)
            
            scores = compute_challenge_score(pred_thresh, true_thresh, pred_runtime, true_runtime)
            
            all_scores.append(scores["combined_score"])
            all_thresh_scores.append(scores["threshold_score"])
            all_runtime_scores.append(scores["runtime_score"])
    
    return {
        "temperature": temperature,
        "model_type": model_type,
        "combined_score_mean": float(np.mean(all_scores)),
        "combined_score_std": float(np.std(all_scores)),
        "threshold_score_mean": float(np.mean(all_thresh_scores)),
        "threshold_score_std": float(np.std(all_thresh_scores)),
        "runtime_score_mean": float(np.mean(all_runtime_scores)),
        "runtime_score_std": float(np.std(all_runtime_scores)),
        "n_evaluations": len(all_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Temperature hyperparameter sweep")
    parser.add_argument("--model", type=str, default="classification", 
                        choices=["classification", "continuous", "both"],
                        help="Model type: classification, continuous, or both")
    parser.add_argument("--min-temp", type=float, default=0.1)
    parser.add_argument("--max-temp", type=float, default=5.0)
    parser.add_argument("--n-values", type=int, default=15)
    parser.add_argument("--log-scale", action="store_true", help="Use log scale for temperature values")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if args.log_scale:
        temp_values = np.logspace(
            np.log10(args.min_temp),
            np.log10(args.max_temp),
            args.n_values
        )
    else:
        temp_values = np.linspace(args.min_temp, args.max_temp, args.n_values)
    
    if args.model == "both":
        model_types = ["classification", "continuous"]
    else:
        model_types = [args.model]
    
    print(f"Temperature sweep: {len(temp_values)} values from {args.min_temp} to {args.max_temp}")
    print(f"Model types: {model_types}")
    print(f"K-folds: {args.n_folds}, Seeds: {args.n_seeds}, Epochs: {args.epochs}")
    print(f"Total evaluations per temperature per model: {args.n_folds * args.n_seeds}")
    print("=" * 70)
    
    results = []
    best_by_model = {}
    
    total_evals = len(model_types) * len(temp_values)
    eval_num = 0
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'='*70}")
        
        best_by_model[model_type] = {"score": -1, "temperature": None}
        
        for temperature in temp_values:
            eval_num += 1
            print(f"\n[{eval_num}/{total_evals}] {model_type} | temperature = {temperature:.3f}")
            
            result = evaluate_temperature(
                temperature=temperature,
                model_type=model_type,
                data_path=data_path,
                circuits_dir=circuits_dir,
                n_folds=args.n_folds,
                n_seeds=args.n_seeds,
                epochs=args.epochs,
                device=args.device,
            )
            results.append(result)
            
            score = result["combined_score_mean"]
            std = result["combined_score_std"]
            
            if score > best_by_model[model_type]["score"]:
                best_by_model[model_type]["score"] = score
                best_by_model[model_type]["temperature"] = temperature
                marker = " *** NEW BEST ***"
            else:
                marker = ""
            
            print(f"  Combined: {score:.4f} ± {std:.4f}{marker}")
            print(f"  Threshold: {result['threshold_score_mean']:.4f} ± {result['threshold_score_std']:.4f}")
            print(f"  Runtime: {result['runtime_score_mean']:.4f} ± {result['runtime_score_std']:.4f}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (sorted by combined score)")
    print("=" * 70)
    
    for model_type in model_types:
        model_results = [r for r in results if r["model_type"] == model_type]
        print(f"\n{model_type.upper()}:")
        print(f"{'Temperature':>12} | {'Combined':>12} | {'Threshold':>12} | {'Runtime':>12}")
        print("-" * 57)
        
        for r in sorted(model_results, key=lambda x: x["combined_score_mean"], reverse=True):
            print(f"{r['temperature']:>12.3f} | {r['combined_score_mean']:>12.4f} | "
                  f"{r['threshold_score_mean']:>12.4f} | {r['runtime_score_mean']:>12.4f}")
        
        best = best_by_model[model_type]
        print(f"\nBest: temperature={best['temperature']:.3f}, score={best['score']:.4f}")
    
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / "results" / f"temperature_sweep_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "config": {
            "model_types": model_types,
            "min_temp": args.min_temp,
            "max_temp": args.max_temp,
            "n_values": args.n_values,
            "log_scale": args.log_scale,
            "n_folds": args.n_folds,
            "n_seeds": args.n_seeds,
            "epochs": args.epochs,
        },
        "best_by_model": best_by_model,
        "results": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare all available models on the quantum circuit prediction task.

This script runs evaluate.py for each model type and displays results.
For detailed single-model evaluation, use evaluate.py directly.

Supports k-fold cross-validation for robust comparison on small datasets.
"""

import sys
from pathlib import Path
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Compare all available models (runs evaluate.py for each)"
    )
    parser.add_argument("--n-runs", type=int, default=32, 
                        help="Number of models to train per fold/split (default: 20)")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="MLP training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8, 
                        help="Batch size (default: 32)")
    parser.add_argument("--val-fraction", type=float, default=0.2, 
                        help="Validation fraction for single split (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Base random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Device for MLP (cpu/cuda/mps)")
    parser.add_argument("--kfold", type=int, default=5,
                        help="Number of folds for cross-validation (0=disabled, default: 5)")
    parser.add_argument("--scoring-loss", action="store_true",
                        help="Use challenge-aligned scoring loss for MLP")
    args = parser.parse_args()
    
    evaluate_script = Path(__file__).parent / "evaluate.py"
    
    models = ["mlp", "xgboost"]
    
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    if args.kfold > 1:
        print(f"Using {args.kfold}-fold cross-validation")
        print(f"{args.n_runs} runs per fold = {args.kfold * args.n_runs} total per model")
    else:
        print(f"Using single {args.val_fraction:.0%} validation split")
        print(f"{args.n_runs} runs per model")
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Running evaluation for: {model.upper()}")
        print(f"{'='*60}\n")
        
        cmd = [
            sys.executable,
            str(evaluate_script),
            "--model", model,
            "--n-runs", str(args.n_runs),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--device", args.device,
        ]
        
        if args.kfold > 1:
            cmd.extend(["--kfold", str(args.kfold)])
        else:
            cmd.extend(["--val-fraction", str(args.val_fraction)])
        
        if args.scoring_loss:
            cmd.append("--scoring-loss")
        
        subprocess.run(cmd)
    
    print("\n" + "="*60)
    print("ALL MODEL EVALUATIONS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

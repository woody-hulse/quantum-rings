#!/usr/bin/env python3
"""
Hyperparameter grid search for the Quantum Circuit GNN.

Searches over model architecture and training hyperparameters,
evaluates each configuration with multiple runs, and reports the best settings.
"""

import sys
from pathlib import Path
import argparse
import json
import itertools
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.train import (
    run_single_evaluation,
    set_all_seeds,
    aggregate_metrics,
    extract_labels,
)
from gnn.dataset import create_graph_data_loaders, GLOBAL_FEAT_DIM
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from scoring import compute_challenge_score


# Default hyperparameter grid
DEFAULT_GRID = {
    "hidden_dim": [32, 48, 64],
    "num_layers": [2, 3, 4],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [1e-3, 5e-4],
    "weight_decay": [1e-4, 5e-4, 1e-3],
    "use_ordinal": [True, False],
    "use_augmentation": [True, False],
    "aug_strength": [0.3, 0.5],
}

# Smaller grid for quick testing
QUICK_GRID = {
    "hidden_dim": [48, 64],
    "num_layers": [2, 3],
    "dropout": [0.2, 0.3],
    "lr": [1e-3],
    "weight_decay": [5e-4, 1e-3],
    "use_ordinal": [False],
    "use_augmentation": [True],
    "aug_strength": [0.3],
}

# Focused grid based on initial experiments
FOCUSED_GRID = {
    "hidden_dim": [48, 64, 80],
    "num_layers": [2, 3],
    "dropout": [0.15, 0.2, 0.25],
    "lr": [1e-3, 7e-4],
    "weight_decay": [3e-4, 5e-4, 7e-4],
    "use_ordinal": [False],
    "use_augmentation": [True],
    "aug_strength": [0.2, 0.3, 0.4],
}


def generate_configs(grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations from the hyperparameter grid."""
    keys = list(grid.keys())
    values = list(grid.values())
    
    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        
        # Skip invalid combinations
        if not config.get("use_augmentation", True) and "aug_strength" in config:
            # Only need one aug_strength value when augmentation is disabled
            if config["aug_strength"] != grid["aug_strength"][0]:
                continue
        
        configs.append(config)
    
    return configs


def evaluate_config(
    config: Dict[str, Any],
    train_loader,
    val_loader,
    n_runs: int = 3,
    epochs: int = 80,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate a single hyperparameter configuration."""
    results = []
    
    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        set_all_seeds(run_seed)
        
        result = run_single_evaluation(
            train_loader=train_loader,
            val_loader=val_loader,
            model_type="basic",
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            epochs=epochs,
            device=device,
            verbose=False,
            use_ordinal=config.get("use_ordinal", False),
            use_augmentation=config.get("use_augmentation", True),
            augmentation_strength=config.get("aug_strength", 0.3),
        )
        results.append(result)
    
    aggregated = aggregate_metrics(results)
    
    return {
        "config": config,
        "n_runs": n_runs,
        "metrics": aggregated,
        "all_runs": results,
    }


def run_grid_search(
    grid: Dict[str, List],
    data_path: Path,
    circuits_dir: Path,
    n_runs: int = 3,
    epochs: int = 80,
    batch_size: int = 16,
    device: str = "cpu",
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run full grid search."""
    configs = generate_configs(grid)
    print(f"Generated {len(configs)} configurations to evaluate")
    
    # Load data once
    set_all_seeds(seed)
    train_loader, val_loader = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=batch_size,
        seed=seed,
    )
    
    print(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Each config will be evaluated with {n_runs} runs")
    print(f"Total evaluations: {len(configs) * n_runs}")
    print()
    
    all_results = []
    best_score = -float("inf")
    best_config = None
    
    start_time = time.time()
    
    for i, config in enumerate(tqdm(configs, desc="Grid search")):
        config_start = time.time()
        
        result = evaluate_config(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            n_runs=n_runs,
            epochs=epochs,
            device=device,
            seed=seed,
        )
        
        config_time = time.time() - config_start
        result["config_time"] = config_time
        all_results.append(result)
        
        # Track best
        score = result["metrics"].get("combined_score", {}).get("mean", 0)
        if score > best_score:
            best_score = score
            best_config = config
            tqdm.write(f"  New best! Score: {score:.4f} | Config: {config}")
        
        # Save intermediate results
        if output_path and (i + 1) % 5 == 0:
            _save_results(all_results, output_path, best_config, best_score)
    
    total_time = time.time() - start_time
    
    # Final save
    if output_path:
        _save_results(all_results, output_path, best_config, best_score)
    
    print(f"\nGrid search complete in {total_time/60:.1f} minutes")
    
    return all_results


def _save_results(
    results: List[Dict],
    output_path: Path,
    best_config: Dict,
    best_score: float,
):
    """Save results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_configs": len(results),
        "best_score": best_score,
        "best_config": best_config,
        "results": convert(results),
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def print_top_results(results: List[Dict], top_k: int = 10):
    """Print the top-k configurations by combined score."""
    # Sort by combined score
    sorted_results = sorted(
        results,
        key=lambda x: x["metrics"].get("combined_score", {}).get("mean", 0),
        reverse=True,
    )
    
    print("\n" + "=" * 80)
    print(f"TOP {top_k} CONFIGURATIONS")
    print("=" * 80)
    
    print(f"\n{'Rank':<5} {'Score':>8} {'MAE (log2)':>10} | Configuration")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results[:top_k]):
        metrics = result["metrics"]
        config = result["config"]
        
        score = metrics.get("combined_score", {}).get("mean", 0)
        mae = metrics.get("runtime_mae", {}).get("mean", 0)
        runtime = mae
        
        config_str = (
            f"hd={config['hidden_dim']}, nl={config['num_layers']}, "
            f"do={config['dropout']}, lr={config['lr']:.0e}, "
            f"wd={config['weight_decay']:.0e}"
        )
        if config.get("use_augmentation"):
            config_str += f", aug={config.get('aug_strength', 0.3)}"
        if config.get("use_ordinal"):
            config_str += ", ordinal"
        
        print(f"{i+1:<5} {score:>8.4f} {mae:>8.4f} | {config_str}")
    
    print("\n" + "=" * 80)
    
    # Print best config details
    best = sorted_results[0]
    print("\nBEST CONFIGURATION:")
    print(json.dumps(best["config"], indent=2))
    print(f"\nMetrics:")
    for key, value in best["metrics"].items():
        if isinstance(value, dict) and "mean" in value:
            print(f"  {key}: {value['mean']:.4f} ± {value['std']:.4f}")


def print_analysis(results: List[Dict]):
    """Print analysis of hyperparameter effects."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER ANALYSIS")
    print("=" * 80)
    
    # Group by each hyperparameter and compute average score
    for param in ["hidden_dim", "num_layers", "dropout", "lr", "weight_decay", 
                  "use_ordinal", "use_augmentation", "aug_strength"]:
        param_values = {}
        
        for result in results:
            config = result["config"]
            if param not in config:
                continue
            
            value = config[param]
            score = result["metrics"].get("combined_score", {}).get("mean", 0)
            
            if value not in param_values:
                param_values[value] = []
            param_values[value].append(score)
        
        if not param_values:
            continue
        
        print(f"\n{param}:")
        for value in sorted(param_values.keys(), key=lambda x: (isinstance(x, bool), x)):
            scores = param_values[value]
            avg = np.mean(scores)
            std = np.std(scores)
            print(f"  {value}: {avg:.4f} ± {std:.4f} (n={len(scores)})")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter grid search for Quantum Circuit GNN"
    )
    parser.add_argument(
        "--grid", type=str, default="quick",
        choices=["default", "quick", "focused"],
        help="Which grid to use (default: quick)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=3,
        help="Number of runs per configuration (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=80,
        help="Training epochs per run (default: 80)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (cpu/cuda/mps)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: auto-generated)"
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top results to show (default: 10)"
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    # Select grid
    if args.grid == "default":
        grid = DEFAULT_GRID
    elif args.grid == "quick":
        grid = QUICK_GRID
    elif args.grid == "focused":
        grid = FOCUSED_GRID
    else:
        raise ValueError(f"Unknown grid: {args.grid}")
    
    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / "results" / f"gnn_grid_search_{timestamp}.json"
        output_path.parent.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("GNN HYPERPARAMETER GRID SEARCH")
    print("=" * 80)
    print(f"\nGrid: {args.grid}")
    print(f"Parameters being searched:")
    for param, values in grid.items():
        print(f"  {param}: {values}")
    print(f"\nRuns per config: {args.n_runs}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Output: {output_path}")
    print()
    
    # Run grid search
    results = run_grid_search(
        grid=grid,
        data_path=data_path,
        circuits_dir=circuits_dir,
        n_runs=args.n_runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        output_path=output_path,
    )
    
    # Print results
    print_top_results(results, args.top_k)
    print_analysis(results)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

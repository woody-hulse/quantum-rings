#!/usr/bin/env python3
"""
Grid search over the underprediction steepness parameter for the asymmetric log2 loss.

Runs k-fold cross-validation by default for each steepness value and reports
the best steepness by combined challenge score.
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import create_data_loaders
from models.mlp import MLPModel
from evaluate import evaluate_model_kfold, set_all_seeds


DEFAULT_STEEPNESS_GRID = [1.0, 2.0, 5.0, 10.0, 20.0]


def run_steepness_grid_search(
    steepness_values: List[float],
    data_path: Path,
    circuits_dir: Path,
    n_folds: int = 5,
    n_runs_per_fold: int = 5,
    base_seed: int = 42,
    batch_size: int = 32,
    epochs: int = 100,
    device: str = "cpu",
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run grid search over steepness with k-fold CV for each value."""
    set_all_seeds(base_seed)
    sample_loader, _ = create_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=batch_size,
        val_fraction=0.2,
        seed=base_seed,
    )
    sample_batch = next(iter(sample_loader))
    input_dim = sample_batch["features"].shape[1]

    print(f"Steepness grid: {steepness_values}")
    print(f"K-fold: {n_folds} folds × {n_runs_per_fold} runs = {n_folds * n_runs_per_fold} per steepness")
    print(f"Total evaluations: {len(steepness_values) * n_folds * n_runs_per_fold}")
    print()

    all_results = []
    best_score = -float("inf")
    best_steepness = None

    for i, steepness in enumerate(steepness_values):
        print("\n" + "=" * 60)
        print(f"Steepness {i + 1}/{len(steepness_values)}: {steepness}")
        print("=" * 60)

        result = evaluate_model_kfold(
            model_class=MLPModel,
            data_path=data_path,
            circuits_dir=circuits_dir,
            input_dim=input_dim,
            n_folds=n_folds,
            n_runs_per_fold=n_runs_per_fold,
            base_seed=base_seed,
            batch_size=batch_size,
            device=device,
            epochs=epochs,
            use_asymmetric_loss=True,
            underprediction_steepness=steepness,
        )

        summary = {
            "underprediction_steepness": steepness,
            "aggregated_metrics": result["aggregated_metrics"],
            "n_folds": result["n_folds"],
            "n_runs_per_fold": result["n_runs_per_fold"],
            "fold_results": [
                {
                    "fold": fr["fold"],
                    "n_train": fr["n_train"],
                    "n_val": fr["n_val"],
                    "metrics": fr["metrics"],
                }
                for fr in result["fold_results"]
            ],
        }
        all_results.append(summary)

        score = result["aggregated_metrics"].get("combined_score", {}).get("mean", 0)
        if score > best_score:
            best_score = score
            best_steepness = steepness
            print(f"  New best! Combined score: {score:.4f}")

        if output_path and (i + 1) % 2 == 0:
            _save_results(all_results, output_path, best_steepness, best_score)

    if output_path:
        _save_results(all_results, output_path, best_steepness, best_score)

    return all_results


def _save_results(
    results: List[Dict],
    output_path: Path,
    best_steepness: Optional[float],
    best_score: float,
) -> None:
    """Save results to JSON."""
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    output = {
        "timestamp": datetime.now().isoformat(),
        "best_steepness": best_steepness,
        "best_combined_score": best_score,
        "n_configs": len(results),
        "results": convert(results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def print_results(results: List[Dict]) -> None:
    """Print summary table of steepness vs scores."""
    print("\n" + "=" * 70)
    print("STEEPNESS GRID SEARCH RESULTS")
    print("=" * 70)
    print(f"\n{'Steepness':>12} {'Combined':>12} {'Thresh Score':>14} {'Runtime Score':>14}")
    print("-" * 70)

    for r in results:
        steepness = r["underprediction_steepness"]
        metrics = r["aggregated_metrics"]
        combined = metrics.get("combined_score", {}).get("mean", 0)
        thresh = metrics.get("threshold_score", {}).get("mean", 0)
        runtime = metrics.get("runtime_score", {}).get("mean", 0)
        print(f"{steepness:>12.1f} {combined:>12.4f} {thresh:>14.4f} {runtime:>14.4f}")

    best = max(results, key=lambda x: x["aggregated_metrics"].get("combined_score", {}).get("mean", 0))
    print("-" * 70)
    print(f"\nBest steepness: {best['underprediction_steepness']}")
    print(f"  Combined score: {best['aggregated_metrics']['combined_score']['mean']:.4f} ± "
          f"{best['aggregated_metrics']['combined_score']['std']:.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over asymmetric loss steepness with k-fold CV"
    )
    parser.add_argument(
        "--steepness", type=float, nargs="+", default=None,
        help="Steepness values to search (default: 1.0 2.0 5.0 10.0 20.0)"
    )
    parser.add_argument(
        "--kfold", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--runs-per-fold", type=int, default=5,
        help="Number of runs per fold (default: 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="MLP training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for MLP (cpu/cuda/mps)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: results/steepness_grid_search_<timestamp>.json)"
    )
    parser.add_argument(
        "--no-kfold", action="store_true",
        help="Disable k-fold; use single train/val split with multiple runs"
    )
    args = parser.parse_args()

    steepness_values = args.steepness if args.steepness is not None else DEFAULT_STEEPNESS_GRID
    n_folds = 1 if args.no_kfold else args.kfold
    if args.no_kfold:
        n_runs_per_fold = args.runs_per_fold
    else:
        n_runs_per_fold = args.runs_per_fold

    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / "results" / f"steepness_grid_search_{timestamp}.json"

    print("=" * 60)
    print("ASYMMETRIC LOSS STEEPNESS GRID SEARCH")
    print("=" * 60)
    print(f"\nSteepness values: {steepness_values}")
    print(f"K-fold CV: {n_folds} folds × {n_runs_per_fold} runs per steepness")
    print(f"Epochs: {args.epochs}  Batch size: {args.batch_size}")
    print(f"Device: {args.device}  Seed: {args.seed}")
    print(f"Output: {output_path}")

    if args.no_kfold:
        print("\nNote: --no-kfold uses a single 80/20 split with multiple runs (no k-fold).")
        from evaluate import evaluate_model
        set_all_seeds(args.seed)
        sample_loader, _ = create_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=args.batch_size,
            val_fraction=0.2,
            seed=args.seed,
        )
        input_dim = next(iter(sample_loader))["features"].shape[1]
        all_results = []
        best_score = -float("inf")
        best_steepness = None
        for steepness in steepness_values:
            print(f"\n--- Steepness {steepness} ---")
            result = evaluate_model(
                model_class=MLPModel,
                data_path=data_path,
                circuits_dir=circuits_dir,
                input_dim=input_dim,
                n_runs=n_runs_per_fold,
                base_seed=args.seed,
                batch_size=args.batch_size,
                val_fraction=0.2,
                device=args.device,
                epochs=args.epochs,
                use_asymmetric_loss=True,
                underprediction_steepness=steepness,
            )
            summary = {
                "underprediction_steepness": steepness,
                "aggregated_metrics": result["aggregated_metrics"],
                "n_folds": 1,
                "n_runs_per_fold": n_runs_per_fold,
            }
            all_results.append(summary)
            score = result["aggregated_metrics"].get("combined_score", {}).get("mean", 0)
            if score > best_score:
                best_score = score
                best_steepness = steepness
        _save_results(all_results, output_path, best_steepness, best_score)
    else:
        all_results = run_steepness_grid_search(
            steepness_values=steepness_values,
            data_path=data_path,
            circuits_dir=circuits_dir,
            n_folds=n_folds,
            n_runs_per_fold=n_runs_per_fold,
            base_seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
            output_path=output_path,
        )

    print_results(all_results)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

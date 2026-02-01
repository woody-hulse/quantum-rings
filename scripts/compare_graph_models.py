#!/usr/bin/env python3
"""
Comprehensive comparison of all graph model architectures.

This script trains and evaluates all GNN architectures on quantum circuit
prediction tasks, providing detailed metrics for comparison.

Tasks:
- duration: Predict log2(runtime) given threshold as input
- threshold: Predict optimal threshold class for target fidelity

Models compared:
- BasicGNN: Simple message-passing GNN
- ImprovedGNN: Attention-based with ordinal regression (threshold) / L1 loss (duration)
- GraphTransformer: Full transformer attention with edge bias
- HeteroGNN: Heterogeneous multi-relation GNN (QCHGT)
- TemporalGNN: Temporal/causal modeling with state memory

Metrics reported (threshold task):
- Accuracy: Exact threshold class match
- Expected Score: Challenge-compatible threshold score
- Underprediction Rate: Predictions below true class (penalized heavily)
- Overprediction Rate: Predictions above true class (acceptable)

Metrics reported (duration task):
- MAE: Mean absolute error in log2 space
- MSE: Mean squared error in log2 space
- Challenge Score: Full challenge score with threshold/runtime

Common metrics:
- Training Time: Wall clock time for training
- Parameters: Number of trainable parameters
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.graph_models import (
    create_graph_model,
    get_all_model_types,
    MODEL_DESCRIPTIONS,
    GraphModelConfig,
)
from gnn.dataset import (
    create_threshold_class_graph_data_loaders,
    create_graph_data_loaders,
)
from gnn.model import create_gnn_model
from gnn.improved_model import create_improved_gnn_model
from gnn.transformer import create_graph_transformer_model
from gnn.hetero_gnn import create_hetero_gnn_model
from gnn.temporal_model import create_temporal_gnn_model
from gnn.train import GNNTrainer
from gnn.augmentation import get_train_augmentation
from scoring import compute_challenge_score


@dataclass
class ModelResult:
    """Results for a single model run."""
    model_type: str
    run_id: int
    train_time: float
    parameters: int
    epochs_trained: int
    task: str = "threshold"
    # Threshold task metrics
    train_accuracy: float = 0.0
    train_score: float = 0.0
    val_accuracy: float = 0.0
    val_score: float = 0.0
    underpred_rate: float = 0.0
    overpred_rate: float = 0.0
    # Duration task metrics
    train_mae: float = 0.0
    train_mse: float = 0.0
    val_mae: float = 0.0
    val_mse: float = 0.0
    challenge_score: float = 0.0


@dataclass 
class AggregatedResult:
    """Aggregated results across multiple runs."""
    model_type: str
    description: str
    n_runs: int
    parameters: int
    task: str = "threshold"
    
    # Threshold task metrics
    val_accuracy_mean: float = 0.0
    val_accuracy_std: float = 0.0
    val_score_mean: float = 0.0
    val_score_std: float = 0.0
    underpred_rate_mean: float = 0.0
    overpred_rate_mean: float = 0.0
    
    # Duration task metrics
    val_mae_mean: float = 0.0
    val_mae_std: float = 0.0
    val_mse_mean: float = 0.0
    val_mse_std: float = 0.0
    challenge_score_mean: float = 0.0
    challenge_score_std: float = 0.0
    
    train_time_mean: float = 0.0
    rank_by_score: int = 0
    rank_by_accuracy: int = 0


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DURATION_MODEL_FACTORIES = {
    "basic": create_gnn_model,
    "improved": create_improved_gnn_model,
    "transformer": create_graph_transformer_model,
    "hetero": create_hetero_gnn_model,
    "temporal": create_temporal_gnn_model,
}


def run_single_experiment_threshold(
    model_type: str,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    run_id: int = 0,
    verbose: bool = False,
) -> ModelResult:
    """Train and evaluate a single model for threshold classification."""
    
    model = create_graph_model(model_type=model_type, **config)
    
    start_time = time.time()
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        verbose=verbose,
        show_progress=True,
    )
    train_time = time.time() - start_time
    
    train_metrics = model.evaluate(train_loader)
    val_metrics = model.evaluate(val_loader)
    
    return ModelResult(
        model_type=model_type,
        run_id=run_id,
        train_time=train_time,
        parameters=model.count_parameters(),
        epochs_trained=len(history.get("history", [])),
        task="threshold",
        train_accuracy=train_metrics["threshold_accuracy"],
        train_score=train_metrics["expected_threshold_score"],
        val_accuracy=val_metrics["threshold_accuracy"],
        val_score=val_metrics["expected_threshold_score"],
        underpred_rate=val_metrics["underprediction_rate"],
        overpred_rate=val_metrics["overprediction_rate"],
    )


def run_single_experiment_duration(
    model_type: str,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    run_id: int = 0,
    verbose: bool = False,
) -> ModelResult:
    """Train and evaluate a single model for duration prediction."""
    
    factory = DURATION_MODEL_FACTORIES.get(model_type)
    if factory is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_kwargs = {
        "hidden_dim": config.get("hidden_dim", 64),
        "num_layers": config.get("num_layers", 4),
        "dropout": config.get("dropout", 0.2),
    }
    if model_type != "basic":
        model_kwargs["num_heads"] = config.get("num_heads", 4)
    if model_type in ("improved", "transformer", "temporal"):
        model_kwargs["model_type"] = "duration"
    if model_type == "hetero":
        model_kwargs["model_type"] = "duration"
    
    model = factory(**model_kwargs)
    device = config.get("device", "cpu")
    
    augmentation = None
    if config.get("use_augmentation", True):
        augmentation = get_train_augmentation()
    
    trainer = GNNTrainer(
        model=model,
        device=device,
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
        augmentation=augmentation,
    )
    
    start_time = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get("epochs", 100),
        early_stopping_patience=config.get("patience", 20),
        verbose=verbose,
        show_progress=True,
    )
    train_time = time.time() - start_time
    
    train_metrics = trainer.evaluate(train_loader)
    val_metrics = trainer.evaluate(val_loader)
    
    pred_thresh, pred_runtime = trainer.predict(val_loader)
    true_thresh = []
    true_runtime = []
    for batch in val_loader:
        true_thresh.extend(batch.threshold.tolist())
        true_runtime.extend((2.0 ** batch.log2_runtime).tolist())
    true_thresh = np.array(true_thresh)
    true_runtime = np.array(true_runtime)
    
    challenge_score = compute_challenge_score(
        threshold_pred=pred_thresh,
        threshold_true=true_thresh,
        runtime_pred=pred_runtime,
        runtime_true=true_runtime,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    
    return ModelResult(
        model_type=model_type,
        run_id=run_id,
        train_time=train_time,
        parameters=n_params,
        epochs_trained=len(history.get("train_loss", [])),
        task="duration",
        train_mae=train_metrics["runtime_mae"],
        train_mse=train_metrics["runtime_mse"],
        val_mae=val_metrics["runtime_mae"],
        val_mse=val_metrics["runtime_mse"],
        challenge_score=challenge_score,
    )


def run_single_experiment(
    model_type: str,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    run_id: int = 0,
    verbose: bool = False,
    task: str = "duration",
) -> ModelResult:
    """Train and evaluate a single model."""
    if task == "threshold":
        return run_single_experiment_threshold(
            model_type, train_loader, val_loader, config, run_id, verbose
        )
    else:
        return run_single_experiment_duration(
            model_type, train_loader, val_loader, config, run_id, verbose
        )


def aggregate_results(results: List[ModelResult], model_type: str, task: str = "duration") -> AggregatedResult:
    """Aggregate results from multiple runs."""
    model_results = [r for r in results if r.model_type == model_type]
    
    if not model_results:
        return None
    
    if task == "threshold":
        return AggregatedResult(
            model_type=model_type,
            description=MODEL_DESCRIPTIONS.get(model_type, "Unknown"),
            n_runs=len(model_results),
            parameters=model_results[0].parameters,
            task=task,
            val_accuracy_mean=np.mean([r.val_accuracy for r in model_results]),
            val_accuracy_std=np.std([r.val_accuracy for r in model_results]),
            val_score_mean=np.mean([r.val_score for r in model_results]),
            val_score_std=np.std([r.val_score for r in model_results]),
            underpred_rate_mean=np.mean([r.underpred_rate for r in model_results]),
            overpred_rate_mean=np.mean([r.overpred_rate for r in model_results]),
            train_time_mean=np.mean([r.train_time for r in model_results]),
        )
    else:
        return AggregatedResult(
            model_type=model_type,
            description=MODEL_DESCRIPTIONS.get(model_type, "Unknown"),
            n_runs=len(model_results),
            parameters=model_results[0].parameters,
            task=task,
            val_mae_mean=np.mean([r.val_mae for r in model_results]),
            val_mae_std=np.std([r.val_mae for r in model_results]),
            val_mse_mean=np.mean([r.val_mse for r in model_results]),
            val_mse_std=np.std([r.val_mse for r in model_results]),
            challenge_score_mean=np.mean([r.challenge_score for r in model_results]),
            challenge_score_std=np.std([r.challenge_score for r in model_results]),
            train_time_mean=np.mean([r.train_time for r in model_results]),
        )


def print_comparison_table(aggregated: List[AggregatedResult], task: str = "duration") -> None:
    """Print formatted comparison table."""
    
    if task == "threshold":
        by_score = sorted(aggregated, key=lambda x: x.val_score_mean, reverse=True)
        for i, r in enumerate(by_score):
            r.rank_by_score = i + 1
        
        by_accuracy = sorted(aggregated, key=lambda x: x.val_accuracy_mean, reverse=True)
        for i, r in enumerate(by_accuracy):
            r.rank_by_accuracy = i + 1
        
        print("\n" + "=" * 100)
        print("GRAPH MODEL COMPARISON RESULTS (Threshold Classification)")
        print("=" * 100)
        
        print(f"\n{'Model':<20} {'Params':>10} {'Val Acc':>12} {'Val Score':>12} {'Underpred':>10} {'Time(s)':>10} {'Rank':>6}")
        print("-" * 100)
        
        for r in by_score:
            print(
                f"{r.model_type:<20} "
                f"{r.parameters:>10,} "
                f"{r.val_accuracy_mean:>8.3f}±{r.val_accuracy_std:.3f} "
                f"{r.val_score_mean:>8.3f}±{r.val_score_std:.3f} "
                f"{r.underpred_rate_mean:>10.3f} "
                f"{r.train_time_mean:>10.1f} "
                f"#{r.rank_by_score:>4}"
            )
        
        print("-" * 100)
        
        best = by_score[0]
        print(f"\nBest Model: {best.model_type}")
        print(f"  - Validation Score: {best.val_score_mean:.4f} ± {best.val_score_std:.4f}")
        print(f"  - Validation Accuracy: {best.val_accuracy_mean:.4f} ± {best.val_accuracy_std:.4f}")
        print(f"  - Underprediction Rate: {best.underpred_rate_mean:.4f}")
        print(f"  - Parameters: {best.parameters:,}")
        print(f"  - Description: {best.description}")
    else:
        by_mae = sorted(aggregated, key=lambda x: x.val_mae_mean)
        for i, r in enumerate(by_mae):
            r.rank_by_score = i + 1
        
        by_challenge = sorted(aggregated, key=lambda x: x.challenge_score_mean, reverse=True)
        for i, r in enumerate(by_challenge):
            r.rank_by_accuracy = i + 1
        
        print("\n" + "=" * 110)
        print("GRAPH MODEL COMPARISON RESULTS (Duration Prediction)")
        print("=" * 110)
        
        print(f"\n{'Model':<20} {'Params':>10} {'Val MAE':>12} {'Val MSE':>12} {'Challenge':>12} {'Time(s)':>10} {'Rank':>6}")
        print("-" * 110)
        
        for r in by_mae:
            print(
                f"{r.model_type:<20} "
                f"{r.parameters:>10,} "
                f"{r.val_mae_mean:>8.4f}±{r.val_mae_std:.4f} "
                f"{r.val_mse_mean:>8.4f}±{r.val_mse_std:.4f} "
                f"{r.challenge_score_mean:>8.4f}±{r.challenge_score_std:.4f} "
                f"{r.train_time_mean:>10.1f} "
                f"#{r.rank_by_score:>4}"
            )
        
        print("-" * 110)
        
        best = by_mae[0]
        print(f"\nBest Model (by MAE): {best.model_type}")
        print(f"  - Validation MAE: {best.val_mae_mean:.4f} ± {best.val_mae_std:.4f}")
        print(f"  - Validation MSE: {best.val_mse_mean:.4f} ± {best.val_mse_std:.4f}")
        print(f"  - Challenge Score: {best.challenge_score_mean:.4f} ± {best.challenge_score_std:.4f}")
        print(f"  - Parameters: {best.parameters:,}")
        print(f"  - Description: {best.description}")
    
    print("\n" + "=" * 100)
    print("MODEL CHARACTERISTICS")
    print("=" * 100)
    
    for r in aggregated:
        print(f"\n{r.model_type}:")
        print(f"  {r.description}")
        print(f"  Parameters: {r.parameters:,}")
        print(f"  Training time: {r.train_time_mean:.1f}s")


def run_comparison(
    data_path: Path,
    circuits_dir: Path,
    task: str = "duration",
    model_types: Optional[List[str]] = None,
    n_runs: int = 3,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.2,
    epochs: int = 100,
    patience: int = 20,
    batch_size: int = 16,
    device: str = "cpu",
    seed: int = 42,
    output_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full comparison of all graph models."""
    
    if model_types is None:
        model_types = get_all_model_types()
    
    task_name = "Duration Prediction" if task == "duration" else "Threshold Classification"
    
    print("=" * 80)
    print(f"QUANTUM CIRCUIT GRAPH MODEL COMPARISON ({task_name})")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Task: {task}")
    print(f"  Models: {model_types}")
    print(f"  Runs per model: {n_runs}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Heads: {num_heads}")
    print(f"  Dropout: {dropout}")
    print(f"  Epochs: {epochs}")
    print(f"  Patience: {patience}")
    print(f"  Device: {device}")
    
    print("\nLoading data...")
    if task == "threshold":
        train_loader, val_loader = create_threshold_class_graph_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=batch_size,
            val_fraction=0.2,
            seed=seed,
        )
    else:
        train_loader, val_loader = create_graph_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=batch_size,
            val_fraction=0.2,
            seed=seed,
        )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    config = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": dropout,
        "epochs": epochs,
        "patience": patience,
        "device": device,
        "use_ordinal": True,
        "use_augmentation": True,
    }
    
    all_results: List[ModelResult] = []
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Model: {model_type}")
        print(f"Description: {MODEL_DESCRIPTIONS.get(model_type, 'Unknown')}")
        print(f"{'='*60}")
        
        for run_id in range(n_runs):
            print(f"\n  Run {run_id + 1}/{n_runs}")
            set_seed(seed + run_id * 1000)
            
            try:
                result = run_single_experiment(
                    model_type=model_type,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    run_id=run_id,
                    verbose=False,
                    task=task,
                )
                all_results.append(result)
                
                if task == "threshold":
                    print(f"    Val Accuracy: {result.val_accuracy:.4f}")
                    print(f"    Val Score: {result.val_score:.4f}")
                    print(f"    Underpred: {result.underpred_rate:.4f}")
                else:
                    print(f"    Val MAE: {result.val_mae:.4f}")
                    print(f"    Val MSE: {result.val_mse:.4f}")
                    print(f"    Challenge Score: {result.challenge_score:.4f}")
                print(f"    Time: {result.train_time:.1f}s")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    aggregated = []
    for model_type in model_types:
        agg = aggregate_results(all_results, model_type, task=task)
        if agg:
            aggregated.append(agg)
    
    print_comparison_table(aggregated, task=task)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "config": config,
        "model_types": model_types,
        "n_runs": n_runs,
        "results": [asdict(r) for r in all_results],
        "aggregated": [asdict(a) for a in aggregated],
        "best_model": aggregated[0].model_type if aggregated else None,
    }
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return output


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare graph model architectures")
    parser.add_argument("--task", type=str, default="duration",
                        choices=["duration", "threshold"],
                        help="Task type: 'duration' (predict log2 runtime) or 'threshold' (predict threshold class)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model types to compare (default: all)")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of runs per model")
    parser.add_argument("--hidden-dim", type=int, default=16,
                        help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    output_file = None
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = project_root / "results" / f"graph_model_comparison_{args.task}_{timestamp}.json"
    
    run_comparison(
        data_path=data_path,
        circuits_dir=circuits_dir,
        task=args.task,
        model_types=args.models,
        n_runs=args.n_runs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()

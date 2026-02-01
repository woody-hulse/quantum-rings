#!/usr/bin/env python3
"""
Comprehensive model comparison script for quantum circuit prediction.

Supports both graph-based and non-graph models on threshold classification
and duration prediction tasks. Generates JSON results and visualizations.

Usage:
    python scripts/compare_models.py --task threshold --suite all
    python scripts/compare_models.py --task duration --suite non-graph
    python scripts/compare_models.py --task threshold --models XGBoost MLP CatBoost
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    create_threshold_class_data_loaders,
    create_data_loaders,
    FEATURE_DIM_WITHOUT_THRESHOLD,
)
from scoring import NUM_THRESHOLD_CLASSES


class ModelTask(Enum):
    THRESHOLD = "threshold"
    DURATION = "duration"


class ModelSuite(Enum):
    ALL = "all"
    GRAPH = "graph"
    NON_GRAPH = "non-graph"


@dataclass
class ModelInfo:
    """Model metadata and configuration."""
    name: str
    model_type: str
    is_graph: bool
    description: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResult:
    """Results for a single model run."""
    model_name: str
    model_type: str
    is_graph: bool
    run_id: int
    task: str
    
    train_time_s: float
    parameters: int
    epochs_trained: int
    
    train_primary_metric: float
    val_primary_metric: float
    train_secondary_metric: float
    val_secondary_metric: float
    
    extra_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Aggregated results across multiple runs."""
    model_name: str
    model_type: str
    is_graph: bool
    description: str
    task: str
    n_runs: int
    parameters: int
    
    val_primary_mean: float
    val_primary_std: float
    val_secondary_mean: float
    val_secondary_std: float
    train_time_mean: float
    
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    rank_by_primary: int = 0


THRESHOLD_CLASS_MODELS = {
    "XGBoost": {
        "class": "XGBoostThresholdClassModel",
        "is_graph": False,
        "description": "Gradient boosted trees with XGBoost",
        "config": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
        },
    },
    "CatBoost": {
        "class": "CatBoostThresholdClassModel",
        "is_graph": False,
        "description": "Gradient boosting with CatBoost (categorical features)",
        "config": {
            "depth": 6,
            "learning_rate": 0.1,
            "iterations": 100,
        },
    },
    "LightGBM": {
        "class": "LightGBMModel",
        "is_graph": False,
        "description": "Light gradient boosting machine",
        "config": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
        },
    },
    "MLP": {
        "class": "MLPThresholdClassModel",
        "is_graph": False,
        "description": "Multi-layer perceptron with dropout",
        "config": {
            "hidden_dims": [256, 128, 64],
            "dropout": 0.3,
            "epochs": 150,
            "early_stopping_patience": 25,
        },
    },
    "BasicGNN": {
        "class": "BasicGNNThresholdClassModel",
        "is_graph": True,
        "description": "Simple message-passing GNN",
        "config": {
            "hidden_dim": 64,
            "num_layers": 4,
            "dropout": 0.2,
            "epochs": 100,
            "patience": 20,
        },
    },
    "ImprovedGNN": {
        "class": "ImprovedGNNThresholdClassModel",
        "is_graph": True,
        "description": "Attention-based GNN with ordinal regression",
        "config": {
            "hidden_dim": 64,
            "num_layers": 4,
            "num_heads": 4,
            "dropout": 0.2,
            "epochs": 100,
            "patience": 20,
        },
    },
    "GraphTransformer": {
        "class": "GraphTransformerThresholdClassModel",
        "is_graph": True,
        "description": "Full transformer attention with edge features",
        "config": {
            "hidden_dim": 64,
            "num_layers": 4,
            "num_heads": 4,
            "dropout": 0.2,
            "epochs": 100,
            "patience": 20,
        },
    },
    "HeteroGNN": {
        "class": "HeteroGNNThresholdClassModel",
        "is_graph": True,
        "description": "Heterogeneous multi-relation GNN (QCHGT)",
        "config": {
            "hidden_dim": 64,
            "num_layers": 4,
            "num_heads": 4,
            "dropout": 0.2,
            "epochs": 100,
            "patience": 20,
        },
    },
    "TemporalGNN": {
        "class": "TemporalGNNThresholdClassModelV2",
        "is_graph": True,
        "description": "Temporal/causal modeling with state memory",
        "config": {
            "hidden_dim": 64,
            "num_layers": 4,
            "num_heads": 4,
            "dropout": 0.2,
            "epochs": 100,
            "patience": 20,
        },
    },
}

DURATION_MODELS = {
    "XGBoost": {
        "class": "XGBoostModel",
        "is_graph": False,
        "description": "XGBoost for duration regression",
        "config": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
        },
    },
    "MLP": {
        "class": "MLPModel",
        "is_graph": False,
        "description": "MLP for duration regression",
        "config": {
            "hidden_dims": [256, 128, 64],
            "dropout": 0.3,
            "epochs": 150,
        },
    },
}


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    """Count trainable parameters in a model."""
    if hasattr(model, "count_parameters"):
        return model.count_parameters()
    
    if hasattr(model, "network"):
        return sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    
    if hasattr(model, "model") and model.model is not None:
        return sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    if hasattr(model, "classifier"):
        try:
            booster = model.classifier.get_booster()
            return int(booster.num_boosted_rounds())
        except:
            pass
        return -1
    
    return -1


def create_model(
    model_name: str,
    model_config: Dict[str, Any],
    task: ModelTask,
    device: str = "cpu",
) -> Any:
    """Create a model instance based on name and task."""
    from models import (
        XGBoostThresholdClassModel,
        CatBoostThresholdClassModel,
        MLPThresholdClassModel,
        XGBoostModel,
        MLPModel,
    )
    
    model_class_name = model_config["class"]
    config = model_config.get("config", {}).copy()
    
    if "device" in config or model_class_name.startswith("MLP"):
        config["device"] = device
    
    if model_class_name == "XGBoostThresholdClassModel":
        return XGBoostThresholdClassModel(**config)
    elif model_class_name == "CatBoostThresholdClassModel":
        return CatBoostThresholdClassModel(**config)
    elif model_class_name == "MLPThresholdClassModel":
        return MLPThresholdClassModel(**config)
    elif model_class_name == "XGBoostModel":
        return XGBoostModel(**config)
    elif model_class_name == "MLPModel":
        return MLPModel(**config)
    elif model_class_name == "LightGBMModel":
        from models import LightGBMModel
        return LightGBMModel(**config)
    elif model_config["is_graph"]:
        from models.graph_models import create_graph_model, get_all_model_types
        
        model_type_map = {
            "BasicGNNThresholdClassModel": "basic",
            "ImprovedGNNThresholdClassModel": "improved",
            "GraphTransformerThresholdClassModel": "transformer",
            "HeteroGNNThresholdClassModel": "hetero",
            "TemporalGNNThresholdClassModelV2": "temporal",
        }
        model_type = model_type_map.get(model_class_name, "basic")
        config["device"] = device
        return create_graph_model(model_type=model_type, **config)
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")


def run_single_experiment(
    model_name: str,
    model_config: Dict[str, Any],
    train_loader,
    val_loader,
    task: ModelTask,
    run_id: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> ModelResult:
    """Train and evaluate a single model."""
    model = create_model(model_name, model_config, task, device)
    
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
    
    epochs = len(history.get("history", [])) if isinstance(history, dict) else 0
    if epochs == 0 and isinstance(history, dict):
        epochs = len(history.get("train_loss", []))
    
    params = count_parameters(model)
    
    if task == ModelTask.THRESHOLD:
        primary_train = train_metrics.get("expected_threshold_score", train_metrics.get("threshold_accuracy", 0))
        primary_val = val_metrics.get("expected_threshold_score", val_metrics.get("threshold_accuracy", 0))
        secondary_train = train_metrics.get("threshold_accuracy", 0)
        secondary_val = val_metrics.get("threshold_accuracy", 0)
        extra = {
            "underpred_rate": val_metrics.get("underprediction_rate", val_metrics.get("underpred_rate", 0)),
            "overpred_rate": val_metrics.get("overprediction_rate", val_metrics.get("overpred_rate", 0)),
        }
    else:
        primary_train = train_metrics.get("runtime_mae", 0)
        primary_val = val_metrics.get("runtime_mae", 0)
        secondary_train = train_metrics.get("runtime_mse", 0)
        secondary_val = val_metrics.get("runtime_mse", 0)
        extra = {}
    
    return ModelResult(
        model_name=model_name,
        model_type=model_config["class"],
        is_graph=model_config["is_graph"],
        run_id=run_id,
        task=task.value,
        train_time_s=train_time,
        parameters=params,
        epochs_trained=epochs,
        train_primary_metric=primary_train,
        val_primary_metric=primary_val,
        train_secondary_metric=secondary_train,
        val_secondary_metric=secondary_val,
        extra_metrics=extra,
    )


def aggregate_results(results: List[ModelResult], model_name: str) -> Optional[AggregatedResult]:
    """Aggregate results from multiple runs."""
    model_results = [r for r in results if r.model_name == model_name]
    
    if not model_results:
        return None
    
    first = model_results[0]
    
    extra_keys = set()
    for r in model_results:
        extra_keys.update(r.extra_metrics.keys())
    
    extra_agg = {}
    for key in extra_keys:
        values = [r.extra_metrics.get(key, 0) for r in model_results]
        extra_agg[f"{key}_mean"] = float(np.mean(values))
        extra_agg[f"{key}_std"] = float(np.std(values))
    
    model_catalog = THRESHOLD_CLASS_MODELS if first.task == "threshold" else DURATION_MODELS
    description = model_catalog.get(model_name, {}).get("description", "Unknown")
    
    return AggregatedResult(
        model_name=model_name,
        model_type=first.model_type,
        is_graph=first.is_graph,
        description=description,
        task=first.task,
        n_runs=len(model_results),
        parameters=first.parameters,
        val_primary_mean=float(np.mean([r.val_primary_metric for r in model_results])),
        val_primary_std=float(np.std([r.val_primary_metric for r in model_results])),
        val_secondary_mean=float(np.mean([r.val_secondary_metric for r in model_results])),
        val_secondary_std=float(np.std([r.val_secondary_metric for r in model_results])),
        train_time_mean=float(np.mean([r.train_time_s for r in model_results])),
        extra_metrics=extra_agg,
    )


def print_results_table(aggregated: List[AggregatedResult], task: ModelTask) -> None:
    """Print formatted comparison table."""
    if task == ModelTask.THRESHOLD:
        by_primary = sorted(aggregated, key=lambda x: x.val_primary_mean, reverse=True)
    else:
        by_primary = sorted(aggregated, key=lambda x: x.val_primary_mean, reverse=False)
    
    for i, r in enumerate(by_primary):
        r.rank_by_primary = i + 1
    
    print("\n" + "=" * 110)
    if task == ModelTask.THRESHOLD:
        print("MODEL COMPARISON: THRESHOLD CLASSIFICATION")
        primary_name = "Val Score"
        secondary_name = "Val Acc"
    else:
        print("MODEL COMPARISON: DURATION PREDICTION")
        primary_name = "Val MAE"
        secondary_name = "Val MSE"
    print("=" * 110)
    
    print(f"\n{'Model':<18} {'Type':<8} {'Params':>12} {primary_name:>14} {secondary_name:>14} {'Time(s)':>10} {'Rank':>6}")
    print("-" * 110)
    
    for r in by_primary:
        type_str = "Graph" if r.is_graph else "Tabular"
        param_str = f"{r.parameters:,}" if r.parameters > 0 else "N/A"
        
        print(
            f"{r.model_name:<18} "
            f"{type_str:<8} "
            f"{param_str:>12} "
            f"{r.val_primary_mean:>8.4f}±{r.val_primary_std:.3f} "
            f"{r.val_secondary_mean:>8.4f}±{r.val_secondary_std:.3f} "
            f"{r.train_time_mean:>10.1f} "
            f"#{r.rank_by_primary:>4}"
        )
    
    print("-" * 110)
    
    best = by_primary[0]
    print(f"\nBest Model: {best.model_name}")
    print(f"  - {primary_name}: {best.val_primary_mean:.4f} ± {best.val_primary_std:.4f}")
    print(f"  - {secondary_name}: {best.val_secondary_mean:.4f} ± {best.val_secondary_std:.4f}")
    print(f"  - Parameters: {best.parameters:,}" if best.parameters > 0 else "  - Parameters: N/A")
    print(f"  - Type: {'Graph-based' if best.is_graph else 'Tabular'}")
    print(f"  - Description: {best.description}")
    
    print("\n" + "=" * 110)


def plot_comparison(
    aggregated: List[AggregatedResult],
    task: ModelTask,
    output_dir: Path,
    timestamp: str,
) -> None:
    """Generate comparison plots."""
    if not aggregated:
        return
    
    if task == ModelTask.THRESHOLD:
        by_primary = sorted(aggregated, key=lambda x: x.val_primary_mean, reverse=True)
    else:
        by_primary = sorted(aggregated, key=lambda x: x.val_primary_mean, reverse=False)
    
    model_names = [r.model_name for r in by_primary]
    is_graph = [r.is_graph for r in by_primary]
    
    colors = ['#2ecc71' if g else '#3498db' for g in is_graph]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Model Comparison: {task.value.title()} Task", fontsize=14, fontweight='bold')
    
    x = np.arange(len(model_names))
    
    ax1 = axes[0, 0]
    primary_vals = [r.val_primary_mean for r in by_primary]
    primary_errs = [r.val_primary_std for r in by_primary]
    bars1 = ax1.bar(x, primary_vals, yerr=primary_errs, color=colors, capsize=4, alpha=0.8)
    if task == ModelTask.THRESHOLD:
        ax1.set_ylabel("Expected Threshold Score")
        ax1.set_title("Validation Score (Higher is Better)")
    else:
        ax1.set_ylabel("Mean Absolute Error (log2)")
        ax1.set_title("Validation MAE (Lower is Better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = axes[0, 1]
    secondary_vals = [r.val_secondary_mean for r in by_primary]
    secondary_errs = [r.val_secondary_std for r in by_primary]
    bars2 = ax2.bar(x, secondary_vals, yerr=secondary_errs, color=colors, capsize=4, alpha=0.8)
    if task == ModelTask.THRESHOLD:
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy (Higher is Better)")
    else:
        ax2.set_ylabel("Mean Squared Error (log2)")
        ax2.set_title("Validation MSE (Lower is Better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    ax3 = axes[1, 0]
    params = [r.parameters if r.parameters > 0 else 1 for r in by_primary]
    bars3 = ax3.bar(x, params, color=colors, alpha=0.8)
    ax3.set_ylabel("Parameters (count)")
    ax3.set_title("Model Size (Parameters)")
    ax3.set_yscale('log')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    ax4 = axes[1, 1]
    times = [r.train_time_mean for r in by_primary]
    bars4 = ax4.bar(x, times, color=colors, alpha=0.8)
    ax4.set_ylabel("Time (seconds)")
    ax4.set_title("Training Time")
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3)
    
    graph_patch = mpatches.Patch(color='#2ecc71', label='Graph-based', alpha=0.8)
    tabular_patch = mpatches.Patch(color='#3498db', label='Tabular', alpha=0.8)
    fig.legend(handles=[graph_patch, tabular_patch], loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = output_dir / f"model_comparison_{task.value}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {plot_path}")
    
    if task == ModelTask.THRESHOLD:
        plot_underpred(by_primary, output_dir, timestamp)


def plot_underpred(aggregated: List[AggregatedResult], output_dir: Path, timestamp: str) -> None:
    """Plot underprediction rates for threshold models."""
    model_names = [r.model_name for r in aggregated]
    underpred = [r.extra_metrics.get("underpred_rate_mean", 0) for r in aggregated]
    overpred = [r.extra_metrics.get("overpred_rate_mean", 0) for r in aggregated]
    
    if all(v == 0 for v in underpred + overpred):
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, underpred, width, label='Underprediction', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, overpred, width, label='Overprediction', color='#f39c12', alpha=0.8)
    
    ax.set_ylabel('Rate')
    ax.set_title('Prediction Error Types (Threshold Task)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / f"prediction_errors_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error plot saved to: {plot_path}")


def run_comparison(
    data_path: Path,
    circuits_dir: Path,
    task: ModelTask,
    suite: ModelSuite,
    model_names: Optional[List[str]] = None,
    n_runs: int = 3,
    batch_size: int = 16,
    device: str = "cpu",
    seed: int = 42,
    output_dir: Path = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run full comparison."""
    if task == ModelTask.THRESHOLD:
        model_catalog = THRESHOLD_CLASS_MODELS
    else:
        model_catalog = DURATION_MODELS
    
    if model_names:
        selected_models = {k: v for k, v in model_catalog.items() if k in model_names}
    elif suite == ModelSuite.GRAPH:
        selected_models = {k: v for k, v in model_catalog.items() if v["is_graph"]}
    elif suite == ModelSuite.NON_GRAPH:
        selected_models = {k: v for k, v in model_catalog.items() if not v["is_graph"]}
    else:
        selected_models = model_catalog
    
    print("=" * 80)
    print("QUANTUM CIRCUIT MODEL COMPARISON")
    print("=" * 80)
    print(f"\nTask: {task.value}")
    print(f"Models: {list(selected_models.keys())}")
    print(f"Runs per model: {n_runs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    print("\nLoading data...")
    if task == ModelTask.THRESHOLD:
        train_loader, val_loader = create_threshold_class_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=batch_size,
            val_fraction=0.2,
            seed=seed,
        )
        
        has_graph = any(v["is_graph"] for v in selected_models.values())
        if has_graph:
            from gnn.dataset import create_threshold_class_graph_data_loaders
            graph_train_loader, graph_val_loader = create_threshold_class_graph_data_loaders(
                data_path=data_path,
                circuits_dir=circuits_dir,
                batch_size=batch_size,
                val_fraction=0.2,
                seed=seed,
            )
    else:
        train_loader, val_loader = create_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=batch_size,
            val_fraction=0.2,
            seed=seed,
        )
        graph_train_loader = graph_val_loader = None
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    all_results: List[ModelResult] = []
    
    for model_name, model_config in selected_models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Description: {model_config['description']}")
        print(f"Type: {'Graph' if model_config['is_graph'] else 'Tabular'}")
        print(f"{'='*60}")
        
        if model_config["is_graph"]:
            current_train = graph_train_loader
            current_val = graph_val_loader
        else:
            current_train = train_loader
            current_val = val_loader
        
        for run_id in range(n_runs):
            print(f"\n  Run {run_id + 1}/{n_runs}")
            set_seed(seed + run_id * 1000)
            
            try:
                result = run_single_experiment(
                    model_name=model_name,
                    model_config=model_config,
                    train_loader=current_train,
                    val_loader=current_val,
                    task=task,
                    run_id=run_id,
                    device=device,
                    verbose=verbose,
                )
                all_results.append(result)
                
                if task == ModelTask.THRESHOLD:
                    print(f"    Val Score: {result.val_primary_metric:.4f}")
                    print(f"    Val Accuracy: {result.val_secondary_metric:.4f}")
                    print(f"    Underpred: {result.extra_metrics.get('underpred_rate', 0):.4f}")
                else:
                    print(f"    Val MAE: {result.val_primary_metric:.4f}")
                    print(f"    Val MSE: {result.val_secondary_metric:.4f}")
                print(f"    Parameters: {result.parameters:,}" if result.parameters > 0 else "    Parameters: N/A")
                print(f"    Time: {result.train_time_s:.1f}s")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    aggregated = []
    for model_name in selected_models.keys():
        agg = aggregate_results(all_results, model_name)
        if agg:
            aggregated.append(agg)
    
    print_results_table(aggregated, task)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "task": task.value,
        "suite": suite.value,
        "n_runs": n_runs,
        "batch_size": batch_size,
        "device": device,
        "seed": seed,
        "model_names": list(selected_models.keys()),
        "results": [asdict(r) for r in all_results],
        "aggregated": [asdict(a) for a in aggregated],
        "best_model": aggregated[0].model_name if aggregated else None,
    }
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = output_dir / f"model_comparison_{task.value}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {json_path}")
        
        plot_comparison(aggregated, task, output_dir, timestamp)
    
    return output


def plot_from_json(json_path: Path, output_dir: Optional[Path] = None) -> None:
    """Generate plots from existing JSON results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    task = ModelTask(data["task"])
    aggregated = [AggregatedResult(**a) for a in data["aggregated"]]
    
    if output_dir is None:
        output_dir = json_path.parent
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_comparison(aggregated, task, output_dir, timestamp)


def main():
    parser = argparse.ArgumentParser(description="Compare models on quantum circuit prediction")
    parser.add_argument("--task", type=str, choices=["threshold", "duration"], default="threshold",
                        help="Task type: threshold classification or duration prediction")
    parser.add_argument("--suite", type=str, choices=["all", "graph", "non-graph"], default="all",
                        help="Model suite to compare")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific model names to compare")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of runs per model")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose training output")
    parser.add_argument("--plot-json", type=str, default=None,
                        help="Generate plots from existing JSON file")
    
    args = parser.parse_args()
    
    if args.plot_json:
        plot_from_json(Path(args.plot_json))
        return
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results"
    
    run_comparison(
        data_path=data_path,
        circuits_dir=circuits_dir,
        task=ModelTask(args.task),
        suite=ModelSuite(args.suite),
        model_names=args.models,
        n_runs=args.n_runs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        output_dir=output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

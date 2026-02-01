"""
Train a model and visualize feature importances with actual feature names.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from data_loader import (
    NUMERIC_FEATURE_KEYS,
    FAMILY_CATEGORIES,
    create_data_loaders,
    create_threshold_class_data_loaders,
)
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.xgboost_threshold_class import XGBoostThresholdClassModel
from models.catboost_threshold_class import CatBoostThresholdClassModel


def get_feature_names(task: str = "duration") -> list:
    """
    Build the list of feature names matching the order in the feature vector.
    
    For duration prediction:
        [NUMERIC_FEATURE_KEYS..., backend, precision, log2_threshold, FAMILY_CATEGORIES...]
    
    For threshold classification:
        [NUMERIC_FEATURE_KEYS..., backend, precision, FAMILY_CATEGORIES...]
    """
    names = list(NUMERIC_FEATURE_KEYS)
    names.extend(["backend", "precision"])
    if task == "duration":
        names.append("log2_threshold")
    names.extend([f"family_{f}" for f in FAMILY_CATEGORIES])
    return names


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list,
    title: str = "Feature Importance",
    top_k: int = 30,
    output_path: Path = None,
):
    """Create horizontal bar chart of top-k most important features."""
    sorted_indices = np.argsort(importances)[::-1]
    top_indices = sorted_indices[:top_k]
    
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    top_importances = top_importances[::-1]
    top_names = top_names[::-1]
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_k * 0.3)))
    
    ax.barh(range(len(top_names)), top_importances, color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=12)
    ax.set_xlabel("Importance", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12)
    ax.xaxis.grid(True, linestyle="--", alpha=0.7)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Train model and visualize feature importance")
    parser.add_argument(
        "--task",
        type=str,
        default="duration",
        choices=["duration", "threshold"],
        help="Prediction task: 'duration' or 'threshold' classification",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "catboost"],
        help="Model type to train",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top features to display",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (optional)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of estimators for the model",
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    results_dir = project_root / "results"
    
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    
    if args.task == "duration":
        train_loader, val_loader = create_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=32,
        )
        if args.model == "xgboost":
            model = XGBoostModel(n_estimators=args.n_estimators)
        else:
            model = CatBoostModel(iterations=args.n_estimators)
    else:
        train_loader, val_loader = create_threshold_class_data_loaders(
            data_path=data_path,
            circuits_dir=circuits_dir,
            batch_size=32,
        )
        if args.model == "xgboost":
            model = XGBoostThresholdClassModel(n_estimators=args.n_estimators)
        else:
            model = CatBoostThresholdClassModel(iterations=args.n_estimators)
    
    print(f"\nTraining samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    print("\nTraining model...")
    metrics = model.fit(train_loader, val_loader, verbose=True)
    
    print("\nTraining metrics:")
    for k, v in metrics["train"].items():
        print(f"  {k}: {v:.4f}")
    print("Validation metrics:")
    for k, v in metrics["val"].items():
        print(f"  {k}: {v:.4f}")
    
    feature_importance = model.get_feature_importance()
    if feature_importance is None:
        print("Error: Model does not have feature importance.")
        return
    
    if args.task == "duration":
        importances = feature_importance["runtime"]
    else:
        importances = feature_importance["threshold_class"]
    
    feature_names = get_feature_names(args.task)
    
    if len(importances) != len(feature_names):
        print(f"Warning: Feature count mismatch - importances: {len(importances)}, names: {len(feature_names)}")
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    print(f"\nTotal features: {len(feature_names)}")
    
    print(f"\nTop {args.top_k} most important features:")
    sorted_indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_indices[:args.top_k], 1):
        print(f"  {rank:2d}. {feature_names[idx]:35s} {importances[idx]:.6f}")
    
    viz_dir = project_root / "figures" / "feature_importance"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = viz_dir / f"feature_importance_{args.task}_{args.model}.png"
    
    title = f"Top {args.top_k} Feature Importances ({args.model.upper()} - {args.task})"
    plot_feature_importance(
        importances=importances,
        feature_names=feature_names,
        title=title,
        top_k=args.top_k,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

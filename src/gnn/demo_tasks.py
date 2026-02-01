#!/usr/bin/env python3
"""
Demo script for Task 1 and Task 2 predictions.

Task 1: Predict optimal threshold for target fidelity 0.75
Task 2: Predict runtime given specific thresholds
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn.model import create_gnn_model
from gnn.dataset import (
    create_graph_data_loaders,
    THRESHOLD_LADDER,
    GLOBAL_FEAT_DIM,
)
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from gnn.train import GNNTrainer


def demo_task1(trainer: GNNTrainer, loader):
    """
    Task 1: Predict optimal threshold for target fidelity 0.75

    Input: Circuit (QASM), backend (GPU/CPU), precision (single/double)
    Output: Optimal threshold value
    """
    print("\n" + "=" * 60)
    print("TASK 1: THRESHOLD PREDICTION")
    print("=" * 60)
    print("Given: Circuit, target fidelity (0.75), backend, precision")
    print("Predict: Optimal threshold\n")

    # Predict optimal thresholds
    predicted_thresholds = trainer.predict_task1(loader)

    print(f"Number of circuits: {len(predicted_thresholds)}")
    print(f"\nFirst 10 predicted thresholds:")
    print(predicted_thresholds[:10])

    # Show distribution
    unique, counts = np.unique(predicted_thresholds, return_counts=True)
    print(f"\nThreshold distribution:")
    for thresh, count in zip(unique, counts):
        print(f"  Threshold {thresh:3d}: {count:3d} circuits ({100*count/len(predicted_thresholds):.1f}%)")

    return predicted_thresholds


def demo_task2_with_predicted_thresholds(trainer: GNNTrainer, loader, predicted_thresholds):
    """
    Task 2a: Predict runtime using predicted optimal thresholds

    Input: Circuit, PREDICTED threshold, backend, precision
    Output: Runtime at that threshold
    """
    print("\n" + "=" * 60)
    print("TASK 2a: RUNTIME PREDICTION (Using Predicted Thresholds)")
    print("=" * 60)
    print("Given: Circuit, predicted threshold, backend, precision")
    print("Predict: Runtime at predicted threshold\n")

    # Convert threshold values to class indices
    threshold_classes = torch.tensor([
        THRESHOLD_LADDER.index(t) for t in predicted_thresholds
    ])

    # Predict runtimes
    predicted_runtimes = trainer.predict_task2(loader, threshold_classes)

    print(f"Number of predictions: {len(predicted_runtimes)}")
    print(f"\nFirst 10 predictions:")
    print("  Threshold → Runtime (seconds)")
    for i in range(min(10, len(predicted_runtimes))):
        print(f"  {predicted_thresholds[i]:3d} → {predicted_runtimes[i]:.4f}s")

    print(f"\nRuntime statistics:")
    print(f"  Mean:   {np.mean(predicted_runtimes):.4f}s")
    print(f"  Median: {np.median(predicted_runtimes):.4f}s")
    print(f"  Min:    {np.min(predicted_runtimes):.4f}s")
    print(f"  Max:    {np.max(predicted_runtimes):.4f}s")

    return predicted_runtimes


def demo_task2_with_custom_thresholds(trainer: GNNTrainer, loader):
    """
    Task 2b: Predict runtime using CUSTOM thresholds

    This demonstrates predicting runtime for ANY threshold value,
    not just the optimal one.
    """
    print("\n" + "=" * 60)
    print("TASK 2b: RUNTIME PREDICTION (Using Custom Thresholds)")
    print("=" * 60)
    print("Given: Circuit, CUSTOM threshold, backend, precision")
    print("Predict: Runtime at custom threshold\n")

    num_samples = len(loader.dataset)

    # Example 1: All circuits at threshold=16
    print("Example 1: Predict runtime at threshold=16 for all circuits")
    threshold_16_class = THRESHOLD_LADDER.index(16)
    threshold_classes = torch.full((num_samples,), threshold_16_class)
    runtimes_at_16 = trainer.predict_task2(loader, threshold_classes)
    print(f"  Mean runtime at threshold=16: {np.mean(runtimes_at_16):.4f}s")
    print(f"  First 5 runtimes: {runtimes_at_16[:5]}")

    # Example 2: All circuits at threshold=64
    print("\nExample 2: Predict runtime at threshold=64 for all circuits")
    threshold_64_class = THRESHOLD_LADDER.index(64)
    threshold_classes = torch.full((num_samples,), threshold_64_class)
    runtimes_at_64 = trainer.predict_task2(loader, threshold_classes)
    print(f"  Mean runtime at threshold=64: {np.mean(runtimes_at_64):.4f}s")
    print(f"  First 5 runtimes: {runtimes_at_64[:5]}")

    # Example 3: Different threshold for each circuit
    print("\nExample 3: Random thresholds for each circuit")
    random_thresholds = np.random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256], size=min(10, num_samples))
    random_threshold_classes = torch.tensor([
        THRESHOLD_LADDER.index(t) for t in random_thresholds
    ])
    # Only predict for first 10 samples
    from torch_geometric.loader import DataLoader as PyGDataLoader
    small_loader = PyGDataLoader(
        list(loader.dataset)[:10],
        batch_size=loader.batch_size,
        shuffle=False
    )
    random_runtimes = trainer.predict_task2(small_loader, random_threshold_classes)
    print("  Threshold → Runtime")
    for thresh, runtime in zip(random_thresholds, random_runtimes):
        print(f"  {thresh:3d} → {runtime:.4f}s")

    # Example 4: Sweep thresholds for a single circuit
    print("\nExample 4: Runtime vs Threshold for a single circuit")
    single_circuit_loader = PyGDataLoader(
        [loader.dataset[0]] * len(THRESHOLD_LADDER),
        batch_size=len(THRESHOLD_LADDER),
        shuffle=False
    )
    all_threshold_classes = torch.arange(len(THRESHOLD_LADDER))
    sweep_runtimes = trainer.predict_task2(single_circuit_loader, all_threshold_classes)
    print("  Threshold ladder sweep:")
    for thresh, runtime in zip(THRESHOLD_LADDER, sweep_runtimes):
        print(f"  Threshold {thresh:3d} → {runtime:.4f}s")


def main():
    """Main demo script."""
    print("=" * 60)
    print("GNN TASK 1 & TASK 2 DEMO")
    print("=" * 60)

    # Setup
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=16,
        val_fraction=0.2,
        seed=42,
    )
    print(f"Validation set: {len(val_loader.dataset)} circuits")

    # Create model
    print("\nCreating model...")
    model = create_gnn_model(
        model_type="basic",
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        global_feat_dim=GLOBAL_FEAT_DIM,
        hidden_dim=48,
        num_layers=3,
        dropout=0.25,
        use_ordinal=True,
    )

    # Create trainer
    trainer = GNNTrainer(
        model=model,
        device=device,
        lr=1e-3,
        weight_decay=1e-3,
        use_ordinal=True,
    )

    # Train model (quick training for demo)
    print("\nTraining model (quick demo training)...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,  # Just a few epochs for demo
        early_stopping_patience=10,
        verbose=False,
        show_progress=True,
    )

    # Demo Task 1: Predict thresholds
    predicted_thresholds = demo_task1(trainer, val_loader)

    # Demo Task 2a: Predict runtime with predicted thresholds
    demo_task2_with_predicted_thresholds(trainer, val_loader, predicted_thresholds)

    # Demo Task 2b: Predict runtime with custom thresholds
    demo_task2_with_custom_thresholds(trainer, val_loader)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Task 1: Model predicts optimal threshold for fidelity 0.75")
    print("2. Task 2: Model predicts runtime for ANY given threshold")
    print("3. The two tasks are independent at inference time")
    print("4. During training, the model learns both simultaneously")
    print("=" * 60)


if __name__ == "__main__":
    main()

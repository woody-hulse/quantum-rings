#!/usr/bin/env python3
"""
Simple inference example for Task 1 and Task 2.

This script shows how to:
1. Load a trained model
2. Make Task 1 predictions (optimal threshold)
3. Make Task 2 predictions (runtime for given threshold)
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn.model import create_gnn_model
from gnn.dataset import create_graph_data_loaders, THRESHOLD_LADDER, GLOBAL_FEAT_DIM
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from gnn.train import GNNTrainer


def load_model(model_path: str, device: str = "cpu"):
    """Load a trained model from checkpoint."""
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

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def predict_task1_example(trainer: GNNTrainer, loader):
    """
    Task 1: Predict optimal threshold for target fidelity 0.75

    Challenge format:
    - Input: QASM file, target_fidelity=0.75, backend (GPU/CPU), precision (single/double)
    - Output: Predicted threshold value
    """
    print("\n" + "=" * 60)
    print("TASK 1: THRESHOLD PREDICTION")
    print("=" * 60)

    # Get predictions
    thresholds = trainer.predict_task1(loader)

    print(f"\nPredicted thresholds for {len(thresholds)} circuits:")
    for i, thresh in enumerate(thresholds[:10]):
        print(f"  Circuit {i}: threshold = {thresh}")

    if len(thresholds) > 10:
        print(f"  ... ({len(thresholds) - 10} more)")

    return thresholds


def predict_task2_example(trainer: GNNTrainer, loader, given_thresholds):
    """
    Task 2: Predict runtime given specific thresholds

    Challenge format:
    - Input: QASM file, threshold, backend (GPU/CPU), precision (single/double)
    - Output: Predicted runtime (seconds)

    Args:
        given_thresholds: List or array of threshold VALUES (e.g., [4, 8, 16, 32, ...])
    """
    print("\n" + "=" * 60)
    print("TASK 2: RUNTIME PREDICTION")
    print("=" * 60)

    # Convert threshold VALUES to class INDICES
    # The model expects indices 0-8, not the actual threshold values
    threshold_classes = torch.tensor([
        THRESHOLD_LADDER.index(t) for t in given_thresholds
    ])

    # Get predictions
    runtimes = trainer.predict_task2(loader, threshold_classes)

    print(f"\nPredicted runtimes for {len(runtimes)} circuits:")
    for i, (thresh, runtime) in enumerate(zip(given_thresholds, runtimes)):
        print(f"  Circuit {i}: threshold={thresh:3d} → runtime={runtime:.4f}s")
        if i >= 9:
            print(f"  ... ({len(runtimes) - 10} more)")
            break

    return runtimes


def main():
    """Simple inference example."""
    print("=" * 60)
    print("INFERENCE EXAMPLE: TASK 1 & TASK 2")
    print("=" * 60)

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading validation data...")
    _, val_loader = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=16,
        val_fraction=0.2,
        seed=42,
    )
    num_circuits = len(val_loader.dataset)
    print(f"Loaded {num_circuits} validation circuits")

    # Option 1: Load pre-trained model (if you have a checkpoint)
    # model = load_model("path/to/checkpoint.pt", device=device)

    # Option 2: Create and use model directly (for demo)
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

    # Create trainer wrapper
    trainer = GNNTrainer(model=model, device=device)

    # Note: For demo, we'll use an untrained model
    # In practice, you would load a trained checkpoint or train first
    print("\nWARNING: Using untrained model for demo purposes")
    print("In practice, you should load a trained checkpoint!\n")

    # ========================================
    # TASK 1: Predict optimal thresholds
    # ========================================
    predicted_thresholds = predict_task1_example(trainer, val_loader)

    # ========================================
    # TASK 2: Predict runtime for given thresholds
    # ========================================

    # Example 1: Use the predicted thresholds from Task 1
    print("\n--- Task 2 Example 1: Using predicted thresholds ---")
    runtimes_1 = predict_task2_example(trainer, val_loader, predicted_thresholds)

    # Example 2: Use custom threshold values provided by judges
    print("\n--- Task 2 Example 2: Using custom thresholds ---")
    # Suppose judges give you these thresholds for each circuit
    custom_thresholds = np.array([16] * num_circuits)  # All circuits at threshold=16
    runtimes_2 = predict_task2_example(trainer, val_loader, custom_thresholds)

    # Example 3: Different threshold for each circuit
    print("\n--- Task 2 Example 3: Different thresholds per circuit ---")
    # Judges might give you different thresholds for each circuit
    mixed_thresholds = np.random.choice([4, 8, 16, 32, 64], size=num_circuits)
    runtimes_3 = predict_task2_example(trainer, val_loader, mixed_thresholds)

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nTo use this in the challenge:")
    print("1. Train your model: python src/gnn/train.py")
    print("2. Save the model checkpoint")
    print("3. Load checkpoint in this script")
    print("4. Use predict_task1() or predict_task2() as needed")
    print("=" * 60)


if __name__ == "__main__":
    main()

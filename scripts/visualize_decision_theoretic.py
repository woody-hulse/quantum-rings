#!/usr/bin/env python3
"""
Diagnostic visualization for decision-theoretic inference.

This script visualizes:
1. When argmax and decision-theoretic predictions differ
2. The probability distributions in those cases
3. Whether the decision-theoretic choice actually helps
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import create_data_loaders, THRESHOLD_LADDER
from models.mlp import MLPModel


def get_score_matrix(num_classes: int = 9) -> np.ndarray:
    """Build the challenge score matrix."""
    score_matrix = np.zeros((num_classes, num_classes))
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if pred_idx < true_idx:
                score_matrix[true_idx, pred_idx] = 0.0
            else:
                steps_over = pred_idx - true_idx
                score_matrix[true_idx, pred_idx] = 2.0 ** (-steps_over)
    return score_matrix


def analyze_predictions(model, val_loader, device="cpu"):
    """Analyze where argmax and decision-theoretic differ."""
    model.network.eval()
    score_matrix = get_score_matrix()
    score_matrix_torch = torch.tensor(score_matrix, dtype=torch.float32, device=device)

    results = {
        "logits": [],
        "probs": [],
        "true_classes": [],
        "argmax_preds": [],
        "dt_preds": [],
        "argmax_scores": [],
        "dt_scores": [],
    }

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            true_classes = batch["threshold_class"].numpy()

            features = model._normalize(features)
            logits, _ = model.network(features)
            probs = F.softmax(logits, dim=1)

            # Argmax predictions
            argmax_preds = logits.argmax(dim=1).cpu().numpy()

            # Decision-theoretic predictions
            expected_scores = probs @ score_matrix_torch
            dt_preds = expected_scores.argmax(dim=1).cpu().numpy()

            # Compute actual scores for each prediction
            for i in range(len(true_classes)):
                true_c = true_classes[i]
                argmax_score = score_matrix[true_c, argmax_preds[i]]
                dt_score = score_matrix[true_c, dt_preds[i]]

                results["logits"].append(logits[i].cpu().numpy())
                results["probs"].append(probs[i].cpu().numpy())
                results["true_classes"].append(true_c)
                results["argmax_preds"].append(argmax_preds[i])
                results["dt_preds"].append(dt_preds[i])
                results["argmax_scores"].append(argmax_score)
                results["dt_scores"].append(dt_score)

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def plot_diagnostics(results, save_path=None):
    """Create diagnostic plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. How often do predictions differ?
    ax = axes[0, 0]
    differs = results["argmax_preds"] != results["dt_preds"]
    n_differ = differs.sum()
    n_total = len(differs)
    ax.bar(["Same", "Different"], [n_total - n_differ, n_differ], color=["green", "orange"])
    ax.set_title(f"Argmax vs Decision-Theoretic\n({n_differ}/{n_total} = {100*n_differ/n_total:.1f}% differ)")
    ax.set_ylabel("Count")

    # 2. When they differ, who wins?
    ax = axes[0, 1]
    diff_mask = differs
    if diff_mask.sum() > 0:
        argmax_wins = (results["argmax_scores"][diff_mask] > results["dt_scores"][diff_mask]).sum()
        dt_wins = (results["dt_scores"][diff_mask] > results["argmax_scores"][diff_mask]).sum()
        ties = diff_mask.sum() - argmax_wins - dt_wins
        ax.bar(["Argmax Wins", "DT Wins", "Tie"], [argmax_wins, dt_wins, ties],
               color=["red", "blue", "gray"])
        ax.set_title(f"When Predictions Differ, Who Scores Better?\n(Argmax: {argmax_wins}, DT: {dt_wins}, Tie: {ties})")
    else:
        ax.text(0.5, 0.5, "No differences", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("When Predictions Differ")
    ax.set_ylabel("Count")

    # 3. Score difference distribution
    ax = axes[0, 2]
    score_diff = results["dt_scores"] - results["argmax_scores"]
    ax.hist(score_diff, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", label="No difference")
    ax.axvline(score_diff.mean(), color="blue", linestyle="-", label=f"Mean: {score_diff.mean():.4f}")
    ax.set_xlabel("DT Score - Argmax Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Difference Distribution\nMean: {score_diff.mean():.4f}")
    ax.legend()

    # 4. Probability distribution entropy vs prediction difference
    ax = axes[1, 0]
    probs = results["probs"]
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    ax.scatter(entropy[~differs], [0] * (~differs).sum(), alpha=0.5, label="Same pred", c="green")
    ax.scatter(entropy[differs], [1] * differs.sum(), alpha=0.5, label="Different pred", c="orange")
    ax.set_xlabel("Entropy of probability distribution")
    ax.set_ylabel("Predictions differ?")
    ax.set_title("Higher Entropy → More Likely to Differ")
    ax.legend()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Same", "Different"])

    # 5. Example probability distributions where they differ
    ax = axes[1, 1]
    if diff_mask.sum() > 0:
        # Pick a few examples where they differ
        diff_indices = np.where(diff_mask)[0][:5]
        x = np.arange(9)
        width = 0.15
        for i, idx in enumerate(diff_indices):
            ax.bar(x + i*width, results["probs"][idx], width, alpha=0.7,
                   label=f"Ex {i+1}: true={results['true_classes'][idx]}, "
                         f"argmax={results['argmax_preds'][idx]}, dt={results['dt_preds'][idx]}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.set_title("Example Distributions Where Predictions Differ")
        ax.legend(fontsize=8)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([f"{t:.0e}" for t in THRESHOLD_LADDER], rotation=45)
    else:
        ax.text(0.5, 0.5, "No differences to show", ha="center", va="center", transform=ax.transAxes)

    # 6. Confidence (max prob) vs correctness
    ax = axes[1, 2]
    max_probs = probs.max(axis=1)
    argmax_correct = results["argmax_preds"] == results["true_classes"]
    dt_correct = results["dt_preds"] == results["true_classes"]

    # Bin by confidence
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    argmax_acc_by_conf = []
    dt_acc_by_conf = []
    counts = []

    for i in range(len(bins) - 1):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
        if mask.sum() > 0:
            argmax_acc_by_conf.append(argmax_correct[mask].mean())
            dt_acc_by_conf.append(dt_correct[mask].mean())
            counts.append(mask.sum())
        else:
            argmax_acc_by_conf.append(np.nan)
            dt_acc_by_conf.append(np.nan)
            counts.append(0)

    width = 0.035
    ax.bar(bin_centers - width/2, argmax_acc_by_conf, width, label="Argmax", alpha=0.7)
    ax.bar(bin_centers + width/2, dt_acc_by_conf, width, label="Decision-Theoretic", alpha=0.7)
    ax.set_xlabel("Max Probability (Confidence)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Confidence Level")
    ax.legend()
    ax.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()

    return fig


def print_summary(results):
    """Print summary statistics."""
    differs = results["argmax_preds"] != results["dt_preds"]

    print("\n" + "="*60)
    print("DECISION-THEORETIC INFERENCE ANALYSIS")
    print("="*60)

    print(f"\nTotal samples: {len(results['true_classes'])}")
    print(f"Predictions differ: {differs.sum()} ({100*differs.mean():.1f}%)")

    # Overall scores
    print(f"\nOverall Scores:")
    print(f"  Argmax mean score:    {results['argmax_scores'].mean():.4f}")
    print(f"  DT mean score:        {results['dt_scores'].mean():.4f}")
    print(f"  Difference:           {(results['dt_scores'] - results['argmax_scores']).mean():.4f}")

    # When they differ
    if differs.sum() > 0:
        print(f"\nWhen predictions differ ({differs.sum()} samples):")
        diff_mask = differs
        argmax_diff_scores = results["argmax_scores"][diff_mask]
        dt_diff_scores = results["dt_scores"][diff_mask]
        print(f"  Argmax mean score:    {argmax_diff_scores.mean():.4f}")
        print(f"  DT mean score:        {dt_diff_scores.mean():.4f}")
        print(f"  DT wins:              {(dt_diff_scores > argmax_diff_scores).sum()}")
        print(f"  Argmax wins:          {(argmax_diff_scores > dt_diff_scores).sum()}")
        print(f"  Ties:                 {(argmax_diff_scores == dt_diff_scores).sum()}")

    # Entropy analysis
    probs = results["probs"]
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    print(f"\nEntropy Analysis:")
    print(f"  Mean entropy (same pred):      {entropy[~differs].mean():.4f}")
    if differs.sum() > 0:
        print(f"  Mean entropy (diff pred):      {entropy[differs].mean():.4f}")

    # Confidence analysis
    max_probs = probs.max(axis=1)
    print(f"\nConfidence Analysis:")
    print(f"  Mean max prob (same pred):     {max_probs[~differs].mean():.4f}")
    if differs.sum() > 0:
        print(f"  Mean max prob (diff pred):     {max_probs[differs].mean():.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize decision-theoretic inference")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None, help="Path to save visualization")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"

    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=32,
        val_fraction=0.2,
        seed=args.seed,
    )

    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["features"].shape[1]

    print(f"Training MLP for {args.epochs} epochs...")
    model = MLPModel(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        device=args.device,
        epochs=args.epochs,
    )
    model.fit(train_loader, val_loader, verbose=False, show_progress=True)

    print("Analyzing predictions...")
    results = analyze_predictions(model, val_loader, device=args.device)

    print_summary(results)

    save_path = args.save or str(project_root / "results" / "decision_theoretic_analysis.png")
    plot_diagnostics(results, save_path=save_path)


if __name__ == "__main__":
    main()

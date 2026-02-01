#!/usr/bin/env python3
"""
Demonstration of expected-score optimization for threshold prediction.

This script shows how the model:
1. Outputs a probability distribution over thresholds
2. Computes expected score for each possible guess
3. Selects the guess that maximizes expected score
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def compute_scoring_matrix(num_classes: int = 9) -> np.ndarray:
    """
    Create scoring matrix where scoring_matrix[guess_idx, true_idx] = score.

    Scoring rules:
    - If guess < true: 0 points (fidelity violated)
    - If guess = true: 1.0 points (optimal)
    - If guess is N steps above true: 1.0 / (2^N) points (partial credit)
    """
    scoring_matrix = np.zeros((num_classes, num_classes))

    for guess_idx in range(num_classes):
        for true_idx in range(num_classes):
            if guess_idx < true_idx:
                # Guess too low - violates fidelity
                scoring_matrix[guess_idx, true_idx] = 0.0
            elif guess_idx == true_idx:
                # Exact match
                scoring_matrix[guess_idx, true_idx] = 1.0
            else:
                # Guess too high - partial credit
                steps_over = guess_idx - true_idx
                scoring_matrix[guess_idx, true_idx] = 1.0 / (2.0 ** steps_over)

    return scoring_matrix


def compute_expected_scores(probs: np.ndarray, scoring_matrix: np.ndarray) -> np.ndarray:
    """
    Compute expected score for each possible guess.

    expected_score[g] = sum_t P(true=t) * score(guess=g, true=t)
    """
    return probs @ scoring_matrix.T


def demonstrate_example(probs: np.ndarray, true_class: int, name: str):
    """Show expected score computation for a given probability distribution."""
    print(f"\n{'='*70}")
    print(f"EXAMPLE: {name}")
    print(f"{'='*70}")

    # Compute scoring matrix
    scoring_matrix = compute_scoring_matrix()

    # Compute expected scores
    expected_scores = compute_expected_scores(probs, scoring_matrix)

    # Find optimal guess
    optimal_guess = np.argmax(expected_scores)
    naive_guess = np.argmax(probs)  # Most likely class

    print(f"\nProbability distribution:")
    for i, (thresh, prob) in enumerate(zip(THRESHOLD_LADDER, probs)):
        marker = " ← TRUE" if i == true_class else ""
        print(f"  Class {i} (threshold={thresh:>3}): P = {prob:.4f}{marker}")

    print(f"\nExpected scores for each possible guess:")
    for i, (thresh, exp_score) in enumerate(zip(THRESHOLD_LADDER, expected_scores)):
        marker = " ← OPTIMAL" if i == optimal_guess else ""
        marker += " (naive)" if i == naive_guess else ""
        print(f"  Guess {i} (threshold={thresh:>3}): E[score] = {exp_score:.4f}{marker}")

    print(f"\nDecision:")
    print(f"  Naive (argmax prob):     Class {naive_guess} (threshold={THRESHOLD_LADDER[naive_guess]})")
    print(f"  Optimal (max exp score): Class {optimal_guess} (threshold={THRESHOLD_LADDER[optimal_guess]})")
    print(f"  True threshold:          Class {true_class} (threshold={THRESHOLD_LADDER[true_class]})")

    # Compute actual score for both strategies
    naive_score = scoring_matrix[naive_guess, true_class]
    optimal_score = scoring_matrix[optimal_guess, true_class]

    print(f"\nActual scores (if true class is {true_class}):")
    print(f"  Naive strategy:   {naive_score:.4f}")
    print(f"  Optimal strategy: {optimal_score:.4f}")
    print(f"  Improvement:      {optimal_score - naive_score:+.4f}")


def main():
    """Run demonstration with several example probability distributions."""

    print("="*70)
    print("EXPECTED-SCORE OPTIMIZATION FOR THRESHOLD PREDICTION")
    print("="*70)
    print("\nScoring rules:")
    print("  • Guess below true threshold: 0 points (fidelity violated)")
    print("  • Guess at true threshold: 1.0 points (optimal)")
    print("  • Guess N steps above true: 1.0 / (2^N) points (partial credit)")
    print("\nStrategy:")
    print("  • Model outputs probability distribution P(threshold)")
    print("  • For each guess G, compute E[score | guess G]")
    print("  • Choose guess that maximizes expected score")

    # Example 1: Confident prediction at class 3
    probs1 = np.array([0.05, 0.10, 0.15, 0.50, 0.10, 0.05, 0.03, 0.01, 0.01])
    demonstrate_example(probs1, true_class=3, name="Confident prediction (peak at class 3)")

    # Example 2: Uncertain between class 2 and 3
    probs2 = np.array([0.05, 0.10, 0.35, 0.35, 0.08, 0.04, 0.02, 0.01, 0.00])
    demonstrate_example(probs2, true_class=2, name="Uncertain between 2 and 3 (true=2)")

    # Example 3: Skewed distribution with long tail
    probs3 = np.array([0.10, 0.20, 0.30, 0.20, 0.10, 0.05, 0.03, 0.01, 0.01])
    demonstrate_example(probs3, true_class=2, name="Skewed distribution (true=2)")

    # Example 4: Uniform-ish distribution (high uncertainty)
    probs4 = np.array([0.10, 0.12, 0.15, 0.18, 0.15, 0.12, 0.10, 0.05, 0.03])
    demonstrate_example(probs4, true_class=4, name="High uncertainty (true=4)")

    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("The expected-score decision rule is NOT the same as argmax probability!")
    print("Due to the asymmetric scoring (zero points if too low, partial credit if too high),")
    print("it's often better to guess conservatively higher when uncertain.")
    print("="*70)


def visualize_scoring_matrix():
    """Create a visualization of the scoring matrix."""
    scoring_matrix = compute_scoring_matrix()

    plt.figure(figsize=(10, 8))
    plt.imshow(scoring_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Score')
    plt.xlabel('True Threshold Class')
    plt.ylabel('Guessed Threshold Class')
    plt.title('Scoring Matrix: Score(guess, true)')

    # Add threshold labels
    plt.xticks(range(9), [str(t) for t in THRESHOLD_LADDER])
    plt.yticks(range(9), [str(t) for t in THRESHOLD_LADDER])

    # Add text annotations
    for i in range(9):
        for j in range(9):
            text = plt.text(j, i, f'{scoring_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig('scoring_matrix.png', dpi=150)
    print("\nVisualization saved to 'scoring_matrix.png'")


if __name__ == "__main__":
    main()

    # Uncomment to generate visualization (requires matplotlib)
    # visualize_scoring_matrix()

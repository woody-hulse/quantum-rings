#!/usr/bin/env python3
"""
Compare single models vs ensembles with different inference strategies.

Tests the hypothesis that:
- Single models are confident → decision-theoretic doesn't help
- Ensembles have more uncertainty → decision-theoretic should help
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import create_data_loaders, create_kfold_data_loaders, THRESHOLD_LADDER
from models.mlp import MLPModel
from models.ensemble import EnsembleModel, EnsembleTrainer
from scoring import compute_challenge_score


def extract_features(loader) -> np.ndarray:
    """Extract features from a data loader."""
    all_features = []
    for batch in loader:
        all_features.append(batch["features"].numpy())
    return np.vstack(all_features)


def extract_labels(loader) -> tuple:
    """Extract ground truth: threshold (input) and runtime in seconds."""
    all_thresh = []
    all_runtime = []
    for batch in loader:
        all_thresh.extend(batch["threshold"])
        all_runtime.extend(np.power(2.0, batch["log2_runtime"].numpy()).tolist())
    return np.array(all_thresh), np.array(all_runtime)


def evaluate_model_with_strategy(model, val_loader, strategy=None):
    """Evaluate a model and compute challenge scores."""
    features = extract_features(val_loader)
    true_thresh, true_runtime = extract_labels(val_loader)

    if hasattr(model, 'predict') and strategy:
        pred_thresh, pred_runtime = model.predict(features, strategy=strategy)
    else:
        pred_thresh, pred_runtime = model.predict(features)

    return compute_challenge_score(pred_thresh, true_thresh, pred_runtime, true_runtime)


def run_comparison(
    data_path: Path,
    circuits_dir: Path,
    n_members: int = 5,
    n_folds: int = 5,
    epochs: int = 50,
    device: str = "cpu",
    seed: int = 42,
):
    """Run the full comparison."""
    print("="*70)
    print("ENSEMBLE + DECISION-THEORETIC INFERENCE COMPARISON")
    print("="*70)

    # Results storage
    results = {
        "single_argmax": [],
        "single_dt": [],
        "ensemble_vote": [],
        "ensemble_dt": [],
    }

    # Get input dimension
    sample_loader, _ = create_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=32,
        val_fraction=0.2,
        seed=seed,
    )
    sample_batch = next(iter(sample_loader))
    input_dim = sample_batch["features"].shape[1]

    # Create k-fold data loaders
    fold_loaders = create_kfold_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=32,
        n_folds=n_folds,
        seed=seed,
    )

    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*70}")

        # Train ensemble members
        print(f"\nTraining {n_members} ensemble members...")
        models = []
        for i in range(n_members):
            member_seed = seed + fold_idx * 1000 + i * 100
            torch.manual_seed(member_seed)
            np.random.seed(member_seed)

            model = MLPModel(
                input_dim=input_dim,
                hidden_dims=[128, 64, 32],
                device=device,
                epochs=epochs,
                inference_strategy="argmax",  # We'll override at prediction time
            )
            print(f"  Training member {i+1}/{n_members}...", end=" ", flush=True)
            model.fit(train_loader, val_loader, verbose=False, show_progress=False)
            print("done")
            models.append(model)

        # Evaluate single model (first member) with different strategies
        single_model = models[0]

        print("\nEvaluating single model...")
        # Single + argmax
        single_model.inference_strategy = "argmax"
        scores = evaluate_model_with_strategy(single_model, val_loader, strategy="argmax")
        results["single_argmax"].append(scores["combined_score"])
        print(f"  Single + Argmax:           {scores['combined_score']:.4f} "
              f"(thresh: {scores['threshold_score']:.4f}, runtime: {scores['runtime_score']:.4f})")

        # Single + decision-theoretic
        scores = evaluate_model_with_strategy(single_model, val_loader, strategy="decision_theoretic")
        results["single_dt"].append(scores["combined_score"])
        print(f"  Single + Decision-Theoretic: {scores['combined_score']:.4f} "
              f"(thresh: {scores['threshold_score']:.4f}, runtime: {scores['runtime_score']:.4f})")

        # Create ensemble
        ensemble = EnsembleModel(models=models, inference_strategy="vote")

        print("\nEvaluating ensemble...")
        # Ensemble + vote
        scores_vote = evaluate_model_with_strategy(ensemble, val_loader, strategy="vote")
        results["ensemble_vote"].append(scores_vote["combined_score"])
        print(f"  Ensemble + Vote:           {scores_vote['combined_score']:.4f} "
              f"(thresh: {scores_vote['threshold_score']:.4f}, runtime: {scores_vote['runtime_score']:.4f})")

        # Ensemble + decision-theoretic
        scores_dt = evaluate_model_with_strategy(ensemble, val_loader, strategy="decision_theoretic")
        results["ensemble_dt"].append(scores_dt["combined_score"])
        print(f"  Ensemble + Decision-Theoretic: {scores_dt['combined_score']:.4f} "
              f"(thresh: {scores_dt['threshold_score']:.4f}, runtime: {scores_dt['runtime_score']:.4f})")

        # Analyze ensemble uncertainty
        print("\nEnsemble uncertainty analysis:")
        features = extract_features(val_loader)
        uncertainty_info = ensemble.predict_with_uncertainty(features)
        agreement = uncertainty_info["member_agreement"]
        print(f"  Mean member agreement: {agreement.mean():.4f}")
        print(f"  Min member agreement:  {agreement.min():.4f}")
        print(f"  Samples with <60% agreement: {(agreement < 0.6).sum()}/{len(agreement)}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY (across all folds)")
    print("="*70)

    print(f"\n{'Strategy':<35} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*70)
    for name, scores in results.items():
        scores = np.array(scores)
        print(f"{name:<35} {scores.mean():>10.4f} {scores.std():>10.4f} "
              f"{scores.min():>10.4f} {scores.max():>10.4f}")

    # Statistical comparison
    print("\n" + "-"*70)
    print("Pairwise improvements:")

    single_argmax = np.array(results["single_argmax"])
    single_dt = np.array(results["single_dt"])
    ensemble_vote = np.array(results["ensemble_vote"])
    ensemble_dt = np.array(results["ensemble_dt"])

    def compare(name1, scores1, name2, scores2):
        diff = scores2 - scores1
        print(f"  {name2} vs {name1}: {diff.mean():+.4f} ± {diff.std():.4f}")

    compare("Single+Argmax", single_argmax, "Single+DT", single_dt)
    compare("Single+Argmax", single_argmax, "Ensemble+Vote", ensemble_vote)
    compare("Ensemble+Vote", ensemble_vote, "Ensemble+DT", ensemble_dt)
    compare("Single+Argmax", single_argmax, "Ensemble+DT", ensemble_dt)

    # Key hypothesis test
    print("\n" + "-"*70)
    print("KEY HYPOTHESIS TEST:")
    print("Does DT help more for ensembles than for single models?")
    single_dt_improvement = single_dt - single_argmax
    ensemble_dt_improvement = ensemble_dt - ensemble_vote
    diff_in_improvements = ensemble_dt_improvement - single_dt_improvement
    print(f"  Single model DT improvement:   {single_dt_improvement.mean():+.4f}")
    print(f"  Ensemble DT improvement:       {ensemble_dt_improvement.mean():+.4f}")
    print(f"  Difference:                    {diff_in_improvements.mean():+.4f}")

    if ensemble_dt_improvement.mean() > single_dt_improvement.mean():
        print("\n  ✓ Ensemble benefits MORE from decision-theoretic inference")
    else:
        print("\n  ✗ Ensemble does NOT benefit more from decision-theoretic inference")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare ensemble + decision-theoretic inference")
    parser.add_argument("--n-members", type=int, default=5, help="Number of ensemble members")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per member")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    run_comparison(
        data_path=data_path,
        circuits_dir=circuits_dir,
        n_members=args.n_members,
        n_folds=args.n_folds,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

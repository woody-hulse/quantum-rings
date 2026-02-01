#!/usr/bin/env python3
"""
Test ensemble + decision-theoretic inference for GNN models.

Compares:
1. Single GNN + argmax
2. Single GNN + decision-theoretic
3. Ensemble GNN + vote
4. Ensemble GNN + decision-theoretic
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.model import create_gnn_model
from gnn.train import GNNTrainer, extract_labels, set_all_seeds
from gnn.dataset import (
    create_graph_data_loaders,
    create_kfold_graph_data_loaders,
    THRESHOLD_LADDER,
    GLOBAL_FEAT_DIM,
)
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from gnn.augmentation import get_train_augmentation
from scoring import compute_challenge_score
from torch_geometric.loader import DataLoader as PyGDataLoader


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


class GNNEnsemble:
    """Ensemble of GNN models with different inference strategies."""

    def __init__(
        self,
        models: List[torch.nn.Module],
        trainers: List[GNNTrainer],
        device: str = "cpu",
    ):
        self.models = models
        self.trainers = trainers
        self.device = device
        self.num_classes = 9
        self.score_matrix = get_score_matrix(self.num_classes)

    @torch.no_grad()
    def get_member_logits(
        self,
        loader: PyGDataLoader,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """Get logits from all ensemble members.

        Returns:
            all_thresh_logits: List of (n_samples, num_classes) arrays per member
            all_runtime_preds: List of (n_samples,) arrays per member
            true_thresh: (n_samples,) ground truth threshold values
            true_runtime: (n_samples,) ground truth runtime values
        """
        all_thresh_logits = []
        all_runtime_preds = []

        for model, trainer in zip(self.models, self.trainers):
            model.eval()
            member_logits = []
            member_runtime = []

            for batch in loader:
                batch = batch.to(self.device)

                threshold_logits, runtime_pred = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    edge_gate_type=batch.edge_gate_type,
                    batch=batch.batch,
                    global_features=batch.global_features,
                )

                member_logits.append(threshold_logits.cpu().numpy())
                member_runtime.append(np.expm1(runtime_pred.cpu().numpy()))

            all_thresh_logits.append(np.concatenate(member_logits, axis=0))
            all_runtime_preds.append(np.concatenate(member_runtime, axis=0))

        true_thresh, true_runtime = extract_labels(loader)

        return all_thresh_logits, all_runtime_preds, true_thresh, true_runtime

    def predict_single(
        self,
        loader: PyGDataLoader,
        member_idx: int = 0,
        strategy: str = "argmax",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from a single ensemble member."""
        all_thresh_logits, all_runtime_preds, _, _ = self.get_member_logits(loader)

        logits = all_thresh_logits[member_idx]
        runtime = all_runtime_preds[member_idx]

        if strategy == "decision_theoretic":
            probs = self._softmax(logits)
            thresh_classes = self._decision_theoretic_predict(probs)
        else:  # argmax
            thresh_classes = logits.argmax(axis=1)

        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])
        return thresh_values, runtime

    def predict_ensemble(
        self,
        loader: PyGDataLoader,
        strategy: str = "vote",
        use_logit_averaging: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble predictions.

        Args:
            strategy: "vote" or "decision_theoretic"
            use_logit_averaging: If True, average logits before softmax (better for DT).
                                 If False, use empirical vote distribution.
        """
        all_thresh_logits, all_runtime_preds, _, _ = self.get_member_logits(loader)

        # Average runtime predictions
        runtime = np.mean(all_runtime_preds, axis=0)

        if use_logit_averaging:
            # Average logits, then softmax
            avg_logits = np.mean(all_thresh_logits, axis=0)
            probs = self._softmax(avg_logits)
        else:
            # Get predictions from each member, build empirical distribution
            member_preds = [logits.argmax(axis=1) for logits in all_thresh_logits]
            probs = self._build_empirical_distribution(member_preds)

        if strategy == "decision_theoretic":
            thresh_classes = self._decision_theoretic_predict(probs)
        else:  # vote / argmax
            thresh_classes = probs.argmax(axis=1)

        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])
        return thresh_values, runtime

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def _build_empirical_distribution(
        self,
        member_preds: List[np.ndarray],
    ) -> np.ndarray:
        """Build empirical probability distribution from member predictions."""
        n_members = len(member_preds)
        n_samples = len(member_preds[0])
        probs = np.zeros((n_samples, self.num_classes))

        for i in range(n_samples):
            for j in range(n_members):
                pred_class = member_preds[j][i]
                probs[i, pred_class] += 1.0 / n_members

        return probs

    def _decision_theoretic_predict(self, probs: np.ndarray) -> np.ndarray:
        """Pick classes that maximize expected challenge score."""
        expected_scores = probs @ self.score_matrix
        return expected_scores.argmax(axis=1)

    def analyze_uncertainty(
        self,
        loader: PyGDataLoader,
    ) -> Dict[str, Any]:
        """Analyze ensemble uncertainty."""
        all_thresh_logits, _, _, _ = self.get_member_logits(loader)

        # Get predictions from each member
        member_preds = [logits.argmax(axis=1) for logits in all_thresh_logits]
        member_preds = np.array(member_preds)  # (n_members, n_samples)

        # Agreement: what fraction of members agree with the majority?
        n_samples = member_preds.shape[1]
        agreements = []
        for i in range(n_samples):
            preds = member_preds[:, i]
            unique, counts = np.unique(preds, return_counts=True)
            agreements.append(counts.max() / len(preds))

        # Average logits entropy
        avg_logits = np.mean(all_thresh_logits, axis=0)
        avg_probs = self._softmax(avg_logits)
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-10), axis=1)

        return {
            "member_agreement": np.array(agreements),
            "avg_entropy": entropy,
            "mean_agreement": np.mean(agreements),
            "mean_entropy": np.mean(entropy),
            "low_agreement_samples": (np.array(agreements) < 0.6).sum(),
        }


def train_gnn_ensemble(
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    n_members: int = 5,
    base_seed: int = 42,
    hidden_dim: int = 32,
    num_layers: int = 2,
    dropout: float = 0.1,
    weight_decay: float = 1e-3,
    aug_strength: float = 0.5,
    use_ordinal: bool = False,
    epochs: int = 100,
    device: str = "cpu",
    verbose: bool = False,
) -> GNNEnsemble:
    """Train an ensemble of GNN models."""
    models = []
    trainers = []

    for i in range(n_members):
        seed = base_seed + i * 100
        set_all_seeds(seed)

        print(f"  Training GNN member {i+1}/{n_members}...", end=" ", flush=True)

        model = create_gnn_model(
            model_type="basic",
            node_feat_dim=NODE_FEAT_DIM,
            edge_feat_dim=EDGE_FEAT_DIM,
            global_feat_dim=GLOBAL_FEAT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_ordinal=use_ordinal,
        )

        augmentation = get_train_augmentation(
            qubit_perm_p=aug_strength,
            edge_dropout_p=0.1 * aug_strength,
            feature_noise_std=0.1 * aug_strength,
            temporal_jitter_std=0.05 * aug_strength,
        )

        trainer = GNNTrainer(
            model=model,
            device=device,
            weight_decay=weight_decay,
            use_ordinal=use_ordinal,
            augmentation=augmentation,
        )

        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            verbose=verbose,
            show_progress=False,
        )

        models.append(model)
        trainers.append(trainer)
        print("done")

    return GNNEnsemble(models=models, trainers=trainers, device=device)


def evaluate_strategy(
    ensemble: GNNEnsemble,
    loader: PyGDataLoader,
    strategy: str,
    is_ensemble: bool,
    use_logit_averaging: bool = True,
) -> Dict[str, float]:
    """Evaluate a specific strategy and return challenge scores."""
    true_thresh, true_runtime = extract_labels(loader)

    if is_ensemble:
        pred_thresh, pred_runtime = ensemble.predict_ensemble(
            loader, strategy=strategy, use_logit_averaging=use_logit_averaging
        )
    else:
        pred_thresh, pred_runtime = ensemble.predict_single(
            loader, member_idx=0, strategy=strategy
        )

    return compute_challenge_score(pred_thresh, true_thresh, pred_runtime, true_runtime)


def run_comparison(
    data_path: Path,
    circuits_dir: Path,
    n_members: int = 5,
    n_folds: int = 3,
    hidden_dim: int = 32,
    num_layers: int = 2,
    dropout: float = 0.1,
    weight_decay: float = 1e-3,
    aug_strength: float = 0.5,
    batch_size: int = 16,
    epochs: int = 100,
    device: str = "cpu",
    seed: int = 42,
):
    """Run full comparison of inference strategies."""
    print("=" * 70)
    print("GNN ENSEMBLE + DECISION-THEORETIC INFERENCE COMPARISON")
    print("=" * 70)
    print(f"\nGNN Configuration:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Aug strength: {aug_strength}")
    print(f"  Batch size: {batch_size}")
    print(f"  Ordinal: False (using softmax classification)")
    print(f"  Ensemble members: {n_members}")
    print(f"  K-fold CV: {n_folds}")

    results = {
        "single_argmax": [],
        "single_dt": [],
        "ensemble_vote": [],
        "ensemble_dt": [],
        "ensemble_logit_avg_vote": [],
        "ensemble_logit_avg_dt": [],
    }

    fold_loaders = create_kfold_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        n_folds=n_folds,
        batch_size=batch_size,
        seed=seed,
    )

    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'=' * 70}")

        # Train ensemble
        print(f"\nTraining {n_members} GNN ensemble members...")
        ensemble = train_gnn_ensemble(
            train_loader=train_loader,
            val_loader=val_loader,
            n_members=n_members,
            base_seed=seed + fold_idx * 1000,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            weight_decay=weight_decay,
            aug_strength=aug_strength,
            use_ordinal=False,  # Use softmax for probability extraction
            epochs=epochs,
            device=device,
        )

        # Analyze uncertainty
        print("\nEnsemble uncertainty analysis:")
        uncertainty = ensemble.analyze_uncertainty(val_loader)
        print(f"  Mean member agreement: {uncertainty['mean_agreement']:.4f}")
        print(f"  Mean entropy: {uncertainty['mean_entropy']:.4f}")
        print(f"  Low agreement samples (<60%): {uncertainty['low_agreement_samples']}/{len(uncertainty['member_agreement'])}")

        # Evaluate all strategies
        print("\nEvaluating strategies...")

        # Single model
        scores = evaluate_strategy(ensemble, val_loader, "argmax", is_ensemble=False)
        results["single_argmax"].append(scores["combined_score"])
        print(f"  Single + Argmax:              {scores['combined_score']:.4f} "
              f"(thresh: {scores['threshold_score']:.4f}, runtime: {scores['runtime_score']:.4f})")

        scores = evaluate_strategy(ensemble, val_loader, "decision_theoretic", is_ensemble=False)
        results["single_dt"].append(scores["combined_score"])
        print(f"  Single + Decision-Theoretic:  {scores['combined_score']:.4f} "
              f"(thresh: {scores['threshold_score']:.4f}, runtime: {scores['runtime_score']:.4f})")

        # Ensemble with empirical vote distribution
        scores = evaluate_strategy(ensemble, val_loader, "vote", is_ensemble=True, use_logit_averaging=False)
        results["ensemble_vote"].append(scores["combined_score"])
        print(f"  Ensemble + Vote (empirical):  {scores['combined_score']:.4f} "
              f"(thresh: {scores['threshold_score']:.4f}, runtime: {scores['runtime_score']:.4f})")

        scores = evaluate_strategy(ensemble, val_loader, "decision_theoretic", is_ensemble=True, use_logit_averaging=False)
        results["ensemble_dt"].append(scores["combined_score"])
        print(f"  Ensemble + DT (empirical):    {scores['combined_score']:.4f} "
              f"(thresh: {scores['threshold_score']:.4f}, runtime: {scores['runtime_score']:.4f})")

        # Ensemble with logit averaging (better for DT)
        scores = evaluate_strategy(ensemble, val_loader, "vote", is_ensemble=True, use_logit_averaging=True)
        results["ensemble_logit_avg_vote"].append(scores["combined_score"])
        print(f"  Ensemble + Vote (logit avg):  {scores['combined_score']:.4f} "
              f"(thresh: {scores['threshold_score']:.4f}, runtime: {scores['runtime_score']:.4f})")

        scores = evaluate_strategy(ensemble, val_loader, "decision_theoretic", is_ensemble=True, use_logit_averaging=True)
        results["ensemble_logit_avg_dt"].append(scores["combined_score"])
        print(f"  Ensemble + DT (logit avg):    {scores['combined_score']:.4f} "
              f"(thresh: {scores['threshold_score']:.4f}, runtime: {scores['runtime_score']:.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (across all folds)")
    print("=" * 70)

    print(f"\n{'Strategy':<35} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 75)
    for name, scores in results.items():
        scores = np.array(scores)
        print(f"{name:<35} {scores.mean():>10.4f} {scores.std():>10.4f} "
              f"{scores.min():>10.4f} {scores.max():>10.4f}")

    # Find best strategy
    best_name = max(results.keys(), key=lambda k: np.mean(results[k]))
    best_score = np.mean(results[best_name])
    print(f"\nBest strategy: {best_name} ({best_score:.4f})")

    # Key comparisons
    print("\n" + "-" * 70)
    print("Key comparisons:")

    single_argmax = np.array(results["single_argmax"])
    single_dt = np.array(results["single_dt"])
    ensemble_vote = np.array(results["ensemble_vote"])
    ensemble_dt = np.array(results["ensemble_dt"])
    ensemble_logit_dt = np.array(results["ensemble_logit_avg_dt"])

    print(f"  Single DT vs Single Argmax:     {(single_dt - single_argmax).mean():+.4f}")
    print(f"  Ensemble Vote vs Single Argmax: {(ensemble_vote - single_argmax).mean():+.4f}")
    print(f"  Ensemble DT vs Ensemble Vote:   {(ensemble_dt - ensemble_vote).mean():+.4f}")
    print(f"  Logit-avg DT vs Logit-avg Vote: {(ensemble_logit_dt - np.array(results['ensemble_logit_avg_vote'])).mean():+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test GNN ensemble + decision-theoretic inference")
    parser.add_argument("--n-members", type=int, default=5, help="Number of ensemble members")
    parser.add_argument("--n-folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--aug-strength", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
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
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        aug_strength=args.aug_strength,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

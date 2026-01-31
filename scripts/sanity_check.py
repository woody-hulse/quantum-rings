#!/usr/bin/env python3
"""
Sanity check script to verify loss functions and model behavior.

Tests:
1. RuntimeScoringLoss computes correct min(r, 1/r) scores
2. ThresholdScoringLoss computes correct 2^(-steps) scores
3. Loss and challenge scoring are aligned
4. Model can overfit on small synthetic data

Usage:
    python scripts/sanity_check.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from losses import (
    ThresholdScoringLoss,
    RuntimeScoringLoss,
    ChallengeScoringLoss,
    compute_scoring_metrics,
)
from scoring import compute_challenge_score
from data_loader import THRESHOLD_LADDER


def test_runtime_scoring_loss():
    """Test that RuntimeScoringLoss matches min(r, 1/r) formula."""
    print("\n" + "="*60)
    print("TEST: RuntimeScoringLoss")
    print("="*60)
    
    loss_fn = RuntimeScoringLoss(reduction="none")
    
    test_cases = [
        # (true_runtime, pred_runtime, expected_score)
        (1.0, 1.0, 1.0),      # Perfect prediction
        (1.0, 2.0, 0.5),      # Off by factor of 2
        (2.0, 1.0, 0.5),      # Off by factor of 2 (reversed)
        (1.0, 4.0, 0.25),     # Off by factor of 4
        (4.0, 1.0, 0.25),     # Off by factor of 4 (reversed)
        (1.0, 10.0, 0.1),     # Off by factor of 10
        (0.1, 0.2, 0.5),      # Small values, factor of 2
        (0.01, 0.01, 1.0),    # Very small, perfect
        (100.0, 100.0, 1.0),  # Large values, perfect
        (100.0, 200.0, 0.5),  # Large values, factor of 2
    ]
    
    all_passed = True
    
    for true_rt, pred_rt, expected_score in test_cases:
        # Convert to log1p space (as the model uses)
        log1p_true = torch.tensor([np.log1p(true_rt)])
        log1p_pred = torch.tensor([np.log1p(pred_rt)])
        
        loss = loss_fn(log1p_pred, log1p_true)
        computed_score = 1.0 - loss.item()
        
        # Also compute directly for comparison
        r = pred_rt / true_rt
        direct_score = min(r, 1/r)
        
        passed = abs(computed_score - expected_score) < 0.01
        status = "PASS" if passed else "FAIL"
        
        if not passed:
            all_passed = False
        
        print(f"  [{status}] true={true_rt:.2f}, pred={pred_rt:.2f}")
        print(f"         Expected score: {expected_score:.4f}")
        print(f"         Computed score: {computed_score:.4f}")
        print(f"         Direct min(r,1/r): {direct_score:.4f}")
    
    return all_passed


def test_threshold_scoring_loss():
    """Test that ThresholdScoringLoss matches 2^(-steps) formula."""
    print("\n" + "="*60)
    print("TEST: ThresholdScoringLoss")
    print("="*60)
    
    num_classes = len(THRESHOLD_LADDER)
    loss_fn = ThresholdScoringLoss(num_classes=num_classes)
    
    test_cases = [
        # (true_class, pred_class, expected_score)
        (0, 0, 1.0),      # Correct: threshold 1 -> 1
        (0, 1, 0.5),      # Over by 1: threshold 1 -> 2
        (0, 2, 0.25),     # Over by 2: threshold 1 -> 4
        (4, 4, 1.0),      # Correct: threshold 16 -> 16
        (4, 5, 0.5),      # Over by 1: threshold 16 -> 32
        (4, 6, 0.25),     # Over by 2: threshold 16 -> 64
        (4, 3, 0.0),      # Under: threshold 16 -> 8 (VIOLATION)
        (8, 0, 0.0),      # Severe under: threshold 256 -> 1
    ]
    
    all_passed = True
    
    for true_cls, pred_cls, expected_score in test_cases:
        # Create one-hot logits (make predicted class have high logit)
        logits = torch.zeros(1, num_classes) - 10.0  # Very negative
        logits[0, pred_cls] = 10.0  # Very positive for predicted class
        
        targets = torch.tensor([true_cls])
        
        loss = loss_fn(logits, targets)
        
        # For one-hot predictions, expected loss = 1 - score (for valid)
        # or = 1.0 (for underprediction, since score = 0)
        if pred_cls < true_cls:
            expected_loss = 1.0  # Underprediction
        else:
            expected_loss = 1.0 - expected_score
        
        passed = abs(loss.item() - expected_loss) < 0.01
        status = "PASS" if passed else "FAIL"
        
        if not passed:
            all_passed = False
        
        true_thresh = THRESHOLD_LADDER[true_cls]
        pred_thresh = THRESHOLD_LADDER[pred_cls]
        
        print(f"  [{status}] true={true_thresh}, pred={pred_thresh}")
        print(f"         Expected score: {expected_score:.4f}, loss: {expected_loss:.4f}")
        print(f"         Computed loss: {loss.item():.4f}")
    
    return all_passed


def test_challenge_score_alignment():
    """Test that our loss aligns with official challenge scoring."""
    print("\n" + "="*60)
    print("TEST: Challenge Score Alignment")
    print("="*60)
    
    np.random.seed(42)
    
    # Generate random test data
    n_samples = 100
    true_thresh_idx = np.random.randint(0, len(THRESHOLD_LADDER), n_samples)
    pred_thresh_idx = np.clip(
        true_thresh_idx + np.random.randint(-2, 3, n_samples),
        0, len(THRESHOLD_LADDER) - 1
    )
    
    true_runtime = np.random.exponential(10.0, n_samples)
    pred_runtime = true_runtime * np.exp(np.random.randn(n_samples) * 0.5)
    
    # Convert to threshold values
    true_thresh = np.array([THRESHOLD_LADDER[i] for i in true_thresh_idx])
    pred_thresh = np.array([THRESHOLD_LADDER[i] for i in pred_thresh_idx])
    
    # Compute official challenge score
    official = compute_challenge_score(pred_thresh, true_thresh, pred_runtime, true_runtime)
    
    # Compute using our loss functions
    log1p_true = torch.tensor(np.log1p(true_runtime), dtype=torch.float32)
    log1p_pred = torch.tensor(np.log1p(pred_runtime), dtype=torch.float32)
    
    # Create one-hot logits for predictions
    logits = torch.zeros(n_samples, len(THRESHOLD_LADDER)) - 10.0
    for i, pred_idx in enumerate(pred_thresh_idx):
        logits[i, pred_idx] = 10.0
    
    targets = torch.tensor(true_thresh_idx)
    
    our_metrics = compute_scoring_metrics(logits, log1p_pred, targets, log1p_true)
    
    print(f"\n  Official Challenge Scoring:")
    print(f"    Threshold Score: {official['threshold_score']:.4f}")
    print(f"    Runtime Score:   {official['runtime_score']:.4f}")
    print(f"    Combined Score:  {official['combined_score']:.4f}")
    
    print(f"\n  Our compute_scoring_metrics:")
    print(f"    Threshold Score: {our_metrics['threshold_score']:.4f}")
    print(f"    Runtime Score:   {our_metrics['runtime_score']:.4f}")
    print(f"    Combined Score:  {our_metrics['combined_score']:.4f}")
    
    thresh_match = abs(official['threshold_score'] - our_metrics['threshold_score']) < 0.01
    runtime_match = abs(official['runtime_score'] - our_metrics['runtime_score']) < 0.01
    combined_match = abs(official['combined_score'] - our_metrics['combined_score']) < 0.01
    
    all_passed = thresh_match and runtime_match and combined_match
    
    print(f"\n  Threshold match: {'PASS' if thresh_match else 'FAIL'}")
    print(f"  Runtime match:   {'PASS' if runtime_match else 'FAIL'}")
    print(f"  Combined match:  {'PASS' if combined_match else 'FAIL'}")
    
    return all_passed


def test_model_overfit():
    """Test that the model can overfit on a tiny synthetic dataset."""
    print("\n" + "="*60)
    print("TEST: Model Overfitting on Synthetic Data")
    print("="*60)
    
    from models.mlp import MLPModel
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create tiny synthetic dataset
    n_samples = 16
    input_dim = 10
    
    # Fixed features
    X = torch.randn(n_samples, input_dim)
    
    # Fixed targets - make them learnable patterns
    threshold_classes = torch.randint(0, len(THRESHOLD_LADDER), (n_samples,))
    runtimes = torch.abs(torch.randn(n_samples) * 5 + 10)  # Positive runtimes
    log_runtimes = torch.log1p(runtimes)
    
    # Create a simple DataLoader-like structure
    class FakeLoader:
        def __init__(self, X, thresh, runtime):
            self.data = {
                "features": X,
                "threshold_class": thresh,
                "log_runtime": runtime,
            }
        
        def __iter__(self):
            yield self.data
        
        def __len__(self):
            return 1
    
    train_loader = FakeLoader(X, threshold_classes, log_runtimes)
    val_loader = train_loader  # Same data for overfitting test
    
    # Create model with standard loss (easier to overfit than scoring loss)
    model = MLPModel(
        input_dim=input_dim,
        hidden_dims=[32, 16],  # Smaller network
        dropout=0.0,  # No dropout for overfitting
        lr=5e-2,  # Higher learning rate
        epochs=30,
        early_stopping_patience=30,  # Don't stop early
        use_scoring_loss=False,  # Standard loss is easier to overfit
        device="cpu",
    )
    
    print("\n  Training model to overfit on synthetic data...")
    model.fit(train_loader, val_loader, verbose=False)
    
    # Evaluate
    model.network.eval()
    with torch.no_grad():
        features = model._normalize(X.to(model.device))
        logits, pred_runtime = model.network(features)
        pred_classes = logits.argmax(dim=1)
        
        # Threshold accuracy
        thresh_acc = (pred_classes == threshold_classes).float().mean().item()
        
        # Runtime error
        runtime_mae = torch.abs(pred_runtime.cpu() - log_runtimes).mean().item()
        
        # Challenge score
        metrics = compute_scoring_metrics(logits.cpu(), pred_runtime.cpu(), threshold_classes, log_runtimes)
    
    print(f"\n  Results after training:")
    print(f"    Threshold Accuracy: {thresh_acc:.2%}")
    print(f"    Runtime MAE (log1p): {runtime_mae:.4f}")
    print(f"    Challenge Threshold Score: {metrics['threshold_score']:.4f}")
    print(f"    Challenge Runtime Score: {metrics['runtime_score']:.4f}")
    print(f"    Challenge Combined Score: {metrics['combined_score']:.4f}")
    
    # Should achieve reasonable overfitting
    thresh_pass = thresh_acc > 0.5  # At least 50% accuracy
    runtime_pass = metrics['runtime_score'] > 0.3  # Reasonable runtime prediction
    combined_pass = metrics['combined_score'] > 0.2  # Reasonable combined score
    
    all_passed = thresh_pass and runtime_pass and combined_pass
    
    print(f"\n  Threshold overfit (>50%): {'PASS' if thresh_pass else 'FAIL'}")
    print(f"  Runtime overfit (>0.3): {'PASS' if runtime_pass else 'FAIL'}")
    print(f"  Combined overfit (>0.2): {'PASS' if combined_pass else 'FAIL'}")
    
    return all_passed


def test_loss_gradient_flow():
    """Test that gradients flow properly through the loss."""
    print("\n" + "="*60)
    print("TEST: Gradient Flow")
    print("="*60)
    
    num_classes = len(THRESHOLD_LADDER)
    batch_size = 16
    
    # Create learnable parameters
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    runtime_pred = torch.randn(batch_size, requires_grad=True)
    
    # Fixed targets
    targets = torch.randint(0, num_classes, (batch_size,))
    runtime_targets = torch.abs(torch.randn(batch_size)) + 0.1
    log_runtime_targets = torch.log1p(runtime_targets)
    
    # Test ChallengeScoringLoss
    loss_fn = ChallengeScoringLoss(multiplicative=True)
    
    result = loss_fn(logits, runtime_pred, targets, log_runtime_targets)
    loss = result["loss"]
    
    # Backward pass
    loss.backward()
    
    logits_grad = logits.grad is not None and not torch.isnan(logits.grad).any()
    runtime_grad = runtime_pred.grad is not None and not torch.isnan(runtime_pred.grad).any()
    
    print(f"\n  Logits have valid gradients: {'PASS' if logits_grad else 'FAIL'}")
    print(f"  Runtime has valid gradients: {'PASS' if runtime_grad else 'FAIL'}")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Threshold loss: {result['threshold_loss'].item():.4f}")
    print(f"  Runtime loss: {result['runtime_loss'].item():.4f}")
    
    return logits_grad and runtime_grad


def main():
    print("\n" + "="*60)
    print("SANITY CHECK: Loss Functions and Model Behavior")
    print("="*60)
    
    results = {}
    
    results["RuntimeScoringLoss"] = test_runtime_scoring_loss()
    results["ThresholdScoringLoss"] = test_threshold_scoring_loss()
    results["ChallengeScoreAlignment"] = test_challenge_score_alignment()
    results["GradientFlow"] = test_loss_gradient_flow()
    results["ModelOverfit"] = test_model_overfit()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {name}: {status}")
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Please review above")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

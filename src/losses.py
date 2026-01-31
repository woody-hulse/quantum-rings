"""
Custom loss functions aligned with the challenge scoring metrics.

The challenge scoring is:
- threshold_score = 2^(-steps_over) if pred >= true, else 0
- runtime_score = min(r, 1/r) where r = pred_time / true_time
- task_score = threshold_score * runtime_score

These losses convert scores to losses (negative scores) for gradient descent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from data_loader import THRESHOLD_LADDER


class ThresholdScoringLoss(nn.Module):
    """
    Loss function for threshold prediction that matches challenge scoring.
    
    Scoring rules:
    - Correct prediction: score = 1.0 (loss = 0)
    - Over by k rungs: score = 2^(-k) (loss increases with k)
    - Under prediction: score = 0 (maximum loss)
    
    Uses soft cross-entropy weighted by the scoring penalties.
    """
    
    def __init__(
        self,
        num_classes: int = 9,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            num_classes: Number of threshold classes (default 9 for ladder)
            label_smoothing: Label smoothing factor (default 0.0)
        """
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        # Precompute the scoring matrix
        # score_matrix[true_class, pred_class] = score for that prediction
        score_matrix = torch.zeros(num_classes, num_classes)
        for true_idx in range(num_classes):
            for pred_idx in range(num_classes):
                if pred_idx < true_idx:
                    # Underprediction: score = 0
                    score_matrix[true_idx, pred_idx] = 0.0
                else:
                    # Correct or overprediction: score = 2^(-steps_over)
                    steps_over = pred_idx - true_idx
                    score_matrix[true_idx, pred_idx] = 2.0 ** (-steps_over)
        
        # Convert to loss matrix: loss = 1 - score (so perfect = 0, worst = 1)
        loss_matrix = 1.0 - score_matrix
        
        self.register_buffer("loss_matrix", loss_matrix)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the scoring-aligned loss.
        
        Args:
            logits: Predicted logits of shape (batch_size, num_classes)
            targets: True class indices of shape (batch_size,)
            
        Returns:
            Scalar loss value
        """
        batch_size = logits.shape[0]
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            probs = (1 - self.label_smoothing) * probs + self.label_smoothing / self.num_classes
        
        # Get the loss weights for each sample based on true class
        # loss_weights[i, j] = loss for predicting class j when true is targets[i]
        loss_weights = self.loss_matrix[targets]  # (batch_size, num_classes)
        
        # Compute expected loss: sum over classes of prob * loss
        sample_losses = (probs * loss_weights).sum(dim=1)
        
        return sample_losses.mean()


class RuntimeScoringLoss(nn.Module):
    """
    Loss function for runtime prediction that matches challenge scoring.
    
    Scoring rule:
    - runtime_score = min(r, 1/r) where r = pred_time / true_time
    
    This is equivalent to:
    - Perfect prediction (r=1): score = 1.0
    - Off by factor of 2: score = 0.5
    - Off by factor of 4: score = 0.25
    
    IMPORTANT: Input is log1p(runtime), so we must convert to linear space
    to correctly compute the ratio score.
    """
    
    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        """
        Args:
            reduction: "mean", "sum", or "none"
            eps: Small value to prevent division by zero
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        log1p_pred: torch.Tensor,
        log1p_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the scoring-aligned runtime loss.
        
        Args:
            log1p_pred: Predicted log(1 + runtime) of shape (batch_size,)
            log1p_true: True log(1 + runtime) of shape (batch_size,)
            
        Returns:
            Loss value (higher = worse prediction)
        """
        # Convert from log1p space back to linear runtime
        # expm1(x) = exp(x) - 1, inverse of log1p
        pred_runtime = torch.expm1(log1p_pred).clamp(min=self.eps)
        true_runtime = torch.expm1(log1p_true).clamp(min=self.eps)
        
        # Compute ratio r = pred / true
        r = pred_runtime / true_runtime
        
        # Score = min(r, 1/r), ranges from 1 (perfect) to 0 (very wrong)
        # Equivalent to: exp(-|log(r)|)
        score = torch.minimum(r, 1.0 / r)
        
        # Loss = 1 - score, ranges from 0 (perfect) to 1 (very wrong)
        loss = 1.0 - score
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ChallengeScoringLoss(nn.Module):
    """
    Combined loss function that matches the full challenge scoring.
    
    The challenge scores each task as:
        task_score = threshold_score * runtime_score
    
    This loss combines both components with configurable weights.
    """
    
    def __init__(
        self,
        threshold_weight: float = 1.0,
        runtime_weight: float = 1.0,
        multiplicative: bool = True,
    ):
        """
        Args:
            threshold_weight: Weight for threshold loss component (used in additive mode)
            runtime_weight: Weight for runtime loss component (used in additive mode)
            multiplicative: If True, combine losses multiplicatively like the scoring (default)
        """
        super().__init__()
        self.threshold_weight = threshold_weight
        self.runtime_weight = runtime_weight
        self.multiplicative = multiplicative
        
        self.threshold_loss = ThresholdScoringLoss()
        self.runtime_loss = RuntimeScoringLoss(reduction="none")
    
    def forward(
        self,
        threshold_logits: torch.Tensor,
        runtime_pred: torch.Tensor,
        threshold_targets: torch.Tensor,
        runtime_targets: torch.Tensor,
    ) -> dict:
        """
        Compute the combined challenge-aligned loss.
        
        Args:
            threshold_logits: Predicted threshold logits (batch_size, num_classes)
            runtime_pred: Predicted log runtime (batch_size,)
            threshold_targets: True threshold class indices (batch_size,)
            runtime_targets: True log runtime (batch_size,)
            
        Returns:
            Dict with 'loss', 'threshold_loss', 'runtime_loss'
        """
        # Compute individual losses
        thresh_loss = self.threshold_loss(threshold_logits, threshold_targets)
        runtime_losses = self.runtime_loss(runtime_pred, runtime_targets)
        
        if self.multiplicative:
            # Approximate multiplicative scoring
            # Get threshold probabilities and compute expected threshold score
            probs = F.softmax(threshold_logits, dim=1)
            score_matrix = 1.0 - self.threshold_loss.loss_matrix.clamp(max=1.0)
            thresh_scores = (probs * score_matrix[threshold_targets]).sum(dim=1)
            
            # Runtime score
            runtime_scores = 1.0 - runtime_losses
            
            # Combined score (what we want to maximize)
            combined_scores = thresh_scores * runtime_scores
            
            # Loss = 1 - score (what we minimize)
            combined_loss = (1.0 - combined_scores).mean()
            
            return {
                "loss": combined_loss,
                "threshold_loss": thresh_loss,
                "runtime_loss": runtime_losses.mean(),
            }
        else:
            # Additive combination
            runtime_loss_mean = runtime_losses.mean()
            combined_loss = (
                self.threshold_weight * thresh_loss +
                self.runtime_weight * runtime_loss_mean
            )
            
            return {
                "loss": combined_loss,
                "threshold_loss": thresh_loss,
                "runtime_loss": runtime_loss_mean,
            }


def compute_scoring_metrics(
    threshold_logits: torch.Tensor,
    runtime_pred: torch.Tensor,
    threshold_targets: torch.Tensor,
    runtime_targets: torch.Tensor,
) -> dict:
    """
    Compute the actual challenge scoring metrics (non-differentiable).
    
    Matches official scoring: when threshold is underpredicted, both
    threshold_score AND runtime_score are 0 for that sample.
    
    Returns:
        Dict with threshold_score, runtime_score, combined_score
    """
    with torch.no_grad():
        # Get predicted threshold class
        pred_class = threshold_logits.argmax(dim=1)
        
        # Compute runtime scores - convert from log1p to linear first
        pred_runtime = torch.expm1(runtime_pred).clamp(min=1e-8)
        true_runtime = torch.expm1(runtime_targets).clamp(min=1e-8)
        r = pred_runtime / true_runtime
        base_runtime_scores = torch.minimum(r, 1.0 / r)
        
        # Compute threshold and runtime scores together (official behavior)
        thresh_scores = []
        runtime_scores = []
        for i, (pred_idx, true_idx) in enumerate(zip(pred_class.tolist(), threshold_targets.tolist())):
            if pred_idx < true_idx:
                # Underprediction: both scores are 0
                thresh_scores.append(0.0)
                runtime_scores.append(0.0)
            else:
                steps_over = pred_idx - true_idx
                thresh_scores.append(2.0 ** (-steps_over))
                runtime_scores.append(base_runtime_scores[i].item())
        
        thresh_scores = torch.tensor(thresh_scores, device=threshold_logits.device)
        runtime_scores = torch.tensor(runtime_scores, device=threshold_logits.device)
        
        # Combined scores (per-sample)
        combined_scores = thresh_scores * runtime_scores
        
        return {
            "threshold_score": thresh_scores.mean().item(),
            "runtime_score": runtime_scores.mean().item(),
            "combined_score": combined_scores.mean().item(),
        }

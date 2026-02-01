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


class AsymmetricLog2Loss(nn.Module):
    """
    Asymmetric loss with weighted cross-entropy for threshold and log2 loss for runtime.
    
    For threshold prediction:
    - Uses weighted cross-entropy with soft targets: true class = 1.0, 
      each class above is "half as correct" (0.5, 0.25, 0.125, ...)
    - Adds explicit penalty for probability mass on classes below true (underprediction)
    - Steepness controls how much worse underprediction is vs overprediction
    
    For runtime prediction:
    - Loss = 1 + |log2(pred/true)|: exact→1, 2x or ½x→2, 4x or ¼x→4
    """
    
    def __init__(
        self,
        num_threshold_classes: int = 9,
        underprediction_steepness: float = 5.0,
        threshold_weight: float = 0.5,
        runtime_weight: float = 0.5,
        eps: float = 1e-8,
    ):
        """
        Args:
            num_threshold_classes: Number of threshold classes in the ladder
            underprediction_steepness: Multiplier for underprediction penalty (>1 = steeper)
            threshold_weight: Weight for threshold component in combined loss
            runtime_weight: Weight for runtime component in combined loss
            eps: Small value to prevent numerical issues
        """
        super().__init__()
        self.num_threshold_classes = num_threshold_classes
        self.underprediction_steepness = underprediction_steepness
        self.threshold_weight = threshold_weight
        self.runtime_weight = runtime_weight
        self.eps = eps
        
        # Precompute soft target matrix for weighted cross-entropy
        # soft_targets[t, i] = target weight for class i when true class is t
        # For i >= t: weight = 2^(-(i-t)), normalized
        # For i < t: weight = 0 (handled separately with underprediction penalty)
        soft_targets = torch.zeros(num_threshold_classes, num_threshold_classes)
        for t in range(num_threshold_classes):
            for i in range(t, num_threshold_classes):
                soft_targets[t, i] = 2.0 ** (-(i - t))
            # Normalize to sum to 1
            soft_targets[t] = soft_targets[t] / soft_targets[t].sum()
        self.register_buffer("soft_targets", soft_targets)
        
        # Underprediction penalty weights: how bad is predicting class i when true is t?
        # For i < t: penalty = 2^(t - i) (exponentially worse for being further below)
        underpred_penalty = torch.zeros(num_threshold_classes, num_threshold_classes)
        for t in range(num_threshold_classes):
            for i in range(t):
                underpred_penalty[t, i] = 2.0 ** (t - i)
        self.register_buffer("underpred_penalty", underpred_penalty)
    
    def threshold_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted cross-entropy with asymmetric underprediction penalty.
        
        Args:
            logits: Predicted logits of shape (batch_size, num_classes)
            targets: True class indices of shape (batch_size,)
        """
        probs = F.softmax(logits, dim=1)  # (batch_size, num_classes)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Get soft targets for each sample's true class
        soft_tgt = self.soft_targets[targets]  # (batch_size, num_classes)
        
        # Weighted cross-entropy: -sum(soft_target * log_prob)
        ce_loss = -(soft_tgt * log_probs).sum(dim=1)
        
        # Underprediction penalty: penalize probability mass on classes below true
        underpred_weights = self.underpred_penalty[targets]  # (batch_size, num_classes)
        underpred_loss = (probs * underpred_weights).sum(dim=1)
        
        # Combined threshold loss
        loss = ce_loss + self.underprediction_steepness * underpred_loss
        return loss.mean()
    
    def runtime_loss(
        self,
        log1p_pred: torch.Tensor,
        log1p_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss = 1 + |log2(pred/true)|: exact→1, 2x or ½x→2, 4x or ¼x→4.
        Uses log2(1 + runtime) for stability when pred is small.
        """
        pred_runtime = torch.expm1(log1p_pred).clamp(min=0.0)
        true_runtime = torch.expm1(log1p_true).clamp(min=0.0)

        log2_pred = torch.log2(1.0 + pred_runtime)
        log2_true = torch.log2(1.0 + true_runtime)
        error = log2_pred - log2_true

        loss = 1.0 + torch.abs(error)
        return loss.mean()
    
    def forward(
        self,
        threshold_logits: torch.Tensor,
        runtime_pred: torch.Tensor,
        threshold_targets: torch.Tensor,
        runtime_targets: torch.Tensor,
    ) -> dict:
        """
        Compute combined asymmetric loss.
        
        Args:
            threshold_logits: Predicted threshold logits (batch_size, num_classes)
            runtime_pred: Predicted log1p runtime (batch_size,)
            threshold_targets: True threshold class indices (batch_size,)
            runtime_targets: True log1p runtime (batch_size,)
            
        Returns:
            Dict with 'loss', 'threshold_loss', 'runtime_loss'
        """
        thresh_loss = self.threshold_loss(threshold_logits, threshold_targets)
        rt_loss = self.runtime_loss(runtime_pred, runtime_targets)
        
        # Weighted average (normalized weights)
        total_weight = self.threshold_weight + self.runtime_weight
        combined_loss = (
            self.threshold_weight * thresh_loss +
            self.runtime_weight * rt_loss
        ) / total_weight
        
        return {
            "loss": combined_loss,
            "threshold_loss": thresh_loss,
            "runtime_loss": rt_loss,
        }


class AsymmetricLog2LossWithHardPenalty(nn.Module):
    """
    Variant of AsymmetricLog2Loss with an additional hard penalty for
    discrete underprediction of threshold.
    
    This adds a term that directly penalizes the probability mass assigned
    to classes below the true threshold, weighted by how many rungs below.
    """
    
    def __init__(
        self,
        num_threshold_classes: int = 9,
        underprediction_steepness: float = 5.0,
        hard_penalty_weight: float = 1.0,
        threshold_weight: float = 0.5,
        runtime_weight: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.hard_penalty_weight = hard_penalty_weight
        
        self.base_loss = AsymmetricLog2Loss(
            num_threshold_classes=num_threshold_classes,
            underprediction_steepness=underprediction_steepness,
            threshold_weight=threshold_weight,
            runtime_weight=runtime_weight,
            eps=eps,
        )
        
        # Precompute underprediction penalty matrix
        # penalty_matrix[true_class, pred_class] = rungs_below if pred < true, else 0
        penalty_matrix = torch.zeros(num_threshold_classes, num_threshold_classes)
        for true_idx in range(num_threshold_classes):
            for pred_idx in range(num_threshold_classes):
                if pred_idx < true_idx:
                    rungs_below = true_idx - pred_idx
                    # Exponential penalty for being further below
                    penalty_matrix[true_idx, pred_idx] = 2.0 ** rungs_below
        
        self.register_buffer("penalty_matrix", penalty_matrix)
    
    def forward(
        self,
        threshold_logits: torch.Tensor,
        runtime_pred: torch.Tensor,
        threshold_targets: torch.Tensor,
        runtime_targets: torch.Tensor,
    ) -> dict:
        """Compute combined loss with hard underprediction penalty."""
        result = self.base_loss(
            threshold_logits, runtime_pred,
            threshold_targets, runtime_targets
        )
        
        # Add hard penalty for probability mass on underprediction classes
        probs = F.softmax(threshold_logits, dim=1)
        penalties = self.penalty_matrix[threshold_targets]  # (batch_size, num_classes)
        hard_penalty = (probs * penalties).sum(dim=1).mean()
        
        result["loss"] = result["loss"] + self.hard_penalty_weight * hard_penalty
        result["hard_penalty"] = hard_penalty
        
        return result


class SafeOverpredictionLoss(nn.Module):
    """
    Loss that heavily biases toward overprediction by treating "1 step over" as nearly optimal.
    
    Key insight: underprediction gives score=0, but overprediction by 1 step gives score=0.5.
    So we should train the model to favor overpredicting when uncertain.
    
    This loss:
    1. Uses soft targets where class+1 gets significant weight (not just decay)
    2. Has configurable underprediction penalty multiplier
    3. Optionally shifts the target up by 1 class entirely
    """
    
    def __init__(
        self,
        num_classes: int = 9,
        underprediction_penalty: float = 10.0,
        overprediction_bonus: float = 0.8,
        shift_target_up: bool = False,
    ):
        """
        Args:
            num_classes: Number of threshold classes
            underprediction_penalty: Multiplier for probability mass on classes below true
            overprediction_bonus: Weight for class+1 relative to true class (0.8 = almost as good)
            shift_target_up: If True, train to predict class+1 instead of true class
        """
        super().__init__()
        self.num_classes = num_classes
        self.underprediction_penalty = underprediction_penalty
        self.shift_target_up = shift_target_up
        
        soft_targets = torch.zeros(num_classes, num_classes)
        for t in range(num_classes):
            if shift_target_up and t < num_classes - 1:
                target = t + 1
            else:
                target = t
            
            soft_targets[t, target] = 1.0
            if target < num_classes - 1:
                soft_targets[t, target + 1] = overprediction_bonus
            
            soft_targets[t] = soft_targets[t] / soft_targets[t].sum()
        self.register_buffer("soft_targets", soft_targets)
        
        underpred_mask = torch.zeros(num_classes, num_classes)
        for t in range(num_classes):
            for i in range(t):
                underpred_mask[t, i] = 1.0
        self.register_buffer("underpred_mask", underpred_mask)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        soft_tgt = self.soft_targets[targets]
        ce_loss = -(soft_tgt * log_probs).sum(dim=1)
        
        underpred_mask = self.underpred_mask[targets]
        underpred_prob = (probs * underpred_mask).sum(dim=1)
        underpred_penalty = self.underprediction_penalty * underpred_prob
        
        return (ce_loss + underpred_penalty).mean()


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


class CrossEntropyMSELog2Loss(nn.Module):
    """
    Combined loss using CrossEntropy for threshold and MSE in log2 space for runtime.
    
    This combines the best of both worlds:
    - CrossEntropy learns sharp decision boundaries for discrete threshold classes
    - MSE in log2 space is well-aligned with runtime scoring (score = 2^(-|log2 error|))
      and provides scaled gradients that prioritize fixing large errors
    
    Runtime scoring: score = min(pred/true, true/pred) = 2^(-|log2(pred) - log2(true)|)
    So MSE on log2(runtime) directly optimizes for minimizing squared log-ratio error.
    
    IMPORTANT: Uses gradient normalization to avoid unstable training from the
    log1p -> log2 conversion chain. Without normalization, gradients for small
    runtimes are 100x+ larger than for large runtimes.
    """
    
    def __init__(
        self,
        threshold_weight: float = 1.0,
        runtime_weight: float = 1.0,
        label_smoothing: float = 0.0,
        normalize_gradient: bool = True,
    ):
        super().__init__()
        self.threshold_weight = threshold_weight
        self.runtime_weight = runtime_weight
        self.normalize_gradient = normalize_gradient
        self.threshold_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(
        self,
        threshold_logits: torch.Tensor,
        runtime_pred: torch.Tensor,
        threshold_targets: torch.Tensor,
        runtime_targets: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            threshold_logits: Predicted threshold logits (batch_size, num_classes)
            runtime_pred: Predicted log1p(runtime) from the MLP (batch_size,)
            threshold_targets: True threshold class indices (batch_size,)
            runtime_targets: True log1p(runtime) (batch_size,)
            
        Returns:
            Dict with 'loss', 'threshold_loss', 'runtime_loss'
        """
        threshold_loss = self.threshold_criterion(threshold_logits, threshold_targets)
        
        pred_runtime = torch.expm1(runtime_pred).clamp(min=1e-8)
        true_runtime = torch.expm1(runtime_targets).clamp(min=1e-8)
        
        log_ratio = torch.log2(pred_runtime / true_runtime)
        
        if self.normalize_gradient:
            # The gradient through log1p -> expm1 -> log2 has scale factor:
            # (1 + runtime) / (runtime * ln(2))
            # 
            # To normalize, we multiply the squared error by the inverse scale^2:
            # [runtime * ln(2) / (1 + runtime)]^2
            #
            # This makes gradients independent of runtime magnitude.
            # We use pred_runtime (detached) as the reference point.
            with torch.no_grad():
                inv_scale = pred_runtime * 0.6931 / (1 + pred_runtime)  # ln(2) ≈ 0.6931
                weight = inv_scale ** 2
                weight = weight / weight.mean()  # Normalize to keep overall loss scale
            
            runtime_loss = (weight * log_ratio ** 2).mean()
        else:
            runtime_loss = (log_ratio ** 2).mean()
        
        combined_loss = (
            self.threshold_weight * threshold_loss +
            self.runtime_weight * runtime_loss
        )
        
        return {
            "loss": combined_loss,
            "threshold_loss": threshold_loss,
            "runtime_loss": runtime_loss,
        }

"""
Scoring utilities matching the official challenge scoring.

Updated for new challenge format with separate tasks:
- Task 1: Threshold prediction for target fidelity 0.75
- Task 2: Runtime prediction given a threshold
"""

from typing import Dict
import numpy as np

from data_loader import THRESHOLD_LADDER


def compute_threshold_score(
    pred_threshold: np.ndarray,
    true_threshold: np.ndarray,
) -> Dict[str, float]:
    """
    Compute threshold prediction score for Task 1.

    Task 1: Given QASM, target fidelity (0.75), GPU/CPU, single/double precision,
    predict the threshold.

    Scoring:
    - If pred_threshold < true_threshold: score = 0 (fidelity violated)
    - Otherwise: score = 2^(-steps_over) where steps_over = pred_idx - true_idx

    Args:
        pred_threshold: Predicted threshold values (from ladder: 1, 2, 4, ..., 256)
        true_threshold: Ground truth threshold values for target fidelity 0.75

    Returns:
        Dictionary with threshold_score and per-sample scores
    """
    threshold_scores = []

    for pred_thr, true_thr in zip(pred_threshold, true_threshold):
        pred_idx = THRESHOLD_LADDER.index(pred_thr) if pred_thr in THRESHOLD_LADDER else -1
        true_idx = THRESHOLD_LADDER.index(true_thr) if true_thr in THRESHOLD_LADDER else -1

        if pred_idx < true_idx:
            # Fidelity violated - score is 0
            threshold_scores.append(0.0)
            continue

        steps_over = pred_idx - true_idx
        thr_score = 2.0 ** (-steps_over)
        threshold_scores.append(thr_score)

    return {
        "threshold_score": float(np.mean(threshold_scores)),
        "threshold_scores_per_sample": threshold_scores,
    }


def compute_runtime_score(
    pred_runtime: np.ndarray,
    true_runtime: np.ndarray,
) -> Dict[str, float]:
    """
    Compute runtime prediction score for Task 2.

    Task 2: Given QASM, threshold, GPU/CPU, single/double precision,
    predict the runtime duration.

    Scoring:
    - runtime_score = min(r, 1/r) where r = pred_time / true_time
    - Penalizes both over and under-prediction

    Args:
        pred_runtime: Predicted forward wall time in seconds
        true_runtime: Ground truth forward wall time in seconds

    Returns:
        Dictionary with runtime_score and per-sample scores
    """
    runtime_scores = []

    for pred_t, true_t in zip(pred_runtime, true_runtime):
        r = pred_t / true_t if true_t > 0 else 0.0
        rt_score = min(r, 1.0 / r) if r > 0 else 0.0
        runtime_scores.append(rt_score)

    return {
        "runtime_score": float(np.mean(runtime_scores)),
        "runtime_scores_per_sample": runtime_scores,
    }


def compute_challenge_score(
    pred_threshold: np.ndarray,
    true_threshold: np.ndarray,
    pred_runtime: np.ndarray,
    true_runtime: np.ndarray,
) -> Dict[str, float]:
    """
    Compute combined scoring metrics (for backwards compatibility during training).

    This computes both threshold and runtime scores separately and also provides
    a combined score (product of threshold_score * runtime_score per task).

    Args:
        pred_threshold: Predicted threshold values (from ladder: 1, 2, 4, ..., 256)
        true_threshold: Ground truth threshold values
        pred_runtime: Predicted forward wall time in seconds
        true_runtime: Ground truth forward wall time in seconds

    Returns:
        Dictionary with threshold_score, runtime_score, and combined_score
    """
    threshold_result = compute_threshold_score(pred_threshold, true_threshold)
    runtime_result = compute_runtime_score(pred_runtime, true_runtime)

    # Compute combined scores per sample (product of threshold and runtime scores)
    combined_scores = []
    for thr_score, rt_score in zip(
        threshold_result["threshold_scores_per_sample"],
        runtime_result["runtime_scores_per_sample"]
    ):
        combined_scores.append(thr_score * rt_score)

    return {
        "threshold_score": threshold_result["threshold_score"],
        "runtime_score": runtime_result["runtime_score"],
        "combined_score": float(np.mean(combined_scores)),
    }

"""
Scoring utilities matching the official challenge scoring.
"""

from typing import Dict
import numpy as np

from data_loader import THRESHOLD_LADDER


def compute_challenge_score(
    pred_threshold: np.ndarray,
    true_threshold: np.ndarray,
    pred_runtime: np.ndarray,
    true_runtime: np.ndarray,
) -> Dict[str, float]:
    """
    Compute scoring metrics matching the official challenge scoring.
    
    Official scoring (Model 1A):
    - If pred_threshold < true_threshold: task_score = 0 (fidelity violated)
    - Otherwise:
        threshold_score = 2^(-steps_over)
        runtime_score = min(r, 1/r) where r = pred_time / true_time
        task_score = threshold_score * runtime_score
    
    Overall score = mean(task_scores)
    
    Args:
        pred_threshold: Predicted threshold values (from ladder: 1, 2, 4, ..., 256)
        true_threshold: Ground truth threshold values
        pred_runtime: Predicted forward wall time in seconds
        true_runtime: Ground truth forward wall time in seconds
        
    Returns:
        Dictionary with threshold_score, runtime_score, and combined_score
    """
    task_scores = []
    threshold_scores = []
    runtime_scores = []
    
    for pred_thr, true_thr, pred_t, true_t in zip(
        pred_threshold, true_threshold, pred_runtime, true_runtime
    ):
        pred_idx = THRESHOLD_LADDER.index(pred_thr) if pred_thr in THRESHOLD_LADDER else -1
        true_idx = THRESHOLD_LADDER.index(true_thr) if true_thr in THRESHOLD_LADDER else -1
        
        if pred_idx < true_idx:
            threshold_scores.append(0.0)
            runtime_scores.append(0.0)
            task_scores.append(0.0)
            continue
        
        steps_over = pred_idx - true_idx
        thr_score = 2.0 ** (-steps_over)
        threshold_scores.append(thr_score)
        
        r = pred_t / true_t if true_t > 0 else 0.0
        rt_score = min(r, 1.0 / r) if r > 0 else 0.0
        runtime_scores.append(rt_score)
        
        task_scores.append(thr_score * rt_score)
    
    return {
        "threshold_score": float(np.mean(threshold_scores)),
        "runtime_score": float(np.mean(runtime_scores)),
        "combined_score": float(np.mean(task_scores)),
    }

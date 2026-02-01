# """
# Scoring utilities matching the official challenge scoring.
# """

# from typing import Dict
# import numpy as np

# from data_loader import THRESHOLD_LADDER


# def compute_challenge_score(
#     pred_threshold: np.ndarray,
#     true_threshold: np.ndarray,
#     pred_runtime: np.ndarray,
#     true_runtime: np.ndarray,
#     validate=False
# ) -> Dict[str, float]:
#     """
#     Compute scoring metrics matching the official challenge scoring.
    
#     Official scoring (Model 1A):
#     - If pred_threshold < true_threshold: task_score = 0 (fidelity violated)
#     - Otherwise:
#         threshold_score = 2^(-steps_over)
#         runtime_score = min(r, 1/r) where r = pred_time / true_time
#         task_score = threshold_score * runtime_score
    
#     Overall score = mean(task_scores)
    
#     Args:
#         pred_threshold: Predicted threshold values (from ladder: 1, 2, 4, ..., 256)
#         true_threshold: Ground truth threshold values
#         pred_runtime: Predicted forward wall time in seconds
#         true_runtime: Ground truth forward wall time in seconds
        
#     Returns:
#         Dictionary with threshold_score, runtime_score, and combined_score
#     """

#     records = []

#     task_scores = []
#     threshold_scores = []
#     runtime_scores = []
    
#     for pred_thr, true_thr, pred_t, true_t in zip(
#         pred_threshold, true_threshold, pred_runtime, true_runtime
#     ):
#         pred_idx = THRESHOLD_LADDER.index(pred_thr) if pred_thr in THRESHOLD_LADDER else -1
#         true_idx = THRESHOLD_LADDER.index(true_thr) if true_thr in THRESHOLD_LADDER else -1
        
#         error_type = "OK"

#         if pred_idx < true_idx:
#             threshold_scores.append(0.0)
#             runtime_scores.append(0.0)
#             task_scores.append(0.0)
#             error_type = "FATAL"
        
#         steps_over = pred_idx - true_idx
#         thr_score = 2.0 ** (-steps_over)
#         threshold_scores.append(thr_score)
        
#         r = pred_t / true_t if true_t > 0 else 0.0
#         rt_score = min(r, 1.0 / r) if r > 0 else 0.0
#         runtime_scores.append(rt_score)
        
#         task_scores.append(thr_score * rt_score)
#         error_type =  str(steps_over) + " Steps Over"


#         # --- 2. Save CSV ---
#         import pandas as pd
#         import os
#         if(validate):
#             records.append({
#             "score": thr_score * rt_score,
#             "time_score": rt_score,
#             "pred_thresh": pred_idx,
#             "true_thresh": true_idx,
#             "pred_time": pred_t,
#             "true_time": true_t,
#             "error_type": error_type,
#             })


#     df = pd.DataFrame(records)
#     report_file = "results.csv"
#     file_exists = os.path.isfile("results.csv")


#     df.to_csv(report_file, index=False, mode='a',header=not file_exists)
#     #print(f"\n[Scoring] Detailed results saved to {report_file}")
    
#     return {
#         "threshold_score": float(np.mean(threshold_scores)),
#         "runtime_score": float(np.mean(runtime_scores)),
#         "combined_score": float(np.mean(task_scores)),
#     }











"""
scoring.py: Official scoring logic + CSV export for visualization.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os


# The official ladder
THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def compute_challenge_score(
pred_threshold: np.ndarray,
true_threshold: np.ndarray,
pred_runtime: np.ndarray,
true_runtime: np.ndarray,
task_ids: Optional[List[str]] = None,
report_file: str = "results.csv", # <--- output filename
validate: bool = False
) -> Dict[str, float]:


    # If no IDs provided, make dummy ones
    if task_ids is None:
        task_ids = [f"task_{i}" for i in range(len(pred_threshold))]


    records = []

    # --- 1. Iterate and Score ---
    for i, (p_thr, t_thr, p_time, t_time, uid) in enumerate(zip(
        pred_threshold, true_threshold, pred_runtime, true_runtime, task_ids
    )):
        # Snap to ladder index
        try: p_idx = THRESHOLD_LADDER.index(p_thr)
        except:
            p_idx = -1
            print(p_thr)
        
        try: t_idx = THRESHOLD_LADDER.index(t_thr)
        except: 
            t_idx = 999
            print(t_thr)




        # Logic
        error_type = "OK"
        steps_diff = 0

        
        if p_idx < t_idx:
            # FATAL
            score = 0.0
            error_type = "FATAL"
            steps_diff = p_idx - t_idx
        else:
            # SAFE or PERFECT
            steps_diff = p_idx - t_idx
            if steps_diff == 0: error_type = "PERFECT"
            else: error_type = f"SAFE (+{steps_diff})"

        # Sub-scores
        if(error_type == "FATAL"):
            s_thresh = 0.0
        else:
            s_thresh = 2.0 ** (-steps_diff)
        r = p_time / t_time if t_time > 0 else 0
        s_time = min(r, 1.0/r) if r > 0 else 0
        score = s_thresh * s_time


        records.append({
            "task_id": uid,
            "score": score,
            "time_score": s_time,
            "error_type": error_type,
            "pred_thresh": p_thr,
            "true_thresh": t_thr,
            "pred_time": p_time,
            "true_time": t_time
        })


    # --- 2. Save CSV ---
    df = pd.DataFrame(records)
    file_exists = os.path.isfile("results.csv")


    df.to_csv(report_file, index=False, mode='a',header=not file_exists)
    #print(f"\n[Scoring] Detailed results saved to {report_file}")

    return {
        "combined_score": df["score"].mean()
    }

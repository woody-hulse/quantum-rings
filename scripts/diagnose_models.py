#!/usr/bin/env python3
"""Diagnose model issues and identify improvement opportunities."""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    create_kfold_data_loaders,
    create_kfold_threshold_class_data_loaders,
    THRESHOLD_LADDER,
)
from scoring import (
    compute_challenge_score,
    mean_threshold_score,
    select_threshold_class_by_expected_score,
    get_threshold_score_matrix,
)

project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "hackathon_public.json"
circuits_dir = project_root / "circuits"

def extract_duration_data(loader):
    X, y, thresh = [], [], []
    for batch in loader:
        X.append(batch["features"].numpy())
        y.extend(batch["log2_runtime"].tolist())
        thresh.extend(batch["threshold"])
    return np.vstack(X), np.array(y), np.array(thresh)

def extract_tc_data(loader):
    X, y = [], []
    for batch in loader:
        X.append(batch["features"].numpy())
        y.extend(batch["threshold_class"].tolist())
    return np.vstack(X), np.array(y)

print("="*80)
print("DURATION MODEL DIAGNOSIS")
print("="*80)

from models.xgboost_model import XGBoostModel

all_mae = []
all_scores = []
all_residuals = []

fold_loaders = create_kfold_data_loaders(data_path, circuits_dir, n_folds=5, batch_size=32)

for fold, (train_loader, val_loader) in enumerate(fold_loaders):
    model = XGBoostModel(max_depth=6, learning_rate=0.1, n_estimators=50)
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    
    X_val, y_val, thresh_val = extract_duration_data(val_loader)
    pred_thresh, pred_runtime = model.predict(X_val)
    true_runtime = np.power(2.0, y_val)
    log2_pred = np.log2(np.maximum(pred_runtime, 1e-10))
    
    mae = np.mean(np.abs(log2_pred - y_val))
    residuals = log2_pred - y_val
    all_residuals.extend(residuals.tolist())
    
    scores = compute_challenge_score(pred_thresh, thresh_val, pred_runtime, true_runtime)
    all_mae.append(mae)
    all_scores.append(scores['combined_score'])
    
    print(f"Fold {fold+1}: MAE={mae:.4f}, Score={scores['combined_score']:.4f}")

print(f"\nOverall MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
print(f"Overall Score: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")

residuals = np.array(all_residuals)
print(f"\nResidual Analysis:")
print(f"  Mean: {np.mean(residuals):.4f}")
print(f"  Std: {np.std(residuals):.4f}")
print(f"  Median: {np.median(residuals):.4f}")
print(f"  P10/P90: {np.percentile(residuals, 10):.4f} / {np.percentile(residuals, 90):.4f}")

print("\n" + "="*80)
print("THRESHOLD-CLASS MODEL DIAGNOSIS")
print("="*80)

from models.xgboost_threshold_class import XGBoostThresholdClassModel

all_scores = []
all_underpred = []
all_proba = []
all_true = []
confusion_matrix = np.zeros((len(THRESHOLD_LADDER), len(THRESHOLD_LADDER)), dtype=int)

fold_loaders = create_kfold_threshold_class_data_loaders(data_path, circuits_dir, n_folds=5, batch_size=32)

for fold, (train_loader, val_loader) in enumerate(fold_loaders):
    model = XGBoostThresholdClassModel(max_depth=6, learning_rate=0.1, n_estimators=50)
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    
    X_val, y_val = extract_tc_data(val_loader)
    proba = model.predict_proba(X_val)
    chosen = select_threshold_class_by_expected_score(proba)
    
    threshold_score = mean_threshold_score(chosen, y_val)
    underpred = np.mean(chosen < y_val)
    
    all_scores.append(threshold_score)
    all_underpred.append(underpred)
    all_proba.extend(proba.tolist())
    all_true.extend(y_val.tolist())
    
    for c, t in zip(chosen, y_val):
        confusion_matrix[c, t] += 1
    
    print(f"Fold {fold+1}: Score={threshold_score:.4f}, Underpred={underpred:.2%}")

print(f"\nOverall Score: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
print(f"Overall Underpred: {np.mean(all_underpred):.2%} ± {np.std(all_underpred):.2%}")

print("\nConfusion Matrix (Pred vs True class):")
print("     ", end="")
for t in range(len(THRESHOLD_LADDER)):
    print(f"  T{t}", end="")
print()
for p in range(len(THRESHOLD_LADDER)):
    print(f"P{p}  ", end="")
    for t in range(len(THRESHOLD_LADDER)):
        print(f"{confusion_matrix[p, t]:4d}", end="")
    print()

print("\nClass-wise analysis (true class):")
true_arr = np.array(all_true)
proba_arr = np.array(all_proba)
for c in range(len(THRESHOLD_LADDER)):
    mask = true_arr == c
    if mask.sum() == 0:
        continue
    class_proba = proba_arr[mask]
    mean_prob_at_class = class_proba[:, c].mean()
    mean_max_prob = class_proba.max(axis=1).mean()
    print(f"  Class {c} (thresh={THRESHOLD_LADDER[c]:3d}): n={mask.sum():3d}, "
          f"mean P(correct)={mean_prob_at_class:.3f}, mean max_P={mean_max_prob:.3f}")

print("\n" + "="*80)
print("ISSUE ANALYSIS & RECOMMENDATIONS")
print("="*80)

print("\n1. DURATION MODEL ISSUES:")
print("   - Check feature importance to identify key predictors")
print("   - Analyze residuals by circuit type to find systematic errors")

print("\n2. THRESHOLD-CLASS MODEL ISSUES:")
print("   - High underprediction rate causes 0 score (critical)")
print("   - Class imbalance may affect model training")
print("   - Consider asymmetric loss to penalize underprediction more")

print("\n3. POTENTIAL IMPROVEMENTS:")
print("   - Use ordinal regression instead of classification")
print("   - Add conservative bias to threshold predictions")
print("   - Ensemble methods with bias towards higher thresholds")
print("   - Use calibrated probabilities for decision-making")

print("\nDone!")

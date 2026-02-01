#!/usr/bin/env python3
"""Evaluate improvements: baseline vs improved models."""

import sys
from pathlib import Path
import numpy as np

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
print("THRESHOLD-CLASS MODEL COMPARISON")
print("="*80)

from models.xgboost_threshold_class import XGBoostThresholdClassModel

fold_loaders = create_kfold_threshold_class_data_loaders(data_path, circuits_dir, n_folds=5, batch_size=32)

print("\n--- Baseline XGBoost (no class weights, no bias) ---")
all_scores_base, all_underpred_base = [], []
for fold, (train_loader, val_loader) in enumerate(fold_loaders):
    model = XGBoostThresholdClassModel(
        max_depth=6, learning_rate=0.1, n_estimators=50,
        use_class_weights=False, conservative_bias=0.0
    )
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    X_val, y_val = extract_tc_data(val_loader)
    proba = model.predict_proba(X_val)
    chosen = select_threshold_class_by_expected_score(proba, conservative_bias=0.0)
    threshold_score = mean_threshold_score(chosen, y_val)
    underpred = np.mean(chosen < y_val)
    all_scores_base.append(threshold_score)
    all_underpred_base.append(underpred)
    print(f"  Fold {fold+1}: Score={threshold_score:.4f}, Underpred={underpred:.2%}")
print(f"  Overall: Score={np.mean(all_scores_base):.4f}±{np.std(all_scores_base):.3f}, Underpred={np.mean(all_underpred_base):.2%}")

print("\n--- XGBoost with class weights ---")
all_scores_cw, all_underpred_cw = [], []
for fold, (train_loader, val_loader) in enumerate(fold_loaders):
    model = XGBoostThresholdClassModel(
        max_depth=6, learning_rate=0.1, n_estimators=50,
        use_class_weights=True, conservative_bias=0.0
    )
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    X_val, y_val = extract_tc_data(val_loader)
    proba = model.predict_proba(X_val)
    chosen = select_threshold_class_by_expected_score(proba, conservative_bias=0.0)
    threshold_score = mean_threshold_score(chosen, y_val)
    underpred = np.mean(chosen < y_val)
    all_scores_cw.append(threshold_score)
    all_underpred_cw.append(underpred)
    print(f"  Fold {fold+1}: Score={threshold_score:.4f}, Underpred={underpred:.2%}")
print(f"  Overall: Score={np.mean(all_scores_cw):.4f}±{np.std(all_scores_cw):.3f}, Underpred={np.mean(all_underpred_cw):.2%}")

print("\n--- XGBoost with class weights + conservative bias 0.1 ---")
all_scores_cb1, all_underpred_cb1 = [], []
for fold, (train_loader, val_loader) in enumerate(fold_loaders):
    model = XGBoostThresholdClassModel(
        max_depth=6, learning_rate=0.1, n_estimators=50,
        use_class_weights=True, conservative_bias=0.1
    )
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    X_val, y_val = extract_tc_data(val_loader)
    proba = model.predict_proba(X_val)
    chosen = select_threshold_class_by_expected_score(proba, conservative_bias=0.1)
    threshold_score = mean_threshold_score(chosen, y_val)
    underpred = np.mean(chosen < y_val)
    all_scores_cb1.append(threshold_score)
    all_underpred_cb1.append(underpred)
    print(f"  Fold {fold+1}: Score={threshold_score:.4f}, Underpred={underpred:.2%}")
print(f"  Overall: Score={np.mean(all_scores_cb1):.4f}±{np.std(all_scores_cb1):.3f}, Underpred={np.mean(all_underpred_cb1):.2%}")

print("\n--- XGBoost with class weights + conservative bias 0.2 ---")
all_scores_cb2, all_underpred_cb2 = [], []
for fold, (train_loader, val_loader) in enumerate(fold_loaders):
    model = XGBoostThresholdClassModel(
        max_depth=6, learning_rate=0.1, n_estimators=50,
        use_class_weights=True, conservative_bias=0.2
    )
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    X_val, y_val = extract_tc_data(val_loader)
    proba = model.predict_proba(X_val)
    chosen = select_threshold_class_by_expected_score(proba, conservative_bias=0.2)
    threshold_score = mean_threshold_score(chosen, y_val)
    underpred = np.mean(chosen < y_val)
    all_scores_cb2.append(threshold_score)
    all_underpred_cb2.append(underpred)
    print(f"  Fold {fold+1}: Score={threshold_score:.4f}, Underpred={underpred:.2%}")
print(f"  Overall: Score={np.mean(all_scores_cb2):.4f}±{np.std(all_scores_cb2):.3f}, Underpred={np.mean(all_underpred_cb2):.2%}")

print("\n" + "="*80)
print("DURATION MODEL COMPARISON")  
print("="*80)

fold_loaders_dur = create_kfold_data_loaders(data_path, circuits_dir, n_folds=5, batch_size=32)

from models.xgboost_model import XGBoostModel
from models.mlp import MLPModel

print("\n--- XGBoost Duration (baseline) ---")
all_mae_xgb, all_scores_xgb = [], []
for fold, (train_loader, val_loader) in enumerate(fold_loaders_dur):
    model = XGBoostModel(max_depth=6, learning_rate=0.1, n_estimators=50)
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    X_val, y_val, thresh_val = extract_duration_data(val_loader)
    pred_thresh, pred_runtime = model.predict(X_val)
    true_runtime = np.power(2.0, y_val)
    log2_pred = np.log2(np.maximum(pred_runtime, 1e-10))
    mae = np.mean(np.abs(log2_pred - y_val))
    scores = compute_challenge_score(pred_thresh, thresh_val, pred_runtime, true_runtime)
    all_mae_xgb.append(mae)
    all_scores_xgb.append(scores['combined_score'])
    print(f"  Fold {fold+1}: MAE={mae:.4f}, Score={scores['combined_score']:.4f}")
print(f"  Overall: MAE={np.mean(all_mae_xgb):.4f}±{np.std(all_mae_xgb):.3f}, Score={np.mean(all_scores_xgb):.4f}")

print("\n--- MLP Duration (improved) ---")
all_mae_mlp, all_scores_mlp = [], []
for fold, (train_loader, val_loader) in enumerate(fold_loaders_dur):
    first_batch = next(iter(train_loader))
    input_dim = first_batch["features"].shape[1]
    model = MLPModel(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        epochs=50,
        early_stopping_patience=15,
        use_huber_loss=True,
        weight_decay=1e-4,
    )
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    X_val, y_val, thresh_val = extract_duration_data(val_loader)
    pred_thresh, pred_runtime = model.predict(X_val)
    true_runtime = np.power(2.0, y_val)
    log2_pred = np.log2(np.maximum(pred_runtime, 1e-10))
    mae = np.mean(np.abs(log2_pred - y_val))
    scores = compute_challenge_score(pred_thresh, thresh_val, pred_runtime, true_runtime)
    all_mae_mlp.append(mae)
    all_scores_mlp.append(scores['combined_score'])
    print(f"  Fold {fold+1}: MAE={mae:.4f}, Score={scores['combined_score']:.4f}")
print(f"  Overall: MAE={np.mean(all_mae_mlp):.4f}±{np.std(all_mae_mlp):.3f}, Score={np.mean(all_scores_mlp):.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nThreshold-Class Models:")
print(f"  Baseline:                    Score={np.mean(all_scores_base):.4f}, Underpred={np.mean(all_underpred_base):.1%}")
print(f"  + Class weights:             Score={np.mean(all_scores_cw):.4f}, Underpred={np.mean(all_underpred_cw):.1%}")
print(f"  + Class weights + bias 0.1:  Score={np.mean(all_scores_cb1):.4f}, Underpred={np.mean(all_underpred_cb1):.1%}")
print(f"  + Class weights + bias 0.2:  Score={np.mean(all_scores_cb2):.4f}, Underpred={np.mean(all_underpred_cb2):.1%}")

print("\nDuration Models:")
print(f"  XGBoost: MAE={np.mean(all_mae_xgb):.4f}, Score={np.mean(all_scores_xgb):.4f}")
print(f"  MLP:     MAE={np.mean(all_mae_mlp):.4f}, Score={np.mean(all_scores_mlp):.4f}")

print("\nDone!")

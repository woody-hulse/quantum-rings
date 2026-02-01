#!/usr/bin/env python3
"""Quick evaluation script - single fold, fewer iterations."""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    create_data_loaders,
    create_threshold_class_data_loaders,
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

print("Loading data...")
train_loader, val_loader = create_data_loaders(data_path, circuits_dir, batch_size=64, val_fraction=0.2)
print(f"Duration: Train batches={len(train_loader)}, Val batches={len(val_loader)}")

train_loader_tc, val_loader_tc = create_threshold_class_data_loaders(data_path, circuits_dir, batch_size=64, val_fraction=0.2)
print(f"Threshold-class: Train batches={len(train_loader_tc)}, Val batches={len(val_loader_tc)}")

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

print("\n=== Duration Models ===")

try:
    from models.xgboost_model import XGBoostModel
    print("\nXGBoost Duration:")
    model = XGBoostModel(max_depth=6, learning_rate=0.1, n_estimators=30)
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    X_val, y_val, thresh_val = extract_duration_data(val_loader)
    pred_thresh, pred_runtime = model.predict(X_val)
    true_runtime = np.power(2.0, y_val)
    log2_pred = np.log2(np.maximum(pred_runtime, 1e-10))
    mae = np.mean(np.abs(log2_pred - y_val))
    scores = compute_challenge_score(pred_thresh, thresh_val, pred_runtime, true_runtime)
    print(f"  MAE (log2): {mae:.4f}")
    print(f"  Combined score: {scores['combined_score']:.4f}")
    print(f"  Runtime score: {scores['runtime_score']:.4f}")
except Exception as e:
    print(f"XGBoost error: {e}")

try:
    from models.catboost_model import CatBoostModel
    print("\nCatBoost Duration:")
    model = CatBoostModel(depth=6, learning_rate=0.1, iterations=30)
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    X_val, y_val, thresh_val = extract_duration_data(val_loader)
    pred_thresh, pred_runtime = model.predict(X_val)
    true_runtime = np.power(2.0, y_val)
    log2_pred = np.log2(np.maximum(pred_runtime, 1e-10))
    mae = np.mean(np.abs(log2_pred - y_val))
    scores = compute_challenge_score(pred_thresh, thresh_val, pred_runtime, true_runtime)
    print(f"  MAE (log2): {mae:.4f}")
    print(f"  Combined score: {scores['combined_score']:.4f}")
    print(f"  Runtime score: {scores['runtime_score']:.4f}")
except Exception as e:
    print(f"CatBoost error: {e}")

try:
    from models.mlp import MLPModel
    print("\nMLP Duration:")
    first_batch = next(iter(train_loader))
    input_dim = first_batch["features"].shape[1]
    model = MLPModel(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.2, epochs=30, early_stopping_patience=10)
    model.fit(train_loader, val_loader, verbose=False, show_progress=False)
    X_val, y_val, thresh_val = extract_duration_data(val_loader)
    pred_thresh, pred_runtime = model.predict(X_val)
    true_runtime = np.power(2.0, y_val)
    log2_pred = np.log2(np.maximum(pred_runtime, 1e-10))
    mae = np.mean(np.abs(log2_pred - y_val))
    scores = compute_challenge_score(pred_thresh, thresh_val, pred_runtime, true_runtime)
    print(f"  MAE (log2): {mae:.4f}")
    print(f"  Combined score: {scores['combined_score']:.4f}")
    print(f"  Runtime score: {scores['runtime_score']:.4f}")
except Exception as e:
    print(f"MLP error: {e}")

print("\n=== Threshold-Class Models ===")

try:
    from models.xgboost_threshold_class import XGBoostThresholdClassModel
    print("\nXGBoost Threshold-Class:")
    model = XGBoostThresholdClassModel(max_depth=6, learning_rate=0.1, n_estimators=30)
    model.fit(train_loader_tc, val_loader_tc, verbose=False, show_progress=False)
    X_val, y_val = extract_tc_data(val_loader_tc)
    proba = model.predict_proba(X_val)
    chosen = select_threshold_class_by_expected_score(proba)
    accuracy = np.mean(chosen == y_val)
    threshold_score = mean_threshold_score(chosen, y_val)
    underpred_rate = np.mean(chosen < y_val)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Threshold score: {threshold_score:.4f}")
    print(f"  Underpred rate: {underpred_rate:.2%}")
except Exception as e:
    print(f"XGBoost TC error: {e}")

try:
    from models.catboost_threshold_class import CatBoostThresholdClassModel
    print("\nCatBoost Threshold-Class:")
    model = CatBoostThresholdClassModel(depth=6, learning_rate=0.1, iterations=30)
    model.fit(train_loader_tc, val_loader_tc, verbose=False, show_progress=False)
    X_val, y_val = extract_tc_data(val_loader_tc)
    proba = model.predict_proba(X_val)
    chosen = select_threshold_class_by_expected_score(proba)
    accuracy = np.mean(chosen == y_val)
    threshold_score = mean_threshold_score(chosen, y_val)
    underpred_rate = np.mean(chosen < y_val)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Threshold score: {threshold_score:.4f}")
    print(f"  Underpred rate: {underpred_rate:.2%}")
except Exception as e:
    print(f"CatBoost TC error: {e}")

try:
    from models.mlp_threshold_class import MLPThresholdClassModel
    print("\nMLP Threshold-Class:")
    first_batch = next(iter(train_loader_tc))
    input_dim = first_batch["features"].shape[1]
    model = MLPThresholdClassModel(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.2, epochs=30, early_stopping_patience=10)
    model.fit(train_loader_tc, val_loader_tc, verbose=False, show_progress=False)
    X_val, y_val = extract_tc_data(val_loader_tc)
    proba = model.predict_proba(X_val)
    chosen = select_threshold_class_by_expected_score(proba)
    accuracy = np.mean(chosen == y_val)
    threshold_score = mean_threshold_score(chosen, y_val)
    underpred_rate = np.mean(chosen < y_val)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Threshold score: {threshold_score:.4f}")
    print(f"  Underpred rate: {underpred_rate:.2%}")
except Exception as e:
    print(f"MLP TC error: {e}")

print("\n=== Analysis of Threshold Class Distribution ===")
X_train, y_train = extract_tc_data(train_loader_tc)
X_val, y_val = extract_tc_data(val_loader_tc)
print("\nThreshold class distribution (train):")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u} (thresh={THRESHOLD_LADDER[u]}): {c} ({c/len(y_train):.1%})")

print("\nThreshold class distribution (val):")
unique, counts = np.unique(y_val, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u} (thresh={THRESHOLD_LADDER[u]}): {c} ({c/len(y_val):.1%})")

print("\nDone!")

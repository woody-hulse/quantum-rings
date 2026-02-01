#!/usr/bin/env python3
"""
Plot correlation heatmap for ALL features in data_loader.

This script creates a comprehensive correlation heatmap showing:
1. All numeric features from NUMERIC_FEATURE_KEYS
2. Backend and precision encodings
3. Threshold class and log runtime targets
"""

import sys
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src folder to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / '..' / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import (
    load_hackathon_data,
    extract_qasm_features,
    compute_min_threshold,
    THRESHOLD_LADDER,
    QuantumCircuitDataset,
    BACKEND_MAP,
    PRECISION_MAP,
)


def load_all_data(data_path: Path, circuits_dir: Path):
    """Load all data with features."""
    circuits, results = load_hackathon_data(data_path)
    circuit_info = {c.file: c for c in circuits}
    
    features_cache = {}
    print("Extracting features from circuits...")
    for i, c in enumerate(circuits):
        qasm_path = circuits_dir / c.file
        if qasm_path.exists():
            features_cache[c.file] = extract_qasm_features(qasm_path)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(circuits)} circuits")
    
    ok_results = [r for r in results if r.status == "ok"]
    return circuits, ok_results, circuit_info, features_cache


def build_feature_matrix(results, circuit_info, features_cache):
    """Build feature matrix with ALL features."""
    
    # All feature names
    all_feature_names = list(QuantumCircuitDataset.NUMERIC_FEATURE_KEYS) + [
        'backend',
        'precision', 
        'threshold_class',
        'log_runtime'
    ]
    
    data_rows = []
    
    for r in results:
        info = circuit_info.get(r.file)
        features = features_cache.get(r.file, {})
        min_thresh = compute_min_threshold(r.threshold_sweep, target=0.99)
        
        if info and features and min_thresh and r.forward_wall_s:
            row = {}
            
            # Add all numeric features
            for key in QuantumCircuitDataset.NUMERIC_FEATURE_KEYS:
                row[key] = features.get(key, 0.0)
            
            # Add backend and precision as numeric
            row['backend'] = BACKEND_MAP.get(r.backend, 0)
            row['precision'] = PRECISION_MAP.get(r.precision, 0)
            
            # Add targets
            row['threshold_class'] = THRESHOLD_LADDER.index(min_thresh) if min_thresh in THRESHOLD_LADDER else -1
            row['log_runtime'] = np.log1p(r.forward_wall_s)
            
            data_rows.append(row)
    
    # Convert to matrix
    n_samples = len(data_rows)
    n_features = len(all_feature_names)
    
    matrix = np.zeros((n_samples, n_features))
    for i, row in enumerate(data_rows):
        for j, feat in enumerate(all_feature_names):
            matrix[i, j] = row.get(feat, 0.0)
    
    return matrix, all_feature_names


def plot_full_correlation_heatmap(matrix, feature_names, save_path, figsize=(24, 20)):
    """Plot correlation heatmap for all features."""
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(matrix.T)
    
    # Handle NaN values (features with zero variance)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Labels
    n_features = len(feature_names)
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    
    # Format labels (replace underscores, truncate)
    labels = [f.replace('_', '\n')[:20] for f in feature_names]
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    
    # Add correlation values (only if matrix is not too large)
    if n_features <= 40:
        for i in range(n_features):
            for j in range(n_features):
                val = corr_matrix[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=color, fontsize=5)
    
    # Highlight target rows/columns
    thresh_idx = feature_names.index('threshold_class')
    runtime_idx = feature_names.index('log_runtime')
    
    # Draw boxes around target rows
    for idx, color, name in [(thresh_idx, 'red', 'threshold'), (runtime_idx, 'blue', 'runtime')]:
        # Horizontal line above and below
        ax.axhline(y=idx - 0.5, color=color, linewidth=2, linestyle='--')
        ax.axhline(y=idx + 0.5, color=color, linewidth=2, linestyle='--')
        # Vertical lines
        ax.axvline(x=idx - 0.5, color=color, linewidth=2, linestyle='--')
        ax.axvline(x=idx + 0.5, color=color, linewidth=2, linestyle='--')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson Correlation', fontsize=12)
    
    # Title
    ax.set_title(f'Full Feature Correlation Heatmap ({n_features} features)\n'
                 f'Red box = threshold_class, Blue box = log_runtime', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    
    return corr_matrix


def plot_correlation_with_targets(matrix, feature_names, save_path):
    """Plot bar chart of correlations with threshold and runtime."""
    
    # Compute correlations
    corr_matrix = np.corrcoef(matrix.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    thresh_idx = feature_names.index('threshold_class')
    runtime_idx = feature_names.index('log_runtime')
    
    thresh_corrs = corr_matrix[thresh_idx, :]
    runtime_corrs = corr_matrix[runtime_idx, :]
    
    # Remove the targets themselves
    feature_mask = [i for i in range(len(feature_names)) 
                    if feature_names[i] not in ['threshold_class', 'log_runtime']]
    
    filtered_names = [feature_names[i] for i in feature_mask]
    filtered_thresh = [thresh_corrs[i] for i in feature_mask]
    filtered_runtime = [runtime_corrs[i] for i in feature_mask]
    
    # Sort by absolute threshold correlation
    sorted_idx = np.argsort(np.abs(filtered_thresh))[::-1]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    
    # Plot 1: Threshold correlations (sorted by threshold)
    ax = axes[0]
    n_features = len(filtered_names)
    y_pos = np.arange(n_features)
    
    colors = ['#2ecc71' if filtered_thresh[i] > 0 else '#e74c3c' for i in sorted_idx]
    ax.barh(y_pos, [filtered_thresh[i] for i in sorted_idx], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([filtered_names[i].replace('_', ' ')[:30] for i in sorted_idx], fontsize=8)
    ax.set_xlabel('Correlation with Threshold Class', fontsize=10)
    ax.set_title('All Features vs Threshold\n(Green=positive, Red=negative)', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=0.1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(x=-0.1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    
    # Plot 2: Runtime correlations (sorted by runtime)
    ax = axes[1]
    sorted_idx_runtime = np.argsort(np.abs(filtered_runtime))[::-1]
    
    colors = ['#3498db' if filtered_runtime[i] > 0 else '#9b59b6' for i in sorted_idx_runtime]
    ax.barh(y_pos, [filtered_runtime[i] for i in sorted_idx_runtime], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([filtered_names[i].replace('_', ' ')[:30] for i in sorted_idx_runtime], fontsize=8)
    ax.set_xlabel('Correlation with Log Runtime', fontsize=10)
    ax.set_title('All Features vs Runtime\n(Blue=positive, Purple=negative)', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=0.1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(x=-0.1, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.invert_yaxis()
    
    plt.suptitle(f'Feature Correlations with Targets ({n_features} features)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_correlation_summary(matrix, feature_names):
    """Print summary of correlations with targets."""
    
    corr_matrix = np.corrcoef(matrix.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    thresh_idx = feature_names.index('threshold_class')
    runtime_idx = feature_names.index('log_runtime')
    
    thresh_corrs = corr_matrix[thresh_idx, :]
    runtime_corrs = corr_matrix[runtime_idx, :]
    
    print("\n" + "=" * 80)
    print("CORRELATION SUMMARY")
    print("=" * 80)
    
    # Threshold correlations
    print("\n📊 TOP 15 THRESHOLD PREDICTORS (by |correlation|):")
    print("-" * 60)
    
    feature_corrs = [(feature_names[i], thresh_corrs[i]) 
                     for i in range(len(feature_names))
                     if feature_names[i] not in ['threshold_class', 'log_runtime']]
    
    sorted_corrs = sorted(feature_corrs, key=lambda x: abs(x[1]), reverse=True)
    
    for i, (name, corr) in enumerate(sorted_corrs[:15]):
        sign = "+" if corr > 0 else ""
        bar = "█" * int(abs(corr) * 20)
        print(f"  {i+1:2}. {name:35} {sign}{corr:+.3f} {bar}")
    
    # Runtime correlations
    print("\n📊 TOP 15 RUNTIME PREDICTORS (by |correlation|):")
    print("-" * 60)
    
    feature_corrs_rt = [(feature_names[i], runtime_corrs[i]) 
                        for i in range(len(feature_names))
                        if feature_names[i] not in ['threshold_class', 'log_runtime']]
    
    sorted_corrs_rt = sorted(feature_corrs_rt, key=lambda x: abs(x[1]), reverse=True)
    
    for i, (name, corr) in enumerate(sorted_corrs_rt[:15]):
        sign = "+" if corr > 0 else ""
        bar = "█" * int(abs(corr) * 20)
        print(f"  {i+1:2}. {name:35} {sign}{corr:+.3f} {bar}")
    
    # Redundant features
    print("\n🔄 HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.95):")
    print("-" * 60)
    
    n = len(feature_names)
    redundant_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr_matrix[i, j]) > 0.95:
                redundant_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for f1, f2, corr in redundant_pairs[:20]:
        print(f"  {f1:30} ↔ {f2:30} r={corr:.3f}")
    
    # Useless features
    print("\n❌ POTENTIALLY USELESS FEATURES (|r| < 0.05 with threshold):")
    print("-" * 60)
    
    useless = [(name, corr) for name, corr in sorted_corrs if abs(corr) < 0.05]
    for name, corr in useless:
        print(f"  {name:40} r={corr:+.3f}")
    
    # Least correlated features for threshold
    print("\n⬇️  LEAST CORRELATED WITH THRESHOLD (by |correlation|):")
    print("-" * 60)
    
    sorted_corrs_asc = sorted(feature_corrs, key=lambda x: abs(x[1]))
    for i, (name, corr) in enumerate(sorted_corrs_asc[:15]):
        sign = "+" if corr > 0 else ""
        bar = "█" * max(1, int(abs(corr) * 20))
        print(f"  {i+1:2}. {name:35} {sign}{corr:+.3f} {bar}")
    
    # Least correlated features for runtime
    print("\n⬇️  LEAST CORRELATED WITH RUNTIME (by |correlation|):")
    print("-" * 60)
    
    sorted_corrs_rt_asc = sorted(feature_corrs_rt, key=lambda x: abs(x[1]))
    for i, (name, corr) in enumerate(sorted_corrs_rt_asc[:15]):
        sign = "+" if corr > 0 else ""
        bar = "█" * max(1, int(abs(corr) * 20))
        print(f"  {i+1:2}. {name:35} {sign}{corr:+.3f} {bar}")
    
    print("\n" + "=" * 80)


def main():
    """Generate correlation heatmap for all features."""
    print("=" * 60)
    print("FULL FEATURE CORRELATION ANALYSIS")
    print("=" * 60 + "\n")
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Try to find data
    possible_paths = [
        project_root / "data" / "hackathon_public.json",
        script_dir / ".." / "data" / "hackathon_public.json",
        Path("./data/hackathon_public.json"),
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p.resolve()
            break
    
    if data_path is None:
        print("ERROR: Could not find hackathon_public.json")
        print("Tried:", [str(p) for p in possible_paths])
        return
    
    circuits_dir = data_path.parent.parent / "circuits"
    
    print(f"Data path: {data_path}")
    print(f"Circuits dir: {circuits_dir}")
    
    # Load data
    print("\nLoading data...")
    circuits, results, circuit_info, features_cache = load_all_data(data_path, circuits_dir)
    print(f"Loaded {len(circuits)} circuits, {len(results)} OK results")
    
    # Build feature matrix
    print("\nBuilding feature matrix...")
    matrix, feature_names = build_feature_matrix(results, circuit_info, features_cache)
    print(f"Matrix shape: {matrix.shape}")
    print(f"Features: {len(feature_names)}")
    
    # Output directory
    output_dir = project_root / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot full heatmap
    print("\nGenerating full correlation heatmap...")
    plot_full_correlation_heatmap(
        matrix, feature_names, 
        output_dir / "full_correlation_heatmap.png",
        figsize=(24, 20)
    )
    
    # Plot bar chart of correlations with targets
    print("Generating target correlation bar chart...")
    plot_correlation_with_targets(
        matrix, feature_names,
        output_dir / "target_correlations.png"
    )
    
    # Print summary
    print_correlation_summary(matrix, feature_names)
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
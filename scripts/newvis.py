#!/usr/bin/env python3
"""
Enhanced visualizations for quantum circuit fingerprint challenge.

EXISTING PLOTS (1-9):
1. MPS-Critical Features vs Threshold
2. Feature Redundancy Analysis
3. Feature Correlation Heatmap
4. Entanglement Structure Analysis
5. Temporal Feature Analysis
6. Feature Importance Proxy
7. Useless Feature Detection
8. Threshold Prediction Difficulty
9. Runtime Prediction Analysis

NEW PLOTS (10-18):
10. n_qubits Deep Dive - The most important feature
11. Gate Pattern Signatures by Family
12. Circuit Complexity Landscape (2D projection)
13. Threshold Transition Analysis (where thresholds change)
14. Fidelity Curve Shapes Analysis
15. Feature Distributions by Threshold Class
16. Interaction Graph Properties
17. Precision Impact Analysis
18. Combined Predictive Features
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src folder to path (relative to this script's location)
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / '..' / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from data_loader import (
    load_hackathon_data,
    extract_qasm_features,
    compute_min_threshold,
    THRESHOLD_LADDER,
    QuantumCircuitDataset,
)


def setup_style():
    """Configure matplotlib style."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('default')
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def load_all_data(data_path: Path, circuits_dir: Path):
    """Load all data with comprehensive features."""
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


def build_feature_dataframe(results, circuit_info, features_cache):
    """Build a comprehensive feature dataset."""
    data_points = []
    
    for r in results:
        info = circuit_info.get(r.file)
        features = features_cache.get(r.file, {})
        min_thresh = compute_min_threshold(r.threshold_sweep, target=0.99)
        
        if info and features and min_thresh and r.forward_wall_s:
            row = {
                'file': r.file,
                'family': info.family,
                'backend': r.backend,
                'precision': r.precision,
                'min_threshold': min_thresh,
                'threshold_class': THRESHOLD_LADDER.index(min_thresh) if min_thresh in THRESHOLD_LADDER else -1,
                'runtime': r.forward_wall_s,
                'log_runtime': np.log1p(r.forward_wall_s),
                'threshold_sweep': r.threshold_sweep,  # Keep for fidelity analysis
            }
            # Add all numeric features
            for key in QuantumCircuitDataset.NUMERIC_FEATURE_KEYS:
                row[key] = features.get(key, 0.0)
            data_points.append(row)
    
    return data_points


# ============================================================================
# ORIGINAL VISUALIZATIONS (1-9)
# ============================================================================

def plot_mps_critical_features(data_points, save_path):
    """Plot the most MPS-critical features against threshold."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    
    thresholds = np.array([d['min_threshold'] for d in data_points])
    log_thresholds = np.log2(thresholds)
    backends = [d['backend'] for d in data_points]
    colors = ['#3498db' if b == 'CPU' else '#e74c3c' for b in backends]
    
    critical_features = [
        ('middle_cut_crossings', 'Middle Cut Crossings', 'Directly predicts bond dimension'),
        ('max_cut_crossings', 'Max Cut Crossings', 'Worst-case entanglement cut'),
        ('max_span', 'Max Interaction Span', 'Longest-range 2Q gate'),
        ('nearest_neighbor_ratio', 'Nearest-Neighbor Ratio', 'Fraction of local interactions'),
        ('long_range_ratio', 'Long-Range Ratio', 'Fraction of non-local interactions'),
        ('light_cone_spread_rate', 'Light Cone Spread Rate', 'Information propagation speed'),
        ('avg_span', 'Average Span', 'Mean interaction distance'),
        ('graph_bandwidth', 'Graph Bandwidth', 'Should equal max_span!'),
        ('n_2q_gates', 'Total 2Q Gates', 'Raw entangling gate count'),
    ]
    
    for idx, (feature, title, desc) in enumerate(critical_features):
        ax = axes[idx // 3, idx % 3]
        values = np.array([d.get(feature, 0) for d in data_points])
        
        jitter = np.random.uniform(-0.15, 0.15, len(log_thresholds))
        ax.scatter(log_thresholds + jitter, values, c=colors, alpha=0.5, s=30, edgecolors='none')
        
        for thresh_val in sorted(set(thresholds)):
            mask = thresholds == thresh_val
            if mask.sum() > 0:
                mean_val = np.mean(values[mask])
                ax.scatter([np.log2(thresh_val)], [mean_val], c='black', s=150, marker='_', linewidths=3)
        
        ax.set_xlabel('Log₂(Min Threshold)')
        ax.set_ylabel(title)
        ax.set_title(f'{title}\n({desc})', fontsize=10)
        ax.set_xticks(np.log2(THRESHOLD_LADDER))
        ax.set_xticklabels([str(t) for t in THRESHOLD_LADDER], fontsize=8)
    
    cpu_patch = mpatches.Patch(color='#3498db', alpha=0.6, label='CPU')
    gpu_patch = mpatches.Patch(color='#e74c3c', alpha=0.6, label='GPU')
    fig.legend(handles=[cpu_patch, gpu_patch], loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    plt.suptitle('MPS-Critical Features vs Minimum Threshold\n(Black bars = mean per threshold)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_redundancy_analysis(data_points, save_path):
    """Visualize redundant feature pairs."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    redundant_pairs = [
        ('max_span', 'graph_bandwidth', 'Should be IDENTICAL'),
        ('n_2q_gates', 'n_cx', 'n_2q_gates ≈ n_cx + n_cz + n_swap + n_ccx'),
        ('middle_cut_crossings', 'cut_crossing_ratio', 'Ratio = crossings / n_2q_gates'),
        ('estimated_depth', 'depth_per_qubit', 'depth_per_qubit = depth / n_qubits'),
        ('qubit_activity_entropy', 'qubit_activity_variance', 'Both measure spread'),
        ('avg_degree', 'n_unique_pairs', 'Related graph metrics'),
    ]
    
    for idx, (feat1, feat2, note) in enumerate(redundant_pairs):
        ax = axes[idx // 3, idx % 3]
        
        x = np.array([d.get(feat1, 0) for d in data_points])
        y = np.array([d.get(feat2, 0) for d in data_points])
        thresholds = np.array([d['min_threshold'] for d in data_points])
        
        scatter = ax.scatter(x, y, c=np.log2(thresholds), cmap='viridis', alpha=0.6, s=30)
        
        valid_mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        if valid_mask.sum() > 2:
            corr = np.corrcoef(x[valid_mask], y[valid_mask])[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(feat1.replace('_', ' ').title())
        ax.set_ylabel(feat2.replace('_', ' ').title())
        ax.set_title(f'{note}', fontsize=10)
        
        if idx == 0:
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, 'r--', alpha=0.7, label='y=x (perfect match)')
            ax.legend(fontsize=8)
    
    plt.colorbar(scatter, ax=axes.ravel().tolist(), label='Log₂(Threshold)', shrink=0.6)
    plt.suptitle('Feature Redundancy Analysis\n(High correlation = potentially redundant)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_correlation_heatmap(data_points, save_path):
    """Plot correlation heatmap of key features."""
    key_features = [
        'n_qubits', 'n_cx', 'n_2q_gates', 'n_1q_gates',
        'avg_span', 'max_span', 'graph_bandwidth',
        'middle_cut_crossings', 'max_cut_crossings',
        'nearest_neighbor_ratio', 'long_range_ratio',
        'light_cone_spread_rate', 'estimated_depth',
        'gate_density', 'qubit_activity_entropy',
        'threshold_class', 'log_runtime'
    ]
    
    n_features = len(key_features)
    matrix = np.zeros((len(data_points), n_features))
    
    for i, d in enumerate(data_points):
        for j, feat in enumerate(key_features):
            matrix[i, j] = d.get(feat, 0)
    
    corr_matrix = np.corrcoef(matrix.T)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    labels = [f.replace('_', '\n') for f in key_features]
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    
    for i in range(n_features):
        for j in range(n_features):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=7)
    
    target_idx = key_features.index('threshold_class')
    runtime_idx = key_features.index('log_runtime')
    ax.axhline(y=target_idx - 0.5, color='red', linewidth=2, linestyle='--')
    ax.axhline(y=target_idx + 0.5, color='red', linewidth=2, linestyle='--')
    ax.axhline(y=runtime_idx - 0.5, color='blue', linewidth=2, linestyle='--')
    ax.axhline(y=runtime_idx + 0.5, color='blue', linewidth=2, linestyle='--')
    
    plt.colorbar(im, ax=ax, label='Pearson Correlation', shrink=0.8)
    ax.set_title('Feature Correlation Heatmap\n(Red = threshold, Blue = runtime)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_entanglement_structure(data_points, save_path):
    """Analyze entanglement structure features by circuit family."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    family_data = defaultdict(list)
    for d in data_points:
        family_data[d['family']].append(d)
    
    families = sorted(family_data.keys())
    n_families = len(families)
    x = np.arange(n_families)
    width = 0.35
    
    ax = axes[0, 0]
    nn_means = [np.mean([d['nearest_neighbor_ratio'] for d in family_data[f]]) for f in families]
    lr_means = [np.mean([d['long_range_ratio'] for d in family_data[f]]) for f in families]
    ax.bar(x - width/2, nn_means, width, label='Nearest-Neighbor', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, lr_means, width, label='Long-Range', color='#e74c3c', alpha=0.8)
    ax.set_ylabel('Ratio')
    ax.set_title('Entanglement Locality by Circuit Family')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n')[:12] for f in families], rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax = axes[0, 1]
    cut_data = [[d['middle_cut_crossings'] for d in family_data[f]] for f in families]
    bp = ax.boxplot(cut_data, patch_artist=True)
    colors = plt.cm.tab20(np.linspace(0, 1, n_families))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Middle Cut Crossings')
    ax.set_title('Entanglement Cut Pressure by Family')
    ax.set_xticks(range(1, n_families + 1))
    ax.set_xticklabels([f.replace('_', '\n')[:12] for f in families], rotation=45, ha='right', fontsize=7)
    
    ax = axes[1, 0]
    family_colors = {f: plt.cm.tab20(i / n_families) for i, f in enumerate(families)}
    for f in families:
        cuts = [d['middle_cut_crossings'] for d in family_data[f]]
        threshs = [d['min_threshold'] for d in family_data[f]]
        ax.scatter(cuts, threshs, c=[family_colors[f]], alpha=0.6, label=f[:10], s=40)
    ax.set_xlabel('Middle Cut Crossings')
    ax.set_ylabel('Minimum Threshold')
    ax.set_yscale('log', base=2)
    ax.set_yticks(THRESHOLD_LADDER)
    ax.set_yticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.set_title('Cut Crossings vs Threshold')
    
    ax = axes[1, 1]
    spread_rates = [[d['light_cone_spread_rate'] for d in family_data[f]] for f in families]
    bp = ax.boxplot(spread_rates, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Light Cone Spread Rate')
    ax.set_title('Information Propagation Speed by Family')
    ax.set_xticks(range(1, n_families + 1))
    ax.set_xticklabels([f.replace('_', '\n')[:12] for f in families], rotation=45, ha='right', fontsize=7)
    
    plt.suptitle('Entanglement Structure Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_temporal_features(data_points, save_path):
    """Analyze when entangling gates occur in circuits."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    thresholds = np.array([d['min_threshold'] for d in data_points])
    log_thresholds = np.log2(thresholds)
    
    ax = axes[0, 0]
    early = np.array([d['early_longrange_ratio'] for d in data_points])
    late = np.array([d['late_longrange_ratio'] for d in data_points])
    scatter = ax.scatter(early, late, c=log_thresholds, cmap='plasma', alpha=0.6, s=40)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Early Long-Range Ratio (first 1/3)')
    ax.set_ylabel('Late Long-Range Ratio (last 1/3)')
    ax.set_title('Temporal Distribution of Long-Range Gates')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Log₂(Threshold)')
    
    ax = axes[0, 1]
    temporal_center = np.array([d['longrange_temporal_center'] for d in data_points])
    jitter = np.random.uniform(-0.15, 0.15, len(log_thresholds))
    backends = [d['backend'] for d in data_points]
    colors = ['#3498db' if b == 'CPU' else '#e74c3c' for b in backends]
    ax.scatter(temporal_center, log_thresholds + jitter, c=colors, alpha=0.5, s=30)
    ax.set_xlabel('Long-Range Gate Temporal Center')
    ax.set_ylabel('Log₂(Min Threshold)')
    ax.set_yticks(np.log2(THRESHOLD_LADDER))
    ax.set_yticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Temporal Center vs Threshold')
    
    ax = axes[1, 0]
    velocity_by_thresh = defaultdict(list)
    for d in data_points:
        velocity_by_thresh[d['min_threshold']].append(d['entanglement_velocity'])
    thresh_list = sorted(velocity_by_thresh.keys())
    data = [velocity_by_thresh[t] for t in thresh_list]
    bp = ax.boxplot(data, patch_artist=True)
    colors_box = plt.cm.viridis(np.linspace(0, 1, len(thresh_list)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([str(t) for t in thresh_list])
    ax.set_xlabel('Minimum Threshold')
    ax.set_ylabel('Entanglement Velocity')
    ax.set_title('Entanglement Velocity by Threshold')
    
    ax = axes[1, 1]
    early_score = early * (1 - temporal_center)
    scatter = ax.scatter(early_score, thresholds, c=[d['runtime'] for d in data_points], 
                        cmap='coolwarm', alpha=0.6, s=40, norm=plt.matplotlib.colors.LogNorm())
    ax.set_xlabel('Early Entanglement Score')
    ax.set_ylabel('Minimum Threshold')
    ax.set_yscale('log', base=2)
    ax.set_yticks(THRESHOLD_LADDER)
    ax.set_yticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.set_title('Early Entanglement Score vs Threshold')
    plt.colorbar(scatter, ax=ax, label='Runtime (s)')
    
    plt.suptitle('Temporal Feature Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance_proxy(data_points, save_path):
    """Estimate feature importance via correlation with targets."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    all_features = QuantumCircuitDataset.NUMERIC_FEATURE_KEYS
    threshold_corrs = []
    runtime_corrs = []
    
    threshold_class = np.array([d['threshold_class'] for d in data_points])
    log_runtime = np.array([d['log_runtime'] for d in data_points])
    
    for feat in all_features:
        values = np.array([d.get(feat, 0) for d in data_points])
        valid = np.isfinite(values) & np.isfinite(threshold_class) & np.isfinite(log_runtime)
        if valid.sum() > 10:
            thresh_corr = np.corrcoef(values[valid], threshold_class[valid])[0, 1]
            runtime_corr = np.corrcoef(values[valid], log_runtime[valid])[0, 1]
        else:
            thresh_corr = 0
            runtime_corr = 0
        threshold_corrs.append(thresh_corr if np.isfinite(thresh_corr) else 0)
        runtime_corrs.append(runtime_corr if np.isfinite(runtime_corr) else 0)
    
    sorted_idx = np.argsort(np.abs(threshold_corrs))[::-1]
    
    ax = axes[0]
    top_n = 25
    top_idx = sorted_idx[:top_n]
    colors = ['#2ecc71' if threshold_corrs[i] > 0 else '#e74c3c' for i in top_idx]
    y_pos = np.arange(top_n)
    ax.barh(y_pos, [threshold_corrs[i] for i in top_idx], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([all_features[i].replace('_', ' ')[:25] for i in top_idx], fontsize=9)
    ax.set_xlabel('Correlation with Threshold Class')
    ax.set_title(f'Top {top_n} Features by Threshold Correlation')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    
    ax = axes[1]
    sorted_idx_runtime = np.argsort(np.abs(runtime_corrs))[::-1]
    top_idx = sorted_idx_runtime[:top_n]
    colors = ['#3498db' if runtime_corrs[i] > 0 else '#9b59b6' for i in top_idx]
    ax.barh(y_pos, [runtime_corrs[i] for i in top_idx], color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([all_features[i].replace('_', ' ')[:25] for i in top_idx], fontsize=9)
    ax.set_xlabel('Correlation with Log(Runtime)')
    ax.set_title(f'Top {top_n} Features by Runtime Correlation')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    
    plt.suptitle('Feature Importance (Correlation Analysis)', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_useless_features(data_points, save_path):
    """Identify and visualize potentially useless features."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    problematic_features = [
        ('layer_uniformity', 'BROKEN: Always 0.5'),
        ('min_span', 'LOW INFO: Usually 1'),
        ('active_qubit_fraction', 'LOW INFO: Usually 1.0'),
        ('final_light_cone_size', 'LOW INFO: Usually 1.0'),
        ('n_custom_gates', 'LOW INFO: Usually 0'),
        ('n_measure', 'LOW INFO: Constant per circuit'),
    ]
    
    for idx, (feat, note) in enumerate(problematic_features):
        ax = axes[idx // 3, idx % 3]
        values = [d.get(feat, 0) for d in data_points]
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color='gray')
        unique_vals = len(set(values))
        mean_val = np.mean(values)
        std_val = np.std(values)
        stats_text = f'Unique: {unique_vals}\nMean: {mean_val:.3f}\nStd: {std_val:.3f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(feat)
        ax.set_ylabel('Count')
        ax.set_title(f'{feat}\n({note})', fontsize=10, color='red')
    
    plt.suptitle('Potentially Useless Features', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_threshold_difficulty(data_points, save_path):
    """Analyze which circuits are hardest to predict."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    thresh_groups = defaultdict(list)
    for d in data_points:
        thresh_groups[d['min_threshold']].append(d)
    thresholds = sorted(thresh_groups.keys())
    
    ax = axes[0, 0]
    counts = [len(thresh_groups[t]) for t in thresholds]
    bars = ax.bar(range(len(thresholds)), counts, color=plt.cm.viridis(np.linspace(0, 1, len(thresholds))))
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel('Minimum Threshold')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution')
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{100*count/total:.1f}%', ha='center', fontsize=8)
    
    ax = axes[0, 1]
    key_features = ['n_qubits', 'n_2q_gates', 'middle_cut_crossings', 'max_span']
    x = np.arange(len(thresholds))
    width = 0.2
    for i, feat in enumerate(key_features):
        variances = []
        for t in thresholds:
            values = [d[feat] for d in thresh_groups[t]]
            variances.append(np.std(values) / (np.mean(values) + 1e-10))
        ax.bar(x + i*width, variances, width, label=feat.replace('_', ' '))
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel('Minimum Threshold')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Feature Variance Within Classes')
    ax.legend(fontsize=8)
    
    ax = axes[1, 0]
    configs = defaultdict(lambda: defaultdict(int))
    for d in data_points:
        config = f"{d['backend'][:1]}/{d['precision'][:1]}"
        configs[d['min_threshold']][config] += 1
    config_names = sorted(set(c for t in configs.values() for c in t.keys()))
    bottom = np.zeros(len(thresholds))
    for config in config_names:
        values = [configs[t][config] for t in thresholds]
        ax.bar(range(len(thresholds)), values, bottom=bottom, label=config)
        bottom += values
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel('Minimum Threshold')
    ax.set_ylabel('Count')
    ax.set_title('Backend/Precision by Threshold')
    ax.legend(title='Config')
    
    ax = axes[1, 1]
    families = sorted(set(d['family'] for d in data_points))
    matrix = np.zeros((len(families), len(thresholds)))
    for d in data_points:
        f_idx = families.index(d['family'])
        t_idx = thresholds.index(d['min_threshold'])
        matrix[f_idx, t_idx] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / (row_sums + 1e-10)
    im = ax.imshow(matrix_norm, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([f[:15] for f in families], fontsize=8)
    ax.set_xlabel('Minimum Threshold')
    ax.set_ylabel('Circuit Family')
    ax.set_title('Threshold by Family (row-normalized)')
    plt.colorbar(im, ax=ax, label='Proportion', shrink=0.8)
    
    plt.suptitle('Threshold Prediction Difficulty', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_runtime_analysis(data_points, save_path):
    """Detailed runtime prediction analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    runtimes = np.array([d['runtime'] for d in data_points])
    log_runtimes = np.log10(runtimes)
    backends = np.array([d['backend'] for d in data_points])
    
    ax = axes[0, 0]
    n_qubits = np.array([d['n_qubits'] for d in data_points])
    n_2q = np.array([d['n_2q_gates'] for d in data_points])
    complexity = n_qubits * np.log1p(n_2q)
    cpu_mask = backends == 'CPU'
    gpu_mask = backends == 'GPU'
    ax.scatter(complexity[cpu_mask], log_runtimes[cpu_mask], alpha=0.5, label='CPU', c='#3498db', s=30)
    ax.scatter(complexity[gpu_mask], log_runtimes[gpu_mask], alpha=0.5, label='GPU', c='#e74c3c', s=30)
    ax.set_xlabel('Complexity (n_qubits × log(n_2q_gates))')
    ax.set_ylabel('Log₁₀(Runtime)')
    ax.set_title('Complexity vs Runtime')
    ax.legend()
    
    ax = axes[0, 1]
    thresholds = np.array([d['min_threshold'] for d in data_points])
    for backend, color, marker in [('CPU', '#3498db', 'o'), ('GPU', '#e74c3c', '^')]:
        mask = backends == backend
        for thresh in sorted(set(thresholds)):
            t_mask = mask & (thresholds == thresh)
            if t_mask.sum() > 0:
                mean_rt = np.mean(log_runtimes[t_mask])
                std_rt = np.std(log_runtimes[t_mask])
                ax.errorbar(np.log2(thresh), mean_rt, yerr=std_rt, 
                           fmt=marker, color=color, capsize=3, markersize=8)
    ax.set_xlabel('Log₂(Threshold)')
    ax.set_ylabel('Log₁₀(Runtime)')
    ax.set_xticks(np.log2(THRESHOLD_LADDER))
    ax.set_xticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.set_title('Runtime by Threshold')
    cpu_patch = mpatches.Patch(color='#3498db', label='CPU')
    gpu_patch = mpatches.Patch(color='#e74c3c', label='GPU')
    ax.legend(handles=[cpu_patch, gpu_patch])
    
    ax = axes[1, 0]
    X = np.column_stack([n_qubits, np.log1p(n_2q)])
    X = np.column_stack([np.ones(len(X)), X])
    try:
        coeffs = np.linalg.lstsq(X, log_runtimes, rcond=None)[0]
        predicted = X @ coeffs
        residuals = log_runtimes - predicted
        ax.hist(residuals[cpu_mask], bins=30, alpha=0.6, label='CPU', color='#3498db')
        ax.hist(residuals[gpu_mask], bins=30, alpha=0.6, label='GPU', color='#e74c3c')
        ax.axvline(x=0, color='black', linestyle='--')
        ax.set_xlabel('Residual (log scale)')
        ax.set_ylabel('Count')
        ax.set_title(f'Residuals (RMSE: {np.sqrt(np.mean(residuals**2)):.3f})')
        ax.legend()
    except:
        ax.text(0.5, 0.5, 'Could not fit model', ha='center', va='center', transform=ax.transAxes)
    
    ax = axes[1, 1]
    circuit_runtimes = defaultdict(dict)
    for d in data_points:
        key = (d['file'], d['precision'])
        circuit_runtimes[key][d['backend']] = d['runtime']
    speedups = []
    cpu_times = []
    for key, times in circuit_runtimes.items():
        if 'CPU' in times and 'GPU' in times:
            speedup = times['CPU'] / times['GPU']
            speedups.append(speedup)
            cpu_times.append(times['CPU'])
    if speedups:
        ax.scatter(cpu_times, speedups, alpha=0.6, c='purple', s=30)
        ax.axhline(y=1, color='gray', linestyle='--', label='No speedup')
        ax.set_xlabel('CPU Runtime (s)')
        ax.set_ylabel('GPU Speedup (CPU/GPU)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'GPU Speedup (Median: {np.median(speedups):.2f}x)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No matched pairs', ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle('Runtime Prediction Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# NEW VISUALIZATIONS (10-18)
# ============================================================================

def plot_nqubits_deep_dive(data_points, save_path):
    """
    PLOT 10: Deep dive into n_qubits - the most important feature.
    Shows how n_qubits relates to threshold, runtime, and other features.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    n_qubits = np.array([d['n_qubits'] for d in data_points])
    thresholds = np.array([d['min_threshold'] for d in data_points])
    log_thresholds = np.log2(thresholds)
    runtimes = np.array([d['runtime'] for d in data_points])
    backends = np.array([d['backend'] for d in data_points])
    
    # Plot 1: n_qubits distribution
    ax = axes[0, 0]
    ax.hist(n_qubits, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.median(n_qubits), color='red', linestyle='--', label=f'Median: {np.median(n_qubits):.0f}')
    ax.axvline(np.mean(n_qubits), color='orange', linestyle='--', label=f'Mean: {np.mean(n_qubits):.1f}')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Count')
    ax.set_title('n_qubits Distribution')
    ax.legend()
    
    # Plot 2: n_qubits vs Threshold (THE KEY RELATIONSHIP)
    ax = axes[0, 1]
    colors = ['#3498db' if b == 'CPU' else '#e74c3c' for b in backends]
    jitter = np.random.uniform(-0.15, 0.15, len(log_thresholds))
    ax.scatter(n_qubits, log_thresholds + jitter, c=colors, alpha=0.5, s=40)
    
    # Add binned means
    qubit_bins = np.arange(0, n_qubits.max() + 5, 5)
    for i in range(len(qubit_bins) - 1):
        mask = (n_qubits >= qubit_bins[i]) & (n_qubits < qubit_bins[i+1])
        if mask.sum() > 0:
            ax.scatter([qubit_bins[i] + 2.5], [np.mean(log_thresholds[mask])], 
                      c='black', s=200, marker='_', linewidths=3, zorder=5)
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Log₂(Min Threshold)')
    ax.set_yticks(np.log2(THRESHOLD_LADDER))
    ax.set_yticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.set_title('n_qubits vs Threshold\n(THE KEY PREDICTOR)')
    
    # Plot 3: n_qubits vs Runtime
    ax = axes[0, 2]
    cpu_mask = backends == 'CPU'
    gpu_mask = backends == 'GPU'
    ax.scatter(n_qubits[cpu_mask], np.log10(runtimes[cpu_mask]), alpha=0.5, label='CPU', c='#3498db', s=30)
    ax.scatter(n_qubits[gpu_mask], np.log10(runtimes[gpu_mask]), alpha=0.5, label='GPU', c='#e74c3c', s=30)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Log₁₀(Runtime)')
    ax.set_title('n_qubits vs Runtime')
    ax.legend()
    
    # Plot 4: n_qubits vs n_2q_gates (scaling)
    ax = axes[1, 0]
    n_2q = np.array([d['n_2q_gates'] for d in data_points])
    scatter = ax.scatter(n_qubits, n_2q, c=log_thresholds, cmap='viridis', alpha=0.6, s=40)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Number of 2Q Gates')
    ax.set_yscale('log')
    ax.set_title('Qubit Count vs Gate Count Scaling')
    plt.colorbar(scatter, ax=ax, label='Log₂(Threshold)')
    
    # Plot 5: n_qubits vs max_span
    ax = axes[1, 1]
    max_span = np.array([d['max_span'] for d in data_points])
    scatter = ax.scatter(n_qubits, max_span, c=log_thresholds, cmap='plasma', alpha=0.6, s=40)
    ax.plot([0, n_qubits.max()], [0, n_qubits.max()], 'k--', alpha=0.3, label='max_span = n_qubits')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Max Span')
    ax.set_title('n_qubits vs Max Interaction Span')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Log₂(Threshold)')
    
    # Plot 6: Threshold distribution by qubit ranges
    ax = axes[1, 2]
    qubit_ranges = [(4, 10), (10, 15), (15, 20), (20, 25), (25, 35)]
    range_data = []
    range_labels = []
    for low, high in qubit_ranges:
        mask = (n_qubits >= low) & (n_qubits < high)
        if mask.sum() > 0:
            range_data.append(log_thresholds[mask])
            range_labels.append(f'{low}-{high}')
    
    bp = ax.boxplot(range_data, patch_artist=True)
    colors_box = plt.cm.coolwarm(np.linspace(0, 1, len(range_data)))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(range_labels)
    ax.set_xlabel('Qubit Range')
    ax.set_ylabel('Log₂(Threshold)')
    ax.set_yticks(np.log2(THRESHOLD_LADDER))
    ax.set_yticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.set_title('Threshold by Qubit Range')
    
    plt.suptitle('n_qubits Deep Dive: The Most Important Feature', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_gate_pattern_signatures(data_points, save_path):
    """
    PLOT 11: Gate pattern signatures that identify algorithm types.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    family_data = defaultdict(list)
    for d in data_points:
        family_data[d['family']].append(d)
    families = sorted(family_data.keys())
    n_families = len(families)
    colors = plt.cm.tab20(np.linspace(0, 1, n_families))
    
    # Plot 1: H-CX pattern count by family
    ax = axes[0, 0]
    h_cx_data = [[d['h_cx_pattern_count'] for d in family_data[f]] for f in families]
    bp = ax.boxplot(h_cx_data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f[:10] for f in families], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('H-CX Pattern Count')
    ax.set_title('Bell/GHZ Signature (H→CX patterns)')
    
    # Plot 2: CX-RZ-CX pattern (variational signature)
    ax = axes[0, 1]
    cx_rz_data = [[d['cx_rz_cx_pattern_count'] for d in family_data[f]] for f in families]
    bp = ax.boxplot(cx_rz_data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f[:10] for f in families], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('CX-RZ-CX Pattern Count')
    ax.set_title('Variational Signature (CX-rotation-CX)')
    
    # Plot 3: Rotation density by family
    ax = axes[0, 2]
    rot_data = [[d['rotation_density'] for d in family_data[f]] for f in families]
    bp = ax.boxplot(rot_data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f[:10] for f in families], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Rotation Density')
    ax.set_title('Parameterized Gate Density')
    
    # Plot 4: CX chain length by family
    ax = axes[1, 0]
    chain_data = [[d['cx_chain_max_length'] for d in family_data[f]] for f in families]
    bp = ax.boxplot(chain_data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f[:10] for f in families], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Max CX Chain Length')
    ax.set_title('Consecutive CX Gates')
    
    # Plot 5: Pattern features vs threshold
    ax = axes[1, 1]
    h_cx = np.array([d['h_cx_pattern_count'] for d in data_points])
    thresholds = np.array([d['min_threshold'] for d in data_points])
    scatter = ax.scatter(h_cx, thresholds, c=[d['n_qubits'] for d in data_points], 
                        cmap='coolwarm', alpha=0.6, s=40)
    ax.set_xlabel('H-CX Pattern Count')
    ax.set_ylabel('Minimum Threshold')
    ax.set_yscale('log', base=2)
    ax.set_yticks(THRESHOLD_LADDER)
    ax.set_yticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.set_title('H-CX Patterns vs Threshold')
    plt.colorbar(scatter, ax=ax, label='n_qubits')
    
    # Plot 6: Gate type entropy by family
    ax = axes[1, 2]
    entropy_data = [[d['gate_type_entropy'] for d in family_data[f]] for f in families]
    bp = ax.boxplot(entropy_data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f[:10] for f in families], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Gate Type Entropy')
    ax.set_title('Gate Diversity (Higher = more gate types)')
    
    plt.suptitle('Gate Pattern Signatures by Circuit Family', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_complexity_landscape(data_points, save_path):
    """
    PLOT 12: 2D projection of circuit complexity space.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Extract key features
    n_qubits = np.array([d['n_qubits'] for d in data_points])
    n_2q = np.array([d['n_2q_gates'] for d in data_points])
    max_span = np.array([d['max_span'] for d in data_points])
    cut_cross = np.array([d['middle_cut_crossings'] for d in data_points])
    thresholds = np.array([d['min_threshold'] for d in data_points])
    log_thresholds = np.log2(thresholds)
    runtimes = np.array([d['runtime'] for d in data_points])
    
    # Plot 1: n_qubits vs gate_density colored by threshold
    ax = axes[0, 0]
    gate_density = n_2q / n_qubits
    scatter = ax.scatter(n_qubits, gate_density, c=log_thresholds, cmap='viridis', alpha=0.6, s=40)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Gate Density (2Q gates / qubit)')
    ax.set_title('Qubit Count vs Gate Density')
    plt.colorbar(scatter, ax=ax, label='Log₂(Threshold)')
    
    # Plot 2: Normalized span vs cut crossings
    ax = axes[0, 1]
    norm_span = max_span / n_qubits
    norm_cuts = cut_cross / (n_2q + 1)
    scatter = ax.scatter(norm_span, norm_cuts, c=log_thresholds, cmap='plasma', alpha=0.6, s=40)
    ax.set_xlabel('Normalized Max Span (max_span / n_qubits)')
    ax.set_ylabel('Normalized Cut Crossings')
    ax.set_title('Span vs Cut Crossings (Normalized)')
    plt.colorbar(scatter, ax=ax, label='Log₂(Threshold)')
    
    # Plot 3: 2D complexity projection
    ax = axes[1, 0]
    # Define two composite axes
    complexity_x = n_qubits * np.log1p(n_2q)  # Size complexity
    complexity_y = max_span * (cut_cross + 1) / (n_qubits + 1)  # Entanglement complexity
    scatter = ax.scatter(complexity_x, complexity_y, c=log_thresholds, cmap='coolwarm', alpha=0.6, s=40)
    ax.set_xlabel('Size Complexity (n_qubits × log(n_2q))')
    ax.set_ylabel('Entanglement Complexity')
    ax.set_title('Circuit Complexity Landscape')
    plt.colorbar(scatter, ax=ax, label='Log₂(Threshold)')
    
    # Add threshold regions
    for thresh in [2, 8, 32]:
        mask = thresholds == thresh
        if mask.sum() > 5:
            hull_x = complexity_x[mask]
            hull_y = complexity_y[mask]
            ax.scatter(hull_x.mean(), hull_y.mean(), marker='*', s=300, 
                      edgecolors='black', linewidths=2, label=f'T={thresh} center')
    ax.legend(fontsize=8)
    
    # Plot 4: Runtime landscape
    ax = axes[1, 1]
    scatter = ax.scatter(complexity_x, complexity_y, c=np.log10(runtimes), cmap='YlOrRd', alpha=0.6, s=40)
    ax.set_xlabel('Size Complexity (n_qubits × log(n_2q))')
    ax.set_ylabel('Entanglement Complexity')
    ax.set_title('Runtime Landscape')
    plt.colorbar(scatter, ax=ax, label='Log₁₀(Runtime)')
    
    plt.suptitle('Circuit Complexity Landscape', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_threshold_transitions(data_points, save_path):
    """
    PLOT 13: Analyze where threshold transitions occur.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Features most related to threshold
    key_features = ['n_qubits', 'max_span', 'middle_cut_crossings', 'n_2q_gates']
    thresholds = np.array([d['min_threshold'] for d in data_points])
    
    for idx, feat in enumerate(key_features):
        ax = axes[idx // 2, idx % 2]
        values = np.array([d[feat] for d in data_points])
        
        # For each threshold transition, find the boundary
        for i in range(len(THRESHOLD_LADDER) - 1):
            t_low = THRESHOLD_LADDER[i]
            t_high = THRESHOLD_LADDER[i + 1]
            
            mask_low = thresholds == t_low
            mask_high = thresholds == t_high
            
            if mask_low.sum() > 0 and mask_high.sum() > 0:
                vals_low = values[mask_low]
                vals_high = values[mask_high]
                
                # Find overlap region
                boundary = (np.percentile(vals_low, 75) + np.percentile(vals_high, 25)) / 2
                
                ax.axhline(y=boundary, color=plt.cm.viridis(i / len(THRESHOLD_LADDER)), 
                          linestyle='--', alpha=0.7, label=f'{t_low}→{t_high}')
        
        # Scatter plot
        scatter = ax.scatter(np.arange(len(values)), values, c=np.log2(thresholds), 
                            cmap='viridis', alpha=0.4, s=20)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(feat.replace('_', ' ').title())
        ax.set_title(f'Threshold Transitions in {feat}')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right', title='Transitions')
    
    plt.suptitle('Threshold Transition Boundaries\n(Dashed lines = approximate decision boundaries)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_distributions_by_threshold(data_points, save_path):
    """
    PLOT 15: Feature distributions for each threshold class.
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    
    key_features = [
        'n_qubits', 'n_2q_gates', 'max_span', 
        'middle_cut_crossings', 'nearest_neighbor_ratio', 'gate_density',
        'estimated_depth', 'light_cone_spread_rate', 'avg_span'
    ]
    
    thresh_groups = defaultdict(list)
    for d in data_points:
        thresh_groups[d['min_threshold']].append(d)
    thresholds = sorted(thresh_groups.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    
    for idx, feat in enumerate(key_features):
        ax = axes[idx // 3, idx % 3]
        
        for t, color in zip(thresholds, colors):
            values = [d[feat] for d in thresh_groups[t]]
            ax.hist(values, bins=20, alpha=0.4, color=color, label=f'T={t}', density=True)
        
        ax.set_xlabel(feat.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(feat)
        if idx == 0:
            ax.legend(fontsize=6, ncol=3)
    
    plt.suptitle('Feature Distributions by Threshold Class', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_interaction_graph_properties(data_points, save_path):
    """
    PLOT 16: Interaction graph properties analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    thresholds = np.array([d['min_threshold'] for d in data_points])
    log_thresholds = np.log2(thresholds)
    
    graph_features = [
        ('max_degree', 'Max Degree', 'Most connected qubit'),
        ('avg_degree', 'Avg Degree', 'Mean connections per qubit'),
        ('degree_entropy', 'Degree Entropy', 'Connection uniformity'),
        ('n_connected_components', 'Connected Components', 'Separate regions'),
        ('clustering_coeff', 'Clustering Coefficient', 'Triangle density'),
        ('n_unique_pairs', 'Unique Pairs', 'Distinct interactions'),
    ]
    
    for idx, (feat, title, desc) in enumerate(graph_features):
        ax = axes[idx // 3, idx % 3]
        values = np.array([d.get(feat, 0) for d in data_points])
        
        scatter = ax.scatter(values, log_thresholds + np.random.uniform(-0.1, 0.1, len(values)), 
                            c=[d['n_qubits'] for d in data_points], cmap='coolwarm', alpha=0.5, s=30)
        
        # Add binned means
        if values.max() > values.min():
            bins = np.linspace(values.min(), values.max(), 10)
            for i in range(len(bins) - 1):
                mask = (values >= bins[i]) & (values < bins[i+1])
                if mask.sum() > 0:
                    ax.scatter([(bins[i] + bins[i+1])/2], [np.mean(log_thresholds[mask])], 
                              c='black', s=100, marker='_', linewidths=2)
        
        ax.set_xlabel(title)
        ax.set_ylabel('Log₂(Threshold)')
        ax.set_yticks(np.log2(THRESHOLD_LADDER))
        ax.set_yticklabels([str(t) for t in THRESHOLD_LADDER])
        ax.set_title(f'{title}\n({desc})')
        
        if idx == 5:
            plt.colorbar(scatter, ax=ax, label='n_qubits')
    
    plt.suptitle('Interaction Graph Properties vs Threshold', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_precision_impact(data_points, save_path):
    """
    PLOT 17: Impact of precision (single vs double) on results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    single_data = [d for d in data_points if d['precision'] == 'single']
    double_data = [d for d in data_points if d['precision'] == 'double']
    
    # Plot 1: Runtime comparison
    ax = axes[0, 0]
    single_rt = [d['runtime'] for d in single_data]
    double_rt = [d['runtime'] for d in double_data]
    ax.hist(np.log10(single_rt), bins=30, alpha=0.6, label=f'Single (n={len(single_rt)})', color='blue')
    ax.hist(np.log10(double_rt), bins=30, alpha=0.6, label=f'Double (n={len(double_rt)})', color='orange')
    ax.set_xlabel('Log₁₀(Runtime)')
    ax.set_ylabel('Count')
    ax.set_title('Runtime by Precision')
    ax.legend()
    
    # Plot 2: Threshold distribution
    ax = axes[0, 1]
    single_thresh = [d['threshold_class'] for d in single_data]
    double_thresh = [d['threshold_class'] for d in double_data]
    x = np.arange(len(THRESHOLD_LADDER))
    width = 0.35
    single_counts = [single_thresh.count(i) for i in range(len(THRESHOLD_LADDER))]
    double_counts = [double_thresh.count(i) for i in range(len(THRESHOLD_LADDER))]
    ax.bar(x - width/2, single_counts, width, label='Single', color='blue', alpha=0.7)
    ax.bar(x + width/2, double_counts, width, label='Double', color='orange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Count')
    ax.set_title('Threshold Distribution by Precision')
    ax.legend()
    
    # Plot 3: Matched pair analysis
    ax = axes[1, 0]
    circuit_data = defaultdict(dict)
    for d in data_points:
        key = (d['file'], d['backend'])
        circuit_data[key][d['precision']] = d
    
    matched_single_rt = []
    matched_double_rt = []
    for key, prec_data in circuit_data.items():
        if 'single' in prec_data and 'double' in prec_data:
            matched_single_rt.append(prec_data['single']['runtime'])
            matched_double_rt.append(prec_data['double']['runtime'])
    
    if matched_single_rt:
        ax.scatter(matched_single_rt, matched_double_rt, alpha=0.6, c='purple', s=30)
        max_val = max(max(matched_single_rt), max(matched_double_rt))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x')
        ax.set_xlabel('Single Precision Runtime')
        ax.set_ylabel('Double Precision Runtime')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'Matched Pair Runtime Comparison (n={len(matched_single_rt)})')
        
        ratio = np.array(matched_double_rt) / np.array(matched_single_rt)
        ax.text(0.05, 0.95, f'Median ratio: {np.median(ratio):.2f}x', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No matched pairs', ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Precision effect by n_qubits
    ax = axes[1, 1]
    single_qubits = [d['n_qubits'] for d in single_data]
    double_qubits = [d['n_qubits'] for d in double_data]
    single_thresh_vals = [d['min_threshold'] for d in single_data]
    double_thresh_vals = [d['min_threshold'] for d in double_data]
    
    ax.scatter(single_qubits, np.log2(single_thresh_vals) + np.random.uniform(-0.1, 0.1, len(single_qubits)), 
              alpha=0.4, label='Single', c='blue', s=30)
    ax.scatter(double_qubits, np.log2(double_thresh_vals) + np.random.uniform(-0.1, 0.1, len(double_qubits)), 
              alpha=0.4, label='Double', c='orange', s=30)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Log₂(Threshold)')
    ax.set_yticks(np.log2(THRESHOLD_LADDER))
    ax.set_yticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.set_title('Precision Effect on Threshold by Qubit Count')
    ax.legend()
    
    plt.suptitle('Precision Impact Analysis (Single vs Double)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_predictive_features(data_points, save_path):
    """
    PLOT 18: Combined/engineered features for better prediction.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract base features
    n_qubits = np.array([d['n_qubits'] for d in data_points])
    n_2q = np.array([d['n_2q_gates'] for d in data_points])
    max_span = np.array([d['max_span'] for d in data_points])
    cut_cross = np.array([d['middle_cut_crossings'] for d in data_points])
    nn_ratio = np.array([d['nearest_neighbor_ratio'] for d in data_points])
    thresholds = np.array([d['min_threshold'] for d in data_points])
    log_thresholds = np.log2(thresholds)
    runtimes = np.array([d['runtime'] for d in data_points])
    
    # Define combined features
    combined_features = [
        (n_qubits * max_span / (n_qubits + 1), 'qubit_span_product', 'n_qubits × max_span / n_qubits'),
        (cut_cross * (1 - nn_ratio), 'cut_nonlocal_product', 'cut_crossings × (1 - nn_ratio)'),
        (n_2q * max_span / (n_qubits ** 2), 'normalized_entanglement', 'n_2q × max_span / n_qubits²'),
        (np.log1p(n_qubits) * np.log1p(cut_cross), 'log_complexity', 'log(n_qubits) × log(cut_crossings)'),
        (max_span ** 2 / n_qubits, 'span_pressure', 'max_span² / n_qubits'),
        (n_2q / n_qubits * (1 - nn_ratio), 'effective_entanglement', 'gate_density × long_range_ratio'),
    ]
    
    for idx, (values, name, formula) in enumerate(combined_features):
        ax = axes[idx // 3, idx % 3]
        
        # Handle potential inf/nan
        valid = np.isfinite(values)
        scatter = ax.scatter(values[valid], log_thresholds[valid], 
                            c=np.log10(runtimes[valid]), cmap='coolwarm', alpha=0.5, s=30)
        
        # Calculate correlation
        if valid.sum() > 10:
            corr = np.corrcoef(values[valid], log_thresholds[valid])[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(name.replace('_', ' ').title())
        ax.set_ylabel('Log₂(Threshold)')
        ax.set_yticks(np.log2(THRESHOLD_LADDER))
        ax.set_yticklabels([str(t) for t in THRESHOLD_LADDER])
        ax.set_title(f'{name}\n({formula})')
        
        if idx == 5:
            plt.colorbar(scatter, ax=ax, label='Log₁₀(Runtime)')
    
    plt.suptitle('Engineered Combined Features for Prediction\n(Higher correlation = better predictor)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("ENHANCED QUANTUM CIRCUIT VISUALIZATIONS")
    print("=" * 60 + "\n")
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    possible_data_paths = [
        project_root / "data" / "hackathon_public.json",
        script_dir / ".." / "data" / "hackathon_public.json",
        Path("./data/hackathon_public.json"),
    ]
    
    data_path = None
    for p in possible_data_paths:
        if p.exists():
            data_path = p.resolve()
            break
    
    if data_path is None:
        print("ERROR: Could not find hackathon_public.json")
        print("Looked in:", [str(p) for p in possible_data_paths])
        return
    
    circuits_dir = data_path.parent.parent / "circuits"
    print(f"Data path: {data_path}")
    print(f"Circuits dir: {circuits_dir}")
    
    if not circuits_dir.exists():
        print(f"ERROR: Circuits directory not found: {circuits_dir}")
        return
    
    # Load data
    print("\nLoading data...")
    circuits, results, circuit_info, features_cache = load_all_data(data_path, circuits_dir)
    print(f"Loaded {len(circuits)} circuits, {len(results)} results")
    
    print("\nBuilding feature dataset...")
    data_points = build_feature_dataframe(results, circuit_info, features_cache)
    print(f"Created {len(data_points)} data points")
    
    # Output directory
    output_dir = project_root / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_style()
    
    print("\n" + "=" * 60)
    print("GENERATING ORIGINAL VISUALIZATIONS (1-9)")
    print("=" * 60)
    
    plot_mps_critical_features(data_points, output_dir / "01_mps_critical_features.png")
    plot_redundancy_analysis(data_points, output_dir / "02_redundancy_analysis.png")
    plot_correlation_heatmap(data_points, output_dir / "03_correlation_heatmap.png")
    plot_entanglement_structure(data_points, output_dir / "04_entanglement_structure.png")
    plot_temporal_features(data_points, output_dir / "05_temporal_features.png")
    plot_feature_importance_proxy(data_points, output_dir / "06_feature_importance.png")
    plot_useless_features(data_points, output_dir / "07_useless_features.png")
    plot_threshold_difficulty(data_points, output_dir / "08_threshold_difficulty.png")
    plot_runtime_analysis(data_points, output_dir / "09_runtime_analysis.png")
    
    print("\n" + "=" * 60)
    print("GENERATING NEW VISUALIZATIONS (10-18)")
    print("=" * 60)
    
    plot_nqubits_deep_dive(data_points, output_dir / "10_nqubits_deep_dive.png")
    plot_gate_pattern_signatures(data_points, output_dir / "11_gate_pattern_signatures.png")
    plot_complexity_landscape(data_points, output_dir / "12_complexity_landscape.png")
    plot_threshold_transitions(data_points, output_dir / "13_threshold_transitions.png")
    plot_feature_distributions_by_threshold(data_points, output_dir / "15_feature_distributions.png")
    plot_interaction_graph_properties(data_points, output_dir / "16_interaction_graph.png")
    plot_precision_impact(data_points, output_dir / "17_precision_impact.png")
    plot_combined_predictive_features(data_points, output_dir / "18_combined_features.png")
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)
    
    # Print summary
    print("\nVISUALIZATION SUMMARY:")
    print("-" * 60)
    print("ORIGINAL (1-9):")
    print("  01: MPS-Critical Features vs Threshold")
    print("  02: Feature Redundancy Analysis")
    print("  03: Feature Correlation Heatmap")
    print("  04: Entanglement Structure by Family")
    print("  05: Temporal Feature Analysis")
    print("  06: Feature Importance (Correlations)")
    print("  07: Useless Feature Detection")
    print("  08: Threshold Prediction Difficulty")
    print("  09: Runtime Prediction Analysis")
    print("\nNEW (10-18):")
    print("  10: n_qubits Deep Dive (MOST IMPORTANT)")
    print("  11: Gate Pattern Signatures by Family")
    print("  12: Circuit Complexity Landscape (2D)")
    print("  13: Threshold Transition Boundaries")
    print("  15: Feature Distributions by Threshold")
    print("  16: Interaction Graph Properties")
    print("  17: Precision Impact Analysis")
    print("  18: Combined/Engineered Features")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Visualization script for quantum circuit fingerprint challenge data.

Generates plots to help understand patterns in threshold selection and runtime.
"""

import sys
from pathlib import Path
from collections import defaultdict
# Print current directory to debug


# Try adding current directory
sys.path.insert(0, '.')

# If data_loader.py is in a 'src' subfolder:
sys.path.insert(0, './src')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_loader import (
    load_hackathon_data,
    extract_qasm_features,
    compute_min_threshold,
    THRESHOLD_LADDER,
    BACKEND_MAP,
    PRECISION_MAP,
)


def setup_style():
    """Configure matplotlib style."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def load_all_data(project_root: Path):
    """Load all data with features."""
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    circuits, results = load_hackathon_data(data_path)
    circuit_info = {c.file: c for c in circuits}
    
    features_cache = {}
    for c in circuits:
        qasm_path = circuits_dir / c.file
        if qasm_path.exists():
            features_cache[c.file] = extract_qasm_features(qasm_path)
    
    ok_results = [r for r in results if r.status == "ok"]
    
    return circuits, ok_results, circuit_info, features_cache


def plot_threshold_distribution_by_family(results, circuit_info, save_path):
    """Plot threshold distribution grouped by circuit family."""
    family_thresholds = defaultdict(list)
    
    for r in results:
        info = circuit_info.get(r.file)
        if info:
            min_thresh = compute_min_threshold(r.threshold_sweep, target=0.99)
            if min_thresh:
                family_thresholds[info.family].append(min_thresh)
    
    families = sorted(family_thresholds.keys())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    positions = []
    labels = []
    data = []
    
    for i, family in enumerate(families):
        thresholds = family_thresholds[family]
        data.append(thresholds)
        positions.append(i)
        short_name = family.replace("_", "\n")
        labels.append(short_name)
    
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(families)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_yscale('log', base=2)
    ax.set_yticks(THRESHOLD_LADDER)
    ax.set_yticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.set_ylabel('Minimum Threshold (meeting 0.99 fidelity)')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Circuit Family')
    ax.set_title('Minimum Threshold Distribution by Circuit Family')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_runtime_vs_threshold(results, save_path):
    """Plot forward runtime vs selected threshold."""
    thresholds = []
    runtimes = []
    backends = []
    
    for r in results:
        if r.forward_wall_s and r.selected_threshold:
            thresholds.append(r.selected_threshold)
            runtimes.append(r.forward_wall_s)
            backends.append(r.backend)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    cpu_mask = [b == "CPU" for b in backends]
    gpu_mask = [b == "GPU" for b in backends]
    
    cpu_thresh = [t for t, m in zip(thresholds, cpu_mask) if m]
    cpu_runtime = [r for r, m in zip(runtimes, cpu_mask) if m]
    gpu_thresh = [t for t, m in zip(thresholds, gpu_mask) if m]
    gpu_runtime = [r for r, m in zip(runtimes, gpu_mask) if m]
    
    ax.scatter(cpu_thresh, cpu_runtime, alpha=0.6, label='CPU', c='blue', s=50)
    ax.scatter(gpu_thresh, gpu_runtime, alpha=0.6, label='GPU', c='orange', s=50)
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(THRESHOLD_LADDER)
    ax.set_xticklabels([str(t) for t in THRESHOLD_LADDER])
    ax.set_xlabel('Selected Threshold')
    ax.set_ylabel('Forward Runtime (seconds)')
    ax.set_title('Forward Runtime vs Selected Threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_fidelity_curves(results, circuit_info, save_path):
    """Plot fidelity vs threshold curves for selected circuits."""
    family_samples = defaultdict(list)
    for r in results:
        info = circuit_info.get(r.file)
        if info and r.threshold_sweep:
            family_samples[info.family].append(r)
    
    sample_results = []
    for family, rs in family_samples.items():
        sample_results.append(rs[0])
    sample_results = sample_results[:12]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, r in enumerate(sample_results):
        ax = axes[idx]
        info = circuit_info.get(r.file)
        
        thresholds = []
        fidelities = []
        for entry in sorted(r.threshold_sweep, key=lambda x: x.threshold):
            if entry.sdk_get_fidelity is not None:
                thresholds.append(entry.threshold)
                fidelities.append(entry.sdk_get_fidelity)
        
        if thresholds:
            ax.plot(thresholds, fidelities, 'o-', linewidth=2, markersize=6)
            ax.axhline(y=0.99, color='r', linestyle='--', alpha=0.7, label='Target (0.99)')
            ax.set_xscale('log', base=2)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Fidelity')
            title = f"{info.family if info else 'Unknown'}\n({r.backend}/{r.precision})"
            ax.set_title(title, fontsize=9)
    
    for idx in range(len(sample_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Fidelity vs Threshold Curves (Sample Circuits)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_runtime_by_backend_precision(results, save_path):
    """Plot runtime distributions by backend and precision."""
    categories = defaultdict(list)
    
    for r in results:
        if r.forward_wall_s:
            key = f"{r.backend}\n{r.precision}"
            categories[key].append(r.forward_wall_s)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    keys = sorted(categories.keys())
    data = [categories[k] for k in keys]
    
    bp = ax1.boxplot(data, tick_labels=keys, patch_artist=True)
    colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_yscale('log')
    ax1.set_ylabel('Forward Runtime (seconds)')
    ax1.set_title('Runtime Distribution by Backend/Precision')
    
    for i, (key, values) in enumerate(zip(keys, data)):
        jitter = np.random.uniform(-0.15, 0.15, len(values))
        ax2.scatter([i + 1 + jitter[j] for j in range(len(values))], 
                   values, alpha=0.5, s=30)
    ax2.set_yscale('log')
    ax2.set_xticks(range(1, len(keys) + 1))
    ax2.set_xticklabels(keys)
    ax2.set_ylabel('Forward Runtime (seconds)')
    ax2.set_title('Runtime by Backend/Precision (with jitter)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_correlations(results, circuit_info, features_cache, save_path):
    """Plot correlations between circuit features and outcomes."""
    data_points = []
    
    for r in results:
        info = circuit_info.get(r.file)
        features = features_cache.get(r.file, {})
        min_thresh = compute_min_threshold(r.threshold_sweep, target=0.99)
        
        if info and features and min_thresh and r.forward_wall_s:
            data_points.append({
                'n_qubits': features.get('n_qubits', 0),
                'n_2q_gates': features.get('n_2q_gates', 0),
                'n_lines': features.get('n_lines', 0),
                'avg_span': features.get('avg_span', 0),
                'gate_density': features.get('gate_density', 0),
                'min_threshold': min_thresh,
                'runtime': r.forward_wall_s,
                'backend': r.backend,
            })
    
    if not data_points:
        print("No data points for correlation plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    n_qubits = [d['n_qubits'] for d in data_points]
    n_2q_gates = [d['n_2q_gates'] for d in data_points]
    n_lines = [d['n_lines'] for d in data_points]
    avg_span = [d['avg_span'] for d in data_points]
    min_thresh = [d['min_threshold'] for d in data_points]
    runtime = [d['runtime'] for d in data_points]
    backends = [d['backend'] for d in data_points]
    
    colors = ['blue' if b == 'CPU' else 'orange' for b in backends]
    
    ax = axes[0, 0]
    ax.scatter(n_qubits, min_thresh, c=colors, alpha=0.6)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Minimum Threshold')
    ax.set_yscale('log', base=2)
    ax.set_title('Qubits vs Min Threshold')
    
    ax = axes[0, 1]
    ax.scatter(n_2q_gates, min_thresh, c=colors, alpha=0.6)
    ax.set_xlabel('Number of 2Q Gates')
    ax.set_ylabel('Minimum Threshold')
    ax.set_xscale('log')
    ax.set_yscale('log', base=2)
    ax.set_title('2Q Gates vs Min Threshold')
    
    ax = axes[0, 2]
    ax.scatter(avg_span, min_thresh, c=colors, alpha=0.6)
    ax.set_xlabel('Average Qubit Span')
    ax.set_ylabel('Minimum Threshold')
    ax.set_yscale('log', base=2)
    ax.set_title('Avg Span vs Min Threshold')
    
    ax = axes[1, 0]
    ax.scatter(n_qubits, runtime, c=colors, alpha=0.6)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Forward Runtime (s)')
    ax.set_yscale('log')
    ax.set_title('Qubits vs Runtime')
    
    ax = axes[1, 1]
    ax.scatter(n_2q_gates, runtime, c=colors, alpha=0.6)
    ax.set_xlabel('Number of 2Q Gates')
    ax.set_ylabel('Forward Runtime (s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('2Q Gates vs Runtime')
    
    ax = axes[1, 2]
    ax.scatter(min_thresh, runtime, c=colors, alpha=0.6)
    ax.set_xlabel('Minimum Threshold')
    ax.set_ylabel('Forward Runtime (s)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_title('Min Threshold vs Runtime')
    
    cpu_patch = mpatches.Patch(color='blue', alpha=0.6, label='CPU')
    gpu_patch = mpatches.Patch(color='orange', alpha=0.6, label='GPU')
    fig.legend(handles=[cpu_patch, gpu_patch], loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle('Feature Correlations with Threshold and Runtime', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_circuit_characteristics(circuits, features_cache, save_path):
    """Plot circuit characteristics by family."""
    family_features = defaultdict(lambda: defaultdict(list))
    
    for c in circuits:
        features = features_cache.get(c.file, {})
        if features:
            family_features[c.family]['n_qubits'].append(features.get('n_qubits', 0))
            family_features[c.family]['n_2q_gates'].append(features.get('n_2q_gates', 0))
            family_features[c.family]['n_1q_gates'].append(features.get('n_1q_gates', 0))
            family_features[c.family]['gate_density'].append(features.get('gate_density', 0))
    
    families = sorted(family_features.keys())
    n_families = len(families)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(n_families)
    width = 0.6
    
    ax = axes[0, 0]
    means = [np.mean(family_features[f]['n_qubits']) for f in families]
    ax.bar(x, means, width, color=plt.cm.tab20(np.linspace(0, 1, n_families)))
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in families], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Avg Qubits')
    ax.set_title('Average Qubit Count by Family')
    
    ax = axes[0, 1]
    means = [np.mean(family_features[f]['n_2q_gates']) for f in families]
    ax.bar(x, means, width, color=plt.cm.tab20(np.linspace(0, 1, n_families)))
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in families], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Avg 2Q Gates')
    ax.set_yscale('log')
    ax.set_title('Average 2-Qubit Gate Count by Family')
    
    ax = axes[1, 0]
    means = [np.mean(family_features[f]['n_1q_gates']) for f in families]
    ax.bar(x, means, width, color=plt.cm.tab20(np.linspace(0, 1, n_families)))
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in families], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Avg 1Q Gates')
    ax.set_yscale('log')
    ax.set_title('Average 1-Qubit Gate Count by Family')
    
    ax = axes[1, 1]
    means = [np.mean(family_features[f]['gate_density']) for f in families]
    ax.bar(x, means, width, color=plt.cm.tab20(np.linspace(0, 1, n_families)))
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in families], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Avg Gate Density (2Q gates / qubit)')
    ax.set_title('Gate Density by Family')
    
    plt.suptitle('Circuit Characteristics by Family', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_threshold_sweep_heatmap(results, circuit_info, save_path):
    """Create heatmap of fidelity across thresholds for all circuits."""
    circuit_data = []
    
    for r in results:
        info = circuit_info.get(r.file)
        if not info or not r.threshold_sweep:
            continue
        
        fidelities = {}
        for entry in r.threshold_sweep:
            if entry.sdk_get_fidelity is not None:
                fidelities[entry.threshold] = entry.sdk_get_fidelity
        
        if fidelities:
            circuit_data.append({
                'name': f"{info.family[:8]}_{r.file.split('_')[-1][:4]}_{r.backend[0]}{r.precision[0]}",
                'fidelities': fidelities,
                'family': info.family,
            })
    
    circuit_data.sort(key=lambda x: (x['family'], x['name']))
    
    thresholds = THRESHOLD_LADDER[:8]
    matrix = np.full((len(circuit_data), len(thresholds)), np.nan)
    
    for i, cd in enumerate(circuit_data):
        for j, t in enumerate(thresholds):
            if t in cd['fidelities']:
                matrix[i, j] = cd['fidelities'][t]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(circuit_data) * 0.25)))
    
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='lightgray')
    
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_xlabel('Threshold')
    
    ax.set_yticks(range(len(circuit_data)))
    ax.set_yticklabels([cd['name'] for cd in circuit_data], fontsize=6)
    ax.set_ylabel('Circuit')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Fidelity')
    
    ax.axvline(x=-0.5, color='white', linewidth=0.5)
    for j in range(len(thresholds)):
        ax.axvline(x=j + 0.5, color='white', linewidth=0.5)
    
    ax.set_title('Fidelity Heatmap Across Thresholds\n(Gray = timeout/missing)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_runtime_distribution_histogram(results, save_path):
    """Plot histogram of log runtimes."""
    runtimes = [r.forward_wall_s for r in results if r.forward_wall_s]
    log_runtimes = np.log10(runtimes)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(log_runtimes, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Log10(Runtime) [seconds]')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Log Runtime')
    ax1.axvline(np.median(log_runtimes), color='red', linestyle='--', label=f'Median: {10**np.median(log_runtimes):.1f}s')
    ax1.legend()
    
    cpu_runtimes = [r.forward_wall_s for r in results if r.forward_wall_s and r.backend == 'CPU']
    gpu_runtimes = [r.forward_wall_s for r in results if r.forward_wall_s and r.backend == 'GPU']
    
    ax2.hist(np.log10(cpu_runtimes), bins=20, alpha=0.6, label='CPU', color='blue')
    ax2.hist(np.log10(gpu_runtimes), bins=20, alpha=0.6, label='GPU', color='orange')
    ax2.set_xlabel('Log10(Runtime) [seconds]')
    ax2.set_ylabel('Count')
    ax2.set_title('Runtime Distribution: CPU vs GPU')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_stats(circuits, results, circuit_info, save_path):
    """Create a summary statistics overview."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    families = [c.family for c in circuits]
    family_counts = {}
    for f in families:
        family_counts[f] = family_counts.get(f, 0) + 1
    sorted_families = sorted(family_counts.items(), key=lambda x: -x[1])
    ax.barh([f[0].replace('_', ' ') for f in sorted_families], 
            [f[1] for f in sorted_families], color='steelblue')
    ax.set_xlabel('Count')
    ax.set_title('Circuits per Family')
    
    ax = axes[0, 1]
    status_counts = {}
    all_results_path = Path(__file__).parent.parent / "data" / "hackathon_public.json"
    _, all_results = load_hackathon_data(all_results_path)
    for r in all_results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
    colors = ['green' if s == 'ok' else 'red' for s in status_counts.keys()]
    ax.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%', colors=colors)
    ax.set_title('Result Status Distribution')
    
    ax = axes[1, 0]
    qubit_counts = [c.n_qubits for c in circuits]
    ax.hist(qubit_counts, bins=15, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Count')
    ax.set_title('Qubit Count Distribution')
    
    ax = axes[1, 1]
    runtimes_cpu = [r.forward_wall_s for r in results if r.forward_wall_s and r.backend == 'CPU']
    runtimes_gpu = [r.forward_wall_s for r in results if r.forward_wall_s and r.backend == 'GPU']
    
    stats_text = f"""Dataset Summary
─────────────────────────────
Circuits: {len(circuits)}
Results (ok): {len(results)}
Families: {len(set(c.family for c in circuits))}

Runtime Statistics (seconds)
─────────────────────────────
CPU:
  Min: {min(runtimes_cpu):.2f}
  Max: {max(runtimes_cpu):.2f}
  Median: {np.median(runtimes_cpu):.2f}
  
GPU:
  Min: {min(runtimes_gpu):.2f}
  Max: {max(runtimes_gpu):.2f}
  Median: {np.median(runtimes_gpu):.2f}

Threshold Ladder: {THRESHOLD_LADDER}
"""
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontfamily='monospace',
            fontsize=10, verticalalignment='center')
    ax.axis('off')
    ax.set_title('Summary Statistics')
    
    plt.suptitle('Quantum Circuit Dataset Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("QUANTUM CIRCUIT DATA VISUALIZATION")
    print("=" * 60 + "\n")
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    circuits, results, circuit_info, features_cache = load_all_data(project_root)
    print(f"Loaded {len(circuits)} circuits, {len(results)} results\n")
    
    setup_style()
    
    print("Generating visualizations...")
    
    plot_summary_stats(circuits, results, circuit_info, 
                      output_dir / "01_summary_stats.png")
    
    plot_threshold_distribution_by_family(results, circuit_info,
                                         output_dir / "02_threshold_by_family.png")
    
    plot_runtime_vs_threshold(results,
                             output_dir / "03_runtime_vs_threshold.png")
    
    plot_fidelity_curves(results, circuit_info,
                        output_dir / "04_fidelity_curves.png")
    
    plot_runtime_by_backend_precision(results,
                                     output_dir / "05_runtime_by_backend.png")
    
    plot_feature_correlations(results, circuit_info, features_cache,
                             output_dir / "06_feature_correlations.png")
    
    plot_circuit_characteristics(circuits, features_cache,
                                output_dir / "07_circuit_characteristics.png")
    
    plot_threshold_sweep_heatmap(results, circuit_info,
                                output_dir / "08_fidelity_heatmap.png")
    
    plot_runtime_distribution_histogram(results,
                                       output_dir / "09_runtime_histogram.png")
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

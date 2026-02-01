#!/usr/bin/env python3
"""
Run model evaluation and visualize score distributions.
"""

import sys
from pathlib import Path
import json
import subprocess
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scoring import compute_challenge_score


def run_evaluation(model: str, n_runs: int = 100, verbose: bool = False):
    """Run the evaluation script and capture results."""
    cmd = [
        "python", "./scripts/evaluate.py",
        "--model", model,
        "--n-runs", str(n_runs),
    ]
    if verbose:
        cmd.append("--verbose")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        raise RuntimeError(f"Evaluation failed with return code {result.returncode}")
    
    return result.stdout


def extract_results_from_output(output: str):
    """Extract JSON results from script output."""
    # Find the JSON output (it's printed at the end)
    lines = output.split('\n')
    json_start = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith('{'):
            json_start = i
            break
    
    if json_start is None:
        raise ValueError("Could not find JSON results in output")
    
    json_str = '\n'.join(lines[json_start:])
    return json.loads(json_str)


def visualize_distributions(results_data, save_path: Path = None):
    """Create distribution visualizations."""
    all_results = results_data.get("all_results", [])
    
    if not all_results:
        print("No results to visualize")
        return
    
    # Extract metrics
    threshold_accs = []
    runtime_maes = []
    combined_scores = []
    
    for result in all_results:
        # Get validation metrics
        val_metrics = result.get("val_metrics", {})
        if val_metrics:
            threshold_accs.append(val_metrics.get("threshold_accuracy", 0))
            runtime_maes.append(val_metrics.get("runtime_mae", 0))
        
        # Compute combined score
        challenge_scores = result.get("challenge_scores", {})
        if challenge_scores:
            combined_scores.append(challenge_scores.get("combined_score", 0))
    
    threshold_accs = np.array(threshold_accs)
    runtime_maes = np.array(runtime_maes)
    combined_scores = np.array(combined_scores)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # Combined Score Distribution
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(combined_scores, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax1.axvline(combined_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {combined_scores.mean():.4f}')
    ax1.axvline(combined_scores.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {combined_scores.median():.4f}')
    ax1.set_xlabel('Combined Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Combined Challenge Scores', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Combined Score Stats
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis('off')
    stats_text = f"""Combined Score Stats:
Mean: {combined_scores.mean():.4f}
Std: {combined_scores.std():.4f}
Min: {combined_scores.min():.4f}
Max: {combined_scores.max():.4f}
Median: {combined_scores.median():.4f}
Q1: {np.percentile(combined_scores, 25):.4f}
Q3: {np.percentile(combined_scores, 75):.4f}"""
    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Threshold Accuracy Distribution
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.hist(threshold_accs, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(threshold_accs.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {threshold_accs.mean():.4f}')
    ax2.axvline(threshold_accs.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {threshold_accs.median():.4f}')
    ax2.set_xlabel('Threshold Accuracy', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Threshold Accuracy', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Threshold Stats
    ax_thresh_stats = fig.add_subplot(gs[1, 2])
    ax_thresh_stats.axis('off')
    thresh_stats_text = f"""Threshold Accuracy Stats:
Mean: {threshold_accs.mean():.4f}
Std: {threshold_accs.std():.4f}
Min: {threshold_accs.min():.4f}
Max: {threshold_accs.max():.4f}
Median: {threshold_accs.median():.4f}
Q1: {np.percentile(threshold_accs, 25):.4f}
Q3: {np.percentile(threshold_accs, 75):.4f}"""
    ax_thresh_stats.text(0.1, 0.5, thresh_stats_text, fontsize=10, verticalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Runtime MAE Distribution
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.hist(runtime_maes, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax3.axvline(runtime_maes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {runtime_maes.mean():.4f}')
    ax3.axvline(runtime_maes.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {runtime_maes.median():.4f}')
    ax3.set_xlabel('Runtime MAE', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution of Runtime MAE', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Runtime Stats
    ax_runtime_stats = fig.add_subplot(gs[2, 2])
    ax_runtime_stats.axis('off')
    runtime_stats_text = f"""Runtime MAE Stats:
Mean: {runtime_maes.mean():.4f}
Std: {runtime_maes.std():.4f}
Min: {runtime_maes.min():.4f}
Max: {runtime_maes.max():.4f}
Median: {runtime_maes.median():.4f}
Q1: {np.percentile(runtime_maes, 25):.4f}
Q3: {np.percentile(runtime_maes, 75):.4f}"""
    ax_runtime_stats.text(0.1, 0.5, runtime_stats_text, fontsize=10, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.suptitle(f'Model Performance Distribution ({len(all_results)} runs)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation and visualize distributions")
    parser.add_argument("--model", type=str, default="separate", 
                       choices=["mlp", "xgboost", "separate"],
                       help="Model to evaluate")
    parser.add_argument("--n-runs", type=int, default=100,
                       help="Number of runs")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--save-plot", type=str, default=None,
                       help="Path to save visualization")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Running {args.n_runs} evaluation runs for {args.model.upper()} model...")
    print(f"{'='*60}\n")
    
    # Run evaluation
    output = run_evaluation(args.model, args.n_runs, args.verbose)
    
    # Extract results
    try:
        results_data = extract_results_from_output(output)
        print(f"\n✅ Successfully extracted results from {len(results_data.get('all_results', []))} runs")
        
        # Visualize
        save_path = Path(args.save_plot) if args.save_plot else None
        visualize_distributions(results_data, save_path)
    
    except Exception as e:
        print(f"Error extracting results: {e}")
        print("\nFull output:")
        print(output)


if __name__ == "__main__":
    main()

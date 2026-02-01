#!/usr/bin/env python3
"""
Run model evaluation multiple times and visualize distributions.
"""

import sys
import json
import subprocess
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_single_evaluation(model: str, seed: int, verbose: bool = False):
    """Run a single evaluation with given seed."""
    cmd = [
        sys.executable, "./scripts/evaluate.py",
        "--model", model,
        "--n-runs", "1",
        "--seed", str(seed),
    ]
    if verbose:
        cmd.append("--verbose")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"Seed {seed} failed!")
        print("STDERR:", result.stderr[:500])
        print("STDOUT:", result.stdout[:500])
        return None
    
    # Parse the JSON from output
    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):  # Search from end
        if line.strip().startswith('{'):
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                continue
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Run model multiple times and visualize distributions")
    parser.add_argument("--model", default="separate", choices=["separate", "mlp", "xgboost"])
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--output", default="visualizations/distribution_plot.png")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print(f"Running {args.n_runs} evaluations with model '{args.model}'...")
    
    combined_scores = []
    threshold_scores = []
    runtime_scores = []
    
    for i in range(args.n_runs):
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{args.n_runs}...")
        
        result = run_single_evaluation(args.model, seed=1000 + i, verbose=args.verbose)
        
        if result and "metrics" in result:
            metrics = result["metrics"]
            combined_scores.append(metrics.get("combined_score", 0))
            threshold_scores.append(metrics.get("threshold_score", 0))
            runtime_scores.append(metrics.get("runtime_score", 0))
    
    print(f"✓ Collected {len(combined_scores)} results")
    
    if len(combined_scores) == 0:
        print("❌ No results collected!")
        return
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    metrics_list = [
        ("Combined Score", combined_scores),
        ("Threshold Score", threshold_scores),
        ("Runtime Score", runtime_scores),
    ]
    
    for idx, (title, values) in enumerate(metrics_list):
        ax = fig.add_subplot(gs[idx, :])
        
        ax.hist(values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_title(f"{title} Distribution ({len(values)} runs)", fontsize=12, fontweight='bold')
        ax.set_xlabel(title)
        ax.set_ylabel("Frequency")
        
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        stats_text = f"Mean: {mean_val:.4f} | Median: {median_val:.4f}\nStd: {std_val:.4f} | Min: {min_val:.4f} | Max: {max_val:.4f}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for title, values in metrics_list:
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        print(f"{title:20} | Mean: {mean_val:8.4f} | Median: {median_val:8.4f} | Std: {std_val:8.4f}")


if __name__ == "__main__":
    main()

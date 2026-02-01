#!/usr/bin/env python3
"""Plot distributions of mirror run_wall_s and forward_wall_s, and correlations with circuit size and n_qubits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_hackathon_data, extract_qasm_features


def main():
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    circuits, results = load_hackathon_data(data_path)
    results = [r for r in results if r.status == "ok"]
    circuit_info = {c.file: c for c in circuits}

    feature_cache = {}
    for c in circuits:
        qasm_path = circuits_dir / c.file
        if qasm_path.exists():
            feature_cache[c.file] = extract_qasm_features(qasm_path)

    mirror_wall_s = []
    mirror_n_qubits = []
    mirror_circuit_size = []
    for r in results:
        info = circuit_info.get(r.file)
        feats = feature_cache.get(r.file, {})
        nq = info.n_qubits if info else feats.get("n_qubits", 0)
        depth = feats.get("estimated_depth", 0)
        for entry in r.threshold_sweep:
            if entry.run_wall_s is not None and entry.run_wall_s > 0:
                mirror_wall_s.append(entry.run_wall_s)
                mirror_n_qubits.append(nq)
                mirror_circuit_size.append(depth)

    forward_wall_s = []
    forward_n_qubits = []
    forward_circuit_size = []
    for r in results:
        if r.forward_wall_s is None or r.forward_wall_s <= 0:
            continue
        info = circuit_info.get(r.file)
        feats = feature_cache.get(r.file, {})
        nq = info.n_qubits if info else feats.get("n_qubits", 0)
        depth = feats.get("estimated_depth", 0)
        forward_wall_s.append(r.forward_wall_s)
        forward_n_qubits.append(nq)
        forward_circuit_size.append(depth)

    out_dir = project_root / "visualizations"
    out_dir.mkdir(exist_ok=True)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, values, label, color in [
        (ax1, mirror_wall_s, "Mirror run_wall_s\n(per threshold_sweep entry)", "steelblue"),
        (ax2, forward_wall_s, "Forward run_wall_s\n(one per result, at selected threshold)", "coral"),
    ]:
        if not values:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        log_vals = np.log10(np.array(values))
        ax.hist(log_vals, bins=40, edgecolor="black", alpha=0.7, color=color)
        ax.set_xlabel("Log10(wall time [s])")
        ax.set_ylabel("Count")
        ax.set_title(label)
        med = np.median(log_vals)
        ax.axvline(med, color="red", linestyle="--", label=f"Median: 10^{med:.2f} s")
        ax.legend()
    fig1.suptitle(
        "Mirror: one value per threshold rung  |  Forward: one value per result (selected threshold only)",
        fontsize=10,
        style="italic",
    )
    plt.tight_layout()
    fig1.savefig(out_dir / "10_mirror_vs_forward_wall_s.png", dpi=150)
    plt.close(fig1)
    print(f"Saved: {out_dir / '10_mirror_vs_forward_wall_s.png'}")
    print(f"Mirror: {len(mirror_wall_s)} values; Forward: {len(forward_wall_s)} values.")

    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    circuit_size_label = "Circuit size (estimated depth)"

    for row, (wall_s, n_qubits, circuit_size, title_prefix, color) in enumerate([
        (mirror_wall_s, mirror_n_qubits, mirror_circuit_size, "Mirror", "steelblue"),
        (forward_wall_s, forward_n_qubits, forward_circuit_size, "Forward", "coral"),
    ]):
        ax_size, ax_nq = axes[row, 0], axes[row, 1]
        if not wall_s:
            ax_size.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_size.transAxes)
            ax_nq.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_nq.transAxes)
            ax_size.set_title(f"{title_prefix} run_wall_s vs {circuit_size_label}")
            ax_nq.set_title(f"{title_prefix} run_wall_s vs n_qubits")
            continue
        wall_s = np.array(wall_s)
        n_qubits = np.array(n_qubits)
        circuit_size = np.array(circuit_size)
        ax_size.scatter(circuit_size, wall_s, alpha=0.3, s=8, c=color)
        ax_size.set_yscale("log")
        ax_size.set_xlabel(circuit_size_label)
        ax_size.set_ylabel("Wall time [s]")
        ax_size.set_title(f"{title_prefix} run_wall_s vs {circuit_size_label}")
        ax_nq.scatter(n_qubits, wall_s, alpha=0.3, s=8, c=color)
        ax_nq.set_yscale("log")
        ax_nq.set_xlabel("Number of qubits")
        ax_nq.set_ylabel("Wall time [s]")
        ax_nq.set_title(f"{title_prefix} run_wall_s vs n_qubits")

    plt.tight_layout()
    fig2.savefig(out_dir / "11_wall_s_vs_circuit_size_and_n_qubits.png", dpi=150)
    plt.close(fig2)
    print(f"Saved: {out_dir / '11_wall_s_vs_circuit_size_and_n_qubits.png'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Graph-based circuit visualizer: qubits as nodes, gates as edges.

Converts OpenQASM circuits into graph form and visualizes:
- 2D: single-layer interaction graph (all gates aggregated); qubits on a circle.
- 3D: time as third dimension; vertical lines = qubits over time, edges at each
  time slice show gate connectivity evolving.
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


GATE_COLORS = {
    "cx": "#e63946",
    "cz": "#457b9d",
    "swap": "#2a9d8f",
    "cp": "#f4a261",
    "crz": "#e9c46a",
    "crx": "#a8dadc",
    "cry": "#264653",
    "cu1": "#9c89b8",
    "cu3": "#f4acb7",
    "rxx": "#6d597a",
    "ryy": "#b56576",
    "rzz": "#e56b6f",
    "ccx": "#8338ec",
    "cswap": "#3a86ff",
    "h": "#e94f37",
    "x": "#ff6b6b",
    "y": "#4ecdc4",
    "z": "#45b7d1",
    "rx": "#f7b731",
    "ry": "#26de81",
    "rz": "#a55eea",
    "s": "#fd9644",
    "t": "#20bf6b",
}
DEFAULT_GATE_COLOR = "#888888"


def get_gate_color(gate_type: str) -> str:
    return GATE_COLORS.get(gate_type, DEFAULT_GATE_COLOR)


def parse_qasm(qasm_path: Path) -> Tuple[int, List[Tuple[str, List[int], List[float]]]]:
    """Parse QASM into n_qubits and list of (gate_type, qubits, params). Skips barrier, measure, comments."""
    text = qasm_path.read_text(encoding="utf-8")
    n_qubits = 0
    qreg_match = re.search(r"qreg\s+\w+\[(\d+)\]", text)
    if qreg_match:
        n_qubits = int(qreg_match.group(1))

    gate_line = re.compile(
        r"\b([a-zA-Z0-9]+)\s*"
        r"(?:\(([^)]*)\))?\s*"
        r"([^;]+);"
    )
    qubit_ref = re.compile(r"\w+\[(\d+)\]")

    gates = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM") or \
           line.startswith("include") or line.startswith("qreg") or \
           line.startswith("creg") or line.startswith("barrier") or \
           line.startswith("measure"):
            continue
        match = gate_line.search(line)
        if not match:
            continue
        gate_type = match.group(1).lower()
        param_str = (match.group(2) or "").strip()
        qubit_str = match.group(3)

        params = []
        if param_str:
            for p in param_str.split(","):
                p = p.strip()
                try:
                    if "pi" in p:
                        p = p.replace("pi", str(math.pi))
                        params.append(float(eval(p)))
                    else:
                        params.append(float(p))
                except Exception:
                    params.append(0.0)

        qubits = [int(m.group(1)) for m in qubit_ref.finditer(qubit_str)]
        if qubits:
            gates.append((gate_type, qubits, params))

    return n_qubits, gates


def assign_layers(
    gates: List[Tuple[str, List[int], List[float]]],
) -> List[List[Tuple[str, List[int], List[float]]]]:
    """Assign each gate to the earliest layer with no qubit overlap."""
    if not gates:
        return []
    layers = []
    for gate_type, qubits, params in gates:
        gate_qubits = set(qubits)
        placed = False
        for layer in layers:
            if not any(gate_qubits & set(qs) for _, qs, _ in layer):
                layer.append((gate_type, qubits, params))
                placed = True
                break
        if not placed:
            layers.append([(gate_type, qubits, params)])
    return layers


def build_aggregate_graph_by_gate(
    gates: List[Tuple[str, List[int], List[float]]],
) -> Tuple[Set[int], Dict[Tuple[int, int], Dict[str, int]]]:
    """
    Build single-layer graph with gate type tracking.
    Returns nodes and edges where each edge maps to {gate_type: count}.
    """
    nodes = set()
    edges: Dict[Tuple[int, int], Dict[str, int]] = {}
    for gt, qubits, _ in gates:
        for q in qubits:
            nodes.add(q)
        if len(qubits) >= 2:
            for i in range(len(qubits)):
                for j in range(i + 1, len(qubits)):
                    iq, jq = qubits[i], qubits[j]
                    key = (min(iq, jq), max(iq, jq))
                    if key not in edges:
                        edges[key] = {}
                    edges[key][gt] = edges[key].get(gt, 0) + 1
    return nodes, edges


def circular_layout(n: int, radius: float = 1.0) -> np.ndarray:
    """Return (n, 2) array of positions on a circle."""
    if n == 0:
        return np.zeros((0, 2))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def draw_2d(
    n_qubits: int,
    nodes: Set[int],
    edges: Dict[Tuple[int, int], Dict[str, int]],
    ax: plt.Axes,
    *,
    node_size: float = 150,
    edge_width_base: float = 2.0,
    edge_width_scale: float = 2.0,
) -> Set[str]:
    """Draw aggregate qubit graph in 2D. Returns set of gate types used."""
    pos = circular_layout(n_qubits)
    ax.set_aspect("equal")
    ax.set_axis_off()

    gate_types_used = set()
    max_weight = max((sum(gt_counts.values()) for gt_counts in edges.values()), default=1)

    for (i, j), gt_counts in edges.items():
        total_weight = sum(gt_counts.values())
        dominant_gate = max(gt_counts.keys(), key=lambda g: gt_counts[g])
        gate_types_used.add(dominant_gate)

        x = [pos[i, 0], pos[j, 0]]
        y = [pos[i, 1], pos[j, 1]]
        lw = edge_width_base + edge_width_scale * (total_weight / max_weight)
        alpha = 0.5 + 0.4 * (total_weight / max_weight)
        color = get_gate_color(dominant_gate)
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=0, solid_capstyle="round")

    ax.scatter(
        pos[:, 0], pos[:, 1],
        s=node_size, c="#2e86ab", edgecolors="#1a1a1a", linewidths=1.5, zorder=1
    )
    return gate_types_used


def draw_3d(
    n_qubits: int,
    layers: List[List[Tuple[str, List[int], List[float]]]],
    ax: plt.Axes,
    *,
    radius: float = 1.0,
    node_size: float = 30,
    edge_lw: float = 2.5,
    qubit_line_lw: float = 1.0,
    edge_alpha: float = 0.8,
) -> Set[str]:
    """
    Draw temporal qubit graph in 3D: x = time (layer index), y/z = qubit position.
    Returns set of gate types used.
    """
    pos_2d = circular_layout(n_qubits, radius=radius)
    T = len(layers)
    x_max = max(T - 1, 1)

    ax.set_xlim(-0.5, x_max + 0.5)
    ax.set_ylim(-radius * 1.3, radius * 1.3)
    ax.set_zlim(-radius * 1.3, radius * 1.3)

    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)

    for q in range(n_qubits):
        ax.plot(
            [0, x_max],
            [pos_2d[q, 0], pos_2d[q, 0]],
            [pos_2d[q, 1], pos_2d[q, 1]],
            color="#888888",
            lw=qubit_line_lw,
            alpha=0.4,
        )

    gate_types_used = set()

    for t, layer in enumerate(layers):
        for gt, qubits, _ in layer:
            color = get_gate_color(gt)
            gate_types_used.add(gt)

            if len(qubits) == 1:
                q = qubits[0]
                ax.scatter(
                    [t],
                    [pos_2d[q, 0]],
                    [pos_2d[q, 1]],
                    s=node_size * 2,
                    c=color,
                    edgecolors="#222",
                    linewidths=0.6,
                    alpha=0.9,
                    depthshade=False,
                )
            else:
                for i in range(len(qubits)):
                    for j in range(i + 1, len(qubits)):
                        qa, qb = qubits[i], qubits[j]
                        ax.plot(
                            [t, t],
                            [pos_2d[qa, 0], pos_2d[qb, 0]],
                            [pos_2d[qa, 1], pos_2d[qb, 1]],
                            color=color,
                            lw=edge_lw,
                            alpha=edge_alpha,
                            solid_capstyle="round",
                        )

    for q in range(n_qubits):
        ax.scatter(
            [0],
            [pos_2d[q, 0]],
            [pos_2d[q, 1]],
            s=node_size,
            c="#2e86ab",
            edgecolors="#1a1a1a",
            linewidths=0.8,
            depthshade=False,
        )

    ax.view_init(elev=20, azim=-60)

    return gate_types_used


def add_legend(
    ax,
    gate_types: Set[str],
    loc: str = "upper right",
    fontsize: int = 9,
    include_qubit: bool = False,
) -> None:
    """Add color-coded gate legend to axes."""
    handles = []
    if include_qubit:
        qubit_marker = mlines.Line2D(
            [], [], color="#2e86ab", marker="o", markersize=8,
            linestyle="None", markeredgecolor="#1a1a1a", label="Qubit"
        )
        handles.append(qubit_marker)
    for gt in sorted(gate_types):
        color = get_gate_color(gt)
        label = gt.upper()
        line = mlines.Line2D([], [], color=color, lw=3, label=label)
        handles.append(line)
    if handles:
        ax.legend(handles=handles, loc=loc, fontsize=fontsize, framealpha=0.9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize circuits as graphs: nodes = qubits, edges = gates."
    )
    parser.add_argument(
        "circuits_dir",
        type=Path,
        nargs="?",
        default=Path("circuits"),
        help="Directory containing .qasm files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("figures/graph_circuits"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "-f", "--files",
        nargs="*",
        help="Specific .qasm files (default: all in circuits_dir)",
    )
    parser.add_argument(
        "--mode",
        choices=["2d", "3d", "both"],
        default="both",
        help="2D single-layer graph, 3D time-evolving graph, or both",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=32,
        help="Skip circuits with more qubits",
    )
    parser.add_argument(
        "--max-gates",
        type=int,
        default=800,
        help="Skip circuits with more gates (affects 3D layer count)",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "both"],
        default="both",
        help="Output image format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output",
    )
    args = parser.parse_args()

    circuits_dir = args.circuits_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        qasm_paths = []
        for f in args.files:
            p = Path(f)
            qasm_paths.append(p if p.exists() else circuits_dir / f)
    else:
        qasm_paths = sorted(circuits_dir.glob("*.qasm"))

    for qasm_path in qasm_paths:
        if not qasm_path.exists():
            print(f"Skipping (not found): {qasm_path}")
            continue
        try:
            n_qubits, gates = parse_qasm(qasm_path)
        except Exception as e:
            print(f"Skipping {qasm_path}: {e}")
            continue

        if n_qubits == 0:
            print(f"Skipping {qasm_path}: no qubits")
            continue
        if n_qubits > args.max_qubits:
            print(f"Skipping {qasm_path}: {n_qubits} qubits > --max-qubits")
            continue
        if len(gates) > args.max_gates:
            print(f"Skipping {qasm_path}: {len(gates)} gates > --max-gates")
            continue

        layers = assign_layers(gates)
        nodes, edges = build_aggregate_graph_by_gate(gates)
        base = output_dir / qasm_path.stem
        title = qasm_path.stem.replace("_", " ")

        exts = ["pdf"] if args.format == "pdf" else ["png"] if args.format == "png" else ["pdf", "png"]
        written = []

        if args.mode in ("2d", "both"):
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            gate_types = draw_2d(n_qubits, nodes, edges, ax)
            add_legend(ax, gate_types, loc="upper right", fontsize=10, include_qubit=True)
            ax.set_title(f"{title} graph connectivity", fontsize=14, fontweight="medium")
            fig.tight_layout()
            for ext in exts:
                path = f"{base}_2d.{ext}"
                fig.savefig(path, bbox_inches="tight", pad_inches=0.08, dpi=args.dpi)
                written.append(path)
            plt.close(fig)

        if args.mode in ("3d", "both"):
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection="3d")
            gate_types = draw_3d(n_qubits, layers, ax)
            add_legend(fig.axes[0], gate_types, loc="upper left", fontsize=10, include_qubit=True)
            ax.set_title(f"{title} graph connectivity", fontsize=14, fontweight="medium", pad=0)
            fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0.02)
            for ext in exts:
                path = f"{base}_3d.{ext}"
                fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=args.dpi)
                written.append(path)
            plt.close(fig)

        for path in written:
            print(f"Wrote {path}")

    print(f"Done. Figures in {output_dir}")


if __name__ == "__main__":
    main()

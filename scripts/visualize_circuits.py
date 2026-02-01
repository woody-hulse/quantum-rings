#!/usr/bin/env python3
"""
Publication-quality quantum circuit visualizer.

Reads OpenQASM files and produces one figure per circuit with qubits as
horizontal lines and gates drawn in standard notation (H, X, CNOT, etc.).
Output is suitable for academic papers: vector PDF, clean typography, optional grayscale.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams


def parse_qasm(qasm_path: Path) -> Tuple[int, List[Tuple[str, List[int], List[float]]]]:
    """
    Parse QASM file into n_qubits and list of (gate_type, qubits, params).
    Skips barrier, measure, comments, and declarations.
    """
    text = qasm_path.read_text(encoding="utf-8")
    n_qubits = 0
    qreg_match = re.search(r"qreg\s+\w+\[(\d+)\]", text)
    if qreg_match:
        n_qubits = int(qreg_match.group(1))

    gate_line = re.compile(
        r"\b([a-zA-Z0-9]+)\s*"           # gate name
        r"(?:\(([^)]*)\))?\s*"            # optional (params)
        r"([^;]+);"                       # qubit args
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
                        import math
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


def assign_layers(gates: List[Tuple[str, List[int], List[float]]]) -> List[List[Tuple[str, List[int], List[float]]]]:
    """
    Assign each gate to the earliest layer such that no two gates in the same
    layer act on the same qubit. Gates are processed in circuit order.
    """
    if not gates:
        return []
    layers = []

    for gate_type, qubits, params in gates:
        gate_qubits = set(qubits)
        placed = False
        for layer_idx, layer in enumerate(layers):
            if not any(gate_qubits & set(qs) for _, qs, _ in layer):
                layer.append((gate_type, qubits, params))
                placed = True
                break
        if not placed:
            layers.append([(gate_type, qubits, params)])
    return layers


def draw_circuit(
    n_qubits: int,
    layers: List[List[Tuple[str, List[int], List[float]]]],
    ax: plt.Axes,
    *,
    qubit_height: float = 0.8,
    gate_width: float = 0.5,
    line_color: str = "#333333",
    gate_fill: str = "#f0f0f0",
    gate_edge: str = "#333333",
    font_size: float = 9,
    show_qubit_labels: bool = True,
) -> None:
    """
    Draw circuit on axes. Qubit 0 at top, qubit n-1 at bottom.
    """
    ax.set_aspect("equal")
    ax.set_axis_off()

    n_layers = len(layers)
    if n_layers == 0:
        n_layers = 1
    width = n_layers * gate_width * 2
    height = (n_qubits - 1) * qubit_height if n_qubits > 1 else qubit_height

    ax.set_xlim(-gate_width, width + gate_width)
    ax.set_ylim(-0.5, height + 0.5)

    def qubit_y(q: int) -> float:
        return (n_qubits - 1 - q) * qubit_height

    for q in range(n_qubits):
        y = qubit_y(q)
        ax.plot([0, width], [y, y], color=line_color, linewidth=1.2, zorder=0)
        if show_qubit_labels:
            ax.text(-gate_width * 0.6, y, f"$|q_{q}\\rangle$", ha="right", va="center", fontsize=font_size - 1)

    box_w = gate_width * 1.4
    box_h = qubit_height * 0.55

    for layer_idx, layer in enumerate(layers):
        x_center = gate_width + layer_idx * gate_width * 2

        for gate_type, qubits, params in layer:
            if len(qubits) == 1:
                q = qubits[0]
                y = qubit_y(q)
                label = _gate_label_1q(gate_type, params)
                rect = mpatches.FancyBboxPatch(
                    (x_center - box_w / 2, y - box_h / 2), box_w, box_h,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    facecolor=gate_fill, edgecolor=gate_edge, linewidth=1.0,
                )
                ax.add_patch(rect)
                ax.text(x_center, y, label, ha="center", va="center", fontsize=font_size - 1, fontweight="normal")

            elif len(qubits) == 2:
                q0, q1 = min(qubits), max(qubits)
                y0, y1 = qubit_y(q0), qubit_y(q1)
                mid_y = (y0 + y1) / 2
                ax.plot([x_center, x_center], [y0, y1], color=line_color, linewidth=1.2, zorder=0)
                if gate_type == "cx":
                    ax.add_patch(mpatches.Circle((x_center, y0), gate_width * 0.15, color=line_color, fill=True, zorder=1))
                    ax.add_patch(mpatches.Circle((x_center, y1), gate_width * 0.2, facecolor="none", edgecolor=line_color, linewidth=1.2, zorder=1))
                    ax.plot([x_center - gate_width * 0.2, x_center + gate_width * 0.2], [y1, y1], color=line_color, linewidth=1.2, zorder=1)
                elif gate_type == "cz":
                    ax.add_patch(mpatches.Circle((x_center, y0), gate_width * 0.12, color=line_color, fill=True, zorder=1))
                    _draw_z_on_line(ax, x_center, y1, gate_width * 0.25, line_color)
                elif gate_type == "swap":
                    _draw_swap(ax, x_center, y0, y1, gate_width, line_color)
                else:
                    ax.add_patch(mpatches.Circle((x_center, y0), gate_width * 0.12, color=line_color, fill=True, zorder=1))
                    label = _gate_label_2q(gate_type, params)
                    rect = mpatches.FancyBboxPatch(
                        (x_center - box_w / 2, y1 - box_h / 2), box_w, box_h,
                        boxstyle="round,pad=0.02,rounding_size=0.05",
                        facecolor=gate_fill, edgecolor=gate_edge, linewidth=1.0,
                    )
                    ax.add_patch(rect)
                    ax.text(x_center, y1, label, ha="center", va="center", fontsize=font_size - 2, fontweight="normal")

            elif len(qubits) >= 3:
                ys = [qubit_y(q) for q in qubits]
                y_min, y_max = min(ys), max(ys)
                ax.plot([x_center, x_center], [y_min, y_max], color=line_color, linewidth=1.2, zorder=0)
                for i, y in enumerate(ys):
                    if i < len(ys) - 1:
                        ax.add_patch(mpatches.Circle((x_center, y), gate_width * 0.12, color=line_color, fill=True, zorder=1))
                    else:
                        ax.add_patch(mpatches.Circle((x_center, y), gate_width * 0.2, facecolor="none", edgecolor=line_color, linewidth=1.2, zorder=1))
                        ax.plot([x_center - gate_width * 0.2, x_center + gate_width * 0.2], [y, y], color=line_color, linewidth=1.2, zorder=1)
                mid_y = (y_min + y_max) / 2
                label = _gate_label_3q(gate_type)
                ax.text(x_center + gate_width * 0.35, mid_y, label, ha="left", va="center", fontsize=font_size - 2)


def _gate_label_1q(gate_type: str, params: List[float]) -> str:
    if gate_type in ("h", "x", "y", "z", "s", "t", "id"):
        return gate_type.upper() if gate_type != "id" else "I"
    if gate_type in ("sdg", "tdg"):
        return gate_type.upper()[:2] + "†"
    if gate_type in ("rx", "ry", "rz"):
        θ = params[0] if params else 0
        if abs(θ - 3.141592653589793) < 1e-6:
            return f"{gate_type.upper()}(π)"
        return f"{gate_type.upper()}(θ)" if params else gate_type.upper()
    if gate_type in ("u1", "u2", "u3"):
        return f"U{len(params)}" if params else "U"
    return gate_type.upper()


def _gate_label_2q(gate_type: str, params: List[float]) -> str:
    if gate_type in ("cp", "cu1", "crx", "cry", "crz", "cu3"):
        return f"{gate_type.upper()}(θ)" if params else gate_type.upper()
    if gate_type in ("rxx", "ryy", "rzz"):
        return gate_type.upper()
    return gate_type.upper()


def _gate_label_3q(gate_type: str) -> str:
    if gate_type == "ccx":
        return "Toffoli"
    if gate_type == "cswap":
        return "Fredkin"
    return gate_type.upper()


def _draw_z_on_line(ax: plt.Axes, x: float, y: float, r: float, color: str) -> None:
    ax.add_patch(mpatches.Circle((x, y), r, facecolor="none", edgecolor=color, linewidth=1.2, zorder=1))
    ax.plot([x - r * 0.6, x + r * 0.6], [y, y], color=color, linewidth=1.0, zorder=1)


def _draw_swap(ax: plt.Axes, x: float, y0: float, y1: float, gate_width: float, color: str) -> None:
    d = gate_width * 0.22
    for y in (y0, y1):
        ax.plot([x - d, x + d], [y - d, y + d], color=color, linewidth=1.2, zorder=1)
        ax.plot([x - d, x + d], [y + d, y - d], color=color, linewidth=1.2, zorder=1)
    ax.plot([x, x], [y0, y1], color=color, linewidth=1.2, zorder=0)


def paper_style() -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times", "serif"]
    rcParams["font.size"] = 10
    rcParams["axes.linewidth"] = 0.8
    rcParams["figure.dpi"] = 150


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize quantum circuits for publication.")
    parser.add_argument(
        "circuits_dir",
        type=Path,
        default=Path("circuits"),
        nargs="?",
        help="Directory containing .qasm files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("figures/circuits"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "-f", "--files",
        nargs="*",
        help="Specific .qasm files to plot (default: all in circuits_dir)",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=32,
        help="Skip circuits with more than this many qubits (default: 32)",
    )
    parser.add_argument(
        "--max-gates",
        type=int,
        default=200,
        help="Skip circuits with more gates than this (default: 200)",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Use grayscale-friendly colors",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Omit qubit labels |q0⟩ etc.",
    )
    args = parser.parse_args()

    circuits_dir = args.circuits_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        qasm_paths = []
        for f in args.files:
            p = Path(f)
            if p.exists():
                qasm_paths.append(p)
            else:
                qasm_paths.append(circuits_dir / f)
    else:
        qasm_paths = sorted(circuits_dir.glob("*.qasm"))

    paper_style()
    line_color = "#333333"
    gate_fill = "#f8f8f8"
    gate_edge = "#333333"
    if args.grayscale:
        line_color = "#1a1a1a"
        gate_fill = "#e8e8e8"
        gate_edge = "#1a1a1a"

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
            print(f"Skipping {qasm_path}: {n_qubits} qubits > --max-qubits {args.max_qubits}")
            continue
        if len(gates) > args.max_gates:
            print(f"Skipping {qasm_path}: {len(gates)} gates > --max-gates {args.max_gates}")
            continue

        layers = assign_layers(gates)
        fig, ax = plt.subplots(1, 1, figsize=(max(6, len(layers) * 0.15), max(3, n_qubits * 0.35)))
        draw_circuit(
            n_qubits,
            layers,
            ax,
            qubit_height=0.8,
            gate_width=0.5,
            line_color=line_color,
            gate_fill=gate_fill,
            gate_edge=gate_edge,
            show_qubit_labels=not args.no_labels,
        )
        title = qasm_path.stem.replace("_", " ")
        ax.set_title(title, fontsize=11)
        fig.tight_layout()

        base = output_dir / qasm_path.stem
        written = []
        if args.format in ("pdf", "both"):
            fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.05)
            written.append("pdf")
        if args.format in ("png", "both"):
            fig.savefig(f"{base}.png", bbox_inches="tight", pad_inches=0.05, dpi=200)
            written.append("png")
        plt.close(fig)
        print("Wrote " + ", ".join(f"{base}.{ext}" for ext in written))

    print(f"Done. Figures in {output_dir}")


if __name__ == "__main__":
    main()

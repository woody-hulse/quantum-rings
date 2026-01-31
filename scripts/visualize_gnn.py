#!/usr/bin/env python3
"""
Beautiful visualization of GNN forward pass on quantum circuits.

Creates an animated GIF showing:
- Qubits as nodes arranged in a circle/line
- Gates as edges with colors indicating gate type
- Message passing computation flowing through the network
- Node activations changing over time
"""

import sys
from pathlib import Path
import argparse
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn.graph_builder import (
    parse_qasm, build_graph_from_qasm, 
    GATE_1Q, GATE_2Q, GATE_3Q, GATE_TO_IDX
)
from gnn.model import create_gnn_model
from gnn.dataset import GLOBAL_FEAT_DIM


GATE_COLORS = {
    'h': '#FF6B6B',      # Red - Hadamard
    'x': '#4ECDC4',      # Teal - Pauli X
    'y': '#45B7D1',      # Blue - Pauli Y
    'z': '#96CEB4',      # Green - Pauli Z
    'cx': '#DDA0DD',     # Plum - CNOT
    'cz': '#98D8C8',     # Mint - CZ
    'swap': '#F7DC6F',   # Yellow - SWAP
    'rz': '#BB8FCE',     # Purple - RZ
    'rx': '#F1948A',     # Light red - RX
    'ry': '#85C1E9',     # Light blue - RY
    't': '#FAD7A0',      # Peach - T gate
    's': '#A9DFBF',      # Light green - S gate
    'cp': '#D7BDE2',     # Lavender - Controlled phase
    'ccx': '#E59866',    # Orange - Toffoli
    'default': '#AEB6BF' # Gray - default
}


def get_gate_color(gate_type: str) -> str:
    """Get color for a gate type."""
    return GATE_COLORS.get(gate_type, GATE_COLORS['default'])


def create_circuit_layout(n_qubits: int, layout: str = 'circular') -> np.ndarray:
    """Create node positions for the circuit visualization."""
    if layout == 'circular':
        angles = np.linspace(0, 2 * np.pi, n_qubits, endpoint=False)
        angles = angles - np.pi / 2  # Start from top
        positions = np.column_stack([np.cos(angles), np.sin(angles)])
    elif layout == 'linear':
        positions = np.column_stack([
            np.zeros(n_qubits),
            np.linspace(1, -1, n_qubits)
        ])
    elif layout == 'grid':
        cols = int(np.ceil(np.sqrt(n_qubits)))
        rows = int(np.ceil(n_qubits / cols))
        positions = []
        for i in range(n_qubits):
            row = i // cols
            col = i % cols
            positions.append([col - cols/2 + 0.5, rows/2 - row - 0.5])
        positions = np.array(positions)
    else:
        raise ValueError(f"Unknown layout: {layout}")
    
    return positions


def extract_edge_info(qasm_text: str, n_qubits: int) -> List[Dict]:
    """Extract edge information with gate types and temporal ordering."""
    _, gates = parse_qasm(qasm_text)
    
    edges = []
    for gate in gates:
        gate_type = gate.gate_type
        
        if gate_type in GATE_1Q:
            if gate.qubits and gate.qubits[0] < n_qubits:
                edges.append({
                    'src': gate.qubits[0],
                    'dst': gate.qubits[0],
                    'gate_type': gate_type,
                    'position': gate.position,
                    'is_self_loop': True,
                })
        
        elif gate_type in GATE_2Q:
            if len(gate.qubits) >= 2:
                q1, q2 = gate.qubits[0], gate.qubits[1]
                if q1 < n_qubits and q2 < n_qubits:
                    edges.append({
                        'src': q1,
                        'dst': q2,
                        'gate_type': gate_type,
                        'position': gate.position,
                        'is_self_loop': False,
                    })
        
        elif gate_type in GATE_3Q:
            if len(gate.qubits) >= 3:
                q1, q2, q3 = gate.qubits[0], gate.qubits[1], gate.qubits[2]
                for q_ctrl in [q1, q2]:
                    if q_ctrl < n_qubits and q3 < n_qubits:
                        edges.append({
                            'src': q_ctrl,
                            'dst': q3,
                            'gate_type': gate_type,
                            'position': gate.position,
                            'is_self_loop': False,
                        })
    
    return edges


class GNNVisualizer:
    """Visualizer for GNN forward pass animation."""
    
    def __init__(
        self,
        qasm_text: str,
        model: torch.nn.Module,
        figsize: Tuple[int, int] = (14, 10),
        layout: str = 'circular',
        fps: int = 10,
        dpi: int = 100,
    ):
        self.qasm_text = qasm_text
        self.model = model
        self.figsize = figsize
        self.layout = layout
        self.fps = fps
        self.dpi = dpi
        
        # Parse circuit
        self.n_qubits, self.gates = parse_qasm(qasm_text)
        self.edges = extract_edge_info(qasm_text, self.n_qubits)
        
        # Build graph
        self.graph_data = build_graph_from_qasm(qasm_text, "CPU", "single")
        
        # Create layout
        self.positions = create_circuit_layout(self.n_qubits, layout)
        
        # Run forward pass and capture intermediate states
        self.node_states = self._capture_forward_pass()
        
        # Setup figure
        self.fig, self.axes = plt.subplots(1, 2, figsize=figsize)
        self.fig.patch.set_facecolor('#1a1a2e')
        
    def _capture_forward_pass(self) -> List[np.ndarray]:
        """Run forward pass and capture node states at each layer."""
        self.model.eval()
        
        x = self.graph_data["x"]
        edge_index = self.graph_data["edge_index"]
        edge_attr = self.graph_data["edge_attr"]
        edge_gate_type = self.graph_data["edge_gate_type"]
        global_features = self.graph_data["global_features"]
        
        # Add batch dimension
        batch = torch.zeros(x.shape[0], dtype=torch.long)
        
        states = []
        
        # Initial state (normalized input features)
        x_norm = (x - x.mean()) / (x.std() + 1e-6)
        states.append(x_norm.mean(dim=1).detach().numpy())
        
        # Get intermediate states through the network
        with torch.no_grad():
            h = self.model.node_embed(x)
            states.append(h.mean(dim=1).detach().numpy())
            
            for i, mp_layer in enumerate(self.model.mp_layers):
                h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
                h = h + h_new
                states.append(h.mean(dim=1).detach().numpy())
        
        return states
    
    def _draw_static_elements(self, ax):
        """Draw static circuit elements (nodes, edges, labels)."""
        ax.set_facecolor('#16213e')
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw edges first (so nodes are on top)
        edge_artists = []
        for edge in self.edges:
            if edge['is_self_loop']:
                continue
            
            src, dst = edge['src'], edge['dst']
            x1, y1 = self.positions[src]
            x2, y2 = self.positions[dst]
            
            color = get_gate_color(edge['gate_type'])
            
            # Draw curved edge
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Offset midpoint perpendicular to the line
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                perp_x = -dy / length * 0.15
                perp_y = dx / length * 0.15
                mid_x += perp_x
                mid_y += perp_y
            
            # Draw with arrow
            arrow = mpatches.FancyArrowPatch(
                (x1, y1), (x2, y2),
                connectionstyle=f"arc3,rad=0.2",
                arrowstyle='-|>',
                mutation_scale=15,
                color=color,
                alpha=0.6,
                linewidth=2,
            )
            ax.add_patch(arrow)
            edge_artists.append(arrow)
        
        return edge_artists
    
    def _create_legend(self, ax):
        """Create a legend for gate types."""
        ax.set_facecolor('#16213e')
        ax.axis('off')
        
        # Find unique gate types in the circuit
        gate_types = set(e['gate_type'] for e in self.edges)
        
        # Create legend patches
        patches = []
        for gate_type in sorted(gate_types):
            color = get_gate_color(gate_type)
            patch = mpatches.Patch(color=color, label=gate_type.upper())
            patches.append(patch)
        
        ax.legend(
            handles=patches,
            loc='upper left',
            fontsize=10,
            facecolor='#1a1a2e',
            edgecolor='white',
            labelcolor='white',
            title='Gate Types',
            title_fontsize=12,
        )
        ax.set_title('Legend', color='white', fontsize=14, pad=10)
    
    def create_animation(self, output_path: str, n_frames_per_layer: int = 15):
        """Create the animated GIF."""
        ax_main = self.axes[0]
        ax_info = self.axes[1]
        
        # Draw static edges
        self._draw_static_elements(ax_main)
        self._create_legend(ax_info)
        
        # Compute total frames
        n_states = len(self.node_states)
        total_frames = (n_states - 1) * n_frames_per_layer + n_frames_per_layer
        
        # Create colormap for node activations
        cmap = LinearSegmentedColormap.from_list(
            'activation', ['#0f3460', '#e94560', '#ffffff']
        )
        
        # Initialize node circles
        node_circles = []
        node_labels = []
        for i in range(self.n_qubits):
            x, y = self.positions[i]
            circle = plt.Circle(
                (x, y), 0.12, 
                facecolor='#0f3460',
                edgecolor='white',
                linewidth=2,
                zorder=10,
            )
            ax_main.add_patch(circle)
            node_circles.append(circle)
            
            # Add qubit label
            label = ax_main.text(
                x, y, f'q{i}',
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='white',
                zorder=11,
            )
            node_labels.append(label)
        
        # Add title and layer indicator
        title = ax_main.set_title(
            'Quantum Circuit GNN Forward Pass',
            color='white', fontsize=16, pad=20,
            fontweight='bold',
        )
        
        layer_text = ax_main.text(
            0, -1.5, 'Layer: Input',
            ha='center', va='center',
            fontsize=14, color='#e94560',
            fontweight='bold',
        )
        
        # Add info panel elements
        info_elements = {}
        
        # Circuit info
        ax_info.text(
            0.5, 0.95, f'Circuit: {self.n_qubits} qubits',
            transform=ax_info.transAxes,
            ha='center', fontsize=12, color='white',
        )
        ax_info.text(
            0.5, 0.88, f'{len(self.gates)} gates',
            transform=ax_info.transAxes,
            ha='center', fontsize=11, color='#aaa',
        )
        ax_info.text(
            0.5, 0.81, f'{len(self.edges)} edges',
            transform=ax_info.transAxes,
            ha='center', fontsize=11, color='#aaa',
        )
        
        # Activation bar chart
        bar_ax = ax_info.inset_axes([0.1, 0.1, 0.8, 0.35])
        bar_ax.set_facecolor('#0f3460')
        bar_ax.tick_params(colors='white', labelsize=8)
        bar_ax.set_xlabel('Qubit', color='white', fontsize=10)
        bar_ax.set_ylabel('Activation', color='white', fontsize=10)
        for spine in bar_ax.spines.values():
            spine.set_color('white')
        
        bars = bar_ax.bar(
            range(self.n_qubits),
            np.zeros(self.n_qubits),
            color='#e94560',
            edgecolor='white',
            linewidth=0.5,
        )
        bar_ax.set_xlim(-0.5, self.n_qubits - 0.5)
        bar_ax.set_ylim(-1, 1)
        
        # Message passing visualization
        message_artists = []
        
        def get_interpolated_state(frame):
            """Get interpolated node state for smooth transitions."""
            layer_idx = frame // n_frames_per_layer
            frame_in_layer = frame % n_frames_per_layer
            t = frame_in_layer / n_frames_per_layer
            
            if layer_idx >= n_states - 1:
                return self.node_states[-1], n_states - 1, 1.0
            
            state1 = self.node_states[layer_idx]
            state2 = self.node_states[layer_idx + 1]
            
            # Smooth interpolation
            t_smooth = t * t * (3 - 2 * t)  # Smoothstep
            
            return state1 + t_smooth * (state2 - state1), layer_idx, t
        
        def animate(frame):
            state, layer_idx, t = get_interpolated_state(frame)
            
            # Normalize state for coloring
            state_norm = (state - state.min()) / (state.max() - state.min() + 1e-6)
            
            # Update node colors
            for i, (circle, val) in enumerate(zip(node_circles, state_norm)):
                color = cmap(val)
                circle.set_facecolor(color)
                
                # Pulse effect during transition
                if t > 0 and t < 1:
                    pulse = 1 + 0.1 * np.sin(t * np.pi)
                    circle.set_radius(0.12 * pulse)
                else:
                    circle.set_radius(0.12)
            
            # Update layer text
            if layer_idx == 0:
                layer_name = "Input Features"
            elif layer_idx == 1:
                layer_name = "Node Embedding"
            else:
                layer_name = f"Message Passing Layer {layer_idx - 1}"
            
            layer_text.set_text(f'{layer_name}')
            
            # Update activation bars
            for bar, val in zip(bars, state):
                bar.set_height(val)
                color = cmap((val - state.min()) / (state.max() - state.min() + 1e-6))
                bar.set_facecolor(color)
            
            # Clear old message artists
            for artist in message_artists:
                artist.remove()
            message_artists.clear()
            
            # Draw message passing arrows during transition
            if t > 0.1 and t < 0.9 and layer_idx > 0:
                for edge in self.edges[:min(20, len(self.edges))]:  # Limit for clarity
                    if edge['is_self_loop']:
                        continue
                    
                    src, dst = edge['src'], edge['dst']
                    x1, y1 = self.positions[src]
                    x2, y2 = self.positions[dst]
                    
                    # Interpolate message position
                    msg_t = (t - 0.1) / 0.8
                    msg_x = x1 + msg_t * (x2 - x1)
                    msg_y = y1 + msg_t * (y2 - y1)
                    
                    # Draw message particle
                    color = get_gate_color(edge['gate_type'])
                    particle = plt.Circle(
                        (msg_x, msg_y), 0.04,
                        facecolor=color,
                        edgecolor='white',
                        linewidth=1,
                        alpha=0.8,
                        zorder=15,
                    )
                    ax_main.add_patch(particle)
                    message_artists.append(particle)
            
            return node_circles + [layer_text] + list(bars) + message_artists
        
        # Create animation
        anim = FuncAnimation(
            self.fig, animate,
            frames=total_frames,
            interval=1000 // self.fps,
            blit=False,
        )
        
        # Add overall title
        self.fig.suptitle(
            'Graph Neural Network on Quantum Circuit',
            fontsize=18, color='white', fontweight='bold',
            y=0.98,
        )
        
        plt.tight_layout()
        
        # Save as GIF
        print(f"Saving animation to {output_path}...")
        writer = PillowWriter(fps=self.fps)
        anim.save(output_path, writer=writer, dpi=self.dpi)
        print(f"Animation saved!")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GNN forward pass on quantum circuit"
    )
    parser.add_argument(
        "--circuit", type=str, default=None,
        help="Path to QASM file (default: use example circuit)"
    )
    parser.add_argument(
        "--output", type=str, default="gnn_forward_pass.gif",
        help="Output GIF path (default: gnn_forward_pass.gif)"
    )
    parser.add_argument(
        "--layout", type=str, default="circular",
        choices=["circular", "linear", "grid"],
        help="Node layout (default: circular)"
    )
    parser.add_argument(
        "--fps", type=int, default=12,
        help="Frames per second (default: 12)"
    )
    parser.add_argument(
        "--dpi", type=int, default=100,
        help="DPI for output (default: 100)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64,
        help="Model hidden dimension (default: 64)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4,
        help="Number of GNN layers (default: 4)"
    )
    args = parser.parse_args()
    
    # Load or create circuit
    if args.circuit:
        circuit_path = Path(args.circuit)
        if not circuit_path.exists():
            print(f"Error: Circuit file not found: {circuit_path}")
            sys.exit(1)
        qasm_text = circuit_path.read_text()
        print(f"Loaded circuit: {circuit_path}")
    else:
        # Use a nice example circuit (GHZ-like)
        qasm_text = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
h q[4];
rz(0.5) q[2];
rz(0.3) q[6];
t q[1];
t q[5];
cx q[0],q[7];
cx q[1],q[6];
cx q[2],q[5];
cx q[3],q[4];
"""
        print("Using example GHZ-like circuit")
    
    # Create model
    from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
    
    print(f"Creating GNN model (hidden_dim={args.hidden_dim}, num_layers={args.num_layers})...")
    model = create_gnn_model(
        model_type='basic',
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        global_feat_dim=GLOBAL_FEAT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    
    # Create visualizer
    print("Creating visualization...")
    visualizer = GNNVisualizer(
        qasm_text=qasm_text,
        model=model,
        layout=args.layout,
        fps=args.fps,
        dpi=args.dpi,
    )
    
    # Generate animation
    output_path = Path(args.output)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualizer.create_animation(str(output_path))
    print(f"\nVisualization complete: {output_path}")


if __name__ == "__main__":
    main()

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
from gnn.dataset import GLOBAL_FEAT_DIM, create_graph_data_loaders
from gnn.train import GNNTrainer


GATE_COLORS = {
    'h': '#E63946',      # Red - Hadamard
    'x': '#2A9D8F',      # Teal - Pauli X
    'y': '#457B9D',      # Steel blue - Pauli Y
    'z': '#1D3557',      # Dark blue - Pauli Z
    'cx': '#9B2335',     # Burgundy - CNOT
    'cz': '#264653',     # Dark teal - CZ
    'swap': '#E9C46A',   # Gold - SWAP
    'rz': '#7209B7',     # Purple - RZ
    'rx': '#F4A261',     # Orange - RX
    'ry': '#2196F3',     # Blue - RY
    't': '#F77F00',      # Bright orange - T gate
    's': '#06D6A0',      # Mint - S gate
    'cp': '#9C6644',     # Brown - Controlled phase
    'ccx': '#D62828',    # Crimson - Toffoli
    'default': '#6C757D' # Gray - default
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
        figsize: Tuple[int, int] = (10, 10),
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
        
        # Setup figure with single axes (square aspect)
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        self.fig.patch.set_facecolor('white')
        
        # Create colormap for activations
        self.cmap = LinearSegmentedColormap.from_list(
            'activation', ['#3B82F6', '#8B5CF6', '#EC4899']
        )
        
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
        ax.set_facecolor('white')
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
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
            
            # Draw with arrow
            arrow = mpatches.FancyArrowPatch(
                (x1, y1), (x2, y2),
                connectionstyle="arc3,rad=0.15",
                arrowstyle='-|>',
                mutation_scale=12,
                color=color,
                alpha=0.7,
                linewidth=1.5,
            )
            ax.add_patch(arrow)
            edge_artists.append(arrow)
        
        return edge_artists
    
    def _create_legend(self, ax):
        """Create a unified legend for gate types."""
        gate_types = set(e['gate_type'] for e in self.edges)
        
        patches = []
        for gate_type in sorted(gate_types):
            color = get_gate_color(gate_type)
            patch = mpatches.Patch(color=color, label=gate_type.upper(), alpha=0.7)
            patches.append(patch)
        
        legend = ax.legend(
            handles=patches,
            loc='upper left',
            fontsize=9,
            facecolor='white',
            edgecolor='#E5E7EB',
            labelcolor='#374151',
            title='Gate Types',
            title_fontsize=10,
            framealpha=0.95,
            borderpad=0.8,
        )
        legend.get_title().set_color('#1F2937')
    
    def create_animation(self, output_path: str, n_frames_per_layer: int = 15):
        """Create the animated GIF."""
        ax = self.ax
        
        # Draw static edges
        self._draw_static_elements(ax)
        self._create_legend(ax)
        
        # Compute total frames
        n_states = len(self.node_states)
        total_frames = (n_states - 1) * n_frames_per_layer + n_frames_per_layer
        
        # Get global min/max across all states for consistent colorbar
        all_states = np.concatenate(self.node_states)
        vmin, vmax = all_states.min(), all_states.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Initialize node circles
        node_circles = []
        node_labels = []
        base_radius = 0.1 if self.n_qubits > 20 else 0.12
        font_size = 7 if self.n_qubits > 20 else 9
        
        for i in range(self.n_qubits):
            x, y = self.positions[i]
            circle = plt.Circle(
                (x, y), base_radius, 
                facecolor=self.cmap(0.5),
                edgecolor='#374151',
                linewidth=1.5,
                zorder=10,
            )
            ax.add_patch(circle)
            node_circles.append(circle)
            
            label = ax.text(
                x, y, f'q{i}',
                ha='center', va='center',
                fontsize=font_size, fontweight='bold',
                color='white',
                zorder=11,
            )
            node_labels.append(label)
        
        # Title
        ax.set_title(
            'GNN Forward Pass',
            color='#1F2937', fontsize=16, pad=15,
            fontweight='bold',
        )
        
        # Layer indicator text
        layer_text = ax.text(
            0, -1.45, 'Input Features',
            ha='center', va='center',
            fontsize=12, color='#6B7280',
            fontweight='medium',
        )
        
        # Circuit info in corner
        info_text = f'{self.n_qubits} qubits · {len(self.gates)} gates'
        ax.text(
            0.98, 0.02, info_text,
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=9, color='#9CA3AF',
        )
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        cbar_ax = self.fig.add_axes([0.88, 0.25, 0.02, 0.5])
        cbar = self.fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Node Activation', fontsize=10, color='#374151')
        cbar.ax.tick_params(labelsize=8, colors='#374151')
        cbar.outline.set_edgecolor('#E5E7EB')
        
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
            
            t_smooth = t * t * (3 - 2 * t)
            return state1 + t_smooth * (state2 - state1), layer_idx, t
        
        def animate(frame):
            state, layer_idx, t = get_interpolated_state(frame)
            
            state_norm = norm(state)
            
            for i, (circle, val) in enumerate(zip(node_circles, state_norm)):
                color = self.cmap(val)
                circle.set_facecolor(color)
                
                if t > 0 and t < 1:
                    pulse = 1 + 0.08 * np.sin(t * np.pi)
                    circle.set_radius(base_radius * pulse)
                else:
                    circle.set_radius(base_radius)
            
            if layer_idx == 0:
                layer_name = "Input Features"
            elif layer_idx == 1:
                layer_name = "Node Embedding"
            else:
                layer_name = f"Message Passing Layer {layer_idx - 1}"
            
            layer_text.set_text(layer_name)
            
            for artist in message_artists:
                artist.remove()
            message_artists.clear()
            
            if t > 0.1 and t < 0.9 and layer_idx > 0:
                for edge in self.edges[:min(25, len(self.edges))]:
                    if edge['is_self_loop']:
                        continue
                    
                    src, dst = edge['src'], edge['dst']
                    x1, y1 = self.positions[src]
                    x2, y2 = self.positions[dst]
                    
                    msg_t = (t - 0.1) / 0.8
                    msg_x = x1 + msg_t * (x2 - x1)
                    msg_y = y1 + msg_t * (y2 - y1)
                    
                    color = get_gate_color(edge['gate_type'])
                    particle = plt.Circle(
                        (msg_x, msg_y), 0.03,
                        facecolor=color,
                        edgecolor='white',
                        linewidth=0.5,
                        alpha=0.9,
                        zorder=15,
                    )
                    ax.add_patch(particle)
                    message_artists.append(particle)
            
            return node_circles + [layer_text] + message_artists
        
        anim = FuncAnimation(
            self.fig, animate,
            frames=total_frames,
            interval=1000 // self.fps,
            blit=False,
        )
        
        plt.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.08)
        
        print(f"Saving animation to {output_path}...")
        writer = PillowWriter(fps=self.fps)
        anim.save(output_path, writer=writer, dpi=self.dpi)
        print(f"Animation saved!")
        
        plt.close()


def train_model(
    model: torch.nn.Module,
    data_path: Path,
    circuits_dir: Path,
    epochs: int = 50,
    device: str = "cpu",
) -> torch.nn.Module:
    """Train the GNN model on the dataset."""
    print("Loading training data...")
    train_loader, val_loader = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=16,
        val_fraction=0.2,
        seed=42,
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    print(f"Training model for up to {epochs} epochs...")
    trainer = GNNTrainer(
        model=model,
        device=device,
        lr=1e-3,
        weight_decay=1e-4,
        use_ordinal=True,
    )
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=15,
        verbose=False,
        show_progress=True,
    )
    
    final_metrics = trainer.evaluate(val_loader)
    print(f"  Validation runtime MAE (log2): {final_metrics.get('runtime_mae', 0):.4f}")
    
    return model


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
        "--hidden-dim", type=int, default=48,
        help="Model hidden dimension (default: 48)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=3,
        help="Number of GNN layers (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs (default: 50)"
    )
    parser.add_argument(
        "--no-train", action="store_true",
        help="Skip training (use random weights)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for training (cpu/cuda/mps)"
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    # Load or create circuit for visualization
    if args.circuit:
        circuit_path = Path(args.circuit)
        if not circuit_path.exists():
            print(f"Error: Circuit file not found: {circuit_path}")
            sys.exit(1)
        qasm_text = circuit_path.read_text()
        print(f"Loaded circuit: {circuit_path}")
    else:
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
    
    if not args.no_train:
        if not data_path.exists():
            print(f"Warning: Data file not found at {data_path}")
            print("Skipping training, using random weights.")
        else:
            model = train_model(
                model=model,
                data_path=data_path,
                circuits_dir=circuits_dir,
                epochs=args.epochs,
                device=args.device,
            )
    else:
        print("Skipping training (--no-train flag set)")
    
    print("Creating visualization...")
    visualizer = GNNVisualizer(
        qasm_text=qasm_text,
        model=model,
        layout=args.layout,
        fps=args.fps,
        dpi=args.dpi,
    )
    
    output_path = Path(args.output)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualizer.create_animation(str(output_path))
    print(f"\nVisualization complete: {output_path}")


if __name__ == "__main__":
    main()

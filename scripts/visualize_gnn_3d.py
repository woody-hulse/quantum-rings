"""
2.5D GIF Visualization of GNN Forward Pass for Quantum Circuit Runtime Prediction.

Creates an animated visualization showing:
- Quantum circuit as a graph (nodes=qubits, edges=gates)
- Color-coded gate types  
- Message passing through layers
- Global pooling and output
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
import matplotlib.patheffects as path_effects

plt.style.use('default')
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#0d1117'
plt.rcParams['text.color'] = '#e6edf3'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

COLORS = {
    'bg': '#0d1117',
    'node_base': '#58a6ff',
    'node_active': '#7ee787',
    'node_pooled': '#f0883e',
    '1q_gate': '#c084fc',
    '2q_gate': '#fb7185',
    '3q_gate': '#f87171',
    'message': '#fbbf24',
    'text': '#e6edf3',
    'dim': '#484f58',
    'global': '#79c0ff',
    'layer_bg': '#161b22',
}


def create_sample_circuit():
    """Create a sample quantum circuit graph for visualization."""
    n_qubits = 5
    
    gates = [
        (0, 0, 'H', 0.05),
        (1, 1, 'H', 0.05),
        (2, 2, 'H', 0.05),
        (3, 3, 'X', 0.08),
        (0, 1, 'CX', 0.18),
        (1, 2, 'CX', 0.30),
        (3, 4, 'CX', 0.25),
        (2, 2, 'RZ', 0.40),
        (2, 3, 'CZ', 0.50),
        (0, 0, 'RZ', 0.55),
        (1, 1, 'X', 0.60),
        (1, 3, 'CX', 0.70),
        (4, 4, 'H', 0.75),
        (0, 4, 'CX', 0.85),
    ]
    
    return n_qubits, gates


def get_gate_color(gate_type):
    """Get color for a gate type."""
    if gate_type in ['H', 'X', 'Y', 'Z', 'S', 'T', 'RX', 'RY', 'RZ', 'ID']:
        return COLORS['1q_gate']
    elif gate_type in ['CX', 'CZ', 'SWAP', 'CP', 'CRZ']:
        return COLORS['2q_gate']
    else:
        return COLORS['3q_gate']


def draw_curved_arrow(ax, x1, y1, x2, y2, color, alpha=0.9, lw=3, connectionstyle="arc3,rad=0.2"):
    """Draw a curved arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=connectionstyle,
        arrowstyle='->,head_length=6,head_width=4',
        color=color,
        alpha=alpha,
        linewidth=lw,
        mutation_scale=1.0,
        zorder=5
    )
    ax.add_patch(arrow)
    return arrow


def create_gnn_visualization():
    """Create the full GNN forward pass visualization."""
    n_qubits, gates = create_sample_circuit()
    
    fig = plt.figure(figsize=(14, 8), facecolor=COLORS['bg'])
    
    ax = fig.add_axes([0.04, 0.08, 0.68, 0.84], facecolor=COLORS['bg'])
    ax_legend = fig.add_axes([0.74, 0.08, 0.24, 0.84], facecolor=COLORS['bg'])
    ax_legend.axis('off')
    
    n_frames = 160
    n_layers = 4
    frames_per_layer = 35
    pooling_frames = 20
    
    layer_spacing = 3.2
    qubit_spacing = 1.2
    
    def get_node_pos(layer, qubit):
        """Get 2D position for a node."""
        x = layer * layer_spacing + 1.5
        y = (n_qubits - 1 - qubit) * qubit_spacing + 1.0
        return x, y
    
    def animate(frame):
        ax.clear()
        ax.set_facecolor(COLORS['bg'])
        
        ax.set_xlim(0, n_layers * layer_spacing + 4.5)
        ax.set_ylim(-0.3, n_qubits * qubit_spacing + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if frame < frames_per_layer * n_layers:
            current_layer = frame // frames_per_layer
            layer_progress = (frame % frames_per_layer) / frames_per_layer
            phase = 'message_passing'
        else:
            current_layer = n_layers - 1
            pooling_progress = min(1.0, (frame - frames_per_layer * n_layers) / pooling_frames)
            phase = 'pooling'
            layer_progress = 1.0
        
        # Draw layer backgrounds
        for layer in range(n_layers):
            is_current = (layer == current_layer)
            layer_alpha = 0.15 if is_current else 0.06
            
            x = layer * layer_spacing + 0.4
            y = -0.1
            w = layer_spacing - 0.3
            h = n_qubits * qubit_spacing + 0.4
            
            rect = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.05,rounding_size=0.3",
                facecolor=COLORS['layer_bg'],
                edgecolor=COLORS['dim'] if not is_current else COLORS['node_base'],
                alpha=layer_alpha,
                linewidth=2 if is_current else 1,
                zorder=1
            )
            ax.add_patch(rect)
            
            # Layer label
            label_alpha = 0.9 if is_current else 0.4
            ax.text(x + w/2, y - 0.15, f'Layer {layer+1}', fontsize=10,
                   color=COLORS['text'], alpha=label_alpha, ha='center', va='top',
                   fontweight='bold' if is_current else 'normal')
        
        # Draw all layers' nodes and edges
        for layer in range(current_layer + 1):
            is_current = (layer == current_layer)
            current_progress = layer_progress if is_current else 1.0
            layer_alpha = 1.0 if is_current else 0.4
            
            node_activations = np.zeros(n_qubits)
            for src, dst, gate_type, gate_pos in gates:
                if current_progress >= gate_pos:
                    activation = min(1.0, (current_progress - gate_pos) * 3.5)
                    node_activations[dst] = max(node_activations[dst], activation)
                    if src != dst:
                        node_activations[src] = max(node_activations[src], activation * 0.5)
            
            if not is_current:
                node_activations = np.ones(n_qubits) * 0.4
            
            # Draw edges (gates) FIRST
            for src, dst, gate_type, gate_pos in gates:
                x1, y1 = get_node_pos(layer, src)
                x2, y2 = get_node_pos(layer, dst)
                color = get_gate_color(gate_type)
                
                if is_current:
                    if current_progress >= gate_pos:
                        edge_alpha = 1.0
                        lw = 4.0
                    else:
                        edge_alpha = 0.25
                        lw = 2.0
                else:
                    edge_alpha = 0.55
                    lw = 2.5
                
                edge_alpha *= layer_alpha
                
                if src == dst:
                    loop_size = 0.28
                    theta = np.linspace(0.3, 2*np.pi - 0.3, 40)
                    loop_x = x1 - loop_size * 0.85 + loop_size * np.cos(theta)
                    loop_y = y1 + loop_size * np.sin(theta)
                    ax.plot(loop_x, loop_y, color=color, alpha=edge_alpha, 
                           linewidth=lw + 0.5, solid_capstyle='round', zorder=4)
                    
                    if is_current and current_progress >= gate_pos:
                        ax.scatter(x1 - loop_size * 0.85, y1, c=[color], s=45, 
                                  alpha=edge_alpha, marker='s', zorder=6)
                else:
                    rad = 0.18 if abs(dst - src) <= 2 else 0.12
                    arrow = FancyArrowPatch(
                        (x1 + 0.18, y1), (x2 + 0.18, y2),
                        connectionstyle=f"arc3,rad={rad}",
                        arrowstyle='->,head_length=7,head_width=5',
                        color=color,
                        alpha=edge_alpha,
                        linewidth=lw,
                        mutation_scale=1.2,
                        zorder=4
                    )
                    ax.add_patch(arrow)
            
            # Draw message particles
            if is_current and phase == 'message_passing':
                for src, dst, gate_type, gate_pos in gates:
                    if src != dst:
                        msg_progress = (current_progress - gate_pos) * 4.0
                        if 0 < msg_progress < 1:
                            x1, y1 = get_node_pos(layer, src)
                            x2, y2 = get_node_pos(layer, dst)
                            
                            x1 += 0.18
                            x2 += 0.18
                            rad = 0.18 if abs(dst - src) <= 2 else 0.12
                            ctrl = ((x1 + x2)/2 + rad * (y2 - y1) * 0.5,
                                   (y1 + y2)/2 - rad * (x2 - x1) * 0.5)
                            
                            t = msg_progress
                            mx = (1-t)**2 * x1 + 2*(1-t)*t * ctrl[0] + t**2 * x2
                            my = (1-t)**2 * y1 + 2*(1-t)*t * ctrl[1] + t**2 * y2
                            
                            glow = plt.Circle((mx, my), 0.18, color=COLORS['message'], 
                                             alpha=0.3, zorder=14)
                            ax.add_patch(glow)
                            
                            particle = plt.Circle((mx, my), 0.1, color=COLORS['message'], 
                                                  ec='white', linewidth=1.5, alpha=0.95, zorder=15)
                            ax.add_patch(particle)
                            
                            for ti in range(4):
                                trail_t = max(0, t - ti * 0.05)
                                trail_x = (1-trail_t)**2 * x1 + 2*(1-trail_t)*trail_t * ctrl[0] + trail_t**2 * x2
                                trail_y = (1-trail_t)**2 * y1 + 2*(1-trail_t)*trail_t * ctrl[1] + trail_t**2 * y2
                                trail_size = 0.06 * (1 - ti/4)
                                trail_alpha = 0.6 * (1 - ti/4)
                                trail = plt.Circle((trail_x, trail_y), trail_size, 
                                                  color=COLORS['message'], alpha=trail_alpha, zorder=13)
                                ax.add_patch(trail)
            
            # Draw nodes ON TOP
            for i in range(n_qubits):
                x, y = get_node_pos(layer, i)
                activation = node_activations[i]
                
                if phase == 'pooling' and is_current:
                    pulse = 0.5 + 0.5 * np.sin(pooling_progress * np.pi * 3)
                    base_color = np.array(to_rgba(COLORS['node_active']))
                    pool_color = np.array(to_rgba(COLORS['node_pooled']))
                    color = base_color * (1 - pooling_progress) + pool_color * pooling_progress
                    radius = 0.22 + 0.04 * pulse
                    node_alpha = 0.95
                else:
                    base_color = np.array(to_rgba(COLORS['node_base']))
                    active_color = np.array(to_rgba(COLORS['node_active']))
                    blend = activation if is_current else 0.3
                    color = base_color * (1 - blend) + active_color * blend
                    radius = 0.2 + 0.03 * activation if is_current else 0.16
                    node_alpha = 0.95 * layer_alpha
                
                glow = plt.Circle((x, y), radius + 0.08, color=color, alpha=0.2 * node_alpha, zorder=8)
                ax.add_patch(glow)
                
                node = plt.Circle((x, y), radius, color=color, ec='white', 
                                  linewidth=2, alpha=node_alpha, zorder=10)
                ax.add_patch(node)
        
        # Qubit labels
        for i in range(n_qubits):
            x, y = get_node_pos(0, i)
            ax.text(x - 0.45, y, f'q{i}', fontsize=10, color=COLORS['text'], 
                   alpha=0.8, ha='right', va='center', fontweight='bold')
        
        # Inter-layer connections (residual)
        for layer in range(min(current_layer, n_layers - 1)):
            for i in range(n_qubits):
                x1, y1 = get_node_pos(layer, i)
                x2, y2 = get_node_pos(layer + 1, i)
                ax.plot([x1 + 0.22, x2 - 0.22], [y1, y2],
                       color=COLORS['dim'], alpha=0.35, linewidth=1.5, 
                       linestyle='--', zorder=2)
        
        # Pooling
        if phase == 'pooling':
            pool_x = (current_layer + 1) * layer_spacing + 1.2
            pool_y = (n_qubits - 1) * qubit_spacing / 2 + 1.0
            
            pool_size = 0.35 + 0.1 * np.sin(pooling_progress * np.pi)
            pool_alpha = min(1.0, pooling_progress * 1.3)
            
            glow = plt.Circle((pool_x, pool_y), pool_size + 0.15, 
                             color=COLORS['node_pooled'], alpha=0.25 * pool_alpha, zorder=18)
            ax.add_patch(glow)
            
            rect = FancyBboxPatch(
                (pool_x - pool_size, pool_y - pool_size), 
                pool_size * 2, pool_size * 2,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=COLORS['node_pooled'],
                edgecolor='white',
                linewidth=2.5,
                alpha=pool_alpha,
                zorder=20
            )
            ax.add_patch(rect)
            
            for i in range(n_qubits):
                x, y = get_node_pos(current_layer, i)
                line_progress = min(1.0, pooling_progress * 1.2)
                end_x = x + (pool_x - x) * line_progress
                end_y = y + (pool_y - y) * line_progress
                
                line_alpha = min(0.75, pooling_progress * 1.0)
                ax.plot([x + 0.22, end_x], [y, end_y],
                       color=COLORS['node_pooled'], alpha=line_alpha, 
                       linewidth=2.5, zorder=15)
            
            if pooling_progress > 0.5:
                output_alpha = min(1.0, (pooling_progress - 0.5) * 2)
                txt = ax.text(pool_x + 0.6, pool_y, '→ log₂(t)', fontsize=14, 
                       color=COLORS['global'], alpha=output_alpha,
                       ha='left', va='center', fontweight='bold')
                txt.set_path_effects([
                    path_effects.withStroke(linewidth=3, foreground=COLORS['bg'])
                ])
            
            if pooling_progress > 0.3:
                label_alpha = min(1.0, (pooling_progress - 0.3) * 2)
                ax.text(pool_x, pool_y - pool_size - 0.25, 'Pooling', fontsize=10,
                       color=COLORS['text'], alpha=label_alpha, ha='center', va='top')
        
        # Title
        if phase == 'message_passing':
            title = f'GNN Forward Pass: Message Passing'
        else:
            title = 'GNN Forward Pass: Global Pooling'
        
        txt = ax.text(n_layers * layer_spacing / 2 + 1.5, n_qubits * qubit_spacing + 0.8, 
                     title, fontsize=16, color=COLORS['text'],
                     ha='center', va='bottom', fontweight='bold')
        txt.set_path_effects([
            path_effects.withStroke(linewidth=4, foreground=COLORS['bg'])
        ])
        
        return ax,
    
    # Legend
    ax_legend.text(0.5, 0.97, 'Legend', fontsize=14, color=COLORS['text'],
                  ha='center', va='top', fontweight='bold', transform=ax_legend.transAxes)
    
    ax_legend.text(0.08, 0.89, 'Nodes (Qubits)', fontsize=11, color=COLORS['text'],
                  ha='left', va='top', fontweight='bold', transform=ax_legend.transAxes)
    
    node_legend = [
        (COLORS['node_base'], 'Qubit (idle)'),
        (COLORS['node_active'], 'Qubit (activated)'),
        (COLORS['node_pooled'], 'Pooled output'),
    ]
    
    for i, (color, label) in enumerate(node_legend):
        y = 0.84 - i * 0.05
        circle = plt.Circle((0.12, y), 0.018, color=color, transform=ax_legend.transAxes,
                            ec='white', linewidth=1)
        ax_legend.add_patch(circle)
        ax_legend.text(0.20, y, label, fontsize=9, color=COLORS['text'],
                      ha='left', va='center', transform=ax_legend.transAxes)
    
    ax_legend.text(0.08, 0.66, 'Edges (Gates)', fontsize=11, color=COLORS['text'],
                  ha='left', va='top', fontweight='bold', transform=ax_legend.transAxes)
    
    gate_legend = [
        (COLORS['1q_gate'], '1-Qubit (H, X, RZ, ...)'),
        (COLORS['2q_gate'], '2-Qubit (CX, CZ, ...)'),
        (COLORS['3q_gate'], '3-Qubit (CCX, ...)'),
    ]
    
    for i, (color, label) in enumerate(gate_legend):
        y = 0.61 - i * 0.05
        ax_legend.annotate('', xy=(0.16, y), xytext=(0.08, y),
                          arrowprops=dict(arrowstyle='->', color=color, lw=3),
                          transform=ax_legend.transAxes)
        ax_legend.text(0.20, y, label, fontsize=9, color=COLORS['text'],
                      ha='left', va='center', transform=ax_legend.transAxes)
    
    ax_legend.text(0.08, 0.43, 'Animation', fontsize=11, color=COLORS['text'],
                  ha='left', va='top', fontweight='bold', transform=ax_legend.transAxes)
    
    circle = plt.Circle((0.12, 0.38), 0.015, color=COLORS['message'], transform=ax_legend.transAxes,
                        ec='white', linewidth=1)
    ax_legend.add_patch(circle)
    ax_legend.text(0.20, 0.38, 'Message particle', fontsize=9, color=COLORS['text'],
                  ha='left', va='center', transform=ax_legend.transAxes)
    
    ax_legend.plot([0.08, 0.16], [0.33, 0.33], color=COLORS['dim'], linewidth=1.5,
                  linestyle='--', transform=ax_legend.transAxes)
    ax_legend.text(0.20, 0.33, 'Residual connection', fontsize=9, color=COLORS['text'],
                  ha='left', va='center', transform=ax_legend.transAxes)
    
    ax_legend.text(0.08, 0.24, 'Architecture', fontsize=11, color=COLORS['text'],
                  ha='left', va='top', fontweight='bold', transform=ax_legend.transAxes)
    
    arch_info = [
        '• 4 Message Passing layers',
        '• Per-gate-type embeddings',
        '• Residual connections',
        '• LayerNorm + Dropout',
        '• Mean/Max/Sum pooling',
        '• Output: log₂(duration)',
    ]
    
    for i, text in enumerate(arch_info):
        y = 0.19 - i * 0.035
        ax_legend.text(0.08, y, text, fontsize=8, color=COLORS['dim'],
                      ha='left', va='center', transform=ax_legend.transAxes)
    
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=False)
    
    output_path = 'gnn_forward_pass.gif'
    print(f"Saving animation to {output_path}...")
    writer = PillowWriter(fps=20)
    anim.save(output_path, writer=writer, dpi=100)
    print(f"Saved to {output_path}")
    
    plt.close()
    return output_path


if __name__ == '__main__':
    create_gnn_visualization()

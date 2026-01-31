"""
Graph Neural Network module for quantum circuit prediction.

This module provides GNN-based models that learn from the topology of quantum circuits,
where qubits are nodes and gates are edges.

Main components:
- graph_builder: Parse QASM files and build PyTorch Geometric graphs
- model: GNN architectures with per-gate-type learnable embeddings
- dataset: PyTorch Geometric datasets and data loaders
- train: Training script with challenge-compatible evaluation
"""

from .graph_builder import (
    build_graph_from_qasm,
    build_graph_from_file,
    parse_qasm,
    NUM_GATE_TYPES,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    GLOBAL_FEAT_DIM_BASE,
    GATE_1Q,
    GATE_2Q,
    GATE_3Q,
    ALL_GATE_TYPES,
    GATE_TO_IDX,
)

from .model import (
    QuantumCircuitGNN,
    QuantumCircuitGNNWithAttention,
    GateTypeMessagePassing,
    create_gnn_model,
)

from .dataset import (
    QuantumCircuitGraphDataset,
    LazyQuantumCircuitGraphDataset,
    create_graph_data_loaders,
    create_kfold_graph_data_loaders,
    THRESHOLD_LADDER,
    FAMILY_CATEGORIES,
    FAMILY_TO_IDX,
    NUM_FAMILIES,
    GLOBAL_FEAT_DIM,
)

__all__ = [
    # Graph building
    "build_graph_from_qasm",
    "build_graph_from_file",
    "parse_qasm",
    "NUM_GATE_TYPES",
    "NODE_FEAT_DIM",
    "EDGE_FEAT_DIM",
    "GLOBAL_FEAT_DIM_BASE",
    "GATE_1Q",
    "GATE_2Q",
    "GATE_3Q",
    "ALL_GATE_TYPES",
    "GATE_TO_IDX",
    # Models
    "QuantumCircuitGNN",
    "QuantumCircuitGNNWithAttention",
    "GateTypeMessagePassing",
    "create_gnn_model",
    # Datasets
    "QuantumCircuitGraphDataset",
    "LazyQuantumCircuitGraphDataset",
    "create_graph_data_loaders",
    "create_kfold_graph_data_loaders",
    "THRESHOLD_LADDER",
    "FAMILY_CATEGORIES",
    "FAMILY_TO_IDX",
    "NUM_FAMILIES",
    "GLOBAL_FEAT_DIM",
]

"""
Graph Neural Network module for quantum circuit prediction.

This module provides GNN-based models that learn from the topology of quantum circuits,
where qubits are nodes and gates are edges.

Main components:
- base: Abstract base classes defining the common interface for all graph models
- graph_builder: Parse QASM files and build PyTorch Geometric graphs
- model: GNN architectures with per-gate-type learnable embeddings
- transformer: Graph Transformer architectures with edge-aware attention
- hetero_gnn: Heterogeneous GNN for multi-relation graphs
- dataset: PyTorch Geometric datasets and data loaders
- augmentation: Data augmentation for training
- train: Training script with challenge-compatible evaluation

Architecture Hierarchy:
    BaseGraphModel
    ├── BaseGraphDurationModel
    │   ├── QuantumCircuitGNN
    │   ├── ImprovedQuantumCircuitGNN
    │   ├── QuantumCircuitGraphTransformer
    │   ├── QuantumCircuitHeteroGNNDuration
    │   └── TemporalQuantumCircuitGNN (NEW - temporal/causal modeling)
    └── BaseGraphThresholdClassModel
        ├── QuantumCircuitGNNThresholdClass
        ├── ImprovedQuantumCircuitGNNThresholdClass
        ├── QuantumCircuitGraphTransformerThresholdClass
        ├── QuantumCircuitHeteroGNN
        └── TemporalQuantumCircuitGNNThresholdClass (NEW)

Key Features:
- Unified interface for all graph models (encode_nodes, pool_graph, forward)
- Multiple architectures: Message-passing GNN, Attentive GNN, Graph Transformer, Hetero GNN
- Ordinal regression for threshold prediction (exploits ordered nature)
- Data augmentation (qubit permutation, edge dropout, feature noise)
- Edge-aware attention with gate-type-specific biases
- Graph positional encodings for structure awareness
"""

from .base import (
    BaseGraphModel,
    BaseGraphDurationModel,
    BaseGraphThresholdClassModel,
    GraphModelConfig,
)

from .graph_builder import (
    build_graph_from_qasm,
    build_graph_from_file,
    parse_qasm,
    NUM_GATE_TYPES,
    NODE_FEAT_DIM,
    NODE_FEAT_DIM_BASIC,
    EDGE_FEAT_DIM,
    EDGE_FEAT_DIM_BASIC,
    GLOBAL_FEAT_DIM_BASE,
    GATE_1Q,
    GATE_2Q,
    GATE_3Q,
    ALL_GATE_TYPES,
    GATE_TO_IDX,
)

from .model import (
    QuantumCircuitGNN,
    QuantumCircuitGNNThresholdClass,
    GateTypeMessagePassing,
    create_gnn_model,
    create_gnn_threshold_class_model,
)

from .dataset import (
    QuantumCircuitGraphDataset,
    LazyQuantumCircuitGraphDataset,
    create_graph_data_loaders,
    create_kfold_graph_data_loaders,
    create_threshold_class_graph_data_loaders,
    create_kfold_threshold_class_graph_data_loaders,
    THRESHOLD_LADDER,
    FAMILY_CATEGORIES,
    FAMILY_TO_IDX,
    NUM_FAMILIES,
    NUM_THRESHOLD_CLASSES,
    GLOBAL_FEAT_DIM,
    GLOBAL_FEAT_DIM_THRESHOLD_CLASS,
)

from .augmentation import (
    QubitPermutation,
    EdgeDropout,
    FeatureNoise,
    TemporalJitter,
    RandomEdgeReverse,
    Compose,
    get_train_augmentation,
    AugmentedDataset,
)

from .improved_model import (
    ImprovedQuantumCircuitGNN,
    ImprovedQuantumCircuitGNNThresholdClass,
    AttentiveGateMessagePassing,
    OrdinalRegressionHead,
    OrdinalLoss,
    FocalLoss,
    ConservativeCrossEntropyLoss,
    create_improved_gnn_model,
)

from .transformer import (
    QuantumCircuitGraphTransformer,
    QuantumCircuitGraphTransformerThresholdClass,
    GraphTransformerEncoder,
    GraphTransformerLayer,
    EdgeAwareMultiHeadAttention,
    GraphPositionalEncoding,
    create_graph_transformer_model,
)

from .hetero_gnn import (
    QuantumCircuitHeteroGNN,
    QuantumCircuitHeteroGNNDuration,
    HeteroMessagePassingBlock,
    HeteroAttentionLayer,
    MetaPathAttention,
    EntanglementAwarePooling,
    EdgeRelation,
    NUM_RELATIONS,
    create_hetero_gnn_model,
    build_hetero_edges_from_standard,
)

from .temporal_model import (
    TemporalQuantumCircuitGNN,
    TemporalQuantumCircuitGNNThresholdClass,
    TemporalGateEmbedding,
    CausalTemporalAttention,
    StateMemoryGRU,
    EntanglementAwareConv,
    MultiScaleTemporalPooling,
    TemporalGraphTransformerLayer,
    create_temporal_gnn_model,
    TEMPORAL_GNN_CONFIGS,
)

from .temporal_graph_builder import (
    build_temporal_graph,
    build_temporal_graph_from_file,
    compute_circuit_layers,
    compute_entanglement_trajectory,
    TEMPORAL_NODE_FEAT_DIM,
    TEMPORAL_EDGE_FEAT_DIM,
    TEMPORAL_GLOBAL_FEAT_DIM_BASE,
)

__all__ = [
    # Base classes
    "BaseGraphModel",
    "BaseGraphDurationModel",
    "BaseGraphThresholdClassModel",
    "GraphModelConfig",
    # Graph building
    "build_graph_from_qasm",
    "build_graph_from_file",
    "parse_qasm",
    "NUM_GATE_TYPES",
    "NODE_FEAT_DIM",
    "NODE_FEAT_DIM_BASIC",
    "EDGE_FEAT_DIM",
    "EDGE_FEAT_DIM_BASIC",
    "GLOBAL_FEAT_DIM_BASE",
    "GATE_1Q",
    "GATE_2Q",
    "GATE_3Q",
    "ALL_GATE_TYPES",
    "GATE_TO_IDX",
    # GNN Models
    "QuantumCircuitGNN",
    "QuantumCircuitGNNThresholdClass",
    "GateTypeMessagePassing",
    "create_gnn_model",
    "create_gnn_threshold_class_model",
    # Datasets
    "QuantumCircuitGraphDataset",
    "LazyQuantumCircuitGraphDataset",
    "create_graph_data_loaders",
    "create_kfold_graph_data_loaders",
    "create_threshold_class_graph_data_loaders",
    "THRESHOLD_LADDER",
    "FAMILY_CATEGORIES",
    "FAMILY_TO_IDX",
    "NUM_FAMILIES",
    "NUM_THRESHOLD_CLASSES",
    "GLOBAL_FEAT_DIM",
    "GLOBAL_FEAT_DIM_THRESHOLD_CLASS",
    # Augmentation
    "QubitPermutation",
    "EdgeDropout",
    "FeatureNoise",
    "TemporalJitter",
    "RandomEdgeReverse",
    "Compose",
    "get_train_augmentation",
    "AugmentedDataset",
    # Improved GNN models
    "ImprovedQuantumCircuitGNN",
    "ImprovedQuantumCircuitGNNThresholdClass",
    "AttentiveGateMessagePassing",
    "OrdinalRegressionHead",
    "OrdinalLoss",
    "FocalLoss",
    "ConservativeCrossEntropyLoss",
    "create_improved_gnn_model",
    # Graph Transformer models
    "QuantumCircuitGraphTransformer",
    "QuantumCircuitGraphTransformerThresholdClass",
    "GraphTransformerEncoder",
    "GraphTransformerLayer",
    "EdgeAwareMultiHeadAttention",
    "GraphPositionalEncoding",
    "create_graph_transformer_model",
    # Heterogeneous GNN (QCHGT)
    "QuantumCircuitHeteroGNN",
    "QuantumCircuitHeteroGNNDuration",
    "HeteroMessagePassingBlock",
    "HeteroAttentionLayer",
    "MetaPathAttention",
    "EntanglementAwarePooling",
    "EdgeRelation",
    "NUM_RELATIONS",
    "create_hetero_gnn_model",
    "build_hetero_edges_from_standard",
    # Temporal GNN models
    "TemporalQuantumCircuitGNN",
    "TemporalQuantumCircuitGNNThresholdClass",
    "TemporalGateEmbedding",
    "CausalTemporalAttention",
    "StateMemoryGRU",
    "EntanglementAwareConv",
    "MultiScaleTemporalPooling",
    "TemporalGraphTransformerLayer",
    "create_temporal_gnn_model",
    "TEMPORAL_GNN_CONFIGS",
    # Temporal graph building
    "build_temporal_graph",
    "build_temporal_graph_from_file",
    "compute_circuit_layers",
    "compute_entanglement_trajectory",
    "TEMPORAL_NODE_FEAT_DIM",
    "TEMPORAL_EDGE_FEAT_DIM",
    "TEMPORAL_GLOBAL_FEAT_DIM_BASE",
]

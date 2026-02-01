"""
State-of-the-Art Graph Neural Network Module for Quantum Circuit Prediction.

This module provides cutting-edge GNN implementations for predicting:
- Quantum circuit runtime/duration
- Optimal threshold selection

Key Improvements over Basic GNN:
================================

1. ARCHITECTURE IMPROVEMENTS:
   - GraphGPS-style hybrid architecture (local MPNN + global attention)
   - Principal Neighborhood Aggregation (PNA) with multiple aggregators
   - Gate-type aware Mixture of Experts (MoE)
   - Virtual nodes for global information flow
   - GATv2-style edge-aware attention
   - Gated GCN layers with GRU-style updates

2. POSITIONAL ENCODINGS:
   - Random Walk Positional Encoding
   - Laplacian Eigenvector Encoding
   - Sinusoidal temporal encoding for gates

3. TRAINING IMPROVEMENTS:
   - Exponential Moving Average (EMA)
   - Mixed precision training (AMP)
   - Gradient accumulation
   - Deep supervision with auxiliary losses
   - Graph Mixup regularization
   - Cosine annealing with warmup
   - Ordinal regression for threshold classification
   - Focal loss for class imbalance

4. FEATURE ENGINEERING:
   - Enhanced node features (spectral, connectivity, temporal)
   - Rich edge features (gate properties, sinusoidal PE)
   - Circuit-level statistics
   - Gate property encoding (Clifford, rotation, diagonal)

Main Components:
================

Models:
-------
- QuantumCircuitGNNSoTA: Full state-of-the-art model for duration prediction
- QuantumCircuitGNNLite: Efficient variant balancing speed and accuracy
- QuantumCircuitGNNThresholdClassSoTA: Classification model for threshold selection

Building Blocks:
----------------
- GPSBlock: GraphGPS block combining local and global processing
- GatedGCNLayer: Gated graph convolution with edge features
- PNAConv: Principal Neighborhood Aggregation
- EdgeAwareGATv2: Attention with edge features
- GateTypeExpertLayer: Mixture of Experts for gate types
- VirtualNode: Global information propagation
- HierarchicalPooling: Multi-scale graph pooling

Training:
---------
- AdvancedGNNTrainer: Full-featured trainer for regression
- AdvancedThresholdClassTrainer: Trainer for classification
- EMA: Exponential Moving Average
- GraphMixup: Graph-level data augmentation

Usage Example:
==============

```python
from gnn import (
    create_sota_gnn_model,
    AdvancedGNNTrainer,
    build_graph_enhanced,
)

# Create model
model = create_sota_gnn_model(
    hidden_dim=128,
    num_layers=6,
    num_heads=8,
    local_type='gated_gcn',
)

# Create trainer with all bells and whistles
trainer = AdvancedGNNTrainer(
    model,
    device='cuda',
    use_amp=True,
    use_ema=True,
    use_mixup=True,
)

# Train
history = trainer.fit(train_loader, val_loader, epochs=100)

# Predict
thresholds, runtimes = trainer.predict(test_loader)
```
"""

# =============================================================================
# ORIGINAL COMPONENTS (backward compatibility)
# =============================================================================

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

# =============================================================================
# STATE-OF-THE-ART COMPONENTS
# =============================================================================

from .model_sota import (
    # Main models
    QuantumCircuitGNNSoTA,
    QuantumCircuitGNNThresholdClassSoTA,
    QuantumCircuitGNNLite,
    
    # Factory functions
    create_sota_gnn_model,
    create_sota_threshold_class_model,
    
    # Building blocks - Positional Encodings
    RandomWalkPE,
    LaplacianPE,
    SinusoidalPE,
    
    # Building blocks - Message Passing
    GatedGCNLayer,
    PNAConv,
    GateTypeExpertLayer,
    
    # Building blocks - Attention
    GraphMultiHeadAttention,
    EdgeAwareGATv2,
    
    # Building blocks - Architecture
    GPSBlock,
    VirtualNode,
    HierarchicalPooling,
)

from .train_advanced import (
    # Trainers
    AdvancedGNNTrainer,
    AdvancedThresholdClassTrainer,
    
    # Training utilities
    EMA,
    GraphMixup,
    
    # Loss functions
    DeepSupervisionLoss,
    OrdinalRegressionLoss,
    FocalLoss,
    
    # Schedulers
    get_cosine_schedule_with_warmup,
)

from .graph_builder_enhanced import (
    build_graph_enhanced,
    build_graph_from_file_enhanced,
    parse_qasm_enhanced,
    compute_spectral_features,
    compute_connectivity_features,
    sinusoidal_encoding,
    NODE_FEAT_DIM_ENHANCED,
    EDGE_FEAT_DIM_ENHANCED,
    GLOBAL_FEAT_DIM_ENHANCED,
    GATE_PROPERTIES,
)


# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Original graph building
    # -------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------
    # Enhanced graph building
    # -------------------------------------------------------------------------
    "build_graph_enhanced",
    "build_graph_from_file_enhanced",
    "parse_qasm_enhanced",
    "compute_spectral_features",
    "compute_connectivity_features",
    "sinusoidal_encoding",
    "NODE_FEAT_DIM_ENHANCED",
    "EDGE_FEAT_DIM_ENHANCED",
    "GLOBAL_FEAT_DIM_ENHANCED",
    "GATE_PROPERTIES",
    
    # -------------------------------------------------------------------------
    # Original models
    # -------------------------------------------------------------------------
    "QuantumCircuitGNN",
    "QuantumCircuitGNNThresholdClass",
    "GateTypeMessagePassing",
    "create_gnn_model",
    "create_gnn_threshold_class_model",
    
    # -------------------------------------------------------------------------
    # State-of-the-art models
    # -------------------------------------------------------------------------
    "QuantumCircuitGNNSoTA",
    "QuantumCircuitGNNThresholdClassSoTA",
    "QuantumCircuitGNNLite",
    "create_sota_gnn_model",
    "create_sota_threshold_class_model",
    
    # -------------------------------------------------------------------------
    # Model building blocks - Positional Encodings
    # -------------------------------------------------------------------------
    "RandomWalkPE",
    "LaplacianPE",
    "SinusoidalPE",
    
    # -------------------------------------------------------------------------
    # Model building blocks - Message Passing
    # -------------------------------------------------------------------------
    "GatedGCNLayer",
    "PNAConv",
    "GateTypeExpertLayer",
    
    # -------------------------------------------------------------------------
    # Model building blocks - Attention
    # -------------------------------------------------------------------------
    "GraphMultiHeadAttention",
    "EdgeAwareGATv2",
    
    # -------------------------------------------------------------------------
    # Model building blocks - Architecture
    # -------------------------------------------------------------------------
    "GPSBlock",
    "VirtualNode",
    "HierarchicalPooling",
    
    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
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
    
    # -------------------------------------------------------------------------
    # Data Augmentation
    # -------------------------------------------------------------------------
    "QubitPermutation",
    "EdgeDropout",
    "FeatureNoise",
    "TemporalJitter",
    "RandomEdgeReverse",
    "Compose",
    "get_train_augmentation",
    "AugmentedDataset",
    
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    "AdvancedGNNTrainer",
    "AdvancedThresholdClassTrainer",
    "EMA",
    "GraphMixup",
    
    # -------------------------------------------------------------------------
    # Loss Functions
    # -------------------------------------------------------------------------
    "DeepSupervisionLoss",
    "OrdinalRegressionLoss",
    "FocalLoss",
    
    # -------------------------------------------------------------------------
    # Schedulers
    # -------------------------------------------------------------------------
    "get_cosine_schedule_with_warmup",
]


# Version info
__version__ = "2.0.0"
__author__ = "Quantum Circuit Prediction Team"


def get_model_info():
    """Print information about available models."""
    info = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           STATE-OF-THE-ART GNN FOR QUANTUM CIRCUIT PREDICTION                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MODELS:                                                                     ║
║  ├── QuantumCircuitGNNSoTA      - Full featured model (~500K params)         ║
║  ├── QuantumCircuitGNNLite      - Efficient variant (~100K params)           ║
║  └── QuantumCircuitGNNThresholdClassSoTA - Classification model              ║
║                                                                              ║
║  KEY FEATURES:                                                               ║
║  ├── GraphGPS Architecture      - Local MPNN + Global Attention              ║
║  ├── Virtual Nodes              - Global information flow                    ║
║  ├── Mixture of Experts         - Gate-type aware processing                 ║
║  ├── Positional Encodings       - Random Walk PE, Laplacian PE               ║
║  ├── Hierarchical Pooling       - Multi-scale representations                ║
║  └── Deep Supervision           - Auxiliary losses for regularization        ║
║                                                                              ║
║  TRAINING FEATURES:                                                          ║
║  ├── Mixed Precision (AMP)      - Faster training with FP16                  ║
║  ├── EMA                        - Exponential Moving Average                 ║
║  ├── Graph Mixup                - Data augmentation                          ║
║  ├── Cosine Annealing           - With warmup                                ║
║  └── Gradient Accumulation      - Larger effective batch sizes               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(info)


# Convenience function to create the best model
def create_best_model(task: str = "runtime", device: str = "cpu", **kwargs):
    """
    Create the best model for a given task.
    
    Args:
        task: "runtime" for duration prediction, "threshold" for classification
        device: "cpu", "cuda", or "mps"
        **kwargs: Additional model arguments
    
    Returns:
        Configured model ready for training
    """
    default_kwargs = {
        "hidden_dim": 128,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "local_type": "gated_gcn",
        "use_virtual_node": True,
        "use_moe": True,
    }
    default_kwargs.update(kwargs)
    
    if task == "runtime":
        model = create_sota_gnn_model(**default_kwargs)
    elif task == "threshold":
        model = create_sota_threshold_class_model(**default_kwargs)
    else:
        raise ValueError(f"Unknown task: {task}. Use 'runtime' or 'threshold'.")
    
    return model.to(device)


def create_best_trainer(model, task: str = "runtime", device: str = "cpu", **kwargs):
    """
    Create the best trainer for a given task.
    
    Args:
        model: The model to train
        task: "runtime" for duration prediction, "threshold" for classification
        device: "cpu", "cuda", or "mps"
        **kwargs: Additional trainer arguments
    
    Returns:
        Configured trainer
    """
    default_kwargs = {
        "device": device,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "use_amp": device != "cpu",
        "use_ema": True,
        "warmup_epochs": 5,
    }
    default_kwargs.update(kwargs)
    
    if task == "runtime":
        trainer = AdvancedGNNTrainer(model, **default_kwargs)
    elif task == "threshold":
        trainer = AdvancedThresholdClassTrainer(model, **default_kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return trainer

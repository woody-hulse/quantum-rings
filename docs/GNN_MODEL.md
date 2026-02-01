# Graph Neural Network Model Specification

This document describes the architecture and features of the Graph Neural Network (GNN) used for quantum circuit runtime prediction.

## Overview

The GNN treats quantum circuits as graphs where:
- **Nodes** represent qubits
- **Edges** represent gates (directed edges for multi-qubit gates, self-loops for single-qubit gates)

The model predicts circuit execution duration given a threshold parameter, outputting `log2(duration)`.

---

## Graph Structure

### Node Representation
Each qubit in the circuit is represented as a node. The graph contains `n_qubits` nodes.

### Edge Representation
Gates are represented as edges:
- **Single-qubit gates**: Self-loop edges from qubit to itself
- **Two-qubit gates**: Directed edge from control qubit to target qubit
- **Three-qubit gates**: Two directed edges (from each control qubit to the target)

---

## Feature Dimensions

| Feature Type | Dimension | Description |
|--------------|-----------|-------------|
| Node Features | 22 | Per-qubit gate statistics and temporal information |
| Edge Features | 4 | Gate parameters and temporal position |
| Global Features | 27 | Circuit-level statistics (base=6 + threshold=1 + family=20) |

---

## Node Features (22 dimensions)

Each node (qubit) has the following features:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| 1Q Gate Counts | 15 | Log-transformed count per gate type: `h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`, `rx`, `ry`, `rz`, `u1`, `u2`, `u3`, `id` |
| 2Q Involvement Count | 1 | Log-transformed count of 2-qubit gates involving this qubit |
| Position | 1 | Normalized position in register `[0, 1]` |
| First 2Q Position | 1 | Normalized position of first 2-qubit gate (1.0 if never entangled) |
| Last 2Q Position | 1 | Normalized position of last 2-qubit gate (0.0 if never entangled) |
| Activity Window | 1 | Duration of 2-qubit gate activity (last - first position) |
| Unique Neighbors | 1 | Count of unique qubits interacted with (normalized by n_qubits-1) |
| Average Span | 1 | Mean interaction distance across all 2-qubit gates |

---

## Edge Features (4 dimensions)

Each edge (gate) has the following features:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Temporal Position | 1 | Normalized position in circuit `[0, 1]` |
| Gate Parameter | 1 | First parameter value (e.g., rotation angle), 0.0 if none |
| Qubit Distance | 1 | Normalized distance between qubits (0 for self-loops) |
| Cumulative 2Q Index | 1 | Cumulative 2-qubit gate count normalized by total gates |

### Edge Gate Type Encoding
Each edge also has a discrete gate type index (0-29) used for learned embeddings:

**Single-qubit gates (0-14):** `h`, `x`, `y`, `z`, `s`, `sdg`, `t`, `tdg`, `rx`, `ry`, `rz`, `u1`, `u2`, `u3`, `id`

**Two-qubit gates (15-26):** `cx`, `cz`, `swap`, `cp`, `crx`, `cry`, `crz`, `cu1`, `cu3`, `rxx`, `ryy`, `rzz`

**Three-qubit gates (27-29):** `ccx`, `cswap`, `ccz`

---

## Global Features (27 dimensions)

Circuit-level features concatenated and passed separately:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Number of Qubits | 1 | Normalized: `n_qubits / 130.0` |
| Number of Gates | 1 | Normalized: `n_gates / 1000.0` |
| Number of 2Q Gates | 1 | Normalized: `n_2q_gates / 500.0` |
| Gate Density | 1 | `n_2q_gates / n_qubits / 10.0` |
| Backend | 1 | Binary: 0.0 = CPU, 1.0 = GPU |
| Precision | 1 | Binary: 0.0 = single, 1.0 = double |
| Log2 Threshold | 1 | `log2(threshold) / 8.0` |
| Family One-Hot | 20 | One-hot encoding of circuit family |

### Circuit Families (20 categories)
```
Amplitude_Estimation, CutBell, Deutsch_Jozsa, GHZ, GraphState,
Ground_State, Grover_NoAncilla, Grover_V_Chain, Portfolio_QAOA,
Portfolio_VQE, Pricing_Call, QAOA, QFT, QFT_Entangled, QNN,
QPE_Exact, Shor, TwoLocalRandom, VQE, W_State
```

---

## Model Architecture

### QuantumCircuitGNN

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Processing                          │
├─────────────────────────────────────────────────────────────┤
│  Node Features (22) ─────► Node Embedding MLP ──► hidden_dim │
│  Global Features (27) ───► Global Projection ──► hidden_dim  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Message Passing Layers (×4)                     │
├─────────────────────────────────────────────────────────────┤
│  GateTypeMessagePassing with:                                │
│    • Per-gate-type learnable embeddings                      │
│    • Edge attribute integration                              │
│    • Residual connections                                    │
│    • Layer normalization                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Graph Pooling                             │
├─────────────────────────────────────────────────────────────┤
│  global_mean_pool ────┐                                      │
│  global_max_pool  ────┼──► Concatenate (3 × hidden_dim)     │
│  global_add_pool  ────┘                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Head                               │
├─────────────────────────────────────────────────────────────┤
│  [Graph Repr + Global Proj] ──► Combined MLP ──► Linear(1)  │
│                                                              │
│  Output: log2(duration) prediction                           │
└─────────────────────────────────────────────────────────────┘
```

### Default Hyperparameters

| Parameter | Default Value |
|-----------|---------------|
| Hidden Dimension | 64 |
| Number of MP Layers | 4 |
| Dropout Rate | 0.1 |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |

---

## GateTypeMessagePassing Layer

Custom message passing layer that learns gate-type-specific transformations:

```python
class GateTypeMessagePassing(MessagePassing):
    """
    Each gate type has its own learned embedding that modulates
    how information propagates through that gate.
    """
    
    Components:
    - gate_type_embed: Embedding(30, hidden_dim)  # Per-gate-type learned vectors
    - msg_mlp: MLP for message transformation
    - update_mlp: MLP for node update
    - norm: LayerNorm(hidden_dim)
```

### Message Computation
```
message = MLP([source_features, gate_embedding, edge_attributes])
```

### Node Update (with residual)
```
h_new = LayerNorm(MLP([h_old, aggregated_messages]))
h = h + Dropout(h_new)
```

---

## Data Augmentation

Training uses the following augmentation strategies:

| Augmentation | Probability | Description |
|--------------|-------------|-------------|
| Qubit Permutation | 0.5 | Randomly relabel qubits (circuit invariant) |
| Edge Dropout | 0.1 | Randomly drop edges for regularization |
| Feature Noise | 0.5 | Gaussian noise (σ=0.1) on node features |
| Temporal Jitter | 0.5 | Perturb temporal positions (σ=0.05) |
| Random Edge Reverse | 0.5 | Reverse direction for symmetric gates (CZ, SWAP) |

---

## Threshold Classification Variant

An alternative model `QuantumCircuitGNNThresholdClass` is available for directly predicting threshold classes instead of runtime:

| Aspect | Duration Model | Threshold Class Model |
|--------|---------------|----------------------|
| Output | log2(duration) | 9-class logits |
| Global Features | 27 (includes threshold) | 26 (excludes threshold) |
| Loss | L1Loss | CrossEntropyLoss |
| Classes | N/A | [1, 2, 4, 8, 16, 32, 64, 128, 256] |

---

## Training Configuration

### Optimizer
- **AdamW** with configurable learning rate and weight decay
- **ReduceLROnPlateau** scheduler (factor=0.5, patience=10, min_lr=1e-6)

### Training Loop
- Gradient clipping: max_norm=1.0
- Early stopping patience: 20 epochs
- Best model selection by validation MAE (log2 space)

### Evaluation Metrics
- Runtime MSE/MAE (in log2 space)
- Challenge threshold score
- Challenge runtime score
- Combined challenge score

---

## Usage Example

```python
from src.gnn import (
    create_gnn_model,
    create_graph_data_loaders,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    GLOBAL_FEAT_DIM,
)
from src.gnn.train import GNNTrainer

# Create data loaders
train_loader, val_loader = create_graph_data_loaders(
    data_path="data/hackathon_public.json",
    circuits_dir="circuits/",
    batch_size=32,
    val_fraction=0.2,
)

# Create model
model = create_gnn_model(
    node_feat_dim=NODE_FEAT_DIM,
    edge_feat_dim=EDGE_FEAT_DIM,
    global_feat_dim=GLOBAL_FEAT_DIM,
    hidden_dim=64,
    num_layers=4,
    dropout=0.1,
)

# Train
trainer = GNNTrainer(model=model, device="cpu", lr=1e-3)
trainer.fit(train_loader, val_loader, epochs=100)
```

---

## Available Graph Model Architectures

The codebase includes multiple GNN architectures, each with different characteristics:

### Architecture Comparison

| Model | Description | Parameters | Key Features |
|-------|-------------|------------|--------------|
| **BasicGNN** | Simple message-passing | ~154K | Per-gate-type embeddings, fast training |
| **ImprovedGNN** | Attention-based | ~222K | Multi-head attention, ordinal regression, stochastic depth |
| **GraphTransformer** | Full transformer | ~300K+ | Edge-aware attention, positional encoding |
| **HeteroGNN (QCHGT)** | Heterogeneous GNN | ~400K+ | Multi-relation edges, meta-path attention |
| **TemporalGNN** | Temporal modeling | ~500K+ | Causal attention, state memory GRU |

### Performance Comparison (Threshold Classification)

Based on empirical testing with default hyperparameters:

| Model | Val Score | Val Accuracy | Underprediction Rate | Training Time |
|-------|-----------|--------------|---------------------|---------------|
| BasicGNN | **0.62 ± 0.07** | **0.53 ± 0.08** | **0.20** | 26s |
| ImprovedGNN | 0.56 ± 0.07 | 0.43 ± 0.08 | 0.31 | 60s |

**Key Findings:**
1. **BasicGNN outperforms more complex models** on this dataset
2. Lower underprediction rate is critical (underpredictions score 0)
3. Simpler models generalize better with limited data
4. More complex architectures may require more data or careful tuning

### Recommended Approach

For the quantum circuit threshold prediction task:

1. **Start with BasicGNN** - it provides the best balance of performance and simplicity
2. **Use data augmentation** - qubit permutation and feature noise improve generalization
3. **Focus on reducing underprediction** - overprediction is preferable (partial score vs. zero)
4. **Consider ensemble approaches** - combining multiple models can reduce variance

### Unified Model Interface

All graph models implement a consistent interface via `BaseGraphModelWrapper`:

```python
from models import create_graph_model

model = create_graph_model(
    model_type="basic",  # or "improved", "transformer", "hetero", "temporal"
    hidden_dim=64,
    num_layers=4,
    num_heads=4,
    dropout=0.2,
    use_ordinal=True,
    use_augmentation=True,
)

history = model.fit(train_loader, val_loader)
probs = model.predict_proba(val_loader)
metrics = model.evaluate(val_loader)
```

---

## File Structure

```
src/gnn/
├── __init__.py              # Module exports
├── base.py                  # Abstract base classes for all graph models
├── graph_builder.py         # QASM parsing and graph construction
├── model.py                 # Basic GNN architecture
├── improved_model.py        # Improved GNN with attention
├── transformer.py           # Graph Transformer architecture
├── hetero_gnn.py           # Heterogeneous GNN (QCHGT)
├── temporal_model.py        # Temporal GNN with causal modeling
├── temporal_graph_builder.py # Temporal graph construction
├── dataset.py               # PyTorch Geometric datasets
├── augmentation.py          # Data augmentation transforms
└── train.py                 # Training loop and evaluation

src/models/
├── graph_models.py          # Unified wrappers for all graph models
├── gnn_threshold_class.py   # Legacy GNN wrapper
└── temporal_gnn_model.py    # Temporal GNN wrappers

scripts/
└── compare_graph_models.py  # Comprehensive model comparison script
```

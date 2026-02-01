# Task 1 & Task 2 Usage Guide

## Quick Reference

### Task 1: Predict Optimal Threshold
**Input:** Circuit (QASM), target fidelity (0.75), backend, precision
**Output:** Optimal threshold value

```python
from gnn.train import GNNTrainer

# After training/loading model
trainer = GNNTrainer(model, device="cpu")

# Predict optimal thresholds
thresholds = trainer.predict_task1(val_loader)
# Returns: np.array([4, 8, 16, 2, ...])  # actual threshold values
```

### Task 2: Predict Runtime for Given Threshold
**Input:** Circuit (QASM), **specific threshold**, backend, precision
**Output:** Predicted runtime (seconds)

```python
import torch
import numpy as np
from gnn.dataset import THRESHOLD_LADDER

# Given threshold VALUES from judges (e.g., [8, 16, 4, 32, ...])
given_thresholds = np.array([8, 16, 4, 32, 64])

# Convert to class indices (0-8)
threshold_classes = torch.tensor([
    THRESHOLD_LADDER.index(t) for t in given_thresholds
])

# Predict runtimes
runtimes = trainer.predict_task2(val_loader, threshold_classes)
# Returns: np.array([0.543, 1.234, 0.123, 2.456, 3.123])  # seconds
```

## Threshold Ladder Mapping

```python
THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
#                   0  1  2  3   4   5   6    7    8  <- class indices

# Examples:
# Threshold value 1   → class index 0
# Threshold value 4   → class index 2
# Threshold value 16  → class index 4
# Threshold value 256 → class index 8
```

## Complete Workflow

### Training

```bash
# Train the model (same as before)
python src/gnn/train.py --hidden-dim 48 --num-layers 3 --n-runs 10
```

### Inference

```python
import torch
import numpy as np
from pathlib import Path
from gnn.model import create_gnn_model
from gnn.dataset import create_graph_data_loaders, THRESHOLD_LADDER, GLOBAL_FEAT_DIM
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
from gnn.train import GNNTrainer

# Load data
_, val_loader = create_graph_data_loaders(
    data_path="data/hackathon_public.json",
    circuits_dir="circuits",
    batch_size=16,
)

# Create model
model = create_gnn_model(
    model_type="basic",
    node_feat_dim=NODE_FEAT_DIM,
    edge_feat_dim=EDGE_FEAT_DIM,
    global_feat_dim=GLOBAL_FEAT_DIM,
    hidden_dim=48,
    num_layers=3,
    dropout=0.25,
    use_ordinal=True,
)

# Load trained weights (if available)
# model.load_state_dict(torch.load("checkpoint.pt"))

# Create trainer
trainer = GNNTrainer(model=model, device="cpu")

# TASK 1: Predict thresholds
predicted_thresholds = trainer.predict_task1(val_loader)
print(f"Predicted thresholds: {predicted_thresholds}")

# TASK 2: Predict runtime for specific thresholds
# Example: Predict runtime at threshold=16 for all circuits
num_samples = len(val_loader.dataset)
threshold_16_classes = torch.full((num_samples,), THRESHOLD_LADDER.index(16))
runtimes_at_16 = trainer.predict_task2(val_loader, threshold_16_classes)
print(f"Runtimes at threshold=16: {runtimes_at_16}")

# Example: Different threshold per circuit
custom_thresholds = np.array([4, 8, 16, 32, 64, 4, 8, ...])  # from judges
custom_classes = torch.tensor([THRESHOLD_LADDER.index(t) for t in custom_thresholds])
custom_runtimes = trainer.predict_task2(val_loader, custom_classes)
print(f"Runtimes: {custom_runtimes}")
```

## Demo Scripts

### Run Full Demo with Training
```bash
python src/gnn/demo_tasks.py
```

This will:
1. Train a model (quick demo training)
2. Demonstrate Task 1 predictions
3. Demonstrate Task 2 predictions with various threshold scenarios

### Run Inference Example
```bash
python src/gnn/inference_example.py
```

This shows how to use a trained model for inference only.

## Key Points

1. **Task 1 is independent:** Predicts optimal threshold for fidelity 0.75
2. **Task 2 requires threshold input:** Can predict runtime for ANY threshold value
3. **Training uses ground truth:** Model learns runtime at ground truth thresholds
4. **Inference is flexible:** Can use predicted thresholds OR custom thresholds for Task 2
5. **Class indices vs values:** Always convert threshold VALUES to class INDICES (0-8) before calling `predict_task2()`

## Challenge Submission Format

When the judges give you inputs:

**For Task 1:**
- They give: Circuit file, target_fidelity=0.75, backend, precision
- You return: Threshold value (e.g., 16)

**For Task 2:**
- They give: Circuit file, threshold value (e.g., 32), backend, precision
- You return: Runtime in seconds (e.g., 1.234)

## Common Patterns

### Pattern 1: Predict both tasks for validation
```python
# Task 1
thresholds = trainer.predict_task1(val_loader)

# Task 2: Runtime at predicted thresholds
threshold_classes = torch.tensor([THRESHOLD_LADDER.index(t) for t in thresholds])
runtimes = trainer.predict_task2(val_loader, threshold_classes)
```

### Pattern 2: Runtime sweep for a single circuit
```python
from torch_geometric.loader import DataLoader as PyGDataLoader

# Create loader with same circuit repeated
circuit = val_loader.dataset[0]
sweep_loader = PyGDataLoader(
    [circuit] * len(THRESHOLD_LADDER),
    batch_size=len(THRESHOLD_LADDER),
)

# All threshold classes 0-8
all_classes = torch.arange(len(THRESHOLD_LADDER))

# Predict runtime at each threshold
runtimes = trainer.predict_task2(sweep_loader, all_classes)

# Plot threshold vs runtime
for thresh, runtime in zip(THRESHOLD_LADDER, runtimes):
    print(f"Threshold {thresh}: {runtime:.4f}s")
```

### Pattern 3: Batch prediction with custom thresholds
```python
# Judges provide a list of (circuit, threshold) pairs
circuits = [circuit1, circuit2, circuit3, ...]
judge_thresholds = [16, 32, 8, 64, ...]  # different threshold per circuit

# Convert to class indices
threshold_classes = torch.tensor([
    THRESHOLD_LADDER.index(t) for t in judge_thresholds
])

# Predict runtimes
loader = PyGDataLoader(circuits, batch_size=16)
runtimes = trainer.predict_task2(loader, threshold_classes)
```

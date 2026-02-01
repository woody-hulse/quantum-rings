# Separate Task 1 & Task 2 Models

## Overview

The models are now **completely separated** into two specialized GNNs:

1. **`ThresholdPredictionGNN`**: Task 1 - Predicts optimal threshold for fidelity 0.75
2. **`RuntimePredictionGNN`**: Task 2 - Predicts runtime given a threshold

## Why Separate Models?

- **Specialized Training**: Each model focuses on one task
- **Independent Optimization**: Different architectures/hyperparameters for each task
- **Challenge Format**: Tasks are evaluated independently
- **Better Performance**: Specialized models often outperform multi-task models

## Task 1: Expected-Score Optimization

The Task 1 model uses **expected-score optimization** to handle the asymmetric scoring function:

### Scoring Rules
- **Guess below true threshold**: 0 points (fidelity violated)
- **Guess at true threshold**: 1.0 points (optimal)
- **Guess N steps above true**: 1.0 / (2^N) points (partial credit)

### Strategy
1. Model outputs a **probability distribution** over threshold classes: P(threshold)
2. For each possible guess G, compute **expected score**: E[score | guess G]
3. Choose the guess that **maximizes expected score** (not just most likely!)

This is different from naive argmax because the scoring is asymmetric - it's worse to guess too low (0 points) than too high (partial credit).

### Example
```python
# Suppose model predicts: P(class 2) = 0.40, P(class 3) = 0.35, ...
# Naive: Choose class 2 (highest probability)
# Smart: Compute expected score for each guess, might choose class 3!
```

See [src/gnn/demo_expected_score.py](demo_expected_score.py) for detailed examples.

## Architecture Differences

### Task 1 Model (ThresholdPredictionGNN)
```
Input: Circuit graph + backend + precision
↓
GNN layers (message passing)
↓
Graph pooling (mean + max + sum)
↓
Global features
↓
MLP → Probability distribution over thresholds
↓
Expected-score decision rule → Optimal threshold (0-8)
```

### Task 2 Model (RuntimePredictionGNN)
```
Input: Circuit graph + THRESHOLD + backend + precision
↓
GNN layers (message passing)
↓
Graph pooling (mean + max + sum)
↓
Global features + Threshold embedding ← KEY DIFFERENCE
↓
MLP → Runtime prediction (log-transformed)
```

**Key Difference**: Task 2 model has a **threshold embedding** that's concatenated with circuit features.

## Training

### Quick Start
```bash
# Train both models separately
python src/gnn/train_separate.py

# With custom parameters
python src/gnn/train_separate.py \
  --hidden-dim 48 \
  --num-layers 3 \
  --dropout 0.25 \
  --n-runs 5 \
  --epochs 100
```

### What Happens During Training

1. **Task 1 Training**:
   - Model learns: Circuit → Optimal threshold
   - Uses ordinal regression loss
   - Evaluated on threshold accuracy

2. **Task 2 Training**:
   - Model learns: Circuit + Threshold → Runtime
   - Uses Huber loss (robust to outliers)
   - Trains with **ground truth thresholds** from data
   - Evaluated on runtime MSE/MAE

## Inference

### Using Trained Models

```python
from gnn.separate_models import create_threshold_model, create_runtime_model
from gnn.train_separate import ThresholdTrainer, RuntimeTrainer
import torch

# Load Task 1 model
threshold_model = create_threshold_model(hidden_dim=48, num_layers=3, dropout=0.25)
# threshold_model.load_state_dict(torch.load("task1_checkpoint.pt"))
threshold_trainer = ThresholdTrainer(threshold_model, device="cpu")

# Load Task 2 model
runtime_model = create_runtime_model(hidden_dim=48, num_layers=3, dropout=0.25)
# runtime_model.load_state_dict(torch.load("task2_checkpoint.pt"))
runtime_trainer = RuntimeTrainer(runtime_model, device="cpu")
```

### Task 1: Predict Thresholds
```python
# Predict optimal thresholds
predicted_thresholds = threshold_trainer.predict(val_loader)
# Returns: np.array([4, 8, 16, 32, ...])
```

### Task 2: Predict Runtimes
```python
import numpy as np
from gnn.dataset import THRESHOLD_LADDER

# Given thresholds (from judges or Task 1 predictions)
given_thresholds = np.array([8, 16, 4, 32, 64])

# Convert to class indices
threshold_classes = torch.tensor([
    THRESHOLD_LADDER.index(t) for t in given_thresholds
])

# Predict runtimes
predicted_runtimes = runtime_trainer.predict(val_loader, threshold_classes)
# Returns: np.array([0.543, 1.234, 0.123, 2.456, 3.123])  # seconds
```

## Complete Workflow

```python
from pathlib import Path
from gnn.separate_models import create_threshold_model, create_runtime_model
from gnn.train_separate import ThresholdTrainer, RuntimeTrainer
from gnn.dataset import create_graph_data_loaders, THRESHOLD_LADDER, GLOBAL_FEAT_DIM
from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM
import torch
import numpy as np

# Setup
project_root = Path(".")
_, val_loader = create_graph_data_loaders(
    data_path=project_root / "data/hackathon_public.json",
    circuits_dir=project_root / "circuits",
    batch_size=16,
)

# ========== TRAIN TASK 1 MODEL ==========
print("Training Task 1 Model...")
threshold_model = create_threshold_model(
    node_feat_dim=NODE_FEAT_DIM,
    edge_feat_dim=EDGE_FEAT_DIM,
    global_feat_dim=GLOBAL_FEAT_DIM,
    hidden_dim=48,
    num_layers=3,
    dropout=0.25,
)
threshold_trainer = ThresholdTrainer(threshold_model, device="cpu")
threshold_trainer.fit(train_loader, val_loader, epochs=100)

# Task 1: Predict thresholds
thresholds = threshold_trainer.predict(val_loader)
print(f"Predicted thresholds: {thresholds}")

# ========== TRAIN TASK 2 MODEL ==========
print("Training Task 2 Model...")
runtime_model = create_runtime_model(
    node_feat_dim=NODE_FEAT_DIM,
    edge_feat_dim=EDGE_FEAT_DIM,
    global_feat_dim=GLOBAL_FEAT_DIM,
    hidden_dim=48,
    num_layers=3,
    dropout=0.25,
)
runtime_trainer = RuntimeTrainer(runtime_model, device="cpu")
runtime_trainer.fit(train_loader, val_loader, epochs=100)

# Task 2: Predict runtimes for predicted thresholds
threshold_classes = torch.tensor([THRESHOLD_LADDER.index(t) for t in thresholds])
runtimes = runtime_trainer.predict(val_loader, threshold_classes)
print(f"Predicted runtimes: {runtimes}")

# Task 2: Predict runtimes for custom thresholds
custom_thresholds = np.array([16] * len(val_loader.dataset))
custom_classes = torch.tensor([THRESHOLD_LADDER.index(t) for t in custom_thresholds])
custom_runtimes = runtime_trainer.predict(val_loader, custom_classes)
print(f"Runtimes at threshold=16: {custom_runtimes}")
```

## Advantages of Separation

1. **Task-Specific Optimization**
   - Task 1: Focus on threshold classification accuracy
   - Task 2: Focus on runtime regression

2. **Independent Hyperparameters**
   - Different learning rates
   - Different model sizes
   - Different regularization

3. **Clearer Code**
   - No conditional logic for task modes
   - Easier to understand and debug
   - Separate training loops

4. **Better Performance**
   - No task interference
   - Each model optimized for its specific objective

## Comparison with Combined Model

### Combined Model (Previous)
```python
# Single model predicts both
threshold_logits, runtime_pred = model(circuit, threshold_class=None)
```
- Multi-task learning
- Shared representations
- Single training loop

### Separate Models (Current)
```python
# Task 1: Threshold only
threshold_logits = threshold_model(circuit)

# Task 2: Runtime only (requires threshold)
runtime_pred = runtime_model(circuit, threshold_class)
```
- Single-task learning
- Specialized representations
- Separate training loops

## Files

- **`separate_models.py`**: Model definitions
  - `ThresholdPredictionGNN`
  - `RuntimePredictionGNN`
  - Factory functions

- **`train_separate.py`**: Training script
  - `ThresholdTrainer`
  - `RuntimeTrainer`
  - Main training loop

## Training Output

```
============================================================
SEPARATE TASK 1 & TASK 2 MODELS
============================================================

Configuration:
  Hidden dim: 48
  Num layers: 3
  ...

============================================================
RUN 1/5
============================================================

--- Training Task 1: Threshold Prediction ---
Training Threshold Model: 100%|██████████| 100/100

Task 1 Results:
  Training time: 45.23s
  Threshold Score: 0.7234

--- Training Task 2: Runtime Prediction ---
Training Runtime Model: 100%|██████████| 100/100

Task 2 Results:
  Training time: 48.56s
  Runtime Score: 0.6789

============================================================
FINAL RESULTS (Target Fidelity = 0.75)
============================================================

Based on 5 runs:

Task 1 (Threshold Prediction):
  Mean Score: 0.7234 ± 0.0189
  Min/Max: 0.7000 / 0.7500

Task 2 (Runtime Prediction):
  Mean Score: 0.6789 ± 0.0234
  Min/Max: 0.6500 / 0.7100
```

## Tips

1. **Train Task 1 First**: Get good threshold predictions, then use them for Task 2 evaluation
2. **Experiment with Different Architectures**: Task 1 and Task 2 might benefit from different model sizes
3. **Save Checkpoints**: Save both models separately for later use
4. **Tune Separately**: Hyperparameter tuning for each task independently

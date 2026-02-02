# Circuit Fingerprint Challenge (iQuHACK 2026)

**Spirit Sprinters:🥇1st Place iQuHACK 2026🥇**

Goal: Predict the simulation cost–accuracy tradeoffs of quantum circuits on an approximate quantum simulator, using Quantum Rings.

## Why this challenge matters

Quantum computing holds the promise to transform entire industries, but today, progress in quantum algorithms is constrained by our ability to execute and test large quantum circuits.

While quantum processing units (QPUs) continue to advance, most real-world development still happens in simulation. The limitation is fundamental: **perfect quantum simulation scales exponentially**, making it infeasible beyond modest circuit sizes.  To get past this, developers rely on **approximate simulation techniques** that generally trade accuracy for runtime and compute resources.

Modern simulators, like those developed by Quantum Rings, allow practitioners to push far beyond traditional limits by exposing configurable  parameters. However, this flexibility introduces a new challenge:

> *How do you know which simulation settings will deliver the fidelity you need—without wasting hours or days of compute?*

This challenge sits at the intersection of **quantum computing and machine learning**. Your task is to develop a model that predicts how a given quantum circuit will behave under different simulation parameters by extracting features from QASM structure, gate composition, entanglement patterns, and topology, and combining them with empirical performance data to **predict**:

- the *minimum approximation threshold* required to meet a target fidelity, and  
- the *expected runtime* to complete execution.

The outcome is quantitative, but the impact will be far more than a leaderboard score. Strong solutions may directly inform **real developer tooling** in an upcoming release, enabling faster iteration, smarter resource allocation, and more accessible quantum experimentation.

## Problem Overview

Predict the minimum MPS threshold and forward wall-clock runtime for quantum circuit simulations given QASM circuit definitions, processor type (CPU/GPU), and precision (single/double).

---

## Features Extracted

### Tabular Features (for CatBoost)

**Structural features** extracted from QASM files:
- Gate counts: 1-qubit gates (H, X, Y, Z, S, T, RX, RY, RZ, etc.), 2-qubit gates (CX, CZ, SWAP), 3-qubit gates (CCX)
- Qubit interaction graph statistics: max/avg degree, degree entropy, connected components, clustering coefficient

**Entanglement complexity features**:
- Graph bandwidth (MPS simulation complexity proxy)
- Treewidth estimation via min-degree heuristic
- Cut crossing metrics (middle-cut crossings, max-cut crossings)
- Entanglement structure: nearest-neighbor ratio, long-range ratio, span Gini coefficient

**Temporal features**:
- Early/late long-range gate ratios
- Light cone spread rate and coverage depth
- Entanglement velocity (cumulative span growth)

**Pattern features**:
- Gate sequence n-grams (H-CX patterns, CX-RZ-CX variational signatures)
- Circuit regularity and repetition scores
- Qubit activity entropy and variance

### Graph Features (for Transformer)

**Node features** (per qubit, 22 dimensions):
- Log-transformed 1-qubit gate counts by type
- 2-qubit gate involvement count
- Normalized position in qubit register
- Temporal: first/last 2Q gate position, activity window, unique neighbors, average interaction span

**Edge features** (per gate, 4 dimensions):
- Normalized temporal position in circuit
- Gate parameter value
- Qubit distance (normalized)
- Cumulative 2Q gate count at edge position

**Global features**:
- Circuit metadata: qubit count, gate counts, gate density
- Execution context: backend (CPU/GPU), precision (single/double)
- Threshold value (for duration model)
- Circuit family one-hot encoding (20 families)

---

## Modeling Strategy

### Threshold Prediction: CatBoost Classifier

- **Task**: 9-class classification (thresholds: 1, 2, 4, 8, 16, 32, 64, 128, 256)
- **Model**: CatBoost with class-weighted training to handle imbalanced threshold distribution
- **Decision rule**: Expected score maximization over predicted class probabilities using the asymmetric challenge scoring matrix (underprediction = 0, overprediction = 2^(-steps))
- **Conservative bias**: Optional bias toward higher thresholds to minimize costly underpredictions

### Duration Prediction: Graph Transformer

- **Task**: Regression on log₂(wall time)
- **Architecture**: Edge-aware multi-head attention with gate-type embeddings
  - Attention bias derived from gate types and edge features
  - Random-walk positional encoding for graph structure
  - 4 transformer layers, 4 attention heads, 64 hidden dimensions
- **Pooling**: Concatenation of mean, max, and sum graph-level representations
- **Output**: Combined graph embedding + global features → MLP → runtime prediction

**Why Graph Transformer for duration?** Quantum simulation runtime scales with entanglement structure. Self-attention naturally captures how entanglement propagates through circuits, with gate-specific attention biases learning which interactions dominate complexity.

---

## Validation Approach

- **Data split**: Train/validation split on hackathon public dataset
- **Primary metric**: Challenge scoring function
  - Threshold score: 0 if underpredicted, else 2^(−steps over)
  - Runtime score: min(r, 1/r) where r = predicted/actual time
  - Combined: threshold_score × runtime_score per task
- **Secondary metrics**: Accuracy, MAE, underprediction rate for threshold; RMSE, MAPE for duration
- **Cross-validation**: Stratified by circuit family and threshold class for robust hyperparameter tuning

---

## Known Limitations

1. **Threshold class imbalance**: Lower thresholds (1, 2) are rare in training data, leading to potential underprediction on simple circuits

2. **Circuit family generalization**: Performance may degrade on holdout circuits from unseen algorithm families not represented in training

3. **Graph size scaling**: Transformer attention scales O(n²) with qubit count; very large circuits may require batch size adjustments

4. **Runtime variance**: Wall-clock time predictions inherit variance from hardware/system load during data collection; model captures expected runtime rather than exact timing

5. **Feature coverage**: Some advanced circuit patterns (e.g., mid-circuit measurement, dynamic circuits) may not be fully captured by current feature engineering

---

## Final Model Selection

| Task | Model | Key Hyperparameters |
|------|-------|---------------------|
| Threshold | CatBoost | depth=6, iterations=100, L2=3.0, class weights |
| Duration | Graph Transformer | layers=4, heads=4, hidden=64, dropout=0.2 |

## Docs

Please find deeper documentation on the Challenge, the Data, the Submissions, and more, using the following links:

- `docs/CHALLENGE.md` — motivation, “what is threshold?”, and feature ideas
- `docs/DATA.md` — dataset format, fields, and interpretation
- `docs/SUBMISSION.md` — exact submission format + local validation instructions
- `docs/CIRCUITS.md` — circuit library notes + provenance guidance
- `docs/THIRD_PARTY.md` — third‑party attributions

## License
This repository is MIT-licensed (see `LICENSE`).

Submission code and writeup are separate; see `docs/SUBMISSION.md` for suggested licensing language.

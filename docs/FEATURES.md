# Feature Documentation

This document describes all features extracted from QASM circuit files in `src/data_loader.py`.

---

## Basic Circuit Statistics

| Feature | Description |
|---------|-------------|
| `n_lines` | Number of non-empty, non-comment lines in the QASM file |
| `n_qubits` | Number of qubits declared in the circuit (`qreg q[N]`) |
| `n_cx` | Count of CNOT (CX) gates |
| `n_cz` | Count of controlled-Z (CZ) gates |
| `n_swap` | Count of SWAP gates |
| `n_ccx` | Count of Toffoli (CCX) gates |
| `n_2q_gates` | Total 2-qubit gates (cx + cz + swap + ccx) |
| `n_1q_gates` | Total 1-qubit gates (h, x, y, z, s, t, rx, ry, rz, u1, u2, u3) |
| `n_measure` | Count of measurement operations |
| `n_barrier` | Count of barrier instructions |
| `n_custom_gates` | Count of custom gate definitions (`gate ...`) |
| `n_h` | Count of Hadamard gates |
| `n_rx`, `n_ry`, `n_rz` | Counts of rotation gates |

---

## Qubit Span Features

These measure the "reach" of 2-qubit gates across the qubit register.

| Feature | Description |
|---------|-------------|
| `n_unique_pairs` | Number of distinct qubit pairs that interact via 2Q gates |
| `avg_span` | Average distance between qubits in 2Q gates (e.g., `cx q[0],q[5]` has span 5) |
| `max_span` | Maximum span across all 2Q gates |
| `min_span` | Minimum span across all 2Q gates |
| `span_std` | Standard deviation of spans (high = varied interaction distances) |

**Why it matters:** Larger spans require more entanglement to track in MPS simulation.

---

## Gate Density Features

| Feature | Description |
|---------|-------------|
| `gate_density` | 2Q gates per qubit (`n_2q_gates / n_qubits`) — circuit "intensity" |
| `gate_ratio_2q` | Fraction of all gates that are 2-qubit gates |

---

## Interaction Graph Features

Built from the qubit interaction graph where nodes = qubits, edges = 2Q gate pairs.

| Feature | Description |
|---------|-------------|
| `max_degree` | Maximum number of unique qubits any single qubit interacts with |
| `avg_degree` | Average degree across all qubits |
| `degree_entropy` | Entropy of degree distribution (high = uniform connectivity) |
| `n_connected_components` | Number of disconnected subgraphs (usually 1 for useful circuits) |
| `clustering_coeff` | Graph clustering coefficient — how often qubit neighbors also neighbor each other |

**Why it matters:** Dense, highly-connected graphs generate more entanglement.

---

## Depth Features

| Feature | Description |
|---------|-------------|
| `estimated_depth` | Estimated circuit depth (max gate layers on any qubit) |
| `depth_per_qubit` | Depth normalized by qubit count |

**Why it matters:** Deeper circuits accumulate more entanglement.

---

## Cut Features (Entanglement Pressure)

Analyze how gates cross bipartitions of the qubit register — critical for MPS bond dimension.

| Feature | Description |
|---------|-------------|
| `middle_cut_crossings` | Number of 2Q gates crossing the middle cut (qubit N/2) |
| `cut_crossing_ratio` | Fraction of 2Q gates that cross the middle cut |
| `max_cut_crossings` | Maximum crossings at any single cut position |

**Why it matters:** More cut crossings → higher entanglement across the cut → larger bond dimension needed.

---

## Graph Bandwidth Features

| Feature | Description |
|---------|-------------|
| `graph_bandwidth` | Maximum span in the interaction graph (same as `max_span`) |
| `normalized_bandwidth` | Bandwidth / n_qubits (0 to 1 scale) |
| `bandwidth_squared` | Squared bandwidth (emphasizes large spans) |

**Why it matters:** Bandwidth directly relates to minimum MPS bond dimension for exact simulation.

---

## Temporal Features

Analyze *when* in the circuit long-range gates appear.

| Feature | Description |
|---------|-------------|
| `early_longrange_ratio` | Fraction of gates in the first 1/3 that are long-range (span > N/4) |
| `late_longrange_ratio` | Fraction of gates in the last 1/3 that are long-range |
| `longrange_temporal_center` | Normalized position (0-1) of long-range gates in the circuit |
| `entanglement_velocity` | Rate of span accumulation (how fast entanglement grows) |

**Why it matters:** Early long-range gates cause entanglement to persist and accumulate throughout the circuit.

---

## Qubit Activity Features

Measure how evenly work is distributed across qubits.

| Feature | Description |
|---------|-------------|
| `qubit_activity_entropy` | Normalized entropy of qubit usage (1 = uniform, 0 = concentrated) |
| `qubit_activity_variance` | Variance in gate counts per qubit |
| `qubit_activity_max_ratio` | Fraction of all gate refs on the most-active qubit |
| `active_qubit_fraction` | Fraction of qubits that participate in at least one gate |

**Why it matters:** Uniform activity often indicates structured algorithms with predictable entanglement.

---

## Gate Pattern Features

Detect algorithmic signatures from gate sequences.

| Feature | Description |
|---------|-------------|
| `cx_chain_max_length` | Longest consecutive sequence of CX gates |
| `h_cx_pattern_count` | Count of H→CX patterns (Bell/GHZ preparation signature) |
| `cx_rz_cx_pattern_count` | Count of CX→rotation→CX patterns (variational circuit signature) |
| `rotation_density` | Fraction of gates that are parameterized rotations |
| `gate_type_entropy` | Entropy of gate type distribution (high = diverse gate set) |
| `cx_h_ratio` | Ratio of CX to H gates |

**Why it matters:** Different algorithms have characteristic gate patterns.

---

## Light Cone Features

Simulate how information spreads from the center qubit through the circuit.

| Feature | Description |
|---------|-------------|
| `light_cone_spread_rate` | How quickly the light cone expands (qubits reached per gate) |
| `light_cone_half_coverage_depth` | Normalized depth to reach 50% of qubits |
| `final_light_cone_size` | Fraction of qubits reached by end of circuit |

**Why it matters:** Fast light cone expansion = rapid entanglement growth = harder for MPS.

---

## Entanglement Structure Features

Analyze the pattern of qubit interactions.

| Feature | Description |
|---------|-------------|
| `nearest_neighbor_ratio` | Fraction of 2Q gates with span = 1 (local interactions) |
| `long_range_ratio` | Fraction of 2Q gates with span ≥ N/3 (global interactions) |
| `span_gini_coefficient` | Inequality in span distribution (0 = uniform, 1 = concentrated) |
| `weighted_span_sum` | Sum of squared spans, normalized (emphasizes long-range) |

**Why it matters:** High nearest-neighbor ratio → easier for MPS. High long-range ratio → harder.

---

## Circuit Regularity Features

Detect repetitive/structured patterns.

| Feature | Description |
|---------|-------------|
| `pattern_repetition_score` | How often 4-gate patterns repeat (high = regular structure) |
| `barrier_regularity` | Regularity of barrier spacing (high = evenly spaced layers) |
| `layer_uniformity` | Placeholder for layer structure analysis |

**Why it matters:** Regular circuits (variational ansätze, QFT) have predictable entanglement behavior.

---

## Summary

**Total features:** ~55 numeric features extracted from QASM

**Most predictive for MPS threshold:**
1. Cut crossing features (direct entanglement proxy)
2. Bandwidth and span features
3. Light cone spread rate
4. Long-range ratio
5. Graph connectivity features

**Circuit family is also included** as a 20-class one-hot encoding, providing algorithm-level prior knowledge.

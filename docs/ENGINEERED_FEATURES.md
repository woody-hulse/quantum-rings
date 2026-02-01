# Engineered Features (Not in Original Data)

All features below are derived from the raw dataset (circuits, results, QASM files). The original data provides: circuit `file`, `backend`, `precision`, `family`, `n_qubits` (in circuits), `selected_threshold`, `forward.run_wall_s`, and threshold sweep data. Everything in these tables is computed for modeling.

---

## 1. Run / task encoding

Encodings of fields that come from the result record (backend, precision, threshold, family).

| Name | Description |
|------|-------------|
| `backend_idx` | 0 = CPU, 1 = GPU. |
| `precision_idx` | 0 = single, 1 = double. |
| `log2_threshold` | log₂(selected_threshold); used as input for duration prediction. |
| `family_onehot` | One-hot over circuit family (20 categories, e.g. QFT, GHZ, QAOA). |

---

## 2. QASM: Basic circuit statistics

Counts and sizes extracted from the QASM file (not stored as such in the original JSON).

| Name | Description |
|------|-------------|
| `n_lines` | Number of non-empty, non-comment lines in the QASM file. |
| `n_qubits` | Number of qubits declared in the circuit (`qreg q[N]`). |
| `n_cx` | Count of CNOT (CX) gates. |
| `n_cz` | Count of controlled-Z (CZ) gates. |
| `n_swap` | Count of SWAP gates. |
| `n_ccx` | Count of Toffoli (CCX) gates. |
| `n_2q_gates` | Total 2-qubit gates (cx + cz + swap + ccx). |
| `n_1q_gates` | Total 1-qubit gates (h, x, y, z, s, t, rx, ry, rz, u1, u2, u3). |
| `n_measure` | Count of measurement operations. |
| `n_barrier` | Count of barrier instructions. |
| `n_custom_gates` | Count of custom gate definitions (`gate ...`). |
| `n_h` | Count of Hadamard gates. |
| `n_rx`, `n_ry`, `n_rz` | Counts of parameterized rotation gates. |

---

## 3. QASM: Qubit span

Measures the “reach” of 2-qubit gates across the qubit register (distance between qubit indices).

| Name | Description |
|------|-------------|
| `n_unique_pairs` | Number of distinct qubit pairs that interact via 2Q gates. |
| `avg_span` | Mean distance between qubits in 2Q gates. |
| `max_span` | Maximum span over all 2Q gates. |
| `min_span` | Minimum span over all 2Q gates. |
| `span_std` | Standard deviation of spans. |

---

## 4. QASM: Gate density

| Name | Description |
|------|-------------|
| `gate_density` | 2Q gates per qubit (`n_2q_gates / n_qubits`). |
| `gate_ratio_2q` | Fraction of all gates that are 2-qubit gates. |

---

## 5. QASM: Interaction graph

Built from the qubit interaction graph (nodes = qubits, edges = 2Q gate pairs).

| Name | Description |
|------|-------------|
| `max_degree` | Maximum number of distinct qubits any single qubit interacts with. |
| `avg_degree` | Average degree over all qubits. |
| `degree_entropy` | Normalized entropy of the degree distribution. |
| `n_connected_components` | Number of disconnected subgraphs. |
| `clustering_coeff` | Graph clustering coefficient. |
| `max_component_size` | Size of the largest connected component. |
| `component_entropy` | Normalized entropy of component size distribution. |

---

## 6. QASM: Depth

| Name | Description |
|------|-------------|
| `estimated_depth` | Estimated circuit depth (max gate layers on any qubit). |
| `depth_per_qubit` | Depth divided by qubit count. |

---

## 7. QASM: Cut (entanglement pressure)

Counts of 2Q gates crossing bipartitions of the qubit register; relevant for MPS bond dimension.

| Name | Description |
|------|-------------|
| `middle_cut_crossings` | Number of 2Q gates crossing the middle cut (qubit N/2). |
| `cut_crossing_ratio` | Fraction of 2Q gates that cross the middle cut. |
| `max_cut_crossings` | Maximum crossings at any single cut position. |

---

## 8. QASM: Graph bandwidth

| Name | Description |
|------|-------------|
| `graph_bandwidth` | Maximum span in the interaction graph (same as `max_span`). |
| `normalized_bandwidth` | Bandwidth / n_qubits. |
| `bandwidth_squared` | Squared bandwidth. |

---

## 9. QASM: Temporal

When in the circuit long-range gates appear (early vs late).

| Name | Description |
|------|-------------|
| `early_longrange_ratio` | Fraction of gates in the first third that are long-range (span > N/4). |
| `late_longrange_ratio` | Fraction of gates in the last third that are long-range. |
| `longrange_temporal_center` | Normalized position (0–1) of long-range gates in the circuit. |
| `entanglement_velocity` | Mean increase in cumulative span per 2Q gate. |

---

## 10. QASM: Qubit activity

Distribution of gate involvement across qubits.

| Name | Description |
|------|-------------|
| `qubit_activity_entropy` | Normalized entropy of qubit usage (1 = uniform). |
| `qubit_activity_variance` | Variance of gate counts per qubit. |
| `qubit_activity_max_ratio` | Fraction of all gate references on the most active qubit. |
| `active_qubit_fraction` | Fraction of qubits that participate in at least one gate. |

---

## 11. QASM: Gate patterns

Sequence-level patterns (n-grams) indicative of algorithm type.

| Name | Description |
|------|-------------|
| `cx_chain_max_length` | Longest consecutive run of CX gates. |
| `h_cx_pattern_count` | Count of H→CX patterns (Bell/GHZ-like). |
| `cx_rz_cx_pattern_count` | Count of CX→rotation→CX patterns (e.g. variational). |
| `rotation_density` | Fraction of gates that are parameterized rotations. |
| `gate_type_entropy` | Entropy of gate type distribution. |
| `cx_h_ratio` | Ratio of CX count to H count. |

---

## 12. QASM: Light cone

How quickly information spreads from a central qubit through 2Q gates.

| Name | Description |
|------|-------------|
| `light_cone_spread_rate` | Rate of growth of “reached” qubits per 2Q gate. |
| `light_cone_half_coverage_depth` | Normalized depth to reach half of the qubits. |
| `final_light_cone_size` | Fraction of qubits reached by the end of the circuit. |

---

## 13. QASM: Entanglement structure

| Name | Description |
|------|-------------|
| `nearest_neighbor_ratio` | Fraction of 2Q gates with span = 1. |
| `long_range_ratio` | Fraction of 2Q gates with span ≥ N/3. |
| `span_gini_coefficient` | Inequality in span distribution (0 = even, 1 = concentrated). |
| `weighted_span_sum` | Sum of squared spans, normalized by n_qubits². |

---

## 14. QASM: Circuit regularity

| Name | Description |
|------|-------------|
| `pattern_repetition_score` | Max repeat count of 4-gate patterns over the circuit. |
| `barrier_regularity` | Regularity of barrier spacing (1 − normalized std of gaps). |
| `layer_uniformity` | Placeholder for layer-structure regularity (fixed 0.5). |

---

## 15. QASM: Treewidth

| Name | Description |
|------|-------------|
| `treewidth_min_degree` | Treewidth estimate from min-degree elimination on the interaction graph. |

---

## 16. Component-model features (gate-level)

Used by the learned component model: per-gate parsing (type, span, layer, position, middle cut) then aggregated. Backend/precision are encoded; threshold is not in this vector.

| Name | Description |
|------|-------------|
| `n_qubits` | Qubit count. |
| `n_gates` | Total gate count. |
| `n_cx`, `n_cz`, `n_swap` | Counts by 2Q type. |
| `n_1q` | Total 1-qubit gate count. |
| `avg_span` | Mean span over 2Q gates. |
| `max_span` | Max span. |
| `span_sum` | Sum of spans. |
| `span_squared_sum` | Sum of squared spans. |
| `n_crossing` | Number of gates crossing the middle cut. |
| `crossing_ratio` | n_crossing / n_gates. |
| `early_entangling` | Count of 2Q gates in the first 30% of positions. |
| `late_entangling` | Count of 2Q gates in the last 30% of positions. |
| `max_layer` | Maximum gate layer index. |
| `weighted_span` | Sum of span × (1 + position/n_gates) over 2Q gates. |
| `n_2q` | Total 2Q gate count. |
| `n_2q_per_qubit` | n_2q / n_qubits. |
| `max_span_per_qubit` | max_span / n_qubits. |
| `span_squared_sum_per_qubit2` | span_squared_sum / n_qubits². |
| `is_gpu` | 1 if backend is GPU else 0. |
| `is_double` | 1 if precision is double else 0. |
| `complexity_proxy` | n_2q × max_span × max_layer / n_qubits. |
| `log1p_n_gates` | log(1 + n_gates). |

---

## 17. Analytical cost features (runtime regression)

Derived from the analytical cost model’s cost dict (total_cost, entanglement_cost, max_layer_cost, cut_crossing_cost) plus run context. Used for runtime prediction in the analytical component wrapper.

| Name | Description |
|------|-------------|
| `total_cost` | Sum of per-gate fidelity costs from the analytical model. |
| `entanglement_cost` | Cost component from entanglement (span-weighted). |
| `max_layer_cost` | Cost component from the worst layer. |
| `cut_crossing_cost` | Cost component from gates crossing the middle cut. |
| `log1p_total_cost` | log(1 + total_cost). |
| `is_gpu` | 1 if GPU else 0. |
| `is_double` | 1 if double precision else 0. |
| `total_cost_double_scale` | total_cost × 1.5 if double else total_cost. |
| `total_cost_gpu_scale` | total_cost × 0.3 if GPU else total_cost. |
| `log2_threshold` | log₂(selected threshold). |

---

## 18. GNN: Node features

Per-qubit features in the graph built from QASM (rich mode). Gate types use the same 1Q/2Q/3Q sets as in the graph builder.

| Name | Description |
|------|-------------|
| `node_1q_log` | log(1 + count) per 1Q gate type (15 dimensions). |
| `node_2q_log` | log(1 + 2Q involvement count) per qubit. |
| `node_positions` | Normalized qubit index in [0, 1]. |
| `node_first_2q_pos` | Normalized position of first 2Q gate involving this qubit (1 if none). |
| `node_last_2q_pos` | Normalized position of last 2Q gate involving this qubit (0 if none). |
| `node_activity_window` | node_last_2q_pos − node_first_2q_pos. |
| `node_degree` | Number of unique qubits this qubit interacts with, normalized by (n_qubits−1). |
| `node_avg_span` | Mean interaction distance (span) for this qubit over 2Q gates. |

---

## 19. GNN: Edge features

Per-edge features; gate type is stored separately as a categorical index.

| Name | Description |
|------|-------------|
| `edge_position` | Normalized temporal position of the gate (0–1). |
| `edge_params` | First gate parameter (e.g. rotation angle) or 0. |
| `edge_qubit_dist` | |q_i − q_j| / (n_qubits − 1) for the edge. |
| `edge_cumulative_idx` | Cumulative 2Q gate index at this edge, normalized by total gates. |

---

## 20. GNN: Global features

Circuit-level and run-level inputs concatenated for the GNN.

| Name | Description |
|------|-------------|
| `n_qubits_norm` | n_qubits / 130. |
| `n_gates_norm` | Total gate count / 1000. |
| `n_2q_gates_norm` | 2Q gate count / 500. |
| `gate_density_norm` | gate_density / 10. |
| `backend_idx` | 0 = CPU, 1 = GPU. |
| `precision_idx` | 0 = single, 1 = double. |
| `log2_threshold_norm` | log2_threshold / 8 (if provided). |
| `family_onehot` | One-hot over circuit family (20 dimensions when used). |

---

## Summary

| Group | Count | Used in |
|-------|--------|---------|
| Run/task encoding | 4 (incl. 20-dim one-hot) | Data loader, GNN global |
| QASM basic | 14 | `extract_qasm_features`, tabular models |
| QASM span | 5 | QASM features |
| QASM density | 2 | QASM features |
| QASM interaction graph | 7 | QASM features |
| QASM depth | 2 | QASM features |
| QASM cut | 3 | QASM features |
| QASM bandwidth | 3 | QASM features |
| QASM temporal | 4 | QASM features |
| QASM qubit activity | 4 | QASM features |
| QASM gate patterns | 6 | QASM features |
| QASM light cone | 3 | QASM features |
| QASM entanglement structure | 4 | QASM features |
| QASM regularity | 3 | QASM features |
| QASM treewidth | 1 | QASM features |
| Component-model (gate-level) | 24 | LearnedComponentModel |
| Analytical cost | 10 | AnalyticalCost runtime regression |
| GNN node | 8 (multi-dim) | Graph models |
| GNN edge | 4 | Graph models |
| GNN global | 8 (+ family) | Graph models |

Total distinct **scalar** engineered names in the tables above is on the order of 120+ (excluding repeated names like `n_qubits` / `backend_idx` that appear in multiple contexts).

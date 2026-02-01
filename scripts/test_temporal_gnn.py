#!/usr/bin/env python3
"""
Test script for the Temporal GNN implementation.

Verifies:
1. Model architecture builds correctly
2. Forward pass works with sample data
3. Training loop runs without errors
4. Model integrates with existing data loaders
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np


def test_temporal_graph_builder():
    """Test the temporal graph builder."""
    print("=" * 60)
    print("Testing Temporal Graph Builder")
    print("=" * 60)
    
    from gnn.temporal_graph_builder import (
        build_temporal_graph,
        compute_circuit_layers,
        compute_entanglement_trajectory,
        TEMPORAL_NODE_FEAT_DIM,
        TEMPORAL_EDGE_FEAT_DIM,
    )
    
    test_qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg c[4];
    h q[0];
    cx q[0],q[1];
    cx q[1],q[2];
    cx q[0],q[3];
    rz(0.5) q[3];
    cx q[2],q[3];
    h q[0];
    """
    
    result = build_temporal_graph(test_qasm, "CPU", "single")
    
    print(f"  Node features: {result['x'].shape}")
    print(f"  Expected node dim: {TEMPORAL_NODE_FEAT_DIM}")
    print(f"  Edge index: {result['edge_index'].shape}")
    print(f"  Edge features: {result['edge_attr'].shape}")
    print(f"  Expected edge dim: {TEMPORAL_EDGE_FEAT_DIM}")
    print(f"  Global features: {result['global_features'].shape}")
    print(f"  Circuit layers: {result['n_layers']}")
    print(f"  Long-range gates: {result['n_longrange']}")
    
    assert result['x'].shape[1] == TEMPORAL_NODE_FEAT_DIM
    assert result['edge_attr'].shape[1] == TEMPORAL_EDGE_FEAT_DIM
    
    print("  [PASS] Temporal graph builder works correctly\n")


def test_temporal_gnn_model():
    """Test the Temporal GNN model architecture."""
    print("=" * 60)
    print("Testing Temporal GNN Model Architecture")
    print("=" * 60)
    
    from gnn.temporal_model import (
        create_temporal_gnn_model,
        TemporalQuantumCircuitGNN,
        TemporalQuantumCircuitGNNThresholdClass,
        TEMPORAL_GNN_CONFIGS,
    )
    from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE
    
    batch_size = 4
    n_nodes = 20
    n_edges = 50
    
    x = torch.randn(n_nodes, NODE_FEAT_DIM)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, EDGE_FEAT_DIM)
    edge_gate_type = torch.randint(0, 30, (n_edges,))
    batch = torch.repeat_interleave(torch.arange(batch_size), n_nodes // batch_size)
    global_features = torch.randn(batch_size, GLOBAL_FEAT_DIM_BASE + 1 + 20)
    
    print("  Testing duration model...")
    model = create_temporal_gnn_model(
        model_type="duration",
        **TEMPORAL_GNN_CONFIGS["small"],
    )
    
    output = model(x, edge_index, edge_attr, edge_gate_type, batch, global_features)
    print(f"    Output shape: {output.shape}")
    assert output.shape == (batch_size,)
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("    [PASS] Duration model forward pass works\n")
    
    print("  Testing threshold classification model...")
    global_features_no_threshold = torch.randn(batch_size, GLOBAL_FEAT_DIM_BASE + 20)
    
    model = create_temporal_gnn_model(
        model_type="threshold",
        **TEMPORAL_GNN_CONFIGS["small"],
    )
    
    output = model(x, edge_index, edge_attr, edge_gate_type, batch, global_features_no_threshold)
    print(f"    Output shape: {output.shape}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("    [PASS] Threshold classification model forward pass works\n")
    
    print("  Testing class probability output...")
    probs = model.get_class_probs(x, edge_index, edge_attr, edge_gate_type, batch, global_features_no_threshold)
    print(f"    Probability shape: {probs.shape}")
    print(f"    Probability sum: {probs.sum(dim=-1)}")
    assert probs.shape == (batch_size, 9)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    print("    [PASS] Class probabilities sum to 1\n")


def test_temporal_gnn_gradients():
    """Test that gradients flow correctly through the model."""
    print("=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    from gnn.temporal_model import create_temporal_gnn_model, TEMPORAL_GNN_CONFIGS
    from gnn.graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE
    
    batch_size = 2
    n_nodes = 10
    n_edges = 20
    
    x = torch.randn(n_nodes, NODE_FEAT_DIM, requires_grad=True)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr = torch.randn(n_edges, EDGE_FEAT_DIM)
    edge_gate_type = torch.randint(0, 30, (n_edges,))
    batch = torch.repeat_interleave(torch.arange(batch_size), n_nodes // batch_size)
    global_features = torch.randn(batch_size, GLOBAL_FEAT_DIM_BASE + 1 + 20)
    
    model = create_temporal_gnn_model(
        model_type="duration",
        **TEMPORAL_GNN_CONFIGS["small"],
    )
    
    output = model(x, edge_index, edge_attr, edge_gate_type, batch, global_features)
    loss = output.mean()
    loss.backward()
    
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    
    assert grad_norm > 0, "Gradients should be non-zero"
    print("  [PASS] Gradients flow correctly\n")


def test_with_real_data():
    """Test with actual circuit data if available."""
    print("=" * 60)
    print("Testing with Real Circuit Data")
    print("=" * 60)
    
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if not data_path.exists():
        print("  [SKIP] Data file not found")
        return
    
    from gnn.dataset import create_graph_data_loaders, create_threshold_class_graph_data_loaders
    from gnn.temporal_model import create_temporal_gnn_model, TEMPORAL_GNN_CONFIGS
    
    print("  Loading data loaders...")
    train_loader, val_loader = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=8,
        val_fraction=0.2,
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    batch = next(iter(train_loader))
    print(f"  Batch node features: {batch.x.shape}")
    print(f"  Batch edge features: {batch.edge_attr.shape}")
    print(f"  Batch global features: {batch.global_features.shape}")
    
    model = create_temporal_gnn_model(
        model_type="duration",
        node_feat_dim=batch.x.size(-1),
        edge_feat_dim=batch.edge_attr.size(-1),
        global_feat_dim=batch.global_features.size(-1),
        **TEMPORAL_GNN_CONFIGS["small"],
    )
    
    print("  Running forward pass on real batch...")
    output = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.edge_gate_type,
        batch.batch,
        batch.global_features,
    )
    print(f"  Output shape: {output.shape}")
    print(f"  Predictions: {output[:5]}")
    print(f"  Targets: {batch.log2_runtime[:5]}")
    print("  [PASS] Real data forward pass works\n")
    
    print("  Testing threshold classification...")
    train_loader_cls, _ = create_threshold_class_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=8,
        val_fraction=0.2,
    )
    
    batch = next(iter(train_loader_cls))
    model = create_temporal_gnn_model(
        model_type="threshold",
        node_feat_dim=batch.x.size(-1),
        edge_feat_dim=batch.edge_attr.size(-1),
        global_feat_dim=batch.global_features.size(-1),
        **TEMPORAL_GNN_CONFIGS["small"],
    )
    
    output = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.edge_gate_type,
        batch.batch,
        batch.global_features,
    )
    probs = model.get_class_probs(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.edge_gate_type,
        batch.batch,
        batch.global_features,
    )
    pred_classes = probs.argmax(dim=-1)
    
    print(f"  Logit output shape: {output.shape}")
    print(f"  Probability shape: {probs.shape}")
    print(f"  Predicted classes: {pred_classes[:5]}")
    print(f"  Target classes: {batch.threshold_class[:5]}")
    print("  [PASS] Threshold classification works\n")


def test_training_step():
    """Test a single training step."""
    print("=" * 60)
    print("Testing Training Step")
    print("=" * 60)
    
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if not data_path.exists():
        print("  [SKIP] Data file not found")
        return
    
    from gnn.dataset import create_graph_data_loaders
    from gnn.temporal_model import create_temporal_gnn_model, TEMPORAL_GNN_CONFIGS
    import torch.nn.functional as F
    
    train_loader, _ = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=8,
        val_fraction=0.2,
    )
    
    batch = next(iter(train_loader))
    
    model = create_temporal_gnn_model(
        model_type="duration",
        node_feat_dim=batch.x.size(-1),
        edge_feat_dim=batch.edge_attr.size(-1),
        global_feat_dim=batch.global_features.size(-1),
        **TEMPORAL_GNN_CONFIGS["small"],
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    optimizer.zero_grad()
    
    output = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.edge_gate_type,
        batch.batch,
        batch.global_features,
    )
    
    loss = F.l1_loss(output, batch.log2_runtime)
    print(f"  Initial loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    
    output2 = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.edge_gate_type,
        batch.batch,
        batch.global_features,
    )
    loss2 = F.l1_loss(output2, batch.log2_runtime)
    print(f"  Loss after step: {loss2.item():.4f}")
    
    print("  [PASS] Training step completed successfully\n")


def test_wrapper_models():
    """Test the wrapper model classes."""
    print("=" * 60)
    print("Testing Wrapper Model Classes")
    print("=" * 60)
    
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    if not data_path.exists():
        print("  [SKIP] Data file not found")
        return
    
    from models.temporal_gnn_model import (
        TemporalGNNDurationModel,
        TemporalGNNThresholdClassModel,
    )
    from gnn.dataset import create_graph_data_loaders, create_threshold_class_graph_data_loaders
    
    print("  Testing duration model wrapper...")
    train_loader, val_loader = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=8,
        val_fraction=0.2,
    )
    
    model = TemporalGNNDurationModel(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        epochs=2,
        patience=5,
    )
    print(f"    Model name: {model.name}")
    
    history = model.fit(train_loader, val_loader, verbose=True, show_progress=False)
    print(f"    Training completed with {len(history['history'])} epochs")
    
    metrics = model.evaluate(val_loader)
    print(f"    Val MAE: {metrics['runtime_mae']:.4f}")
    print("  [PASS] Duration model wrapper works\n")
    
    print("  Testing threshold model wrapper...")
    train_loader_cls, val_loader_cls = create_threshold_class_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=8,
        val_fraction=0.2,
    )
    
    model = TemporalGNNThresholdClassModel(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        epochs=2,
        patience=5,
    )
    print(f"    Model name: {model.name}")
    
    history = model.fit(train_loader_cls, val_loader_cls, verbose=True, show_progress=False)
    print(f"    Training completed with {len(history['history'])} epochs")
    
    metrics = model.evaluate(val_loader_cls)
    print(f"    Val accuracy: {metrics['threshold_accuracy']:.4f}")
    print("  [PASS] Threshold model wrapper works\n")


def main():
    print("\n" + "=" * 60)
    print(" TEMPORAL GNN TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_temporal_graph_builder()
        test_temporal_gnn_model()
        test_temporal_gnn_gradients()
        test_with_real_data()
        test_training_step()
        test_wrapper_models()
        
        print("=" * 60)
        print(" ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAILED] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

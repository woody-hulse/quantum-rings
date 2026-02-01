#!/usr/bin/env python3
"""
Test script to validate the data loading functionality.

Runs various checks to ensure data is loaded correctly and is compatible with PyTorch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from collections import Counter

from data_loader import (
    load_hackathon_data,
    extract_qasm_features,
    QuantumCircuitDataset,
    create_data_loaders,
    get_feature_statistics,
    compute_min_threshold,
    THRESHOLD_LADDER,
)


def test_load_hackathon_data():
    """Test basic data loading."""
    print("=" * 60)
    print("TEST: load_hackathon_data")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    
    circuits, results = load_hackathon_data(data_path)
    
    print(f"Loaded {len(circuits)} circuits")
    print(f"Loaded {len(results)} result entries")
    
    assert len(circuits) > 0, "No circuits loaded"
    assert len(results) > 0, "No results loaded"
    
    print("\nSample circuits:")
    for c in circuits[:3]:
        print(f"  - {c.file}: {c.family}, {c.n_qubits} qubits")
    
    print("\nSample results:")
    for r in results[:3]:
        print(f"  - {r.file}: {r.backend}/{r.precision}, status={r.status}")
        print(f"    threshold_sweep entries: {len(r.threshold_sweep)}")
        print(f"    selected_threshold: {r.selected_threshold}")
        print(f"    forward_wall_s: {r.forward_wall_s}")
    
    status_counts = Counter(r.status for r in results)
    print(f"\nStatus distribution: {dict(status_counts)}")
    
    backend_counts = Counter(r.backend for r in results)
    print(f"Backend distribution: {dict(backend_counts)}")
    
    precision_counts = Counter(r.precision for r in results)
    print(f"Precision distribution: {dict(precision_counts)}")
    
    print("\n✓ load_hackathon_data passed")
    return circuits, results


def test_extract_qasm_features():
    """Test QASM feature extraction."""
    print("\n" + "=" * 60)
    print("TEST: extract_qasm_features")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    circuits_dir = project_root / "circuits"
    
    test_files = [
        "ghz_indep_qiskit_15.qasm",
        "qft_indep_qiskit_15.qasm",
        "grover-noancilla_indep_qiskit_7.qasm",
    ]
    
    for fname in test_files:
        fpath = circuits_dir / fname
        if fpath.exists():
            features = extract_qasm_features(fpath)
            print(f"\n{fname}:")
            print(f"  n_qubits: {features.get('n_qubits')}")
            print(f"  n_lines: {features.get('n_lines')}")
            print(f"  n_cx: {features.get('n_cx')}")
            print(f"  n_2q_gates: {features.get('n_2q_gates')}")
            print(f"  n_1q_gates: {features.get('n_1q_gates')}")
            print(f"  avg_span: {features.get('avg_span'):.2f}")
            print(f"  max_span: {features.get('max_span')}")
            
            assert features.get('n_qubits', 0) > 0, f"Invalid n_qubits for {fname}"
            assert features.get('n_lines', 0) > 0, f"Invalid n_lines for {fname}"
    
    print("\n✓ extract_qasm_features passed")


def test_dataset_creation():
    """Test PyTorch Dataset creation."""
    print("\n" + "=" * 60)
    print("TEST: QuantumCircuitDataset")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    train_dataset = QuantumCircuitDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="train",
        val_fraction=0.2,
        seed=42,
    )
    
    val_dataset = QuantumCircuitDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="val",
        val_fraction=0.2,
        seed=42,
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Feature dimension: {train_dataset.feature_dim}")
    
    assert len(train_dataset) > 0, "Empty train dataset"
    assert len(val_dataset) > 0, "Empty val dataset"
    
    train_files = set(train_dataset.results[i].file for i in range(len(train_dataset)))
    val_files = set(val_dataset.results[i].file for i in range(len(val_dataset)))
    overlap = train_files & val_files
    assert len(overlap) == 0, f"Train/val overlap: {overlap}"
    print(f"\n✓ No overlap between train and val circuits")
    
    sample = train_dataset[0]
    print(f"\nSample item:")
    print(f"  features shape: {sample['features'].shape}")
    print(f"  features dtype: {sample['features'].dtype}")
    print(f"  threshold: {sample['threshold']}")
    print(f"  log2_runtime: {sample['log2_runtime']:.4f}")
    print(f"  file: {sample['file']}")
    print(f"  backend: {sample['backend']}")
    print(f"  precision: {sample['precision']}")
    
    assert sample['features'].shape[0] == train_dataset.feature_dim
    assert sample['log2_runtime'].dtype == torch.float32
    
    print("\n✓ QuantumCircuitDataset passed")
    return train_dataset, val_dataset


def test_data_loaders():
    """Test DataLoader creation and batching."""
    print("\n" + "=" * 60)
    print("TEST: create_data_loaders")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    train_loader, val_loader = create_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=8,
        val_fraction=0.2,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  features: {batch['features'].shape}")
    print(f"  threshold: {batch['threshold']}")
    print(f"  log2_runtime: {batch['log2_runtime'].shape}")
    print(f"  files: {batch['file']}")
    print(f"  backends: {batch['backend']}")
    print(f"  precisions: {batch['precision']}")
    
    assert batch['features'].shape[0] == 8, "Incorrect batch size"
    assert batch['log2_runtime'].shape[0] == 8, "Incorrect batch size"
    
    print("\n✓ create_data_loaders passed")
    return train_loader, val_loader


def test_feature_statistics():
    """Test feature statistics computation for normalization."""
    print("\n" + "=" * 60)
    print("TEST: get_feature_statistics")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    train_loader, _ = create_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=32,
    )
    
    mean, std = get_feature_statistics(train_loader)
    
    print(f"Feature mean shape: {mean.shape}")
    print(f"Feature std shape: {std.shape}")
    print(f"\nFeature statistics (first 10):")
    for i in range(min(10, len(mean))):
        print(f"  Feature {i}: mean={mean[i]:.4f}, std={std[i]:.4f}")
    
    assert not torch.isnan(mean).any(), "NaN in mean"
    assert not torch.isnan(std).any(), "NaN in std"
    assert (std > 0).all(), "Zero std values"
    
    print("\n✓ get_feature_statistics passed")


def test_threshold_label_distribution():
    """Analyze threshold class distribution."""
    print("\n" + "=" * 60)
    print("TEST: Threshold Label Distribution")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    dataset = QuantumCircuitDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="train",
        val_fraction=0.0,
    )
    
    threshold_counts = Counter()
    for i in range(len(dataset)):
        item = dataset[i]
        threshold_counts[item['threshold']] += 1
    
    print("Threshold distribution:")
    for threshold_val in sorted(threshold_counts.keys()):
        count = threshold_counts[threshold_val]
        pct = 100 * count / len(dataset)
        print(f"  Threshold={threshold_val}: {count} ({pct:.1f}%)")
    
    print("\n✓ Threshold distribution analysis complete")


def test_runtime_distribution():
    """Analyze runtime distribution."""
    print("\n" + "=" * 60)
    print("TEST: Runtime Distribution")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    dataset = QuantumCircuitDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="train",
        val_fraction=0.0,
    )
    
    log2_runtimes = []
    for i in range(len(dataset)):
        item = dataset[i]
        log2_runtimes.append(item['log2_runtime'].item())
    
    log2_runtimes = np.array(log2_runtimes)
    
    print("Log2 runtime statistics:")
    print(f"  Min: {log2_runtimes.min():.4f}")
    print(f"  Max: {log2_runtimes.max():.4f}")
    print(f"  Mean: {log2_runtimes.mean():.4f}")
    print(f"  Std: {log2_runtimes.std():.4f}")
    print(f"  Median: {np.median(log2_runtimes):.4f}")
    
    actual_runtimes = np.power(2.0, log2_runtimes)
    print("\nActual runtime statistics (seconds):")
    print(f"  Min: {actual_runtimes.min():.4f}s")
    print(f"  Max: {actual_runtimes.max():.4f}s")
    print(f"  Mean: {actual_runtimes.mean():.4f}s")
    print(f"  Median: {np.median(actual_runtimes):.4f}s")
    
    print("\n✓ Runtime distribution analysis complete")


def test_pytorch_compatibility():
    """Test full PyTorch training loop compatibility."""
    print("\n" + "=" * 60)
    print("TEST: PyTorch Training Compatibility")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    train_loader, val_loader = create_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=16,
    )
    
    feature_dim = train_loader.dataset.feature_dim
    
    model = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Running mini training loop (1 epoch) for duration regression...")
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        features = batch['features']
        targets = batch['log2_runtime']
        
        outputs = model(features).squeeze(-1)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"  Average training loss (MSE): {avg_loss:.4f}")
    
    print("\nRunning validation loop...")
    model.eval()
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features']
            targets = batch['log2_runtime']
            
            outputs = model(features).squeeze(-1)
            mse = ((outputs - targets) ** 2).sum().item()
            
            total_samples += targets.size(0)
            total_mse += mse
    
    avg_mse = total_mse / total_samples
    print(f"  Validation MSE: {avg_mse:.4f}")
    
    print("\n✓ PyTorch training compatibility verified")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("QUANTUM CIRCUIT DATA LOADER TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_load_hackathon_data()
        test_extract_qasm_features()
        test_dataset_creation()
        test_data_loaders()
        test_feature_statistics()
        test_threshold_label_distribution()
        test_runtime_distribution()
        test_pytorch_compatibility()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
Data augmentation for quantum circuit graphs.

Augmentation strategies:
1. Qubit permutation - relabel qubits (circuit is equivalent under relabeling)
2. Edge dropout - randomly drop edges during training
3. Feature noise - add Gaussian noise to continuous features
4. Temporal jitter - slightly perturb temporal positions
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Optional, List, Callable
import copy


class QubitPermutation:
    """
    Randomly permute qubit indices.
    
    Quantum circuits are equivalent under qubit relabeling, so this is a 
    valid augmentation that doesn't change the circuit's behavior.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying the augmentation
        """
        self.p = p
    
    def __call__(self, data: Data) -> Data:
        if np.random.random() > self.p:
            return data
        
        data = copy.copy(data)
        n_nodes = data.x.shape[0]
        
        # Generate random permutation
        perm = torch.randperm(n_nodes)
        inv_perm = torch.argsort(perm)
        
        # Permute node features
        data.x = data.x[perm]
        
        # Permute edge indices
        if data.edge_index.numel() > 0:
            data.edge_index = inv_perm[data.edge_index]
        
        return data


class EdgeDropout:
    """
    Randomly drop edges during training.
    
    This encourages the model to not rely too heavily on any single edge
    and improves generalization.
    """
    
    def __init__(self, p: float = 0.1):
        """
        Args:
            p: Probability of dropping each edge
        """
        self.p = p
    
    def __call__(self, data: Data) -> Data:
        if self.p == 0:
            return data
        
        data = copy.copy(data)
        n_edges = data.edge_index.shape[1]
        
        if n_edges == 0:
            return data
        
        # Keep mask
        keep_mask = torch.rand(n_edges) > self.p
        
        # Ensure at least one edge is kept
        if not keep_mask.any():
            keep_mask[0] = True
        
        data.edge_index = data.edge_index[:, keep_mask]
        data.edge_attr = data.edge_attr[keep_mask]
        data.edge_gate_type = data.edge_gate_type[keep_mask]
        
        return data


class FeatureNoise:
    """
    Add Gaussian noise to continuous node features.
    
    Helps with regularization and prevents overfitting to exact feature values.
    """
    
    def __init__(self, std: float = 0.1, p: float = 0.5):
        """
        Args:
            std: Standard deviation of Gaussian noise
            p: Probability of applying noise
        """
        self.std = std
        self.p = p
    
    def __call__(self, data: Data) -> Data:
        if np.random.random() > self.p:
            return data
        
        data = copy.copy(data)
        
        # Add noise to node features
        noise = torch.randn_like(data.x) * self.std
        data.x = data.x + noise
        
        return data


class TemporalJitter:
    """
    Add small jitter to temporal edge positions.
    
    Temporal positions are normalized to [0, 1], so small perturbations
    simulate uncertainty in exact gate ordering within a layer.
    """
    
    def __init__(self, std: float = 0.05, p: float = 0.5):
        """
        Args:
            std: Standard deviation of jitter
            p: Probability of applying jitter
        """
        self.std = std
        self.p = p
    
    def __call__(self, data: Data) -> Data:
        if np.random.random() > self.p:
            return data
        
        data = copy.copy(data)
        
        if data.edge_attr.numel() > 0:
            # Jitter the temporal position (first column of edge_attr)
            jitter = torch.randn(data.edge_attr.shape[0]) * self.std
            data.edge_attr = data.edge_attr.clone()
            data.edge_attr[:, 0] = torch.clamp(data.edge_attr[:, 0] + jitter, 0, 1)
        
        return data


class RandomEdgeReverse:
    """
    Randomly reverse edge directions for symmetric gates (CZ, SWAP).
    
    These gates are symmetric, so the direction shouldn't matter.
    """
    
    # Gate type indices for symmetric gates (from GATE_TO_IDX)
    SYMMETRIC_GATES = {'cz', 'swap', 'rxx', 'ryy', 'rzz'}
    
    def __init__(self, p: float = 0.5):
        self.p = p
        # Import gate mapping
        from .graph_builder import GATE_TO_IDX
        self.symmetric_indices = {
            GATE_TO_IDX[g] for g in self.SYMMETRIC_GATES if g in GATE_TO_IDX
        }
    
    def __call__(self, data: Data) -> Data:
        if np.random.random() > self.p:
            return data
        
        data = copy.copy(data)
        
        if data.edge_index.numel() == 0:
            return data
        
        edge_index = data.edge_index.clone()
        
        for i in range(edge_index.shape[1]):
            gate_type = data.edge_gate_type[i].item()
            if gate_type in self.symmetric_indices and np.random.random() < 0.5:
                # Swap source and destination
                edge_index[0, i], edge_index[1, i] = edge_index[1, i].clone(), edge_index[0, i].clone()
        
        data.edge_index = edge_index
        return data


class Compose:
    """Compose multiple augmentations."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, data: Data) -> Data:
        for t in self.transforms:
            data = t(data)
        return data


def get_train_augmentation(
    qubit_perm_p: float = 0.5,
    edge_dropout_p: float = 0.1,
    feature_noise_std: float = 0.1,
    temporal_jitter_std: float = 0.05,
) -> Compose:
    """
    Get default training augmentation pipeline.
    
    Args:
        qubit_perm_p: Probability of qubit permutation
        edge_dropout_p: Probability of dropping each edge
        feature_noise_std: Std of feature noise
        temporal_jitter_std: Std of temporal jitter
    """
    transforms = [
        QubitPermutation(p=qubit_perm_p),
        EdgeDropout(p=edge_dropout_p),
        FeatureNoise(std=feature_noise_std, p=0.5),
        TemporalJitter(std=temporal_jitter_std, p=0.5),
        RandomEdgeReverse(p=0.5),
    ]
    return Compose(transforms)


class AugmentedDataset:
    """
    Wrapper that applies augmentation to a dataset.
    
    For training, applies random augmentations on each access.
    """
    
    def __init__(self, dataset, augmentation: Optional[Callable] = None):
        self.dataset = dataset
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.augmentation is not None:
            data = self.augmentation(data)
        return data
    
    def get(self, idx):
        return self.__getitem__(idx)


if __name__ == "__main__":
    # Test augmentations
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from gnn.graph_builder import build_graph_from_qasm
    
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    h q[0];
    cx q[0],q[1];
    cx q[1],q[2];
    cz q[2],q[3];
    """
    
    graph = build_graph_from_qasm(qasm, "CPU", "single")
    
    # Create Data object
    data = Data(
        x=graph["x"],
        edge_index=graph["edge_index"],
        edge_attr=graph["edge_attr"],
        edge_gate_type=graph["edge_gate_type"],
    )
    
    print("Original:")
    print(f"  Nodes: {data.x.shape}")
    print(f"  Edges: {data.edge_index.shape}")
    print(f"  Edge index:\n{data.edge_index}")
    
    # Test qubit permutation
    perm = QubitPermutation(p=1.0)
    data_perm = perm(data)
    print("\nAfter qubit permutation:")
    print(f"  Edge index:\n{data_perm.edge_index}")
    
    # Test edge dropout
    dropout = EdgeDropout(p=0.3)
    data_drop = dropout(data)
    print("\nAfter edge dropout:")
    print(f"  Edges: {data_drop.edge_index.shape}")
    
    # Test full pipeline
    aug = get_train_augmentation()
    data_aug = aug(data)
    print("\nAfter full augmentation:")
    print(f"  Nodes: {data_aug.x.shape}")
    print(f"  Edges: {data_aug.edge_index.shape}")

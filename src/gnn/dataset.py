"""
PyTorch Geometric dataset for quantum circuit graphs.

Provides:
- QuantumCircuitGraphDataset: Full dataset with train/val split by circuit file
- KFoldQuantumCircuitGraphDataset: K-fold cross-validation dataset
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from .graph_builder import (
    build_graph_from_file,
    NODE_FEAT_DIM,
    NODE_FEAT_DIM_BASIC,
    EDGE_FEAT_DIM,
    EDGE_FEAT_DIM_BASIC,
    GLOBAL_FEAT_DIM_BASE,
)


THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]

FAMILY_CATEGORIES = [
    "Amplitude_Estimation", "CutBell", "Deutsch_Jozsa", "GHZ", "GraphState",
    "Ground_State", "Grover_NoAncilla", "Grover_V_Chain", "Portfolio_QAOA",
    "Portfolio_VQE", "Pricing_Call", "QAOA", "QFT", "QFT_Entangled", "QNN",
    "QPE_Exact", "Shor", "TwoLocalRandom", "VQE", "W_State"
]
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_CATEGORIES)}
NUM_FAMILIES = len(FAMILY_CATEGORIES)


@dataclass
class ThresholdSweepEntry:
    threshold: int
    sdk_get_fidelity: Optional[float]
    p_return_zero: Optional[float]


def load_hackathon_data(data_path: Path) -> Tuple[Dict, List[Dict]]:
    """Load hackathon data and return circuit info dict and results list."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    circuit_info = {}
    for c in data["circuits"]:
        circuit_info[c["file"]] = {
            "family": c["family"],
            "n_qubits": c["n_qubits"],
        }
    
    results = []
    for r in data["results"]:
        if r["status"] != "ok":
            continue
        
        sweep = []
        for s in r.get("threshold_sweep", []):
            sweep.append(ThresholdSweepEntry(
                threshold=s["threshold"],
                sdk_get_fidelity=s.get("sdk_get_fidelity"),
                p_return_zero=s.get("p_return_zero"),
            ))
        
        forward = r.get("forward", {})
        
        results.append({
            "file": r["file"],
            "backend": r["backend"],
            "precision": r["precision"],
            "threshold_sweep": sweep,
            "forward_wall_s": forward.get("run_wall_s") if forward else None,
        })
    
    return circuit_info, results


def compute_min_threshold(sweep: List[ThresholdSweepEntry], target: float = 0.99) -> Optional[int]:
    """Find minimum threshold meeting target fidelity."""
    for entry in sorted(sweep, key=lambda x: x.threshold):
        fid = entry.sdk_get_fidelity
        if fid is not None and fid >= target:
            return entry.threshold
    return None


def get_threshold_class(sweep: List[ThresholdSweepEntry]) -> int:
    """Get threshold as class index for target fidelity 0.75."""
    min_thresh = compute_min_threshold(sweep, target=0.75)
    if min_thresh is None:
        return len(THRESHOLD_LADDER) - 1
    try:
        return THRESHOLD_LADDER.index(min_thresh)
    except ValueError:
        for i, t in enumerate(THRESHOLD_LADDER):
            if t >= min_thresh:
                return i
        return len(THRESHOLD_LADDER) - 1


class QuantumCircuitGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for quantum circuit graphs.
    
    Processes all circuits once and stores them for fast loading.
    """
    
    def __init__(
        self,
        data_path: Path,
        circuits_dir: Path,
        root: Optional[str] = None,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        self.data_path = Path(data_path)
        self.circuits_dir = Path(circuits_dir)
        self.split = split
        self.val_fraction = val_fraction
        self.seed = seed
        
        # Load raw data before calling super().__init__
        self.circuit_info, self.all_results = load_hackathon_data(self.data_path)
        
        # Split by circuit file
        circuit_files = sorted(list(set(r["file"] for r in self.all_results)))
        rng = np.random.RandomState(seed)
        rng.shuffle(circuit_files)
        
        n_val = int(len(circuit_files) * val_fraction)
        val_files = set(circuit_files[:n_val])
        train_files = set(circuit_files[n_val:])
        
        if split == "train":
            self.results = [r for r in self.all_results if r["file"] in train_files]
        else:
            self.results = [r for r in self.all_results if r["file"] in val_files]
        
        # We don't use the caching mechanism - process in memory directly
        super().__init__(root, transform, pre_transform)
        
        # Process data
        self._data_list = self._process_data()
    
    def _process_data(self) -> List[Data]:
        """Process all circuits into PyG Data objects."""
        data_list = []
        
        for result in self.results:
            file = result["file"]
            backend = result["backend"]
            precision = result["precision"]
            sweep = result["threshold_sweep"]
            forward_time = result["forward_wall_s"]
            
            info = self.circuit_info.get(file, {})
            family = info.get("family", "")
            
            qasm_path = self.circuits_dir / file
            if not qasm_path.exists():
                continue
            
            try:
                graph_dict = build_graph_from_file(
                    qasm_path,
                    backend=backend,
                    precision=precision,
                    family=family,
                    family_to_idx=FAMILY_TO_IDX,
                    num_families=NUM_FAMILIES,
                )
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
            
            threshold_class = get_threshold_class(sweep)
            log_runtime = np.log1p(forward_time) if forward_time else 0.0
            
            data = Data(
                x=graph_dict["x"],
                edge_index=graph_dict["edge_index"],
                edge_attr=graph_dict["edge_attr"],
                edge_gate_type=graph_dict["edge_gate_type"],
                global_features=graph_dict["global_features"],
                threshold_class=torch.tensor(threshold_class, dtype=torch.long),
                log_runtime=torch.tensor(log_runtime, dtype=torch.float32),
                file=file,
                backend=backend,
                precision=precision,
            )
            
            data_list.append(data)
        
        return data_list
    
    def len(self) -> int:
        return len(self._data_list)
    
    def get(self, idx: int) -> Data:
        return self._data_list[idx]
    
    @property
    def processed_file_names(self):
        return []  # We don't save to disk
    
    @property
    def raw_file_names(self):
        return []


class LazyQuantumCircuitGraphDataset(Dataset):
    """
    Lazy-loading dataset that builds graphs on-demand.
    More memory efficient for large datasets.
    """
    
    def __init__(
        self,
        data_path: Path,
        circuits_dir: Path,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.circuits_dir = Path(circuits_dir)
        self.split = split
        
        self.circuit_info, all_results = load_hackathon_data(self.data_path)
        
        circuit_files = sorted(list(set(r["file"] for r in all_results)))
        rng = np.random.RandomState(seed)
        rng.shuffle(circuit_files)
        
        n_val = int(len(circuit_files) * val_fraction)
        val_files = set(circuit_files[:n_val])
        train_files = set(circuit_files[n_val:])
        
        if split == "train":
            self.results = [r for r in all_results if r["file"] in train_files]
        else:
            self.results = [r for r in all_results if r["file"] in val_files]
        
        self._cache: Dict[int, Data] = {}
    
    def len(self) -> int:
        return len(self.results)
    
    def get(self, idx: int) -> Data:
        if idx in self._cache:
            return self._cache[idx]
        
        result = self.results[idx]
        file = result["file"]
        backend = result["backend"]
        precision = result["precision"]
        sweep = result["threshold_sweep"]
        forward_time = result["forward_wall_s"]
        
        info = self.circuit_info.get(file, {})
        family = info.get("family", "")
        
        qasm_path = self.circuits_dir / file
        graph_dict = build_graph_from_file(
            qasm_path,
            backend=backend,
            precision=precision,
            family=family,
            family_to_idx=FAMILY_TO_IDX,
            num_families=NUM_FAMILIES,
        )
        
        threshold_class = get_threshold_class(sweep)
        log_runtime = np.log1p(forward_time) if forward_time else 0.0
        
        data = Data(
            x=graph_dict["x"],
            edge_index=graph_dict["edge_index"],
            edge_attr=graph_dict["edge_attr"],
            edge_gate_type=graph_dict["edge_gate_type"],
            global_features=graph_dict["global_features"],
            threshold_class=torch.tensor(threshold_class, dtype=torch.long),
            log_runtime=torch.tensor(log_runtime, dtype=torch.float32),
            file=file,
            backend=backend,
            precision=precision,
        )
        
        self._cache[idx] = data
        return data


def create_graph_data_loaders(
    data_path: Path,
    circuits_dir: Path,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[PyGDataLoader, PyGDataLoader]:
    """Create train and validation PyG DataLoaders."""
    train_dataset = QuantumCircuitGraphDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="train",
        val_fraction=val_fraction,
        seed=seed,
    )
    
    val_dataset = QuantumCircuitGraphDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="val",
        val_fraction=val_fraction,
        seed=seed,
    )
    
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


def create_kfold_graph_data_loaders(
    data_path: Path,
    circuits_dir: Path,
    n_folds: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> List[Tuple[PyGDataLoader, PyGDataLoader]]:
    """Create k-fold cross-validation data loaders."""
    circuit_info, all_results = load_hackathon_data(data_path)
    
    circuit_files = sorted(list(set(r["file"] for r in all_results)))
    rng = np.random.RandomState(seed)
    rng.shuffle(circuit_files)
    
    fold_size = len(circuit_files) // n_folds
    fold_loaders = []
    
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else len(circuit_files)
        
        val_files = set(circuit_files[start_idx:end_idx])
        train_files = set(circuit_files) - val_files
        
        # Create datasets for this fold
        train_results = [r for r in all_results if r["file"] in train_files]
        val_results = [r for r in all_results if r["file"] in val_files]
        
        train_data_list = _process_results(
            train_results, circuit_info, circuits_dir
        )
        val_data_list = _process_results(
            val_results, circuit_info, circuits_dir
        )
        
        # Create simple list-based datasets
        train_loader = PyGDataLoader(
            train_data_list,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=len(train_data_list) > batch_size,
        )
        
        val_loader = PyGDataLoader(
            val_data_list,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders


def _process_results(
    results: List[Dict],
    circuit_info: Dict,
    circuits_dir: Path,
) -> List[Data]:
    """Process a list of results into PyG Data objects."""
    data_list = []
    
    for result in results:
        file = result["file"]
        backend = result["backend"]
        precision = result["precision"]
        sweep = result["threshold_sweep"]
        forward_time = result["forward_wall_s"]
        
        info = circuit_info.get(file, {})
        family = info.get("family", "")
        
        qasm_path = circuits_dir / file
        if not qasm_path.exists():
            continue
        
        try:
            graph_dict = build_graph_from_file(
                qasm_path,
                backend=backend,
                precision=precision,
                family=family,
                family_to_idx=FAMILY_TO_IDX,
                num_families=NUM_FAMILIES,
            )
        except Exception as e:
            continue
        
        threshold_class = get_threshold_class(sweep)
        log_runtime = np.log1p(forward_time) if forward_time else 0.0
        
        data = Data(
            x=graph_dict["x"],
            edge_index=graph_dict["edge_index"],
            edge_attr=graph_dict["edge_attr"],
            edge_gate_type=graph_dict["edge_gate_type"],
            global_features=graph_dict["global_features"],
            threshold_class=torch.tensor(threshold_class, dtype=torch.long),
            log_runtime=torch.tensor(log_runtime, dtype=torch.float32),
            file=file,
            backend=backend,
            precision=precision,
        )
        
        data_list.append(data)
    
    return data_list


# Constants for external use
GLOBAL_FEAT_DIM = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    print("Loading datasets...")
    train_loader, val_loader = create_graph_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=16,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Node features: {batch.x.shape}")
    print(f"  Edge index: {batch.edge_index.shape}")
    print(f"  Edge attr: {batch.edge_attr.shape}")
    print(f"  Edge gate types: {batch.edge_gate_type.shape}")
    print(f"  Global features: {batch.global_features.shape}")
    print(f"  Batch assignment: {batch.batch.shape}")
    print(f"  Threshold classes: {batch.threshold_class}")

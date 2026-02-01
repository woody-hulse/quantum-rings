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
NUM_THRESHOLD_CLASSES = len(THRESHOLD_LADDER)

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
    run_wall_s: Optional[float] = None


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
                run_wall_s=s.get("run_wall_s"),
            ))
        selection = r.get("selection", {})
        forward = r.get("forward", {})
        results.append({
            "file": r["file"],
            "backend": r["backend"],
            "precision": r["precision"],
            "threshold_sweep": sweep,
            "selected_threshold": selection.get("selected_threshold"),
            "forward_wall_s": forward.get("run_wall_s") if forward else None,
        })
    return circuit_info, results


def _compute_min_threshold(sweep: List, target: float = 0.75):
    """Find minimum threshold meeting fidelity target. Returns threshold value or None."""
    for entry in sorted(sweep, key=lambda x: x.threshold):
        fid = getattr(entry, "sdk_get_fidelity", None)
        if fid is not None and fid >= target:
            return entry.threshold
    return None


def _threshold_to_class(min_threshold: int) -> int:
    if min_threshold in THRESHOLD_LADDER:
        return THRESHOLD_LADDER.index(min_threshold)
    for i, t in enumerate(THRESHOLD_LADDER):
        if min_threshold <= t:
            return i
    return len(THRESHOLD_LADDER) - 1


class QuantumCircuitGraphDataset(InMemoryDataset):
    """
    Duration prediction: one sample per result using forward run_wall_s and selected_threshold only (no mirror times).
    Global features include log2(threshold). Target is log2_runtime.
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
        self.circuit_info, self.all_results = load_hackathon_data(self.data_path)

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

        self._samples: List[Dict] = []
        for r in self.results:
            if r.get("forward_wall_s") is not None and r["forward_wall_s"] > 0 and r.get("selected_threshold") is not None:
                self._samples.append(r)

        super().__init__(root, transform, pre_transform)
        self._data_list = self._process_data()

    def _process_data(self) -> List[Data]:
        data_list = []
        for result in self._samples:
            file = result["file"]
            backend = result["backend"]
            precision = result["precision"]
            threshold = result["selected_threshold"]
            info = self.circuit_info.get(file, {})
            family = info.get("family", "")
            log2_threshold = np.log2(max(threshold, 1))
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
                    log2_threshold=log2_threshold,
                )
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
            log2_runtime = np.log2(max(float(result["forward_wall_s"]), 1e-10))
            data = Data(
                x=graph_dict["x"],
                edge_index=graph_dict["edge_index"],
                edge_attr=graph_dict["edge_attr"],
                edge_gate_type=graph_dict["edge_gate_type"],
                global_features=graph_dict["global_features"],
                log2_runtime=torch.tensor(log2_runtime, dtype=torch.float32),
                file=file,
                backend=backend,
                precision=precision,
                threshold=threshold,
            )
            data_list.append(data)
        return data_list

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    @property
    def processed_file_names(self):
        return []

    @property
    def raw_file_names(self):
        return []


GLOBAL_FEAT_DIM = GLOBAL_FEAT_DIM_BASE + 1 + NUM_FAMILIES
GLOBAL_FEAT_DIM_THRESHOLD_CLASS = GLOBAL_FEAT_DIM_BASE + NUM_FAMILIES


class ThresholdClassGraphDataset(InMemoryDataset):
    """
    Threshold-class prediction: one sample per result.
    Global features exclude log2(threshold) and duration. Target is threshold_class.
    """

    def __init__(
        self,
        data_path: Path,
        circuits_dir: Path,
        root: Optional[str] = None,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        fidelity_target: float = 0.75,
        fold: Optional[int] = None,
        n_folds: Optional[int] = None,
        transform=None,
        pre_transform=None,
    ):
        self.data_path = Path(data_path)
        self.circuits_dir = Path(circuits_dir)
        self.split = split
        self.val_fraction = val_fraction
        self.seed = seed
        self.fidelity_target = fidelity_target
        self.circuit_info, self.all_results = load_hackathon_data(self.data_path)
        circuit_files = sorted(list(set(r["file"] for r in self.all_results)))
        rng = np.random.RandomState(seed)
        rng.shuffle(circuit_files)
        if fold is not None and n_folds is not None:
            fold_size = len(circuit_files) // n_folds
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else len(circuit_files)
            val_files = set(circuit_files[start:end])
            train_files = set(circuit_files) - val_files
        else:
            n_val = int(len(circuit_files) * val_fraction)
            val_files = set(circuit_files[:n_val])
            train_files = set(circuit_files[n_val:])
        if split == "train":
            results = [r for r in self.all_results if r["file"] in train_files]
        else:
            results = [r for r in self.all_results if r["file"] in val_files]
        self._samples = []
        for r in results:
            min_thr = _compute_min_threshold(r["threshold_sweep"], target=self.fidelity_target)
            if min_thr is not None:
                self._samples.append(r)
        super().__init__(root, transform, pre_transform)
        self._data_list = self._process_data()

    def _process_data(self) -> List[Data]:
        data_list = []
        for result in self._samples:
            file = result["file"]
            backend = result["backend"]
            precision = result["precision"]
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
                    log2_threshold=None,
                )
            except Exception:
                continue
            min_thr = _compute_min_threshold(result["threshold_sweep"], target=self.fidelity_target)
            threshold_class = _threshold_to_class(min_thr)
            data = Data(
                x=graph_dict["x"],
                edge_index=graph_dict["edge_index"],
                edge_attr=graph_dict["edge_attr"],
                edge_gate_type=graph_dict["edge_gate_type"],
                global_features=graph_dict["global_features"],
                threshold_class=torch.tensor(threshold_class, dtype=torch.long),
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
        return []

    @property
    def raw_file_names(self):
        return []


class LazyQuantumCircuitGraphDataset(Dataset):
    """Lazy-loading duration dataset: one sample per result using forward run_wall_s and selected_threshold only."""

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
            results = [r for r in all_results if r["file"] in train_files]
        else:
            results = [r for r in all_results if r["file"] in val_files]
        self._samples: List[Dict] = []
        for r in results:
            if r.get("forward_wall_s") is not None and r["forward_wall_s"] > 0 and r.get("selected_threshold") is not None:
                self._samples.append(r)
        self._cache: Dict[int, Data] = {}

    def len(self) -> int:
        return len(self._samples)

    def get(self, idx: int) -> Data:
        if idx in self._cache:
            return self._cache[idx]
        result = self._samples[idx]
        file = result["file"]
        backend = result["backend"]
        precision = result["precision"]
        threshold = result["selected_threshold"]
        info = self.circuit_info.get(file, {})
        family = info.get("family", "")
        log2_threshold = np.log2(max(threshold, 1))
        qasm_path = self.circuits_dir / file
        graph_dict = build_graph_from_file(
            qasm_path,
            backend=backend,
            precision=precision,
            family=family,
            family_to_idx=FAMILY_TO_IDX,
            num_families=NUM_FAMILIES,
            log2_threshold=log2_threshold,
        )
        log2_runtime = np.log2(max(float(result["forward_wall_s"]), 1e-10))
        data = Data(
            x=graph_dict["x"],
            edge_index=graph_dict["edge_index"],
            edge_attr=graph_dict["edge_attr"],
            edge_gate_type=graph_dict["edge_gate_type"],
            global_features=graph_dict["global_features"],
            log2_runtime=torch.tensor(log2_runtime, dtype=torch.float32),
            file=file,
            backend=backend,
            precision=precision,
            threshold=threshold,
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
    """Create train and validation PyG DataLoaders. Threshold as input, log2(duration) target."""
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


def create_threshold_class_graph_data_loaders(
    data_path: Path,
    circuits_dir: Path,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    fidelity_target: float = 0.75,
) -> Tuple[PyGDataLoader, PyGDataLoader]:
    """Create train and validation PyG DataLoaders for threshold-class prediction (no duration, no threshold in features)."""
    train_dataset = ThresholdClassGraphDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="train",
        val_fraction=val_fraction,
        seed=seed,
        fidelity_target=fidelity_target,
    )
    val_dataset = ThresholdClassGraphDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="val",
        val_fraction=val_fraction,
        seed=seed,
        fidelity_target=fidelity_target,
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


def create_kfold_threshold_class_graph_data_loaders(
    data_path: Path,
    circuits_dir: Path,
    n_folds: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    fidelity_target: float = 0.75,
) -> List[Tuple[PyGDataLoader, PyGDataLoader]]:
    """Create k-fold cross-validation PyG DataLoaders for threshold-class prediction."""
    fold_loaders = []
    for fold in range(n_folds):
        train_dataset = ThresholdClassGraphDataset(
            data_path=data_path,
            circuits_dir=circuits_dir,
            split="train",
            val_fraction=0.2,
            seed=seed,
            fidelity_target=fidelity_target,
            fold=fold,
            n_folds=n_folds,
        )
        val_dataset = ThresholdClassGraphDataset(
            data_path=data_path,
            circuits_dir=circuits_dir,
            split="val",
            val_fraction=0.2,
            seed=seed,
            fidelity_target=fidelity_target,
            fold=fold,
            n_folds=n_folds,
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
        fold_loaders.append((train_loader, val_loader))
    return fold_loaders


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
    """Process results into PyG Data objects (duration: one per result using forward run_wall_s and selected_threshold only)."""
    data_list = []
    for result in results:
        if result.get("forward_wall_s") is None or result["forward_wall_s"] <= 0 or result.get("selected_threshold") is None:
            continue
        file = result["file"]
        backend = result["backend"]
        precision = result["precision"]
        threshold = result["selected_threshold"]
        info = circuit_info.get(file, {})
        family = info.get("family", "")
        qasm_path = circuits_dir / file
        if not qasm_path.exists():
            continue
        try:
            log2_threshold = np.log2(max(threshold, 1))
            graph_dict = build_graph_from_file(
                qasm_path,
                backend=backend,
                precision=precision,
                family=family,
                family_to_idx=FAMILY_TO_IDX,
                num_families=NUM_FAMILIES,
                log2_threshold=log2_threshold,
            )
        except Exception:
            continue
        log2_runtime = np.log2(max(float(result["forward_wall_s"]), 1e-10))
        data = Data(
            x=graph_dict["x"],
            edge_index=graph_dict["edge_index"],
            edge_attr=graph_dict["edge_attr"],
            edge_gate_type=graph_dict["edge_gate_type"],
            global_features=graph_dict["global_features"],
            log2_runtime=torch.tensor(log2_runtime, dtype=torch.float32),
            file=file,
            backend=backend,
            precision=precision,
            threshold=threshold,
        )
        data_list.append(data)
    return data_list


# GLOBAL_FEAT_DIM defined above (duration: base + 1 + num_families)


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
    print(f"  log2_runtime: {batch.log2_runtime.shape}")

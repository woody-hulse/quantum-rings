"""
PyTorch-compatible data loading for the quantum circuit fingerprint challenge.

Provides Dataset classes for:
- Duration-only formulation: threshold as input parameter, predict log2(duration).
- Legacy threshold+runtime: predict threshold class and runtime (log1p).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from qasm_features import extract_qasm_features


THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BACKEND_MAP = {"CPU": 0, "GPU": 1}
PRECISION_MAP = {"single": 0, "double": 1}


@dataclass
class CircuitInfo:
    """Circuit metadata from the dataset."""
    file: str
    family: str
    n_qubits: int
    source_name: str = ""
    source_url: str = ""


@dataclass
class ThresholdSweepEntry:
    """Single entry from a threshold sweep."""
    threshold: int
    sdk_get_fidelity: Optional[float]
    p_return_zero: Optional[float]
    run_wall_s: Optional[float]
    peak_rss_mb: Optional[float]
    returncode: int
    note: str


@dataclass
class ResultEntry:
    """Result entry for a circuit/backend/precision combination."""
    file: str
    backend: str
    precision: str
    status: str
    selected_threshold: Optional[int] = None
    target_fidelity: Optional[float] = None
    threshold_sweep: List[ThresholdSweepEntry] = field(default_factory=list)
    forward_wall_s: Optional[float] = None
    forward_shots: Optional[int] = None


def load_hackathon_data(data_path: Path) -> Tuple[List[CircuitInfo], List[ResultEntry]]:
    """Load the hackathon public dataset."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    circuits = []
    for c in data["circuits"]:
        circuits.append(CircuitInfo(
            file=c["file"],
            family=c["family"],
            n_qubits=c["n_qubits"],
            source_name=c.get("source", {}).get("name", ""),
            source_url=c.get("source", {}).get("url", "") or "",
        ))
    
    results = []
    for r in data["results"]:
        sweep = []
        for s in r.get("threshold_sweep", []):
            sweep.append(ThresholdSweepEntry(
                threshold=s["threshold"],
                sdk_get_fidelity=s.get("sdk_get_fidelity"),
                p_return_zero=s.get("p_return_zero"),
                run_wall_s=s.get("run_wall_s"),
                peak_rss_mb=s.get("peak_rss_mb"),
                returncode=s.get("returncode", 0),
                note=s.get("note", ""),
            ))
        
        selection = r.get("selection", {})
        forward = r.get("forward", {})
        
        results.append(ResultEntry(
            file=r["file"],
            backend=r["backend"],
            precision=r["precision"],
            status=r["status"],
            selected_threshold=selection.get("selected_threshold"),
            target_fidelity=selection.get("target"),
            threshold_sweep=sweep,
            forward_wall_s=forward.get("run_wall_s") if forward else None,
            forward_shots=forward.get("shots") if forward else None,
        ))
    
    return circuits, results


def compute_min_threshold(sweep: List[ThresholdSweepEntry], target: float = 0.75) -> Optional[int]:
    """Find the minimum threshold that meets the target fidelity."""
    for entry in sorted(sweep, key=lambda x: x.threshold):
        fid = entry.sdk_get_fidelity
        if fid is not None and fid >= target:
            return entry.threshold
    return None


def threshold_to_class(min_threshold: int) -> int:
    """Map minimum threshold value to ladder class index (0..len(THRESHOLD_LADDER)-1)."""
    if min_threshold in THRESHOLD_LADDER:
        return THRESHOLD_LADDER.index(min_threshold)
    for i, t in enumerate(THRESHOLD_LADDER):
        if min_threshold <= t:
            return i
    return len(THRESHOLD_LADDER) - 1


NUMERIC_FEATURE_KEYS = [
    "n_qubits", "n_lines", "n_cx", "n_cz", "n_swap", "n_ccx",
    "n_2q_gates", "n_1q_gates", "n_unique_pairs",
    "n_custom_gates", "n_measure", "n_barrier",
    "n_h", "n_rx", "n_ry", "n_rz",
    "avg_span", "max_span", "min_span", "span_std",
    "gate_density", "gate_ratio_2q",
    "max_degree", "avg_degree", "degree_entropy",
    "n_connected_components", "clustering_coeff", "max_component_size", "component_entropy",
    "estimated_depth", "depth_per_qubit",
    "middle_cut_crossings", "cut_crossing_ratio", "max_cut_crossings",
    "graph_bandwidth", "normalized_bandwidth", "bandwidth_squared",
    "early_longrange_ratio", "late_longrange_ratio",
    "longrange_temporal_center", "entanglement_velocity",
    "qubit_activity_entropy", "qubit_activity_variance",
    "qubit_activity_max_ratio", "active_qubit_fraction",
    "cx_chain_max_length", "h_cx_pattern_count", "cx_rz_cx_pattern_count",
    "rotation_density", "gate_type_entropy", "cx_h_ratio",
    "light_cone_spread_rate", "light_cone_half_coverage_depth", "final_light_cone_size",
    "nearest_neighbor_ratio", "long_range_ratio",
    "span_gini_coefficient", "weighted_span_sum",
    "pattern_repetition_score", "barrier_regularity", "layer_uniformity",
    "treewidth_min_degree",
]

FAMILY_CATEGORIES = [
    "Amplitude_Estimation", "CutBell", "Deutsch_Jozsa", "GHZ", "GraphState",
    "Ground_State", "Grover_NoAncilla", "Grover_V_Chain", "Portfolio_QAOA",
    "Portfolio_VQE", "Pricing_Call", "QAOA", "QFT", "QFT_Entangled", "QNN",
    "QPE_Exact", "Shor", "TwoLocalRandom", "VQE", "W_State"
]


class QuantumCircuitDataset(Dataset):
    """
    Dataset for duration prediction: threshold as input parameter.
    One sample per result using forward run_wall_s and selected_threshold only (no mirror times).
    Target: log2(duration).
    """

    FAMILY_CATEGORIES = FAMILY_CATEGORIES
    NUMERIC_FEATURE_KEYS = NUMERIC_FEATURE_KEYS

    def __init__(
        self,
        data_path: Path,
        circuits_dir: Path,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        filter_ok_only: bool = True,
    ):
        self.circuits_dir = circuits_dir
        self.split = split

        circuits, results = load_hackathon_data(data_path)
        self.circuit_info = {c.file: c for c in circuits}
        self.family_to_idx = {f: i for i, f in enumerate(self.FAMILY_CATEGORIES)}

        if filter_ok_only:
            results = [r for r in results if r.status == "ok"]

        circuit_files = list(set(r.file for r in results))
        rng = np.random.RandomState(seed)
        rng.shuffle(circuit_files)
        n_val = int(len(circuit_files) * val_fraction)
        val_files = set(circuit_files[:n_val])
        train_files = set(circuit_files[n_val:])

        if split == "train":
            results = [r for r in results if r.file in train_files]
        else:
            results = [r for r in results if r.file in val_files]

        self._samples: List[ResultEntry] = []
        for r in results:
            if r.forward_wall_s is not None and r.forward_wall_s > 0 and r.selected_threshold is not None:
                self._samples.append(r)

        self.results = results
        self._feature_cache: Dict[str, Dict] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def _get_circuit_features(self, file: str) -> Dict[str, Any]:
        if file not in self._feature_cache:
            qasm_path = self.circuits_dir / file
            self._feature_cache[file] = extract_qasm_features(qasm_path)
        return self._feature_cache[file]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = self._samples[idx]
        circuit = self.circuit_info.get(result.file)
        qasm_features = self._get_circuit_features(result.file)

        family_idx = self.family_to_idx.get(circuit.family if circuit else "", 0)
        family_onehot = torch.zeros(len(self.FAMILY_CATEGORIES))
        family_onehot[family_idx] = 1.0

        backend_idx = BACKEND_MAP.get(result.backend, 0)
        precision_idx = PRECISION_MAP.get(result.precision, 0)
        threshold = result.selected_threshold
        log2_threshold = np.log2(max(threshold, 1))

        numeric_values = [qasm_features.get(k, 0.0) for k in self.NUMERIC_FEATURE_KEYS]
        numeric_values.extend([float(backend_idx), float(precision_idx), float(log2_threshold)])
        numeric_features = torch.tensor(numeric_values, dtype=torch.float32)
        features = torch.cat([numeric_features, family_onehot])

        duration_s = float(result.forward_wall_s)
        log2_runtime = np.log2(max(duration_s, 1e-10))

        return {
            "features": features,
            "log2_runtime": torch.tensor(log2_runtime, dtype=torch.float32),
            "threshold": threshold,
            "file": result.file,
            "backend": result.backend,
            "precision": result.precision,
        }

    @property
    def feature_dim(self) -> int:
        return len(self.NUMERIC_FEATURE_KEYS) + 3 + len(self.FAMILY_CATEGORIES)


THRESHOLD_FEATURE_IDX = len(NUMERIC_FEATURE_KEYS) + 2
FEATURE_DIM_WITHOUT_THRESHOLD = len(NUMERIC_FEATURE_KEYS) + 2 + len(FAMILY_CATEGORIES)


class ThresholdClassDataset(Dataset):
    """
    Dataset for threshold-class prediction: one sample per result.
    Features: all except duration and threshold (circuit + backend + precision + family).
    Target: threshold class index (0..len(THRESHOLD_LADDER)-1).
    """

    FAMILY_CATEGORIES = FAMILY_CATEGORIES
    NUMERIC_FEATURE_KEYS = NUMERIC_FEATURE_KEYS

    def __init__(
        self,
        data_path: Path,
        circuits_dir: Path,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        filter_ok_only: bool = True,
        fidelity_target: float = 0.75,
    ):
        self.circuits_dir = circuits_dir
        self.split = split
        self.fidelity_target = fidelity_target
        circuits, results = load_hackathon_data(data_path)
        self.circuit_info = {c.file: c for c in circuits}
        self.family_to_idx = {f: i for i, f in enumerate(self.FAMILY_CATEGORIES)}
        if filter_ok_only:
            results = [r for r in results if r.status == "ok"]
        circuit_files = list(set(r.file for r in results))
        rng = np.random.RandomState(seed)
        rng.shuffle(circuit_files)
        n_val = int(len(circuit_files) * val_fraction)
        val_files = set(circuit_files[:n_val])
        train_files = set(circuit_files[n_val:])
        if split == "train":
            results = [r for r in results if r.file in train_files]
        else:
            results = [r for r in results if r.file in val_files]
        self._samples: List[ResultEntry] = []
        for r in results:
            min_thr = compute_min_threshold(r.threshold_sweep, target=self.fidelity_target)
            if min_thr is not None:
                self._samples.append(r)
        self.results = results
        self._feature_cache: Dict[str, Dict] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def _get_circuit_features(self, file: str) -> Dict[str, Any]:
        if file not in self._feature_cache:
            qasm_path = self.circuits_dir / file
            self._feature_cache[file] = extract_qasm_features(qasm_path)
        return self._feature_cache[file]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = self._samples[idx]
        circuit = self.circuit_info.get(result.file)
        qasm_features = self._get_circuit_features(result.file)
        family_idx = self.family_to_idx.get(circuit.family if circuit else "", 0)
        family_onehot = torch.zeros(len(self.FAMILY_CATEGORIES))
        family_onehot[family_idx] = 1.0
        backend_idx = BACKEND_MAP.get(result.backend, 0)
        precision_idx = PRECISION_MAP.get(result.precision, 0)
        numeric_values = [qasm_features.get(k, 0.0) for k in self.NUMERIC_FEATURE_KEYS]
        numeric_values.extend([float(backend_idx), float(precision_idx)])
        numeric_features = torch.tensor(numeric_values, dtype=torch.float32)
        features = torch.cat([numeric_features, family_onehot])
        min_thr = compute_min_threshold(result.threshold_sweep, target=self.fidelity_target)
        threshold_class = threshold_to_class(min_thr)
        return {
            "features": features,
            "threshold_class": torch.tensor(threshold_class, dtype=torch.long),
            "file": result.file,
            "backend": result.backend,
            "precision": result.precision,
        }

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM_WITHOUT_THRESHOLD

    @property
    def num_threshold_classes(self) -> int:
        return len(THRESHOLD_LADDER)


def collate_fn_threshold_class(batch: List[Dict]) -> Dict[str, Any]:
    features = torch.stack([item["features"] for item in batch])
    threshold_class = torch.stack([item["threshold_class"] for item in batch])
    return {
        "features": features,
        "threshold_class": threshold_class,
        "file": [item["file"] for item in batch],
        "backend": [item["backend"] for item in batch],
        "precision": [item["precision"] for item in batch],
    }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    features = torch.stack([item["features"] for item in batch])
    log2_runtime = torch.stack([item["log2_runtime"] for item in batch])
    return {
        "features": features,
        "log2_runtime": log2_runtime,
        "threshold": [item["threshold"] for item in batch],
        "file": [item["file"] for item in batch],
        "backend": [item["backend"] for item in batch],
        "precision": [item["precision"] for item in batch],
    }


class HoldoutDataset(Dataset):
    """Dataset for holdout predictions (no labels)."""
    
    def __init__(
        self,
        holdout_path: Path,
        circuits_dir: Path,
        circuit_id_map: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            holdout_path: Path to holdout_public.json
            circuits_dir: Path to circuits directory
            circuit_id_map: Optional mapping from task ID to circuit file
        """
        self.circuits_dir = circuits_dir
        self.circuit_id_map = circuit_id_map or {}
        
        with open(holdout_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.tasks = data["tasks"]
        self._feature_cache: Dict[str, Dict] = {}
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task = self.tasks[idx]
        task_id = task["id"]
        processor = task["processor"]
        precision = task["precision"]
        
        circuit_file = self.circuit_id_map.get(task_id, "")
        
        return {
            "task_id": task_id,
            "processor": processor,
            "precision": precision,
            "circuit_file": circuit_file,
        }


def create_data_loaders(
    data_path: Path,
    circuits_dir: Path,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders. Threshold as input, target is log2(duration)."""
    train_dataset = QuantumCircuitDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="train",
        val_fraction=val_fraction,
        seed=seed,
    )
    val_dataset = QuantumCircuitDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="val",
        val_fraction=val_fraction,
        seed=seed,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def create_threshold_class_data_loaders(
    data_path: Path,
    circuits_dir: Path,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for threshold-class prediction (no duration, no threshold in features)."""
    train_dataset = ThresholdClassDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="train",
        val_fraction=val_fraction,
        seed=seed,
    )
    val_dataset = ThresholdClassDataset(
        data_path=data_path,
        circuits_dir=circuits_dir,
        split="val",
        val_fraction=val_fraction,
        seed=seed,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_threshold_class,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_threshold_class,
    )
    return train_loader, val_loader


class KFoldThresholdClassDataset(Dataset):
    """K-fold cross-validation for threshold-class prediction (one sample per result, no duration/threshold in features)."""

    FAMILY_CATEGORIES = ThresholdClassDataset.FAMILY_CATEGORIES
    NUMERIC_FEATURE_KEYS = ThresholdClassDataset.NUMERIC_FEATURE_KEYS

    def __init__(
        self,
        data_path: Path,
        circuits_dir: Path,
        fold: int,
        n_folds: int = 5,
        is_train: bool = True,
        seed: int = 42,
        filter_ok_only: bool = True,
        fidelity_target: float = 0.75,
    ):
        self.circuits_dir = circuits_dir
        self.fold = fold
        self.n_folds = n_folds
        self.is_train = is_train
        self.fidelity_target = fidelity_target
        circuits, results = load_hackathon_data(data_path)
        self.circuit_info = {c.file: c for c in circuits}
        self.family_to_idx = {f: i for i, f in enumerate(self.FAMILY_CATEGORIES)}
        if filter_ok_only:
            results = [r for r in results if r.status == "ok"]
        circuit_files = sorted(list(set(r.file for r in results)))
        rng = np.random.RandomState(seed)
        rng.shuffle(circuit_files)
        fold_size = len(circuit_files) // n_folds
        fold_starts = [i * fold_size for i in range(n_folds)]
        fold_starts.append(len(circuit_files))
        val_files = set(circuit_files[fold_starts[fold]:fold_starts[fold + 1]])
        train_files = set(circuit_files) - val_files
        if is_train:
            results = [r for r in results if r.file in train_files]
        else:
            results = [r for r in results if r.file in val_files]
        self._samples: List[ResultEntry] = []
        for r in results:
            min_thr = compute_min_threshold(r.threshold_sweep, target=self.fidelity_target)
            if min_thr is not None:
                self._samples.append(r)
        self.results = results
        self._feature_cache: Dict[str, Dict] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def _get_circuit_features(self, file: str) -> Dict[str, Any]:
        if file not in self._feature_cache:
            qasm_path = self.circuits_dir / file
            self._feature_cache[file] = extract_qasm_features(qasm_path)
        return self._feature_cache[file]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = self._samples[idx]
        circuit = self.circuit_info.get(result.file)
        qasm_features = self._get_circuit_features(result.file)
        family_idx = self.family_to_idx.get(circuit.family if circuit else "", 0)
        family_onehot = torch.zeros(len(self.FAMILY_CATEGORIES))
        family_onehot[family_idx] = 1.0
        backend_idx = BACKEND_MAP.get(result.backend, 0)
        precision_idx = PRECISION_MAP.get(result.precision, 0)
        numeric_values = [qasm_features.get(k, 0.0) for k in self.NUMERIC_FEATURE_KEYS]
        numeric_values.extend([float(backend_idx), float(precision_idx)])
        numeric_features = torch.tensor(numeric_values, dtype=torch.float32)
        features = torch.cat([numeric_features, family_onehot])
        min_thr = compute_min_threshold(result.threshold_sweep, target=self.fidelity_target)
        threshold_class = threshold_to_class(min_thr)
        return {
            "features": features,
            "threshold_class": torch.tensor(threshold_class, dtype=torch.long),
            "file": result.file,
            "backend": result.backend,
            "precision": result.precision,
        }

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM_WITHOUT_THRESHOLD

    @property
    def num_threshold_classes(self) -> int:
        return len(THRESHOLD_LADDER)


def create_kfold_threshold_class_data_loaders(
    data_path: Path,
    circuits_dir: Path,
    n_folds: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader]]:
    """Create k-fold cross-validation DataLoaders for threshold-class prediction."""
    fold_loaders = []
    for fold in range(n_folds):
        train_dataset = KFoldThresholdClassDataset(
            data_path=data_path,
            circuits_dir=circuits_dir,
            fold=fold,
            n_folds=n_folds,
            is_train=True,
            seed=seed,
        )
        val_dataset = KFoldThresholdClassDataset(
            data_path=data_path,
            circuits_dir=circuits_dir,
            fold=fold,
            n_folds=n_folds,
            is_train=False,
            seed=seed,
        )
        train_generator = torch.Generator()
        train_generator.manual_seed(seed + fold)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn_threshold_class,
            generator=train_generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_threshold_class,
        )
        fold_loaders.append((train_loader, val_loader))
    return fold_loaders


class KFoldQuantumCircuitDataset(Dataset):
    """K-fold cross-validation: duration prediction (threshold as input, log2(duration) target)."""

    FAMILY_CATEGORIES = QuantumCircuitDataset.FAMILY_CATEGORIES
    NUMERIC_FEATURE_KEYS = QuantumCircuitDataset.NUMERIC_FEATURE_KEYS

    def __init__(
        self,
        data_path: Path,
        circuits_dir: Path,
        fold: int,
        n_folds: int = 5,
        is_train: bool = True,
        seed: int = 42,
        filter_ok_only: bool = True,
    ):
        self.circuits_dir = circuits_dir
        self.fold = fold
        self.n_folds = n_folds
        self.is_train = is_train
        circuits, results = load_hackathon_data(data_path)
        self.circuit_info = {c.file: c for c in circuits}
        self.family_to_idx = {f: i for i, f in enumerate(self.FAMILY_CATEGORIES)}
        if filter_ok_only:
            results = [r for r in results if r.status == "ok"]
        circuit_files = sorted(list(set(r.file for r in results)))
        rng = np.random.RandomState(seed)
        rng.shuffle(circuit_files)
        fold_size = len(circuit_files) // n_folds
        fold_starts = [i * fold_size for i in range(n_folds)]
        fold_starts.append(len(circuit_files))
        val_files = set(circuit_files[fold_starts[fold]:fold_starts[fold + 1]])
        train_files = set(circuit_files) - val_files
        if is_train:
            results = [r for r in results if r.file in train_files]
        else:
            results = [r for r in results if r.file in val_files]
        self._samples: List[ResultEntry] = []
        for r in results:
            if r.forward_wall_s is not None and r.forward_wall_s > 0 and r.selected_threshold is not None:
                self._samples.append(r)
        self._feature_cache: Dict[str, Dict] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def _get_circuit_features(self, file: str) -> Dict[str, Any]:
        if file not in self._feature_cache:
            qasm_path = self.circuits_dir / file
            self._feature_cache[file] = extract_qasm_features(qasm_path)
        return self._feature_cache[file]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = self._samples[idx]
        circuit = self.circuit_info.get(result.file)
        qasm_features = self._get_circuit_features(result.file)
        family_idx = self.family_to_idx.get(circuit.family if circuit else "", 0)
        family_onehot = torch.zeros(len(self.FAMILY_CATEGORIES))
        family_onehot[family_idx] = 1.0
        backend_idx = BACKEND_MAP.get(result.backend, 0)
        precision_idx = PRECISION_MAP.get(result.precision, 0)
        threshold = result.selected_threshold
        log2_threshold = np.log2(max(threshold, 1))
        numeric_values = [qasm_features.get(k, 0.0) for k in self.NUMERIC_FEATURE_KEYS]
        numeric_values.extend([float(backend_idx), float(precision_idx), float(log2_threshold)])
        numeric_features = torch.tensor(numeric_values, dtype=torch.float32)
        features = torch.cat([numeric_features, family_onehot])
        log2_runtime = np.log2(max(float(result.forward_wall_s), 1e-10))
        return {
            "features": features,
            "log2_runtime": torch.tensor(log2_runtime, dtype=torch.float32),
            "threshold": threshold,
            "file": result.file,
            "backend": result.backend,
            "precision": result.precision,
        }

    @property
    def feature_dim(self) -> int:
        return len(self.NUMERIC_FEATURE_KEYS) + 3 + len(self.FAMILY_CATEGORIES)


def create_kfold_data_loaders(
    data_path: Path,
    circuits_dir: Path,
    n_folds: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader]]:
    """Create k-fold cross-validation data loaders.
    
    Returns:
        List of (train_loader, val_loader) tuples, one per fold.
    """
    fold_loaders = []
    
    for fold in range(n_folds):
        train_dataset = KFoldQuantumCircuitDataset(
            data_path=data_path,
            circuits_dir=circuits_dir,
            fold=fold,
            n_folds=n_folds,
            is_train=True,
            seed=seed,
        )
        
        val_dataset = KFoldQuantumCircuitDataset(
            data_path=data_path,
            circuits_dir=circuits_dir,
            fold=fold,
            n_folds=n_folds,
            is_train=False,
            seed=seed,
        )
        
        train_generator = torch.Generator()
        train_generator.manual_seed(seed + fold)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            generator=train_generator,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders


def get_feature_statistics(data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of features for normalization."""
    all_features = []
    for batch in data_loader:
        all_features.append(batch["features"])
    
    all_features = torch.cat(all_features, dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)
    std[std == 0] = 1.0
    
    return mean, std


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "hackathon_public.json"
    circuits_dir = project_root / "circuits"
    
    train_loader, val_loader = create_data_loaders(
        data_path=data_path,
        circuits_dir=circuits_dir,
        batch_size=16,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Features shape: {batch['features'].shape}")
    print(f"  log2_runtime shape: {batch['log2_runtime'].shape}")
    print(f"  Files: {batch['file'][:3]}...")

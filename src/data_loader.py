"""
PyTorch-compatible data loading for the quantum circuit fingerprint challenge.

Provides Dataset classes for training threshold prediction and runtime estimation models.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


THRESHOLD_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BACKEND_MAP = {"CPU": 0, "GPU": 1}
PRECISION_MAP = {"single": 0, "double": 1}


@dataclass
class CircuitInfo:
    file: str
    family: str
    n_qubits: int
    source_name: str = ""
    source_url: str = ""


@dataclass
class ThresholdSweepEntry:
    threshold: int
    sdk_get_fidelity: Optional[float]
    p_return_zero: Optional[float]
    run_wall_s: Optional[float]
    peak_rss_mb: Optional[float]
    returncode: int
    note: str


@dataclass
class ResultEntry:
    file: str
    backend: str
    precision: str
    status: str
    selected_threshold: Optional[int] = None
    target_fidelity: Optional[float] = None
    threshold_sweep: List[ThresholdSweepEntry] = field(default_factory=list)
    forward_wall_s: Optional[float] = None
    forward_shots: Optional[int] = None


def extract_qasm_features(qasm_path: Path) -> Dict[str, Any]:
    """Extract features from a QASM file for model input."""
    if not qasm_path.exists():
        return {}
    
    text = qasm_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    
    non_empty_lines = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("//"))
    n_qubits = 0
    qreg_match = re.search(r"qreg\s+\w+\[(\d+)\]", text)
    if qreg_match:
        n_qubits = int(qreg_match.group(1))
    
    n_cx = len(re.findall(r"\bcx\b", text))
    n_cz = len(re.findall(r"\bcz\b", text))
    n_swap = len(re.findall(r"\bswap\b", text))
    n_ccx = len(re.findall(r"\bccx\b", text))
    n_2q_gates = n_cx + n_cz + n_swap + n_ccx
    
    n_h = len(re.findall(r"\bh\b", text))
    n_x = len(re.findall(r"\bx\b", text))
    n_y = len(re.findall(r"\by\b", text))
    n_z = len(re.findall(r"\bz\b", text))
    n_s = len(re.findall(r"\bs\b", text))
    n_t = len(re.findall(r"\bt\b", text))
    n_rx = len(re.findall(r"\brx\b", text))
    n_ry = len(re.findall(r"\bry\b", text))
    n_rz = len(re.findall(r"\brz\b", text))
    n_u1 = len(re.findall(r"\bu1\b", text))
    n_u2 = len(re.findall(r"\bu2\b", text))
    n_u3 = len(re.findall(r"\bu3\b", text))
    n_1q_gates = n_h + n_x + n_y + n_z + n_s + n_t + n_rx + n_ry + n_rz + n_u1 + n_u2 + n_u3
    
    n_measure = len(re.findall(r"\bmeasure\b", text))
    n_barrier = len(re.findall(r"\bbarrier\b", text))
    n_custom_gates = len(re.findall(r"\bgate\s+\w+", text))
    
    qubit_pairs = set()
    for match in re.finditer(r"\bcx\s+(\w+)\[(\d+)\]\s*,\s*(\w+)\[(\d+)\]", text):
        q1, q2 = int(match.group(2)), int(match.group(4))
        qubit_pairs.add((min(q1, q2), max(q1, q2)))
    for match in re.finditer(r"\bcz\s+(\w+)\[(\d+)\]\s*,\s*(\w+)\[(\d+)\]", text):
        q1, q2 = int(match.group(2)), int(match.group(4))
        qubit_pairs.add((min(q1, q2), max(q1, q2)))
    
    n_unique_pairs = len(qubit_pairs)
    
    if qubit_pairs:
        spans = [abs(q2 - q1) for q1, q2 in qubit_pairs]
        avg_span = np.mean(spans)
        max_span = max(spans)
    else:
        avg_span = 0.0
        max_span = 0
    
    gate_density = n_2q_gates / max(n_qubits, 1)
    
    return {
        "n_lines": non_empty_lines,
        "n_qubits": n_qubits,
        "n_cx": n_cx,
        "n_cz": n_cz,
        "n_swap": n_swap,
        "n_ccx": n_ccx,
        "n_2q_gates": n_2q_gates,
        "n_1q_gates": n_1q_gates,
        "n_measure": n_measure,
        "n_barrier": n_barrier,
        "n_custom_gates": n_custom_gates,
        "n_unique_pairs": n_unique_pairs,
        "avg_span": avg_span,
        "max_span": max_span,
        "gate_density": gate_density,
        "n_h": n_h,
        "n_rx": n_rx,
        "n_ry": n_ry,
        "n_rz": n_rz,
    }


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


def compute_min_threshold(sweep: List[ThresholdSweepEntry], target: float = 0.99) -> Optional[int]:
    """Find the minimum threshold that meets the target fidelity."""
    for entry in sorted(sweep, key=lambda x: x.threshold):
        fid = entry.sdk_get_fidelity
        if fid is not None and fid >= target:
            return entry.threshold
    return None


class QuantumCircuitDataset(Dataset):
    """PyTorch Dataset for quantum circuit threshold/runtime prediction."""
    
    FAMILY_CATEGORIES = [
        "Amplitude_Estimation", "CutBell", "Deutsch_Jozsa", "GHZ", "GraphState",
        "Ground_State", "Grover_NoAncilla", "Grover_V_Chain", "Portfolio_QAOA",
        "Portfolio_VQE", "Pricing_Call", "QAOA", "QFT", "QFT_Entangled", "QNN",
        "QPE_Exact", "Shor", "TwoLocalRandom", "VQE", "W_State"
    ]
    
    def __init__(
        self,
        data_path: Path,
        circuits_dir: Path,
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        filter_ok_only: bool = True,
    ):
        """
        Args:
            data_path: Path to hackathon_public.json
            circuits_dir: Path to circuits directory
            split: "train" or "val"
            val_fraction: Fraction of circuits for validation (split by circuit file)
            seed: Random seed for split
            filter_ok_only: Only include results with status="ok"
        """
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
            self.results = [r for r in results if r.file in train_files]
        else:
            self.results = [r for r in results if r.file in val_files]
        
        self._feature_cache: Dict[str, Dict] = {}
    
    def __len__(self) -> int:
        return len(self.results)
    
    def _get_circuit_features(self, file: str) -> Dict[str, Any]:
        if file not in self._feature_cache:
            qasm_path = self.circuits_dir / file
            self._feature_cache[file] = extract_qasm_features(qasm_path)
        return self._feature_cache[file]
    
    def _get_threshold_label(self, result: ResultEntry) -> int:
        """Get the minimum threshold meeting fidelity target as a class index."""
        min_thresh = compute_min_threshold(result.threshold_sweep, target=0.99)
        if min_thresh is None:
            return len(THRESHOLD_LADDER) - 1
        try:
            return THRESHOLD_LADDER.index(min_thresh)
        except ValueError:
            for i, t in enumerate(THRESHOLD_LADDER):
                if t >= min_thresh:
                    return i
            return len(THRESHOLD_LADDER) - 1
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = self.results[idx]
        circuit = self.circuit_info.get(result.file)
        qasm_features = self._get_circuit_features(result.file)
        
        family_idx = self.family_to_idx.get(circuit.family if circuit else "", 0)
        family_onehot = torch.zeros(len(self.FAMILY_CATEGORIES))
        family_onehot[family_idx] = 1.0
        
        backend_idx = BACKEND_MAP.get(result.backend, 0)
        precision_idx = PRECISION_MAP.get(result.precision, 0)
        
        numeric_features = torch.tensor([
            qasm_features.get("n_qubits", 0),
            qasm_features.get("n_lines", 0),
            qasm_features.get("n_cx", 0),
            qasm_features.get("n_cz", 0),
            qasm_features.get("n_2q_gates", 0),
            qasm_features.get("n_1q_gates", 0),
            qasm_features.get("n_unique_pairs", 0),
            qasm_features.get("avg_span", 0.0),
            qasm_features.get("max_span", 0),
            qasm_features.get("gate_density", 0.0),
            qasm_features.get("n_custom_gates", 0),
            backend_idx,
            precision_idx,
        ], dtype=torch.float32)
        
        features = torch.cat([numeric_features, family_onehot])
        
        threshold_class = self._get_threshold_label(result)
        
        forward_time = result.forward_wall_s if result.forward_wall_s else 0.0
        log_runtime = np.log1p(forward_time)
        
        return {
            "features": features,
            "threshold_class": torch.tensor(threshold_class, dtype=torch.long),
            "log_runtime": torch.tensor(log_runtime, dtype=torch.float32),
            "file": result.file,
            "backend": result.backend,
            "precision": result.precision,
        }
    
    @property
    def feature_dim(self) -> int:
        return 13 + len(self.FAMILY_CATEGORIES)
    
    @property
    def num_threshold_classes(self) -> int:
        return len(THRESHOLD_LADDER)


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
    """Create train and validation DataLoaders."""
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate function to handle string fields."""
    features = torch.stack([item["features"] for item in batch])
    threshold_class = torch.stack([item["threshold_class"] for item in batch])
    log_runtime = torch.stack([item["log_runtime"] for item in batch])
    
    return {
        "features": features,
        "threshold_class": threshold_class,
        "log_runtime": log_runtime,
        "file": [item["file"] for item in batch],
        "backend": [item["backend"] for item in batch],
        "precision": [item["precision"] for item in batch],
    }


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
    print(f"  Threshold classes: {batch['threshold_class']}")
    print(f"  Log runtimes: {batch['log_runtime']}")
    print(f"  Files: {batch['file'][:3]}...")

import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch.utils.data import DataLoader

from models.base import BaseModel
import joblib

# Ladder values defined by challenge
THRESHOLD_LADDER = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])

class HierarchicalDecoupledModel(BaseModel):
    """
    Decouples threshold and runtime prediction. 
    Threshold is predicted as a class, then used as a feature for runtime regression.
    """

    def __init__(self, conservative_mode: bool = True):
        self.threshold_model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.runtime_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.conservative_mode = conservative_mode
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "HierarchicalDecoupledRF"

    def _loader_to_numpy(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper to extract features, threshold labels, and runtimes from DataLoader."""
        all_features, all_thresholds, all_runtimes = [], [], []
        
        for batch in loader:
            # FIX: Unpack 3 items directly instead of expecting a nested tuple
            # If your dataset returns (x, threshold, runtime), this will work:
            x, y_t, y_r = batch
            
            all_features.append(x.numpy())
            all_thresholds.append(y_t.numpy())
            all_runtimes.append(y_r.numpy())
            
        return (np.vstack(all_features), 
                np.concatenate(all_thresholds), 
                np.concatenate(all_runtimes))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = False) -> Dict[str, Any]:
        X_train, y_t_train, y_r_train = self._loader_to_numpy(train_loader)
        
        if verbose: print(f"[{self.name}] Training threshold classifier...")
        self.threshold_model.fit(X_train, y_t_train)
        
        # Augment features for runtime: Use the true thresholds during training (Teacher Forcing)
        # Using log2 of threshold to linearize the ladder
        X_runtime_train = np.column_stack([X_train, np.log2(y_t_train)])
        
        if verbose: print(f"[{self.name}] Training runtime regressor on log(runtime)...")
        # Model log(runtime) to handle scaling across orders of magnitude
        self.runtime_model.fit(X_runtime_train, np.log(y_r_train))
        
        self._is_fitted = True
        return self.evaluate(val_loader)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        # 1. Predict threshold class
        # Classifier returns the index or the ladder value depending on your label encoding
        pred_thresholds = self.threshold_model.predict(features)
        
        if self.conservative_mode:
            # Optional: Heuristic to bump threshold if uncertainty is high 
            # to avoid the 'score 0' penalty
            pass 

        # 2. Predict Runtime using the threshold prediction as a feature
        X_runtime_test = np.column_stack([features, np.log2(pred_thresholds)])
        log_runtimes = self.runtime_model.predict(X_runtime_test)
        
        return pred_thresholds.astype(int), np.exp(log_runtimes)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        X, y_t_true, y_r_true = self._loader_to_numpy(loader)
        y_t_pred, y_r_pred = self.predict(X)
        
        # Accuracy for classification
        t_acc = np.mean(y_t_pred == y_t_true)
        
        # Error for runtime in log-space (as it is scored symmetrically/normalized)
        log_r_true = np.log(y_r_true)
        log_r_pred = np.log(y_r_pred)
        
        mse = np.mean((log_r_true - log_r_pred)**2)
        mae = np.mean(np.abs(log_r_true - log_r_pred))
        
        return {
            "threshold_accuracy": float(t_acc),
            "runtime_mse": float(mse),
            "runtime_mae": float(mae)
        }

    def save(self, path: Path) -> None:
        import job_lib # type: ignore
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.threshold_model, path / "threshold_rf.joblib")
        joblib.dump(self.runtime_model, path / "runtime_rf.joblib")

    def load(self, path: Path) -> None:
        import joblib # type: ignore
        self.threshold_model = joblib.load(path / "threshold_rf.joblib")
        self.runtime_model = joblib.load(path / "runtime_rf.joblib")
        self._is_fitted = True
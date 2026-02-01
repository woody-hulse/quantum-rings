"""
The Darwinian Model (Hybrid GPU Edition - Fixed for XGBoost 1.6).
- CPU: Evolves new mathematical formulas (Genetic Programming).
- GPU: Trains massive XGBoost ensembles using 'gpu_hist'.

Fixes:
- Replaced device='cuda' with tree_method='gpu_hist' to support your installed version.
"""

from typing import Dict, Tuple, Any
from pathlib import Path

import numpy as np
import joblib
from torch.utils.data import DataLoader

# Scikit-learn imports
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator

# Genetic Programming (CPU Only)
try:
    from gplearn.genetic import SymbolicTransformer
except ImportError:
    raise ImportError("You MUST run: pip install gplearn")

# XGBoost (GPU Enabled)
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    raise ImportError("You MUST run: pip install xgboost")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER
from models.base import BaseModel


class PyTorchCascadingModel(BaseModel):
    """
    Renamed to PyTorchCascadingModel for pipeline compatibility.
    """
    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.threshold_model = None
        self.runtime_model = None
        
        # --- STAGE 1: GENETIC EVOLUTION (CPU) ---
        # "The Scientist": Invents new features using math.
        self.genetic_engine = SymbolicTransformer(
            generations=10,
            population_size=500,     # Safe size for RAM
            hall_of_fame=50,
            n_components=20, 
            function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'),
            metric='pearson',
            verbose=1,
            random_state=random_state,
            n_jobs=2                 # Limit CPU cores to prevent RAM crash
        )
        
        # --- STAGE 2: THE GPU STACK (XGBoost 1.6 Compatible) ---
        # "The Machine": Trains massive models on the evolved features.
        estimators = [
            ('xgb_deep', XGBClassifier(
                n_estimators=500,
                max_depth=8,            
                learning_rate=0.02,     
                tree_method='gpu_hist', # <--- FIXED: Explicit GPU mode for v1.6
                predictor='gpu_predictor', # <--- FIXED: Accelerates prediction
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('xgb_wide', XGBClassifier(
                n_estimators=500,
                max_depth=4,            
                learning_rate=0.05,
                tree_method='gpu_hist', # <--- FIXED
                predictor='gpu_predictor',
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            ))
        ]
        
        # --- STAGE 3: META LEARNER ---
        self.threshold_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000),
            cv=3,
            n_jobs=1  # XGBoost manages its own threads/GPU
        )

        # Basic Runtime Model (Also on GPU)
        self.runtime_model = XGBRegressor(
            n_estimators=200, 
            max_depth=3, 
            learning_rate=0.05,
            tree_method='gpu_hist', # <--- FIXED
            predictor='gpu_predictor',
            random_state=random_state
        )
        
        # Full Pipeline
        self.full_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('genetics', self.genetic_engine),
            ('stack', self.threshold_model)
        ])

    @property
    def name(self) -> str:
        return "Darwinian_GPU_Stack"

    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_features = []
        all_thresh = []
        all_runtime = []

        for batch in loader:
            all_features.append(batch["features"].numpy())
            all_thresh.extend(batch["threshold_class"].tolist())
            all_runtime.extend(batch["log_runtime"].tolist())

        return np.vstack(all_features), np.array(all_thresh), np.array(all_runtime)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, verbose: bool = False, **kwargs) -> Dict[str, Any]:
        
        X_train, y_thresh_train, y_runtime_train = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val = self._extract_data(val_loader)

        if verbose:
            print(f"--- EVOLVING (CPU) & TRAINING (GPU) ---")

        # 1. Train Full Pipeline
        self.full_pipeline.fit(X_train, y_thresh_train)
        
        # Evaluation
        train_pred = self.full_pipeline.predict(X_train)
        val_pred = self.full_pipeline.predict(X_val)
        
        train_acc = accuracy_score(y_thresh_train, train_pred)
        val_acc = accuracy_score(y_thresh_val, val_pred)
        
        if verbose:
            print(f" -> Train Acc: {train_acc:.2%}")
            print(f" -> Val Acc:   {val_acc:.2%}")

        # 2. Train Runtime Model
        self.runtime_model.fit(X_train, y_runtime_train)
        
        runtime_log_val = self.runtime_model.predict(X_val)
        runtime_log_val = np.clip(runtime_log_val, 0, 15) 
        runtime_pred_val = np.expm1(runtime_log_val)
        
        val_mse = mean_squared_error(y_runtime_val, runtime_pred_val)

        return {
            "val": {
                "threshold_accuracy": val_acc,
                "runtime_mse": val_mse
            }
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        X, y_thresh, y_runtime = self._extract_data(loader)
        
        t_pred = self.full_pipeline.predict(X)
        
        r_log = self.runtime_model.predict(X) 
        r_log = np.clip(r_log, 0, 15)
        r_pred_expm1 = np.expm1(r_log)

        return {
            "threshold_accuracy": accuracy_score(y_thresh, t_pred),
            "runtime_mse": mean_squared_error(y_runtime, r_pred_expm1),
            "runtime_mae": mean_absolute_error(y_runtime, r_pred_expm1),
        }

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Genetics + Stack
        class_indices = self.full_pipeline.predict(features)
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in class_indices])
        
        # Runtime
        runtime_log = self.runtime_model.predict(features)
        runtime_log = np.clip(runtime_log, 0, 15)
        runtime_values = np.expm1(runtime_log)
        
        return thresh_values, runtime_values

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.full_pipeline, path / "genetic_stack.joblib")
        joblib.dump(self.runtime_model, path / "runtime_basic.joblib")

    def load(self, path: Path) -> None:
        path = Path(path)
        self.full_pipeline = joblib.load(path / "genetic_stack.joblib")
        self.runtime_model = joblib.load(path / "runtime_basic.joblib")
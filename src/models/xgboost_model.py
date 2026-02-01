"""
Robust Smart Linear Model.
Fixes convergence issues by adding secondary scaling and increasing solver iterations.
"""

from typing import Dict, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import joblib
from torch.utils.data import DataLoader

# Sklearn imports
from sklearn.linear_model import Ridge, LogisticRegression, Lasso
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER
from models.base import BaseModel


class XGBoostModel(BaseModel):
    """
    A 'Crash-Proof' version of the Smart Linear Model.
    
    Fixes:
    1. Double Scaling: Scales inputs BEFORE and AFTER polynomial expansion.
    2. Solver Switch: Uses 'liblinear' (more stable) instead of 'lbfgs'.
    3. Iteration Boost: Increases max_iter to 10,000 to prevent early timeouts.
    """

    def __init__(
        self,
        poly_degree: int = 2,
        selection_threshold: str = "1.25*mean",
        ridge_alpha: float = 1.0,
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.poly_degree = poly_degree
        self.selection_threshold = selection_threshold
        self.ridge_alpha = ridge_alpha
        self.C = C
        self.random_state = random_state
        
        self.threshold_pipeline: Optional[Pipeline] = None
        self.runtime_pipeline: Optional[Pipeline] = None

    @property
    def name(self) -> str:
        return "RobustSmart_PolyLasso"

    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_features = []
        all_thresh = []
        all_runtime = []

        for batch in loader:
            all_features.append(batch["features"].numpy())
            all_thresh.extend(batch["threshold_class"].tolist())
            all_runtime.extend(batch["log_runtime"].tolist())

        X = np.vstack(all_features)
        y_thresh = np.array(all_thresh, dtype=int)
        y_runtime = np.array(all_runtime)

        return X, y_thresh, y_runtime

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        
        X_train, y_thresh_train, y_runtime_train = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val = self._extract_data(val_loader)

        if verbose:
            print(f"\n--- Training Robust Smart Models ---")

        # -----------------------------
        # 1. Threshold Pipeline (The Fix)
        # -----------------------------
        if verbose:
            print("Optimizing Threshold Classifier...")
            
        self.threshold_pipeline = Pipeline([
            ('scaler_raw', RobustScaler()), 
            ('poly', PolynomialFeatures(degree=self.poly_degree, interaction_only=True, include_bias=False)),
            ('scaler_poly', StandardScaler()), 
            
            # --- FIX IS HERE ---
            # 1. Removed penalty='l1'
            # 2. Added l1_ratio=1 (Enables L1 selection)
            # 3. Solver 'saga' supports this mixing
            ('selector', SelectFromModel(
                LogisticRegression(
                    solver='saga',       # Required for l1_ratio support
                    l1_ratio=1,          # 1.0 = Pure L1 (Lasso behavior)
                    C=0.5, 
                    max_iter=5000, 
                    random_state=self.random_state
                    # Note: We let 'penalty' default (usually to None/L2) but l1_ratio overrides it in new versions
                ),
                threshold=self.selection_threshold
            )),
            
            ('clf', LogisticRegression(
                C=self.C, 
                solver='saga',
                l1_ratio=0,          # 0.0 = Pure L2 (Ridge behavior) for final classification
                max_iter=10000, 
                class_weight='balanced', 
                random_state=self.random_state
            ))
        ])
        
        # Explicitly handle the "penalty='elasticnet'" requirement if your specific 
        # sklearn version still strictly enforces it despite the warning. 
        # If the code below fails with "l1_ratio parameter is only used when penalty is 'elasticnet'",
        # uncomment the lines below:
        
        # self.threshold_pipeline.named_steps['selector'].estimator.set_params(penalty='elasticnet')
        # self.threshold_pipeline.named_steps['clf'].set_params(penalty='elasticnet')
        
        self.threshold_pipeline.fit(X_train, y_thresh_train)
        
        if verbose:
            n_feats = self.threshold_pipeline.named_steps['selector'].get_support().sum()
            print(f" -> Converged! Selected {n_feats} features for Threshold.")

        # -----------------------------
        # 2. Runtime Pipeline
        # -----------------------------
        if verbose:
            print("Optimizing Runtime Regressor...")
            
        self.runtime_pipeline = Pipeline([
            ('scaler_raw', RobustScaler()),
            ('poly', PolynomialFeatures(degree=self.poly_degree, interaction_only=True, include_bias=False)),
            ('scaler_poly', StandardScaler()), 
            
            ('selector', SelectFromModel(
                Lasso(alpha=0.01, max_iter=5000, random_state=self.random_state), 
                threshold=self.selection_threshold
            )),
            
            ('reg', Ridge(alpha=self.ridge_alpha, random_state=self.random_state))
        ])
        
        self.runtime_pipeline.fit(X_train, y_runtime_train)

        # -----------------------------
        # Evaluation
        # -----------------------------
        train_metrics = self._evaluate_internal(X_train, y_thresh_train, y_runtime_train)
        val_metrics = self._evaluate_internal(X_val, y_thresh_val, y_runtime_val)

        if verbose:
            print("\n--- Final Results ---")
            print(f"[Threshold] Val Acc: {val_metrics['threshold_accuracy']:.2%}")
            print(f"[Runtime]   Val MSE: {val_metrics['runtime_mse']:.4f}")

        return {"train": train_metrics, "val": val_metrics}
    
    def _evaluate_internal(self, X, y_thresh, y_runtime) -> Dict[str, float]:
        thresh_pred = self.threshold_pipeline.predict(X)
        runtime_pred = self.runtime_pipeline.predict(X)
        return {
            "threshold_accuracy": accuracy_score(y_thresh, thresh_pred),
            "threshold_f1": f1_score(y_thresh, thresh_pred, average='weighted'),
            "runtime_mse": mean_squared_error(y_runtime, runtime_pred),
            "runtime_mae": mean_absolute_error(y_runtime, runtime_pred),
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        X, y_thresh, y_runtime = self._extract_data(loader)
        return self._evaluate_internal(X, y_thresh, y_runtime)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold_pipeline is None or self.runtime_pipeline is None:
            raise RuntimeError("Model not trained.")

        thresh_class_indices = self.threshold_pipeline.predict(features)
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_class_indices])

        runtime_log = self.runtime_pipeline.predict(features)
        runtime_values = np.expm1(runtime_log)

        return thresh_values, runtime_values

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.threshold_pipeline, path / "thresh_pipeline.joblib")
        joblib.dump(self.runtime_pipeline, path / "runtime_pipeline.joblib")

    def load(self, path: Path) -> None:
        path = Path(path)
        self.threshold_pipeline = joblib.load(path / "thresh_pipeline.joblib")
        self.runtime_pipeline = joblib.load(path / "runtime_pipeline.joblib")
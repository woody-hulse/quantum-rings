"""
Rohan7: Optimized ensemble model emphasizing top correlated features.

This model performs intelligent feature selection based on correlation analysis,
then trains separate gradient boosting models for threshold and runtime prediction.

Top Features:
- Threshold: avg_degree, n_unique_pairs, clustering_coeff, n_cz, cut_crossing_ratio, n_ry
- Runtime: max_span, graph_bandwidth, span_std, entanglement_velocity, avg_span, n_measure
"""

from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER
from models.base import BaseModel

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class Rohan7Model(BaseModel):
    """
    Optimized ensemble model with correlation-based feature selection.
    
    Uses the top correlated features for each prediction task:
    - Threshold prediction: focuses on degree/connectivity metrics
    - Runtime prediction: focuses on entanglement/span metrics
    """
    
    # Top features for each task (based on correlation analysis)
    THRESHOLD_FEATURES = [
        'avg_degree', 'n_unique_pairs', 'clustering_coeff', 'n_cz', 
        'cut_crossing_ratio', 'n_ry', 'cx_rz_cx_pattern_count', 'n_barrier',
        'rotation_density', 'n_swap', 'n_connected_components', 'early_longrange_ratio',
        'final_light_cone_size', 'entanglement_velocity', 'avg_span', 'long_range_ratio', 
        'cx_chain_max_length', 'n_qubits', 'degree_entropy', 'n_lines', 'estimated_depth'
    ]
    
    RUNTIME_FEATURES = [
        'max_span', 'graph_bandwidth', 'span_std', 'entanglement_velocity',
        'avg_span', 'n_measure', 'n_qubits', 'n_swap', 'n_connected_components',
        'degree_entropy', 'n_unique_pairs', 'max_degree', 'normalized_bandwidth',
        'cut_crossing_ratio', 'nearest_neighbor_ratio', 'n_lq_gates', 'n_h', "rotation_density",
        "quibit_activity_max_ratio","fine_light_cone_size","cx_rz_cx_pattern_count"
    ]
    
    def __init__(
        self,
        max_depth: int = 7,
        learning_rate: float = 0.15,
        n_estimators: int = 150,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        min_child_weight: int = 2,
        random_state: int = 42,
        use_lightgbm: bool = False,
    ):
        if not HAS_XGBOOST and not HAS_LIGHTGBM:
            raise ImportError("xgboost or lightgbm is required")
        
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        
        self.threshold_model: Optional[Any] = None
        self.runtime_model: Optional[Any] = None
        
        self.threshold_scaler = StandardScaler()
        self.runtime_scaler = StandardScaler()
        
        # Will store selected feature indices after fitting
        self.threshold_feature_indices: List[int] = []
        self.runtime_feature_indices: List[int] = []
        
        self.all_feature_names: List[str] = []
    
    @property
    def name(self) -> str:
        return "Rohan7"
    
    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Extract features and targets from data loader."""
        all_features = []
        all_thresh = []
        all_runtime = []
        
        for batch in loader:
            all_features.append(batch["features"].numpy())
            all_thresh.extend(batch["threshold_class"].tolist())
            all_runtime.extend(batch["log_runtime"].tolist())
        
        X = np.vstack(all_features)
        y_thresh = np.array(all_thresh, dtype=np.float32)
        y_runtime = np.array(all_runtime)
        
        # Get feature names from dataset
        dataset = loader.dataset
        feature_names = (
            list(dataset.NUMERIC_FEATURE_KEYS) + 
            ["backend", "precision"] + 
            [f.replace("_", "") for f in dataset.FAMILY_CATEGORIES]
        )
        
        return X, y_thresh, y_runtime, feature_names
    
    def _select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                        target_features: List[str]) -> List[int]:
        """Select feature indices based on target feature names."""
        indices = []
        for target in target_features:
            for i, name in enumerate(feature_names):
                if name == target:
                    indices.append(i)
                    break
        return indices if indices else list(range(X.shape[1]))
    
    def _round_to_class(self, pred: np.ndarray) -> np.ndarray:
        """Round regression predictions to nearest valid class index."""
        return np.clip(np.round(pred), 0, len(THRESHOLD_LADDER) - 1).astype(int)
    
    def _create_model(self, model_type: str = "xgboost"):
        """Create a new gradient boosting model."""
        if self.use_lightgbm:
            return lgb.LGBMRegressor(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                random_state=self.random_state,
                verbose=-1,
            )
        else:
            return xgb.XGBRegressor(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                random_state=self.random_state,
                objective="reg:squarederror",
            )
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train both threshold and runtime models with selected features."""
        X_train, y_thresh_train, y_runtime_train, feature_names = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val, _ = self._extract_data(val_loader)
        
        self.all_feature_names = feature_names
        
        # Select features for each task
        self.threshold_feature_indices = self._select_features(
            X_train, y_thresh_train, feature_names, self.THRESHOLD_FEATURES
        )
        self.runtime_feature_indices = self._select_features(
            X_train, y_runtime_train, feature_names, self.RUNTIME_FEATURES
        )
        
        if verbose:
            print(f"Threshold features selected: {len(self.threshold_feature_indices)}")
            print(f"Runtime features selected: {len(self.runtime_feature_indices)}")
        
        # Extract selected features
        X_train_thresh = X_train[:, self.threshold_feature_indices]
        X_val_thresh = X_val[:, self.threshold_feature_indices]
        X_train_runtime = X_train[:, self.runtime_feature_indices]
        X_val_runtime = X_val[:, self.runtime_feature_indices]
        
        # Scale features
        X_train_thresh_scaled = self.threshold_scaler.fit_transform(X_train_thresh)
        X_val_thresh_scaled = self.threshold_scaler.transform(X_val_thresh)
        X_train_runtime_scaled = self.runtime_scaler.fit_transform(X_train_runtime)
        X_val_runtime_scaled = self.runtime_scaler.transform(X_val_runtime)
        
        # Train threshold model
        if verbose:
            print("Training threshold model...")
        self.threshold_model = self._create_model()
        
        if self.use_lightgbm:
            self.threshold_model.fit(
                X_train_thresh_scaled, y_thresh_train,
                eval_set=[(X_val_thresh_scaled, y_thresh_val)],
                eval_metric="rmse",
                callbacks=[lgb.log_evaluation(period=0)],
            )
        else:
            self.threshold_model.fit(
                X_train_thresh_scaled, y_thresh_train,
                eval_set=[(X_train_thresh_scaled, y_thresh_train), 
                         (X_val_thresh_scaled, y_thresh_val)],
                verbose=False,
            )
        
        # Train runtime model
        if verbose:
            print("Training runtime model...")
        self.runtime_model = self._create_model()
        
        if self.use_lightgbm:
            self.runtime_model.fit(
                X_train_runtime_scaled, y_runtime_train,
                eval_set=[(X_val_runtime_scaled, y_runtime_val)],
                eval_metric="rmse",
                callbacks=[lgb.log_evaluation(period=0)],
            )
        else:
            self.runtime_model.fit(
                X_train_runtime_scaled, y_runtime_train,
                eval_set=[(X_train_runtime_scaled, y_runtime_train), 
                         (X_val_runtime_scaled, y_runtime_val)],
                verbose=False,
            )
        
        # Evaluate
        train_metrics = self._evaluate_internal(
            X_train_thresh_scaled, y_thresh_train,
            X_train_runtime_scaled, y_runtime_train
        )
        val_metrics = self._evaluate_internal(
            X_val_thresh_scaled, y_thresh_val,
            X_val_runtime_scaled, y_runtime_val
        )
        
        if verbose:
            print(f"Train Threshold Acc: {train_metrics['threshold_accuracy']:.4f} | "
                  f"Train Runtime MSE: {train_metrics['runtime_mse']:.4f}")
            print(f"Val Threshold Acc: {val_metrics['threshold_accuracy']:.4f} | "
                  f"Val Runtime MSE: {val_metrics['runtime_mse']:.4f}")
        
        return {"train": train_metrics, "val": val_metrics}
    
    def _evaluate_internal(
        self,
        X_thresh: np.ndarray,
        y_thresh: np.ndarray,
        X_runtime: np.ndarray,
        y_runtime: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate on scaled, selected features."""
        thresh_pred_raw = self.threshold_model.predict(X_thresh)
        thresh_pred = self._round_to_class(thresh_pred_raw)
        runtime_pred = self.runtime_model.predict(X_runtime)
        
        return {
            "threshold_accuracy": accuracy_score(y_thresh.astype(int), thresh_pred),
            "runtime_mse": mean_squared_error(y_runtime, runtime_pred),
            "runtime_mae": mean_absolute_error(y_runtime, runtime_pred),
        }
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on data loader."""
        X, y_thresh, y_runtime, _ = self._extract_data(loader)
        
        X_thresh = X[:, self.threshold_feature_indices]
        X_runtime = X[:, self.runtime_feature_indices]
        
        X_thresh_scaled = self.threshold_scaler.transform(X_thresh)
        X_runtime_scaled = self.runtime_scaler.transform(X_runtime)
        
        return self._evaluate_internal(X_thresh_scaled, y_thresh, X_runtime_scaled, y_runtime)
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on full feature set."""
        if self.threshold_model is None or self.runtime_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X_thresh = features[:, self.threshold_feature_indices]
        X_runtime = features[:, self.runtime_feature_indices]
        
        X_thresh_scaled = self.threshold_scaler.transform(X_thresh)
        X_runtime_scaled = self.runtime_scaler.transform(X_runtime)
        
        thresh_pred_raw = self.threshold_model.predict(X_thresh_scaled)
        thresh_classes = self._round_to_class(thresh_pred_raw)
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])
        
        runtime_log = self.runtime_model.predict(X_runtime_scaled)
        runtime_values = np.expm1(runtime_log)
        
        return thresh_values, runtime_values
    
    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        """Get feature importances from both models."""
        if self.threshold_model is None or self.runtime_model is None:
            return None
        
        try:
            return {
                "threshold": self.threshold_model.feature_importances_,
                "runtime": self.runtime_model.feature_importances_,
            }
        except AttributeError:
            return None
    
    def save(self, path: Path) -> None:
        """Save model and scalers."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.use_lightgbm:
            self.threshold_model.booster_.save_model(str(path / "threshold_model.txt"))
            self.runtime_model.booster_.save_model(str(path / "runtime_model.txt"))
        else:
            self.threshold_model.save_model(path / "threshold_model.json")
            self.runtime_model.save_model(path / "runtime_model.json")
        
        np.save(path / "threshold_scaler_mean.npy", self.threshold_scaler.mean_)
        np.save(path / "threshold_scaler_scale.npy", self.threshold_scaler.scale_)
        np.save(path / "runtime_scaler_mean.npy", self.runtime_scaler.mean_)
        np.save(path / "runtime_scaler_scale.npy", self.runtime_scaler.scale_)
        np.save(path / "threshold_indices.npy", np.array(self.threshold_feature_indices))
        np.save(path / "runtime_indices.npy", np.array(self.runtime_feature_indices))
    
    def load(self, path: Path) -> None:
        """Load model and scalers."""
        path = Path(path)
        
        if self.use_lightgbm:
            self.threshold_model = lgb.Booster(model_file=str(path / "threshold_model.txt"))
            self.runtime_model = lgb.Booster(model_file=str(path / "runtime_model.txt"))
        else:
            self.threshold_model = xgb.XGBRegressor()
            self.threshold_model.load_model(path / "threshold_model.json")
            
            self.runtime_model = xgb.XGBRegressor()
            self.runtime_model.load_model(path / "runtime_model.json")
        
        self.threshold_scaler.mean_ = np.load(path / "threshold_scaler_mean.npy")
        self.threshold_scaler.scale_ = np.load(path / "threshold_scaler_scale.npy")
        self.runtime_scaler.mean_ = np.load(path / "runtime_scaler_mean.npy")
        self.runtime_scaler.scale_ = np.load(path / "runtime_scaler_scale.npy")
        
        self.threshold_feature_indices = np.load(path / "threshold_indices.npy").tolist()
        self.runtime_feature_indices = np.load(path / "runtime_indices.npy").tolist()

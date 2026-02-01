"""
Separate XGBoost models for threshold and runtime with top 10 most correlated features.

Trains models independently using only the 10 most correlated features for each task.
Includes per-epoch loss printing and 5-epoch curve visualization.
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
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SeparateThresholdRuntimeModel(BaseModel):
    """Separate XGBoost models using top 10 most correlated features for each task."""
    
    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        min_epochs: int = 100,
        save_curves: bool = True,
        curves_output_dir: Optional[Path] = None,
        save_interval: int = 5,
    ):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required. Install with: pip install xgboost")
        
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.min_epochs = min_epochs
        self.save_curves = save_curves
        self.save_interval = save_interval
        
        # Models and scalers
        self.threshold_model: Optional[xgb.XGBRegressor] = None
        self.runtime_model: Optional[xgb.XGBRegressor] = None
        
        self.threshold_scaler = StandardScaler()
        self.runtime_scaler = StandardScaler()
        
        # Feature indices and names
        self.threshold_feature_indices: List[int] = []
        self.threshold_feature_names: List[str] = []
        self.runtime_feature_indices: List[int] = []
        self.runtime_feature_names: List[str] = []
        
        # Training history
        self.threshold_history: Dict[str, List[float]] = {"train": [], "val": []}
        self.runtime_history: Dict[str, List[float]] = {"train": [], "val": []}
        
        # Output directory
        if curves_output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            curves_output_dir = project_root / "visualizations" / "training_curves"
        self.curves_output_dir = Path(curves_output_dir)
        if self.save_curves:
            self.curves_output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def name(self) -> str:
        return "SeparateThresholdRuntime"
    
    def _extract_data(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Extract all data from loader."""
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
        feature_names = dataset.NUMERIC_FEATURE_KEYS + ["backend", "precision"] + \
                       [f.replace("_", "") for f in dataset.FAMILY_CATEGORIES]
        
        return X, y_thresh, y_runtime, feature_names
    
    def _get_top_correlated_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_features: int = 10,
    ) -> Tuple[List[int], List[str]]:
        """Get indices and names of top N most correlated features."""
        correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        correlations = np.abs(correlations)  # Use absolute correlation
        correlations = np.nan_to_num(correlations, nan=0.0)  # Replace NaN with 0
        
        top_indices = np.argsort(correlations)[-n_features:][::-1]  # Top 10, descending
        top_names = [feature_names[i] for i in top_indices]
        
        return top_indices.tolist(), top_names
    
    def _round_to_class(self, pred: np.ndarray) -> np.ndarray:
        """Round regression predictions to nearest valid class index."""
        return np.clip(np.round(pred), 0, len(THRESHOLD_LADDER) - 1).astype(int)
    
    def _save_epoch_curves(self, model_type: str, epoch: int) -> None:
        """Save training and validation loss curves for current epoch."""
        if not HAS_MATPLOTLIB:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            if model_type == "threshold":
                history = self.threshold_history
            else:
                history = self.runtime_history
            
            epochs_range = range(1, len(history["train"]) + 1)
            
            ax.plot(epochs_range, history["train"], label="Train Loss", linewidth=2, color='#3498db', marker='o')
            ax.plot(epochs_range, history["val"], label="Val Loss", linewidth=2, color='#e74c3c', marker='s')
            ax.axvline(x=epoch, color='green', linestyle='--', alpha=0.5, label=f'Epoch {epoch}')
            
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("RMSE Loss", fontsize=12)
            ax.set_title(f"{model_type.title()} Model - Epoch {epoch}", fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = self.curves_output_dir / f"{model_type}_epoch_{epoch:03d}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"  ⚠️  Failed to save {model_type} curves at epoch {epoch}: {e}")
    
    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = "threshold",
        verbose: bool = False,
        patience: int = 10,
    ) -> xgb.XGBRegressor:
        """Train a model with per-epoch loss tracking and early stopping."""
        
        model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=1,  # Train one tree at a time
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="rmse",
        )
        
        history = {"train": [], "val": []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.min_epochs + 1):
            # Train for 1 more boosting round
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
                xgb_model=model if epoch > 1 else None,
            )
            
            # Get current predictions and losses
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            train_loss = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_loss = np.sqrt(mean_squared_error(y_val, y_val_pred))
            
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            
            # Print loss
            print(f"  Epoch {epoch:3d}: Train Loss={train_loss:.6f} | Val Loss={val_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save curves every N epochs
            if epoch % self.save_interval == 0:
                if model_type == "threshold":
                    self.threshold_history = history.copy()
                else:
                    self.runtime_history = history.copy()
                self._save_epoch_curves(model_type, epoch)
            
            # Stop if patience exceeded
            if patience_counter >= patience:
                print(f"  ⏸️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        # Store final history
        if model_type == "threshold":
            self.threshold_history = history
        else:
            self.runtime_history = history
        
        return model
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train separate models for threshold and runtime."""
        
        X_train, y_thresh_train, y_runtime_train, feature_names = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val, _ = self._extract_data(val_loader)
        
        # Get top 25 most correlated features for each model
        self.threshold_feature_indices, self.threshold_feature_names = self._get_top_correlated_features(
            X_train, y_thresh_train, feature_names, n_features=25
        )
        self.runtime_feature_indices, self.runtime_feature_names = self._get_top_correlated_features(
            X_train, y_runtime_train, feature_names, n_features=25
        )
        
        if verbose:
            print(f"\n📊 Top 25 Features for Threshold: {self.threshold_feature_names}")
            print(f"📊 Top 25 Features for Runtime: {self.runtime_feature_names}")
        
        # Extract and scale features
        X_train_thresh = X_train[:, self.threshold_feature_indices]
        X_val_thresh = X_val[:, self.threshold_feature_indices]
        X_train_thresh_scaled = self.threshold_scaler.fit_transform(X_train_thresh)
        X_val_thresh_scaled = self.threshold_scaler.transform(X_val_thresh)
        
        X_train_runtime = X_train[:, self.runtime_feature_indices]
        X_val_runtime = X_val[:, self.runtime_feature_indices]
        X_train_runtime_scaled = self.runtime_scaler.fit_transform(X_train_runtime)
        X_val_runtime_scaled = self.runtime_scaler.transform(X_val_runtime)
        
        # Train threshold model
        if verbose:
            print(f"\n🎯 Training Threshold Model for {self.min_epochs} epochs...")
        self.threshold_model = self._train_model(
            X_train_thresh_scaled, y_thresh_train,
            X_val_thresh_scaled, y_thresh_val,
            model_type="threshold",
            verbose=verbose,
        )
        
        # Train runtime model
        if verbose:
            print(f"\n⏱️  Training Runtime Model for {self.min_epochs} epochs...")
        self.runtime_model = self._train_model(
            X_train_runtime_scaled, y_runtime_train,
            X_val_runtime_scaled, y_runtime_val,
            model_type="runtime",
            verbose=verbose,
        )
        
        # Evaluate
        train_metrics = self.evaluate(train_loader)
        val_metrics = self.evaluate(val_loader)
        
        if verbose:
            print(f"\n✅ Final Results:")
            print(f"  Train Threshold Acc: {train_metrics['threshold_accuracy']:.4f}")
            print(f"  Train Runtime MAE: {train_metrics['runtime_mae']:.4f}")
            print(f"  Val Threshold Acc: {val_metrics['threshold_accuracy']:.4f}")
            print(f"  Val Runtime MAE: {val_metrics['runtime_mae']:.4f}")
        
        return {"train": train_metrics, "val": val_metrics}
    
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on data."""
        X, y_thresh, y_runtime, _ = self._extract_data(loader)
        
        X_thresh = X[:, self.threshold_feature_indices]
        X_runtime = X[:, self.runtime_feature_indices]
        
        X_thresh_scaled = self.threshold_scaler.transform(X_thresh)
        X_runtime_scaled = self.runtime_scaler.transform(X_runtime)
        
        # Predictions
        thresh_pred_raw = self.threshold_model.predict(X_thresh_scaled)
        thresh_pred = self._round_to_class(thresh_pred_raw)
        runtime_pred = self.runtime_model.predict(X_runtime_scaled)
        
        return {
            "threshold_accuracy": accuracy_score(y_thresh.astype(int), thresh_pred),
            "runtime_mse": mean_squared_error(y_runtime, runtime_pred),
            "runtime_mae": mean_absolute_error(y_runtime, runtime_pred),
        }
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on new features."""
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
        """Get feature importance for each model."""
        if self.threshold_model is None or self.runtime_model is None:
            return None
        return {
            "threshold": self.threshold_model.feature_importances_,
            "runtime": self.runtime_model.feature_importances_,
        }
    
    def save(self, path: Path) -> None:
        """Save both models and scalers."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.threshold_model.save_model(str(path / "threshold_model.json"))
        self.runtime_model.save_model(str(path / "runtime_model.json"))
        
        np.save(path / "threshold_scaler_mean.npy", self.threshold_scaler.mean_)
        np.save(path / "threshold_scaler_scale.npy", self.threshold_scaler.scale_)
        np.save(path / "runtime_scaler_mean.npy", self.runtime_scaler.mean_)
        np.save(path / "runtime_scaler_scale.npy", self.runtime_scaler.scale_)
        
        np.save(path / "threshold_feature_indices.npy", np.array(self.threshold_feature_indices))
        np.save(path / "runtime_feature_indices.npy", np.array(self.runtime_feature_indices))
    
    def load(self, path: Path) -> None:
        """Load both models and scalers."""
        path = Path(path)
        
        self.threshold_model = xgb.XGBRegressor()
        self.threshold_model.load_model(str(path / "threshold_model.json"))
        
        self.runtime_model = xgb.XGBRegressor()
        self.runtime_model.load_model(str(path / "runtime_model.json"))
        
        self.threshold_scaler.mean_ = np.load(path / "threshold_scaler_mean.npy")
        self.threshold_scaler.scale_ = np.load(path / "threshold_scaler_scale.npy")
        self.runtime_scaler.mean_ = np.load(path / "runtime_scaler_mean.npy")
        self.runtime_scaler.scale_ = np.load(path / "runtime_scaler_scale.npy")
        
        self.threshold_feature_indices = np.load(path / "threshold_feature_indices.npy").tolist()
        self.runtime_feature_indices = np.load(path / "runtime_feature_indices.npy").tolist()

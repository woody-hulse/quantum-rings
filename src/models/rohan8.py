"""
LightGBM model v2 — fixed version.

Changes from v1:
- No threshold bias (was destroying accuracy)
- Use ALL features instead of aggressive pre-filtering (let LightGBM handle selection)
- Only 5 interaction features per target (from top-10 base features)
- Much fewer trees (60), shallower (max_depth=3), higher min_child_samples (5)
- Stronger regularization (reg_alpha=0.5, reg_lambda=2.0)
- Single scaler (no need for per-target scaling when using all features)
- Early stopping to prevent overfitting
"""

from typing import Dict, Tuple, Any, Optional, List
from pathlib import Path
from itertools import combinations

import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import THRESHOLD_LADDER
from models.base import BaseModel

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class Rohan8Model(BaseModel):
    """LightGBM with light interaction features and anti-overfitting defaults."""

    def __init__(
        self,
        top_k_base: int = 10,
        max_interactions: int = 5,
        n_estimators: int = 60,
        max_depth: int = 3,
        learning_rate: float = 0.08,
        num_leaves: int = 7,
        min_child_samples: int = 5,
        subsample: float = 0.7,
        colsample_bytree: float = 0.7,
        reg_alpha: float = 0.5,
        reg_lambda: float = 2.0,
        threshold_bias: int = 0,
        random_state: int = 42,
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm required. Install with: pip install lightgbm")

        self.top_k_base = top_k_base
        self.max_interactions = max_interactions
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.threshold_bias = threshold_bias
        self.random_state = random_state

        self.threshold_model = None
        self.runtime_model = None
        self.scaler = StandardScaler()

        # Interaction feature state (computed on training data)
        self.thresh_interaction_pairs: Optional[List[Tuple[int, int]]] = None
        self.runtime_interaction_pairs: Optional[List[Tuple[int, int]]] = None
        self.thresh_top_base_idx: Optional[np.ndarray] = None
        self.runtime_top_base_idx: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "Rohan8"

    # ── data extraction ──────────────────────────────────────────────

    def _extract_data(
        self, loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_features, all_thresh, all_runtime = [], [], []
        for batch in loader:
            all_features.append(batch["features"].numpy())
            all_thresh.extend(batch["threshold_class"].tolist())
            all_runtime.extend(batch["log_runtime"].tolist())
        return (
            np.vstack(all_features),
            np.array(all_thresh, dtype=np.float32),
            np.array(all_runtime),
        )

    # ── feature engineering ──────────────────────────────────────────

    @staticmethod
    def _top_correlated_idx(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Indices of the k features with highest |correlation| to y."""
        corrs = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            col = X[:, i]
            if np.std(col) < 1e-12:
                continue
            valid = np.isfinite(col) & np.isfinite(y)
            if valid.sum() > 2:
                corrs[i] = abs(np.corrcoef(col[valid], y[valid])[0, 1])
        corrs = np.nan_to_num(corrs, nan=0.0)
        return np.sort(np.argsort(corrs)[::-1][: min(k, len(corrs))])

    @staticmethod
    def _best_interaction_pairs(
        X: np.ndarray, y: np.ndarray, base_idx: np.ndarray, max_pairs: int
    ) -> List[Tuple[int, int]]:
        """Find the most correlated pairwise products among base features."""
        if len(base_idx) < 2 or max_pairs == 0:
            return []
        candidates = list(combinations(range(len(base_idx)), 2))
        X_base = X[:, base_idx]
        pair_corrs = []
        for i, j in candidates:
            product = X_base[:, i] * X_base[:, j]
            if np.std(product) < 1e-12:
                pair_corrs.append(0.0)
                continue
            valid = np.isfinite(product) & np.isfinite(y)
            if valid.sum() > 2:
                c = abs(np.corrcoef(product[valid], y[valid])[0, 1])
                pair_corrs.append(c if np.isfinite(c) else 0.0)
            else:
                pair_corrs.append(0.0)
        pair_corrs = np.array(pair_corrs)
        top = np.argsort(pair_corrs)[::-1][:max_pairs]
        return [candidates[t] for t in top]

    def _append_interactions(
        self,
        X: np.ndarray,
        base_idx: np.ndarray,
        interaction_pairs: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Append interaction columns to the FULL feature matrix."""
        if not interaction_pairs:
            return X
        interactions = []
        for i, j in interaction_pairs:
            interactions.append(
                (X[:, base_idx[i]] * X[:, base_idx[j]]).reshape(-1, 1)
            )
        return np.hstack([X] + interactions)

    # ── threshold helpers ────────────────────────────────────────────

    def _round_to_class(self, pred: np.ndarray) -> np.ndarray:
        return np.clip(
            np.round(pred) + self.threshold_bias, 0, len(THRESHOLD_LADDER) - 1
        ).astype(int)

    # ── LightGBM params ─────────────────────────────────────────────

    def _lgb_params(self) -> dict:
        return {
            "objective": "regression",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": 1,
        }

    # ── fit / evaluate / predict ─────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        X_train, y_thresh_train, y_runtime_train = self._extract_data(train_loader)
        X_val, y_thresh_val, y_runtime_val = self._extract_data(val_loader)

        # Scale ALL base features first
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Find top-k base features for interaction generation only
        self.thresh_top_base_idx = self._top_correlated_idx(
            X_train_scaled, y_thresh_train, self.top_k_base
        )
        self.runtime_top_base_idx = self._top_correlated_idx(
            X_train_scaled, y_runtime_train, self.top_k_base
        )

        # Find best interaction pairs from those top features
        self.thresh_interaction_pairs = self._best_interaction_pairs(
            X_train_scaled, y_thresh_train,
            self.thresh_top_base_idx, self.max_interactions,
        )
        self.runtime_interaction_pairs = self._best_interaction_pairs(
            X_train_scaled, y_runtime_train,
            self.runtime_top_base_idx, self.max_interactions,
        )

        # Build: all base features + a few interactions
        Xt_train = self._append_interactions(
            X_train_scaled, self.thresh_top_base_idx, self.thresh_interaction_pairs
        )
        Xt_val = self._append_interactions(
            X_val_scaled, self.thresh_top_base_idx, self.thresh_interaction_pairs
        )
        Xr_train = self._append_interactions(
            X_train_scaled, self.runtime_top_base_idx, self.runtime_interaction_pairs
        )
        Xr_val = self._append_interactions(
            X_val_scaled, self.runtime_top_base_idx, self.runtime_interaction_pairs
        )

        if verbose:
            print(f"Threshold: {X_train.shape[1]} base + "
                  f"{len(self.thresh_interaction_pairs)} interactions = {Xt_train.shape[1]}")
            print(f"Runtime:   {X_train.shape[1]} base + "
                  f"{len(self.runtime_interaction_pairs)} interactions = {Xr_train.shape[1]}")

        # ── train threshold ──
        thresh_evals = {}
        self.threshold_model = lgb.LGBMRegressor(**self._lgb_params())
        self.threshold_model.fit(
            Xt_train, y_thresh_train,
            eval_set=[(Xt_train, y_thresh_train), (Xt_val, y_thresh_val)],
            eval_names=["train", "val"],
            callbacks=[
                lgb.log_evaluation(0),
                lgb.early_stopping(10, verbose=False),
                lgb.record_evaluation(thresh_evals),
            ],
        )
 
        # ── train runtime ──
        runtime_evals = {}
        self.runtime_model = lgb.LGBMRegressor(**self._lgb_params())
        self.runtime_model.fit(
            Xr_train, y_runtime_train,
            eval_set=[(Xr_train, y_runtime_train), (Xr_val, y_runtime_val)],
            eval_names=["train", "val"],
            callbacks=[
                lgb.log_evaluation(0),
                lgb.early_stopping(10, verbose=False),
                lgb.record_evaluation(runtime_evals),
            ],
        )

      

        # Store training history for plotting
        self.training_history_ = {
            "threshold_train_loss": thresh_evals.get("train", {}).get("l2", []),
            "threshold_val_loss": thresh_evals.get("val", {}).get("l2", []),
            "runtime_train_loss": runtime_evals.get("train", {}).get("l2", []),
            "runtime_val_loss": runtime_evals.get("val", {}).get("l2", []),
        }

        train_metrics = self._evaluate_internal(
            Xt_train, y_thresh_train, Xr_train, y_runtime_train
        )
        val_metrics = self._evaluate_internal(
            Xt_val, y_thresh_val, Xr_val, y_runtime_val
        )

        if verbose:
            print(f"Train  thresh acc: {train_metrics['threshold_accuracy']:.4f}  "
                  f"runtime MSE: {train_metrics['runtime_mse']:.4f}")
            print(f"Val    thresh acc: {val_metrics['threshold_accuracy']:.4f}  "
                  f"runtime MSE: {val_metrics['runtime_mse']:.4f}")

        return {"train": train_metrics, "val": val_metrics}

    def _evaluate_internal(
        self,
        Xt: np.ndarray,
        y_thresh: np.ndarray,
        Xr: np.ndarray,
        y_runtime: np.ndarray,
    ) -> Dict[str, float]:
        thresh_pred = self._round_to_class(self.threshold_model.predict(Xt))
        runtime_pred = self.runtime_model.predict(Xr)
        return {
            "threshold_accuracy": accuracy_score(y_thresh.astype(int), thresh_pred),
            "runtime_mse": mean_squared_error(y_runtime, runtime_pred),
            "runtime_mae": mean_absolute_error(y_runtime, runtime_pred),
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        X, y_thresh, y_runtime = self._extract_data(loader)
        X_scaled = self.scaler.transform(X)
        Xt = self._append_interactions(
            X_scaled, self.thresh_top_base_idx, self.thresh_interaction_pairs
        )
        Xr = self._append_interactions(
            X_scaled, self.runtime_top_base_idx, self.runtime_interaction_pairs
        )
        return self._evaluate_internal(Xt, y_thresh, Xr, y_runtime)

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold_model is None or self.runtime_model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(features)
        Xt = self._append_interactions(
            X_scaled, self.thresh_top_base_idx, self.thresh_interaction_pairs
        )
        Xr = self._append_interactions(
            X_scaled, self.runtime_top_base_idx, self.runtime_interaction_pairs
        )

        thresh_classes = self._round_to_class(self.threshold_model.predict(Xt))
        thresh_values = np.array([THRESHOLD_LADDER[c] for c in thresh_classes])

        runtime_log = self.runtime_model.predict(Xr)
        runtime_values = np.expm1(runtime_log)

        return thresh_values, runtime_values

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        if self.threshold_model is None or self.runtime_model is None:
            return None
        return {
            "threshold": self.threshold_model.feature_importances_,
            "runtime": self.runtime_model.feature_importances_,
            "threshold_top_base_indices": self.thresh_top_base_idx,
            "runtime_top_base_indices": self.runtime_top_base_idx,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.threshold_model.booster_.save_model(str(path / "threshold_model.txt"))
        self.runtime_model.booster_.save_model(str(path / "runtime_model.txt"))
        np.save(path / "scaler_mean.npy", self.scaler.mean_)
        np.save(path / "scaler_scale.npy", self.scaler.scale_)
        np.save(path / "thresh_top_base_idx.npy", self.thresh_top_base_idx)
        np.save(path / "runtime_top_base_idx.npy", self.runtime_top_base_idx)
        np.save(path / "thresh_interaction_pairs.npy", np.array(self.thresh_interaction_pairs))
        np.save(path / "runtime_interaction_pairs.npy", np.array(self.runtime_interaction_pairs))

    def load(self, path: Path) -> None:
        path = Path(path)
        self.threshold_model = lgb.Booster(model_file=str(path / "threshold_model.txt"))
        self.runtime_model = lgb.Booster(model_file=str(path / "runtime_model.txt"))
        self.scaler.mean_ = np.load(path / "scaler_mean.npy")
        self.scaler.scale_ = np.load(path / "scaler_scale.npy")
        self.thresh_top_base_idx = np.load(path / "thresh_top_base_idx.npy")
        self.runtime_top_base_idx = np.load(path / "runtime_top_base_idx.npy")
        self.thresh_interaction_pairs = [
            tuple(p) for p in np.load(path / "thresh_interaction_pairs.npy")
        ]
        self.runtime_interaction_pairs = [
            tuple(p) for p in np.load(path / "runtime_interaction_pairs.npy")
        ]

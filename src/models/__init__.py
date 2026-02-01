"""
Model implementations: duration prediction (threshold as input, log2 runtime)
and threshold-class prediction (no duration, no threshold in features; P(class), select by max expected score).
"""

from models.base import BaseModel, ThresholdClassBaseModel
from models.gradient_boosting_base import (
    GradientBoostingRegressionModel,
    GradientBoostingClassificationModel,
)
from models.mlp import MLPModel
from models.mlp_threshold_class import MLPThresholdClassModel
from models.xgboost_model import XGBoostModel
from models.xgboost_threshold_class import XGBoostThresholdClassModel
from models.catboost_model import CatBoostModel
from models.catboost_threshold_class import CatBoostThresholdClassModel
from models.lightgbm_model import LightGBMModel
from models.gnn_threshold_class import GNNThresholdClassModel

__all__ = [
    "BaseModel",
    "ThresholdClassBaseModel",
    "GradientBoostingRegressionModel",
    "GradientBoostingClassificationModel",
    "MLPModel",
    "MLPThresholdClassModel",
    "XGBoostModel",
    "XGBoostThresholdClassModel",
    "CatBoostModel",
    "CatBoostThresholdClassModel",
    "LightGBMModel",
    "GNNThresholdClassModel",
]

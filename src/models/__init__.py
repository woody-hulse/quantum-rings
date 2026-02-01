"""
Model implementations: duration prediction (threshold as input, log2 runtime)
and threshold-class prediction (no duration, no threshold in features; P(class), select by max expected score).

Graph Models:
- BasicGNNThresholdClassModel: Simple message-passing GNN
- ImprovedGNNThresholdClassModel: Attention-based with ordinal regression
- GraphTransformerThresholdClassModel: Full transformer with edge attention
- HeteroGNNThresholdClassModel: Heterogeneous multi-relation GNN
- TemporalGNNThresholdClassModelV2: Temporal/causal modeling

Use create_graph_model() factory function for easy instantiation.
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
from models.temporal_gnn_model import (
    TemporalGNNDurationModel,
    TemporalGNNThresholdClassModel,
)
from models.graph_models import (
    BaseGraphModelWrapper,
    GraphModelConfig,
    BasicGNNThresholdClassModel,
    ImprovedGNNThresholdClassModel,
    GraphTransformerThresholdClassModel,
    HeteroGNNThresholdClassModel,
    TemporalGNNThresholdClassModelV2,
    create_graph_model,
    get_all_model_types,
    MODEL_DESCRIPTIONS,
)

__all__ = [
    # Base classes
    "BaseModel",
    "ThresholdClassBaseModel",
    # Gradient Boosting
    "GradientBoostingRegressionModel",
    "GradientBoostingClassificationModel",
    # MLP
    "MLPModel",
    "MLPThresholdClassModel",
    # XGBoost
    "XGBoostModel",
    "XGBoostThresholdClassModel",
    # CatBoost
    "CatBoostModel",
    "CatBoostThresholdClassModel",
    # LightGBM
    "LightGBMModel",
    # GNN (legacy wrapper)
    "GNNThresholdClassModel",
    "TemporalGNNDurationModel",
    "TemporalGNNThresholdClassModel",
    # Unified Graph Models
    "BaseGraphModelWrapper",
    "GraphModelConfig",
    "BasicGNNThresholdClassModel",
    "ImprovedGNNThresholdClassModel",
    "GraphTransformerThresholdClassModel",
    "HeteroGNNThresholdClassModel",
    "TemporalGNNThresholdClassModelV2",
    "create_graph_model",
    "get_all_model_types",
    "MODEL_DESCRIPTIONS",
]

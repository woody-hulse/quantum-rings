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

from .base import BaseModel, ThresholdClassBaseModel
from .gradient_boosting_base import (
    GradientBoostingRegressionModel,
    GradientBoostingClassificationModel,
)
from .mlp import MLPModel
from .mlp_threshold_class import MLPThresholdClassModel
from .xgboost_model import XGBoostModel
from .xgboost_threshold_class import XGBoostThresholdClassModel
from .catboost_model import CatBoostModel
from .catboost_threshold_class import CatBoostThresholdClassModel
from .lightgbm_model import LightGBMModel
from .gnn_threshold_class import GNNThresholdClassModel
from .temporal_gnn_model import (
    TemporalGNNDurationModel,
    TemporalGNNThresholdClassModel,
)
from .graph_models import (
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

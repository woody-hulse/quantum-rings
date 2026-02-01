"""
Separate specialized models for Task 1 and Task 2.

Task 1: Threshold prediction for target fidelity 0.75
Task 2: Runtime prediction given a threshold
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from .model import (
    OrdinalRegressionHead,
    OrdinalRegressionLoss,
    GateTypeMessagePassing,
)
from .graph_builder import NUM_GATE_TYPES, NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_FEAT_DIM_BASE


class ThresholdPredictionGNN(nn.Module):
    """
    Task 1: Predict optimal threshold for target fidelity 0.75.

    Uses expected-score optimization:
    - Outputs probability distribution over thresholds
    - Selects guess that maximizes expected score given scoring function:
      * guess < true: 0 points
      * guess = true: 1.0 points
      * guess is N steps above true: 1.0 / (2^N) points

    Input: Circuit graph, backend, precision
    Output: Threshold class (0-8)
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 20,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_threshold_classes: int = 9,
        dropout: float = 0.1,
        use_ordinal: bool = False,  # Deprecated - always use softmax now
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_threshold_classes = num_threshold_classes

        # Initial node projection
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            GateTypeMessagePassing(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_feat_dim=edge_feat_dim,
            )
            for _ in range(num_layers)
        ])

        self.mp_dropout = nn.Dropout(dropout)

        # Graph pooling produces 3 * hidden_dim features
        pool_dim = 3 * hidden_dim

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Combined representation processing
        combined_dim = pool_dim + hidden_dim

        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Threshold prediction head - standard classification
        self.threshold_head = nn.Linear(hidden_dim, num_threshold_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for threshold prediction.

        Returns:
            threshold_logits: [batch_size, num_threshold_classes]
        """
        # Initial node embedding
        h = self.node_embed(x)

        # Message passing
        for mp_layer in self.mp_layers:
            h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
            h = h + self.mp_dropout(h_new)

        # Graph-level pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max, h_sum], dim=-1)

        # Process global features
        g = self.global_proj(global_features)

        # Combine representations
        combined = torch.cat([h_graph, g], dim=-1)
        combined = self.combined_mlp(combined)

        # Predict threshold logits
        threshold_logits = self.threshold_head(combined)

        return threshold_logits

    def get_probabilities(self, threshold_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probability distribution.

        Args:
            threshold_logits: [batch_size, num_classes]

        Returns:
            probs: [batch_size, num_classes] probability distribution
        """
        return F.softmax(threshold_logits, dim=-1)

    def compute_expected_scores(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute expected score for each possible guess.

        Scoring function:
        - If guess_idx < true_idx: 0 points
        - If guess_idx == true_idx: 1.0 points
        - If guess_idx > true_idx: 1.0 / (2 ** (guess_idx - true_idx)) points

        Args:
            probs: [batch_size, num_classes] probability distribution over true threshold

        Returns:
            expected_scores: [batch_size, num_classes] expected score for each guess
        """
        batch_size = probs.shape[0]
        num_classes = probs.shape[1]

        # Create scoring matrix [num_guesses, num_true_thresholds]
        # scoring_matrix[g, t] = score when guessing g and true threshold is t
        scoring_matrix = torch.zeros(
            (num_classes, num_classes),
            device=probs.device,
            dtype=probs.dtype
        )

        for guess_idx in range(num_classes):
            for true_idx in range(num_classes):
                if guess_idx < true_idx:
                    # Guess too low - violates fidelity
                    scoring_matrix[guess_idx, true_idx] = 0.0
                elif guess_idx == true_idx:
                    # Exact match
                    scoring_matrix[guess_idx, true_idx] = 1.0
                else:
                    # Guess too high - partial credit
                    steps_over = guess_idx - true_idx
                    scoring_matrix[guess_idx, true_idx] = 1.0 / (2.0 ** steps_over)

        # Compute expected score for each guess
        # expected_scores[b, g] = sum_t P(true=t) * score(guess=g, true=t)
        # = sum_t probs[b, t] * scoring_matrix[g, t]
        expected_scores = torch.matmul(probs, scoring_matrix.T)  # [batch_size, num_classes]

        return expected_scores

    def predict_threshold_class(self, threshold_logits: torch.Tensor) -> torch.Tensor:
        """
        Predict threshold class by maximizing expected score.

        Args:
            threshold_logits: [batch_size, num_classes]

        Returns:
            predicted_classes: [batch_size] optimal threshold class indices
        """
        probs = self.get_probabilities(threshold_logits)
        expected_scores = self.compute_expected_scores(probs)
        return expected_scores.argmax(dim=1)


class RuntimePredictionGNN(nn.Module):
    """
    Task 2: Predict runtime given a specific threshold.

    Input: Circuit graph, threshold, backend, precision
    Output: Runtime (log-transformed)
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 20,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_threshold_classes: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_threshold_classes = num_threshold_classes

        # Initial node projection
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            GateTypeMessagePassing(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_feat_dim=edge_feat_dim,
            )
            for _ in range(num_layers)
        ])

        self.mp_dropout = nn.Dropout(dropout)

        # Graph pooling produces 3 * hidden_dim features
        pool_dim = 3 * hidden_dim

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Threshold embedding - CRITICAL: threshold is an input
        self.threshold_embed = nn.Sequential(
            nn.Embedding(num_threshold_classes, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )

        # Combined representation processing
        # Now includes threshold embedding
        combined_dim = pool_dim + hidden_dim + hidden_dim // 2

        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Runtime prediction head
        self.runtime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
        threshold_class: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for runtime prediction.

        Args:
            threshold_class: Threshold class indices [batch_size]

        Returns:
            runtime_pred: [batch_size] log-transformed runtime
        """
        # Initial node embedding
        h = self.node_embed(x)

        # Message passing
        for mp_layer in self.mp_layers:
            h_new = mp_layer(h, edge_index, edge_attr, edge_gate_type)
            h = h + self.mp_dropout(h_new)

        # Graph-level pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch)
        h_graph = torch.cat([h_mean, h_max, h_sum], dim=-1)

        # Process global features
        g = self.global_proj(global_features)

        # Embed threshold input
        threshold_emb = self.threshold_embed(threshold_class)

        # Combine all representations including threshold
        combined = torch.cat([h_graph, g, threshold_emb], dim=-1)
        combined = self.combined_mlp(combined)

        # Predict runtime
        runtime_pred = self.runtime_head(combined).squeeze(-1)

        return runtime_pred


def create_threshold_model(
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 20,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_threshold_classes: int = 9,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Create Task 1 threshold prediction model with expected-score optimization.

    The model outputs a probability distribution and uses a decision rule that
    maximizes expected score given the asymmetric scoring function.
    """
    return ThresholdPredictionGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        global_feat_dim=global_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_threshold_classes=num_threshold_classes,
        dropout=dropout,
    )


def create_runtime_model(
    node_feat_dim: int = NODE_FEAT_DIM,
    edge_feat_dim: int = EDGE_FEAT_DIM,
    global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 20,
    hidden_dim: int = 64,
    num_layers: int = 4,
    num_threshold_classes: int = 9,
    dropout: float = 0.1,
) -> nn.Module:
    """Create Task 2 runtime prediction model."""
    return RuntimePredictionGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        global_feat_dim=global_feat_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_threshold_classes=num_threshold_classes,
        dropout=dropout,
    )

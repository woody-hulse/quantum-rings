"""
Abstract base class for graph neural network models.

This module defines a common interface for all graph-based models that operate
on quantum circuit graph representations. Models implementing this interface
can be used interchangeably for threshold prediction or duration prediction.

The graph representation for quantum circuits:
- Nodes: Qubits with per-qubit gate statistics and temporal features
- Edges: Gates (directed for multi-qubit, self-loops for single-qubit)
- Global: Circuit-level statistics (size, backend, family encoding)
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from .graph_builder import NODE_FEAT_DIM, EDGE_FEAT_DIM, NUM_GATE_TYPES, GLOBAL_FEAT_DIM_BASE


@dataclass
class GraphModelConfig:
    """Configuration for graph models."""
    node_feat_dim: int = NODE_FEAT_DIM
    edge_feat_dim: int = EDGE_FEAT_DIM
    num_gate_types: int = NUM_GATE_TYPES
    global_feat_dim: int = GLOBAL_FEAT_DIM_BASE + 20
    hidden_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_feat_dim": self.node_feat_dim,
            "edge_feat_dim": self.edge_feat_dim,
            "num_gate_types": self.num_gate_types,
            "global_feat_dim": self.global_feat_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }


class BaseGraphModel(nn.Module, ABC):
    """
    Abstract base class for graph-based quantum circuit models.
    
    All graph models (GNN, Graph Transformer, etc.) should inherit from this
    class and implement the required methods. This ensures a consistent
    interface for training, inference, and evaluation.
    
    Graph Input Structure:
        x: Node features [n_nodes, node_feat_dim]
        edge_index: Edge connectivity [2, n_edges]
        edge_attr: Edge features [n_edges, edge_feat_dim]
        edge_gate_type: Gate type indices [n_edges]
        batch: Batch assignment [n_nodes]
        global_features: Circuit-level features [batch_size, global_feat_dim]
    """
    
    def __init__(self, config: Optional[GraphModelConfig] = None):
        super().__init__()
        self.config = config or GraphModelConfig()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name for reporting."""
        pass
    
    @abstractmethod
    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode node features using graph structure.
        
        This is the core graph processing step that transforms raw node features
        into contextualized representations by propagating information through
        the graph structure (gates between qubits).
        
        For quantum circuits, this captures:
        - How information/entanglement flows between qubits
        - Gate-specific effects on state complexity
        - Temporal ordering of operations
        
        Args:
            x: Node features [n_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, n_edges]
            edge_attr: Edge features [n_edges, edge_feat_dim]
            edge_gate_type: Gate type indices [n_edges]
            batch: Batch assignment for pooling [n_nodes]
            
        Returns:
            Encoded node representations [n_nodes, hidden_dim]
        """
        pass
    
    @abstractmethod
    def pool_graph(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool node embeddings to graph-level representation.
        
        Aggregates node-level features into a single vector per graph,
        capturing the overall circuit properties.
        
        Args:
            node_embeddings: Node representations [n_nodes, hidden_dim]
            batch: Batch assignment [n_nodes]
            
        Returns:
            Graph representations [batch_size, pool_dim]
        """
        pass
    
    @abstractmethod
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
        Full forward pass from graph to predictions.
        
        Args:
            x: Node features [n_nodes, node_feat_dim]
            edge_index: Edge connectivity [2, n_edges]
            edge_attr: Edge features [n_edges, edge_feat_dim]
            edge_gate_type: Gate type indices [n_edges]
            batch: Batch assignment [n_nodes]
            global_features: Circuit-level features [batch_size, global_feat_dim]
            
        Returns:
            Model predictions (shape depends on task)
        """
        pass
    
    def forward_batch(self, batch: Batch) -> torch.Tensor:
        """
        Convenience method to run forward pass on a PyG Batch object.
        
        Args:
            batch: PyTorch Geometric Batch containing graph data
            
        Returns:
            Model predictions
        """
        return self.forward(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            edge_gate_type=batch.edge_gate_type,
            batch=batch.batch,
            global_features=batch.global_features,
        )
    
    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get intermediate node embeddings (useful for visualization/analysis).
        
        Default implementation uses encode_nodes. Override if additional
        processing is needed.
        """
        return self.encode_nodes(x, edge_index, edge_attr, edge_gate_type, batch)
    
    def get_attention_weights(self) -> Optional[List[torch.Tensor]]:
        """
        Return attention weights if the model uses attention.
        
        Override in attention-based models (Graph Transformer, GAT).
        
        Returns:
            List of attention weight tensors per layer, or None
        """
        return None
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseGraphDurationModel(BaseGraphModel):
    """
    Base class for duration prediction models.
    
    Predicts log2(runtime) given circuit graph and threshold.
    Global features include log2(threshold).
    """
    
    @abstractmethod
    def predict_runtime(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict log2(runtime) for each graph in the batch.
        
        Returns:
            Log2 runtime predictions [batch_size]
        """
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_runtime(
            x, edge_index, edge_attr, edge_gate_type, batch, global_features
        )


class BaseGraphThresholdClassModel(BaseGraphModel):
    """
    Base class for threshold class prediction models.
    
    Predicts probability distribution over threshold classes.
    Threshold classes: [1, 2, 4, 8, 16, 32, 64, 128, 256]
    """
    
    NUM_THRESHOLD_CLASSES = 9
    
    @abstractmethod
    def predict_logits(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict class logits for each graph in the batch.
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_logits(
            x, edge_index, edge_attr, edge_gate_type, batch, global_features
        )
    
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Returns:
            Class probabilities [batch_size, num_classes]
        """
        logits = self.predict_logits(
            x, edge_index, edge_attr, edge_gate_type, batch, global_features
        )
        return torch.softmax(logits, dim=-1)
    
    def predict_class(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate_type: torch.Tensor,
        batch: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict most likely class.
        
        Returns:
            Class indices [batch_size]
        """
        logits = self.predict_logits(
            x, edge_index, edge_attr, edge_gate_type, batch, global_features
        )
        return logits.argmax(dim=-1)

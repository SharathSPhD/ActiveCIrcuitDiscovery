# Core data structures for YorK_RP Active Inference Circuit Discovery

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import torch
from pathlib import Path
import json
import pickle
from collections import defaultdict
try:
    from scipy import sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import configuration with proper relative import
try:
    from config.experiment_config import InterventionType
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.experiment_config import InterventionType

# Add caching utilities
class FeatureCache:
    """Simple LRU cache for feature computations."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove oldest
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()

# Global cache instance
feature_cache = FeatureCache()

@dataclass 
class SAEFeature:
    """Represents a Sparse Autoencoder feature with comprehensive metadata."""
    feature_id: int
    layer: int  
    activation_threshold: float
    description: str
    max_activation: float
    examples: List[str]
    feature_vector: Optional[np.ndarray] = None
    decoder_weights: Optional[np.ndarray] = None
    
    def __post_init__(self):
        # Allow larger activations for real transformer features
        if self.max_activation < 0:
            raise ValueError(f"max_activation must be non-negative, got {self.max_activation}")
        if self.layer < 0:
            raise ValueError(f"layer must be non-negative, got {self.layer}")

@dataclass
class InterventionResult:
    """Results from a circuit intervention experiment."""
    intervention_type: InterventionType
    target_feature: SAEFeature
    original_logits: torch.Tensor
    intervened_logits: torch.Tensor
    effect_size: float
    target_token_change: float
    intervention_layer: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate tensors have same shape
        if self.original_logits.shape != self.intervened_logits.shape:
            raise ValueError("Original and intervened logits must have same shape")
    
    @property
    def logit_diff_l2(self) -> float:
        """Calculate L2 norm of logit difference."""
        return torch.norm(self.intervened_logits - self.original_logits).item()
    
    @property 
    def kl_divergence(self) -> float:
        """Calculate KL divergence between original and intervened distributions."""
        original_probs = torch.softmax(self.original_logits, dim=-1)
        intervened_probs = torch.softmax(self.intervened_logits, dim=-1)
        return torch.nn.functional.kl_div(
            torch.log(intervened_probs + 1e-10), 
            original_probs, 
            reduction='sum'
        ).item()

@dataclass
class CircuitNode:
    """Node in the discovered circuit graph."""
    feature: SAEFeature
    activation_value: float
    causal_influence: float  
    downstream_nodes: List[int] = field(default_factory=list)
    upstream_nodes: List[int] = field(default_factory=list)
    
    def add_downstream(self, node_id: int):
        """Add downstream connection."""
        if node_id not in self.downstream_nodes:
            self.downstream_nodes.append(node_id)
    
    def add_upstream(self, node_id: int):
        """Add upstream connection."""
        if node_id not in self.upstream_nodes:
            self.upstream_nodes.append(node_id)

@dataclass
class GraphNode:
    """Node in the attribution graph with enhanced metadata."""
    node_id: str
    layer: int
    feature_id: int
    importance: float
    description: str
    
    def __post_init__(self):
        if not 0 <= self.importance <= 1:
            raise ValueError(f"importance must be in [0,1], got {self.importance}")

@dataclass
class GraphEdge:
    """Edge in the attribution graph with enhanced metadata."""
    source_id: str
    target_id: str
    weight: float
    confidence: float
    edge_type: str = "causal"
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")

@dataclass 
class AttributionGraph:
    """Efficient attribution graph for circuit discovery with sparse representation."""
    input_text: str
    nodes: List[GraphNode]  # Node list for easy iteration
    edges: List[GraphEdge]  # Edge list for easy iteration
    target_output: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Efficient lookup structures
    _node_index: Dict[str, int] = field(default_factory=dict, init=False)
    _adjacency_matrix: Optional[Any] = field(default=None, init=False)  # Sparse matrix
    _edge_weights: Dict[Tuple[str, str], float] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Build efficient lookup structures."""
        self._build_node_index()
        self._build_adjacency_matrix()
        self._build_edge_weights()
    
    def _build_node_index(self):
        """Build node ID to index mapping."""
        self._node_index = {node.node_id: i for i, node in enumerate(self.nodes)}
    
    def _build_adjacency_matrix(self):
        """Build sparse adjacency matrix for efficient graph operations."""
        if not SCIPY_AVAILABLE or not self.nodes:
            return
        
        n = len(self.nodes)
        row_indices = []
        col_indices = []
        data = []
        
        for edge in self.edges:
            if edge.source_id in self._node_index and edge.target_id in self._node_index:
                source_idx = self._node_index[edge.source_id]
                target_idx = self._node_index[edge.target_id]
                row_indices.append(source_idx)
                col_indices.append(target_idx)
                data.append(edge.weight)
        
        if data:  # Only create matrix if we have edges
            self._adjacency_matrix = sparse.csr_matrix(
                (data, (row_indices, col_indices)), shape=(n, n)
            )
    
    def _build_edge_weights(self):
        """Build edge weight lookup dictionary."""
        self._edge_weights = {
            (edge.source_id, edge.target_id): edge.weight 
            for edge in self.edges
        }
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID with O(1) lookup."""
        idx = self._node_index.get(node_id)
        return self.nodes[idx] if idx is not None else None
    
    def add_edge(self, source_id: str, target_id: str, weight: float, confidence: float = 1.0):
        """Add edge between nodes and update efficient structures."""
        edge = GraphEdge(source_id, target_id, weight, confidence)
        self.edges.append(edge)
        self._edge_weights[(source_id, target_id)] = weight
        
        # Update adjacency matrix if it exists
        if (self._adjacency_matrix is not None and 
            source_id in self._node_index and target_id in self._node_index):
            source_idx = self._node_index[source_id]
            target_idx = self._node_index[target_id]
            self._adjacency_matrix[source_idx, target_idx] = weight
    
    def get_node_degree(self, node_id: str) -> Tuple[int, int]:
        """Get in-degree and out-degree of node efficiently."""
        if self._adjacency_matrix is not None and node_id in self._node_index:
            idx = self._node_index[node_id]
            out_degree = (self._adjacency_matrix[idx, :] != 0).sum()
            in_degree = (self._adjacency_matrix[:, idx] != 0).sum()
            return (int(in_degree), int(out_degree))
        
        # Fallback to linear search
        in_degree = sum(1 for edge in self.edges if edge.target_id == node_id)
        out_degree = sum(1 for edge in self.edges if edge.source_id == node_id)
        return (in_degree, out_degree)
    
    def get_neighbors(self, node_id: str) -> Tuple[List[str], List[str]]:
        """Get predecessors and successors of a node."""
        predecessors = [edge.source_id for edge in self.edges if edge.target_id == node_id]
        successors = [edge.target_id for edge in self.edges if edge.source_id == node_id]
        return (predecessors, successors)
    
    def get_edge_weight(self, source_id: str, target_id: str) -> Optional[float]:
        """Get edge weight with O(1) lookup."""
        return self._edge_weights.get((source_id, target_id))
    
    def prune_graph(self, node_threshold: float = 0.8, edge_threshold: float = 0.98) -> 'AttributionGraph':
        """Prune graph by removing low-importance nodes and edges."""
        # Sort nodes by importance
        sorted_nodes = sorted(self.nodes, key=lambda n: n.importance, reverse=True)
        
        # Keep top nodes by cumulative importance
        cumulative_importance = 0.0
        total_importance = sum(node.importance for node in sorted_nodes)
        kept_nodes = []
        
        for node in sorted_nodes:
            cumulative_importance += node.importance
            kept_nodes.append(node)
            if cumulative_importance / total_importance >= node_threshold:
                break
        
        kept_node_ids = {node.node_id for node in kept_nodes}
        
        # Sort edges by weight and keep top edges
        sorted_edges = sorted(self.edges, key=lambda e: abs(e.weight), reverse=True)
        cumulative_weight = 0.0
        total_weight = sum(abs(edge.weight) for edge in sorted_edges)
        kept_edges = []
        
        for edge in sorted_edges:
            # Only keep edges between kept nodes
            if edge.source_id in kept_node_ids and edge.target_id in kept_node_ids:
                cumulative_weight += abs(edge.weight)
                kept_edges.append(edge)
                if cumulative_weight / total_weight >= edge_threshold:
                    break
        
        # Create new pruned graph
        return AttributionGraph(
            input_text=self.input_text,
            nodes=kept_nodes,
            edges=kept_edges,
            target_output=self.target_output,
            confidence=self.confidence,
            metadata={**self.metadata, 'pruned': True, 'original_size': (len(self.nodes), len(self.edges))}
        )
    
    def to_json(self) -> str:
        """Serialize graph to JSON format."""
        graph_dict = {
            'input_text': self.input_text,
            'target_output': self.target_output,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'nodes': [
                {
                    'node_id': node.node_id,
                    'layer': node.layer,
                    'feature_id': node.feature_id,
                    'importance': node.importance,
                    'description': node.description
                } for node in self.nodes
            ],
            'edges': [
                {
                    'source_id': edge.source_id,
                    'target_id': edge.target_id,
                    'weight': edge.weight,
                    'confidence': edge.confidence,
                    'edge_type': edge.edge_type
                } for edge in self.edges
            ]
        }
        return json.dumps(graph_dict, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AttributionGraph':
        """Deserialize graph from JSON format."""
        graph_dict = json.loads(json_str)
        
        nodes = [
            GraphNode(
                node_id=node_data['node_id'],
                layer=node_data['layer'],
                feature_id=node_data['feature_id'],
                importance=node_data['importance'],
                description=node_data['description']
            ) for node_data in graph_dict['nodes']
        ]
        
        edges = [
            GraphEdge(
                source_id=edge_data['source_id'],
                target_id=edge_data['target_id'],
                weight=edge_data['weight'],
                confidence=edge_data['confidence'],
                edge_type=edge_data.get('edge_type', 'causal')
            ) for edge_data in graph_dict['edges']
        ]
        
        return cls(
            input_text=graph_dict['input_text'],
            nodes=nodes,
            edges=edges,
            target_output=graph_dict['target_output'],
            confidence=graph_dict['confidence'],
            metadata=graph_dict.get('metadata', {})
        )
    
    def save(self, filepath: Path):
        """Save graph to file with compression."""
        filepath = Path(filepath)
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                f.write(self.to_json())
        else:
            # Use pickle for binary format
            with open(filepath, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, filepath: Path) -> 'AttributionGraph':
        """Load graph from file."""
        filepath = Path(filepath)
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                return cls.from_json(f.read())
        else:
            # Load from pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)

@dataclass
class CorrespondenceMetrics:
    """Metrics measuring correspondence between Active Inference and circuit mechanisms."""
    belief_updating_correspondence: float  # AI belief updates vs circuit behavior
    precision_weighting_correspondence: float  # AI precision vs attention patterns
    prediction_error_correspondence: float  # AI prediction errors vs circuit errors
    overall_correspondence: float  # Combined score for RQ1 validation
    
    # Supporting evidence
    circuit_attention_patterns: List[float] = field(default_factory=list)
    ai_precision_patterns: List[float] = field(default_factory=list)
    circuit_prediction_errors: List[float] = field(default_factory=list)
    ai_prediction_errors: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        # Validate all scores are in [0, 1]
        for field_name in ['belief_updating_correspondence', 'precision_weighting_correspondence', 
                          'prediction_error_correspondence', 'overall_correspondence']:
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be in [0,1], got {value}")

@dataclass
class NovelPrediction:
    """Novel prediction about circuit behavior from Active Inference analysis."""
    prediction_type: str  # 'attention_pattern', 'feature_interaction', 'failure_mode'
    description: str
    testable_hypothesis: str
    expected_outcome: str
    test_method: str
    confidence: float
    validation_status: str = "untested"
    validation_evidence: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        
        valid_types = ['attention_pattern', 'feature_interaction', 'failure_mode']
        if self.prediction_type not in valid_types:
            raise ValueError(f"prediction_type must be one of {valid_types}")
        
        valid_statuses = ['untested', 'validated', 'falsified']
        if self.validation_status not in valid_statuses:
            raise ValueError(f"validation_status must be one of {valid_statuses}")

@dataclass
class ValidationResult:
    """Result from prediction validation."""
    prediction_id: str
    validation_status: str  # 'validated', 'falsified', 'inconclusive'
    confidence: float
    evidence: Dict[str, Any]
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    
    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        
        valid_statuses = ['validated', 'falsified', 'inconclusive']
        if self.validation_status not in valid_statuses:
            raise ValueError(f"validation_status must be one of {valid_statuses}")

@dataclass
class EfficiencyMetrics:
    """Comprehensive efficiency metrics for RQ2 validation."""
    ai_intervention_count: int
    baseline_intervention_counts: Dict[str, int]
    improvement_percentages: Dict[str, float]
    overall_improvement: float
    statistical_significance: Dict[str, float]  # p-values
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    def __post_init__(self):
        if self.ai_intervention_count < 0:
            raise ValueError("ai_intervention_count must be non-negative")
        if not 0 <= self.overall_improvement <= 100:
            raise ValueError("overall_improvement must be in [0,100]")

@dataclass
class BeliefState:
    """Enhanced state representation for Active Inference agent with temporal tracking."""
    # Core pymdp components
    qs: np.ndarray  # Posterior beliefs over states
    
    # Feature-specific beliefs
    feature_importances: Dict[int, float]
    connection_beliefs: Dict[Tuple[int, int], float]
    uncertainty: Dict[int, float]
    confidence: float
    
    # Enhanced pymdp integration
    generative_model: Optional[Dict[str, Any]] = None
    posterior_beliefs: Optional[np.ndarray] = None
    precision_matrix: Optional[np.ndarray] = None
    
    # Temporal tracking
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    step_number: int = 0
    intervention_history: List[str] = field(default_factory=list)
    
    # Expected Free Energy components
    epistemic_value: float = 0.0
    pragmatic_value: float = 0.0
    predicted_surprise: float = 0.0
    
    # Belief updating metadata
    learning_rate: float = 0.1
    prediction_errors: Dict[int, float] = field(default_factory=dict)
    belief_changes: Dict[int, float] = field(default_factory=dict)
    
    def get_entropy(self) -> float:
        """Calculate entropy of current belief state."""
        try:
            from scipy.stats import entropy
            return entropy(self.qs + 1e-10)
        except ImportError:
            # Fallback calculation
            return -np.sum(self.qs * np.log(self.qs + 1e-10))
    
    def get_average_uncertainty(self) -> float:
        """Get average uncertainty across all features."""
        if not self.uncertainty:
            return 1.0
        return sum(self.uncertainty.values()) / len(self.uncertainty)
    
    def update_feature_belief(self, feature_id: int, new_importance: float, 
                            prediction_error: float = 0.0) -> float:
        """Update belief about a specific feature and track changes."""
        old_importance = self.feature_importances.get(feature_id, 0.5)
        
        # Track prediction error
        self.prediction_errors[feature_id] = prediction_error
        
        # Update importance with learning rate
        updated_importance = old_importance + self.learning_rate * (new_importance - old_importance)
        updated_importance = max(0.0, min(1.0, updated_importance))  # Clamp to [0,1]
        
        # Track belief change
        change = abs(updated_importance - old_importance)
        self.belief_changes[feature_id] = change
        
        # Update uncertainty (reduce with evidence)
        if feature_id in self.uncertainty:
            uncertainty_reduction = min(0.1, abs(prediction_error) * 0.2)
            self.uncertainty[feature_id] = max(0.1, self.uncertainty[feature_id] - uncertainty_reduction)
        
        self.feature_importances[feature_id] = updated_importance
        return change
    
    def get_total_belief_change(self) -> float:
        """Calculate total belief change in this update."""
        return sum(self.belief_changes.values())
    
    def get_expected_free_energy(self) -> float:
        """Calculate Expected Free Energy for current state."""
        return self.epistemic_value + self.pragmatic_value + self.predicted_surprise
    
    def clone(self) -> 'BeliefState':
        """Create a deep copy of the belief state for history tracking."""
        return BeliefState(
            qs=self.qs.copy(),
            feature_importances=self.feature_importances.copy(),
            connection_beliefs=self.connection_beliefs.copy(),
            uncertainty=self.uncertainty.copy(),
            confidence=self.confidence,
            generative_model=self.generative_model.copy() if self.generative_model else None,
            posterior_beliefs=self.posterior_beliefs.copy() if self.posterior_beliefs is not None else None,
            precision_matrix=self.precision_matrix.copy() if self.precision_matrix is not None else None,
            timestamp=self.timestamp,
            step_number=self.step_number,
            intervention_history=self.intervention_history.copy(),
            epistemic_value=self.epistemic_value,
            pragmatic_value=self.pragmatic_value,
            predicted_surprise=self.predicted_surprise,
            learning_rate=self.learning_rate,
            prediction_errors=self.prediction_errors.copy(),
            belief_changes=self.belief_changes.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'qs': self.qs.tolist(),
            'feature_importances': self.feature_importances,
            'connection_beliefs': {f"{k[0]},{k[1]}": v for k, v in self.connection_beliefs.items()},
            'uncertainty': self.uncertainty,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'step_number': self.step_number,
            'intervention_history': self.intervention_history,
            'epistemic_value': self.epistemic_value,
            'pragmatic_value': self.pragmatic_value,
            'predicted_surprise': self.predicted_surprise,
            'prediction_errors': self.prediction_errors,
            'belief_changes': self.belief_changes
        }

@dataclass
class ExperimentResult:
    """Results from a complete experiment run."""
    experiment_name: str
    timestamp: str
    config_used: Dict[str, Any]
    
    # Core results
    correspondence_metrics: List[CorrespondenceMetrics]
    efficiency_metrics: Dict[str, float]
    novel_predictions: List[NovelPrediction]
    
    # Research question validation
    rq1_passed: bool
    rq2_passed: bool  
    rq3_passed: bool
    overall_success: bool
    
    # Supporting data
    intervention_results: List[InterventionResult] = field(default_factory=list)
    belief_history: List[BeliefState] = field(default_factory=list)
    circuit_graphs: List[AttributionGraph] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        passed = sum([self.rq1_passed, self.rq2_passed, self.rq3_passed])
        return passed / 3.0
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the experiment."""
        return {
            'total_interventions': len(self.intervention_results),
            'average_effect_size': np.mean([r.effect_size for r in self.intervention_results]) if self.intervention_results else 0.0,
            'average_correspondence': np.mean([c.overall_correspondence for c in self.correspondence_metrics]) if self.correspondence_metrics else 0.0,
            'predictions_generated': len(self.novel_predictions),
            'predictions_validated': len([p for p in self.novel_predictions if p.validation_status == 'validated']),
            'success_rate': self.success_rate,
            'experiment_duration': self.metadata.get('duration_seconds', 0)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'config_used': self.config_used,
            'rq1_passed': self.rq1_passed,
            'rq2_passed': self.rq2_passed,
            'rq3_passed': self.rq3_passed,
            'overall_success': self.overall_success,
            'summary_stats': self.get_summary_stats()
        }
    
    def save_comprehensive(self, filepath: Path):
        """Save complete experiment results with all data."""
        filepath = Path(filepath)
        
        # Create comprehensive export
        export_data = {
            'experiment_info': self.to_dict(),
            'correspondence_metrics': [
                {
                    'belief_updating': cm.belief_updating_correspondence,
                    'precision_weighting': cm.precision_weighting_correspondence,
                    'prediction_error': cm.prediction_error_correspondence,
                    'overall': cm.overall_correspondence
                } for cm in self.correspondence_metrics
            ],
            'intervention_results': [
                {
                    'intervention_type': ir.intervention_type.value,
                    'feature_id': ir.target_feature.feature_id,
                    'layer': ir.target_feature.layer,
                    'effect_size': ir.effect_size,
                    'target_token_change': ir.target_token_change,
                    'kl_divergence': ir.kl_divergence,
                    'metadata': ir.metadata
                } for ir in self.intervention_results
            ],
            'belief_evolution': [
                bs.to_dict() for bs in self.belief_history
            ],
            'predictions': [
                {
                    'type': pred.prediction_type,
                    'description': pred.description,
                    'hypothesis': pred.testable_hypothesis,
                    'confidence': pred.confidence,
                    'status': pred.validation_status
                } for pred in self.novel_predictions
            ]
        }
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)

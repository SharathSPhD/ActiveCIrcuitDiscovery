# Core data structures for YorK_RP Active Inference Circuit Discovery

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import torch
from pathlib import Path

# Import configuration with proper relative import
try:
    from ..config.experiment_config import InterventionType
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.experiment_config import InterventionType

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
        if not 0 <= self.max_activation <= 1:
            raise ValueError(f"max_activation must be in [0,1], got {self.max_activation}")
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
class AttributionGraph:
    """Complete attribution graph for circuit discovery."""
    input_text: str
    nodes: Dict[int, CircuitNode]
    edges: Dict[Tuple[int, int], float]  # (source, target) -> weight
    target_output: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_edge(self, source_id: int, target_id: int, weight: float):
        """Add edge between nodes."""
        self.edges[(source_id, target_id)] = weight
        if source_id in self.nodes:
            self.nodes[source_id].add_downstream(target_id)
        if target_id in self.nodes:
            self.nodes[target_id].add_upstream(source_id)
    
    def get_node_degree(self, node_id: int) -> Tuple[int, int]:
        """Get in-degree and out-degree of node."""
        if node_id not in self.nodes:
            return (0, 0)
        return (len(self.nodes[node_id].upstream_nodes), 
                len(self.nodes[node_id].downstream_nodes))

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
class BeliefState:
    """State representation for Active Inference agent."""
    # Core pymdp components
    qs: np.ndarray  # Posterior beliefs over states
    
    # Feature-specific beliefs
    feature_importances: Dict[int, float]
    connection_beliefs: Dict[Tuple[int, int], float]
    uncertainty: Dict[int, float]
    confidence: float
    
    # Optional pymdp integration
    generative_model: Optional[Dict[str, Any]] = None
    posterior_beliefs: Optional[np.ndarray] = None
    precision_matrix: Optional[np.ndarray] = None
    
    def get_entropy(self) -> float:
        """Calculate entropy of current belief state."""
        from scipy.stats import entropy
        return entropy(self.qs + 1e-10)
    
    def get_average_uncertainty(self) -> float:
        """Get average uncertainty across all features."""
        if not self.uncertainty:
            return 1.0
        return sum(self.uncertainty.values()) / len(self.uncertainty)

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

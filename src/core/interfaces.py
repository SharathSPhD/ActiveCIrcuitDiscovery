# Abstract interfaces for YorK_RP Active Inference Circuit Discovery
# Following Interface Segregation Principle

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np

from core.data_structures import (
    SAEFeature, InterventionResult, AttributionGraph, BeliefState,
    CorrespondenceMetrics, NovelPrediction, ExperimentResult
)
from config.experiment_config import CompleteConfig, InterventionType

class ICircuitTracer(ABC):
    """Interface for circuit discovery and analysis."""
    
    @abstractmethod
    def find_active_features(self, text: str, threshold: float = 0.05) -> Dict[int, List[SAEFeature]]:
        """Find active SAE features for given input text."""
        pass
    
    @abstractmethod
    def perform_intervention(self, text: str, feature: SAEFeature, 
                           intervention_type: InterventionType) -> InterventionResult:
        """Perform intervention on specified feature."""
        pass
    
    @abstractmethod
    def build_attribution_graph(self, text: str) -> AttributionGraph:
        """Build complete attribution graph for circuit analysis."""
        pass
    
    @abstractmethod
    def get_feature_activations(self, text: str, layer: int) -> torch.Tensor:
        """Get feature activations for specific layer."""
        pass

class IInterventionStrategy(ABC):
    """Interface for intervention selection strategies."""
    
    @abstractmethod
    def select_intervention(self, available_features: List[SAEFeature], 
                          belief_state: Optional[BeliefState] = None) -> Optional[SAEFeature]:
        """Select next feature for intervention."""
        pass
    
    @abstractmethod
    def update_strategy(self, intervention_result: InterventionResult):
        """Update strategy based on intervention result."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get name of the strategy."""
        pass

class IActiveInferenceAgent(ABC):
    """Interface for Active Inference agent following pymdp patterns."""
    
    @abstractmethod
    def initialize_beliefs(self, features: Dict[int, List[SAEFeature]]) -> BeliefState:
        """Initialize belief state from discovered features."""
        pass
    
    @abstractmethod
    def calculate_expected_free_energy(self, feature: SAEFeature, 
                                     intervention_type: InterventionType) -> float:
        """Calculate expected free energy for potential intervention."""
        pass
    
    @abstractmethod
    def update_beliefs(self, intervention_result: InterventionResult) -> CorrespondenceMetrics:
        """Update beliefs based on intervention result."""
        pass
    
    @abstractmethod
    def generate_predictions(self) -> List[NovelPrediction]:
        """Generate novel predictions from current beliefs."""
        pass
    
    @abstractmethod
    def check_convergence(self, threshold: float = 0.15) -> bool:
        """Check if beliefs have converged."""
        pass

class IMetricsCalculator(ABC):
    """Interface for calculating research question metrics."""
    
    @abstractmethod
    def calculate_correspondence(self, ai_beliefs: BeliefState, 
                               circuit_behavior: List[InterventionResult]) -> CorrespondenceMetrics:
        """Calculate RQ1: Active Inference correspondence metrics."""
        pass
    
    @abstractmethod
    def calculate_efficiency(self, ai_interventions: int, 
                           baseline_results: Dict[str, int]) -> Dict[str, float]:
        """Calculate RQ2: Efficiency improvement metrics."""
        pass
    
    @abstractmethod
    def validate_predictions(self, predictions: List[NovelPrediction], 
                           test_data: List[str]) -> int:
        """Validate RQ3: Novel predictions."""
        pass

class IVisualizationGenerator(ABC):
    """Interface for generating experiment visualizations."""
    
    @abstractmethod
    def create_circuit_diagram(self, graph: AttributionGraph, 
                             output_path: str) -> str:
        """Create circuit diagram visualization."""
        pass
    
    @abstractmethod
    def create_metrics_dashboard(self, result: ExperimentResult, 
                               output_path: str) -> str:
        """Create comprehensive metrics dashboard."""
        pass
    
    @abstractmethod
    def create_belief_evolution_plot(self, belief_history: List[BeliefState], 
                                   output_path: str) -> str:
        """Create plot showing belief evolution over time."""
        pass

class IExperimentRunner(ABC):
    """Interface for running complete experiments."""
    
    @abstractmethod
    def setup_experiment(self, config: CompleteConfig):
        """Setup experiment with given configuration."""
        pass
    
    @abstractmethod
    def run_experiment(self, test_inputs: List[str]) -> ExperimentResult:
        """Run complete experiment on test inputs."""
        pass
    
    @abstractmethod
    def validate_research_questions(self, result: ExperimentResult) -> Dict[str, bool]:
        """Validate all research questions against targets."""
        pass
    
    @abstractmethod
    def save_results(self, result: ExperimentResult, output_dir: str):
        """Save experiment results to specified directory."""
        pass

class IConfigurationValidator(ABC):
    """Interface for configuration validation."""
    
    @abstractmethod
    def validate_model_config(self, config: CompleteConfig) -> List[str]:
        """Validate model configuration."""
        pass
    
    @abstractmethod
    def validate_experiment_config(self, config: CompleteConfig) -> List[str]:
        """Validate experiment configuration."""
        pass
    
    @abstractmethod
    def validate_dependencies(self) -> List[str]:
        """Validate required dependencies are available."""
        pass

class IResultsAnalyzer(ABC):
    """Interface for analyzing experiment results."""
    
    @abstractmethod
    def analyze_intervention_patterns(self, results: List[InterventionResult]) -> Dict[str, Any]:
        """Analyze patterns in intervention results."""
        pass
    
    @abstractmethod
    def compare_strategies(self, results: Dict[str, ExperimentResult]) -> Dict[str, Any]:
        """Compare different intervention strategies."""
        pass
    
    @abstractmethod
    def generate_insights(self, result: ExperimentResult) -> List[str]:
        """Generate insights from experiment results."""
        pass

class IStatisticalValidator(ABC):
    """Interface for enhanced statistical validation."""
    
    @abstractmethod
    def validate_correspondence_significance(self, correspondence_metrics: List[CorrespondenceMetrics],
                                          target_threshold: float = 70.0) -> Dict[str, Any]:
        """Validate correspondence significance with statistical testing."""
        pass
    
    @abstractmethod
    def validate_efficiency_improvement(self, ai_interventions: List[int],
                                      baseline_interventions: Dict[str, List[int]],
                                      target_improvement: float = 30.0) -> Dict[str, Any]:
        """Validate efficiency improvement with statistical testing."""
        pass
    
    @abstractmethod
    def validate_prediction_success_rate(self, predictions: List[NovelPrediction],
                                       target_count: int = 3) -> Dict[str, Any]:
        """Validate prediction success rate with statistical testing."""
        pass

class IPredictionGenerator(ABC):
    """Interface for enhanced prediction generation."""
    
    @abstractmethod
    def generate_attention_pattern_predictions(self, belief_state: BeliefState,
                                             circuit_data: Dict[str, Any]) -> List[NovelPrediction]:
        """Generate predictions about attention patterns."""
        pass
    
    @abstractmethod
    def generate_feature_interaction_predictions(self, belief_state: BeliefState,
                                               circuit_graph: AttributionGraph) -> List[NovelPrediction]:
        """Generate predictions about feature interactions."""
        pass
    
    @abstractmethod
    def generate_failure_mode_predictions(self, belief_state: BeliefState,
                                        intervention_history: List[InterventionResult]) -> List[NovelPrediction]:
        """Generate predictions about circuit failure modes."""
        pass

class IPredictionValidator(ABC):
    """Interface for enhanced prediction validation."""
    
    @abstractmethod
    def validate_attention_pattern_prediction(self, prediction: NovelPrediction,
                                            test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate attention pattern predictions."""
        pass
    
    @abstractmethod
    def validate_feature_interaction_prediction(self, prediction: NovelPrediction,
                                              test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature interaction predictions."""
        pass
    
    @abstractmethod
    def validate_failure_mode_prediction(self, prediction: NovelPrediction,
                                       test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate failure mode predictions."""
        pass

# Factory interfaces for dependency injection

class ICircuitTracerFactory(ABC):
    """Factory for creating circuit tracers."""
    
    @abstractmethod
    def create_tracer(self, config: CompleteConfig) -> ICircuitTracer:
        """Create circuit tracer based on configuration."""
        pass

class IStrategyFactory(ABC):
    """Factory for creating intervention strategies."""
    
    @abstractmethod
    def create_strategy(self, strategy_name: str, config: CompleteConfig) -> IInterventionStrategy:
        """Create intervention strategy by name."""
        pass
    
    @abstractmethod
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        pass

class IAgentFactory(ABC):
    """Factory for creating Active Inference agents."""
    
    @abstractmethod
    def create_agent(self, config: CompleteConfig, tracer: ICircuitTracer) -> IActiveInferenceAgent:
        """Create Active Inference agent with configuration."""
        pass

# Exception classes for proper error handling

class CircuitDiscoveryError(Exception):
    """Base exception for circuit discovery errors."""
    pass

class InterventionError(CircuitDiscoveryError):
    """Exception for intervention-related errors."""
    pass

class ActiveInferenceError(CircuitDiscoveryError):
    """Exception for Active Inference-related errors."""
    pass

class ConfigurationError(CircuitDiscoveryError):
    """Exception for configuration-related errors."""
    pass

class IVisualizationBackend(ABC):
    """Interface for visualization backends (static, interactive, etc.)."""
    
    @abstractmethod
    def create_interactive_graph(self, graph: AttributionGraph, 
                               output_path: Optional[str] = None) -> str:
        """Create visualization of attribution graph."""
        pass

class ValidationError(CircuitDiscoveryError):
    """Exception for validation-related errors."""
    pass

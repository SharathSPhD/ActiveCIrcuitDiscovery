# YorK_RP Core Module  
"""Core data structures and interfaces for Active Inference Circuit Discovery."""

from .data_structures import (
    CircuitFeature,
    SAEFeature,  # Backward compatibility alias
    InterventionResult,
    CircuitNode,
    AttributionGraph,
    CorrespondenceMetrics,
    NovelPrediction,
    BeliefState,
    ExperimentResult
)

from .interfaces import (
    ICircuitTracer,
    IInterventionStrategy,
    IActiveInferenceAgent,
    IMetricsCalculator,
    IVisualizationGenerator,
    IExperimentRunner,
    IConfigurationValidator,
    IResultsAnalyzer,
    ICircuitTracerFactory,
    IStrategyFactory,
    IAgentFactory,
    CircuitDiscoveryError,
    InterventionError,
    ActiveInferenceError,
    ConfigurationError,
    ValidationError
)

from .metrics import (
    CorrespondenceCalculator,
    EfficiencyCalculator,
    ValidationCalculator,
    StatisticalResult
)

from .prediction_system import (
    PredictionGenerator,
    AttentionPatternPredictor,
    FeatureInteractionPredictor,
    FailureModePredictor,
    EnhancedPredictionGenerator,
    PredictionEvidence,
    ValidationResult
)

from .prediction_validator import (
    ValidationMethod,
    AttentionPatternValidator,
    FeatureInteractionValidator,
    FailureModeValidator,
    PredictionValidator,
    ValidationConfig
)

from .statistical_validation import (
    StatisticalValidator,
    StatisticalTest,
    BootstrapResult,
    MultipleComparisonsCorrection
)

__all__ = [
    # Data structures
    'CircuitFeature',
    'SAEFeature',  # Backward compatibility
    'InterventionResult', 
    'CircuitNode',
    'AttributionGraph',
    'CorrespondenceMetrics',
    'NovelPrediction',
    'BeliefState',
    'ExperimentResult',
    
    # Interfaces
    'ICircuitTracer',
    'IInterventionStrategy',
    'IActiveInferenceAgent',
    'IMetricsCalculator',
    'IVisualizationGenerator',
    'IExperimentRunner',
    'IConfigurationValidator',
    'IResultsAnalyzer',
    'ICircuitTracerFactory',
    'IStrategyFactory',
    'IAgentFactory',
    
    # Exceptions
    'CircuitDiscoveryError',
    'InterventionError',
    'ActiveInferenceError',
    'ConfigurationError',
    'ValidationError',
    
    # Metrics
    'CorrespondenceCalculator',
    'EfficiencyCalculator',
    'ValidationCalculator',
    'StatisticalResult',
    
    # Prediction System
    'PredictionGenerator',
    'AttentionPatternPredictor',
    'FeatureInteractionPredictor',
    'FailureModePredictor',
    'EnhancedPredictionGenerator',
    'PredictionEvidence',
    'ValidationResult',
    
    # Prediction Validation
    'ValidationMethod',
    'AttentionPatternValidator',
    'FeatureInteractionValidator',
    'FailureModeValidator',
    'PredictionValidator',
    'ValidationConfig',
    
    # Statistical Validation
    'StatisticalValidator',
    'StatisticalTest',
    'BootstrapResult',
    'MultipleComparisonsCorrection'
]

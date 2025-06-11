# YorK_RP: Active Inference Approach to Circuit Discovery in Large Language Models
"""
Research Project: Active Inference Approach to Circuit Discovery in Large Language Models
Module: COM00150M - Research Proposal (Computer Science)

This package provides a comprehensive implementation of Active Inference principles
for circuit discovery in transformer language models, integrating with established
libraries like TransformerLens, SAE-lens, and pymdp.

Main Components:
- config: Configuration management
- core: Data structures and interfaces  
- circuit_analysis: Circuit discovery with real SAE analysis
- active_inference: Active Inference agent with pymdp integration
- experiments: Experiment runners and orchestration
"""

from .config import get_config, CompleteConfig
from .core import (
    SAEFeature, InterventionResult, ExperimentResult,
    CorrespondenceCalculator, EnhancedPredictionGenerator,
    PredictionValidator, StatisticalValidator
)
from .circuit_analysis import CircuitTracer
from .active_inference import ActiveInferenceAgent
from .experiments import YorKExperimentRunner
from .visualization import CircuitVisualizer

__version__ = "1.0.0"
__author__ = "YorK Research Project"

__all__ = [
    'get_config',
    'CompleteConfig',
    'SAEFeature',
    'InterventionResult', 
    'ExperimentResult',
    'CircuitTracer',
    'ActiveInferenceAgent',
    'YorKExperimentRunner',
    'CircuitVisualizer',
    'CorrespondenceCalculator',
    'EnhancedPredictionGenerator',
    'PredictionValidator',
    'StatisticalValidator'
]

# Quick setup function for common use cases
def create_experiment(config_path=None):
    """Create a complete experiment setup with default configuration."""
    config = get_config(config_path)
    runner = YorKExperimentRunner()
    runner.setup_experiment(config)
    return runner, config

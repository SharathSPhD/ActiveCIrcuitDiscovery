# YorK_RP: Active Inference Approach to Circuit Discovery in Large Language Models
"""
Research Project: Enhanced Active Inference for Circuit Discovery in Large Language Models
Module: COM00150M - Research Proposal (Computer Science)

This package provides a production-ready implementation of Enhanced Active Inference 
for circuit discovery in transformer language models, integrating circuit-tracer 
with real pymdp Active Inference.

Main Components:
- config: Configuration management for Gemma-2-2B + circuit-tracer
- core: Unified data structures and interfaces  
- circuit_analysis: Real circuit discovery with circuit-tracer transcoders
- active_inference: Real pymdp Active Inference agent (no fallbacks)
- experiments: Integrated experiment runners with RQ validation
"""

from .config import get_config, CompleteConfig
from .core import (
    CircuitFeature, InterventionResult, ExperimentResult,
    CorrespondenceCalculator, EnhancedPredictionGenerator,
    PredictionValidator, StatisticalValidator
)
from .circuit_analysis import RealCircuitTracer
from .active_inference import ProperActiveInferenceAgent
from .experiments import CircuitDiscoveryIntegration
from .visualization import CircuitVisualizer

__version__ = "1.0.0"
__author__ = "YorK Research Project"

__all__ = [
    'get_config',
    'CompleteConfig', 
    'CircuitFeature',
    'InterventionResult', 
    'ExperimentResult',
    'RealCircuitTracer',
    'ProperActiveInferenceAgent',
    'CircuitDiscoveryIntegration',
    'CircuitVisualizer',
    'CorrespondenceCalculator',
    'EnhancedPredictionGenerator',
    'PredictionValidator',
    'StatisticalValidator'
]

# Quick setup function for common use cases
def create_experiment(config_path=None):
    """Create a complete experiment setup with circuit-tracer + Active Inference."""
    config = get_config(config_path)
    integration = CircuitDiscoveryIntegration(model_name=config.model.name)
    integration.initialize()
    return integration, config

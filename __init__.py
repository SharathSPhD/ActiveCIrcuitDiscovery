# Active Circuit Discovery
"""
Active Inference Approach to Circuit Discovery in Large Language Models
YorK Research Project - COM00150M Research Proposal

A comprehensive framework for discovering circuits in large language models using 
Active Inference principles, integrating with TransformerLens, SAE-lens, and pymdp.
"""

from src import (
    get_config,
    CompleteConfig,
    SAEFeature,
    InterventionResult,
    ExperimentResult,
    CircuitTracer,
    ActiveInferenceAgent,
    YorKExperimentRunner,
    CircuitVisualizer,
    create_experiment
)

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
    'create_experiment'
]